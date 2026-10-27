use chandra::algorithm::rc::{RcConfig, RcData};
use chandra::algorithm::trace::TraceData;
use chandra::algorithm::{CommandList, CoreData};
use keter::prelude::*;

#[test]
fn extra_diffuse_samples_keep_the_same_first_path_and_secondary_hit() {
    let core = CoreData::new([16, 16], chandra::white_room::world());
    let view = chandra::white_room::camera(0, [16, 16]).view();
    core.view.write_host(view).execute();
    core.compute_gbuffer().execute();
    let mut rc = RcData::new(RcConfig {
        base_probes: 4096,
        num_cascades: 3,
        ..Default::default()
    });
    let trace = TraceData::new_fused(&core, &rc, 16 * 16 * 3 * 64);
    let mut commands = CommandList::new();
    rc.reset_table(&mut commands, true);
    commands += rc.cache.set_view(view);
    commands += rc.init_probes_base(&core);
    for c in 1..rc.num_cascades {
        commands += rc.init_probes(c);
    }
    commands += rc.set_lifetime_base(&core);
    for c in 1..rc.num_cascades {
        commands += rc.set_lifetime(c);
    }
    rc.reset_table(&mut commands, false);
    // Assign each pixel an immutable rank for this test. Scheduling variation
    // must not be confused with a change in the samples-per-pixel budget.
    let mut trace = trace;
    trace.enable_stable_samples(&core, &rc);
    let schedule = |trace: &TraceData| {
        let mut commands = CommandList::new();
        commands += trace.reset();
        commands += trace.compute_ray_count_base(&core, &rc);
        for c in 1..rc.num_cascades {
            commands += trace.compute_ray_count(&rc, c);
        }
        for c in (0..rc.num_cascades).rev() {
            commands += trace.compute_ray_offset(&rc, c);
        }
        commands += trace.prepare_stable_samples(&core, &rc);
        commands.execute();
    };
    let budget = DEVICE.create_buffer_from_fn::<u32>(256, |_| 1);
    let length =
        ((2 * rc.base_angular_resolution.pow(2)) as f32 / (4. * std::f32::consts::PI)).sqrt();
    schedule(&trace);
    trace
        .compute_rays_adaptive(&core, &rc, 140, length, false, 0, 1, Some(&budget))
        .execute();
    let one = trace.traced_hits.copy_to_vec();
    let first_mass: f32 = rc
        .cache
        .weights
        .view(..rc.cache.total_directions_per_cascade() as usize)
        .copy_to_vec()
        .iter()
        .sum();
    assert!((first_mass - 256.).abs() < 1e-3);
    trace.enable_diffuse_work(&core);
    trace.set_primary_register_limit(64);
    budget.copy_from(&vec![32u32; 256]);
    schedule(&trace);
    trace
        .compute_rays_adaptive(&core, &rc, 140, length, false, 0, 1, Some(&budget))
        .execute();
    let many = trace.traced_hits.copy_to_vec();
    let total_mass: f32 = rc
        .cache
        .weights
        .view(..rc.cache.total_directions_per_cascade() as usize)
        .copy_to_vec()
        .iter()
        .sum();
    assert!(
        (total_mass - 512.).abs() < 1e-3,
        "Adaptive sampling changed the surface measure: {total_mass}"
    );
    assert!(one.iter().all(|h| h.position.x.is_finite()));
    assert_eq!(
        one, many,
        "Changing the budget changed the path used for secondary transport"
    );
}
