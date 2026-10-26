use chandra::algorithm::rc::{GiAvailability, RcConfig, RcData};
use chandra::algorithm::trace::{Hit, TraceData};
use chandra::algorithm::{CommandList, CoreData, WorldData};
use chandra::world::TriangleWorld;
use keter::lang::types::vector::Vec3;
use keter::prelude::*;

#[test]
fn full_hash_tables_terminate_and_recovery_coarsens_primary_and_secondary_allocations() {
    let table = chandra::table::HashTable::new(1, 4, [8].into_iter());
    table.freelist.copy_from(&[0, 1, 2, 3, 4, 5, 6, 7]);
    table.freelist_index.copy_from(&[0]);
    table.freelist_end.copy_from(&[8]);
    let result = DEVICE.create_buffer::<u32>(2);
    DEVICE
        .create_kernel::<fn()>(&track!(|| {
            for key in 1u64..=5u64 {
                table.insert(key, 0u32.expr(), |_| {});
            }
            result.write(0, table.get(999u64.expr(), 0u32.expr()));
            result.write(1, table.cost(999u64.expr(), 0u32.expr()));
        }))
        .dispatch([1, 1, 1]);
    assert_eq!(result.copy_to_vec(), [u32::MAX, u32::MAX]);
    assert_eq!(table.allocation_failures.copy_to_vec(), [1]);

    let core = CoreData::new(
        [64, 64],
        WorldData {
            world: TriangleWorld::new(
                &[
                    [-50000., -50000., 0.],
                    [50000., -50000., 0.],
                    [0., 50000., 0.],
                ],
                &[[0, 1, 2]],
                &[[0, 0]],
                &[0],
                &[Vec3::splat(0.8)],
            ),
            envmap: image::Rgb32FImage::new(1, 1),
            envmap_scale: 0.,
            sun_dir: Vec3::new(0., 0., 1.),
            sun_color: Vec3::splat(1.),
            custom_emitters: vec![],
        },
    );
    let camera = yesod::camera::Camera {
        screen_size: glam::Vec2::splat(64.),
        pos: glam::Vec3::new(0., 0., 1000.),
        yaw: 0.,
        pitch: 1.5,
        fov: 1.05,
    };
    core.view.write_host(camera.view()).execute();
    core.compute_gbuffer().execute();
    let mut rc = RcData::new(RcConfig {
        base_probes: 256,
        num_cascades: 3,
        base_grid_size: 6.,
        ..Default::default()
    });
    let trace = TraceData::new_fused(&core, &rc, 64 * 64 * 12);
    let hits: Vec<_> = (0..64 * 64)
        .map(|i| Hit {
            position: Vec3::new(5000. + (i % 64) as f32 * 40., (i / 64) as f32 * 40., 0.),
            normal: Vec3::new(0., 0., 1.),
        })
        .collect();
    trace.traced_hits.copy_from(&hits);
    let mut commands = CommandList::new();
    rc.reset_table(&mut commands, true);
    commands += rc.cache.set_view(camera.view());
    commands += rc.init_probes_base(&core);
    commands += rc.init_probes_secondary(&trace);
    for c in 1..rc.num_cascades {
        commands += rc.init_probes(c);
    }
    commands += rc.set_lifetime_base(&core);
    commands += rc.set_lifetime_secondary(&trace);
    for c in 1..rc.num_cascades {
        commands += rc.set_lifetime(c);
    }
    assert!(!rc.try_reset_table(&mut commands, false));
    let mut recovered = false;
    for _ in 0..4 {
        if rc.recover_probe_capacity(&core, Some(&trace), &mut commands) {
            recovered = true;
            break;
        }
    }
    assert!(recovered, "Small probe pool never recovered");
    assert!(rc.cache.lod_bias.read_blocking() > 0);
    let missing = Singleton::new(0u32);
    DEVICE
        .create_kernel::<fn()>(&track!(|| {
            let position = trace.traced_hits.read(dispatch_id().x).position;
            let probe = rc.cache.quantize(
                position,
                rc.cache.estimate_lod_secondary(position),
                0u32.expr(),
            );
            if rc.cache.get(probe).invalid() {
                missing.atomic().fetch_add(1u32);
            }
        }))
        .dispatch([64 * 64, 1, 1]);
    assert_eq!(
        missing.read_blocking(),
        0,
        "Recovery dropped secondary surfaces"
    );
}

#[test]
fn failed_frames_hide_gi_until_two_successful_frames() {
    let mut gate = GiAvailability::default();
    assert!(!gate.update(true));
    assert!(gate.update(true));
    assert!(!gate.update(false));
    assert!(!gate.update(true));
    assert!(!gate.update(false));
    assert!(!gate.update(true));
    assert!(gate.update(true));
}
