use chandra::algorithm::rc::{Ray, RcConfig, RcData};
use chandra::algorithm::trace::TraceData;
use chandra::algorithm::{CommandList, CoreData, WorldData};
use chandra::cache::Probe;
use chandra::world::TriangleWorld;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter::rtx::SurfaceHit;

#[test]
fn exhausted_probe_pool_reports_failure_without_cross_cascade_writes() {
    let table = chandra::table::HashTable::new(2, 16, [2, 2].into_iter());
    table.freelist.copy_from(&[0, 1, 2, 3]);
    table.freelist_index.copy_from(&[0, 2]);
    table.freelist_end.copy_from(&[2, 4]);
    let output = DEVICE.create_buffer::<u32>(4);
    let insert = DEVICE.create_kernel::<fn()>(&track!(|| {
        table.insert(10u64.expr(), 0u32.expr(), |_| {});
        table.insert(11u64.expr(), 0u32.expr(), |_| {});
        table.insert(12u64.expr(), 0u32.expr(), |_| {});
        table.insert(20u64.expr(), 1u32.expr(), |_| {});
        output.write(0, table.get(10u64.expr(), 0u32.expr()));
        output.write(1, table.get(11u64.expr(), 0u32.expr()));
        output.write(2, table.get(12u64.expr(), 0u32.expr()));
        output.write(3, table.get(20u64.expr(), 1u32.expr()));
    }));
    insert.dispatch([1, 1, 1]);
    assert_eq!(table.allocation_failures.copy_to_vec(), vec![1]);
    assert_eq!(output.copy_to_vec(), vec![0, 1, u32::MAX, 2]);
    assert_eq!(table.data_key.copy_to_vec(), vec![10, 11, 20, u64::MAX]);
}

#[test]
fn fused_transport_matches_reference_and_divergent_ray_queue_has_no_lost_samples() {
    let world = TriangleWorld::new(
        &[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]],
        &[[0, 1, 2]],
        &[[0, 0]],
        &[0],
        &[Vec3::splat(0.7)],
    );
    let core = CoreData::new(
        [8, 8],
        WorldData {
            world,
            envmap: image::Rgb32FImage::new(2, 2),
            envmap_scale: 0.,
            sun_dir: Vec3::new(0., 0., 1.),
            sun_color: Vec3::splat(1.),
            custom_emitters: vec![],
        },
    );
    let mut rc = RcData::new(RcConfig {
        base_probes: 256,
        num_cascades: 3,
        ..Default::default()
    });
    let mut commands = CommandList::new();
    rc.reset_table(&mut commands, true);
    let initialize = DEVICE.create_kernel::<fn()>(&track!(|| {
        for lod in 0..2u32 {
            for cascade in 0..3u32 {
                let probe = rc.cache.quantize(Vec3::splat_expr(1.), lod, cascade);
                rc.cache
                    .insert(probe, |index| rc.probe_lifetime.write(index, 100u32));
            }
        }
    }));
    initialize.dispatch([1, 1, 1]);
    rc.reset_table(&mut commands, false);
    let rays: Vec<_> = (0..96u32)
        .map(|i| Ray {
            base_probe: Probe {
                position: Vec4::new(1 << 17, 1 << 17, 1 << 17, i % 2),
                cascade: 0,
            },
            radiance: Vec3::new(
                (i % 7) as f32 / 16.,
                (i % 11) as f32 / 32.,
                (i % 13) as f32 / 16.,
            ),
            cascade: i % 3,
            dir: Vec2::new(
                ((i % 8) as f32 + 0.5) / 8.,
                (((i / 8) % 4) as f32 + 0.5) / 4.,
            ),
        })
        .collect();
    let rays = DEVICE.create_buffer_from_slice(&rays);
    let reference = DEVICE.create_kernel::<fn()>(&track!(|| {
        let ray = rays.read(dispatch_id().x);
        for cascade in 0..3u32 {
            if cascade > ray.cascade {
                break;
            }
            let node = rc.cache.get(ray.base_probe.shift_up(cascade));
            let encoder = rc.cache.get_encoder(cascade);
            let value = if cascade >= ray.cascade {
                ray.radiance.extend(0.0)
            } else {
                Vec3::splat_expr(0.0f32).extend(1.0)
            };
            rc.cache.deposit(
                node.data_offset + encoder.grid_to_linear(encoder.quantize(ray.dir)),
                value,
            );
            rc.probe_quality.atomic_ref(node.index).fetch_add(1.0f32);
        }
    }));
    reference.dispatch([96, 1, 1]);
    let expected_fluence = rc.cache.weighted_fluence.copy_to_vec();
    let expected_weights = rc.cache.weights.copy_to_vec();
    let expected_quality = rc.probe_quality.copy_to_vec();
    let clear = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        rc.cache.weighted_fluence.write(i, Vec4::splat(0.0f32));
        rc.cache.weights.write(i, 0.0f32);
        if i < rc.probe_quality.len() as u32 {
            rc.probe_quality.write(i, 0.0f32);
        }
    }));
    clear.dispatch([rc.cache.weights.len() as u32, 1, 1]);
    // Two contributions in one invocation also check that terminating the first
    // deposit does not return from its caller and drop the next LOD contribution.
    let fused = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x * 2u32;
        rc.deposit_ray_linked(rays.read(i));
        rc.deposit_ray_linked(rays.read(i + 1u32));
    }));
    fused.dispatch([48, 1, 1]);
    assert_eq!(rc.cache.weighted_fluence.copy_to_vec(), expected_fluence);
    assert_eq!(rc.cache.weights.copy_to_vec(), expected_weights);
    assert_eq!(rc.probe_quality.copy_to_vec(), expected_quality);

    let trace = TraceData::new(&core, &rc, 256);
    let miss = DEVICE.create_buffer_from_slice(&[SurfaceHit {
        inst: u32::MAX,
        prim: u32::MAX,
        bary: Vec2::splat(0.),
        committed_ray_t: f32::INFINITY,
    }]);
    let append = DEVICE.create_kernel::<fn()>(&track!(|| {
        set_block_size([64, 1, 1]);
        let i = dispatch_id().x;
        if i % 3u32 == 0u32 {
            return;
        }
        for lod in 0..2u32 {
            trace.compute_deposit_ray(
                &rc,
                Vec3::splat_expr(1.),
                lod,
                Vec3::new(0., 0., 1.).expr(),
                Vec3::splat_expr(i.cast_f32()),
                miss.read(0u32),
                1.0f32.expr(),
            );
        }
    }));
    append.dispatch([128, 1, 1]);
    let count = trace.next_traced_ray.read_blocking();
    assert_eq!(count, 170);
    let mut actual: Vec<_> = trace
        .traced_rays
        .view(..count as usize)
        .copy_to_vec()
        .iter()
        .map(|r| (r.radiance.x as u32, r.base_probe.position.w))
        .collect();
    let mut expected: Vec<_> = (0..128u32)
        .filter(|i| i % 3 != 0)
        .flat_map(|i| [(i, 0), (i, 1)])
        .collect();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);

    let direction_error = DEVICE.create_buffer::<f32>(rc.cache.sample_directions.len());
    let check = DEVICE.create_kernel::<fn(u32)>(&track!(|cascade| {
        let i = dispatch_id().x;
        let cached = rc.cache.direction(cascade, i);
        let original = rc.cache.get_encoder(cascade).linear_to_dir(i);
        direction_error.write(
            rc.cache.direction_index(cascade, i),
            (cached - original).abs().reduce_max(),
        );
    }));
    for cascade in 0..3 {
        check.dispatch([rc.cache.directions_per_probe(cascade), 1, 1], &cascade);
    }
    assert!(direction_error.copy_to_vec().iter().all(|&e| e <= 1e-7));
}

#[test]
fn opaque_any_hit_agrees_with_closest_hit_visibility_for_both_faces_and_misses() {
    let world = TriangleWorld::new(
        &[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]],
        &[[0, 1, 2]],
        &[[0, 0]],
        &[0],
        &[Vec3::splat(0.7)],
    );
    let errors = Singleton::new(0u32);
    let query = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let z = (i % 2u32 == 0u32).select(5.0f32.expr(), (-5.0f32).expr());
        let origin = Vec3::expr(
            (i % 64u32).cast_f32() / 16. - 1.,
            (i / 64u32).cast_f32() / 16. - 1.,
            z,
        );
        let dir = Vec3::expr(0.1f32, 0.0f32, -z.signum()).normalize();
        if world.occluded(origin, dir) != !world.trace(origin, dir).miss() {
            errors.atomic().fetch_add(1u32);
        }
    }));
    query.dispatch([4096, 1, 1]);
    assert_eq!(errors.read_blocking(), 0);
}
