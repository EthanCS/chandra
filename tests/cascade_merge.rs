use chandra::algorithm::CommandList;
use chandra::algorithm::rc::{RcConfig, RcData};
use chandra::cache::Probe;
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;

#[test]
fn sparse_neighbours_with_divergent_hash_lookups_match_scalar_interpolation() {
    let mut rc = RcData::new(RcConfig {
        base_probes: 4096,
        num_cascades: 2,
        quality_decay: 1.0,
        ..Default::default()
    });
    let probes: Vec<_> = (0..64)
        .map(|i| Probe {
            position: Vec4::new(
                (1 << 17) + i * 8 + (i & 1),
                (1 << 17) + ((i >> 1) & 1),
                (1 << 17) + ((i >> 2) & 1),
                0,
            ),
            cascade: 0,
        })
        .collect();
    let probes = DEVICE.create_buffer_from_slice(&probes);
    let expected = DEVICE.create_buffer::<Vec3<f32>>(64);
    let mut commands = CommandList::new();
    rc.reset_table(&mut commands, true);
    let insert = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let probe = probes.read(i);
        rc.cache
            .insert(probe, |index| rc.probe_lifetime.write(index, 100u32));
        let mask = (i * 73u32 + 19u32) & 255u32;
        for corner in 0u32..8 {
            if (mask & (1u32 << corner)) != 0u32 {
                rc.cache.insert(probe.next_trilinear(corner).0, |index| {
                    rc.probe_lifetime.write(index, 100u32)
                });
            }
        }
    }));
    insert.dispatch([64, 1, 1]);
    rc.reset_table(&mut commands, false);
    let initialize = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let probe = probes.read(i);
        let base = rc.cache.get(probe);
        for d in 0u32..32 {
            rc.cache.weights.write(base.data_offset + d, 1.0f32);
            rc.cache
                .weighted_fluence
                .write(base.data_offset + d, Vec4::new(0., 0., 0., 1.));
        }
        let sum = Vec3::splat(0.0f32).var();
        let weights = 0.0f32.var();
        for corner in 0u32..8 {
            let (next, weight) = probe.next_trilinear(corner);
            let next = rc.cache.get(next);
            if next.loaded() {
                let quality = 1.0 + corner.cast_f32() * 0.25;
                let color = Vec3::expr(
                    (i + 1u32).cast_f32() * 0.1,
                    corner.cast_f32() + 0.5,
                    0.75f32,
                );
                rc.probe_quality.write(next.index, quality);
                for d in 0u32..32 {
                    rc.radiance.write(next.data_offset / 4u32 + d, color);
                }
                *sum += weight * quality * color;
                *weights += weight * quality;
            }
        }
        expected.write(i, sum / keter::max(**weights, 1e-5f32));
    }));
    initialize.dispatch([64, 1, 1]);
    rc.merge_cache(0).execute();
    let errors = Singleton::new(0u32);
    let compare = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let node = rc.cache.get(probes.read(i));
        for d in 0u32..32 {
            let actual = rc.base_radiance.read(node.data_offset + d);
            if !actual.is_finite().all() || (actual - expected.read(i)).abs().reduce_max() > 1e-5f32
            {
                errors.atomic().fetch_add(1u32);
            }
        }
    }));
    compare.dispatch([64, 1, 1]);
    assert_eq!(errors.read_blocking(), 0);
}
