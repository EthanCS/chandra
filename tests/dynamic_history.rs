use chandra::algorithm::dynamic_history::DynamicHistory;
use chandra::algorithm::rc::{RcConfig, RcData};
use chandra::algorithm::trace::{Hit, TraceData};
use chandra::algorithm::{CommandList, CoreData, WorldData};
use chandra::dynamic::Bounds;
use chandra::world::TriangleWorld;
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;
use yesod::utils::dispatch0;

#[test]
fn motion_discards_affected_radiance_and_secondary_hits_while_lighting_decays_history() {
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
        num_cascades: 2,
        base_grid_size: 6.,
        ..Default::default()
    });
    let trace = TraceData::new(&core, &rc, 192);
    let history = DynamicHistory::new(&rc, &trace);
    let data_offset = Singleton::new(0u32);
    let mut commands = CommandList::new();
    rc.reset_table(&mut commands, true);
    dispatch0(
        track!(|| {
            let probe = rc
                .cache
                .quantize(Vec3::splat_expr(1.), 0u32.expr(), 0u32.expr());
            rc.cache.insert(probe, |index| {
                rc.probe_lifetime.write(index, 100u32);
            });
            data_offset.write(rc.cache.get(probe).data_offset);
        }),
        [1, 1, 1],
    )
    .execute();
    rc.reset_table(&mut commands, false);
    dispatch0(
        track!(|| {
            let i = data_offset.read() + dispatch_id().x;
            rc.cache.weights.write(i, 4.);
            rc.cache.weighted_fluence.write(i, Vec4::splat(8.));
            rc.base_radiance.write(i, Vec3::splat(2.));
            if dispatch_id().x % 4 == 0 {
                rc.radiance.write(i / 4, Vec3::splat(2.));
            }
        }),
        [32, 1, 1],
    )
    .execute();
    let hit = Hit {
        position: Vec3::splat(3.),
        normal: Vec3::new(0., 0., 1.),
    };
    trace.traced_hits.copy_from_vec(vec![hit; 64]).execute();
    history.update(&rc, &[], 0.5).execute();
    let offset = data_offset.read_blocking() as usize;
    assert!(
        rc.cache
            .weights
            .view(offset..offset + 32)
            .copy_to_vec()
            .iter()
            .all(|&w| w == 2.)
    );
    assert!(trace.traced_hits.copy_to_vec()[0].position.x.is_finite());
    history
        .update(
            &rc,
            &[Bounds {
                min: glam::Vec3::splat(-1.),
                max: glam::Vec3::splat(8.),
            }],
            1.,
        )
        .execute();
    assert!(
        rc.cache
            .weights
            .view(offset..offset + 32)
            .copy_to_vec()
            .iter()
            .all(|&w| w == 0.)
    );
    assert!(
        rc.base_radiance
            .view(offset..offset + 32)
            .copy_to_vec()
            .iter()
            .all(|v| v.x == 0. && v.y == 0. && v.z == 0.)
    );
    assert!(
        rc.radiance
            .view(offset / 4..offset / 4 + 8)
            .copy_to_vec()
            .iter()
            .all(|v| v.x == 0.)
    );
    assert!(
        trace
            .traced_hits
            .copy_to_vec()
            .iter()
            .all(|h| h.position.x.is_infinite())
    );

    // More than 16 independently moving objects must not merge into one giant
    // box and invalidate stationary hits lying in the space between them.
    let many = DynamicHistory::with_region_capacity(&rc, &trace, 100);
    let mut regions: Vec<_> = (0..99)
        .map(|i| Bounds {
            min: glam::Vec3::splat(10000. + i as f32 * 20.),
            max: glam::Vec3::splat(10005. + i as f32 * 20.),
        })
        .collect();
    regions.push(Bounds {
        min: glam::Vec3::splat(2.),
        max: glam::Vec3::splat(4.),
    });
    let mut hits = vec![hit; 64];
    hits[1].position = Vec3::splat(5000.);
    trace.traced_hits.copy_from_vec(hits).execute();
    many.update(&rc, &regions, 1.).execute();
    let hits = trace.traced_hits.copy_to_vec();
    assert!(
        hits[0].position.x.is_infinite(),
        "The 100th region was ignored"
    );
    assert_eq!(hits[1].position.x, 5000., "Separate bounds were collapsed");
}
