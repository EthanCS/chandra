use chandra::algorithm::{CoreData, WorldData};
use chandra::world::{PbrMaterial, TriangleWorld};
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;

#[test]
fn cached_reflection_convolution_preserves_constants_and_clears_cold_probes() {
    use chandra::algorithm::CommandList;
    use chandra::algorithm::cached_specular::CachedSpecular;
    use chandra::algorithm::rc::{RcConfig, RcData};
    let world = TriangleWorld::new(
        &[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        &[[0, 1, 2]],
        &[[0, 0]],
        &[0],
        &[Vec3::splat(0.5)],
    );
    let core = CoreData::new(
        [1, 1],
        WorldData {
            world,
            envmap: image::Rgb32FImage::new(1, 1),
            envmap_scale: 0.,
            sun_dir: Vec3::z(),
            sun_color: Vec3::splat(0.),
            custom_emitters: vec![],
        },
    );
    let mut rc = RcData::new(RcConfig {
        base_probes: 256,
        num_cascades: 2,
        use_quality_weight: false,
        ..Default::default()
    });
    let mut commands = CommandList::new();
    rc.reset_table(&mut commands, true);
    let insert = DEVICE.create_kernel::<fn()>(&track!(|| {
        let pos = Vec3::splat_expr(0.1);
        rc.cache.iter_lods(pos, |lod, _, _| {
            let probes = rc.cache.quantize_trilinear(pos, lod, 0u32.expr());
            for j in 0u32..8 {
                let (p, _) = probes.get(j);
                rc.cache.insert(
                    chandra::cache::Probe::from_comps_expr(chandra::cache::ProbeComps {
                        position: p.extend(lod),
                        cascade: 0u32.expr(),
                    }),
                    |i| rc.probe_lifetime.write(i, 100u32),
                );
            }
        });
    }));
    insert.dispatch([1, 1, 1]);
    rc.reset_table(&mut commands, false);
    let data = CachedSpecular::new(&core, &rc, 2);
    let result = DEVICE.create_buffer::<Vec4<f32>>(2048);
    let check = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let z = (i.cast_f32() + 0.5) / 1024.0 - 1.0;
        let phi = i.cast_f32() * 2.39996323;
        let dir = Vec3::expr(
            (1.0 - z * z).sqrt() * phi.cos(),
            (1.0 - z * z).sqrt() * phi.sin(),
            z,
        );
        let r = (i % 129u32).cast_f32() / 128.0;
        result.write(i, data.lookup(&rc, Vec3::splat_expr(0.1), dir, r));
    }));
    rc.base_radiance
        .copy_from(&vec![Vec3::new(0.25, 0.5, 1.0); rc.base_radiance.len()]);
    data.prefilter(&rc).execute();
    check.dispatch([2048, 1, 1]);
    for p in result.copy_to_vec() {
        assert!(
            (p.x - 0.25).abs() < 1e-5
                && (p.y - 0.5).abs() < 1e-5
                && (p.z - 1.0).abs() < 1e-5
                && p.w > 0.99,
            "Constant energy / octahedral seam: {p:?}"
        );
    }
    rc.base_radiance
        .copy_from(&vec![Vec3::splat(f32::NAN); rc.base_radiance.len()]);
    data.prefilter(&rc).execute();
    check.dispatch([2048, 1, 1]);
    assert!(
        result.copy_to_vec().iter().all(|v| *v == Vec4::splat(0.)),
        "Cold/recycled probe retained stale reflected light"
    );
}

#[test]
fn cached_reflections_cover_odd_sizes_and_keep_constant_environment_after_history_reset() {
    use chandra::algorithm::cached_specular::CachedSpecular;
    use chandra::algorithm::rc::{RcConfig, RcData};
    use chandra::algorithm::splat::SplatData;
    let mut world = TriangleWorld::new(
        &[[-100., -100., 0.], [100., -100., 0.], [0., 100., 0.]],
        &[[0, 1, 2]],
        &[[0, 0]],
        &[0],
        &[Vec3::splat(1.)],
    );
    let mut metal = PbrMaterial::default();
    metal.parameters = Vec4::new(0.3, 1., 0.04, 1.);
    world.set_pbr(&[metal], &[[0., 0., 1.]; 3], &[[0.; 2]; 3], &[0.], vec![]);
    let core = CoreData::new(
        [17, 13],
        WorldData {
            world,
            envmap: image::Rgb32FImage::from_pixel(4, 2, image::Rgb([0.25, 0.5, 1.])),
            envmap_scale: 1.,
            sun_dir: Vec3::z(),
            sun_color: Vec3::splat(0.),
            custom_emitters: vec![],
        },
    );
    let camera = yesod::camera::Camera {
        pos: glam::Vec3::new(0., -2., 2.),
        yaw: 0.,
        pitch: 0.7853982,
        screen_size: glam::Vec2::new(17., 13.),
        fov: 0.6,
    }
    .view();
    core.view.write_host(camera).execute();
    core.compute_gbuffer().execute();
    let rc = RcData::new(RcConfig {
        base_probes: 256,
        num_cascades: 2,
        ..Default::default()
    });
    let splat = SplatData::new(&core, &rc, 1.);
    let diffuse = chandra::algorithm::diffuse::DiffuseData::with_sample_limit(&core, 8);
    diffuse.prepare(&core, 0, camera, false, 2, 0).execute();
    assert!(
        diffuse.samples.copy_to_vec().iter().all(|v| *v == 8),
        "Cold RC update escaped its budget"
    );
    diffuse.prepare(&core, 0, camera, false, 2, 32).execute();
    assert!(
        diffuse.samples.copy_to_vec().iter().all(|v| *v == 32),
        "Explicit reference override was ignored"
    );
    for scale in [1, 2, 4] {
        let mut lighting = core.lighting.read_blocking();
        lighting.environment_scale = 1.;
        core.lighting.write_host(lighting).execute();
        let data = CachedSpecular::new(&core, &rc, scale);
        let first = std::cell::RefCell::new(Vec::new());
        for frame in 0..6 {
            data.trace(&core, &rc, &splat, frame, false).execute();
            for (_, command) in data.reconstruct(
                &core,
                &rc,
                &splat,
                frame,
                camera,
                frame != 0 && frame != 3,
                2.,
                false,
            ) {
                command.execute();
            }
            let values = data.output.copy_to_vec();
            assert!(
                values
                    .iter()
                    .all(|v| v.x.is_finite() && v.x > 0.1 && v.z > 0.4),
                "Missing/black pixels at scale {scale}"
            );
            if frame == 0 {
                *first.borrow_mut() = values;
            } else {
                for (a, b) in values.iter().zip(first.borrow().iter()) {
                    assert!(
                        (a.x - b.x).abs() < 1e-5 && (a.z - b.z).abs() < 1e-5,
                        "Reset or sampling phase introduced flicker"
                    );
                }
            }
            assert!(
                data.diagnostics.copy_to_vec().iter().all(|v| v.w == 4.),
                "History loss changed fixed ray budget"
            );
        }
        let mut lighting = core.lighting.read_blocking();
        lighting.environment_scale = 0.;
        core.lighting.write_host(lighting).execute();
        for frame in 6..32 {
            data.trace(&core, &rc, &splat, frame, false).execute();
            for (_, command) in data.reconstruct(&core, &rc, &splat, frame, camera, true, 2., false)
            {
                command.execute();
            }
            let values = data.output.copy_to_vec();
            assert!(
                values
                    .iter()
                    .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite() && v.x >= 0.),
                "Fade to black generated nonfinite radiance"
            );
            data.composition.copy_from(
                &values
                    .iter()
                    .map(|v| Vec4::new(v.x + 0.1, v.y + 0.2, v.z + 0.3, 1.))
                    .collect::<Vec<_>>(),
            );
            data.compose(&core, frame, camera, frame > 6).execute();
            for value in data.composition.copy_to_vec() {
                assert!(
                    value.x.is_finite()
                        && value.y.is_finite()
                        && value.z.is_finite()
                        && value.x >= 0.
                        && value.y >= 0.
                        && value.z >= 0.,
                    "Perceptual composition must remain finite through a fade"
                );
            }
        }
    }
}
