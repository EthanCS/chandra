use chandra::algorithm::{CoreData, WorldData, specular_brdf as brdf};
use chandra::world::{PbrMaterial, TriangleWorld};
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::rand::pcg3df;

#[test]
fn short_temporal_windows_preserve_mature_reflection_support_and_reset_on_disocclusion() {
    use chandra::algorithm::rc::{RcConfig, RcData};
    use chandra::algorithm::specular::SpecularData;
    let mut world = TriangleWorld::new(
        &[[-100., -100., 0.], [100., -100., 0.], [0., 100., 0.]],
        &[[0, 1, 2]], &[[0, 0]], &[0], &[Vec3::splat(1.)]);
    let mut metal = PbrMaterial::default();
    metal.parameters = Vec4::new(0.3, 1., 0.04, 1.);
    world.set_pbr(&[metal], &[[0., 0., 1.]; 3], &[[0.; 2]; 3], &[0.], vec![]);
    let core = CoreData::new([16, 16], WorldData {
        world, envmap: image::Rgb32FImage::from_pixel(4, 2, image::Rgb([1., 1., 1.])),
        envmap_scale: 1., sun_dir: Vec3::z(), sun_color: Vec3::splat(0.), custom_emitters: vec![],
    });
    let camera = yesod::camera::Camera {
        pos: glam::Vec3::new(0., -2., 2.), yaw: 0., pitch: 0.7853982,
        screen_size: glam::Vec2::splat(16.), fov: 0.6,
    }.view();
    core.view.write_host(camera).execute();
    core.compute_gbuffer().execute();
    let rc = RcData::new(RcConfig {base_probes: 256, num_cascades: 2, ..Default::default()});
    let data = SpecularData::new(&core);
    for limit in [1., 4. / 3., 2., 8. / 3., 4.] {
        for frame in 0..8 {
            data.trace(&core, &rc, None, frame, 4, false, false, true, camera, frame > 0).execute();
            let budgets = data.diagnostics.copy_to_vec();
            if frame == 0 {
                assert!(budgets.iter().all(|v| v.w == 128.), "Invalid history did not restore disocclusion sampling");
            }
            if frame >= 4 {
                assert!(budgets.iter().filter(|v| v.w == 16.).count() > 150,
                    "Short history limit {limit} kept mature metal at 64 rays");
            }
            data.reconstruct(&core, frame, camera, frame > 0, limit).execute();
            assert!(data.output.copy_to_vec().iter().all(|v| v.w <= limit + 1e-5));
        }
    }
}

#[test]
fn shared_reflection_shading_preserves_sun_point_shadows_and_both_material_lobes() {
    use chandra::algorithm::PointLight;
    for smooth in [false, true] {
        let mut world = TriangleWorld::new(
            &[
                [-20., -20., 0.],
                [60., -20., 0.],
                [-20., 60., 0.],
                [-8., -8., 5.],
                [8., -8., 5.],
                [-8., 8., 5.],
            ],
            &[[0, 1, 2], [0, 1, 2]],
            &[[0, 0], [1, 3]],
            &[0, 1],
            &[Vec3::new(0.6, 0.25, 0.12), Vec3::splat(0.5)],
        );
        let mut surface = PbrMaterial::default();
        surface.parameters = Vec4::new(0.35, 0.4, 0.04, 1.);
        surface.emission_alpha = Vec4::new(0.1, 0.02, 0., 1.);
        let normal = if smooth {
            [0.2, 0., 0.9797959]
        } else {
            [0., 0., 1.]
        };
        world.set_pbr(
            &[surface, PbrMaterial::default()],
            &[normal; 6],
            &[[0.; 2]; 6],
            &[0.; 2],
            vec![],
        );
        let mut core = CoreData::new(
            [16, 16],
            WorldData {
                world,
                envmap: image::Rgb32FImage::new(1, 1),
                envmap_scale: 0.,
                sun_dir: Vec3::new(0., 0., 1.),
                sun_color: Vec3::new(2., 1., 0.5),
                custom_emitters: vec![(0, Vec3::new(0.02, 0.01, 0.))],
            },
        );
        core.set_point_lights(
            &(0..8)
                .map(|i| PointLight {
                    position_range: Vec4::new(
                        i as f32 * 4. - 12.,
                        0.,
                        if i % 2 == 0 { 4. } else { 8. },
                        60.,
                    ),
                    intensity: Vec4::new(70., 40., 20., 0.),
                })
                .collect::<Vec<_>>(),
        );
        let results = DEVICE.create_buffer::<Vec4<f32>>(256);
        let query = DEVICE.create_kernel::<fn()>(&track!(|| {
            let i = dispatch_id().x;
            let start = Vec3::expr((i % 16u32).cast_f32() - 8., (i / 16u32).cast_f32() - 8., 2.);
            let direction = Vec3::new(0., 0., -1.).expr();
            let hit = core.world.trace(start, direction);
            let pos = core.world.get_position(hit);
            let n = core.world.get_shading_normal(hit, direction);
            let ng = core.world.get_normal_flat(hit, direction);
            let (r, f0, albedo) = core.world.get_pbr_lobes(hit);
            let expected = core.get_surface_irradiance(None, pos, hit, direction)
                * core.world.get_diffuse(hit)
                + core.direct_specular(pos, hit, -direction)
                + core.get_surface_emission(hit);
            let value = core.direct_reflection(pos, hit, -direction, n, ng, r, f0, albedo);
            let material_error = (f0 - core.world.get_f0(hit)).abs().reduce_max()
                + (albedo - core.world.get_diffuse(hit)).abs().reduce_max()
                + (r - core.world.get_roughness_metallic(hit).x).abs();
            results.write(
                i,
                Vec4::expr(
                    (value - expected).abs().reduce_max(),
                    expected.reduce_max(),
                    value.reduce_max(),
                    material_error,
                ),
            );
        }));
        for sun in [
            Vec3::new(0., 0., 1.),
            Vec3::new(0.6, 0., 0.8),
            Vec3::new(0.6, 0., -0.8),
        ] {
            let mut lighting = core.lighting.read_blocking();
            lighting.sun_dir = sun;
            core.lighting.write_host(lighting).execute();
            query.dispatch([256, 1, 1]);
            let output = results.copy_to_vec();
            assert!(
                output
                    .iter()
                    .all(|p| p.x < 1e-5 && p.w < 1e-6 && p.z.is_finite()),
                "Shared shading changed visibility or a material lobe: {:?}",
                output.iter().max_by(|a, b| a.x.total_cmp(&b.x))
            );
            assert!(output.iter().any(|p| p.y > 0.2));
        }
    }
}

#[test]
fn ggx_visible_normal_sampling_matches_hemisphere_integration_and_mirror_energy() {
    let count = 131072u32;
    let values = DEVICE.create_buffer::<Vec4<f32>>(count as usize);
    let sample = DEVICE.create_kernel::<fn(f32, f32)>(&track!(|rough, nv| {
        let i = dispatch_id().x;
        let n = Vec3::new(0., 0., 1.).expr();
        let v = Vec3::expr((1.0 - nv * nv).sqrt(), 0., nv);
        let u = pcg3df(Vec3::expr(i, 7u32, 11u32)).xy();
        let (l, w) = brdf::sample(n, v, rough, Vec3::splat_expr(0.04), u);
        let z = (i.cast_f32() + 0.5) / count as f32;
        let phi = u.y * std::f32::consts::TAU;
        let q = Vec3::expr(
            (1.0 - z * z).sqrt() * phi.cos(),
            (1.0 - z * z).sqrt() * phi.sin(),
            z,
        );
        let integral =
            brdf::evaluate(n, v, q, rough, Vec3::splat_expr(0.04)).x * std::f32::consts::TAU;
        values.write(i, Vec4::expr(w.x, integral, l.length(), w.reduce_max()));
    }));
    for (rough, nv) in [(0.3, 1.), (0.6, 1.), (0.9, 0.25), (0.5, 0.1)] {
        sample.dispatch([count, 1, 1], &rough, &nv);
        let data = values.copy_to_vec();
        assert!(
            data.iter()
                .all(|p| p.x.is_finite() && p.x >= 0. && p.w <= 1.001 && (p.z - 1.).abs() < 1e-4)
        );
        let a = data.iter().map(|p| p.x as f64).sum::<f64>() / count as f64;
        let b = data.iter().map(|p| p.y as f64).sum::<f64>() / count as f64;
        assert!(
            (a - b).abs() < 0.002,
            "GGX mismatch r={rough} nv={nv}: VNDF={a}, quadrature={b}"
        );
    }
    sample.dispatch([count, 1, 1], &0.0, &1.0);
    let mean = values.copy_to_vec().iter().map(|p| p.x).sum::<f32>() / count as f32;
    assert!(
        (mean - 0.04).abs() < 0.0002,
        "Mirror Fresnel energy: {mean}"
    );
}

#[test]
fn pbr_cutout_is_consistent_for_primary_reflections_and_shadow_rays() {
    let positions = [
        [-2., -2., 1.],
        [2., -2., 1.],
        [0., 2., 1.],
        [-2., -2., 0.],
        [2., -2., 0.],
        [0., 2., 0.],
    ];
    let mut world = TriangleWorld::new(
        &positions,
        &[[0, 1, 2], [0, 1, 2]],
        &[[0, 0], [1, 3]],
        &[0, 1],
        &[Vec3::splat(1.), Vec3::new(1., 0., 0.)],
    );
    let mut mask = PbrMaterial::default();
    mask.alpha_cutoff = 0.5;
    mask.emission_alpha.w = 0.;
    let mut metal = PbrMaterial::default();
    metal.parameters.y = 1.;
    metal.parameters.x = 0.2;
    world.set_pbr(
        &[mask, metal],
        &[[0., 0., 1.]; 6],
        &[[0.; 2]; 6],
        &[0.; 2],
        vec![],
    );
    let output = DEVICE.create_buffer::<Vec4<f32>>(1);
    let test = DEVICE.create_kernel::<fn()>(&track!(|| {
        let o = Vec3::new(0., 0., 2.).expr();
        let d = Vec3::new(0., 0., -1.).expr();
        let hit = world.trace(o, d);
        let near = world.occluded_segment(o, d, 1.5f32.expr());
        let far = world.occluded_segment(o, d, 3.0f32.expr());
        output.write(
            0u32,
            Vec4::expr(
                hit.inst.cast_f32(),
                near.cast_u32().cast_f32(),
                far.cast_u32().cast_f32(),
                world.get_diffuse(hit).reduce_max(),
            ),
        );
    }));
    test.dispatch([1, 1, 1]);
    let p = output.copy_to_vec()[0];
    assert_eq!([p.x, p.y, p.z, p.w], [1., 0., 1., 0.]);
}

#[test]
fn cutout_traversal_rejects_more_than_thirty_two_transparent_layers() {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut offsets = Vec::new();
    let mut materials = Vec::new();
    for i in 0..41u32 {
        let z = if i == 40 { 0. } else { 0.05 + i as f32 * 0.04 };
        offsets.push([i, i * 3]);
        positions.extend([[-2., -2., z], [2., -2., z], [0., 2., z]]);
        indices.push([0, 1, 2]);
        materials.push(if i == 40 { 1 } else { 0 });
    }
    let mut world = TriangleWorld::new(
        &positions,
        &indices,
        &offsets,
        &materials,
        &[Vec3::splat(1.); 2],
    );
    let mut cutout = PbrMaterial::default();
    cutout.alpha_cutoff = 0.5;
    cutout.emission_alpha.w = 0.;
    world.set_pbr(
        &[cutout, PbrMaterial::default()],
        &vec![[0., 0., 1.]; positions.len()],
        &vec![[0.; 2]; positions.len()],
        &vec![0.; indices.len()],
        vec![],
    );
    let output = DEVICE.create_buffer::<Vec4<f32>>(1);
    let check = DEVICE.create_kernel::<fn()>(&track!(|| {
        let start = Vec3::new(0., 0., 2.).expr();
        let direction = Vec3::new(0., 0., -1.).expr();
        let hit = world.trace(start, direction);
        output.write(
            0u32,
            Vec4::expr(
                hit.inst.cast_f32(),
                hit.committed_ray_t,
                world
                    .occluded_segment(start, direction, 1.9f32.expr())
                    .cast_u32()
                    .cast_f32(),
                world
                    .occluded_segment(start, direction, 3.0f32.expr())
                    .cast_u32()
                    .cast_f32(),
            ),
        );
    }));
    check.dispatch([1, 1, 1]);
    let p = output.copy_to_vec()[0];
    assert_eq!([p.x, p.y, p.z, p.w], [40., 2., 0., 1.]);
}

#[test]
fn reflection_ray_sees_offscreen_surface() {
    // The test has no screen-space source: radiance comes from the ray hit.
    let positions = [
        [-10., -10., 0.],
        [10., -10., 0.],
        [0., 10., 0.],
        [-10., 5., 0.],
        [0., 5., 20.],
        [10., 5., 0.],
    ];
    let mut world = TriangleWorld::new(
        &positions,
        &[[0, 1, 2], [0, 1, 2]],
        &[[0, 0], [1, 3]],
        &[0, 1],
        &[Vec3::splat(1.), Vec3::new(0.8, 0.02, 0.01)],
    );
    let mut mirror = PbrMaterial::default();
    mirror.parameters = Vec4::new(0., 1., 0.04, 1.);
    world.set_pbr(
        &[mirror, PbrMaterial::default()],
        &[[0., 0., 1.]; 6],
        &[[0.; 2]; 6],
        &[0.; 2],
        vec![],
    );
    let core = CoreData::new(
        [1, 1],
        WorldData {
            world,
            envmap: image::Rgb32FImage::new(1, 1),
            envmap_scale: 0.,
            sun_dir: Vec3::new(0., -1., 0.),
            sun_color: Vec3::splat(1.),
            custom_emitters: vec![],
        },
    );
    let output = DEVICE.create_buffer::<Vec4<f32>>(1);
    let test = DEVICE.create_kernel::<fn()>(&track!(|| {
        let o = Vec3::new(0., 0., 0.01).expr();
        let d = Vec3::<f32>::new(0., 1., 1.).expr().normalize();
        let hit = core.world.trace(o, d);
        let value = core.get_radiance(None, o, d, hit);
        output.write(0u32, value.extend(hit.inst.cast_f32()));
    }));
    test.dispatch([1, 1, 1]);
    let p = output.copy_to_vec()[0];
    assert_eq!(p.w, 1.);
    assert!(
        p.x > 0.2 && p.x > p.y * 10.,
        "Offscreen red reflection is missing: {p:?}"
    );
}

#[test]
fn hybrid_pipeline_reflects_emission_and_discards_invalid_history() {
    use chandra::algorithm::rc::{RcConfig, RcData};
    use chandra::algorithm::specular::SpecularData;
    fn trace_sized(
        data: &SpecularData,
        core: &CoreData,
        rc: &RcData,
        frame: u32,
        samples: u32,
        camera: yesod::camera::View,
        valid: bool,
    ) {
        use chandra::algorithm::specular::SpecularTraceStage;
        let run = |stage| {
            data.trace_stages(
                core, rc, None, frame, samples, false, false, true, camera, valid, stage,
            )
            .into_iter()
            .map(|(_, node)| node)
            .collect::<Vec<_>>()
            .chain()
            .execute()
        };
        run(SpecularTraceStage::Base);
        run(SpecularTraceStage::Refinement(
            data.pending_refinement_count(),
        ));
    }
    fn assert_matches(data: &SpecularData, serial: &SpecularData) {
        assert_eq!(
            data.diagnostics.copy_to_vec(),
            serial.diagnostics.copy_to_vec(),
            "Batched tracing changed accepted rays or adaptive allocation"
        );
        for (a, b) in data.raw.copy_to_vec().iter().zip(serial.raw.copy_to_vec()) {
            assert!(
                (a.x - b.x).abs() < 1e-5
                    && (a.y - b.y).abs() < 1e-5
                    && (a.z - b.z).abs() < 1e-5
                    && a.w == b.w,
                "Batched tracing changed a reflection estimate: {a:?} versus {b:?}"
            );
        }
    }
    for (primary_roughness, metallic, secondary_roughness) in [
        (0., 1., 0.6),
        (0., 1., 0.2),
        (0.25, 1., 0.2),
        (0.65, 0., 0.2),
        (0.25, 0., 0.6),
    ] {
        let positions = [
            [-20., -20., 0.],
            [20., -20., 0.],
            [0., 20., 0.],
            [-10., 5., 0.],
            [0., 5., 20.],
            [10., 5., 0.],
        ];
        let mut world = TriangleWorld::new(
            &positions,
            &[[0, 1, 2], [0, 1, 2]],
            &[[0, 0], [1, 3]],
            &[0, 1],
            &[Vec3::splat(1.), Vec3::splat(0.)],
        );
        let mut mirror = PbrMaterial::default();
        mirror.parameters = Vec4::new(primary_roughness, metallic, 0.04, 1.);
        let mut emitter = PbrMaterial::default();
        emitter.parameters.x = secondary_roughness;
        emitter.emission_alpha = Vec4::new(2., 0.2, 0.05, 1.);
        world.set_pbr(
            &[mirror, emitter],
            &[
                [0., 0., 1.],
                [0., 0., 1.],
                [0., 0., 1.],
                [0., -1., 0.],
                [0., -1., 0.],
                [0., -1., 0.],
            ],
            &[[0.; 2]; 6],
            &[0.; 2],
            vec![],
        );
        let core = CoreData::new(
            [16, 16],
            WorldData {
                world,
                envmap: image::Rgb32FImage::new(1, 1),
                envmap_scale: 0.,
                sun_dir: Vec3::new(0., 0., 1.),
                sun_color: Vec3::splat(0.),
                custom_emitters: vec![],
            },
        );
        let camera = yesod::camera::Camera {
            pos: glam::Vec3::new(0., -2., 2.),
            yaw: 0.,
            pitch: 0.7853982,
            screen_size: glam::Vec2::splat(16.),
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
        let data = SpecularData::with_refinement_batch(&core, true, 1);
        trace_sized(&data, &core, &rc, 0, 4, camera, false);
        let serial = SpecularData::with_refinement_queue(&core, false);
        serial
            .trace(&core, &rc, None, 0, 4, false, false, true, camera, false)
            .execute();
        assert_matches(&data, &serial);
        for batch in [2, 4, 8, 16, 32] {
            let other = SpecularData::with_refinement_batch(&core, true, batch);
            other
                .trace(&core, &rc, None, 0, 4, false, false, true, camera, false)
                .execute();
            assert_matches(&other, &serial);
        }
        if primary_roughness > 0. {
            // Exercise fresh and mature history, non-power-of-two requests,
            // dielectric batches and metal fallbacks with the same Sobol prefix.
            for (frame, samples) in [4, 4, 4, 4, 4, 7, 16, 32].into_iter().enumerate() {
                let frame = frame as u32;
                for tracer in [&data, &serial] {
                    trace_sized(tracer, &core, &rc, frame, samples, camera, frame > 0);
                }
                assert_matches(&data, &serial);
                for tracer in [&data, &serial] {
                    tracer
                        .reconstruct(&core, frame, camera, frame > 0, 24.)
                        .execute();
                }
            }
            continue;
        }
        data.reconstruct(&core, 0, camera, false, 24.).execute();
        let reflected = data.output.copy_to_vec();
        let raw = data.raw.copy_to_vec();
        assert!(
            reflected
                .iter()
                .enumerate()
                .filter(|(_, p)| p.x > 1.0)
                .all(|(i, _)| (raw[i].w > 0.5) == (secondary_roughness < 0.45)),
            "A deterministic mirror ray hid its stochastic secondary reflection"
        );
        assert!(
            reflected
                .iter()
                .filter(|p| p.x > 1. && p.x > p.y * 5.)
                .count()
                > 64,
            "Mirror does not reflect the emissive wall"
        );
        data.trace(&core, &rc, None, 1, 4, false, false, true, camera, true)
            .execute();
        data.reconstruct(&core, 1, camera, true, 24.).execute();
        if secondary_roughness < 0.45 {
            assert!(data.output.copy_to_vec().iter().any(|p| p.w > 1.));
        }
        let current = data.output.copy_to_vec();
        for (i, p) in reflected.iter().enumerate().filter(|(_, p)| p.x > 1.0) {
            assert!(
                (current[i].x - p.x).abs() < 1e-5 && (current[i].y - p.y).abs() < 1e-5,
                "Reconstruction blurred a deterministic reflected emitter"
            );
        }
        // Even an exact mirror looking at a rough surface reads an RC
        // estimator. It needs history without filtering the emissive detail.
        data.trace(&core, &rc, None, 2, 4, true, false, true, camera, true)
            .execute();
        data.reconstruct(&core, 2, camera, true, 24.).execute();
        let cached = data.output.copy_to_vec();
        let cached_raw = data.raw.copy_to_vec();
        for (i, p) in reflected.iter().enumerate().filter(|(_, p)| p.x > 1.0) {
            assert!(
                cached_raw[i].w > 0.5,
                "RC-backed mirror was marked noise-free"
            );
            assert!(
                (cached[i].x - p.x).abs() < 1e-5,
                "Filtering cached mirror energy changed deterministic emission"
            );
        }
        let mut lighting = core.lighting.read_blocking();
        lighting.emission_scale = 0.;
        core.lighting.write_host(lighting).execute();
        data.trace(&core, &rc, None, 3, 4, false, false, true, camera, true)
            .execute();
        data.reconstruct(&core, 3, camera, true, 4.).execute();
        if secondary_roughness >= 0.45 {
            assert!(
                data.output
                    .copy_to_vec()
                    .iter()
                    .all(|p| p.x == 0. && p.y == 0. && p.z == 0.),
                "Neighborhood clipping retained a vanished reflected emitter"
            );
        }
        data.trace(&core, &rc, None, 4, 4, false, false, true, camera, false)
            .execute();
        data.reconstruct(&core, 4, camera, false, 24.).execute();
        assert!(
            data.output
                .copy_to_vec()
                .iter()
                .all(|p| p.x == 0. && p.y == 0. && p.z == 0.),
            "Invalid reflection history survived a lighting reset"
        );
    }
}

#[test]
fn adaptive_path_samples_keep_stratified_prefixes() {
    let output = DEVICE.create_buffer::<Vec4<f32>>(128);
    let sample = DEVICE.create_kernel::<fn(u32)>(&track!(|frame| {
        let i = dispatch_id().x;
        output.write(
            i,
            brdf::path_samples(Vec2::new(17u32, 93u32).expr(), frame, i),
        );
    }));
    let mut previous = None;
    for frame in [0u32, 1, 241, 8191] {
        sample.dispatch([128, 1, 1], &frame);
        let values = output.copy_to_vec();
        for n in [4usize, 8, 16, 32, 64, 128] {
            for dimension in 0..4 {
                let mut bins = vec![0u32; n];
                for v in &values[..n] {
                    let x = [v.x, v.y, v.z, v.w][dimension];
                    assert!(x >= 0. && x < 1. && x.is_finite());
                    bins[(x * n as f32) as usize] += 1;
                }
                assert!(
                    bins.iter().all(|&v| v == 1),
                    "Missing stratum at frame {frame}, budget {n}, dimension {dimension}"
                );
            }
        }
        if let Some(p) = previous {
            assert_ne!(values[0].x, p, "Frozen temporal samples");
        }
        previous = Some(values[0].x);
    }
}

#[test]
fn secondary_irradiance_atlas_preserves_constant_energy_and_bounds_directional_error() {
    use chandra::algorithm::CommandList;
    use chandra::algorithm::rc::{RcConfig, RcData};
    use chandra::algorithm::splat::SplatData;
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
        ..Default::default()
    });
    let mut commands = CommandList::new();
    rc.reset_table(&mut commands, true);
    let insert = DEVICE.create_kernel::<fn()>(&track!(|| {
        let p = Vec3::splat_expr(0.1);
        rc.cache.insert(
            rc.cache
                .quantize(p, rc.cache.estimate_lod_secondary(p), 0u32.expr()),
            |i| rc.probe_lifetime.write(i, 100u32),
        );
    }));
    insert.dispatch([1, 1, 1]);
    rc.reset_table(&mut commands, false);
    let splat = SplatData::new(&core, &rc, 1.0);
    let initialize = DEVICE.create_kernel::<fn(bool)>(&track!(|directional| {
        let i = dispatch_id().x;
        let d = rc.cache.get_encoder(0u32.expr()).linear_to_dir(i % 32u32);
        let value = if directional {
            Vec3::splat_expr(1.0) + d * 0.8
        } else {
            Vec3::new(0.25, 0.5, 1.0).expr()
        };
        rc.base_radiance.write(i, value);
    }));
    let values = DEVICE.create_buffer::<Vec4<f32>>(2048);
    let feedback = DEVICE.create_buffer::<Vec4<f32>>(1);
    let check_feedback = DEVICE.create_kernel::<fn()>(&track!(|| {
        let start = Vec3::new(0.1, 0.1, 1.).expr();
        let dir = Vec3::new(0., 0., -1.).expr();
        let hit = core.world.trace(start, dir);
        let exact = core.get_radiance(Some(&rc), start, dir, hit);
        let fast = core.get_radiance_cached(Some(&rc), Some(&splat), start, dir, hit);
        feedback.write(0u32, (fast - exact).extend(fast.reduce_max()));
    }));
    let check = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let z = (i.cast_f32() + 0.5) / 1024.0 - 1.0;
        let phi = i.cast_f32() * 2.39996323;
        let n = Vec3::expr(
            (1.0 - z * z).sqrt() * phi.cos(),
            (1.0 - z * z).sqrt() * phi.sin(),
            z,
        );
        let p = Vec3::splat_expr(0.1);
        let exact = rc.sample_secondary(p, n);
        let fast = splat.sample_secondary(&rc, p, n);
        values.write(i, (fast - exact).extend(fast.reduce_min()));
    }));
    for directional in [false, true] {
        initialize.dispatch([rc.base_radiance.len() as u32, 1, 1], &directional);
        splat.calculate_irradiance(&rc).execute();
        splat.copy_irradiance_borders(&rc).execute();
        check_feedback.dispatch([1, 1, 1]);
        let value = feedback.copy_to_vec()[0];
        assert!(value.w.is_finite() && value.w > 0.0);
        assert!(value.x.abs().max(value.y.abs()).max(value.z.abs()) < if directional {0.05} else {0.001});
        check.dispatch([2048, 1, 1]);
        let data = values.copy_to_vec();
        let max_error = data
            .iter()
            .flat_map(|v| [v.x.abs(), v.y.abs(), v.z.abs()])
            .fold(0.0f32, f32::max);
        assert!(data.iter().all(|v| v.w.is_finite() && v.w > 0.0));
        // A coarse six-texel normal atlas is approximate: bound its absolute
        // directional error to 10% of the unit mean, while constants stay exact.
        assert!(
            max_error < if directional { 0.1 } else { 0.001 },
            "Secondary irradiance error {max_error}, directional={directional}"
        );
    }
    // Recycled/cold probes must not retain a previous atlas tile's energy.
    rc.base_radiance.copy_from(&vec![Vec3::splat(f32::NAN); rc.base_radiance.len()]);
    splat.calculate_irradiance(&rc).execute();
    splat.copy_irradiance_borders(&rc).execute();
    check_feedback.dispatch([1, 1, 1]);
    let value = feedback.copy_to_vec()[0];
    assert_eq!(value, Vec4::splat(0.));
}
