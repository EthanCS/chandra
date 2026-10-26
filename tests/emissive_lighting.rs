use chandra::algorithm::{CoreData, Lighting, PointLight, WorldData};
use chandra::world::TriangleWorld;
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;

#[test]
fn current_frame_light_grid_matches_full_loop_across_moves_and_range_changes() {
    let make = || CoreData::new([1,1], WorldData {
        world: TriangleWorld::new(&[[-2000.,-2000.,0.],[6000.,-2000.,0.],[-2000.,6000.,0.]],
            &[[0,1,2]], &[[0,0]], &[0], &[Vec3::splat(0.7)]),
        envmap: image::Rgb32FImage::new(1,1), envmap_scale: 0., sun_dir: Vec3::new(0.,0.,1.),
        sun_color: Vec3::splat(0.), custom_emitters: vec![],
    });
    let lights = |step: f32| (0..32).map(|i| PointLight {
        position_range: Vec4::new((i % 8) as f32 * 200. - 600. + step * 71., (i / 8) as f32 * 230. - 500., 130. + step * 11., 220. + (i % 5) as f32 * 90. + step * 5.),
        intensity: Vec4::new(2000. * i as f32, 10000., 3000., 0.),
    }).collect::<Vec<_>>();
    let mut reference = make();
    let mut accelerated = make();
    reference.set_point_lights_grid(&lights(0.), false);
    accelerated.set_point_lights(&lights(0.));
    let errors = Singleton::new(0u32);
    let compare = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let origin = Vec3::expr((i % 128u32).cast_f32() * 20. - 1200., (i / 128u32).cast_f32() * 20. - 1200., 100.0f32);
        let direction = Vec3::new(0.,0.,-1.).expr();
        let hit = reference.world.trace(origin, direction);
        if hit.miss() { return; }
        let a = reference.get_radiance(None, origin, direction, hit);
        let b = accelerated.get_radiance(None, origin, direction, hit);
        if (a - b).abs().reduce_max() > 1e-6f32 { errors.atomic().fetch_add(1u32); }
    }));
    for step in [-4.,0.,1.,9.] {
        (reference.update_point_lights(lights(step)), accelerated.update_point_lights(lights(step))).chain().execute();
        compare.dispatch([128*128,1,1]);
        assert_eq!(errors.read_blocking(), 0, "Light culling changed shading at step {step}");
    }
}

#[test]
fn moving_point_lights_use_finite_shadow_segments_and_range_cutoff() {
    let world = TriangleWorld::new(
        &[
            [0., 0., 0.],
            [200., 0., 0.],
            [0., 200., 0.],
            [0., 0., 300.],
            [200., 0., 300.],
            [0., 200., 300.],
        ],
        &[[0, 1, 2], [0, 1, 2]],
        &[[0, 0], [1, 3]],
        &[0, 0],
        &[Vec3::splat(1.)],
    );
    let mut core = CoreData::new(
        [1, 1],
        WorldData {
            world,
            envmap: image::Rgb32FImage::new(1, 1),
            envmap_scale: 0.,
            sun_dir: Vec3::new(0., 0., 1.),
            sun_color: Vec3::splat(0.),
            custom_emitters: vec![],
        },
    );
    let light = |z, range| PointLight {
        position_range: Vec4::new(50., 50., z, range),
        intensity: Vec4::new(10000., 5000., 2500., 0.),
    };
    core.set_point_lights(&[light(200., 1000.)]);
    let output = DEVICE.create_buffer::<Vec3<f32>>(1);
    let query = DEVICE.create_kernel::<fn()>(&track!(|| {
        let origin = Vec3::new(50., 50., 100.).expr();
        let dir = Vec3::new(0., 0., -1.).expr();
        let hit = core.world.trace(origin, dir);
        output.write(0, core.get_radiance(None, origin, dir, hit));
    }));
    let value = 0.25f32 * (1. - 0.04f32.powi(2)).powi(2);
    for (z, range, expected) in [
        (200., 1000., value),
        (400., 1000., 0.),
        (200., 100., 0.),
        (200., 1000., value),
    ] {
        core.update_point_lights(vec![light(z, range)]).execute();
        query.dispatch([1, 1, 1]);
        let result = output.copy_to_vec()[0];
        assert!(
            (result.x - expected).abs() < 1e-5,
            "z={z}, range={range}: {} != {expected}",
            result.x
        );
        assert!((result.y - expected * 0.5).abs() < 1e-5);
        assert!((result.z - expected * 0.25).abs() < 1e-5);
    }
}

#[test]
fn hundred_emissive_materials_feed_gi_rays_without_diffuse_or_external_light() {
    let mut vertices = Vec::new();
    let mut offsets = Vec::new();
    let mut materials = Vec::new();
    let mut emission = Vec::new();
    for i in 0..101u32 {
        let x = i as f32 * 4.;
        vertices.extend([[x, 0., 0.], [x + 2., 0., 0.], [x, 2., 0.]]);
        offsets.push([i, i * 3]);
        // Material IDs intentionally differ from instance IDs.
        materials.push(if i < 100 { (i * 37) % 100 } else { 100 });
        emission.push(if i < 100 {
            Vec3::new((i + 1) as f32 / 128., (100 - i) as f32 / 128., 0.25)
        } else {
            Vec3::splat(0.)
        });
    }
    let mut world = TriangleWorld::new(
        &vertices,
        &[[0, 1, 2]; 101],
        &offsets,
        &materials,
        // Black diffuse material must still emit its full specified radiance.
        &[Vec3::splat(0.); 101],
    );
    world.set_material_emission(&emission);
    let core = CoreData::new(
        [1, 1],
        WorldData {
            world,
            envmap: image::Rgb32FImage::from_pixel(1, 1, image::Rgb([1., 1., 1.])),
            envmap_scale: 0.,
            sun_dir: Vec3::new(0., 0., 1.),
            sun_color: Vec3::splat(0.),
            custom_emitters: vec![],
        },
    );
    let results = DEVICE.create_buffer::<Vec3<f32>>(102);
    let query = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let origin = Vec3::expr(i.cast_f32() * 4. + 0.5, 0.5f32, 5.0f32);
        let direction = Vec3::new(0., 0., -1.).expr();
        let hit = core.world.trace(origin, direction);
        // Exactly the radiance function used by primary and secondary GI rays.
        results.write(i, core.get_radiance(None, origin, direction, hit));
    }));
    for scale in [1., 0.25, 0., 8.] {
        core.lighting
            .write_host(Lighting {
                sun_dir: Vec3::new(0., 0., 1.),
                sun_color: Vec3::splat(0.),
                environment_scale: 0.,
                emission_scale: scale,
            })
            .execute();
        query.dispatch([102, 1, 1]);
        for (i, actual) in results.copy_to_vec().iter().enumerate() {
            let expected = if i < 101 {
                emission[materials[i] as usize] * scale
            } else {
                Vec3::splat(0.)
            };
            for (a, b) in [actual.x, actual.y, actual.z]
                .into_iter()
                .zip([expected.x, expected.y, expected.z])
            {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Emitter {i}, scale {scale}: {a} != {b}"
                );
            }
        }
    }
}
