use chandra::dynamic::{RigidBatch, RigidBinding, SkinBinding, SkinWeights};
use chandra::world::TriangleWorld;
use glam::{Mat4 as HostMat4, Vec3 as HostVec3};
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;

#[test]
fn hundred_rigid_instances_preserve_hits_normals_and_static_and_skinned_meshes() {
    // Keep a static prefix and a skinned suffix: instance indices and mixed
    // BLAS ownership must remain correct after repeated transform-only updates.
    let mut positions = Vec::new();
    let mut offsets = Vec::new();
    for i in 0..102 {
        offsets.push([i, i * 3]);
        let x = i as f32 * 4.;
        positions.extend([[x, 0., 0.], [x + 1., 0., 0.], [x, 1., 0.]]);
    }
    let dynamic: Vec<usize> = (1..102).collect();
    let mut world = TriangleWorld::new_dynamic(
        &positions,
        &[[0, 1, 2]; 102],
        &offsets,
        &[0; 102],
        &[Vec3::splat(0.7)],
        &dynamic,
    );
    let rigid_indices: Vec<_> = (1..101).collect();
    let batch = RigidBatch::new(&mut world, &rigid_indices).unwrap();
    assert!(RigidBinding::new(&world, 1).is_err());
    let skin = SkinBinding::new(
        &world,
        101,
        &[SkinWeights {
            joints: [0; 4],
            weights: [1., 0., 0., 0.],
        }; 3],
        &[HostMat4::IDENTITY],
    )
    .unwrap();
    let inputs = DEVICE.create_buffer::<Vec4<f32>>(204);
    let outputs = DEVICE.create_buffer::<Vec4<f32>>(204);
    let errors = Singleton::new(0u32);
    let query = DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        let origin = inputs.read(i * 2u32).xyz();
        let direction = inputs.read(i * 2u32 + 1u32).xyz();
        let hit = world.trace(origin, direction);
        if hit.miss() || hit.inst != i || !world.occluded(origin, direction) {
            errors.atomic().fetch_add(1u32);
            return;
        }
        outputs.write(i * 2u32, world.get_position(hit).extend(1.0));
        outputs.write(i * 2u32 + 1u32, world.get_normal_oriented(hit).extend(1.0));
        if i > 0u32 && i < 101u32 {
            let old = Vec3::expr(i.cast_f32() * 4. + 0.25, 0.25f32, 0.25f32);
            if !world
                .trace_segment(
                    old,
                    Vec3::new(0., 0., -1.).expr(),
                    keter::lang::types::vector::Vec2::expr(0.0f32, 0.5f32),
                )
                .miss()
            {
                errors.atomic().fetch_add(1u32);
            }
        }
    }));
    for frame in 0..12 {
        let transforms: Vec<_> = (1..101)
            .map(|i| {
                let pivot = HostVec3::new(i as f32 * 4., 0., 0.);
                let scale = HostVec3::new(if i % 13 == 0 { -1.1 } else { 0.9 }, 1.15, 0.8);
                HostMat4::from_translation(pivot + HostVec3::new(0.1, 0.2, 2. + frame as f32 * 0.1))
                    * HostMat4::from_rotation_y(0.2 + frame as f32 * 0.015)
                    * HostMat4::from_rotation_z(i as f32 * 0.01)
                    * HostMat4::from_scale(scale)
                    * HostMat4::from_translation(-pivot)
            })
            .collect();
        let skin_transform = HostMat4::from_translation(HostVec3::Z * (0.4 + frame as f32 * 0.03));
        let (vertices, bounds) = batch.update(&world, &transforms).unwrap();
        let (skin_vertices, _) = skin.update(&[skin_transform]).unwrap();
        (
            vertices,
            skin_vertices,
            world.update_acceleration(&dynamic, frame == 6),
        )
            .chain()
            .execute();
        let actual = world.positions.copy_to_vec();
        let mut expected = Vec::new();
        let mut queries = Vec::new();
        for i in 0..102 {
            let transform = if i == 0 {
                HostMat4::IDENTITY
            } else if i == 101 {
                skin_transform
            } else {
                transforms[i - 1]
            };
            let p: [HostVec3; 3] =
                std::array::from_fn(|j| transform.transform_point3(positions[i * 3 + j].into()));
            for j in 0..3 {
                assert!(HostVec3::from(actual[i * 3 + j]).distance(p[j]) < 2e-4);
                if (1..101).contains(&i) {
                    assert!(bounds[i - 1].expanded(2e-4).contains(p[j]));
                }
            }
            let center = (p[0] + p[1] + p[2]) / 3.;
            let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
            queries.push(Vec4::from((center + normal * 1.5).extend(1.)));
            queries.push(Vec4::from((-normal).extend(0.)));
            expected.push((center, normal));
        }
        inputs.copy_from_vec(queries).execute();
        query.dispatch([102, 1, 1]);
        assert_eq!(
            errors.read_blocking(),
            0,
            "Stale, wrong-instance or missing hit at frame {frame}"
        );
        let output = outputs.copy_to_vec();
        for (i, (position, normal)) in expected.into_iter().enumerate() {
            let p = output[i * 2];
            let n = output[i * 2 + 1];
            assert!(HostVec3::new(p.x, p.y, p.z).distance(position) < 5e-4);
            assert!(HostVec3::new(n.x, n.y, n.z).distance(normal) < 5e-4);
        }
    }
    assert!(batch.update(&world, &[]).is_err());
    assert!(batch.update(&world, &[HostMat4::ZERO; 100]).is_err());
}

fn triangle() -> TriangleWorld {
    TriangleWorld::new_dynamic(
        &[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]],
        &[[0, 1, 2]],
        &[[0, 0]],
        &[0],
        &[Vec3::splat(0.7)],
        &[0],
    )
}

#[test]
fn rigid_only_pipeline_keeps_order_across_empty_blas_stage() {
    use chandra::algorithm::CommandList;
    let mut world = triangle();
    let batch = RigidBatch::new(&mut world, &[0]).unwrap();
    let output = DEVICE.create_buffer::<Vec4<f32>>(2);
    let query = query_kernel(&world, &output);
    let mut transform = HostMat4::IDENTITY;
    for frame in 0..24 {
        let moving = frame % 6 < 4;
        let mut commands = CommandList::new();
        if moving {
            transform = HostMat4::from_translation(HostVec3::new(5. + frame as f32, 0., 1.))
                * HostMat4::from_rotation_y(0.7);
            let (vertices, _) = batch.update(&world, &[transform]).unwrap();
            commands += vertices;
        }
        // This is the viewer's unprofiled submission path: no intermediate
        // dispatch, readback or synchronization may hide a broken dependency.
        commands += world.update_bottom_level(if moving { &[0] } else { &[] }, frame == 12);
        commands += world.update_top_level(moving, frame == 12);
        let center = transform.transform_point3(HostVec3::new(2. / 3., 2. / 3., 0.));
        let normal = transform.transform_vector3(HostVec3::Z).normalize();
        commands.push(query.dispatch_async(
            [1, 1, 1],
            &Vec3::from(center + normal * 5.),
            &Vec3::from(-normal),
        ));
        commands.execute();
        let result = output.copy_to_vec();
        assert_eq!(result[0].w, 1., "Missing hit at frame {frame}");
        assert!(
            HostVec3::new(result[0].x, result[0].y, result[0].z).distance(center) < 1e-4,
            "Shading vertices and instance transform disagree at frame {frame}"
        );
        assert!(HostVec3::new(result[1].x, result[1].y, result[1].z).distance(normal) < 1e-4);
    }
}

#[test]
fn textured_diffuse_samples_and_blends_the_expected_mip_levels() {
    let mut world = triangle();
    let texture = DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, 4, 4, 3);
    texture
        .view(0)
        .copy_from(&[Vec4::new(1.0f32, 0., 0., 1.); 16]);
    texture
        .view(1)
        .copy_from(&[Vec4::new(0.0f32, 1., 0., 1.); 4]);
    texture.view(2).copy_from(&[Vec4::new(0.0f32, 0., 1., 1.)]);
    world.set_diffuse_textures(&[[0., 0.], [1., 0.], [0., 1.]], &[1.], &[0], vec![texture]);
    let output = DEVICE.create_buffer::<Vec3<f32>>(1);
    let sample = DEVICE.create_kernel::<fn(f32)>(&track!(|distance| {
        let hit = world.trace(
            Vec3::expr(0.5f32, 0.5f32, distance),
            Vec3::new(0., 0., -1.).expr(),
        );
        output.write(0, world.get_diffuse(hit));
    }));
    for (lod, expected) in [
        (0.0f32, [0.7, 0., 0.]),
        (1., [0., 0.7, 0.]),
        (1.5, [0., 0.35, 0.35]),
        (2., [0., 0., 0.7]),
    ] {
        sample.dispatch([1, 1, 1], &(2.0f32.powf(lod) / (0.0008 * 4.)));
        let actual = output.copy_to_vec()[0];
        for (actual, expected) in [actual.x, actual.y, actual.z].into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.001,
                "LOD {lod}: {actual} != {expected}"
            );
        }
    }
}

fn query_kernel(
    world: &TriangleWorld,
    output: &Buffer<Vec4<f32>>,
) -> Kernel<fn(Vec3<f32>, Vec3<f32>)> {
    DEVICE.create_kernel::<fn(Vec3<f32>, Vec3<f32>)>(&track!(|origin, direction| {
        let hit = world.trace(origin, direction);
        if hit.miss() {
            output.write(0, Vec4::splat(-1.0));
            output.write(1, Vec4::splat(0.0));
        } else {
            output.write(0, world.get_position(hit).extend(1.0));
            output.write(1, world.get_normal_oriented(hit).extend(1.0));
        }
    }))
}

#[test]
fn rigid_refit_updates_vertices_hits_and_normals_without_recompilation() {
    let world = triangle();
    let binding = RigidBinding::new(&world, 0).unwrap();
    let output = DEVICE.create_buffer::<Vec4<f32>>(2);
    let query = query_kernel(&world, &output);
    query.dispatch([1, 1, 1], &Vec3::new(0.5, 0.5, 5.), &Vec3::new(0., 0., -1.));
    assert_eq!(output.copy_to_vec()[0].w, 1.);
    for frame in 0..12 {
        let transform = HostMat4::from_translation(HostVec3::new(5. + frame as f32, 0., 1.))
            * HostMat4::from_rotation_y(0.7);
        let (deform, bounds) = binding.update(transform).unwrap();
        (deform, world.update_acceleration(&[0], frame == 6))
            .chain()
            .execute();
        let vertices = world.positions.copy_to_vec();
        for (vertex, original) in vertices
            .iter()
            .zip([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        {
            let actual = HostVec3::from(*vertex);
            assert!(actual.distance(transform.transform_point3(original.into())) < 1e-4);
            assert!(bounds.expanded(1e-4).contains(actual));
        }
        query.dispatch([1, 1, 1], &Vec3::new(0.5, 0.5, 5.), &Vec3::new(0., 0., -1.));
        assert_eq!(
            output.copy_to_vec()[0].w,
            -1.,
            "Old location still intersects"
        );
        let center = transform.transform_point3(HostVec3::new(2. / 3., 2. / 3., 0.));
        let normal = transform.transform_vector3(HostVec3::Z).normalize();
        query.dispatch(
            [1, 1, 1],
            &Vec3::from(center + normal * 5.),
            &Vec3::from(-normal),
        );
        let result = output.copy_to_vec();
        assert_eq!(result[0].w, 1.);
        assert!(HostVec3::new(result[0].x, result[0].y, result[0].z).distance(center) < 1e-4);
        assert!(HostVec3::new(result[1].x, result[1].y, result[1].z).distance(normal) < 1e-4);
    }
}

#[test]
fn gpu_skinning_matches_cpu_and_refitted_rays_over_multiple_poses() {
    let world = triangle();
    let weights = [
        SkinWeights {
            joints: [0, 1, 0, 0],
            weights: [1., 0., 0., 0.],
        },
        SkinWeights {
            joints: [0, 1, 0, 0],
            weights: [0., 1., 0., 0.],
        },
        SkinWeights {
            joints: [0, 1, 0, 0],
            weights: [0.5, 0.5, 0., 0.],
        },
    ];
    let binding = SkinBinding::new(&world, 0, &weights, &[HostMat4::IDENTITY; 2]).unwrap();
    let output = DEVICE.create_buffer::<Vec4<f32>>(2);
    let query = query_kernel(&world, &output);
    for frame in 0..20 {
        let height = 0.2 + frame as f32 * 0.12;
        let joints = [
            HostMat4::IDENTITY,
            HostMat4::from_translation(HostVec3::Z * height),
        ];
        let (update, bounds) = binding.update(&joints).unwrap();
        (update, world.update_acceleration(&[0], frame == 10))
            .chain()
            .execute();
        let expected = [
            HostVec3::ZERO,
            HostVec3::new(2., 0., height),
            HostVec3::new(0., 2., height * 0.5),
        ];
        for (actual, expected) in world.positions.copy_to_vec().iter().zip(expected) {
            let actual = HostVec3::from(*actual);
            assert!(actual.distance(expected) < 1e-5);
            assert!(bounds.expanded(1e-5).contains(actual));
        }
        let center = (expected[0] + expected[1] + expected[2]) / 3.;
        let normal = (expected[1] - expected[0])
            .cross(expected[2] - expected[0])
            .normalize();
        query.dispatch(
            [1, 1, 1],
            &Vec3::from(center + normal * 5.),
            &Vec3::from(-normal),
        );
        let result = output.copy_to_vec();
        assert_eq!(result[0].w, 1.);
        assert!(HostVec3::new(result[0].x, result[0].y, result[0].z).distance(center) < 1e-4);
        assert!(HostVec3::new(result[1].x, result[1].y, result[1].z).distance(normal) < 1e-4);
    }
    assert!(binding.update(&[]).is_err());
    assert!(
        binding
            .update(&[HostMat4::from_cols_array(&[f32::NAN; 16]); 2])
            .is_err()
    );
}
