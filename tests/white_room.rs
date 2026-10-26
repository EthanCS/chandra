use chandra::algorithm::CoreData;
use chandra::utils::offset_ray;
use keter::lang::types::vector::Vec3;
use keter::prelude::*;

#[test]
fn sealed_room_thin_partitions_and_real_openings_have_correct_visibility() {
    let core = CoreData::new([8, 8], chandra::white_room::world());
    let errors = Singleton::new(0u32);
    DEVICE
        .create_kernel::<fn()>(&track!(|| {
            let i = dispatch_id().x;
            let dir = yesod::direction::project_line(
                Vec3::expr(i, 17u32, 29u32).cast_f32().xy().fract()
                    + yesod::rand::pcg3df(Vec3::expr(i, 237u32, 99u32)).xy(),
            );
            let start = Vec3::new(2120., 150., 110.).expr();
            let hit = core.world.trace(start, dir);
            if hit.miss() {
                errors.atomic().fetch_add(1u32);
            } else {
                let position = core.hit_position(start, dir, hit);
                if core.get_emission(None, position, hit, dir).reduce_max() > 1e-6f32 {
                    errors.atomic().fetch_add(1u32);
                }
            }
            // Starting exactly on the floor must not self-occlude an upward ray.
            let up = Vec3::new(0., 0., 1.).expr();
            let slit = offset_ray(Vec3::new(2520., 150., 0.).expr(), up);
            if core.world.occluded(slit, up) {
                errors.atomic().fetch_add(1u32);
            }
            if !core.world.occluded(Vec3::new(2520., 100., 1.).expr(), up) {
                errors.atomic().fetch_add(1u32);
            }
            // The old integer-ULP offset could cross both this thin partition and
            // the sealed room's 4 cm shell at these world coordinates.
            let right = Vec3::new(1., 0., 0.).expr();
            let wall = offset_ray(Vec3::new(2869.999, 150., 50.).expr(), right);
            let blocker = core.world.trace(wall, right);
            if blocker.miss() || blocker.committed_ray_t > 2.01f32 {
                errors.atomic().fetch_add(1u32);
            }
            let corner = Vec3::new(3300., 50., 50.).expr();
            if core.world.occluded(corner, up) {
                errors.atomic().fetch_add(1u32);
            }
        }))
        .dispatch([1024, 1, 1]);
    assert_eq!(errors.read_blocking(), 0);
}

#[test]
fn gbuffer_kernel_uses_each_worlds_own_resources() {
    let a = CoreData::new([8, 8], chandra::white_room::world());
    let b = CoreData::new([8, 8], chandra::white_room::world());
    a.view
        .write_host(chandra::white_room::camera(0, [8, 8]).view())
        .execute();
    let mut outside = chandra::white_room::camera(0, [8, 8]);
    outside.pos = glam::Vec3::new(0., 0., 10000.);
    outside.pitch = -1.5;
    b.view.write_host(outside.view()).execute();
    a.compute_gbuffer().execute();
    b.compute_gbuffer().execute();
    let hit_counts = DEVICE.create_buffer::<u32>(2);
    hit_counts.copy_from(&[0, 0]);
    DEVICE
        .create_kernel::<fn()>(&track!(|| {
            let pixel = dispatch_id().xy();
            if a.read_depth(pixel) > 0.0 {
                hit_counts.atomic_ref(0u32).fetch_add(1u32);
            }
            if b.read_depth(pixel) > 0.0 {
                hit_counts.atomic_ref(1u32).fetch_add(1u32);
            }
        }))
        .dispatch([8, 8, 1]);
    assert_eq!(hit_counts.copy_to_vec(), [64, 0]);
}
