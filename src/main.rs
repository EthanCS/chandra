use std::f32::consts::TAU;

use glam::Mat3 as FMat3;
use keter::lang::types::vector::{Mat3, Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::utils::{inferno, pcg3df, reinhard};
use keter_testbed::{App, KeyCode};

use crate::cache::HashCache;
use crate::world::World;

pub mod cache;
pub mod color;
pub mod fluence;
pub mod utils;
pub mod world;

#[derive(Debug, Clone, Copy)]
struct Camera {
    screen_size: Vec2<f32>,
    yaw: f32,
    pitch: f32,
    pos: Vec3<f32>,
    fov: f32,
}
impl Camera {
    fn rotation(&self) -> FMat3 {
        let view = FMat3::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0).transpose();
        FMat3::from_cols(view.x_axis, view.z_axis, view.y_axis).transpose()
    }
    fn view(&self) -> View {
        let view = self.rotation();
        let ratio = (self.fov / 2.0).tan() / (self.screen_size.y / 2.0);
        let view = FMat3::from_cols(view.x_axis * ratio, -view.y_axis * ratio, view.z_axis);

        View {
            screen_size: self.screen_size,
            pos: self.pos,
            transform: view.into(),
        }
    }
    fn forward(&self) -> Vec3<f32> {
        self.rotation().z_axis.into()
    }
    fn right(&self) -> Vec3<f32> {
        self.rotation().x_axis.into()
    }
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
struct View {
    screen_size: Vec2<f32>,
    pos: Vec3<f32>,
    transform: Mat3,
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [2048; 2]).agx().resize().init();

    let file = dot_vox::load("assets/castle.vox").unwrap();

    let world = World::new(file);

    let cache = HashCache::new(4.0, 1 << 20);

    let lower_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        let value = cache.nodes.read(index).var();
        value.reduce(0.5_f32.expr(), 5.0.expr());
        cache.nodes.write(index, value);
    }));

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32)>(
        KernelBuildOptions {
            name: Some("draw_kernel_4".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick| {
            set_block_size([8, 8, 1]);
            let pixel = dispatch_id().xy();
            let ray_dir = view.transform * (pixel.cast_f32() - view.screen_size / 2.0).extend(1.0);
            let ray_dir = ray_dir.normalize();

            let hit = world.trace(view.pos, ray_dir);

            if hit.procedural_hit() {
                let world_pos = view.pos + ray_dir * hit.committed_ray_t;
                let world_voxel = world_pos.floor().cast_i32();
                let delta_dist = (1.0 / ray_dir).abs();
                let side_dist = ((-ray_dir).signum() * (world_voxel.cast_f32() - world_pos)
                    + (-ray_dir).signum() * 0.5
                    + 0.5)
                    * delta_dist;
                let mask = side_dist <= keter::min(side_dist.yzx(), side_dist.zxy());
                let normal = -mask.select(ray_dir.signum().cast_i32(), Vec3::splat_expr(0));
                let prev_voxel = world_voxel + normal;
                let shadowed = world
                    .trace(
                        prev_voxel.cast_f32() + 0.5,
                        Vec3::<f32>::expr(0.3, 1.5, 1.0).normalize(),
                    )
                    .procedural_hit();
                let sun_radiance =
                    shadowed.select(0.0_f32.expr(), 1.0_f32.expr()) * Vec3::new(1.0, 0.95, 0.8);

                let rng = pcg3df(pixel.extend(tick));
                let theta = rng.x.acos();
                let phi = rng.y * TAU;
                let v = Vec3::expr(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let normal = normal.cast_f32();
                let perp1 = normal.yzx();
                let perp2 = normal.zxy();
                let bounce_ray = v.x * perp1 + v.y * perp2 + v.z * normal;
                let bounce_start = world_pos + normal;
                let bounce_hit = world.trace(bounce_start, bounce_ray);
                let bounce_radiance = if bounce_hit.procedural_hit() {
                    let bounce_pos = bounce_start + bounce_ray * bounce_hit.committed_ray_t;
                    let color = world.get_color(bounce_start, bounce_ray, bounce_hit);
                    let value = cache.query(bounce_pos);
                    if value.x == -1.0 {
                        Vec3::splat_expr(0.0)
                    } else {
                        value * color
                    }
                } else {
                    0.3_f32.expr() * Vec3::splat_expr(1.0) // ambient
                };
                let radiance = sun_radiance + bounce_radiance;
                // Todo: This is a race condition.
                cache.deposit(world_pos, radiance);

                // Vec3::splat_expr(num_ddas.cast_f32() * 0.1); // pcg3df(Vec3::expr(hit.inst, hit.prim, 1));
                let color = world.get_color(view.pos, ray_dir, hit) * cache.query(world_pos); // inferno(1.0 - reinhard(hit.committed_ray_t / 1000.0)); //
                app.set_pixel(pixel.cast_i32(), color);
            }
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 0.5,
        pitch: 0.6,
        pos: Vec3::new(-500.0, -500.0, 500.0),
        fov: 1.15,
    };

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = if rt.key_down(KeyCode::ControlLeft) {
            5.0
        } else {
            200.0
        } * rt.frame_time();
        for key in &rt.keys_down {
            match key {
                KeyCode::ArrowLeft => camera.yaw -= 1.0 * rot_speed,
                KeyCode::ArrowRight => camera.yaw += 1.0 * rot_speed,
                KeyCode::ArrowUp => camera.pitch -= 1.0 * rot_speed,
                KeyCode::ArrowDown => camera.pitch += 1.0 * rot_speed,
                KeyCode::KeyW => camera.pos = camera.pos + camera.forward() * move_speed,
                KeyCode::KeyS => camera.pos = camera.pos - camera.forward() * move_speed,
                KeyCode::KeyA => camera.pos = camera.pos - camera.right() * move_speed,
                KeyCode::KeyD => camera.pos = camera.pos + camera.right() * move_speed,
                KeyCode::Space => camera.pos = camera.pos + Vec3::new(0.0, 0.0, 1.0) * move_speed,
                KeyCode::ShiftLeft => {
                    camera.pos = camera.pos - Vec3::new(0.0, 0.0, 1.0) * move_speed
                }
                _ => (),
            }
        }

        rt.log_fps();

        draw_kernel.dispatch(rt.dispatch_size(), &camera.view(), &rt.tick);
        lower_cache.dispatch([cache.nodes.len() as u32, 1, 1]);
    });
}
