#![feature(array_windows)]

use std::f32::consts::TAU;

use glam::Mat3 as FMat3;
use keter::lang::types::vector::{Mat3, Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};
use yesod::color::{inferno, reinhard};
use yesod::rand::pcg3df;

use crate::cache::{
    HashCache, Node, NodeComps, POSITION_INVALID, Probe, ProbeComps, RadianceDeposit,
    decode_position,
};
use crate::world::{TriangleWorld, VoxelWorld};

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

    let world = TriangleWorld::new("assets/sponza.obj");

    let cache = HashCache::new(16.0, 4, 1 << 18, 6);

    let cache_memory = cache.data.len() * std::mem::size_of::<RadianceDeposit>()
        + cache.nodes.len() * std::mem::size_of::<Node>();
    println!("Cache memory: {} MB", cache_memory / 1024 / 1024);

    let lower_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        let value = cache.data.read(index).var();
        value.reduce(0.9_f32.expr(), 10.0.expr());
        cache.data.write(index, value);
    }));
    let reset_cache_nodes = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache.nodes.write(dispatch_id().x, Node::default());
    }));
    let reset_cache_data = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache.data.write(
            dispatch_id().x,
            RadianceDeposit {
                fluence: Vec4::new(0.0, 0.0, 0.0, 1.0),
                radiance_count: Vec4::new(0.0, 0.0, 0.0, 1.0),
            },
        );
    }));
    // TODO: Test conjoined merging method instead where the merging is combined with the raytracing instead and fluence isn't stored.
    let merge_cache = DEVICE.create_kernel::<fn(u32)>(&track!(|level| {
        let node_index = dispatch_id().x + cache.get_node_offset(level);
        let node = cache.nodes.read(node_index);
        if node.position == POSITION_INVALID {
            return;
        }
        let position = decode_position(node.position);
        let probe = Probe::from_comps_expr(ProbeComps { position, level });
        // TODO: needs bilinear interpolation
        let next_node_index = cache.get_node_index(probe.next_nearest());
        let next_node = cache.nodes.read(next_node_index);
        if next_node.position == POSITION_INVALID {
            return;
        }
        let encoder = cache.get_encoder_by_level(level);
        let next_encoder = cache.get_encoder_by_level(level + 1);
        for index in 0_u32.expr()..encoder.linear_size() {
            let dir = encoder.linear_to_grid(index);
            let total_radiance = Vec3::splat(0.0_f32).var();
            for offset in [[0, 0], [0, 1], [1, 0], [1, 1]].map(Vec2::from) {
                let sample_dir = dir * 2 + offset;
                let sample_index = next_encoder.grid_to_linear(sample_dir);
                *total_radiance += cache
                    .data
                    .read(cache.get_data_index_raw(next_node_index, level + 1, sample_index))
                    .radiance_count
                    .xyz();
            }
            let data_index = cache.get_data_index_raw(node_index, level, index);
            let radiance_deposit = cache.data.read(data_index).var();
            radiance_deposit.merge(total_radiance / 4.0);
            cache.data.write(data_index, radiance_deposit);
        }
        // TODO: Switch to pre-merging? How does that actually work.
    }));
    let merge_final_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        let level = (cache.num_levels - 1).expr();
        let node_index = dispatch_id().x + cache.get_node_offset(level);
        let node = cache.nodes.read(node_index);
        if node.position == POSITION_INVALID {
            return;
        }
        let encoder = cache.get_encoder_by_level(level);
        for index in 0_u32.expr()..encoder.linear_size() {
            let dir = encoder.get_dir_from_linear(index);
            let sunlight_dir = Vec3::expr(0.3_f32, 1.5, 1.0).normalize();
            let data_index = cache.get_data_index_raw(node_index, level, index);
            let radiance_deposit = cache.data.read(data_index).var();
            radiance_deposit.merge(if dir.dot(sunlight_dir) > 0.9 {
                Vec3::splat_expr(10.0_f32)
            } else {
                Vec3::splat_expr(0.01_f32)
            });
            cache.data.write(data_index, radiance_deposit);
        }
    }));

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32, u32)>(
        KernelBuildOptions {
            name: Some("draw_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick, mode| {
            set_block_size([8, 8, 1]);
            let pixel = dispatch_id().xy();
            let ray_dir = view.transform * (pixel.cast_f32() - view.screen_size / 2.0).extend(1.0);
            let ray_dir = ray_dir.normalize();

            let hit = world.trace(view.pos, ray_dir);

            if hit.valid() {
                let world_pos = view.pos + ray_dir * hit.committed_ray_t;
                let normal = world.get_normal_flat(hit);
                // let shadowed = world
                //     .trace(offset_pos, Vec3::<f32>::expr(0.3, 1.5, 1.0).normalize())
                //     .valid();
                // let sun_radiance =
                //     shadowed.select(0.0_f32.expr(), 1.0_f32.expr()) * Vec3::new(1.0, 0.95, 0.8);

                // TODO: Decide whether to have the circle cutoffs be around the sample point or be around the cell center.

                // https://ameye.dev/notes/sampling-the-hemisphere/
                let rng = pcg3df(pixel.extend(tick));
                let theta = rng.x.acos();
                let phi = rng.y * TAU;
                let v = Vec3::expr(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let perp1 = normal.yzx();
                let perp2 = normal.zxy();
                let bounce_ray = v.x * perp1 + v.y * perp2 + v.z * normal;

                let probe = Probe::base(cache.quantize(world_pos));
                let has_probe = true.var();
                for level in (0_u32..cache.num_levels).rev() {
                    if **has_probe
                        && !cache.init(
                            probe.shift_level(level.expr()),
                            // Vec4::expr(0.0, 0.0, 0.0, 1.0),
                            // bounce_ray,
                        )
                    {
                        *has_probe = false;
                    }
                }

                for level in (0_u32..cache.num_levels).rev() {
                    let mut ray_end_t = 12.0 * cache.base_grid_size * (1_u32 << level) as f32;
                    let mut ray_start_t = ray_end_t / 2.0;
                    escape!({
                        if level == 0 {
                            ray_start_t = 0.01;
                        }
                        if level == cache.num_levels - 1 {
                            ray_end_t = 10000.0;
                        }
                    });
                    let bounce_hit = world.trace_segment(
                        world_pos,
                        bounce_ray,
                        Vec2::expr(ray_start_t, ray_end_t),
                    );
                    let fluence = if bounce_hit.valid() {
                        Vec4::splat_expr(0.0)
                        /*
                        let bounce_pos = world_pos + bounce_ray * bounce_hit.committed_ray_t;
                        let color = Vec3::splat_expr(0.7); // world.get_color(bounce_start, bounce_ray, bounce_hit);
                        let value =
                            cache.query_diffuse(bounce_pos, world.get_normal_flat(bounce_hit));
                        if value.x == -1.0 {
                            Vec3::splat_expr(0.0_f32).extend(-1.0) // skip because undefined
                        } else {
                            (value * color).extend(0.0)
                        }
                        */
                    } else {
                        Vec3::splat_expr(0.0_f32).extend(1.0) // miss
                    };
                    if fluence.w >= 0.0 {
                        cache.deposit(probe.shift_level(level.expr()), fluence, bounce_ray);
                    }
                }

                let color = if mode == 0 {
                    cache.query_diffuse(world_pos, normal)
                } else if mode == 1 {
                    cache.query(probe, ray_dir)
                } else if mode == 2 {
                    if has_probe {
                        pcg3df(probe.position)
                    } else {
                        Vec3::splat_expr(0.0)
                    }
                } else if mode == 3 {
                    inferno(1.0 - reinhard(hit.committed_ray_t / 1000.0))
                } else if mode == 4 {
                    world_pos / 1000.0
                } else {
                    Vec3::expr(1.0, 0.0, 0.0)
                };
                app.set_pixel(pixel.cast_i32(), color);
            }
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 2.1,
        pitch: 0.15,
        pos: Vec3::new(-360.0, 1250.0, 10.0),
        fov: 1.15,
    };

    let mut mode = 0;

    reset_cache_nodes.dispatch([cache.nodes.len() as u32, 1, 1]);
    reset_cache_data.dispatch([cache.data.len() as u32, 1, 1]);

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = if rt.key_down(KeyCode::ControlLeft) {
            20.0
        } else {
            800.0
        } * rt.frame_time();
        for key in &rt.keys_down {
            match key {
                KeyCode::ArrowLeft => camera.yaw -= rot_speed,
                KeyCode::ArrowRight => camera.yaw += rot_speed,
                KeyCode::ArrowUp => camera.pitch -= rot_speed,
                KeyCode::ArrowDown => camera.pitch += rot_speed,
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
        if rt.key_pressed(KeyCode::KeyP) {
            println!("Camera: {camera:?}");
        }
        if rt.key_pressed(KeyCode::KeyM) {
            mode = (mode + 1) % 5;
            println!("Drawing level {mode}");
        }

        rt.log_fps();

        draw_kernel.dispatch(rt.dispatch_size(), &camera.view(), &rt.tick, &mode);
        // lower_cache.dispatch([cache.data.len() as u32, 1, 1]);

        merge_final_cache.dispatch([cache.num_nodes_at_level(cache.num_levels - 1), 1, 1]);
        for l in (0..cache.num_levels - 1).rev() {
            merge_cache.dispatch([cache.num_nodes_at_level(l), 1, 1], &l);
        }

        if rt.key_pressed(KeyCode::Backslash) {
            println!("Resetting cache");
            reset_cache_nodes.dispatch([cache.nodes.len() as u32, 1, 1]);
            reset_cache_data.dispatch([cache.data.len() as u32, 1, 1]);
        }
    });
}
