#![feature(array_windows)]

/*
TODO:
- Screenspace version of everything?
- [X] or screenspace grid.
- [X] Blue noise.
- [X] Make octahedral actually equal area (fix current bug)
- Comparison to SHARC / PT
- [X] Bilinear interpolation at the end as well
*/

use std::f32::consts::TAU;
use std::path::PathBuf;

use glam::Mat3 as FMat3;
use image::{GenericImageView, ImageReader};
use keter::graph::profile::Profiler;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};
use yesod::camera::{Camera, View};
use yesod::color::{inferno, plasma, reinhard};
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::rand::{pcg3d, pcg3df};

use crate::cache::{
    Fluence, HashCache, LoadedNode, LoadedNodeComps, Node, POSITION_INVALID, Probe, ProbeComps,
    Radiance, RadianceDeposit, RadianceElement, decode_position,
};
use crate::world::TriangleWorld;

pub mod cache;
pub mod color;
pub mod fluence;
pub mod utils;
pub mod world;

const RADIANCE_DIRECT_ONLY: u32 = 0b1;

#[tracked]
fn reflect(dir: Expr<Vec3<f32>>, normal: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    dir - normal * (2.0_f32.expr() * dir.dot(normal))
}

#[tracked]
fn rotate_hemi(normal: Expr<Vec3<f32>>, v: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    // Find perpendicular basis; due to hairy ball theorem this cannot be continuous.
    let perp1 = if normal.z.abs() < 0.9 {
        normal.cross(Vec3::expr(0.0_f32, 0.0, 1.0)).normalize()
    } else {
        normal.cross(Vec3::expr(0.0_f32, 1.0, 0.0)).normalize()
    };
    let perp2 = normal.cross(perp1);
    v.x * perp1 + v.y * perp2 + v.z * normal
}

// https://ameye.dev/notes/sampling-the-hemisphere/
#[tracked]
fn sample_bounce(normal: Expr<Vec3<f32>>, rng: Expr<Vec2<f32>>) -> Expr<Vec3<f32>> {
    let theta = rng.x.acos();
    let phi = rng.y * TAU;
    let v = Vec3::expr(
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    );
    rotate_hemi(normal, v)
}

#[tracked]
fn over(fluence: Expr<Fluence>, radiance: Expr<Radiance>) -> Expr<Radiance> {
    fluence.xyz() + radiance * fluence.w
}

#[tracked]
fn sunlight_radiance(dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
    let sunlight_dir = Vec3::expr(0.3_f32, 1.5, 1.0).normalize();
    if dir.dot(sunlight_dir) > 0.9 {
        Vec3::splat_expr(10.0_f32)
    } else {
        Vec3::splat_expr(0.01_f32)
    }
}

type NoiseTexture = Tex3d<Vec4<f32>>;
fn load_noise_textures(names: impl Fn(u32) -> PathBuf) -> NoiseTexture {
    let storage = PixelStorage::Byte4;
    let mut num_images = 0;
    let mut data = vec![];
    let mut shape = (0, 0);
    while let Ok(image) = ImageReader::open(names(num_images)) {
        let image = image.decode().unwrap();
        if shape == (0, 0) {
            shape = image.dimensions();
        } else {
            assert_eq!(shape, image.dimensions());
        }
        num_images += 1;
        assert_eq!(image.color(), image::ColorType::Rgba8);

        data.extend(image.into_rgba8().pixels().map(|p| Vec4::from(p.0)));
    }
    let texture = DEVICE.create_tex3d(storage, shape.0, shape.1, num_images, 1);
    texture.view(0).copy_from(&data);
    texture
}
#[tracked]
fn as_normal(v: Expr<Vec4<f32>>) -> Expr<Vec3<f32>> {
    v.xyz() * 2.0 - 1.0
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [2048; 2]).agx().resize().init();

    // let sphere_noise =

    let hemi_noise = load_noise_textures(|i| {
        PathBuf::from(format!(
            "assets/noise/sphere/exp/sphere_coshemi_binomial3x3_exp0101_product_{i}.png",
        ))
    });

    let world = TriangleWorld::load(
        "assets/sponza.obj",
        FMat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    );

    let depth_buf =
        DEVICE.create_tex2d::<f32>(PixelStorage::Float1, app.size()[0], app.size()[1], 1);
    let normal_buf =
        DEVICE.create_tex2d::<Vec2<f32>>(PixelStorage::Short2, app.size()[0], app.size()[1], 1);
    let accum_buf =
        DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, app.size()[0], app.size()[1], 1);
    // let cache = HashCache::new(16.0, 100000.0, 4, 1 << 18, 6);
    let cache = HashCache::new(16.0 / 8.0, 16.0 * 50.0 / 8.0, 4, 1 << 18, 6);

    let cache_memory = cache.data.len() * std::mem::size_of::<RadianceElement>()
        + cache.nodes.len() * std::mem::size_of::<Node>();
    println!("Cache memory: {} MB", cache_memory / 1024 / 1024);

    let lower_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        let value = cache.data.read(index).var();
        value.reduce(0.9_f32.expr(), 0.0.expr());
        cache.data.write(index, value);
    }));
    let reset_tex2d = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>)>(&track!(|tex| {
        let pixel = dispatch_id().xy();
        tex.write(pixel, Vec4::splat_expr(0.0_f32));
    }));
    let reset_cache_nodes = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache.nodes.write(dispatch_id().x, Node::default());
    }));
    let reset_cache_data = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache
            .data
            .write(dispatch_id().x, RadianceElement::default());
    }));
    // TODO: Test conjoined merging method instead where the merging is combined with the raytracing instead and fluence isn't stored.
    let merge_cache = DEVICE.create_kernel::<fn(u32)>(&track!(|level| {
        let node_index = dispatch_id().x;
        let node = cache.read_node(level, node_index);
        if node.position == POSITION_INVALID {
            return;
        }
        let position = decode_position(node.position);
        let probe = Probe::from_comps_expr(ProbeComps { position, level });
        let node = cache.load_index(probe, node_index);
        // TODO: best to process in batches of 8.
        /*
        let total_weight = 1.0_f32.var();
        let next = [(
            cache.query(
                probe.next_nearest(),
                |x| x,
                || {
                    return;
                    LoadedNode::expr_zeroed()
                },
            ),
            1.0_f32.expr(),
        )];
        */
        let total_weight = 0.0_f32.var();
        let next = probe.next_bilinear().map(|(probe, weight)| {
            let (next, weight) = cache.query(
                probe,
                |n| (n, weight),
                || {
                    (
                        LoadedNode::from_comps_expr(LoadedNodeComps {
                            probe,
                            encoder: cache.get_encoder(probe.level),
                            data_offset: 0_u32.expr(),
                        }),
                        0_f32.expr(),
                    )
                },
            );
            *total_weight += weight;
            (next, weight)
        });
        let next = next.map(|(next, weight)| (next, weight / total_weight));
        for index in 0_u32.expr()..(node.encoder.linear_size() / dispatch_size().y) {
            let index = index * dispatch_size().y + dispatch_id().y;
            let dir = node.encoder.linear_to_grid(index);
            let total_radiance = Vec3::splat(0.0_f32).var();
            for offset in [[0, 0], [0, 1], [1, 0], [1, 1]].map(Vec2::from) {
                let sample_dir = dir * 2 + offset;
                for (next, weight) in &next {
                    let sample_index = next.encoder.grid_to_linear(sample_dir);
                    *total_radiance +=
                        weight * cache.data.read(next.data_offset + sample_index).radiance();
                }
            }
            let data_index = node.data_offset + index;
            let radiance_deposit = cache.data.read(data_index).var();
            radiance_deposit.merge(total_radiance / 4.0);
            cache.data.write(data_index, radiance_deposit);
        }
        // TODO: Switch to pre-merging? How does that actually work.
    }));
    let merge_final_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        let level = (cache.num_levels - 1).expr();
        let node_index = dispatch_id().x;
        let node = cache.read_node(level, node_index);
        if node.position == POSITION_INVALID {
            return;
        }
        let position = decode_position(node.position);
        let probe = Probe::from_comps_expr(ProbeComps { position, level });
        let node = cache.load_index(probe, node_index);
        for index in 0_u32.expr()..(node.encoder.linear_size() / dispatch_size().y) {
            let index = index * dispatch_size().y + dispatch_id().y;
            let data_index = node.data_offset + index;
            let radiance_deposit = cache.data.read(data_index).var();
            radiance_deposit.merge(
                // Should never happen.
                Vec3::expr(1000.0, 0.0, 0.0),
            );
            cache.data.write(data_index, radiance_deposit);
        }
    }));
    let compute_gbuffer_kernel = DEVICE.create_kernel::<fn(View)>(&track!(|view| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let hit = world.trace(view.pos, ray_dir);
        if !hit.valid() {
            depth_buf.write(pixel, 0.0);
            return;
        }
        depth_buf.write(pixel, hit.committed_ray_t);
        let normal = world.get_normal_flat(hit);
        normal_buf.write(pixel, SphericalEncoder::encode(normal));
    }));
    let compute_radiance_kernel = DEVICE.create_kernel_with_options::<fn(View, u32, u32)>(
        KernelBuildOptions {
            name: Some("compute_radiance_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick, flags| {
            set_block_size([8, 8, 1]);
            let pixel = dispatch_id().xy();
            let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

            let depth = depth_buf.read(pixel);
            if depth == 0.0 {
                return;
            }
            let normal = SphericalEncoder::decode(normal_buf.read(pixel));
            let world_pos = view.pos + ray_dir * depth;

            let bounce_ray = as_normal(hemi_noise.read(
                pixel.extend(tick)
                    % Vec3::new(hemi_noise.width(), hemi_noise.height(), hemi_noise.depth()),
            ));
            let bounce_ray = rotate_hemi(normal, bounce_ray);
            // TODO: It should probably be uniform rather than cos sampling but it doesn't actually change much.
            // let weight = bounce_ray.dot(normal);
            // let bounce_ray = sample_bounce(normal, pcg3df(pixel.extend(tick)).xy());

            let probe = cache.quantize(world_pos);

            // TODO: Can do a single ray trace instead of 6.
            let bounce_hit = world.trace_segment(
                world_pos,
                bounce_ray,
                Vec2::expr(0.01, 100000.0) * (1_u32 << probe.position.w).cast_f32(),
            );
            let (hit_level, fluence) = if bounce_hit.valid() {
                let level = keter::max(
                    (bounce_hit.committed_ray_t / (12.0 * 16.0_f32)).log2(), // TODO: change to depend on grid size.
                    0.0,
                )
                .ceil()
                .cast_u32();
                let level = keter::min(level, cache.num_levels.expr() - 1);
                let radiance = if (flags & RADIANCE_DIRECT_ONLY) != 0 {
                    Vec3::splat_expr(0.0_f32)
                } else {
                    let bounce_pos = world_pos + bounce_ray * bounce_hit.committed_ray_t;
                    let color = Vec3::splat_expr(0.7); // world.get_color(bounce_start, bounce_ray, bounce_hit);
                    let value = cache.query_diffuse(
                        cache.quantize(bounce_pos),
                        world.get_normal_flat(bounce_hit),
                    );
                    if value.x == -1.0 {
                        Vec3::splat_expr(0.0_f32) // skip because undefined
                    } else {
                        value * color
                    }
                };
                (level, radiance.extend(0.0))
            } else {
                (
                    cache.num_levels.expr() - 1,
                    sunlight_radiance(bounce_ray).extend(0.0),
                )
            };
            for level in 0.expr()..hit_level {
                cache.deposit(
                    probe.shift_level(level),
                    bounce_ray,
                    RadianceDeposit::expr(Vec3::splat_expr(0.0_f32).extend(1.0)),
                );
            }
            cache.deposit(
                probe.shift_level(hit_level),
                bounce_ray,
                RadianceDeposit::expr(fluence),
            );
        }),
    );

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32, u32, u32)>(
        KernelBuildOptions {
            name: Some("draw_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick, mode, level| {
            set_block_size([8, 8, 1]);

            let pixel = dispatch_id().xy();
            let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

            let depth = depth_buf.read(pixel);
            if depth == 0.0 {
                app.set_pixel(pixel.cast_i32(), sunlight_radiance(ray_dir));
                return;
            }
            let normal = SphericalEncoder::decode(normal_buf.read(pixel));
            let world_pos = view.pos + ray_dir * depth;

            // TODO: Decide whether to have the circle cutoffs be around the sample point or be around the cell center.

            let probe = cache.quantize(world_pos).shift_level(level);

            let color = if mode == 0 {
                cache.query_diffuse(probe, normal)
            } else if mode == 1 {
                cache.query_diffuse_bilinear(world_pos, level, normal)
            } else if mode == 2 {
                if cache.contains(probe) {
                    pcg3df(probe.position.xyz())
                } else {
                    Vec3::splat_expr(0.0)
                }
            } else if mode == 3 {
                let (has_probe, cost) = cache.contains_cost(probe);
                if has_probe {
                    plasma(cost.cast_f32() / 9.0)
                } else {
                    Vec3::splat_expr(0.0)
                }
            } else if mode == 4 {
                inferno(1.0 - reinhard(depth / 1000.0))
            } else if mode == 5 {
                normal * 0.5 + Vec3::splat_expr(0.5)
            } else if mode == 6 || mode == 7 {
                let offset = pcg3d(Vec3::expr(tick / hemi_noise.depth(), 392, 132)).xy()
                    % Vec2::new(hemi_noise.width(), hemi_noise.height());
                let bounce_ray = as_normal(hemi_noise.read(
                    (pixel + offset).extend(tick)
                        % Vec3::new(hemi_noise.width(), hemi_noise.height(), hemi_noise.depth()),
                ));
                let bounce_ray = rotate_hemi(normal, bounce_ray);
                // let bounce_ray = sample_bounce(normal, pcg3df(pixel.extend(tick)).xy());

                let bounce_hit =
                    world.trace_segment(world_pos, bounce_ray, Vec2::expr(0.1_f32, 10000.0));
                let radiance = if bounce_hit.valid() {
                    Vec3::splat_expr(0.0_f32)
                } else {
                    sunlight_radiance(bounce_ray)
                };
                if mode == 7 {
                    let value = accum_buf.read(pixel) + radiance.extend(1.0);
                    accum_buf.write(pixel, value);
                    value.xyz() / value.w
                } else {
                    radiance
                }
            /*} else if mode == 7 || mode == 8 {
                let grid_size = cache.base_grid_size * (1_u32 << level).cast_f32();
                let start = view.pos / grid_size;
                let color = Vec3::splat(0.0_f32).var();
                utils::dda(
                    start,
                    ray_dir,
                    10000.0_f32.expr(),
                    (Vec3::splat_expr(-1000.0_f32), Vec3::splat_expr(1000.0_f32)),
                    |pos, last_t, _next_t| {
                        let probe = cache.get_probe(pos, level);
                        let start_pos =
                            view.pos + ray_dir * last_t - (pos.cast_f32() + 0.5) * grid_size;
                        if cache.contains(probe) {
                            let (time, intersected) =
                                utils::intersect_sphere(start_pos, ray_dir, 0.3 * grid_size);
                            if intersected {
                                let intersect_dir = (start_pos + ray_dir * time.x).normalize();
                                *color = // inferno(1.0 - reinhard((time.x + last_t) / 1000.0));

                                if mode == 7 { cache.query_dir(probe, intersect_dir)
                                    + Vec3::expr(0.0_f32, 0.0, 0.)
                                } else {
                                    let encoder = cache.get_encoder(level);
                                    encoder.grid_to_dir(
                                        encoder.dir_to_grid(intersect_dir)
                                    ) * 0.5 + 0.5
                                };
                            }
                            intersected
                        } else {
                            false.expr()
                        }
                    },
                );
                **color
                // cache.query_diffuse(probe, normal)
                //     + cache.query_dir_bilinear(probe, reflect(ray_dir, normal))
            */
            } else {
                Vec3::expr(1.0, 0.0, 0.0)
            };
            app.set_pixel(pixel.cast_i32(), color);
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 4.868212,
        pitch: 0.4147132,
        pos: glam::Vec3::new(132.45715, -404.1544, 651.1271),
        fov: 1.15,
        // yaw: 3.2746875,
        // pitch: 1.0245812,
        // pos: glam::Vec3::new(16.342783, 671.79944, 1391.7502),
        // fov: 1.15,
    };
    let mut running = true;
    let mut mode = 0;
    let mut level = 0;
    let mut cutoff_mode = 1;
    let mut radiance_flags = RADIANCE_DIRECT_ONLY;

    let mut profiler = Profiler::new();

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = if rt.key_down(KeyCode::ControlLeft) {
            20.0
        } else {
            800.0
        } * rt.frame_time();
        let camera_updated = camera.apply_movement(&rt.keys_down, rot_speed, move_speed);
        if rt.key_pressed(KeyCode::KeyP) {
            println!("Camera: {camera:?}");
        }
        if rt.key_pressed(KeyCode::KeyM) {
            mode = (mode + 1) % 9;
            println!("Drawing mode {mode}");
        }
        if rt.key_pressed(KeyCode::KeyL) {
            level = (level + 1) % cache.num_levels;
            println!("Level set {level}");
        }
        if rt.key_pressed(KeyCode::KeyC) {
            cutoff_mode = (cutoff_mode + 1) % 3;
            println!("Cutoff mode set {cutoff_mode}");
        }
        if rt.key_pressed(KeyCode::KeyO) {
            running = !running;
            println!("Running: {running}");
        }
        if rt.key_pressed(KeyCode::KeyR) {
            radiance_flags ^= RADIANCE_DIRECT_ONLY;
        }
        let mut commands = vec![];
        commands.push(
            compute_gbuffer_kernel
                .dispatch_async(rt.dispatch_size(), &camera.view())
                .debug("Compute GBuffer"),
        );

        if running || rt.key_pressed(KeyCode::Period) {
            commands.extend([
                cache.set_view(camera.view()),
                lower_cache
                    .dispatch_async([cache.data.len() as u32, 1, 1])
                    .debug("Lower Cache"),
                compute_radiance_kernel
                    .dispatch_async(
                        rt.dispatch_size(),
                        &camera.view(),
                        &rt.tick,
                        &radiance_flags,
                    )
                    .debug("Compute Radiance"),
            ]);
            for l in (0..cache.num_levels).rev() {
                let dispatch_size = [cache.num_nodes_at_level(l), 1 << l, 1];
                commands.push(
                    (if l == cache.num_levels - 1 {
                        merge_final_cache.dispatch_async(dispatch_size)
                    } else {
                        merge_cache.dispatch_async(dispatch_size, &l)
                    })
                    .debug(format!("Merge Cache Level {l}")),
                );
            }
        }

        if camera_updated {
            commands.push(
                reset_tex2d
                    .dispatch_async([accum_buf.width(), accum_buf.height(), 1], &accum_buf)
                    .debug("Reset Accumulation"),
            );
        }

        commands.push(
            draw_kernel
                .dispatch_async(rt.dispatch_size(), &camera.view(), &rt.tick, &mode, &level)
                .debug("Draw"),
        );

        profiler.record(commands.chain().execute_timed());

        if profiler.time() > 1000.0 {
            profiler.print(
                &[
                    "Compute GBuffer",
                    "Merge",
                    "Draw",
                    "Lower Cache",
                    "Compute Radiance",
                ],
                true,
            );
            profiler.reset();
        }

        if rt.key_down(KeyCode::Backslash) {
            // println!("Resetting cache");
            reset_cache_nodes.dispatch([cache.nodes.len() as u32, 1, 1]);
            reset_cache_data.dispatch([cache.data.len() as u32, 1, 1]);
        }
    });
}
