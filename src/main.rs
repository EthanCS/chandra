#![feature(array_windows)]

/*
TODOS:
[X] Add more lights, fancier primary ray stuff?
[X] C-1 merging for better contacts
[ ] Optimize, switch to u64 everywhere
[ ] Better hashmap impl without dropping things? Bias towards primary rays first.
[ ] Some sorta anderson acceleration to make the infinite bounce convergence less obvious?

[ ] Precompute ndotl lighting separately? At least for offscreen?
[ ] Add separate world-space cache with lower resolution? Preferably get both to use the same map to prevent overflow...
[ ] Ray counting / quality metrics.
[ ] Test how much memory saved if each probe element is sparse?
[ ] Optimize the c-1 merging or remove it.

Hashmap:
It's fine if missing takes many iterations to hit (especially in 2nd / 3rd bounce, can cap the lookup as well).
Missing on primary should not happen at all. For efficiency, also make primary override secondary? Can that be done atomically??

So: Each node has a "rating" bit. When inserting:
* Scan for the first node with a higher rating; atomically set (fetch_min) it.
  * Then set its position to invalid.
* Scan for the first node with position invalid, and claim it.
This probably works fine, but only for the min-level insert.

Idea: Two keys per node: primary and secondary. Look up primary over secondary.

BUGS:
[ ] Convergence is just pretty damn slow currently. Make it so that lower cascades use higher cascade values if they don't have enough info?

Split into two caches. Also measure ray divergence and use that to stochastically reduce ray count.
*/

use std::f32::consts::TAU;
use std::path::PathBuf;

use glam::Mat3 as FMat3;
use image::{GenericImageView, ImageReader};
use keter::bindless::BINDLESS;
use keter::graph::profile::Profiler;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};
use yesod::camera::{Camera, View};
use yesod::color::{inferno, plasma, reinhard};
use yesod::device_println;
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::printer::global::{PrintExt, flush_printer};
use yesod::rand::{pcg3d, pcg3df};

use crate::cache::{
    Fluence, HashCache, INDEX_SCALING, LoadedNode, LoadedNodeComps, Node, POSITION_INVALID, Probe,
    ProbeComps, Radiance, RadianceDeposit, RadianceElement, decode_position,
};
use crate::world::TriangleWorld;

pub mod cache;
pub mod color;
pub mod fluence;
pub mod utils;
pub mod world;

const RAY_LENGTH: f32 = 12.0;

const RADIANCE_DIRECT_ONLY: u32 = 0b1;
const C_1_MERGE: u32 = 0b10;
const C_0_START: f32 = 1.0;

const QUALITY_CUTOFF: f32 = 0.0; // 0.3;
const TARGET_QUALITY_MULTIPLIER: f32 = 50.0;
const QUALITY_DECAY: f32 = 1.0;
const MAX_RAYS: u32 = 1 << 22;

fn sunlight_dir() -> Vec3<f32> {
    let dir = glam::Vec3::new(0.1, 0.5, 1.0);
    dir.normalize().into()
}

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

#[repr(C)]
#[derive(Value, Debug, Clone, Copy)]
struct Ray {
    dir: Vec3<f32>,
    start: Vec3<f32>,
    base_probe: Probe,
    // base_probe: Probe,
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [2048; 2]).agx().resize().init();

    yesod::printer::global::set_capacity(1024 * 1024);

    // let sphere_noise =

    let hemi_noise = load_noise_textures(|i| {
        PathBuf::from(format!(
            "assets/noise/sphere/exp/sphere_coshemi_binomial3x3_exp0101_product_{i}.png",
        ))
    });

    /*let world = {
        let a = 6000.0;
        let b = 50.0;
        let h1 = 7.0;
        let h2 = 307.0;

        TriangleWorld::new(
            &[
                [-a, -a, h1],
                [-a, a, h1],
                [a, -a, h1],
                [a, a, h1],
                [-b, -b, h2],
                [-b, b, h2],
                [-b, -b, h1],
                [-b, b, h1],
            ],
            &[[0.0; 3]; 4],
            &[[0, 2, 1], [1, 2, 3], [0, 2, 1], [1, 2, 3]],
            &[[0, 0], [2, 4]],
        )
    };*/

    let world = TriangleWorld::load(
        "assets/sponza.obj",
        FMat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    );

    let depth_buf =
        DEVICE.create_tex2d::<f32>(PixelStorage::Float1, app.size()[0], app.size()[1], 1);
    let depth_buf = BINDLESS.push_tex2d(
        depth_buf,
        Sampler {
            filter: SamplerFilter::LinearLinear,
            address: SamplerAddress::Edge,
        },
    );
    BINDLESS.update();
    let normal_buf =
        DEVICE.create_tex2d::<Vec2<f32>>(PixelStorage::Short2, app.size()[0], app.size()[1], 1);
    let accum_buf =
        DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, app.size()[0], app.size()[1], 1);
    let sunlight_buf =
        DEVICE.create_tex3d::<Vec3<f32>>(PixelStorage::Float4, app.size()[0], app.size()[1], 2, 1);
    // let cache = HashCache::new(16.0, 100000.0, 4, 1 << 18, 6);
    let cache = HashCache::new(16.0 / 8.0, 16.0 * 50.0 / 8.0, 4, 1 << 18, 6);
    let primary_cache = DEVICE.create_buffer_from_fn::<Vec4<f32>>(
        (cache.base_nodes
            * cache.base_angular_resolution.pow(2)
            * INDEX_SCALING.x
            * INDEX_SCALING.y) as usize,
        |_| Vec4::new(0.0, 0.0, 0.0, 0.00001),
    );

    let base_probe_num_rays =
        DEVICE.create_buffer_from_fn::<u32>(cache.base_nodes as usize, |_| 0_u32);
    let node_quality = DEVICE.create_buffer_from_fn::<f32>(cache.nodes.len(), |_| 0.0_f32);
    let traced_rays = DEVICE.create_buffer::<Ray>(MAX_RAYS as usize);
    let next_traced_ray = Singleton::<u32>::new(0_u32);

    let cache_memory = cache.data.len() * std::mem::size_of::<RadianceElement>()
        + cache.nodes.len() * std::mem::size_of::<Node>();
    println!("Cache memory: {} MB", cache_memory / 1024 / 1024);

    let decay_cache = DEVICE.create_kernel::<fn(u32)>(&track!(|level| {
        let node_index = dispatch_id().x;
        let node = cache.read_node(level, node_index).var();
        if node.position == POSITION_INVALID {
            return;
        }
        let quality = node_quality.read(node_index).var();
        *quality *= QUALITY_DECAY;
        let is_invalid =
            quality < QUALITY_CUTOFF * cache.get_encoder(level).linear_size().cast_f32();
        if is_invalid {
            *node.position = POSITION_INVALID;
            *quality = 0.0;
        }
        cache.write_node(level, node_index, **node);
        node_quality.write(node_index + cache.get_node_offset(level), **quality);

        if is_invalid {
            let position = decode_position(**node.position);
            let probe = Probe::from_comps_expr(ProbeComps { position, level });
            let node = cache.load_index(probe, node_index);

            for index in 0_u32.expr()..node.encoder.linear_size() {
                cache
                    .data
                    .write(node.data_offset + index, RadianceElement::default());
            }
            if level == 0 {
                for index in 0_u32.expr()..node.encoder.linear_size() {
                    primary_cache
                        .write(node.data_offset + index, Vec4::new(0.0, 0.0, 0.0, 0.00001));
                }
            }
        }
    }));
    let reset_tex2d = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>)>(&track!(|tex| {
        let pixel = dispatch_id().xy();
        tex.write(pixel, Vec4::splat_expr(0.0_f32));
    }));
    let reset_cache_nodes = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache.nodes.write(dispatch_id().x, Node::default());
        node_quality.write(dispatch_id().x, 0.0_f32);
    }));
    let reset_cache_data = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache
            .data
            .write(dispatch_id().x, RadianceElement::default());
    }));
    let reset_primary_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        primary_cache.write(dispatch_id().x, Vec4::new(0.0, 0.0, 0.0, 0.00001));
    }));

    let reset_base_probe_num_rays = DEVICE.create_kernel::<fn()>(&track!(|| {
        base_probe_num_rays.write(dispatch_id().x, 0_u32);
    }));
    let reset_next_traced_ray = DEVICE.create_kernel::<fn()>(&track!(|| {
        next_traced_ray.write(0_u32);
    }));

    #[derive(Clone, Copy)]
    enum MergeLevel {
        // First,
        Last,
        Middle(Expr<u32>),
    }
    let merge_cache_base = track!(|merge_level: MergeLevel| {
        let level = match merge_level {
            // MergeLevel::First => 0.expr(),
            MergeLevel::Last => (cache.num_levels - 1).expr(),
            MergeLevel::Middle(l) => l,
        };
        let node_index = dispatch_id().x;
        let node = cache.read_node(level, node_index);
        if node.position == POSITION_INVALID {
            return;
        }
        let position = decode_position(node.position);
        let probe = Probe::from_comps_expr(ProbeComps { position, level });
        let node = cache.load_index(probe, node_index);
        // TODO: best to process in batches of 8.

        // let total_weight = 1.0_f32.var();
        // let next = [(
        //     cache.query(
        //         probe.next_nearest(),
        //         |x| x,
        //         || {
        //             return;
        //             LoadedNode::expr_zeroed()
        //         },
        //     ),
        //     1.0_f32.expr(),
        // )];

        // TODO: pre-load all of the data batched.
        let total_weight = 0.0_f32.var();
        let next_probes = (!matches!(merge_level, MergeLevel::Last)).then(|| {
            let next_probes = probe.next_bilinear().map(|(probe, weight)| {
                let (next, weight) = cache.query(
                    probe,
                    |n| (n, weight),
                    || {
                        (
                            LoadedNode::from_comps_expr(LoadedNodeComps {
                                probe,
                                encoder: cache.get_encoder(probe.level),
                                data_offset: 0_u32.expr(),
                                node_index: 0_u32.expr(),
                            }),
                            0_f32.expr(),
                        )
                    },
                );
                *total_weight += weight;
                (next, weight)
            });
            *total_weight = keter::max(total_weight, 0.001);
            next_probes.map(|(next, weight)| (next, weight / total_weight))
        });
        for index in 0_u32.expr()..(node.encoder.linear_size() / dispatch_size().y) {
            let index = index * dispatch_size().y + dispatch_id().y;
            let dir = node.encoder.linear_to_grid(index);
            let total_radiance = Vec3::splat(0.0_f32).var();
            if let MergeLevel::Last = merge_level {
                // Shouldn't be used.
                *total_radiance = Vec3::expr(10000.0, 0.0, 0.0);
            } else {
                for offset in [[0, 0], [0, 1], [1, 0], [1, 1]].map(Vec2::from) {
                    let sample_dir = dir * 2 + offset;
                    // TODO: Can optimize because half are empty.
                    for (next, weight) in next_probes.unwrap() {
                        let sample_index = next.encoder.grid_to_linear(sample_dir);
                        *total_radiance +=
                            weight * cache.data.read(next.data_offset + sample_index).radiance();
                    }
                }
            }
            let data_index = node.data_offset + index;
            let radiance_deposit = cache.data.read(data_index).var();
            radiance_deposit.merge(total_radiance / 4.0);
            cache.data.write(data_index, radiance_deposit);
        }
    });

    let add_primary_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        let node = cache.read_node(0.expr(), index);
        if node.position == POSITION_INVALID {
            return;
        }
        let node = cache.load_index(
            Probe::from_comps_expr(ProbeComps {
                position: decode_position(node.position),
                level: 0.expr(),
            }),
            index,
        );
        for index in 0_u32.expr()..node.encoder.linear_size() {
            let data_index = node.data_offset + index;
            let radiance_deposit = cache.data.read(data_index).var();
            let primary = primary_cache.read(data_index);
            *radiance_deposit.radiance_weight += (primary.xyz() / primary.w).extend(0.0);
            cache.data.write(data_index, radiance_deposit);
        }
    }));

    let merge_cache = DEVICE.create_kernel::<fn(u32)>(&|level| {
        merge_cache_base(MergeLevel::Middle(level));
    });
    let merge_last_cache = DEVICE.create_kernel::<fn()>(&|| {
        merge_cache_base(MergeLevel::Last);
    });
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

    let compute_num_rays_kernel = DEVICE.create_kernel::<fn(View, u32)>(&track!(|view, tick| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let depth = depth_buf.read(pixel);
        if depth == 0.0 {
            return;
        }
        let world_pos = (view.pos + ray_dir * depth).var();

        let interpolate_side = pixel.x % 2 == pixel.y % 2;

        let probe = cache.quantize(**world_pos, interpolate_side);

        let loaded_node = cache.query_init(
            probe,
            |n| n,
            || {
                return;
                **LoadedNode::var_zeroed()
            },
        );
        base_probe_num_rays
            .atomic_ref(loaded_node.node_index)
            .fetch_add(1_u32.expr());
    }));
    let compute_rays_kernel = DEVICE.create_kernel::<fn(View, u32)>(&track!(|view, tick| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let depth = depth_buf.read(pixel);
        if depth == 0.0 {
            return;
        }
        let normal = SphericalEncoder::decode(normal_buf.read(pixel));

        let world_pos = view.pos + ray_dir * depth;

        let interpolate_side = pixel.x % 2 == pixel.y % 2;

        let probe = cache.quantize(world_pos, interpolate_side);

        let loaded_node = cache.query_init(
            probe,
            |n| n,
            || {
                return;
                **LoadedNode::var_zeroed()
            },
        );

        let bounce_ray = rotate_hemi(
            normal,
            as_normal(hemi_noise.read(
                pixel.extend(tick)
                    % Vec3::new(hemi_noise.width(), hemi_noise.height(), hemi_noise.depth()),
            )),
        );

        let num_rays = base_probe_num_rays.read(loaded_node.node_index);
        let target_rays = keter::max(
            cache.get_encoder(0.expr()).linear_size().cast_f32() * TARGET_QUALITY_MULTIPLIER
                - node_quality.read(loaded_node.node_index),
            0.0,
        );
        let extinction = target_rays / keter::max(num_rays, 1).cast_f32();

        if pcg3df((pixel + Vec2::new(3528, 9306)).extend(tick)).x < extinction {
            let ray = Ray::from_comps_expr(RayComps {
                dir: bounce_ray,
                start: world_pos,
                base_probe: probe,
            });
            let ray_index = next_traced_ray.atomic().fetch_add(1_u32.expr());
            if ray_index < traced_rays.len_expr_u32() {
                traced_rays.write(ray_index, ray);
            }
        }
    }));

    let compute_radiance_kernel = DEVICE.create_kernel_with_options::<fn(u32, u32)>(
        KernelBuildOptions {
            name: Some("compute_radiance_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|_tick, flags| {
            set_block_size([64, 1, 1]);

            if dispatch_id().x >= next_traced_ray.read() {
                return;
            }

            let ray = traced_rays.read(dispatch_id().x);
            let ray_dir = ray.dir.normalize();
            let world_pos = ray.start.var();

            for i in 0_u32.expr()..1_u32.expr() {
                let bounce_ray = ray_dir;

                let probe = ray.base_probe;

                let bounce_hit = world.trace_segment(
                    **world_pos,
                    bounce_ray,
                    Vec2::expr(
                        if flags & C_1_MERGE != 0 {
                            C_0_START
                                * cache.base_grid_size
                                * (1_u32 << probe.position.w).cast_f32()
                        } else {
                            0.01.expr()
                        },
                        100000.0,
                    ), // * (1_u32 << probe.position.w).cast_f32(),
                );
                let bounce_pos = world_pos + bounce_ray * bounce_hit.committed_ray_t;

                let (hit_level, fluence) = if bounce_hit.valid() {
                    let level = keter::max(
                        (bounce_hit.committed_ray_t
                            / (RAY_LENGTH
                                * cache.base_grid_size
                                * (1_u32 << probe.position.w).cast_f32()))
                        .log2(), // TODO: change to depend on grid size.
                        0.0,
                    )
                    .ceil()
                    .cast_u32();
                    let level = keter::min(level, cache.num_levels.expr() - 1);
                    let radiance = if (flags & RADIANCE_DIRECT_ONLY) != 0 {
                        if bounce_hit.inst == 1 {
                            Vec3::expr(10.0, 0.0, 0.0)
                        } else {
                            Vec3::splat_expr(0.0_f32)
                        }
                    } else {
                        let color = Vec3::splat_expr(0.7); // world.get_color(bounce_start, bounce_ray, bounce_hit);
                        let value = cache.query_diffuse(
                            cache.quantize(bounce_pos, false.expr()),
                            world.get_normal_flat(bounce_hit),
                            0.expr(),
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
                        Vec4::splat_expr(0.0), // sunlight_radiance(bounce_ray).extend(0.0),
                    )
                };

                let deposit = |level, fluence| {
                    cache.query_init(
                        probe.shift_level(level),
                        |node| {
                            let data = cache.data.atomic_ref(
                                node.data_offset + node.encoder.dir_to_linear(bounce_ray),
                            );
                            data.deposit(RadianceDeposit::expr(fluence));
                            node_quality
                                .atomic_ref(node.node_index + cache.get_node_offset(level))
                                .fetch_add(1.0);
                        },
                        || {},
                    )
                    // cache.deposit(probe.shift_level(level), bounce_ray, RadianceDeposit::expr(fluence));
                };

                for level in 0.expr()..hit_level {
                    deposit(level, Vec3::splat_expr(0.0_f32).extend(1.0));
                }
                deposit(hit_level, fluence);

                let sunlight = world
                    .trace_segment(
                        **world_pos,
                        sunlight_dir().expr(),
                        Vec2::expr(1.0, 100000.0),
                    )
                    .valid()
                    .select(Vec3::splat_expr(0.0_f32), Vec3::splat_expr(10.0_f32));
                // sunlight_buf.write(pixel.extend(i), sunlight);
                // TODO: Added sunlight is downsampled at primary currently.
                cache.query(
                    probe,
                    |node| {
                        let value = primary_cache.atomic_ref(
                            node.data_offset + node.encoder.dir_to_linear(sunlight_dir().expr()),
                        );
                        value.x.fetch_add(sunlight.x);
                        value.y.fetch_add(sunlight.y);
                        value.z.fetch_add(sunlight.z);
                        value.w.fetch_add(1.0);
                    },
                    || {},
                );

                if !bounce_hit.valid() || (flags & RADIANCE_DIRECT_ONLY) != 0 {
                    break;
                }

                *world_pos = bounce_pos;
                // *normal = bounce_normal;
            }
        }),
    );

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32, u32, u32, u32, Vec2<u32>)>(
        KernelBuildOptions {
            name: Some("draw_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick, flags, mode, level, cursor_pixel| {
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
            // let sunlight = sunlight_buf.read(pixel.extend(0));

            let probe = cache.quantize(world_pos, false.expr()).shift_level(level);

            let shadow_mask = 0u32.var();

            if (flags & C_1_MERGE) != 0 && mode < 2 {
                let c_1_len =
                    C_0_START * cache.base_grid_size * (1_u32 << probe.position.w).cast_f32();
                let encoder = cache.get_encoder(0.expr());

                for i in 0_u32.expr()..32_u32.expr() {
                    let offset = pcg3df(pixel.extend(i + tick * 32)).xy();
                    let c_1_dir =
                        encoder.grid_to_dir_f32(encoder.linear_to_grid(i).cast_f32() + offset);
                    let c_1_dir = c_1_dir * c_1_len;
                    // raymarch in screenspace
                    for j in 0_u32.expr()..4_u32.expr() {
                        let sample_pos = world_pos + c_1_dir * ((j.cast_f32() + 0.5) / 4.0);
                        let screen_pos = view.project(sample_pos);
                        let sample_depth = (sample_pos - view.pos).length();
                        let object_depth = depth_buf
                            .var()
                            .sample((screen_pos + 0.5) / view.screen_size)
                            .x;
                        if object_depth != 0.0 && object_depth < sample_depth {
                            *shadow_mask |= 1 << i;
                            break;
                        }
                    }
                }
            }

            let color = if mode == 0 {
                let other_probe = cache.quantize(world_pos, true.expr()).shift_level(level);
                let factor = cache.projection().interpolation_factor(world_pos);
                let result = cache.query_diffuse(probe, normal, **shadow_mask);
                let result = if factor == 1.0 {
                    result
                } else {
                    let other_result = cache.query_diffuse(other_probe, normal, **shadow_mask);
                    result * factor + other_result * (1.0 - factor)
                };
                if result.x.is_nan() {
                    Vec3::expr(0.0, 0.0, 1.0)
                } else if result.x.is_infinite() {
                    Vec3::expr(1.0, 0.0, 1.0)
                } else {
                    result // + sunlight * keter::max(normal.dot(sunlight_dir()), 0.0)
                }
            } else if mode == 1 {
                let factor = cache.projection().interpolation_factor(world_pos);
                let result = cache.query_diffuse_bilinear(
                    world_pos,
                    level,
                    normal,
                    false.expr(),
                    **shadow_mask,
                );
                let result = if factor == 1.0 {
                    result
                } else {
                    let other_result = cache.query_diffuse_bilinear(
                        world_pos,
                        level,
                        normal,
                        true.expr(),
                        **shadow_mask,
                    );
                    result * factor + other_result * (1.0 - factor)
                };
                result // + sunlight * keter::max(normal.dot(sunlight_dir()), 0.0)
            } else if mode == 2 {
                if cache.contains(probe) {
                    inferno((probe.position.w.cast_f32() + 0.5) / 9.0)
                        * pcg3df(probe.position.xyz()).x
                } else {
                    Vec3::splat_expr(0.0)
                }
            } else if mode == 3 {
                cache.query(
                    probe,
                    |node| {
                        let quality =
                            node_quality.read(node.node_index + cache.get_node_offset(level));
                        let quality_ratio = quality
                            / (TARGET_QUALITY_MULTIPLIER
                                * cache.get_encoder(level).linear_size().cast_f32());
                        // if (cursor_pixel == pixel).all() {
                        //     device_println!(
                        //         "Probe at quality: {}",
                        //         // probe.position.w.host(),
                        //         quality_ratio.host()
                        //     );
                        // }
                        inferno(quality_ratio)
                    },
                    || Vec3::splat_expr(0.0),
                )
            } else if mode == 4 {
                let probe = cache.quantize(world_pos, false.expr());
                cache.query(
                    probe,
                    |node| {
                        let num_rays = base_probe_num_rays.read(node.node_index);
                        let target_rays = keter::max(
                            cache.get_encoder(0.expr()).linear_size().cast_f32()
                                * TARGET_QUALITY_MULTIPLIER
                                - node_quality.read(node.node_index),
                            0.0,
                        );
                        let extinction = target_rays / keter::max(num_rays, 1).cast_f32();
                        inferno(extinction)
                    },
                    || Vec3::splat_expr(0.0),
                )
                // inferno(1.0 - reinhard(depth / 1000.0))
            } else if mode == 5 {
                Vec3::splat_expr(cache.projection().interpolation_factor(world_pos))
                // normal * 0.5 + Vec3::splat_expr(0.5)
            } else if mode == 6 {
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
                let value = accum_buf.read(pixel) + radiance.extend(1.0);
                accum_buf.write(pixel, value);
                value.xyz() / value.w
            } else if mode == 7 {
                inferno(1.0 - reinhard((cache.dequantize(probe) - world_pos).length()))
            } else if mode == 8 {
                let hit = world.trace(view.pos, ray_dir);
                if (cursor_pixel == pixel).all() {
                    let inst = hit.inst;
                    device_println!("Hit at {}", inst.host());
                }
                pcg3df(Vec3::expr(hit.inst, 37, 926))
            } else if mode == 9 {
                let (has_probe, cost) = cache.contains_cost(probe);
                if has_probe {
                    plasma(cost.cast_f32() / 9.0)
                } else {
                    Vec3::splat_expr(0.0)
                }
            } else {
                Vec3::expr(1.0, 0.0, 0.0)
            };
            app.set_pixel(pixel.cast_i32(), color);
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 7.4975963,
        pitch: 0.332344,
        pos: glam::Vec3::new(-78.53391, -402.59995, 495.0998),
        fov: 1.15,
    };
    let mut running = true;
    let mut mode = 1;
    let mut level = 0;
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
            mode = (mode + 1) % 10;
            println!("Drawing mode {mode}");
            let tonemap = matches!(mode, 0 | 1 | 6);
            rt.tonemap(tonemap);
        }
        if rt.key_pressed(KeyCode::KeyL) {
            level = (level + 1) % cache.num_levels;
            println!("Level set {level}");
        }
        if rt.key_pressed(KeyCode::KeyO) {
            running = !running;
            println!("Running: {running}");
        }
        if rt.key_pressed(KeyCode::KeyR) {
            radiance_flags ^= RADIANCE_DIRECT_ONLY;
        }
        if rt.key_pressed(KeyCode::KeyT) {
            radiance_flags ^= C_1_MERGE;
        }
        let mut commands = vec![];
        commands.push(
            compute_gbuffer_kernel
                .dispatch_async(rt.dispatch_size(), &camera.view())
                .debug("Compute GBuffer"),
        );

        let running = running || rt.key_pressed(KeyCode::Period);
        if running {
            if mode == 3 {
                println!("Num rays traced: {}", next_traced_ray.read_blocking());
            }
            commands.push(cache.set_view(camera.view()));
            for l in (0..cache.num_levels).rev() {
                let dispatch_size = [cache.num_nodes_at_level(l), 1 << l, 1];
                commands.push(
                    decay_cache
                        .dispatch_async(dispatch_size, &l)
                        .debug("Decay Cache"),
                );
            }

            // Reset ray counts
            commands.push(
                reset_base_probe_num_rays
                    .dispatch_async([base_probe_num_rays.len() as u32, 1, 1])
                    .debug("Reset Base Probe Num Rays"),
            );
            commands.push(
                reset_next_traced_ray
                    .dispatch_async([1, 1, 1])
                    .debug("Reset Next Traced Ray"),
            );

            commands.push(
                compute_num_rays_kernel
                    .dispatch_async(rt.dispatch_size(), &camera.view(), &rt.tick)
                    .debug("Compute Radiance: Num Rays"),
            );
            commands.push(
                compute_rays_kernel
                    .dispatch_async(rt.dispatch_size(), &camera.view(), &rt.tick)
                    .debug("Compute Radiance: Rays"),
            );
            commands.push(
                compute_radiance_kernel
                    .dispatch_async([MAX_RAYS, 1, 1], &rt.tick, &radiance_flags)
                    .debug("Compute Radiance: Trace"),
            );

            for l in (0..cache.num_levels).rev() {
                let dispatch_size = [cache.num_nodes_at_level(l), 1 << l, 1];
                commands.push(
                    (if l == cache.num_levels - 1 {
                        merge_last_cache.dispatch_async(dispatch_size)
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
                .dispatch_async(
                    rt.dispatch_size(),
                    &camera.view(),
                    &rt.tick,
                    &radiance_flags,
                    &mode,
                    &level,
                    &rt.cursor_position.map(|x| x as u32),
                )
                .debug("Draw"),
        );

        if running {
            commands.push(
                add_primary_cache
                    .dispatch_async([cache.num_nodes_at_level(0), 1, 1])
                    .debug("Add Primary Cache"),
            );
        }

        profiler.record(commands.chain().execute_timed());

        if profiler.time() > 1000.0 {
            profiler.print(
                &[
                    "Compute GBuffer",
                    "Decay Cache",
                    "Compute Radiance",
                    "Merge",
                    "Add Primary Cache",
                    "Draw",
                ],
                true,
            );
            profiler.reset();
        }

        if rt.key_down(KeyCode::Backslash) {
            // println!("Resetting cache");
            reset_cache_nodes.dispatch([cache.nodes.len() as u32, 1, 1]);
            reset_cache_data.dispatch([cache.data.len() as u32, 1, 1]);
            reset_primary_cache.dispatch([primary_cache.len() as u32, 1, 1]);
        }
        flush_printer();
    });
}
