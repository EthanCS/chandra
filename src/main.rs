#![feature(array_windows)]

/*
TODOS:
[X] Add more lights, fancier primary ray stuff?
[ ] Optimize, switch to u64 everywhere
[ ] Better hashmap impl without dropping things? Bias towards primary rays first.
[ ] Some sorta anderson acceleration to make the infinite bounce convergence less obvious?

[ ] Precompute ndotl lighting separately? At least for offscreen?
[ ] Add separate world-space cache with lower resolution? Preferably get both to use the same map to prevent overflow...
[ ] Ray counting / quality metrics.
[ ] Test how much memory saved if each probe element is sparse?

[X] Test completely normal RC implementation for performance
[ ] Compress probe lists. Then can use bigger hashmap.
  - Could use a freelist to avoid duplication, or just double everything.
[ ] Need to fix LODs. How??
  [ ] Write every ray to two LODs and have perfect blending.
[ ] Merging optimization..
[ ] Convert to SH at C0?
[ ] Alternative final-level gathering?

[ ] Fix the ray start bias to actually use stuff from that paper.
[ ] Fix the shadow fix causing overdarkening / inconsistency between cascades.
[ ] Init cascades upon screenspace rather than rays and merge upwards.
[ ] Investigate light leaking near pillars without shadow fix

BUGS:
[ ] Convergence is just pretty damn slow currently. Make it so that lower cascades use higher cascade values if they don't have enough info?

Split into two caches. Also measure ray divergence and use that to stochastically reduce ray count.
*/

use std::f32::consts::TAU;
use std::path::PathBuf;
use std::sync::Arc;

use chandra::cache::{
    DirectionIndexer, DirectionIndexerComps, Fluence, HashCache, INDEX_SCALING, LoadedNode,
    LoadedNodeComps, Probe, ProbeComps, Radiance, RadianceElement, RadianceElementComps,
};
use chandra::table;
use chandra::utils::{corner_to_index, index_to_corner};
use chandra::world::TriangleWorld;
use glam::Mat3 as FMat3;
use image::{GenericImageView, ImageReader};
use keter::bindless::BINDLESS;
use keter::graph::profile::Profiler;
use keter::lang::functions::{
    block_id, sync_block, thread_id, warp_active_count_bits, warp_prefix_count_bits,
    warp_read_lane_at,
};
use keter::lang::types::shared::Shared;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter::rtx::SurfaceHit;
use keter_testbed::{App, KeyCode};
use parking_lot::Mutex;
use yesod::camera::{Camera, View};
use yesod::color::{inferno, plasma, reinhard};
use yesod::direction::{DirectionEncoder, SphericalEncoder, project_line};
use yesod::printer::global::flush_printer;
use yesod::rand::{pcg, pcg3d, pcg3df, pcgf, r2};
use yesod::shapes::intersect_sphere;
use yesod::utils::{dispatch0, dispatch0_async};
use yesod::{device_println, host};

const TERMINATE_DEPOSIT: u32 = 0b1;
// TODO: Probably should scale with distance

const TARGET_QUALITY_MULTIPLIER: f32 = 10.0;
const QUALITY_DECAY: f32 = 0.95;
const MAX_RAYS: u32 = 1 << 22;
const NODE_LIFETIME: u32 = 100;
const INCREASED_BASE_RESOLUTION: bool = false;

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
fn load_noise2_textures(names: impl Fn(u32) -> PathBuf) -> Tex3d<Vec2<u32>> {
    let storage = PixelStorage::Byte2;
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
        assert_eq!(image.color(), image::ColorType::Rgb8);

        data.extend(
            image
                .into_rgb8()
                .pixels()
                .map(|p| Vec2::new(p.0[0], p.0[1])),
        );
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
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [1920, 1080])
        .scale(2)
        .agx()
        .resize()
        .init();

    yesod::printer::global::set_capacity(1024 * 1024);

    // let sphere_noise =

    let hemi_noise = load_noise_textures(|i| {
        PathBuf::from(format!(
            "assets/noise/sphere/exp/sphere_coshemi_binomial3x3_exp0101_product_{i}.png",
        ))
    });
    let noise2 = load_noise2_textures(|i| PathBuf::from(format!("assets/noise2/{i}.png",)));

    /*let world = {
        let a = 6000.0;
        let b = 150.0;
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
    let radiance_buf = DEVICE.create_buffer::<Vec4<f32>>((app.size()[0] * app.size()[1]) as usize);
    let radiance_depth_buf = DEVICE.create_buffer::<f32>((app.size()[0] * app.size()[1]) as usize);
    // let cache = HashCache::new(16.0, 100000.0, 4, 1 << 18, 6);
    let cache = HashCache::new(10.0, 100000.0, 4, 1 << 18, 4);
    let cache_radiance = DEVICE.create_buffer::<Vec3<f32>>(cache.weights.len() / 4);
    let cache_base_radiance = DEVICE.create_buffer::<Vec3<f32>>(cache.base_probes as usize * 32);

    let base_probe_num_rays =
        DEVICE.create_buffer_from_fn::<u32>(cache.base_probes as usize, |_| 0_u32);
    let node_quality =
        DEVICE.create_buffer_from_fn::<f32>(cache.total_probes() as usize, |_| 0.0_f32);
    let node_lifetime =
        DEVICE.create_buffer_from_fn::<u32>(cache.total_probes() as usize, |_| 0_u32);
    let traced_rays = DEVICE.create_buffer::<Ray>(MAX_RAYS as usize);
    let next_traced_ray = Singleton::<u32>::new(0_u32);

    let cache_memory = cache.weighted_fluence.size_bytes()
        + cache.weights.size_bytes()
        + cache.table.memory_size();
    println!("Cache memory: {} MB", cache_memory / 1024 / 1024);

    let build_freelists = DEVICE.create_kernel::<fn(bool)>(&track!(|total| {
        let data_index = dispatch_id().x;
        cache.table.build_freelists(data_index, |key, cascade| {
            let lifetime = node_lifetime.read(data_index).var();
            let quality = node_quality.read(data_index).var();
            let quality = quality * QUALITY_DECAY;
            let invalid = lifetime == 0 || total;
            *lifetime -= 1_u32;
            if invalid {
                // device_println!("Deleting: {}", host!(data_index));
                let node = cache.load_index(cache.decode(key, cascade), data_index);
                let encoder = cache.get_encoder(cascade);

                for index in 0_u32.expr()..encoder.linear_size() {
                    cache
                        .weighted_fluence
                        .write(node.data_offset + index, Vec4::splat(0.0));
                    cache.weights.write(node.data_offset + index, 0.0);
                }
                node_quality.write(data_index, 0.0);
            } else {
                node_lifetime.write(data_index, **lifetime);
                node_quality.write(data_index, quality);
            }
            invalid
        });
    }));

    let get_fluence = track!(|start: Expr<Vec3<f32>>,
                              dir: Expr<Vec3<f32>>,
                              hit: Expr<SurfaceHit>| {
        if hit.valid() {
            let bounce_pos = start + dir * hit.committed_ray_t;
            let normal = world.get_normal_flat(hit);
            let weight = normal.dot(sunlight_dir());
            let radiance = Vec3::splat(0.0_f32).var();

            if weight > 0.0 {
                *radiance += world
                    .trace_segment(
                        bounce_pos,
                        sunlight_dir().expr(),
                        Vec2::expr(0.01, 100000.0),
                    )
                    .valid()
                    .select(Vec3::splat_expr(0.0_f32), Vec3::splat_expr(10.0_f32))
                    * weight;
            }
            if hit.inst == 1 {
                *radiance += Vec3::expr(10.0, 0.0, 0.0);
            }
            radiance.extend(0.0)
        } else {
            Vec3::splat_expr(0.0_f32).extend(1.0)
        }
    });

    let get_quality_weight = track!(|node: Expr<LoadedNode>| {
        node_quality.read(node.index)
        // let quality = node_quality.read(node.index);
        // let encoder = cache.get_encoder(node.probe.cascade);
        // let ratio = quality / encoder.linear_size().cast_f32();
        // keter::min(1.0, ratio * TARGET_QUALITY_MULTIPLIER)
    });

    #[derive(Clone, Copy)]
    enum MergeLevel {
        First,
        Last,
        Middle(Expr<u32>),
    }

    // Note: can additionally load all lower probes at once as well for increased efficiency.
    let merge_cache_base = track!(|merge_level: MergeLevel| {
        set_block_size([1, 32, 1]);
        let cascade = match merge_level {
            MergeLevel::First => 0.expr(),
            MergeLevel::Last => (cache.num_cascades - 1).expr(),
            MergeLevel::Middle(l) => l,
        };
        let node = cache.get_occupied(cascade, dispatch_id().x);
        if node.invalid() {
            return;
        }

        let probe = node.probe;

        let n = 0_u32.var();
        let next_offsets = Shared::<u32>::new(8);
        let next_weights = Shared::<f32>::new(8);

        if !matches!(merge_level, MergeLevel::Last) {
            if thread_id().y < 8 {
                let i = thread_id().y;
                let (next_probe, weight) = probe.next_bilinear(i);

                let next = cache.get(next_probe);
                if next.loaded() {
                    let index = warp_prefix_count_bits(true.expr());

                    next_offsets.write(index, next.data_offset);
                    next_weights.write(index, weight * get_quality_weight(next));
                }
                *n = warp_active_count_bits(next.loaded());
            }
            sync_block();
        }
        let n = warp_read_lane_at(**n, 0);

        let index = dispatch_id().y;
        let total_radiance = Vec4::splat(0.0_f32).var();
        if !matches!(merge_level, MergeLevel::Last) {
            for i in 0.expr()..n {
                let radiance = cache_radiance.read(next_offsets.read(i) / 4 + index);
                // necessary?
                if radiance.is_finite().all() {
                    *total_radiance += next_weights.read(i) * radiance.extend(1.0);
                }
            }
        }
        let data_index = node.data_offset + index;

        let far = total_radiance.xyz() / keter::max(total_radiance.w, 0.00001);

        let weight = cache.weights.read(data_index);
        let fluence = cache.weighted_fluence.read(data_index) / weight;

        let radiance = if weight < 0.0001 {
            Vec4::splat_expr(0.0)
        } else {
            over(fluence, far).extend(1.0)
        };

        if INCREASED_BASE_RESOLUTION && matches!(merge_level, MergeLevel::First) {
            cache_base_radiance.write(data_index, radiance.xyz() / radiance.w);
        } else {
            let adj_radiance = warp_read_lane_at(radiance, (index + 1) % 32);
            if index % 2 == 0 {
                let radiance = radiance + adj_radiance;
                let adj_radiance = warp_read_lane_at(radiance, (index + 2) % 32);
                if index % 4 == 0 {
                    let radiance = radiance + adj_radiance;
                    cache_radiance.write(data_index / 4, radiance.xyz() / radiance.w);
                }
            }
        }
    });

    let merge_first_cache = DEVICE.create_kernel::<fn()>(&|| {
        merge_cache_base(MergeLevel::First);
    });
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

    let init_c0_probe_kernel = DEVICE.create_kernel::<fn(View)>(&track!(|view| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let depth = depth_buf.read(pixel);
        if depth == 0.0 {
            return;
        }
        let world_pos = view.pos + ray_dir * depth;

        let lod = cache.estimate_lod(world_pos);
        let probe = cache.quantize(world_pos, lod, 0.expr());
        cache.insert(probe, |_| {});
    }));
    let init_probe_kernel = DEVICE.create_kernel::<fn(u32)>(&track!(|cascade| {
        let index = dispatch_id().x + cache.get_index_offset(cascade);
        let key = cache.table.data_key.read(index);
        if key == table::INVALID_KEY {
            return;
        }
        let probe = cache.decode(key, cascade);
        let probe = probe.next_nearest();
        cache.insert(probe, |_| {});
    }));

    let compute_num_rays_kernel = DEVICE.create_kernel::<fn(View, u32)>(&track!(|view, _tick| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let depth = depth_buf.read(pixel);
        if depth == 0.0 {
            return;
        }
        let world_pos = view.pos + ray_dir * depth;

        let lod = cache.estimate_lod(world_pos);

        let probe = cache.quantize(world_pos, lod, 0.expr());

        let node = cache.get(probe);

        // TODO: This shouldn't happen!
        if node.invalid() {
            return;
        }

        base_probe_num_rays
            .atomic_ref(node.index)
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

        let lod = cache.estimate_lod(world_pos);

        let probe = cache.quantize(world_pos, lod, 0.expr());

        let node = cache.get(probe);

        if node.invalid() {
            return;
        }

        /*let bounce_ray = noise2
            .read(pixel.extend(tick) % Vec3::new(noise2.width(), noise2.height(), noise2.depth()))
            / 2;
        let bounce_ray = bounce_ray.cast_f32() + 0.5;
        let bounce_ray = bounce_ray / 128.0;
        let bounce_ray = bounce_ray * Vec2::new(0.5, 1.0); // X-axis is doubled because only hemisphere / mirroring.
        let bounce_ray = SphericalEncoder::decode(bounce_ray);

        let bounce_ray = bounce_ray * bounce_ray.dot(normal).signum();
        */
        let bounce_ray = rotate_hemi(
            normal,
            as_normal(hemi_noise.read(
                pixel.extend(tick)
                    % Vec3::new(hemi_noise.width(), hemi_noise.height(), hemi_noise.depth()),
            )),
        );

        let num_rays = base_probe_num_rays.read(node.index);
        let target_rays = keter::max(
            cache.get_encoder(0.expr()).linear_size().cast_f32() * TARGET_QUALITY_MULTIPLIER
                - node_quality.read(node.index),
            0.0,
        );
        // let extinction = target_rays / keter::max(num_rays, 1).cast_f32();

        // if pcg3df((pixel + Vec2::new(3528, 9306)).extend(tick)).x < extinction {
        let ray = Ray::from_comps_expr(RayComps {
            dir: bounce_ray,
            start: world_pos,
            base_probe: probe,
        });
        let ray_index = next_traced_ray.atomic().fetch_add(1_u32.expr());
        if ray_index < traced_rays.len_expr_u32() {
            traced_rays.write(ray_index, ray);
        }
        // }
    }));

    let compute_radiance_kernel = DEVICE.create_kernel_with_options::<fn(u32, u32, f32)>(
        KernelBuildOptions {
            name: Some("compute_radiance_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|_tick, flags, c0_len| {
            set_block_size([64, 1, 1]);

            if dispatch_id().x >= next_traced_ray.read() {
                return;
            }

            let ray = traced_rays.read(dispatch_id().x);
            let ray_dir = ray.dir.normalize();
            let world_pos = ray.start;

            let bounce_ray = ray_dir;

            let probe = ray.base_probe;

            let bounce_hit = world.trace_segment(
                world_pos,
                bounce_ray,
                Vec2::expr(0.01, 100000.0), // * (1_u32 << probe.position.w).cast_f32(),
            );

            let (hit_level, fluence) = if bounce_hit.valid() {
                let cascade = keter::max(
                    (bounce_hit.committed_ray_t
                        / (c0_len * cache.base_grid_size * (1_u32 << probe.position.w).cast_f32()))
                    .log2(),
                    0.0,
                )
                .ceil()
                .cast_u32();
                (
                    keter::min(cascade, cache.num_cascades.expr() - 1),
                    get_fluence(world_pos, bounce_ray, bounce_hit),
                )
            } else {
                (cache.num_cascades.expr() - 1, Vec4::splat_expr(0.0))
            };

            let deposit = |cascade, fluence| {
                let node = cache.get(probe.shift_level(cascade));
                if node.loaded() {
                    let encoder = cache.get_encoder(cascade);
                    cache.deposit(
                        node.data_offset + encoder.dir_to_linear(bounce_ray),
                        fluence,
                    );
                    node_quality.atomic_ref(node.index).fetch_add(1.0);
                    node_lifetime
                        .atomic_ref(node.index)
                        .fetch_max(NODE_LIFETIME.expr());
                } // Otherwise should be impossible
            };

            if (flags & TERMINATE_DEPOSIT) == 0 {
                for level in 0..cache.num_cascades {
                    deposit(
                        level,
                        if level < hit_level {
                            Vec3::splat_expr(0.0_f32).extend(1.0)
                        } else {
                            fluence
                        },
                    );
                }
            } else {
                // TODO: If we invert transmittance then the lower-deposits are just count increments.
                // note: since lifetime is deposited, short rays can cause higher-cascades to vanish
                for level in 0.expr()..hit_level {
                    deposit(level, Vec3::splat_expr(0.0_f32).extend(1.0));
                }
                deposit(hit_level, fluence);
            }
        }),
    );

    let compute_screen_kernel = DEVICE.create_kernel::<fn(View, f32, u32, u32)>(&track!(
        |view, blend_radius, cascade, draw_mode| {
            set_block_size([1, 32, 1]);
            let node = cache.get_occupied(cascade, dispatch_id().x);
            if node.invalid() {
                return;
            }
            let probe = node.probe;
            let encoder = DirectionIndexer::from_comps_expr(DirectionIndexerComps {
                resolution: if INCREASED_BASE_RESOLUTION && cascade == 0 {
                    2_u32.expr()
                } else {
                    (cache.base_angular_resolution / 2) * (1 << cascade)
                },
            });
            assert_eq!(cache.base_angular_resolution, 4);

            // let quality = node_quality.read(node.index);

            // 8 if !INCREASED_BASE_RESOLUTION
            let radiance_data = Shared::<Vec3<f32>>::new(32);
            if INCREASED_BASE_RESOLUTION && cascade == 0 {
                let v = cache_base_radiance.read(node.data_offset + dispatch_id().y);
                radiance_data.write(thread_id().y, v);
            } else if cascade != 0 || dispatch_id().y < 8 {
                let v = cache_radiance.read(node.data_offset / 4 + dispatch_id().y);
                radiance_data.write(thread_id().y, v);
            }
            sync_block();

            let screen_min = Vec2::from(app.size()).expr().cast_f32().var();
            let screen_max = Vec2::splat(0.0_f32).var();

            let spacing = cache.base_grid_size
                * (1 << (probe.position.w + cascade)).cast_f32()
                * blend_radius;
            let probe_pos = cache.dequantize(probe);

            /*if dispatch_id().y < 8 {
                let i = dispatch_id().y;

                let coord = probe_pos
                    + spacing
                        * (Vec3::<bool>::expr((i & 1) != 0, (i & 2) != 0, (i & 4) != 0)
                            .select(Vec3::splat_expr(-1.0_f32), Vec3::splat_expr(1.0_f32)));
                let screen_pos = view.project3(coord);
                if screen_pos.z <= 0.0 {
                    return;
                }
                let screen_pos = screen_pos.xy();

                screen_min = warp_active_min(screen_pos);
                screen_max = warp_active_max(screen_pos);
            }*/

            // TODO: May go schizo if the probe is behind the camera...
            for i in 0..8 {
                let coord = probe_pos
                    + spacing
                        * (Vec3::<bool>::expr((i & 1) != 0, (i & 2) != 0, (i & 4) != 0)
                            .select(Vec3::splat_expr(-1.0_f32), Vec3::splat_expr(1.0_f32)));
                let screen_pos = view.project3(coord);
                if screen_pos.z <= 0.0 {
                    return;
                }
                let screen_pos = screen_pos.xy();
                *screen_min = keter::min(**screen_min, screen_pos);
                *screen_max = keter::max(**screen_max, screen_pos);
            }
            let screen_min = keter::max(0.0_f32, **screen_min).floor().cast_u32();
            let screen_max = keter::min(Vec2::from(app.size()).expr().cast_f32(), **screen_max)
                .ceil()
                .cast_u32();
            let size = screen_max - screen_min;
            let area = size.reduce_prod();
            let i = dispatch_id().y.var();
            while i < area {
                let pixel = screen_min + Vec2::expr(i % size.x, i / size.x);
                *i += 32;

                let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

                if draw_mode == 0 {
                    let depth = depth_buf.read(pixel);
                    if depth == 0.0 {
                        continue;
                    }
                    let world_pos = view.pos + ray_dir * depth;
                    let normal = SphericalEncoder::decode(normal_buf.read(pixel));
                    let weight = keter::max(1.0 - (world_pos - probe_pos).abs() / spacing, 0.0)
                        .reduce_prod()
                        * (1 << (2 * probe.position.w)).cast_f32().recip()
                        * get_quality_weight(node);
                    if weight == 0.0 {
                        continue;
                    }
                    // let (factor, cascade) = cache.projection().interpolation_factor_lod(world_pos);
                    // let factor = if cascade == probe.position.w {
                    //     factor
                    // } else {
                    //     1.0 - factor
                    // };
                    // let weight = weight * factor;

                    let total_radiance = Vec4::splat(0.0_f32).var();
                    for j in 0_u32.expr()..if cascade == 0 && !INCREASED_BASE_RESOLUTION {
                        8_u32.expr()
                    } else {
                        32_u32.expr()
                    } {
                        let dir = encoder.linear_to_dir(j + block_id().y * 32);
                        let weight = dir.dot(normal);
                        if weight > 0.0 {
                            let radiance = radiance_data.read(j);
                            if radiance.is_finite().all() {
                                *total_radiance += weight * radiance.extend(1.0);
                            }
                        }
                    } // TODO: Figure out the biasing properly? Just directly bias based on amount of missing data?
                    if total_radiance.w < 0.0001 {
                        continue;
                    }

                    let total_radiance = total_radiance.xyz() / total_radiance.w;
                    // let total_radiance = pcg3df(probe.position.xyz());
                    let deposit_radiance = total_radiance.extend(1.0) * weight;
                    let dest = radiance_buf.atomic_ref(pixel.y * app.size()[0] + pixel.x);
                    dest.x.fetch_add(deposit_radiance.x);
                    dest.y.fetch_add(deposit_radiance.y);
                    dest.z.fetch_add(deposit_radiance.z);
                    dest.w.fetch_add(deposit_radiance.w);
                } else {
                    let (time, intersected) =
                        intersect_sphere(view.pos - probe_pos, ray_dir, spacing);
                    if !intersected {
                        continue;
                    }
                    let depth = time.x;
                    let intersect_pos = view.pos + ray_dir * depth - probe_pos;

                    let intersect_dir = intersect_pos.normalize();
                    let index = encoder.dir_to_linear(intersect_dir) - block_id().y * 32;
                    if index >= 32 {
                        continue;
                    }

                    if draw_mode == 2 {
                        // depth prepass
                        radiance_depth_buf
                            .atomic_ref(pixel.y * app.size()[0] + pixel.x)
                            .fetch_min(depth);
                    } else if draw_mode == 1 {
                        let actual_depth =
                            radiance_depth_buf.read(pixel.y * app.size()[0] + pixel.x);
                        if depth == actual_depth {
                            let radiance = radiance_data.read(index);
                            if radiance.is_finite().all() {
                                let deposit_radiance = radiance.extend(1.0);
                                let dest =
                                    radiance_buf.atomic_ref(pixel.y * app.size()[0] + pixel.x);
                                dest.x.fetch_add(deposit_radiance.x);
                                dest.y.fetch_add(deposit_radiance.y);
                                dest.z.fetch_add(deposit_radiance.z);
                                dest.w.fetch_add(deposit_radiance.w);
                            }
                        }
                    }
                }
            }
        }
    ));

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32, u32, u32, u32, Vec2<u32>)>(
        KernelBuildOptions {
            name: Some("draw_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick, _flags, mode, cascade, cursor_pixel| {
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

            let lod = cache.estimate_lod(world_pos);
            let probe = cache.quantize(world_pos, lod, cascade);

            let sunlight = world
                .trace_segment(world_pos, sunlight_dir().expr(), Vec2::expr(0.01, 100000.0))
                .valid()
                .select(Vec3::splat_expr(0.0_f32), Vec3::splat_expr(10.0_f32));

            let color = if mode == 0 {
                let r = radiance_buf.read(pixel.y * app.size()[0] + pixel.x);
                let result = r.xyz() / r.w;
                result + sunlight * keter::max(normal.dot(sunlight_dir()), 0.0)
            } else if mode == 1 {
                if cache.contains(probe) {
                    let target_probe = cache.decode(cache.encode(probe), cascade);
                    if (target_probe.position != probe.position).any()
                        || (cache.quantize(world_pos, lod, cascade + 1).position
                            != probe.shift_level(1.expr()).position)
                            .any()
                    {
                        Vec3::splat_expr(1.0)
                    } else {
                        inferno((probe.position.w.cast_f32() + 0.5) / 9.0)
                            * pcg3df(probe.position.xyz()).x
                    }
                } else {
                    Vec3::splat_expr(0.0)
                }
            } else if mode == 2 && cascade < cache.num_cascades - 1 {
                let num_next = 0_f32.var();
                for i in 0..8u32 {
                    let (next_probe, weight) = probe.next_bilinear(i);

                    let next = cache.get(next_probe);
                    *num_next += weight * next.loaded().select(1.0_f32.expr(), 0.0_f32.expr());
                }
                inferno(**num_next)
                /*let inner = dispatch_id().xy() % Vec2::expr(8, 4);
                let outer = dispatch_id().xy() / Vec2::expr(8, 4);
                let i = 32 * (outer.y * (dispatch_size().x / 4) + outer.x) + inner.y * 8 + inner.x;
                if i < cache.table.data_capacity {
                    let occupied = cache.table.data_key.read(i) != table::INVALID_KEY;
                    if occupied {
                        Vec3::splat_expr(1.0)
                    } else {
                        Vec3::splat_expr(0.0)
                    }
                } else {
                    Vec3::expr(0.0, 0.0, 1.0)
                }*/
            } else if mode == 3 {
                /*let node = cache.get(probe);
                if node.loaded() {
                    let encoder = DirectionIndexer::from_comps_expr(DirectionIndexerComps {
                        resolution: (cache.base_angular_resolution / 2) << cascade,
                    });
                    let total_radiance = Vec4::splat(0.0_f32).var();
                    for j in 0_u32.expr()..encoder.linear_size() {
                        let dir = encoder.linear_to_dir(j);
                        let weight = dir.dot(normal);
                        if weight > 0.0 {
                            let radiance = cache_radiance.read(node.data_offset / 4 + j);
                            if radiance.is_finite().all() {
                                *total_radiance += weight * radiance.extend(1.0);
                            }
                        }
                    }
                    total_radiance.xyz() / total_radiance.w
                } else {
                    Vec3::expr(1.0, 0.0, 0.0)
                }*/
                pcg3df(probe.position.xyz())
            } else if mode == 4 {
                // let offset = pcg3d(Vec3::expr(tick / hemi_noise.depth(), 392, 132)).xy()
                //     % Vec2::new(hemi_noise.width(), hemi_noise.height());
                // let bounce_ray = as_normal(hemi_noise.read(
                //     (pixel + offset).extend(tick)
                //         % Vec3::new(hemi_noise.width(), hemi_noise.height(), hemi_noise.depth()),
                // ));
                // let bounce_ray = rotate_hemi(normal, bounce_ray);

                let bounce_ray = project_line((r2(tick) + pcg3df(pixel.extend(968)).xy()).fract());
                let bounce_ray = bounce_ray * bounce_ray.dot(normal).signum();

                // let bounce_ray = sample_bounce(normal, pcg3df(pixel.extend(tick)).xy());

                let bounce_hit =
                    world.trace_segment(world_pos, bounce_ray, Vec2::expr(0.1_f32, 10000.0));
                let radiance = get_fluence(world_pos, bounce_ray, bounce_hit).xyz();
                let value = accum_buf.read(pixel) + radiance.extend(1.0);
                accum_buf.write(pixel, value);
                value.xyz() / value.w + sunlight * keter::max(normal.dot(sunlight_dir()), 0.0)
            } else if mode == 5 {
                let node = cache.get(probe);
                if node.loaded() {
                    let quality = node_quality.read(node.index);
                    let quality_ratio = quality
                        / (TARGET_QUALITY_MULTIPLIER
                            * cache.get_encoder(cascade).linear_size().cast_f32());
                    inferno(quality_ratio)
                } else {
                    Vec3::splat_expr(0.0)
                }
                // inferno(1.0 - reinhard((cache.dequantize(probe) - world_pos).length()))
            } else if mode == 6 {
                let hit = world.trace(view.pos, ray_dir);
                if (cursor_pixel == pixel).all() {
                    let inst = hit.inst;
                    device_println!("Hit at {}", host!(inst));
                }
                pcg3df(Vec3::expr(hit.inst, 37, 926))
            } else if mode == 7 {
                let cost = cache.cost(probe);
                if cost != u32::MAX {
                    if cost > 8 {
                        Vec3::expr(1.0, 1.0, 1.0)
                    } else {
                        plasma(cost.cast_f32() / 9.0)
                    }
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
        // yaw: 6.857436,
        // pitch: 0.12587574,
        // pos: glam::Vec3::new(308.5938, -573.01685, 240.64691),
        yaw: 6.437008,
        pitch: 0.47492924,
        pos: glam::Vec3::new(-94.45221, -937.2442, 613.4341),
        fov: 1.15,
    };
    let mut running = true;
    let mut mode = 0;
    let mut cascade = 0;
    let mut radiance_flags = TERMINATE_DEPOSIT;
    let mut c0_len_mode = 0;
    let mut blend_radius_mode = 0;
    let mut num_probes = vec![0; cache.num_cascades as usize];
    let mut max_probes = vec![0; cache.num_cascades as usize];

    let mut profiler = Profiler::new();

    let nc = vec![0; cache.num_cascades as usize];

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
            mode = (mode + 1) % 8;
            println!("Drawing mode {mode}");
            let tonemap = matches!(mode, 0 | 4);
            rt.tonemap(tonemap);
        }
        if rt.key_pressed(KeyCode::KeyL) {
            cascade = (cascade + 1) % cache.num_cascades;
            println!("Level set {cascade}");
        }
        if rt.key_pressed(KeyCode::KeyK) {
            c0_len_mode = (c0_len_mode + 1) % 3;
        }
        if rt.key_pressed(KeyCode::KeyO) {
            running = !running;
            println!("Running: {running}");
        }
        if rt.key_pressed(KeyCode::KeyT) {
            radiance_flags ^= TERMINATE_DEPOSIT;
        }
        if rt.key_pressed(KeyCode::KeyB) {
            blend_radius_mode = (blend_radius_mode + 1) % 3;
        }
        let mut timings = vec![];
        let mut commands = vec![];

        let mut reset_table = |commands: &mut Vec<_>, total: bool| {
            // reset table
            {
                let table = &cache.table;
                let clear_entries = dispatch0_async(
                    track!(|| {
                        table.entries.write(dispatch_id().x, table::Entry::EMPTY);
                    }),
                    [table.entries.len() as u32, 1, 1],
                );
                commands.push(
                    (
                        cache
                            .table
                            .table_data_start
                            .copy_to_buffer_async(&table.freelist_index),
                        cache
                            .table
                            .table_data_start
                            .copy_to_buffer_async(&table.last_occupied_index),
                        table.total_occupied.copy_from_async(&nc),
                        clear_entries,
                        build_freelists.dispatch_async([table.freelist.len() as u32, 1, 1], &total),
                        cache
                            .table
                            .table_data_start
                            .copy_to_buffer_async(&table.freelist_index),
                    )
                        .chain(),
                );
            }

            // find probe counts
            let total_occupied = Arc::new(Mutex::new(vec![0_u32; cache.num_cascades as usize]));
            commands.push(cache.table.total_occupied.copy_to_shared(&total_occupied));
            let last_occupied_index =
                Arc::new(Mutex::new(vec![0_u32; cache.num_cascades as usize]));
            commands.push(
                cache
                    .table
                    .last_occupied_index
                    .copy_to_shared(&last_occupied_index),
            );
            timings.extend(std::mem::take(commands).chain().execute_timed());
            let last_occupied_index = last_occupied_index.lock();

            num_probes = total_occupied.lock().to_vec();
            max_probes = last_occupied_index
                .iter()
                .enumerate()
                .map(|(c, &x)| x - cache.get_index_offset_host(c as u32) + 1)
                .collect();
        };
        if rt.tick == 0 {
            reset_table(&mut commands, false);
        }

        if rt.key_down(KeyCode::Backslash) {
            println!("Resetting cache");

            let mut commands = vec![];

            reset_table(&mut commands, true);
        }

        commands.push(
            compute_gbuffer_kernel
                .dispatch_async(rt.dispatch_size(), &camera.view())
                .debug("Compute GBuffer"),
        );

        let running = running || rt.key_pressed(KeyCode::Period);
        if running {
            // if mode == 3 {
            //     println!("Num rays traced: {}", next_traced_ray.read_blocking());
            // }
            commands.push(cache.set_view(camera.view()));

            // add c0 probes
            commands.push(
                init_c0_probe_kernel
                    .dispatch_async(rt.dispatch_size(), &camera.view())
                    .debug("Init Probes Cascade 0"),
            );
            for c in 0..(cache.num_cascades - 1) {
                commands.push(
                    init_probe_kernel
                        .dispatch_async([cache.max_probes_at_cascade(c), 1, 1], &c)
                        .debug(format!("Init Probes Cascade {}", c + 1)),
                );
            }

            /*
            for l in (0..cache.num_levels).rev() {
                let dispatch_size = [cache.num_nodes_at_cascade(l), 1 << l, 1];
                commands.push(
                    decay_cache
                        .dispatch_async(dispatch_size, &l)
                        .debug("Decay Cache"),
                );
            }
            */

            // Reset ray counts
            commands.push(
                dispatch0_async(
                    track!(|| {
                        base_probe_num_rays.write(dispatch_id().x, 0_u32);
                    }),
                    [base_probe_num_rays.len() as u32, 1, 1],
                )
                .debug("Reset Base Probe Num Rays"),
            );
            commands.push(
                dispatch0_async(
                    track!(|| {
                        next_traced_ray.write(0_u32);
                    }),
                    [1, 1, 1],
                )
                .debug("Reset Next Traced Ray"),
            );

            commands.push(
                compute_num_rays_kernel
                    .dispatch_async(rt.dispatch_size(), &camera.view(), &rt.tick)
                    .debug("Compute Radiance Num Rays"),
            );
            commands.push(
                compute_rays_kernel
                    .dispatch_async(rt.dispatch_size(), &camera.view(), &rt.tick)
                    .debug("Compute Radiance Rays"),
            );
            commands.push(
                compute_radiance_kernel
                    .dispatch_async(
                        [MAX_RAYS, 1, 1],
                        &rt.tick,
                        &radiance_flags,
                        &match c0_len_mode {
                            0 => 4.0,
                            1 => 12.0,
                            2 => 999.0,
                            _ => unreachable!(),
                        },
                    )
                    .debug("Compute Radiance Trace"),
            );

            reset_table(&mut commands, false);

            for l in (0..cache.num_cascades).rev() {
                let dispatch_size = [max_probes[l as usize], 32 << (l * 2), 1];
                commands.push(
                    (match l {
                        0 => merge_first_cache.dispatch_async(dispatch_size),
                        l if l == cache.num_cascades - 1 => {
                            merge_last_cache.dispatch_async(dispatch_size)
                        }
                        l => merge_cache.dispatch_async(dispatch_size, &l),
                    })
                    .debug(format!("Merge Cache Level {l}")),
                );
            }
        }

        if camera_updated {
            commands.push(
                dispatch0_async(
                    track!(|| {
                        let pixel = dispatch_id().xy();
                        accum_buf.write(pixel, Vec4::splat_expr(0.0_f32));
                    }),
                    [accum_buf.width(), accum_buf.height(), 1],
                )
                .debug("Reset Accumulation"),
            );
        }

        commands.push(
            dispatch0_async(
                track!(|| {
                    radiance_buf.write(dispatch_id().x, Vec4::splat_expr(0.0_f32));
                    radiance_depth_buf.write(dispatch_id().x, 100000.0_f32);
                }),
                [radiance_buf.len() as u32, 1, 1],
            )
            .debug("Reset Radiance Buffer"),
        );
        // TODO: Freelist reset doesn't properly count number of probes.
        // println!("Num probes: {num_probes:?}");

        if mode == 0 {
            let blend_radius = match blend_radius_mode {
                0 => 1.0,
                1 => 0.5,
                2 => 0.4,
                _ => unreachable!(),
            };
            let draw_circle = blend_radius_mode == 2;
            let dispatch_size = 32 << (cascade.saturating_sub(1) * 2);
            if draw_circle {
                commands.push(
                    compute_screen_kernel
                        .dispatch_async(
                            [max_probes[cascade as usize], dispatch_size, 1],
                            &camera.view(),
                            &blend_radius,
                            &cascade,
                            &2,
                        )
                        .debug("Compute Screen (Circle Depth)"),
                );
            }
            commands.push(
                compute_screen_kernel
                    .dispatch_async(
                        [max_probes[cascade as usize], dispatch_size, 1],
                        &camera.view(),
                        &blend_radius,
                        &cascade,
                        &if draw_circle { 1 } else { 0 },
                    )
                    .debug("Compute Screen"),
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
                    &cascade,
                    &rt.cursor_position.map(|x| x as u32),
                )
                .debug("Draw"),
        );

        timings.extend(commands.chain().execute_timed());
        profiler.record(timings);

        if profiler.time() > 1000.0 {
            profiler.print(
                &[
                    "Compute GBuffer",
                    "Init Probes",
                    "Decay Cache",
                    "Compute Radiance",
                    "Merge",
                    "Add Primary Cache",
                    "Compute Screen",
                    "Draw",
                ],
                true,
            );
            profiler.reset();
            println!("Num probes: {num_probes:?}");
            println!("Max probe: {max_probes:?}");
            println!(
                "Limit: {:?}",
                (0..cache.num_cascades)
                    .map(|n| cache.max_probes_at_cascade(n))
                    .collect::<Vec<_>>()
            );
        }

        flush_printer();
    });
}
