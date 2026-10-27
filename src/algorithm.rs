use std::ops::AddAssign;

use image::Rgb32FImage;
use keter::bindless::{BINDLESS, Tex2dHandle};
use keter::graph::NodeConfigs;
use keter::lang::types::shared::Shared;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::camera::View;
use yesod::device_println;
use yesod::direction::{DirectionEncoder, EquirectangularEncoder, SphericalEncoder};
use yesod::rand::pcg3df;
use yesod::utils::{dispatch0, dispatch1, memo};
use crate::utils::offset_ray;

use crate::algorithm::rc::RcData;
use crate::cache::{Fluence, Probe, Radiance};

#[cfg(not(feature = "voxel"))]
type World = crate::world::TriangleWorld;
#[cfg(not(feature = "voxel"))]
type Hit = keter::rtx::SurfaceHit;
#[cfg(feature = "voxel")]
type World = crate::world::VoxelWorld;
#[cfg(feature = "voxel")]
type Hit = keter::rtx::CommittedHit;

pub struct WorldData {
    pub world: World,
    pub envmap: Rgb32FImage,
    pub envmap_scale: f32,
    pub sun_dir: Vec3<f32>,
    pub sun_color: Vec3<f32>,
    pub custom_emitters: Vec<(u32, Vec3<f32>)>,
}

/// Lighting shared by all ray tracing kernels, updated without recompiling them.
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Lighting {
    pub sun_dir: Vec3<f32>,
    pub sun_color: Vec3<f32>,
    pub environment_scale: f32,
    pub emission_scale: f32,
}

/// Finite-range point light. RGB is intensity in the renderer's centimetre
/// coordinate system; range culls lights before issuing a visibility ray.
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct PointLight {
    pub position_range: Vec4<f32>,
    pub intensity: Vec4<f32>,
}

pub struct CoreData {
    pub world: World,
    pub view: Singleton<View>,
    pub depth: Tex2dHandle<f32>,
    pub normal: Tex2d<Vec2<f32>>,
    pub hit: Buffer<Hit>,

    pub lighting: Singleton<Lighting>,
    pub environment: Tex2dHandle<Vec4<f32>>,
    constant_environment: Option<Vec3<f32>>,
    pub custom_emitters: Vec<(u32, Vec3<f32>)>,
    point_lights: Option<Buffer<PointLight>>,
    light_grid: Option<light_grid::LightGrid>,
    gbuffer_kernel: std::sync::OnceLock<Kernel<fn()>>,
}
impl CoreData {
    pub fn new(
        [width, height]: [u32; 2],
        WorldData {
            world,
            envmap,
            envmap_scale,
            sun_dir,
            sun_color,
            custom_emitters,
        }: WorldData,
    ) -> Self {
        let depth = DEVICE.create_tex2d::<f32>(PixelStorage::Float1, width, height, 1);
        let depth = BINDLESS.push_tex2d(
            depth,
            Sampler {
                filter: SamplerFilter::LinearLinear,
                address: SamplerAddress::Edge,
            },
        );
        let environment =
            DEVICE.create_tex2d(PixelStorage::Float4, envmap.width(), envmap.height(), 1);
        let constant_environment = envmap.pixels().next().and_then(|first| {
            envmap.pixels().all(|p| p == first).then(|| Vec3::new(first[0], first[1], first[2]))
        });
        let envmap = envmap
            .pixels()
            .map(|p| Vec4::new(p[0], p[1], p[2], 1.0))
            .collect::<Vec<_>>();
        environment.copy_from_vec(envmap).execute_blocking();
        let environment = BINDLESS.push_tex2d(
            environment,
            Sampler {
                filter: SamplerFilter::LinearLinear,
                address: SamplerAddress::Repeat,
            },
        );
        BINDLESS.update();
        // is the compression worth it?
        Self {
            world,
            view: Singleton::new_uninit(),
            depth,
            normal: DEVICE.create_tex2d::<Vec2<f32>>(PixelStorage::Short2, width, height, 1),
            hit: DEVICE.create_buffer::<Hit>(width as usize * height as usize),
            lighting: Singleton::new(Lighting {
                sun_dir,
                sun_color,
                environment_scale: envmap_scale,
                emission_scale: 1.0,
            }),
            environment,
            constant_environment,
            custom_emitters,
            point_lights: None,
            light_grid: None,
            gbuffer_kernel: std::sync::OnceLock::new(),
        }
    }
    pub fn set_point_lights(&mut self, lights: &[PointLight]) {
        self.set_point_lights_grid(lights, true);
    }
    pub fn set_point_lights_grid(&mut self, lights: &[PointLight], use_grid: bool) {
        assert!(!lights.is_empty());
        self.point_lights = Some(DEVICE.create_buffer_from_slice(lights));
        self.light_grid = (use_grid && (8..=32).contains(&lights.len())).then(|| light_grid::LightGrid::new(lights));
    }
    pub fn update_point_lights(&self, lights: Vec<PointLight>) -> NodeConfigs<'static> {
        let buffer = self.point_lights.as_ref().expect("Point lights were not configured");
        assert_eq!(buffer.len(), lights.len());
        if let Some(grid) = &self.light_grid {
            (grid.update(&lights), buffer.copy_from_vec(lights)).chain()
        } else { buffer.copy_from_vec(lights) }
    }
    pub fn screen_size(&self) -> Vec2<u32> {
        Vec2::new(self.normal.width(), self.normal.height())
    }
    pub fn screen_dispatch_size(&self) -> [u32; 3] {
        [self.normal.width(), self.normal.height(), 1]
    }
    pub fn num_pixels(&self) -> u32 {
        self.normal.width() * self.normal.height()
    }
    #[tracked]
    pub fn get_emission(
        &self,
        rc: Option<&RcData>,
        pos: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        dir: Expr<Vec3<f32>>,
    ) -> Expr<Radiance> {
        self.get_surface_irradiance(rc, pos, hit, dir) * self.world.get_diffuse(hit)
            + self.get_surface_emission(hit)
    }
    #[tracked]
    pub fn get_surface_emission(&self, hit: Expr<Hit>) -> Expr<Radiance> {
        #[cfg(not(feature = "voxel"))]
        let emission = self.world.get_emission(hit) * self.lighting.read().emission_scale;
        #[cfg(feature = "voxel")]
        let emission = Vec3::splat_expr(0.0f32);
        emission
    }
    #[tracked]
    pub fn get_surface_irradiance(
        &self,
        rc: Option<&RcData>,
        pos: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        dir: Expr<Vec3<f32>>,
    ) -> Expr<Radiance> {
        let normal = self.world.get_shading_normal(hit, dir);
        let lighting = self.lighting.read();
        let weight = normal.dot(lighting.sun_dir);
        let radiance = Vec3::splat(0.0_f32).var();

        if weight > 0.0 && lighting.sun_color.reduce_max() > 0.0 {
            #[cfg(not(feature = "voxel"))]
            let visible = !self.world.occluded(offset_ray(pos, normal), lighting.sun_dir);
            #[cfg(feature = "voxel")]
            let visible = self.world.trace(offset_ray(pos, normal), lighting.sun_dir).miss();
            *radiance += visible.select(lighting.sun_color, Vec3::splat_expr(0.0_f32))
                * weight;
        }
        if let Some(rc) = rc {
            *radiance += rc.sample_secondary(pos, normal);
        }

        if let Some(lights) = &self.point_lights {
            let candidates = self.light_grid.as_ref().map(|grid| grid.candidates(pos).var());
            for step in 0..lights.len() as u32 {
                let i = if let Some(mask) = &candidates {
                    if ***mask == 0u32 { break; }
                    let index = mask.trailing_zeros();
                    **mask &= ***mask - 1u32;
                    index
                } else { step };
                let light = lights.read(i);
                let delta = light.position_range.xyz() - pos;
                let d2 = delta.dot(delta);
                let range2 = light.position_range.w * light.position_range.w;
                if d2 > 0.0001 && d2 < range2 && light.intensity.xyz().reduce_max() > 0.0 {
                    let distance = d2.sqrt();
                    let direction = delta / distance;
                    let cosine = normal.dot(direction);
                    if cosine > 0.0 {
                        #[cfg(not(feature = "voxel"))]
                        let visible = !self.world.occluded_segment(offset_ray(pos, normal), direction, distance - 0.01);
                        #[cfg(feature = "voxel")]
                        let visible = {
                            let blocker = self.world.trace(offset_ray(pos, normal), direction);
                            blocker.miss() || blocker.committed_ray_t >= distance - 0.01
                        };
                        let q = d2 / range2;
                        let fade = 1.0 - q * q;
                        let attenuation = fade * fade / keter::max(d2, 100.0);
                        *radiance += visible.select(light.intensity.xyz(), Vec3::splat_expr(0.0f32)) * (cosine * attenuation);
                    }
                }
            }
        }

        for (emitter_inst, emitter_color) in self.custom_emitters.iter() {
            if hit.inst == *emitter_inst {
                *radiance += *emitter_color;
            }
        }

        **radiance
    }
    #[tracked]
    pub fn get_radiance(
        &self,
        rc: Option<&RcData>,
        start: Expr<Vec3<f32>>,
        dir: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
    ) -> Expr<Radiance> {
        if hit.miss() {
            self.get_environment(dir)
        } else {
            self.get_emission(rc, self.hit_position(start, dir, hit), hit, dir)
        }
    }
    #[tracked]
    pub fn get_environment(&self, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        // Vec3::splat_expr(0.0f32)
        /*let facing = dir.dot(Vec3::z());
        let base = Vec3::splat(1.0f32);
        let falloff = Vec3::new(0.9, 0.9, 0.2);
        let downwards_falloff = 1.0 / (1.0 + 2000.0 * keter::max(-facing, 0.0));
        downwards_falloff * (base - keter::max(facing, 0.0).powf(0.1) * falloff)*/
        let color = if let Some(color) = self.constant_environment {
            color.expr()
        } else {
            let uv = EquirectangularEncoder::encode(dir);
            self.environment.var().sample(uv).xyz()
        };
        color * self.lighting.read().environment_scale
    }
    /// Reuse cosine-convolved radiance for secondary diffuse feedback. Rebuild
    /// the atlas from base_radiance after cache allocation and before tracing.
    #[tracked]
    pub fn get_radiance_cached(
        &self, rc: Option<&RcData>, irradiance: Option<&splat::SplatData>,
        start: Expr<Vec3<f32>>, dir: Expr<Vec3<f32>>, hit: Expr<Hit>,
    ) -> Expr<Radiance> {
        if let (Some(rc), Some(irradiance)) = (rc, irradiance) {
            if hit.miss() {
                self.get_environment(dir)
            } else {
                let pos = self.hit_position(start, dir, hit);
                let albedo = self.world.get_diffuse(hit);
                let diffuse = if albedo.reduce_max() > 0.0 {
                    let normal = self.world.get_shading_normal(hit, dir);
                    (self.get_surface_irradiance(None, pos, hit, dir)
                        + irradiance.sample_secondary(rc, pos, normal)) * albedo
                } else { Vec3::splat_expr(0.0f32) };
                diffuse + self.get_surface_emission(hit)
            }
        } else {
            self.get_radiance(rc, start, dir, hit)
        }
    }
    #[tracked]
    pub fn get_fluence(
        &self,
        rc: Option<&RcData>,
        start: Expr<Vec3<f32>>,
        dir: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        infinite: Expr<bool>,
    ) -> Expr<Fluence> {
        if hit.miss() {
            if infinite {
                self.get_environment(dir).extend(0.0)
            } else {
                Vec3::splat_expr(0.0_f32).extend(1.0)
            }
        } else {
            self.get_emission(rc, self.hit_position(start, dir, hit), hit, dir)
                .extend(0.0)
        }
    }
    #[tracked]
    pub fn hit_position(
        &self,
        _start: Expr<Vec3<f32>>,
        _dir: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
    ) -> Expr<Vec3<f32>> {
        #[cfg(not(feature = "voxel"))]
        let position = self.world.get_position(hit);
        #[cfg(feature = "voxel")]
        let position = _start + _dir * hit.committed_ray_t;
        position
    }
    #[tracked]
    pub fn get_world_pos(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        let view = self.view.read();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();
        let depth = self.depth.read(pixel);
        if depth == 0.0 {
            return;
        }
        let output = self.hit_position(
            view.pos,
            ray_dir,
            self.hit.read(self.pixel_to_linear(pixel)),
        );
        output
    }
    #[tracked]
    pub fn read_normal(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        SphericalEncoder::decode(self.normal.read(pixel))
    }
    /*#[tracked]
    pub fn read_normal_smooth(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        let hit = self.hit.read(self.pixel_to_linear(pixel));
        self.world.get_normal(hit)
    }*/
    #[tracked]
    pub fn read_depth(&self, pixel: Expr<Vec2<u32>>) -> Expr<f32> {
        // could use hit.committed_ray_t instead
        self.depth.read(pixel)
    }
    #[tracked]
    pub fn pixel_to_linear(&self, pixel: Expr<Vec2<u32>>) -> Expr<u32> {
        pixel.y * self.screen_size().x + pixel.x
    }
    pub fn compute_gbuffer(&self) -> NodeConfigs<'static> {
        self.gbuffer_kernel
            .get_or_init(|| DEVICE.create_kernel::<fn()>(&track!(|| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();
                let view = self.view.read();
                let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

                let hit = self.world.trace(view.pos, ray_dir);
                if hit.miss() {
                    self.depth.write(pixel, 0.0);
                    return;
                }
                self.depth.write(pixel, hit.committed_ray_t);
                let normal = self.world.get_shading_normal(hit, ray_dir);
                self.normal.write(pixel, SphericalEncoder::encode(normal));
                self.hit.write(self.pixel_to_linear(pixel), hit);
            })))
            .dispatch_async(self.screen_dispatch_size())
            .debug("Compute G-Buffer")
    }
}

pub mod ddgi;
pub mod dynamic_history;
pub mod diffuse;
pub mod specular_brdf;
pub mod specular;
pub mod cached_specular;
mod light_grid;
pub mod rc;
pub mod ris;
pub mod splat;
pub mod trace;

#[derive(Default)]
pub struct CommandList<'a> {
    unfinished: Vec<NodeConfigs<'a>>,
    pub timings: Vec<(String, f32)>,
}
impl<'a> AddAssign<NodeConfigs<'a>> for CommandList<'a> {
    fn add_assign(&mut self, rhs: NodeConfigs<'a>) {
        self.unfinished.push(rhs);
    }
}
impl<'a> CommandList<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn append(&mut self, mut other: Self) {
        self.unfinished.append(&mut other.unfinished);
        self.timings.append(&mut other.timings);
    }
    pub fn execute(&mut self) {
        let commands = std::mem::take(&mut self.unfinished).chain();
        #[cfg(feature = "trace")]
        self.timings.extend(commands.execute_timed());
        #[cfg(not(feature = "trace"))]
        commands.execute();
    }
    pub fn push(&mut self, node: impl AsNodes<'a>) {
        self.unfinished.push(node.into_node_configs());
    }
}
