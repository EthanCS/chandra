use std::f32::consts::TAU;
use std::ops::Deref;

use keter::bindless::Tex3dHandle;
use keter::lang::types::vector::Mat3;
use yesod::direction::OctahedralEncoder;
use yesod::rand::pcg3df;
use yesod::utils::{cross_matrix, outer_product_3, spherical_fibonacci};

use super::*;
use crate::utils::{from_linear_expr, index_to_corner};

pub struct DdgiConfig {
    pub origin: Vec3<f32>,
    pub spacing: Vec3<f32>,
    pub probe_dims: Vec3<u32>,
    pub rays_per_probe: u32,
    pub irradiance_resolution: u32,
    pub distance_resolution: u32,
    pub hysteresis: f32,
    pub shadow_bias: f32,
    pub max_distance: f32,
}
impl Default for DdgiConfig {
    fn default() -> Self {
        Self {
            origin: Vec3::new(-160.0, 0.0, -160.0),
            spacing: Vec3::splat(10.0),
            probe_dims: Vec3::new(32, 32, 16),
            rays_per_probe: 128,
            irradiance_resolution: 8,
            distance_resolution: 16,
            hysteresis: 0.97,
            shadow_bias: 0.3,
            max_distance: 10000.0,
        }
    }
}
impl DdgiConfig {
    pub fn probe_irradiance_size(&self) -> Vec3<u32> {
        Vec3::new(
            self.irradiance_resolution + 2,
            self.irradiance_resolution + 2,
            1,
        )
    }
    pub fn probe_distance_size(&self) -> Vec3<u32> {
        Vec3::new(
            self.distance_resolution + 2,
            self.distance_resolution + 2,
            1,
        )
    }
    pub fn num_probes(&self) -> u32 {
        self.probe_dims.x * self.probe_dims.y * self.probe_dims.z
    }
    pub fn min_spacing(&self) -> f32 {
        self.spacing.x.min(self.spacing.y).min(self.spacing.z)
    }
}

#[tracked]
fn random_rotation(rand: Expr<Vec3<f32>>) -> Expr<Mat3> {
    let dir = SphericalEncoder::decode(rand.xy());
    let angle = rand.z * TAU;
    let (asin, acos) = angle.sin_cos();
    acos * Mat3::identity().expr()
        + asin * cross_matrix(dir)
        + (1.0 - acos) * outer_product_3(dir, dir)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct Ray {
    pub dir: Vec2<f32>,
    pub distance: f32,
    pub hit_backface: bool,
    pub radiance: Radiance,
}
const _: fn() = || {
    let _ = std::mem::transmute::<Ray, [f32; 8]>;
};

pub struct DdgiData {
    pub ray_data: Buffer<Ray>,
    pub irradiance: Tex3dHandle<Radiance>,
    pub distance: Tex3dHandle<Vec2<f32>>,
    config: DdgiConfig,
}
impl Deref for DdgiData {
    type Target = DdgiConfig;
    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
impl DdgiData {
    pub fn new(config: DdgiConfig) -> Self {
        let ray_data =
            DEVICE.create_buffer::<Ray>((config.num_probes() * config.rays_per_probe) as usize);
        let irradiance_size = config.irradiance_resolution + 2;
        let distance_size = config.distance_resolution + 2;
        let irradiance = DEVICE.create_tex3d::<Radiance>(
            PixelStorage::Float4,
            config.probe_dims.x * irradiance_size,
            config.probe_dims.y * irradiance_size,
            config.probe_dims.z,
            1,
        );
        let distance = DEVICE.create_tex3d::<Vec2<f32>>(
            PixelStorage::Float4,
            config.probe_dims.x * distance_size,
            config.probe_dims.y * distance_size,
            config.probe_dims.z,
            1,
        );
        let irradiance = BINDLESS.push_tex3d(
            irradiance,
            Sampler {
                filter: SamplerFilter::LinearPoint,
                address: SamplerAddress::Edge,
            },
        );
        let distance = BINDLESS.push_tex3d(
            distance,
            Sampler {
                filter: SamplerFilter::LinearPoint,
                address: SamplerAddress::Edge,
            },
        );
        BINDLESS.update();
        Self {
            config,
            ray_data,
            irradiance,
            distance,
        }
    }
    #[tracked]
    fn world_to_probe(&self, world_pos: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
        let pos = (world_pos - self.origin) / self.spacing;
        pos.clamp(
            Vec3::splat_expr(0.0),
            self.probe_dims.expr().cast_f32() - 1.001,
        )
    }
    #[tracked]
    fn probe_to_world(&self, probe_pos: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
        probe_pos * self.spacing + self.origin
    }
    #[tracked]
    pub fn sample_distance(&self, probe: Expr<Vec3<u32>>, dir: Expr<Vec2<f32>>) -> Expr<Vec2<f32>> {
        let offset = (probe * self.probe_distance_size()).cast_f32();
        let uv = dir * self.distance_resolution as f32 + 1.0;
        self.distance
            .var()
            .sample((offset + uv.extend(0.5)) / self.distance.size().expr().cast_f32())
            .xy()
    }
    #[tracked]
    pub fn sample_irradiance(
        &self,
        probe: Expr<Vec3<u32>>,
        dir: Expr<Vec2<f32>>,
    ) -> Expr<Radiance> {
        let offset = (probe * self.probe_irradiance_size()).cast_f32();
        let uv = dir * self.irradiance_resolution as f32 + 1.0;
        self.irradiance
            .var()
            .sample((offset + uv.extend(0.5)) / self.irradiance.size().expr().cast_f32())
            .xyz()
    }
    #[tracked]
    pub fn query_irradiance(
        &self,
        pos: Expr<Vec3<f32>>,
        normal: Expr<Vec3<f32>>,
        // facing away from the point
        camera_dir: Expr<Vec3<f32>>,
    ) -> Expr<Radiance> {
        let normal_bias = 0.2f32 * 0.75 * self.min_spacing() * self.shadow_bias;
        let camera_bias = 0.8f32 * 0.75 * self.min_spacing() * self.shadow_bias;
        let bias = normal * normal_bias + camera_dir * camera_bias;
        let biased_pos = pos + bias;
        let probe_pos = self.world_to_probe(biased_pos);
        let base_probe = probe_pos.floor().cast_u32();
        let alpha = probe_pos.fract();

        let total_irradiance = Vec4::splat(0.0f32).var();

        for i in 0_u32..8_u32 {
            let adj_probe_offset = index_to_corner(i);
            let adj_probe = base_probe + adj_probe_offset;
            let adj_probe_pos = self.probe_to_world(adj_probe.cast_f32());

            let pos_to_adj = (adj_probe_pos - pos).normalize();
            let biased_pos_to_adj = (adj_probe_pos - biased_pos).normalize();
            let dist_to_adj = (adj_probe_pos - biased_pos).length();

            let weight = 1.0_f32.var();

            let trilinear = keter::max(0.001, (adj_probe_offset == 1).select(alpha, 1.0 - alpha));
            let trilinear_weight = trilinear.reduce_prod();
            let wrap_shading = 0.5 + 0.5 * camera_dir.dot(pos_to_adj);
            *weight *= wrap_shading.sqr() + 0.2;

            let distance =
                self.sample_distance(adj_probe, OctahedralEncoder::encode(-biased_pos_to_adj));

            let variance = (distance.y - distance.x.sqr()).abs();
            if dist_to_adj > distance.x {
                let v = dist_to_adj - distance.x;
                let chebyshev_weight = variance / (variance + v.sqr());
                let chebyshev_weight = chebyshev_weight.cube();
                *weight *= chebyshev_weight;
            }
            *weight = keter::max(0.000001, weight);

            let crush_threshold = 0.2f32;
            if weight < crush_threshold {
                *weight *= weight.sqr() / crush_threshold.powi(2);
            }

            *weight *= trilinear_weight;

            let irradiance = self.sample_irradiance(adj_probe, OctahedralEncoder::encode(normal));

            *total_irradiance += irradiance.sqrt().extend(1.0) * weight;
        }
        let irradiance = total_irradiance.xyz() / total_irradiance.w;
        irradiance.sqr()
    }
    pub fn trace_rays(&self, core: &CoreData, tick: u32) -> NodeConfigs<'static> {
        dispatch1(
            track!(|tick| {
                let probe_index = dispatch_id().x;
                let ray_index = dispatch_id().y;
                let rand = pcg3df(Vec3::expr(probe_index, tick, 173));
                let rot = random_rotation(rand);
                let dir = rot * spherical_fibonacci(ray_index, self.rays_per_probe.expr());
                let origin =
                    self.probe_to_world(from_linear_expr(probe_index, self.probe_dims).cast_f32());
                let hit = core.world.trace(origin, dir);
                let encoded_dir = OctahedralEncoder::encode(dir);
                let ray = if hit.miss() {
                    Ray::from_comps_expr(RayComps {
                        dir: encoded_dir,
                        distance: self.max_distance.expr(),
                        hit_backface: false.expr(),
                        radiance: Vec3::splat_expr(0.0),
                    })
                } else {
                    let normal = core.world.get_normal_flat(hit);
                    let hit_backface = normal.dot(dir) > 0.0;
                    let radiance = if hit_backface {
                        Vec3::splat_expr(0.0)
                    } else {
                        core.get_emission(origin + dir * hit.committed_ray_t, hit)
                    };
                    let distance = if hit_backface {
                        0.2 * hit.committed_ray_t
                    } else {
                        hit.committed_ray_t
                    };
                    Ray::from_comps_expr(RayComps {
                        dir: encoded_dir,
                        distance,
                        hit_backface,
                        radiance,
                    })
                };
                self.ray_data
                    .write(probe_index * self.rays_per_probe + ray_index, ray);
            }),
            [self.num_probes(), self.rays_per_probe, 1],
            tick,
        )
        .debug("DDGI Trace Rays")
    }
    pub fn splat_irradiance(&self) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([1, 8, 8]);
                let probe_index = dispatch_id().x;
                let probe_pos = from_linear_expr(probe_index, self.probe_dims);
                let local_pos = dispatch_id().yz();
                let texture_index =
                    probe_pos * self.probe_irradiance_size() + (local_pos + 1).extend(0);
                let previous_irradiance = self.irradiance.read(texture_index);
                let irradiance = Vec4::splat(0.0f32).var();
                let direction = OctahedralEncoder::decode(
                    (local_pos.cast_f32() + 0.5) / self.irradiance_resolution as f32,
                );
                for ray_index in 0..self.rays_per_probe {
                    let ray = self
                        .ray_data
                        .read(probe_index * self.rays_per_probe + ray_index);
                    let weight = direction.dot(OctahedralEncoder::decode(ray.dir));
                    if weight > 0.0 {
                        *irradiance += ray.radiance.extend(1.0) * weight;
                    }
                }
                let irradiance = irradiance.xyz() / irradiance.w;
                let irradiance = irradiance.lerp(previous_irradiance, self.hysteresis);
                self.irradiance.write(texture_index, irradiance);
            }),
            [
                self.num_probes(),
                self.irradiance_resolution,
                self.irradiance_resolution,
            ],
        )
        .debug("DDGI Splat Irradiance")
    }
    pub fn splat_distance(&self) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([1, 8, 8]);
                let probe_index = dispatch_id().x;
                let probe_pos = from_linear_expr(probe_index, self.probe_dims);
                let local_pos = dispatch_id().yz();
                let texture_index =
                    probe_pos * self.probe_distance_size() + (local_pos + 1).extend(0);
                let previous_distance = self.distance.read(texture_index);
                let distance = Vec3::splat(0.0f32).var();
                let direction = OctahedralEncoder::decode(
                    (local_pos.cast_f32() + 0.5) / self.distance_resolution as f32,
                );
                for ray_index in 0..self.rays_per_probe {
                    let ray = self
                        .ray_data
                        .read(probe_index * self.rays_per_probe + ray_index);
                    let weight = direction.dot(OctahedralEncoder::decode(ray.dir));
                    if weight > 0.0 {
                        let weight = weight.sqr();
                        *distance += Vec3::expr(ray.distance, ray.distance.sqr(), 1.0) * weight;
                    }
                }
                let distance = distance.xy() / distance.z;
                let distance = distance.lerp(previous_distance, self.hysteresis);
                self.distance.write(texture_index, distance);
            }),
            [
                self.num_probes(),
                self.distance_resolution,
                self.distance_resolution,
            ],
        )
        .debug("DDGI Splat Distance")
    }
    pub fn copy_borders(&self) -> NodeConfigs<'static> {
        (
            dispatch0(
                track!(|| {
                    set_block_size([1, 8, 4]);
                    self.copy_borders_raw(false);
                }),
                [self.num_probes(), self.irradiance_resolution, 4],
            )
            .debug("DDGI Copy Irradiance Borders"),
            dispatch0(
                track!(|| {
                    set_block_size([1, 16, 4]);
                    self.copy_borders_raw(true);
                }),
                [self.num_probes(), self.distance_resolution, 4],
            )
            .debug("DDGI Copy Distance Borders"),
            dispatch0(
                track!(|| {
                    set_block_size([8, 4, 1]);
                    let probe_index = dispatch_id().x;
                    let probe_pos = from_linear_expr(probe_index, self.probe_dims);
                    let corner = Vec2::expr(dispatch_id().y % 2, dispatch_id().y / 2);
                    // Irradiance
                    let write_pos = corner * (self.irradiance_resolution + 1);
                    let read_pos = (1 - corner) * (self.irradiance_resolution) + 1;
                    let value = self
                        .irradiance
                        .read(probe_pos * self.probe_irradiance_size() + read_pos.extend(0));
                    self.irradiance.write(
                        probe_pos * self.probe_irradiance_size() + write_pos.extend(0),
                        value,
                    );
                    // Distance
                    let write_pos = corner * (self.distance_resolution + 1);
                    let read_pos = (1 - corner) * (self.distance_resolution) + 1;
                    let value = self
                        .distance
                        .read(probe_pos * self.probe_distance_size() + read_pos.extend(0));
                    self.distance.write(
                        probe_pos * self.probe_distance_size() + write_pos.extend(0),
                        value,
                    );
                }),
                [self.num_probes(), 4, 1],
            )
            .debug("DDGI Copy Corners"),
        )
            .into_node_configs()
    }
    #[tracked]
    fn copy_borders_raw(&self, is_distance: bool) {
        let size = if is_distance {
            self.probe_distance_size()
        } else {
            self.probe_irradiance_size()
        };
        let res = if is_distance {
            self.distance_resolution
        } else {
            self.irradiance_resolution
        };

        let probe_index = dispatch_id().x;
        let probe_pos = from_linear_expr(probe_index, self.probe_dims);
        let texture_offset = probe_pos * size;

        let write_index = dispatch_id().y + 1;
        let read_index = res + 2 - write_index;
        let side = dispatch_id().z;
        let near = side % 2 == 0;
        let write_pos = Vec2::expr(
            write_index,
            if near { 0_u32.expr() } else { (res + 1).expr() },
        );
        let read_pos = Vec2::expr(read_index, if near { 1_u32.expr() } else { res.expr() });
        let horiz = side / 2 == 0;
        let write_pos = if horiz { write_pos } else { write_pos.yx() };
        let read_pos = if horiz { read_pos } else { read_pos.yx() };
        if is_distance {
            let value = self.distance.read(texture_offset + read_pos.extend(0));
            self.distance
                .write(texture_offset + write_pos.extend(0), value);
        } else {
            let value = self.irradiance.read(texture_offset + read_pos.extend(0));
            self.irradiance
                .write(texture_offset + write_pos.extend(0), value);
        }
    }
}
