use keter::lang::functions::{block_id, sync_block, thread_id};
use yesod::color::inferno;
use yesod::shapes::intersect_sphere;
use yesod::utils::FetchAddVector;

use super::*;
use crate::algorithm::rc::RcData;
use crate::cache::{DirectionIndexer, DirectionIndexerComps};

pub struct SplatData {
    pub radiance: Buffer<Vec4<f32>>,
    // debug only
    pub depth: Buffer<f32>,
    // for base probes only, currently
    pub average_pos_normal: Buffer<AveragePosNormal>,
    // settings
    pub use_planar_weights: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Value, Default)]
#[repr(C)]
pub struct AveragePosNormal {
    position_count: Vec4<f32>,
    total_normal: Vec3<f32>,
}
impl AveragePosNormalExpr {
    #[tracked]
    pub fn position(&self) -> Expr<Vec3<f32>> {
        self.position_count.xyz() / self.position_count.w
    }
    #[tracked]
    pub fn normal(&self) -> Expr<Vec3<f32>> {
        self.total_normal / self.position_count.w
    }
}
impl AveragePosNormalAtomicRef {
    pub fn add_sample(&self, pos: Expr<Vec3<f32>>, normal: Expr<Vec3<f32>>) {
        self.position_count.fetch_add(pos.extend(1.0));
        self.total_normal.fetch_add(normal);
    }
}

impl SplatData {
    pub fn new(core: &CoreData, rc: &RcData, use_planar_weights: bool) -> Self {
        Self {
            radiance: DEVICE.create_buffer::<Vec4<f32>>(core.num_pixels() as usize),
            depth: DEVICE.create_buffer::<f32>(core.num_pixels() as usize),
            average_pos_normal: DEVICE
                .create_buffer_from_fn(rc.cache.base_probes as usize, |_| Default::default()),
            use_planar_weights,
        }
    }
    // TODO: Apply to more cascades.
    pub fn compute_average_pos_normal(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        dispatch0_async(
            track!(|| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();
                let world_pos = core.get_world_pos(pixel);
                let normal = core.read_normal(pixel);

                let lod = rc.cache.estimate_lod(world_pos);
                let probe = rc.cache.quantize(world_pos, lod, 0.expr());
                let node = rc.cache.get(probe);

                self.average_pos_normal
                    .atomic_ref(node.index)
                    .add_sample(world_pos, normal);
            }),
            core.screen_dispatch_size(),
        )
        .debug("Compute Average Position and Normal")
    }
    pub fn splat_radiance_raw(
        &self,
        core: &CoreData,
        rc: &RcData,
    ) -> &'static Kernel<fn(f32, u32, u32)> {
        memo(
            (self.use_planar_weights, rc.increased_base_resolution),
            track!(|| {
                DEVICE.create_kernel::<fn(f32, u32, u32)>(&|blend_radius, cascade, draw_mode| {
                    set_block_size([1, 32, 1]);
                    let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                    if node.invalid() {
                        return;
                    }
                    let view = core.view.read();

                    let probe = node.probe;
                    let encoder = DirectionIndexer::from_comps_expr(DirectionIndexerComps {
                        resolution: if rc.increased_base_resolution && cascade == 0 {
                            rc.cache.base_angular_resolution.expr()
                        } else {
                            (rc.cache.base_angular_resolution / 2) * (1 << cascade)
                        },
                    });
                    assert_eq!(rc.cache.base_angular_resolution, 4);

                    // let quality = node_quality.read(node.index);

                    // 8 if !INCREASED_BASE_RESOLUTION
                    let radiance_data = Shared::<Vec3<f32>>::new(32);
                    if rc.increased_base_resolution && cascade == 0 {
                        let v = rc.base_radiance.read(node.data_offset + dispatch_id().y);
                        radiance_data.write(thread_id().y, v);
                    } else if cascade != 0 || dispatch_id().y < 8 {
                        let v = rc.radiance.read(node.data_offset / 4 + dispatch_id().y);
                        radiance_data.write(thread_id().y, v);
                    }
                    sync_block();

                    let screen_min = core.screen_size().expr().cast_f32().var();
                    let screen_max = Vec2::splat(0.0_f32).var();

                    let spacing = rc.cache.base_grid_size
                        * (1 << (probe.position.w + cascade)).cast_f32()
                        * blend_radius;
                    let probe_pos = rc.cache.dequantize(probe);

                    let avg_pos_normal = if self.use_planar_weights && cascade == 0 {
                        self.average_pos_normal.read(node.index)
                    } else {
                        AveragePosNormal::default().expr()
                    };
                    let avg_pos = avg_pos_normal.position();
                    // Perhaps unnormalized could have some value? It shows the variance.
                    let avg_normal = avg_pos_normal.normal().normalize();

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
                    let screen_max = keter::min(core.screen_size().expr().cast_f32(), **screen_max)
                        .ceil()
                        .cast_u32();
                    let size = screen_max - screen_min;
                    let area = size.reduce_prod();
                    let i = dispatch_id().y.var();
                    while i < area {
                        let pixel = screen_min + Vec2::expr(i % size.x, i / size.x);
                        let pixel_index = core.pixel_to_linear(pixel);
                        *i += 32;

                        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

                        // TODO: This is a complete fucking mess and needs a lot of cleanup.
                        if draw_mode == 0 {
                            let depth = core.read_depth(pixel);
                            if depth == 0.0 {
                                continue;
                            }
                            let world_pos = view.pos + ray_dir * depth;
                            let normal = core.read_normal(pixel);
                            let trilinear_weight =
                                keter::max(1.0 - (world_pos - probe_pos).abs() / spacing, 0.0)
                                    .reduce_prod();
                            let planar_weight = if self.use_planar_weights && cascade == 0 {
                                keter::max(
                                    0.4 - (avg_pos - world_pos).dot(avg_normal).abs() / spacing,
                                    0.0,
                                )
                            } else {
                                1.0.expr()
                            };

                            let weight =
                                trilinear_weight * planar_weight * rc.get_quality_weight(node);
                            // * (1 << (2 * probe.position.w)).cast_f32().recip()
                            if weight == 0.0 {
                                continue;
                            }
                            // let (factor, cascade) = rc.cache.projection().interpolation_factor_lod(world_pos);
                            // let factor = if cascade == probe.position.w {
                            //     factor
                            // } else {
                            //     1.0 - factor
                            // };
                            // let weight = weight * factor;

                            let total_radiance = Vec4::splat(0.0_f32).var();
                            for j in 0_u32.expr()
                                ..if cascade == 0 && !rc.increased_base_resolution {
                                    8_u32.expr()
                                } else {
                                    32_u32.expr()
                                }
                            {
                                let dir = encoder.linear_to_dir(j + block_id().y * 32);
                                let weight = dir.dot(normal);
                                if weight > 0.0 {
                                    let radiance = radiance_data.read(j);
                                    if radiance.is_finite().all() {
                                        *total_radiance += weight * radiance.extend(1.0);
                                    } else {
                                        *total_radiance.w = 0.0;
                                        break;
                                    }
                                }
                            } // TODO: Figure out the biasing properly? Just directly bias based on amount of missing data?
                            if total_radiance.w < 0.0001 {
                                continue;
                            }

                            let total_radiance = total_radiance.xyz() / total_radiance.w;
                            // let total_radiance = pcg3df(probe.position.xyz());
                            let deposit_radiance = total_radiance.extend(1.0) * weight;
                            let dest = self.radiance.atomic_ref(pixel_index);
                            dest.fetch_add(deposit_radiance);
                        } else if draw_mode == 1 || draw_mode == 2 {
                            let (time, intersected) =
                                intersect_sphere(view.pos - probe_pos, ray_dir, spacing);
                            if !intersected {
                                continue;
                            }
                            let depth = time.x;
                            let intersect_pos = view.pos + ray_dir * depth;

                            let intersect_dir = (intersect_pos - probe_pos).normalize();
                            let index = encoder.dir_to_linear(intersect_dir) - block_id().y * 32;
                            if index >= 32 {
                                continue;
                            }

                            if draw_mode == 2 {
                                // depth prepass
                                self.depth.atomic_ref(pixel_index).fetch_min(depth);
                            } else if draw_mode == 1 {
                                let actual_depth = self.depth.read(pixel_index);
                                if depth == actual_depth {
                                    let radiance = radiance_data.read(index);
                                    if radiance.is_finite().all() {
                                        let deposit_radiance = radiance.extend(1.0);
                                        let dest = self.radiance.atomic_ref(pixel_index);
                                        dest.fetch_add(deposit_radiance);
                                    }
                                }
                            }
                        } else if draw_mode == 3 || draw_mode == 4 {
                            let depth =
                                (avg_pos - view.pos).dot(avg_normal) / ray_dir.dot(avg_normal);
                            let intersect_pos = view.pos + ray_dir * depth;
                            if depth > 0.0 && ((intersect_pos - probe_pos).abs() < spacing).all() {
                                if draw_mode == 4 {
                                    // depth prepass
                                    self.depth.atomic_ref(pixel_index).fetch_min(depth);
                                } else if draw_mode == 3 {
                                    let actual_depth = self.depth.read(pixel_index);
                                    if depth == actual_depth {
                                        let radiance = inferno(
                                            (intersect_pos - avg_pos).length() / (2.0 * spacing),
                                        );
                                        if radiance.is_finite().all() {
                                            let deposit_radiance = radiance.extend(1.0);
                                            let dest = self.radiance.atomic_ref(pixel_index);
                                            dest.fetch_add(deposit_radiance);
                                        }
                                    }
                                }
                            }
                        }
                    }
                })
            }),
        )
    }
}
