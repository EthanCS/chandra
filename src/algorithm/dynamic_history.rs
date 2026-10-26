//! Reactive world-space history for changing geometry and illumination.
//!
//! Swept AABBs reduce confidence in affected directional groups without erasing
//! their normalized radiance. Lighting changes use a separate response window.
use keter::graph::NodeConfigs;
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;

use crate::algorithm::rc::RcData;
use crate::algorithm::trace::{Hit, TraceData};
use crate::dynamic::Bounds;

const MAX_REGIONS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Value)]
struct Region {
    min: Vec3<f32>,
    max: Vec3<f32>,
}

pub struct DynamicHistory {
    regions: Buffer<Region>,
    invalidate: Kernel<fn(u32, f32, f32, u32)>,
    invalidate_hits: Kernel<fn(u32)>,
    hit_count: u32,
    _inverse_directions: Buffer<Vec3<f32>>,
}

impl DynamicHistory {
    pub fn new(rc: &RcData, trace: &TraceData) -> Self {
        Self::with_region_capacity(rc, trace, MAX_REGIONS)
    }

    pub fn with_region_capacity(rc: &RcData, trace: &TraceData, capacity: usize) -> Self {
        assert!(capacity > 0 && capacity <= u32::MAX as usize);
        let regions = DEVICE.create_buffer::<Region>(capacity);
        let inverse_directions =
            DEVICE.create_buffer::<Vec3<f32>>(rc.cache.sample_directions.len());
        let initialize = DEVICE.create_kernel::<fn()>(&track!(|| {
            let index = dispatch_id().x;
            let dir = rc.cache.sample_directions.read(index);
            let safe_dir = (dir.abs() < 1e-7).select(Vec3::splat_expr(1e-7), dir);
            inverse_directions.write(index, 1.0 / safe_dir);
        }));
        initialize.dispatch([inverse_directions.len() as u32, 1, 1]);
        let invalidate = DEVICE.create_kernel::<fn(u32, f32, f32, u32)>(&track!(
            |count, retention, motion_retention, cascade| {
                // Neighbouring lanes cover adjacent directional bins, making
                // the four-value cache updates coalesce within each probe.
                set_block_size([4, 8, 1]);
                let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                if node.invalid() {
                    return;
                }
                let origin = rc.cache.dequantize(node.probe);
                let padding = rc.cache.probe_spacing(node.probe) * 2.0;
                let group = dispatch_id().y * 4;
                let end = if cascade == rc.last_cascade() {
                    f32::INFINITY.expr()
                } else {
                    let c0 = ((2 * rc.base_angular_resolution.pow(2)) as f32
                        / (4.0 * std::f32::consts::PI))
                        .sqrt();
                    c0 * rc.cache.spacing(node.probe.position.w, 0)
                        * rc.length_scale.expr().powi(cascade.cast_i32())
                };
                let affected = false.var();
                for k in 0..4u32 {
                    let inv = inverse_directions.read(rc.cache.direction_index(cascade, group + k));
                    for r in 0u32.expr()..count {
                        let region = regions.read(r);
                        let a = (region.min - padding - origin) * inv;
                        let b = (region.max + padding - origin) * inv;
                        let near = keter::min(a, b);
                        let far = keter::max(a, b);
                        let enter = keter::max(keter::max(near.x, near.y), keter::max(near.z, 0.0));
                        let leave = keter::min(keter::min(far.x, far.y), keter::min(far.z, end));
                        *affected |= enter <= leave;
                        // OR reduction: once any ray footprint hits a moving
                        // region the whole four-bin group loses confidence.
                        if affected {
                            break;
                        }
                    }
                    if affected {
                        break;
                    }
                }
                // A hash-cell centre is not the origin of its surface samples.
                // Clearing it also erases valid estimates on moving surfaces.
                let factor = affected.select(keter::min(retention, motion_retention), retention);
                let cap = keter::max(
                    16.0f32,
                    trace.ray_count.read(node.index).cast_f32() * 8.0
                        / rc.cache.get_encoder(cascade).linear_size().cast_f32(),
                );
                for k in 0..4u32 {
                    let index = node.data_offset + group + k;
                    let old_weight = rc.cache.weights.read(index);
                    let weight =
                        affected.select(keter::min(old_weight * factor, cap), old_weight * factor);
                    if weight < 0.0001 {
                        rc.cache.weights.write(index, 0.0);
                        rc.cache.weighted_fluence.write(index, Vec4::splat(0.0));
                    } else {
                        rc.cache.weights.write(index, weight);
                        rc.cache.weighted_fluence.write(
                            index,
                            rc.cache.weighted_fluence.read(index) * (weight / old_weight),
                        );
                    }
                }
            }
        ));
        let invalidate_hits = DEVICE.create_kernel::<fn(u32)>(&track!(|count| {
            let i = dispatch_id().x;
            let hit = trace.traced_hits.read(i);
            if !hit.valid() {
                return;
            }
            for r in 0u32.expr()..count {
                let region = regions.read(r);
                if (hit.position >= region.min - 0.1).all()
                    && (hit.position <= region.max + 0.1).all()
                {
                    trace.traced_hits.write(i, Hit::invalid());
                    return;
                }
            }
        }));
        Self {
            regions,
            invalidate,
            invalidate_hits,
            hit_count: trace.traced_hits.len() as u32,
            _inverse_directions: inverse_directions,
        }
    }

    pub fn update(
        &self,
        rc: &RcData,
        swept_bounds: &[Bounds],
        retention: f32,
    ) -> NodeConfigs<'static> {
        self.update_with_dt(rc, swept_bounds, retention, 1.0 / 60.0)
    }

    pub fn update_with_dt(
        &self,
        rc: &RcData,
        swept_bounds: &[Bounds],
        retention: f32,
        dt: f32,
    ) -> NodeConfigs<'static> {
        assert!(retention.is_finite() && (0.0..=1.0).contains(&retention));
        assert!(dt.is_finite() && dt > 0.0);
        let motion_retention = 0.5f32.powf(dt / 0.100);
        assert!(
            swept_bounds
                .iter()
                .all(|b| b.min.is_finite() && b.max.is_finite() && b.min.cmple(b.max).all())
        );
        let bounds = if swept_bounds.len() > self.regions.len() {
            vec![swept_bounds.iter().copied().reduce(Bounds::union).unwrap()]
        } else {
            swept_bounds.to_vec()
        };
        let mut commands = Vec::new();
        let count = bounds.len() as u32;
        if count > 0 {
            let regions: Vec<_> = bounds
                .iter()
                .map(|b| Region {
                    min: b.min.into(),
                    max: b.max.into(),
                })
                .collect();
            commands.push(self.regions.view(..regions.len()).copy_from_vec(regions));
            commands.push(
                self.invalidate_hits
                    .dispatch_async([self.hit_count, 1, 1], &count)
                    .into_node_configs(),
            );
        }
        if count > 0 || retention < 1.0 {
            for cascade in 0..rc.num_cascades {
                commands.push(
                    self.invalidate
                        .dispatch_async(
                            [
                                rc.max_probes[cascade as usize],
                                rc.cache.directions_per_probe(cascade) / 4,
                                1,
                            ],
                            &count,
                            &retention,
                            &motion_retention,
                            &cascade,
                        )
                        .into_node_configs(),
                );
            }
        }
        commands.chain().debug("Dynamic history invalidation")
    }
}

/// Direction edits recover quickly after dragging stops. Energy edits keep a
/// short flushing window so old multiple-bounce energy can leave the cache.
#[derive(Default)]
pub struct LightingResponse {
    direction_remaining: f32,
    energy_remaining: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct HistoryResponse {
    pub retention: f32,
    pub primary_samples: u32,
}

impl LightingResponse {
    pub fn update(
        &mut self,
        dt: f32,
        direction_changed: bool,
        energy_changed: bool,
        multibounce: bool,
    ) -> HistoryResponse {
        assert!(dt.is_finite() && dt > 0.0);
        let duration = if multibounce { 0.500 } else { 0.150 };
        if direction_changed {
            self.direction_remaining = duration;
        }
        if energy_changed {
            self.energy_remaining = duration;
        }
        let energy_active = self.energy_remaining > 0.0;
        let strength = if energy_active {
            1.0
        } else {
            (self.direction_remaining / duration).clamp(0.0, 1.0)
        };
        self.direction_remaining = (self.direction_remaining - dt).max(0.0);
        self.energy_remaining = (self.energy_remaining - dt).max(0.0);
        HistoryResponse {
            retention: 0.5f32.powf(dt * strength / if energy_active { 0.025 } else { 0.045 }),
            primary_samples: if strength > 0.0 { 4 } else { 1 },
        }
    }
}
