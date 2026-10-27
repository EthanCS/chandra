//! Surface-validated diffuse reconstruction and disocclusion sampling.
//! History is unexposed irradiance, before diffuse albedo and direct lighting.
use super::*;
use crate::algorithm::splat::SplatData;

pub struct DiffuseData {
    pub raw: Buffer<Vec4<f32>>,
    pub filtered: Buffer<Vec4<f32>>,
    pub output: Buffer<Vec4<f32>>,
    pub samples: Buffer<u32>,
    position: [Buffer<Vec4<f32>>; 2],
    normal: [Buffer<Vec4<f32>>; 2],
    history: [Buffer<Vec4<f32>>; 2],
    // Valid observations are independent of the time-scaled blend weight.
    // A short response window must not make a mature surface disoccluded.
    history_age: [Buffer<f32>; 2],
    prepare: std::sync::OnceLock<Kernel<fn(u32, View, bool, u32, u32)>>,
    capture: std::sync::OnceLock<Kernel<fn()>>,
    spatial: std::sync::OnceLock<Kernel<fn(u32)>>,
    temporal: std::sync::OnceLock<Kernel<fn(u32, View, bool, f32)>>,
}

impl DiffuseData {
    pub fn new(core: &CoreData) -> Self {
        let create = || {
            DEVICE
                .create_buffer_from_fn::<Vec4<f32>>(core.num_pixels() as usize, |_| Vec4::splat(0.))
        };
        Self {
            raw: create(),
            filtered: create(),
            output: create(),
            samples: DEVICE.create_buffer(core.num_pixels() as usize),
            position: [create(), create()],
            normal: [create(), create()],
            history: [create(), create()],
            history_age: std::array::from_fn(|_| DEVICE.create_buffer_from_fn(core.num_pixels() as usize, |_| 0.0f32)),
            prepare: Default::default(),
            capture: Default::default(),
            spatial: Default::default(),
            temporal: Default::default(),
        }
    }
    #[tracked]
    fn position(&self, slot: Expr<u32>, index: Expr<u32>) -> Expr<Vec4<f32>> {
        if slot == 0u32 {
            self.position[0].read(index)
        } else {
            self.position[1].read(index)
        }
    }
    #[tracked]
    fn normal(&self, slot: Expr<u32>, index: Expr<u32>) -> Expr<Vec4<f32>> {
        if slot == 0u32 {
            self.normal[0].read(index)
        } else {
            self.normal[1].read(index)
        }
    }
    #[tracked]
    fn history(&self, slot: Expr<u32>, index: Expr<u32>) -> Expr<Vec4<f32>> {
        if slot == 0u32 {
            self.history[0].read(index)
        } else {
            self.history[1].read(index)
        }
    }
    #[tracked]
    fn reproject(
        &self,
        core: &CoreData,
        slot: Expr<u32>,
        view: Expr<View>,
        position: Expr<Vec4<f32>>,
        normal: Expr<Vec3<f32>>,
    ) -> (Expr<Vec4<f32>>, Expr<f32>) {
        let local = view.transform.transpose() * (position.xyz() - view.pos);
        let screen = view.project(position.xyz()) - 0.5;
        let output = Vec4::splat(0.0f32).var();
        let age = 0.0f32.var();
        if local.z > 0.0
            && (screen >= 0.0).all()
            && (screen < core.screen_size().expr().cast_f32() - 1.0).all()
        {
            let base = screen.floor().cast_u32();
            let fraction = screen.fract();
            let sum = Vec4::splat(0.0f32).var();
            let weight_sum = 0.0f32.var();
            let footprint = keter::max(0.05, local.z * view.scaling * 2.0);
            for corner in 0u32..4 {
                let p = base + Vec2::expr(corner & 1u32, corner >> 1u32);
                let index = core.pixel_to_linear(p);
                let previous = self.position(slot, index);
                let n = self.normal(slot, index).xyz();
                if previous.w != position.w
                    || n.dot(normal) < 0.98
                    || (previous.xyz() - position.xyz()).length() > footprint
                {
                    continue;
                }
                // Reject an adjacent parallel surface even if it is in the same mesh.
                if (previous.xyz() - position.xyz()).dot(normal).abs()
                    > keter::max(0.03, footprint * 0.1)
                {
                    continue;
                }
                let history = self.history(slot, index);
                if history.w <= 0.0 || !history.is_finite().all() {
                    continue;
                }
                let weight = ((corner & 1u32) != 0u32).select(fraction.x, 1.0 - fraction.x)
                    * ((corner >> 1u32) != 0u32).select(fraction.y, 1.0 - fraction.y);
                *sum += history * weight;
                *age += (if slot == 0u32 { self.history_age[0].read(index) }
                    else { self.history_age[1].read(index) }) * weight;
                *weight_sum += weight;
            }
            if weight_sum >= 0.75 {
                *output = sum / weight_sum;
                *age /= weight_sum;
            } else {
                *age = 0.0;
            }
        }
        (**output, **age)
    }
    pub fn prepare(
        &self,
        core: &CoreData,
        phase: u32,
        previous: View,
        valid: bool,
        base_samples: u32,
        override_samples: u32,
    ) -> NodeConfigs<'static> {
        self.prepare
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32, View, bool, u32, u32)>(&track!(
                    |phase, previous, valid, base_samples, override_samples| {
                        let pixel = dispatch_id().xy();
                        let index = core.pixel_to_linear(pixel);
                        let position = Vec4::splat(0.0f32).var();
                        let normal = Vec4::splat(0.0f32).var();
                        let count = base_samples.var();
                        if core.read_depth(pixel) > 0.0 {
                            let hit = core.hit.read(index);
                            *position = core
                                .get_world_pos(pixel)
                                .extend((hit.inst + 1u32).cast_f32());
                            *normal = core.read_normal(pixel).extend(1.0);
                            let support = 0u32.var();
                            for d in 0u32..4 {
                                let delta = <[Vec2<i32>; 4]>::expr([
                                    Vec2::new(-1, 0),
                                    Vec2::new(1, 0),
                                    Vec2::new(0, -1),
                                    Vec2::new(0, 1),
                                ])
                                .read(d);
                                let q = pixel.cast_i32() + delta;
                                if !(q >= 0).all()
                                    || !(q < core.screen_size().expr().cast_i32()).all()
                                {
                                    continue;
                                }
                                let q = q.cast_u32();
                                if core.read_depth(q) > 0.0
                                    && core.hit.read(core.pixel_to_linear(q)).inst == hit.inst
                                    && core.read_normal(q).dot(normal.xyz()) > 0.98
                                {
                                    *support += 1u32;
                                }
                            }
                            let age = if valid {
                                self.reproject(
                                    core,
                                    1u32 - phase,
                                    previous,
                                    **position,
                                    normal.xyz(),
                                ).1
                            } else {
                                0.0f32.expr()
                            };
                            // Bilinear reprojection of a four-frame history can
                            // round just below four. That is not disocclusion.
                            *count = if age < 3.999 {
                                64u32.expr()
                            } else if support < 3u32 {
                                32u32.expr()
                            } else if support < 4u32 {
                                16u32.expr()
                            } else {
                                base_samples
                            };
                        }
                        if override_samples > 0u32 {
                            *count = override_samples;
                        }
                        self.samples.write(index, **count);
                        if phase == 0u32 {
                            self.position[0].write(index, **position);
                            self.normal[0].write(index, **normal);
                        } else {
                            self.position[1].write(index, **position);
                            self.normal[1].write(index, **normal);
                        }
                    }
                ))
            })
            .dispatch_async(
                core.screen_dispatch_size(),
                &phase,
                &previous,
                &valid,
                &base_samples,
                &override_samples,
            )
            .debug("Diffuse sample support")
    }
    pub fn capture(&self, core: &CoreData, splat: &SplatData) -> NodeConfigs<'static> {
        self.capture
            .get_or_init(|| {
                DEVICE.create_kernel::<fn()>(&track!(|| {
                    let pixel = dispatch_id().xy();
                    let index = core.pixel_to_linear(pixel);
                    let value = if core.read_depth(pixel) > 0.0 {
                        splat.get_radiance(index).extend(1.0)
                    } else {
                        Vec4::splat_expr(0.0f32)
                    };
                    self.raw.write(index, value);
                }))
            })
            .dispatch_async(core.screen_dispatch_size())
            .debug("Diffuse raw irradiance")
    }
    pub fn reconstruct(
        &self,
        core: &CoreData,
        phase: u32,
        previous: View,
        valid: bool,
        max_frames: f32,
    ) -> NodeConfigs<'static> {
        let spatial = self
            .spatial
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32)>(&track!(|phase| {
                    let pixel = dispatch_id().xy();
                    let index = core.pixel_to_linear(pixel);
                    let position = self.position(phase, index);
                    let normal = self.normal(phase, index).xyz();
                    let center = self.raw.read(index);
                    if position.w == 0.0 {
                        self.filtered.write(index, center);
                        return;
                    }
                    let sigma = keter::max(
                        0.01,
                        core.read_depth(pixel) * core.view.read().scaling * 2.0,
                    );
                    let sum = Vec3::splat(0.0f32).var();
                    let weight_sum = 0.0f32.var();
                    for dy in -2i32..=2i32 {
                        for dx in -2i32..=2i32 {
                            let q = pixel.cast_i32() + Vec2::expr(dx, dy);
                            if !(q >= 0).all() || !(q < core.screen_size().expr().cast_i32()).all()
                            {
                                continue;
                            }
                            let j = core.pixel_to_linear(q.cast_u32());
                            let other = self.position(phase, j);
                            let n = self.normal(phase, j).xyz();
                            let delta = other.xyz() - position.xyz();
                            if other.w != position.w
                                || n.dot(normal) < 0.98
                                || delta.dot(normal).abs() > keter::max(0.03, sigma * 0.1)
                            {
                                continue;
                            }
                            let weight = (-delta.length_squared() / (2.0 * sigma * sigma)).exp();
                            *sum += self.raw.read(j).xyz() * weight;
                            *weight_sum += weight;
                        }
                    }
                    self.filtered
                        .write(index, (sum / keter::max(**weight_sum, 1e-6f32)).extend(1.0));
                }))
            })
            .dispatch_async(core.screen_dispatch_size(), &phase);
        let temporal = self
            .temporal
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32, View, bool, f32)>(&track!(
                    |phase, previous, valid, max_frames| {
                        let pixel = dispatch_id().xy();
                        let index = core.pixel_to_linear(pixel);
                        let position = self.position(phase, index);
                        let normal = self.normal(phase, index).xyz();
                        let current = self.filtered.read(index).xyz();
                        let observations = 0.0f32.var();
                        let history = if valid && position.w > 0.0 {
                            let (history, age) = self.reproject(core, 1u32 - phase, previous, position, normal);
                            *observations = age;
                            history
                        } else {
                            Vec4::splat_expr(0.0f32)
                        };
                        let low = current.var();
                        let high = current.var();
                        for dy in -1i32..=1i32 {
                            for dx in -1i32..=1i32 {
                                let q = pixel.cast_i32() + Vec2::expr(dx, dy);
                                if !(q >= 0).all()
                                    || !(q < core.screen_size().expr().cast_i32()).all()
                                {
                                    continue;
                                }
                                let j = core.pixel_to_linear(q.cast_u32());
                                if self.position(phase, j).w != position.w
                                    || self.normal(phase, j).xyz().dot(normal) < 0.98
                                {
                                    continue;
                                }
                                let color = self.filtered.read(j).xyz();
                                *low = keter::min(**low, color);
                                *high = keter::max(**high, color);
                            }
                        }
                        let margin = (high - low) * 0.1;
                        let retained = history.xyz().clamp(low - margin, high + margin);
                        let count = keter::min(history.w, max_frames - 1.0);
                        let color = (current + retained * count) / (count + 1.0);
                        let age = (position.w > 0.0).select(count + 1.0, 0.0f32.expr());
                        let value = color.extend(age);
                        let confidence = (position.w > 0.0).select(keter::min(observations + 1.0, 4.0), 0.0f32.expr());
                        self.output.write(index, value);
                        if phase == 0u32 {
                            self.history[0].write(index, value);
                            self.history_age[0].write(index, confidence);
                        } else {
                            self.history[1].write(index, value);
                            self.history_age[1].write(index, confidence);
                        }
                    }
                ))
            })
            .dispatch_async(
                core.screen_dispatch_size(),
                &phase,
                &previous,
                &valid,
                &max_frames,
            );
        (spatial, temporal)
            .chain()
            .debug("Diffuse surface reconstruction")
    }
}
