//! Hybrid reflection: broad GGX integration of RC, traced glossy paths, and
//! surface/reflector-validated history. No reservoir resampling is used.
use super::{specular_brdf as brdf, *};

impl CoreData {
    /// Direct lighting at a reflection hit. Share sun visibility only when the
    /// diffuse and geometric-normal ray origins are bitwise identical.
    #[tracked]
    pub fn direct_reflection(
        &self,
        pos: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        v: Expr<Vec3<f32>>,
        n: Expr<Vec3<f32>>,
        ng: Expr<Vec3<f32>>,
        r: Expr<f32>,
        f0: Expr<Vec3<f32>>,
        albedo: Expr<Vec3<f32>>,
    ) -> Expr<Vec3<f32>> {
        let lighting = self.lighting.read();
        let diffuse = Vec3::splat(0.0f32).var();
        let specular = Vec3::splat(0.0f32).var();
        let diffuse_origin = offset_ray(pos, n);
        let specular_origin = offset_ray(pos, ng);
        let has_diffuse = albedo.reduce_max() > 0.0;
        let nl = n.dot(lighting.sun_dir);
        if nl > 0.0 && lighting.sun_color.reduce_max() > 0.0 {
            let diffuse_visible = false.var();
            if has_diffuse {
                *diffuse_visible = !self.world.occluded(diffuse_origin, lighting.sun_dir);
                *diffuse += diffuse_visible.select(lighting.sun_color, Vec3::splat_expr(0.0)) * nl;
            }
            if ng.dot(lighting.sun_dir) > 0.0 {
                let visible = if has_diffuse && (diffuse_origin == specular_origin).all() {
                    **diffuse_visible
                } else {
                    !self.world.occluded(specular_origin, lighting.sun_dir)
                };
                if visible {
                    *specular += lighting.sun_color
                        * brdf::evaluate(n, v, lighting.sun_dir, r, f0)
                        * std::f32::consts::PI;
                }
            }
        }
        if let Some(lights) = &self.point_lights {
            for index in 0..lights.len() as u32 {
                let light = lights.read(index);
                let delta = light.position_range.xyz() - pos;
                let d2 = delta.dot(delta);
                let range2 = light.position_range.w * light.position_range.w;
                if d2 > 0.0001 && d2 < range2 {
                    let distance = d2.sqrt();
                    let l = delta / distance;
                    let cosine = n.dot(l);
                    let q = d2 / range2;
                    let fade = 1.0 - q * q;
                    let attenuation = fade * fade / keter::max(d2, 100.0);
                    if has_diffuse && cosine > 0.0 && light.intensity.xyz().reduce_max() > 0.0 {
                        let visible =
                            !self
                                .world
                                .occluded_segment(diffuse_origin, l, distance - 0.01);
                        *diffuse += visible.select(light.intensity.xyz(), Vec3::splat_expr(0.0))
                            * (cosine * attenuation);
                    }
                    // Keep the existing specular segment endpoint, which differs
                    // from diffuse. Reusing this visibility would change near-light occlusion.
                    if cosine > 0.0
                        && ng.dot(l) > 0.0
                        && !self
                            .world
                            .occluded_segment(specular_origin, l, distance - 0.001)
                    {
                        *specular += light.intensity.xyz()
                            * attenuation
                            * brdf::evaluate(n, v, l, r, f0)
                            * std::f32::consts::PI;
                    }
                }
            }
        }
        for (instance, color) in &self.custom_emitters {
            if hit.inst == *instance {
                *diffuse += *color;
            }
        }
        diffuse * albedo + specular + self.get_surface_emission(hit)
    }
    #[tracked]
    pub fn direct_specular(
        &self,
        pos: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        v: Expr<Vec3<f32>>,
    ) -> Expr<Vec3<f32>> {
        self.direct_specular_roughness(pos, hit, v, self.world.get_roughness_metallic(hit).x)
    }
    #[tracked]
    fn direct_specular_roughness(
        &self,
        pos: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        v: Expr<Vec3<f32>>,
        r: Expr<f32>,
    ) -> Expr<Vec3<f32>> {
        let n = self.world.get_shading_normal(hit, -v);
        let ng = self.world.get_normal_flat(hit, -v);
        let f0 = self.world.get_f0(hit);
        let lighting = self.lighting.read();
        let result = Vec3::splat(0.0f32).var();
        // The legacy sun/point intensity is diffuse irradiance divided by pi.
        if n.dot(lighting.sun_dir) > 0.0
            && ng.dot(lighting.sun_dir) > 0.0
            && lighting.sun_color.reduce_max() > 0.0
        {
            if !self.world.occluded(offset_ray(pos, ng), lighting.sun_dir) {
                *result += lighting.sun_color
                    * brdf::evaluate(n, v, lighting.sun_dir, r, f0)
                    * std::f32::consts::PI;
            }
        }
        if let Some(lights) = &self.point_lights {
            for i in 0..lights.len() as u32 {
                let light = lights.read(i);
                let delta = light.position_range.xyz() - pos;
                let d2 = delta.dot(delta);
                let range2 = light.position_range.w * light.position_range.w;
                if d2 > 0.0001 && d2 < range2 {
                    let distance = d2.sqrt();
                    let l = delta / distance;
                    if n.dot(l) > 0.0
                        && ng.dot(l) > 0.0
                        && !self
                            .world
                            .occluded_segment(offset_ray(pos, ng), l, distance - 0.001)
                    {
                        let fade = 1.0 - (d2 / range2).powf(2.0);
                        *result += light.intensity.xyz()
                            * (fade * fade / keter::max(d2, 100.0))
                            * brdf::evaluate(n, v, l, r, f0)
                            * std::f32::consts::PI;
                    }
                }
            }
        }
        **result
    }
}

#[derive(Clone, Copy)]
pub enum SpecularTraceStage {
    All,
    Base,
    Refinement(u32),
}

pub struct SpecularData {
    pub raw: Buffer<Vec4<f32>>,
    pub output: Buffer<Vec4<f32>>,
    /// Per-pixel diagnostics: roughness, mirror/geometric cosine, accepted rays, budget.
    pub diagnostics: Buffer<Vec4<f32>>,
    deterministic: Buffer<Vec4<f32>>,
    variance: Buffer<Vec4<f32>>,
    guide_normal: Buffer<Vec4<f32>>,
    filtered: Buffer<Vec4<f32>>,
    hit_distance: Buffer<f32>,
    position: [Buffer<Vec4<f32>>; 2],
    normal: [Buffer<Vec4<f32>>; 2],
    reflector: [Buffer<Vec4<f32>>; 2],
    history: [Buffer<Vec4<f32>>; 2],
    history_age: [Buffer<f32>; 2],
    trace: [std::sync::OnceLock<Kernel<fn(u32, u32, bool, bool, bool, View, bool, bool, u32)>>; 2],
    // Pixel, target samples, completed initial samples, needs an extra pass.
    refine_pixels: Buffer<Vec4<u32>>,
    partial_sum: Buffer<Vec4<f32>>,
    partial_fixed: Buffer<Vec4<f32>>,
    partial_noise: Buffer<Vec4<f32>>,
    sample_energy: Buffer<f32>,
    resolve: std::sync::OnceLock<Kernel<fn(u32, View, bool, bool, u32)>>,
    refine_count: Singleton<u32>,
    compact_refinement: bool,
    refinement_batch: u32,
    refinement_capacity: u32,
    max_registers: u32,
    spatial: std::sync::OnceLock<Kernel<fn(u32)>>,
    temporal: std::sync::OnceLock<Kernel<fn(u32, View, bool, f32)>>,
}
impl SpecularData {
    pub fn new(core: &CoreData) -> Self {
        Self::with_refinement_queue(core, true)
    }
    pub fn with_refinement_queue(core: &CoreData, compact_refinement: bool) -> Self {
        Self::with_refinement_batch(
            core,
            compact_refinement,
            if core.num_pixels() <= 1920 * 1080 {
                1
            } else {
                32
            },
        )
    }
    pub fn with_refinement_batch(
        core: &CoreData,
        compact_refinement: bool,
        refinement_batch: u32,
    ) -> Self {
        Self::with_trace_options(core, compact_refinement, refinement_batch, 96)
    }
    pub fn with_trace_options(
        core: &CoreData,
        compact_refinement: bool,
        refinement_batch: u32,
        max_registers: u32,
    ) -> Self {
        assert!(max_registers <= 255);
        assert!(matches!(refinement_batch, 1 | 2 | 4 | 8 | 16 | 32));
        let refinement_capacity = core.num_pixels().div_ceil((4 / refinement_batch).max(1));
        let batches = (128 / refinement_batch) as usize;
        let create =
            || DEVICE.create_buffer_from_fn(core.num_pixels() as usize, |_| Vec4::splat(0.0f32));
        Self {
            raw: create(),
            output: create(),
            diagnostics: create(),
            deterministic: create(),
            variance: create(),
            guide_normal: create(),
            filtered: create(),
            hit_distance: DEVICE.create_buffer(core.num_pixels() as usize),
            position: [create(), create()],
            normal: [create(), create()],
            reflector: [create(), create()],
            history: [create(), create()],
            history_age: std::array::from_fn(|_| DEVICE.create_buffer_from_fn(core.num_pixels() as usize, |_| 0.0f32)),
            trace: Default::default(),
            refine_pixels: DEVICE.create_buffer(core.num_pixels() as usize),
            partial_sum: DEVICE.create_buffer(if compact_refinement {
                refinement_capacity as usize * batches
            } else {
                1
            }),
            partial_fixed: DEVICE.create_buffer(if compact_refinement {
                refinement_capacity as usize * batches
            } else {
                1
            }),
            partial_noise: DEVICE.create_buffer(if compact_refinement {
                refinement_capacity as usize * batches
            } else {
                1
            }),
            sample_energy: DEVICE.create_buffer(if compact_refinement {
                refinement_capacity as usize * 128
            } else {
                1
            }),
            resolve: Default::default(),
            refine_count: Singleton::new(0),
            compact_refinement,
            refinement_batch,
            refinement_capacity,
            max_registers,
            spatial: Default::default(),
            temporal: Default::default(),
        }
    }
    #[tracked]
    fn read(&self, b: &[Buffer<Vec4<f32>>; 2], phase: Expr<u32>, i: Expr<u32>) -> Expr<Vec4<f32>> {
        if phase == 0u32 {
            b[0].read(i)
        } else {
            b[1].read(i)
        }
    }
    #[tracked]
    fn write(
        &self,
        b: &[Buffer<Vec4<f32>>; 2],
        phase: Expr<u32>,
        i: Expr<u32>,
        v: Expr<Vec4<f32>>,
    ) {
        if phase == 0u32 {
            b[0].write(i, v);
        } else {
            b[1].write(i, v);
        }
    }
    #[tracked]
    pub fn cache_specular(
        core: &CoreData,
        rc: &RcData,
        pos: Expr<Vec3<f32>>,
        n: Expr<Vec3<f32>>,
        v: Expr<Vec3<f32>>,
        r: Expr<f32>,
        f0: Expr<Vec3<f32>>,
    ) -> Expr<Vec3<f32>> {
        let result = Vec3::splat(0.0f32).var();
        let norm = 0.0f32.var();
        let encoder = rc.cache.get_encoder(0u32.expr());
        rc.cache.iter_lods(pos, |lod, _, lod_weight| {
            let probes = rc.cache.quantize_trilinear(pos, lod, 0u32.expr());
            for j in 0u32..8 {
                let (p, w) = probes.get(j);
                let node = rc
                    .cache
                    .get(Probe::from_comps_expr(crate::cache::ProbeComps {
                        position: p.extend(lod),
                        cascade: 0u32.expr(),
                    }));
                if node.loaded() {
                    let weight = w * lod_weight * rc.get_quality_weight(node);
                    let sum = Vec3::splat(0.0f32).var();
                    for d in 0u32.expr()..encoder.linear_size() {
                        let l = encoder.linear_to_dir(d);
                        let radiance = rc.base_radiance.read(node.data_offset + d);
                        if n.dot(l) > 0.0 && radiance.is_finite().all() {
                            *sum += radiance
                                * brdf::evaluate(n, v, l, r, f0)
                                * (4.0 * std::f32::consts::PI / encoder.linear_size().cast_f32());
                        }
                    }
                    *result += sum * weight;
                    *norm += weight;
                }
            }
        });
        let _ = core;
        result / keter::max(norm, 0.000001)
    }
    #[tracked]
    fn reproject_history(
        &self,
        core: &CoreData,
        phase: Expr<u32>,
        i: Expr<u32>,
        previous: Expr<View>,
        valid: Expr<bool>,
    ) -> (Expr<Vec4<f32>>, Expr<f32>) {
        let pos = self.read(&self.position, phase, i);
        let n = self.read(&self.normal, phase, i);
        let history = Vec4::splat(0.0f32).var();
        let age = 0.0f32.var();
        let local = previous.transform.transpose() * (pos.xyz() - previous.pos);
        let screen = previous.project(pos.xyz()) - 0.5;
        if valid
            && pos.w > 0.0
            && local.z > 0.0
            && (screen >= 0.0).all()
            && (screen < core.screen_size().expr().cast_f32() - 1.0).all()
        {
            let base = screen.floor().cast_u32();
            let f = screen.fract();
            let support = 0.0f32.var();
            for c in 0u32..4 {
                let q = base + Vec2::expr(c & 1u32, c >> 1u32);
                let j = core.pixel_to_linear(q);
                let pp = self.read(&self.position, 1u32 - phase, j);
                let pn = self.read(&self.normal, 1u32 - phase, j);
                let fp = keter::max(0.005, local.z * previous.scaling * 2.0);
                if pp.w != pos.w
                    || (pn.w <= 0.001) != (n.w <= 0.001)
                    || self.read(&self.history, 1u32 - phase, j).w < 0.5
                    || pn.xyz().dot(n.xyz()) < 0.995
                    || (pn.w - n.w).abs() > 0.05
                    || (pp.xyz() - pos.xyz()).length() > fp
                    || (pp.xyz() - pos.xyz()).dot(n.xyz()).abs() > fp * 0.1
                {
                    continue;
                }
                // Glossy history additionally validates the reflected object
                // and its hit position. Smooth reflection motion is not the
                // same as first-surface motion.
                if n.w < 0.25 {
                    let a = self.read(&self.reflector, phase, i);
                    let b = self.read(&self.reflector, 1u32 - phase, j);
                    let distance = self.hit_distance.read(i);
                    if a.w != b.w
                        || (a.xyz() - b.xyz()).length()
                            > if n.w <= 0.001 {
                                keter::max(fp * 0.75, distance * 0.001)
                            } else {
                                keter::max(fp * 3.0, distance * (0.003 + n.w * n.w * 2.0))
                            }
                    {
                        continue;
                    }
                }
                let weight = (c & 1u32 != 0u32).select(f.x, 1.0 - f.x)
                    * (c >> 1u32 != 0u32).select(f.y, 1.0 - f.y);
                *history += self.read(&self.history, 1u32 - phase, j) * weight;
                *age += (if phase == 0u32 { self.history_age[1].read(j) }
                    else { self.history_age[0].read(j) }) * weight;
                *support += weight;
            }
            if support > 0.7 {
                *history /= support;
                *age /= support;
            } else {
                *history = Vec4::splat_expr(0.0);
                *age = 0.0;
            }
        }
        (**history, **age)
    }
    pub fn trace(
        &self,
        core: &CoreData,
        rc: &RcData,
        irradiance: Option<&super::splat::SplatData>,
        frame: u32,
        samples: u32,
        cache_valid: bool,
        reference: bool,
        regularize: bool,
        previous: View,
        history_valid: bool,
    ) -> NodeConfigs<'static> {
        self.trace_stages(
            core,
            rc,
            irradiance,
            frame,
            samples,
            cache_valid,
            reference,
            regularize,
            previous,
            history_valid,
            SpecularTraceStage::All,
        )
        .into_iter()
        .map(|(_, node)| node)
        .collect::<Vec<_>>()
        .chain()
        .debug("Hybrid reflections")
    }
    /// Expose the dispatch boundaries to the application's GPU event profiler.
    pub fn trace_stages(
        &self,
        core: &CoreData,
        rc: &RcData,
        irradiance: Option<&super::splat::SplatData>,
        frame: u32,
        samples: u32,
        cache_valid: bool,
        reference: bool,
        regularize: bool,
        previous: View,
        history_valid: bool,
        stage: SpecularTraceStage,
    ) -> Vec<(&'static str, NodeConfigs<'static>)> {
        // Compact expensive pixels, then trace independent short ray batches.
        // OptiX can migrate work between physical warps: reduction must be a
        // separate compute dispatch, never shuffle operations around traversal.
        // A second bounded pass extends uncertain estimates to 128 samples.
        let mut commands = if matches!(stage, SpecularTraceStage::Refinement(_)) {
            vec![]
        } else {
            vec![(
                "Reflection queue reset",
                self.refine_count.write_host(0).into_node_configs(),
            )]
        };
        let groups = core.num_pixels().div_ceil(self.refinement_capacity);
        let begin = if matches!(stage, SpecularTraceStage::Refinement(_)) {
            1
        } else {
            0
        };
        let active_work = if let SpecularTraceStage::Refinement(count) = stage {
            assert!(count <= core.num_pixels());
            count
        } else {
            core.num_pixels()
        };
        for pass in begin
            ..if !matches!(stage, SpecularTraceStage::Base)
                && self.compact_refinement
                && samples < 128
            {
                1 + groups * 2
            } else {
                1
            }
        {
            let refine = pass != 0;
            let extra = refine && (pass - 1) % 2 == 1;
            let group = if refine { (pass - 1) / 2 } else { 0 };
            let work_offset = group * self.refinement_capacity;
            let work_size = active_work
                .saturating_sub(work_offset)
                .min(self.refinement_capacity);
            let ray_threads = work_size * (128 / self.refinement_batch);
            let dispatch = if refine {
                [ray_threads, 1, 1]
            } else {
                core.screen_dispatch_size()
            };
            let command = if refine && work_size == 0 {
                NodeConfigs::default()
            } else {
                self.trace[usize::from(refine)]
                .get_or_init(|| {
                    DEVICE.create_kernel_with_options::<fn(u32, u32, bool, bool, bool, View, bool, bool, u32)>(
                        KernelBuildOptions {max_registers:if refine {self.max_registers} else {0},..Default::default()},
                        &track!(|frame,
                                 samples,
                                 cache_valid,
                                 reference,
                                 regularize,
                                 previous,
                                 history_valid,
                                 extra,
                                 work_offset| {
                            let available = keter::min(
                                keter::max(self.refine_count.read(), work_offset) - work_offset,
                                self.refinement_capacity,
                            );
                            let jobs = available * (128 / self.refinement_batch);
                            let iterations = 1u32.expr();
                            for _iteration in 0u32.expr()..iterations {
                                let task = dispatch_id().x;
                                if refine && task >= jobs {
                                    break;
                                }
                                let local_index = task / (128 / self.refinement_batch);
                                let work_index = work_offset + local_index;
                                let batch = task % (128 / self.refinement_batch);
                                let work = if refine {
                                    self.refine_pixels.read(work_index)
                                } else {
                                    Vec4::splat_expr(0u32)
                                };
                                let sample_start = if refine {
                                    batch * self.refinement_batch
                                } else {
                                    0u32.expr()
                                };
                                if refine
                                    && (sample_start >= work.y
                                        || (extra && (work.w == 0u32 || sample_start < work.z)))
                                {
                                    continue;
                                }
                                let pixel = if refine {
                                    Vec2::expr(
                                        work.x % core.screen_size().x,
                                        work.x / core.screen_size().x,
                                    )
                                } else {
                                    dispatch_id().xy()
                                };
                                let i = core.pixel_to_linear(pixel);
                                let phase = frame % 2u32;
                                let zero = Vec4::splat_expr(0.0f32);
                                if !refine {
                                    self.raw.write(i, zero);
                                    self.diagnostics.write(i, zero);
                                    self.deterministic.write(i, zero);
                                    self.variance.write(i, zero);
                                    self.guide_normal.write(i, zero);
                                    self.hit_distance.write(i, 0.0);
                                    self.write(&self.position, phase, i, zero);
                                    self.write(&self.normal, phase, i, zero);
                                    self.write(&self.reflector, phase, i, zero);
                                }
                                if core.read_depth(pixel) == 0.0 {
                                    continue;
                                }
                                let hit = core.hit.read(i);
                                let pos = core.get_world_pos(pixel);
                                let n = core.read_normal(pixel);
                                let v = (core.view.read().pos - pos).normalize();
                                let ng = core.world.get_normal_flat(hit, -v);
                                let (roughness, f0, _) = core.world.get_pbr_lobes(hit);
                                let stochastic = (roughness > 0.001).var();
                                let random_path = (roughness > 0.001).var();
                                if !refine {
                                    self.write(
                                        &self.position,
                                        phase,
                                        i,
                                        pos.extend((hit.inst + 1u32).cast_f32()),
                                    );
                                    self.write(&self.normal, phase, i, n.extend(roughness));
                                }
                                let origin = offset_ray(pos, ng);
                                let guide = hit.var();
                                let guide_distance = 0.0f32.var();
                                if roughness <= 0.001 || (!refine && roughness < 0.25) {
                                    let mirror = n * (2.0 * n.dot(v)) - v;
                                    *guide = core.world.trace(origin, mirror);
                                    let point = if guide.miss() {
                                        mirror * 10000.0
                                    } else {
                                        core.world.get_position(**guide)
                                    };
                                    let id = guide
                                        .miss()
                                        .select(0.0f32.expr(), (guide.inst + 1u32).cast_f32());
                                    if !refine {
                                        self.write(&self.reflector, phase, i, point.extend(id));
                                    }
                                    if !refine && !guide.miss() {
                                        self.guide_normal.write(
                                            i,
                                            core.world.get_shading_normal(**guide, mirror).extend(
                                                core.world.get_roughness_metallic(**guide).x,
                                            ),
                                        );
                                    }
                                    *guide_distance = guide
                                        .miss()
                                        .select(10000.0f32.expr(), **guide.committed_ray_t);
                                }
                                if !refine {
                                    self.hit_distance.write(i, **guide_distance);
                                }
                                // Use exactly the reconstruction's validity test. Pixels
                                // without history must resolve current lighting themselves.
                                let observations = 0.0f32.var();
                                let prior = if refine {
                                    Vec4::splat_expr(0.0)
                                } else {
                                    let (history, age) = self.reproject_history(core, phase, i, previous, history_valid);
                                    *observations = age;
                                    history
                                };
                                let disoccluded = prior.w < 0.5;
                                // Cache use fades in only for broad lobes. First segment visibility
                                // remains traced even on rough surfaces; only the far segment is cached.
                                let mix = if reference || !cache_valid {
                                    0.0f32.expr()
                                } else {
                                    ((roughness - 0.45) / 0.15).clamp(0., 1.)
                                };
                                let sum = Vec3::splat(0.0f32).var();
                                let fixed_sum = Vec3::splat(0.0f32).var();
                                let noise_squared = Vec3::splat(0.0f32).var();
                                let mirror_base = Vec3::splat(0.0f32).var();
                                let mirror_fixed = Vec3::splat(0.0f32).var();
                                let distances = 0.0f32.var();
                                // Thin features and grazing glossy surfaces cannot borrow
                                // enough spatial support. Spend samples there instead of
                                // blurring across different objects to hide their variance.
                                let neighbors = 0u32.var();
                                if !refine {
                                    for y in -1i32..=1 {
                                        for x in -1i32..=1 {
                                            let q = pixel.cast_i32() + Vec2::expr(x, y);
                                            if !(q >= 0).all()
                                                || !(q < core.screen_size().expr().cast_i32()).all()
                                            {
                                                continue;
                                            }
                                            let q = q.cast_u32();
                                            if core.read_depth(q) > 0.0
                                                && core.hit.read(core.pixel_to_linear(q)).inst
                                                    == hit.inst
                                                && core.read_normal(q).dot(n) > 0.985
                                            {
                                                *neighbors += 1u32;
                                            }
                                        }
                                    }
                                }
                                let sample_count = if roughness > 0.001
                                    && roughness < 0.45
                                    && f0.reduce_max() > 0.2
                                {
                                    keter::max(
                                        samples,
                                        if observations >= 3.999 && neighbors >= 6u32 {
                                            16u32.expr()
                                        } else {
                                            64u32.expr()
                                        },
                                    )
                                } else if roughness > 0.001 && (neighbors < 6u32 || n.dot(v) < 0.2)
                                {
                                    keter::max(samples, keter::min(samples * 2u32, 32u32))
                                } else {
                                    if roughness > 0.001 {
                                        keter::max(samples, 4u32)
                                    } else {
                                        samples
                                    }
                                };
                                let budget = sample_count.var();
                                if roughness > 0.001 && disoccluded {
                                    *budget = keter::max(
                                        samples,
                                        if f0.reduce_max() > 0.2 {
                                            128u32.expr()
                                        } else {
                                            keter::min(samples * 8u32, 128u32)
                                        },
                                    );
                                }
                                // Repeating the exact same deterministic mirror ray adds
                                // no information. A stochastic continuation raises this below.
                                if roughness <= 0.001 {
                                    *budget = 1u32;
                                }
                                if refine {
                                    *budget = work.y;
                                }
                                let peak = 0.0f32.var();
                                let total_energy = 0.0f32.var();
                                let accepted = 0u32.var();
                                let sample_end = if refine {
                                    sample_start + self.refinement_batch
                                } else {
                                    keter::max(samples, 128u32)
                                };
                                for s in sample_start..sample_end {
                                    // Adaptive batches round up equally in serial and queued
                                    // execution, including non-power-of-two requested budgets.
                                    if budget > keter::max(samples, 16u32) {
                                        *budget = ((budget + 31u32) / 32u32) * 32u32;
                                    }
                                    if refine {
                                        self.sample_energy.write(local_index * 128u32 + s, 0.0);
                                    }
                                    if self.compact_refinement
                                        && !refine
                                        && budget > keter::max(samples, 16u32)
                                    {
                                        let index = self.refine_count.atomic().fetch_add(1u32);
                                        self.refine_pixels.write(
                                            index,
                                            Vec4::expr(i, **budget, **budget, 0u32.expr()),
                                        );
                                        return;
                                    }
                                    if s >= budget {
                                        break;
                                    }
                                    let path_sample = brdf::path_samples(pixel, frame, s);
                                    let u = path_sample.xy();
                                    let (l, weight) = brdf::sample(n, v, roughness, f0, u);
                                    if l.dot(ng) > 0.0 && weight.reduce_max() > 0.0 {
                                        *accepted += 1u32;
                                        let secondary = if roughness <= 0.001 {
                                            **guide
                                        } else {
                                            core.world.trace(origin, l)
                                        };
                                        let distance = secondary
                                            .miss()
                                            .select(10000.0f32.expr(), secondary.committed_ray_t);
                                        let local_radius = rc.base_grid_size
                                            * 2.0
                                            * (1u32 << rc.cache.estimate_lod(pos)).cast_f32();
                                        let use_cache = mix > 0.0 && distance > local_radius;
                                        let cache = Vec3::splat(0.0f32).var();
                                        let usable = false.var();
                                        if use_cache {
                                            *cache =
                                                rc.sample(pos, (rc.num_cascades - 1).expr(), l);
                                            for c in (0..rc.num_cascades - 1).rev() {
                                                *cache = super::rc::over(
                                                    rc.cache.sample(pos, c.expr(), l),
                                                    **cache,
                                                );
                                            }
                                            *usable = cache.is_finite().all();
                                        }
                                        let fixed_li = Vec3::splat(0.0f32).var();
                                        let incoming = if **usable && mix >= 0.9999 {
                                            *fixed_li = **cache;
                                            **cache
                                        } else if secondary.miss() {
                                            *fixed_li = core.get_environment(l);
                                            **fixed_li
                                        } else {
                                            let p = core.world.get_position(secondary);
                                            let nr = core.world.get_shading_normal(secondary, l);
                                            let gn = core.world.get_normal_flat(secondary, l);
                                            let (material_r, secondary_f0, albedo) =
                                                core.world.get_pbr_lobes(secondary);
                                            // Path regularization limits unresolved near-delta
                                            // sun connections after a rough bounce. This is a
                                            // deliberate bias in indirect caustics, never a
                                            // radiance clamp; pure mirror chains stay unchanged.
                                            let r = if regularize
                                                && roughness > 0.001
                                                && material_r * material_r < 0.3
                                            {
                                                // PBRT-style GGX path regularization, applied
                                                // only after non-perfect scattering.
                                                (2.0 * material_r * material_r)
                                                    .clamp(0.1, 0.3)
                                                    .sqrt()
                                            } else {
                                                material_r
                                            };
                                            if r > 0.001 && r < 0.45 {
                                                *stochastic = true;
                                                *random_path = true;
                                                if !refine && roughness <= 0.001 {
                                                    // A perfect first bounce cannot spatially blur
                                                    // away noise introduced at the reflected surface.
                                                    // Give that glossy continuation its own budget.
                                                    *budget = keter::max(
                                                        **budget,
                                                        keter::max(
                                                            samples,
                                                            disoccluded.select(
                                                                128u32.expr(),
                                                                64u32.expr(),
                                                            ),
                                                        ),
                                                    );
                                                } else if !refine && neighbors < 6u32 {
                                                    *budget = keter::max(**budget, 32u32);
                                                }
                                            }
                                            // RC is an estimator even when both reflection
                                            // directions are deterministic. Keep its energy out
                                            // of the unfiltered mirror-detail component.
                                            if cache_valid {
                                                *stochastic = true;
                                            }
                                            let stable = if roughness <= 0.001 && s > sample_start {
                                                *fixed_li = **mirror_fixed;
                                                **mirror_base
                                            } else {
                                                let cached_diffuse =
                                                    if cache_valid && albedo.reduce_max() > 0.0 {
                                                        (if let Some(splat) = irradiance {
                                                            splat.sample_secondary(rc, p, nr)
                                                        } else {
                                                            rc.sample_secondary(p, nr)
                                                        }) * albedo
                                                    } else {
                                                        Vec3::splat_expr(0.0)
                                                    };
                                                let fixed = core.direct_reflection(
                                                    p,
                                                    secondary,
                                                    -l,
                                                    nr,
                                                    gn,
                                                    r,
                                                    secondary_f0,
                                                    albedo,
                                                );
                                                let value = fixed + cached_diffuse;
                                                *fixed_li = fixed;
                                                if roughness <= 0.001 {
                                                    *mirror_base = value;
                                                    *mirror_fixed = fixed;
                                                }
                                                value
                                            };
                                            let extra = Vec3::splat(0.0f32).var();
                                            // One additional glossy bounce captures reflections in metal
                                            // seen inside the primary reflection. Further transport ends
                                            // in diffuse RC; this is a finite-depth hybrid, not full PT.
                                            // A rough first bounce already integrates directions.
                                            // Sample its continuation once per path instead of
                                            // running a nested 32-direction RC integral per ray.
                                            // Exact first mirrors still use the deterministic RC
                                            // integral for a broad secondary lobe.
                                            if r < 0.45 || roughness > 0.001 {
                                                let u2 = Vec2::expr(path_sample.z, path_sample.w);
                                                let (l2, w2) =
                                                    brdf::sample(nr, -l, r, secondary_f0, u2);
                                                if l2.dot(gn) > 0.0 {
                                                    let p2 = offset_ray(p, gn);
                                                    let h2 = core.world.trace(p2, l2);
                                                    let albedo2 = if h2.miss() {
                                                        Vec3::splat_expr(0.0)
                                                    } else {
                                                        core.world.get_diffuse(h2)
                                                    };
                                                    let direct = if h2.miss() {
                                                        core.get_environment(l2)
                                                    } else if albedo2.reduce_max() > 0.0 {
                                                        core.get_radiance(None, p2, l2, h2)
                                                    } else {
                                                        core.get_surface_emission(h2)
                                                    };
                                                    let li = direct.var();
                                                    if r <= 0.001 {
                                                        *fixed_li += w2 * direct;
                                                    }
                                                    if cache_valid
                                                        && !h2.miss()
                                                        && albedo2.reduce_max() > 0.0
                                                    {
                                                        let p3 = core.world.get_position(h2);
                                                        let n3 =
                                                            core.world.get_shading_normal(h2, l2);
                                                        *li += (if let Some(splat) = irradiance {
                                                            splat.sample_secondary(rc, p3, n3)
                                                        } else {
                                                            rc.sample_secondary(p3, n3)
                                                        }) * albedo2;
                                                    }
                                                    *extra = w2 * li;
                                                }
                                            } else if cache_valid {
                                                *extra = Self::cache_specular(
                                                    core,
                                                    rc,
                                                    p,
                                                    nr,
                                                    -l,
                                                    r,
                                                    secondary_f0,
                                                );
                                            }
                                            stable + extra
                                        };
                                        // Cache is only substituted beyond the local probe footprint.
                                        // The hit remains available to reject wrong reflector histories.
                                        let contribution = weight
                                            * if usable {
                                                incoming * (1.0 - mix) + cache * mix
                                            } else {
                                                incoming
                                            };
                                        *sum += contribution;
                                        let fixed_contribution = if roughness <= 0.001 {
                                            weight * fixed_li
                                        } else {
                                            Vec3::splat_expr(0.0)
                                        };
                                        let noise = contribution - fixed_contribution;
                                        *noise_squared += noise * noise;
                                        if roughness <= 0.001 {
                                            *fixed_sum += weight * fixed_li;
                                        }
                                        *peak = keter::max(**peak, contribution.reduce_max());
                                        *total_energy += contribution.reduce_max();
                                        if refine {
                                            self.sample_energy.write(
                                                local_index * 128u32 + s,
                                                contribution.reduce_max(),
                                            );
                                        }
                                        *distances += distance;
                                    }
                                    // A rare high-energy path needs more evidence, rather
                                    // than a radiance clamp that would erase real lighting.
                                    // This bounded adaptive estimator is still a biased
                                    // reconstruction, not an unbiased reference integrator.
                                    if !refine
                                        && s >= 3u32
                                        && peak > 0.25
                                        && peak > 8.0 * (total_energy - peak) / s.cast_f32()
                                    {
                                        *budget = keter::max(**budget, 128u32);
                                    }
                                    // A current estimate that disagrees with a valid
                                    // reprojected history needs more rays, not a brightness
                                    // floor or additional temporal lag. Genuine illumination
                                    // changes are still free to become dark after sampling.
                                    if !refine
                                        && s + 1u32 >= budget
                                        && budget < 128u32
                                        && roughness > 0.001
                                        && prior.w > 0.0
                                        && (sum / (s + 1u32).cast_f32() - prior.xyz())
                                            .abs()
                                            .reduce_max()
                                            > keter::max(0.02, prior.xyz().reduce_max() * 0.25)
                                    {
                                        *budget = 128u32;
                                    }
                                    // Zero from a handful of paths is not evidence of a
                                    // dark newly-visible metal surface. Resolve uncertain
                                    // zero estimates with a bounded high-sample fallback.
                                    if !refine
                                        && s + 1u32 >= budget
                                        && budget < 128u32
                                        && **random_path
                                        && f0.reduce_max() > 0.2
                                        && sum.reduce_max() <= 1e-8
                                    {
                                        *budget = 128u32;
                                    }
                                }
                                if refine {
                                    let partial =
                                        local_index * (128 / self.refinement_batch) + batch;
                                    self.partial_sum.write(partial, sum.extend(**distances));
                                    self.partial_fixed
                                        .write(partial, fixed_sum.extend(accepted.cast_f32()));
                                    self.partial_noise.write(
                                        partial,
                                        noise_squared.extend(
                                            stochastic.select(1.0f32.expr(), 0.0f32.expr())
                                                + random_path.select(2.0f32.expr(), 0.0f32.expr()),
                                        ),
                                    );
                                    continue;
                                }
                                let result = sum / budget.cast_f32();
                                let noise_mean = (sum - fixed_sum) / budget.cast_f32();
                                let variance = keter::max(
                                    noise_squared / budget.cast_f32() - noise_mean * noise_mean,
                                    Vec3::splat_expr(0.0),
                                ) / budget.cast_f32();
                                self.variance.write(i, variance.extend(1.0));
                                self.deterministic
                                    .write(i, (fixed_sum / budget.cast_f32()).extend(1.0));
                                self.diagnostics.write(
                                    i,
                                    Vec4::expr(
                                        roughness,
                                        (n * (2.0 * n.dot(v)) - v).dot(ng),
                                        accepted.cast_f32(),
                                        budget.cast_f32(),
                                    ),
                                );
                                // Alpha marks estimated energy, not surface coverage:
                                // exact mirror directions may still read noisy RC or
                                // have a stochastic glossy continuation at the next hit.
                                self.raw.write(
                                    i,
                                    result.extend(stochastic.select(1.0f32.expr(), 0.0f32.expr())),
                                );
                                self.hit_distance.write(
                                    i,
                                    if roughness < 0.25 {
                                        **guide_distance
                                    } else {
                                        distances / budget.cast_f32()
                                    },
                                );
                            }
                        }),
                    )
                })
                .dispatch_async(
                    dispatch,
                    &frame,
                    &samples,
                    &cache_valid,
                    &reference,
                    &regularize,
                    &previous,
                    &history_valid,
                    &extra,
                    &work_offset,
                )
                .debug(if refine {
                    "Hybrid reflection refinement"
                } else {
                    "Hybrid reflection base"
                })
            };
            commands.push((
                if !refine {
                    "Reflection base"
                } else if extra {
                    [
                        "Reflection extra rays 0",
                        "Reflection extra rays 1",
                        "Reflection extra rays 2",
                        "Reflection extra rays 3",
                    ][group as usize]
                } else {
                    [
                        "Reflection refinement rays 0",
                        "Reflection refinement rays 1",
                        "Reflection refinement rays 2",
                        "Reflection refinement rays 3",
                    ][group as usize]
                },
                command,
            ));
            if refine {
                commands.push((
                    if extra {
                        [
                            "Reflection extra resolve 0",
                            "Reflection extra resolve 1",
                            "Reflection extra resolve 2",
                            "Reflection extra resolve 3",
                        ][group as usize]
                    } else {
                        [
                            "Reflection initial resolve 0",
                            "Reflection initial resolve 1",
                            "Reflection initial resolve 2",
                            "Reflection initial resolve 3",
                        ][group as usize]
                    },
                    self.resolve_refinement(
                        core,
                        frame % 2,
                        previous,
                        history_valid,
                        extra,
                        work_offset,
                        work_size,
                    ),
                ));
            }
        }
        commands
    }
    pub fn pending_refinement_count(&self) -> u32 {
        self.refine_count.read_blocking()
    }
    fn resolve_refinement(
        &self,
        core: &CoreData,
        phase: u32,
        previous: View,
        history_valid: bool,
        extra: bool,
        work_offset: u32,
        work_size: u32,
    ) -> NodeConfigs<'static> {
        if work_size == 0 {
            return NodeConfigs::default();
        }
        self.resolve
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32, View, bool, bool, u32)>(&track!(
                    |phase, previous, history_valid, extra, work_offset| {
                        let local_index = dispatch_id().x;
                        let index = local_index + work_offset;
                        if index >= self.refine_count.read() {
                            return;
                        }
                        let work = self.refine_pixels.read(index);
                        if extra && work.w == 0u32 {
                            return;
                        }
                        let i = work.x;
                        let budget = work.y;
                        let sum = Vec4::splat(0.0f32).var();
                        let fixed = Vec4::splat(0.0f32).var();
                        let noise = Vec3::splat(0.0f32).var();
                        let flags = 0.0f32.var();
                        for batch in 0u32.expr()..budget / self.refinement_batch {
                            let partial = local_index * (128 / self.refinement_batch) + batch;
                            *sum += self.partial_sum.read(partial);
                            *fixed += self.partial_fixed.read(partial);
                            let value = self.partial_noise.read(partial);
                            *noise += value.xyz();
                            *flags = keter::max(**flags, value.w);
                        }
                        let n = self.read(&self.normal, phase, i);
                        let result = sum.xyz() / budget.cast_f32();
                        if !extra && budget < 128u32 {
                            let peak = 0.0f32.var();
                            let energy = 0.0f32.var();
                            let extend = false.var();
                            for sample in 0u32.expr()..budget {
                                let value = self.sample_energy.read(local_index * 128u32 + sample);
                                *peak = keter::max(**peak, value);
                                *energy += value;
                                if sample >= 3u32
                                    && peak > 0.25
                                    && peak > 8.0 * (energy - peak) / sample.cast_f32()
                                {
                                    *extend = true;
                                }
                            }
                            let prior =
                                self.reproject_history(core, phase, i, previous, history_valid).0;
                            if n.w > 0.001
                                && prior.w > 0.0
                                && (result - prior.xyz()).abs().reduce_max()
                                    > keter::max(0.02, prior.xyz().reduce_max() * 0.25)
                            {
                                *extend = true;
                            }
                            if flags > 1.5
                                && core.world.get_f0(core.hit.read(i)).reduce_max() > 0.2
                                && sum.xyz().reduce_max() <= 1e-8
                            {
                                *extend = true;
                            }
                            if extend {
                                self.refine_pixels.write(
                                    index,
                                    Vec4::expr(i, 128u32.expr(), budget, 1u32.expr()),
                                );
                                return;
                            }
                        }
                        let mean = (sum.xyz() - fixed.xyz()) / budget.cast_f32();
                        let variance = keter::max(
                            noise / budget.cast_f32() - mean * mean,
                            Vec3::splat_expr(0.0),
                        ) / budget.cast_f32();
                        self.variance.write(i, variance.extend(1.0));
                        self.deterministic
                            .write(i, (fixed.xyz() / budget.cast_f32()).extend(1.0));
                        let pos = self.read(&self.position, phase, i).xyz();
                        let v = (core.view.read().pos - pos).normalize();
                        let ng = core.world.get_normal_flat(core.hit.read(i), -v);
                        self.diagnostics.write(
                            i,
                            Vec4::expr(
                                n.w,
                                (n.xyz() * (2.0 * n.xyz().dot(v)) - v).dot(ng),
                                fixed.w,
                                budget.cast_f32(),
                            ),
                        );
                        self.raw.write(
                            i,
                            result.extend((flags > 0.5).select(1.0f32.expr(), 0.0f32.expr())),
                        );
                        if n.w >= 0.25 {
                            self.hit_distance.write(i, sum.w / budget.cast_f32());
                        }
                    }
                ))
            })
            .dispatch_async(
                [work_size, 1, 1],
                &phase,
                &previous,
                &history_valid,
                &extra,
                &work_offset,
            )
            .debug("Reflection batch reduction")
            .into_node_configs()
    }
    pub fn reconstruct(
        &self,
        core: &CoreData,
        frame: u32,
        previous: View,
        valid: bool,
        max_frames: f32,
    ) -> NodeConfigs<'static> {
        let phase = frame % 2;
        let spatial = self
            .spatial
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32)>(&track!(|phase| {
                    let p = dispatch_id().xy();
                    let i = core.pixel_to_linear(p);
                    let raw = self.raw.read(i);
                    let center = keter::max(
                        raw.xyz() - self.deterministic.read(i).xyz(),
                        Vec3::splat_expr(0.0),
                    )
                    .extend(raw.w);
                    let pos = self.read(&self.position, phase, i);
                    let n = self.read(&self.normal, phase, i);
                    if pos.w == 0.0
                        || (n.w < 0.08 && (n.w > 0.001 || raw.w < 0.5))
                        || (n.w <= 0.001 && self.diagnostics.read(i).w < 1.5)
                    {
                        self.filtered.write(i, center);
                        return;
                    }
                    // Robust radiance weights prevent one rare, very bright path
                    // from contaminating the whole spatial footprint. Uniform
                    // highlights retain their value because weights normalize.
                    let center_weight = 1.0 / (1.0 + center.xyz().reduce_max());
                    let sum = (center.xyz() * center_weight).var();
                    let total = center_weight.var();
                    let radius = if n.w > 0.4 { 2i32.expr() } else { 1i32.expr() };
                    for y in -2i32..=2 {
                        for x in -2i32..=2 {
                            if (x == 0 && y == 0) || x.abs() > radius || y.abs() > radius {
                                continue;
                            }
                            let q = p.cast_i32() + Vec2::expr(x, y);
                            if !(q >= 0).all() || !(q < core.screen_size().expr().cast_i32()).all()
                            {
                                continue;
                            }
                            let j = core.pixel_to_linear(q.cast_u32());
                            let pn = self.read(&self.position, phase, j);
                            let nn = self.read(&self.normal, phase, j);
                            let footprint = keter::max(
                                0.005,
                                core.read_depth(p) * core.view.read().scaling * 3.0,
                            );
                            if pn.w != pos.w
                                || (nn.w <= 0.001) != (n.w <= 0.001)
                                || self.raw.read(j).w < 0.5
                                || (nn.w - n.w).abs() > 0.08
                                || nn.xyz().dot(n.xyz()) < 0.985
                                || (pn.xyz() - pos.xyz()).dot(n.xyz()).abs() > footprint * 0.15
                            {
                                continue;
                            }
                            if n.w <= 0.001 {
                                let a = self.read(&self.reflector, phase, i);
                                let b = self.read(&self.reflector, phase, j);
                                let an = self.guide_normal.read(i);
                                let bn = self.guide_normal.read(j);
                                if a.w != b.w
                                    || an.xyz().dot(bn.xyz())
                                        < 1.0 - keter::max(1e-6, an.w.powf(4.0) * 0.5)
                                    || (an.w - bn.w).abs() > 0.08
                                {
                                    continue;
                                }
                                let reflected_footprint = keter::max(
                                    0.005,
                                    (core.read_depth(p) + self.hit_distance.read(i))
                                        * core.view.read().scaling
                                        * 3.0,
                                );
                                if (a.xyz() - b.xyz()).dot(an.xyz()).abs()
                                    > reflected_footprint * 0.15
                                {
                                    continue;
                                }
                            }
                            let d = self.hit_distance.read(i);
                            let dj = self.hit_distance.read(j);
                            let dw = (-(d - dj).abs() / keter::max(1.0, d * 0.25)).exp();
                            let value = keter::max(
                                self.raw.read(j).xyz() - self.deterministic.read(j).xyz(),
                                Vec3::splat_expr(0.0),
                            );
                            let weight = (-(x * x + y * y).cast_f32() * 0.5).exp() * dw
                                / (1.0 + value.reduce_max());
                            *sum += value * weight;
                            *total += weight;
                        }
                    }
                    self.filtered.write(i, (sum / total).extend(1.0));
                }))
            })
            .dispatch_async(core.screen_dispatch_size(), &phase);
        let temporal = self
            .temporal
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32, View, bool, f32)>(&track!(
                    |phase, previous, valid, max_frames| {
                        let p = dispatch_id().xy();
                        let i = core.pixel_to_linear(p);
                        let current = self.filtered.read(i);
                        let pos = self.read(&self.position, phase, i);
                        let n = self.read(&self.normal, phase, i);
                        if pos.w == 0.0 || self.raw.read(i).w < 0.5 {
                            self.write(&self.history, phase, i, Vec4::splat_expr(0.0));
                            if phase == 0u32 { self.history_age[0].write(i, 0.0); }
                            else { self.history_age[1].write(i, 0.0); }
                            self.output.write(i, self.deterministic.read(i));
                            return;
                        }
                        let (history, age) = self.reproject_history(core, phase, i, previous, valid);
                        // Clip stale history to current same-surface support.
                        // This also lets real moving illumination leave promptly.
                        let low = current.xyz().var();
                        let high = current.xyz().var();
                        let clipping_support = 0u32.var();
                        for y in -1i32..=1 {
                            for x in -1i32..=1 {
                                let q = p.cast_i32() + Vec2::expr(x, y);
                                if !(q >= 0).all()
                                    || !(q < core.screen_size().expr().cast_i32()).all()
                                {
                                    continue;
                                }
                                let j = core.pixel_to_linear(q.cast_u32());
                                let pp = self.read(&self.position, phase, j);
                                let nn = self.read(&self.normal, phase, j);
                                if pp.w != pos.w
                                    || (nn.w <= 0.001) != (n.w <= 0.001)
                                    || self.raw.read(j).w < 0.5
                                    || nn.xyz().dot(n.xyz()) < 0.995
                                    || (nn.w - n.w).abs() > 0.05
                                {
                                    continue;
                                }
                                if n.w < 0.25
                                    && self.read(&self.reflector, phase, j).w
                                        != self.read(&self.reflector, phase, i).w
                                {
                                    continue;
                                }
                                let value = self.filtered.read(j).xyz();
                                *clipping_support += 1u32;
                                *low = keter::min(**low, value);
                                *high = keter::max(**high, value);
                            }
                        }
                        let margin = (high - low) * 0.1;
                        let retained = if clipping_support >= 6u32
                            || (n.w <= 0.001 && self.raw.read(i).w < 0.5)
                        {
                            history.xyz().clamp(low - margin, high + margin)
                        } else {
                            // A single current sample is not a variance bound.
                            // Clamping to it would silently disable accumulation
                            // on thin metallic details and cause black/bright pops.
                            history.xyz()
                        };
                        // RC already filters its cached estimate. Exact rays that
                        // only read RC need a prompt response and no extra spatial
                        // blur across reflected texture/visibility boundaries.
                        let history_limit = if n.w <= 0.001 && self.diagnostics.read(i).w < 1.5 {
                            keter::min(max_frames, 1.25)
                        } else {
                            max_frames
                        };
                        let count = keter::min(history.w + 1.0, history_limit);
                        let output = if pos.w == 0.0 {
                            Vec4::splat_expr(0.0)
                        } else {
                            let blended =
                                (retained * (count - 1.0) + current.xyz()) / keter::max(count, 1.0);
                            // Bound only the stochastic component using measured
                            // standard error. Deterministic mirror detail is added
                            // unchanged below; neighbors cannot overwhelm a precise
                            // current estimate with stale or unrelated radiance.
                            let mean = keter::max(
                                self.raw.read(i).xyz() - self.deterministic.read(i).xyz(),
                                Vec3::splat_expr(0.0),
                            );
                            let error =
                                self.variance.read(i).xyz().sqrt() * 3.0 + mean * 0.1 + 0.002;
                            let bounded = blended.clamp(
                                keter::max(mean - error, Vec3::splat_expr(0.0)),
                                mean + error,
                            );
                            if self.diagnostics.read(i).w >= 16.0
                                && (mean.reduce_max() > 0.001
                                    || self.diagnostics.read(i).w >= 128.0)
                            {
                                bounded.extend(count)
                            } else {
                                blended.extend(count)
                            }
                        };
                        self.write(&self.history, phase, i, output);
                        let confidence = keter::min(age + 1.0, 4.0);
                        if phase == 0u32 { self.history_age[0].write(i, confidence); }
                        else { self.history_age[1].write(i, confidence); }
                        self.output.write(
                            i,
                            (output.xyz() + self.deterministic.read(i).xyz()).extend(output.w),
                        );
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
            .debug("Reflection reconstruction")
    }
}
