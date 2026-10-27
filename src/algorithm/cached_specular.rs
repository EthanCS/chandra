//! Bounded-cost reflection approximation: four GGX visibility rays per glossy
//! reduced-resolution pixel; sharp mirrors and uncovered edges get exact rays.
//! A mirror may continue once before the remaining transport terminates in RC.
//! Broad lobes use a probe-space GGX convolution and a split-sum BRDF table.
//! This is deliberately biased; the full GGX path remains a quality reference.
use keter::lang::functions::sync_block;
use yesod::direction::OctahedralEncoder;

use super::splat::SplatData;
use super::{specular_brdf as brdf, *};

const RES: u32 = 6;
const TILE: u32 = RES + 2;
const LEVELS: u32 = 3;

pub struct CachedSpecular {
    pub raw: Buffer<Vec4<f32>>,
    pub output: Buffer<Vec4<f32>>,
    pub diagnostics: Buffer<Vec4<f32>>,
    pub scale: u32,
    size: Vec2<u32>,
    spp: u32,
    atlas: Tex2dHandle<Vec4<f32>>,
    atlas_dims: Vec2<u32>,
    weights: Buffer<f32>,
    brdf: Tex2dHandle<Vec4<f32>>,
    signal: Buffer<Vec4<f32>>,
    filtered: Buffer<Vec4<f32>>,
    positions: [Buffer<Vec4<f32>>; 2],
    normals: [Buffer<Vec4<f32>>; 2],
    reflectors: [Buffer<Vec4<f32>>; 2],
    history: [Buffer<Vec4<f32>>; 2],
    full_positions: [Buffer<Vec4<f32>>; 2],
    full_normals: [Buffer<Vec4<f32>>; 2],
    prefilter_kernel: std::sync::OnceLock<Kernel<fn()>>,
    trace_kernel: std::sync::OnceLock<Kernel<fn(u32, bool, u32)>>,
    spatial_kernel: std::sync::OnceLock<Kernel<fn(u32)>>,
    temporal_kernel: std::sync::OnceLock<Kernel<fn(u32, View, bool, f32)>>,
    upscale_kernel: std::sync::OnceLock<Kernel<fn(u32)>>,
}

impl CachedSpecular {
    pub fn new(core: &CoreData, rc: &RcData, scale: u32, spp: u32) -> Self {
        assert!(matches!(scale, 1 | 2 | 4));
        assert!((1..=8).contains(&spp));
        assert!(rc.increased_base_resolution && rc.base_angular_resolution == 4);
        let size = Vec2::new(
            core.screen_size().x.div_ceil(scale),
            core.screen_size().y.div_ceil(scale),
        );
        let x = 1 << (rc.cache.base_probes.ilog2() / 2);
        let atlas_dims = Vec2::new(x, rc.cache.base_probes.div_ceil(x));
        let texture = |w, h| {
            BINDLESS.push_tex2d(
                DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, w, h, 1),
                Sampler {
                    filter: SamplerFilter::LinearLinear,
                    address: SamplerAddress::Edge,
                },
            )
        };
        let atlas = texture(atlas_dims.x * TILE, atlas_dims.y * TILE * LEVELS);
        let brdf = texture(64, 64);
        BINDLESS.update();
        let weights = DEVICE.create_buffer::<f32>((LEVELS * TILE * TILE * 32) as usize);
        DEVICE
            .create_kernel::<fn()>(&track!(|| {
                let index = dispatch_id().x;
                let level = index / (TILE * TILE);
                let p = Vec2::expr(index % TILE, (index / TILE) % TILE);
                // Fold octahedral borders directly into the quadrature. No stale
                // neighbour tile or previous-frame border can enter a lookup.
                let q = p.var();
                if p.x == 0u32 || p.x == TILE - 1 {
                    *q = Vec2::expr(
                        (p.x == 0u32).select(1u32.expr(), RES.expr()),
                        TILE - 1 - p.y,
                    );
                }
                if p.y == 0u32 || p.y == TILE - 1 {
                    *q = Vec2::expr(
                        TILE - 1 - p.x,
                        (p.y == 0u32).select(1u32.expr(), RES.expr()),
                    );
                }
                *q = q.clamp(Vec2::splat_expr(1u32), Vec2::splat_expr(RES));
                let axis = OctahedralEncoder::decode((q.cast_f32() - 0.5) / RES as f32);
                let roughness = Self::level_roughness(level);
                let a2 = roughness.powf(4.0);
                let d = dispatch_id().y;
                let l = rc.cache.get_encoder(0u32.expr()).linear_to_dir(d);
                let nl = keter::max(axis.dot(l), 0.0);
                let nh2 = (1.0 + nl) * 0.5;
                let denominator = nh2 * (a2 - 1.0) + 1.0;
                weights.write(
                    d * LEVELS * TILE * TILE + index,
                    nl * a2 / (denominator * denominator),
                );
            }))
            .dispatch([LEVELS * TILE * TILE, 32, 1]);
        DEVICE
            .create_kernel::<fn()>(&track!(|| {
                let p = dispatch_id().xy();
                let nv = (p.x.cast_f32() + 0.5) / 64.0;
                let roughness = (p.y.cast_f32() + 0.5) / 64.0;
                let n = Vec3::z().expr();
                let v = Vec3::expr((1.0 - nv * nv).sqrt(), 0., nv);
                let white = 0.0f32.var();
                let black = 0.0f32.var();
                for s in 0u32..128 {
                    let u = brdf::path_samples(Vec2::splat_expr(0u32), 0u32.expr(), s).xy();
                    let (l, w) = brdf::sample(n, v, roughness, Vec3::splat_expr(1.0), u);
                    let (_, b) = brdf::sample(n, v, roughness, Vec3::splat_expr(0.0), u);
                    if l.z > 0.0 {
                        *white += w.x;
                        *black += b.x;
                    }
                }
                brdf.write(
                    p,
                    Vec4::expr((white - black) / 128.0, black / 128.0, 0., 0.),
                );
            }))
            .dispatch([64, 64, 1]);
        let full =
            || DEVICE.create_buffer_from_fn(core.num_pixels() as usize, |_| Vec4::splat(0.0f32));
        let low =
            || DEVICE.create_buffer_from_fn((size.x * size.y) as usize, |_| Vec4::splat(0.0f32));
        Self {
            raw: full(),
            output: full(),
            diagnostics: full(),
            scale,
            size,
            spp,
            atlas,
            atlas_dims,
            weights,
            brdf,
            signal: low(),
            filtered: low(),
            positions: [low(), low()],
            normals: [low(), low()],
            reflectors: [low(), low()],
            history: [low(), low()],
            full_positions: [full(), full()],
            full_normals: [full(), full()],
            prefilter_kernel: Default::default(),
            trace_kernel: Default::default(),
            spatial_kernel: Default::default(),
            temporal_kernel: Default::default(),
            upscale_kernel: Default::default(),
        }
    }

    #[tracked]
    fn level_roughness(level: Expr<u32>) -> Expr<f32> {
        if level == 0u32 {
            0.35f32.expr()
        } else if level == 1u32 {
            0.65f32.expr()
        } else {
            1.0f32.expr()
        }
    }
    #[tracked]
    fn read(
        &self,
        buffers: &[Buffer<Vec4<f32>>; 2],
        phase: Expr<u32>,
        i: Expr<u32>,
    ) -> Expr<Vec4<f32>> {
        if phase == 0u32 {
            buffers[0].read(i)
        } else {
            buffers[1].read(i)
        }
    }
    #[tracked]
    fn write(
        &self,
        buffers: &[Buffer<Vec4<f32>>; 2],
        phase: Expr<u32>,
        i: Expr<u32>,
        value: Expr<Vec4<f32>>,
    ) {
        if phase == 0u32 {
            buffers[0].write(i, value);
        } else {
            buffers[1].write(i, value);
        }
    }
    #[tracked]
    fn index(&self, p: Expr<Vec2<u32>>) -> Expr<u32> {
        p.y * self.size.x + p.x
    }
    #[tracked]
    pub fn response(
        &self,
        n: Expr<Vec3<f32>>,
        v: Expr<Vec3<f32>>,
        r: Expr<f32>,
        f0: Expr<Vec3<f32>>,
    ) -> Expr<Vec3<f32>> {
        let nv = n.dot(v).clamp(0.0, 1.0);
        if r <= 0.001 {
            brdf::fresnel(f0, nv)
        } else {
            let ab = self.brdf.var().sample(Vec2::expr(nv, r));
            f0 * ab.x + ab.y
        }
    }
    /// One convolution per occupied probe, shared by all screen queries. Run
    /// after RC merge, including after cache resets / slot recycling.
    pub fn prefilter(&self, rc: &RcData) -> NodeConfigs<'static> {
        self.prefilter_kernel
            .get_or_init(|| {
                DEVICE.create_kernel::<fn()>(&track!(|| {
                    set_block_size([1, 64, 1]);
                    let node = rc.cache.get_occupied(0u32.expr(), dispatch_id().x);
                    if node.invalid() {
                        return;
                    }
                    let lane = dispatch_id().y;
                    let level = dispatch_id().z;
                    let values = Shared::<Vec4<f32>>::new(32);
                    if lane < 32u32 {
                        let value = rc.base_radiance.read(node.data_offset + lane);
                        values.write(
                            lane,
                            value.is_finite().all().select(
                                keter::max(value, Vec3::splat_expr(0.0)).extend(1.0),
                                Vec4::splat_expr(0.0),
                            ),
                        );
                    }
                    sync_block();
                    let sum = Vec4::splat(0.0f32).var();
                    for d in 0u32..32 {
                        *sum += values.read(d)
                            * self
                                .weights
                                .read(d * LEVELS * TILE * TILE + level * TILE * TILE + lane);
                    }
                    let tile = Vec2::expr(
                        node.index % self.atlas_dims.x,
                        node.index / self.atlas_dims.x + level * self.atlas_dims.y,
                    );
                    self.atlas.write(
                        tile * TILE + Vec2::expr(lane % TILE, lane / TILE),
                        (sum.xyz() / keter::max(sum.w, 0.000001))
                            .extend((sum.w > 0.000001).cast_u32().cast_f32()),
                    );
                }))
            })
            .dispatch_async([rc.max_probes[0], TILE * TILE, LEVELS])
            .debug("Prefilter reflection RC")
    }
    /// Spatially normalized lookup: absent/cold probes cannot create black
    /// contributions. Alpha is support, allowing a traced fallback at startup.
    #[tracked]
    pub fn lookup(
        &self,
        rc: &RcData,
        pos: Expr<Vec3<f32>>,
        direction: Expr<Vec3<f32>>,
        r: Expr<f32>,
    ) -> Expr<Vec4<f32>> {
        let t = if r < 0.65 {
            ((r - 0.35) / 0.30).clamp(0.0, 1.0)
        } else {
            1.0 + ((r - 0.65) / 0.35).clamp(0.0, 1.0)
        };
        let lower = keter::min(t.floor().cast_u32(), 1u32);
        let f = t - lower.cast_f32();
        let direction = OctahedralEncoder::encode(direction) * RES as f32 + 1.0;
        let result = Vec4::splat(0.0f32).var();
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
                    let tile = Vec2::expr(
                        node.index % self.atlas_dims.x,
                        node.index / self.atlas_dims.x + lower * self.atlas_dims.y,
                    );
                    let uv = (tile * TILE).cast_f32() + direction;
                    let dims =
                        Vec2::new(self.atlas_dims.x * TILE, self.atlas_dims.y * TILE * LEVELS)
                            .expr()
                            .cast_f32();
                    let a = self.atlas.var().sample(uv / dims);
                    let b = self
                        .atlas
                        .var()
                        .sample((uv + Vec2::expr(0.0, (self.atlas_dims.y * TILE) as f32)) / dims);
                    let value = a * (1.0 - f) + b * f;
                    let weight = w * lod_weight * rc.get_quality_weight(node) * value.w;
                    *result += value.xyz().extend(1.0) * weight;
                }
            }
        });
        (result.xyz() / keter::max(result.w, 0.000001)).extend(result.w.clamp(0.0, 1.0))
    }
    #[tracked]
    fn distant(
        &self,
        rc: &RcData,
        pos: Expr<Vec3<f32>>,
        direction: Expr<Vec3<f32>>,
    ) -> Expr<Vec3<f32>> {
        let value = rc
            .sample(pos, (rc.num_cascades - 1).expr(), direction)
            .var();
        for c in (0..rc.num_cascades - 1).rev() {
            *value = super::rc::over(rc.cache.sample(pos, c.expr(), direction), **value);
        }
        value.is_finite().all().select(
            keter::max(**value, Vec3::splat_expr(0.0)),
            Vec3::splat_expr(0.0),
        )
    }
    /// One visibility query with at most one extra mirror continuation.
    #[tracked]
    fn shade_direction(
        &self,
        core: &CoreData,
        rc: &RcData,
        splat: &SplatData,
        p: Expr<Vec2<u32>>,
        cache_valid: Expr<bool>,
        l: Expr<Vec3<f32>>,
    ) -> (Expr<Vec3<f32>>, Expr<Vec4<f32>>) {
        let hit = core.hit.read(core.pixel_to_linear(p));
        let pos = core.get_world_pos(p);
        let v = (core.view.read().pos - pos).normalize();
        let ng = core.world.get_normal_flat(hit, -v);
        let r = core.world.get_roughness_metallic(hit).x;
        let signal = Vec3::splat(0.0f32).var();
        let reflector = Vec4::splat(0.0f32).var();
        if l.dot(ng) > 0.0 {
            let secondary = core.world.trace(offset_ray(pos, ng), l);
            let distance = secondary
                .miss()
                .select(10000.0f32.expr(), secondary.committed_ray_t);
            if secondary.miss() {
                *signal = core.get_environment(l);
                *reflector = (l * 10000.0).extend(0.0);
            } else {
                let point = core.world.get_position(secondary);
                let normal = core.world.get_shading_normal(secondary, l);
                let geometric = core.world.get_normal_flat(secondary, l);
                let (secondary_r, f0, albedo) = core.world.get_pbr_lobes(secondary);
                // Broaden secondary direct highlights instead of relying on
                // many random paths to resolve a narrow, high-energy lobe.
                let effective_r = if r <= 0.001 {
                    secondary_r
                } else {
                    keter::max(secondary_r, 0.20 + 0.25 * r)
                };
                *signal = core.direct_reflection(
                    point,
                    secondary,
                    -l,
                    normal,
                    geometric,
                    effective_r,
                    f0,
                    albedo,
                );
                if cache_valid {
                    let irradiance = splat.sample_secondary(rc, point, normal);
                    *signal += irradiance * albedo;
                    let reflected = normal * (2.0 * normal.dot(-l)) + l;
                    let continuation =
                        if r <= 0.001 && secondary_r < 0.45 && reflected.dot(geometric) > 0.0 {
                            let origin = offset_ray(point, geometric);
                            let last = core.world.trace(origin, reflected);
                            core.get_radiance_cached(Some(rc), Some(splat), origin, reflected, last)
                        } else if r > 0.001 && secondary_r >= 0.45 && f0.reduce_max() < 0.1 {
                            // A broad, weak dielectric lobe can share the diffuse
                            // angular prefilter; reserve directional lookups for
                            // metals and narrow secondary highlights.
                            irradiance
                        } else {
                            let cached =
                                self.lookup(rc, point, reflected, keter::max(secondary_r, 0.35));
                            if cached.w >= 0.999 {
                                cached.xyz()
                            } else {
                                let fallback = self.distant(rc, point, reflected);
                                cached.xyz() * cached.w + fallback * (1.0 - cached.w)
                            }
                        };
                    *signal += if r <= 0.001 && secondary_r >= 0.45 {
                        super::specular::SpecularData::cache_specular(
                            core,
                            rc,
                            point,
                            normal,
                            -l,
                            secondary_r,
                            f0,
                        )
                    } else {
                        continuation * self.response(normal, -l, effective_r, f0)
                    };
                }
                *reflector = point.extend((secondary.inst + 1u32).cast_f32());
            }
            if cache_valid && r > 0.45 {
                let broad = self.lookup(rc, pos, l, keter::max(r, 0.35));
                let blend = ((r - 0.45) / 0.15).clamp(0.0, 1.0);
                let blend = blend * blend * (3.0 - 2.0 * blend);
                // Near blockers retain traced visibility. A single direction
                // is only an occlusion approximation for a broad GGX cone.
                let footprint = rc.base_grid_size * (1u32 << rc.cache.estimate_lod(pos)).cast_f32();
                let visibility =
                    0.85 + 0.15 * (distance / keter::max(footprint * 2.0, 0.001)).clamp(0.0, 1.0);
                let weight = blend * broad.w;
                *signal = signal * (1.0 - weight) + broad.xyz() * weight * visibility;
            }
        }
        (
            signal.is_finite().all().select(
                keter::max(**signal, Vec3::splat_expr(0.0)),
                Vec3::splat_expr(0.0),
            ),
            **reflector,
        )
    }
    /// Self-normalized GGX quadrature keeps a constant incident field constant
    /// with four samples. The split-sum response supplies material reflectance.
    /// This is a biased low-budget estimator, with no disocclusion sample burst.
    #[tracked]
    fn shade(
        &self,
        core: &CoreData,
        rc: &RcData,
        splat: &SplatData,
        p: Expr<Vec2<u32>>,
        frame: Expr<u32>,
        spp: Expr<u32>,
        cache_valid: Expr<bool>,
    ) -> (Expr<Vec3<f32>>, Expr<Vec4<f32>>) {
        let hit = core.hit.read(core.pixel_to_linear(p));
        let pos = core.get_world_pos(p);
        let n = core.read_normal(p);
        let v = (core.view.read().pos - pos).normalize();
        let ng = core.world.get_normal_flat(hit, -v);
        let (r, f0, _) = core.world.get_pbr_lobes(hit);
        let mirror = n * (2.0 * n.dot(v)) - v;
        let dominant = if r <= 0.001 || mirror.dot(ng) > 0.001 {
            mirror
        } else {
            ng * (2.0 * ng.dot(v)) - v
        };
        if r < 0.08 || r >= 0.6 {
            self.shade_direction(core, rc, splat, p, cache_valid, dominant)
        } else {
            let sum = Vec3::splat(0.0f32).var();
            let weights = Vec3::splat(0.0f32).var();
            let reflector = Vec4::splat(0.0f32).var();
            for s in 0u32..8 {
                if s >= spp {
                    break;
                }
                let (l, w) = brdf::sample(n, v, r, f0, brdf::path_samples(p, frame, s).xy());
                if l.dot(ng) > 0.0 && l.dot(n) > 0.0 {
                    let (radiance, guide) =
                        self.shade_direction(core, rc, splat, p, cache_valid, l);
                    *sum += radiance * w;
                    *weights += w;
                    *reflector = guide;
                }
            }
            if (weights > 0.000001).any() {
                (
                    sum / keter::max(**weights, Vec3::splat_expr(0.000001)),
                    **reflector,
                )
            } else {
                // A grazing normal map can reject every sample. Keep the
                // geometric fallback explicit instead of producing a hole.
                self.shade_direction(core, rc, splat, p, cache_valid, dominant)
            }
        }
    }
    pub fn trace(
        &self,
        core: &CoreData,
        rc: &RcData,
        splat: &SplatData,
        frame: u32,
        cache_valid: bool,
        spp: u32,
    ) -> NodeConfigs<'static> {
        let spp = spp.min(self.spp).max(1);
        self.trace_kernel
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32, bool, u32)>(&track!(|frame, cache_valid, spp| {
                    let phase = frame % 2u32;
                    set_block_size([8, 8, 1]);
                    let p = dispatch_id().xy();
                    let i = self.index(p);
                    let full = p * self.scale;
                    let zero = Vec4::splat_expr(0.0f32);
                    self.write(&self.positions, phase, i, zero);
                    self.write(&self.normals, phase, i, zero);
                    self.write(&self.reflectors, phase, i, zero);
                    self.signal.write(i, zero);
                    if core.read_depth(full) == 0.0 {
                        return;
                    }
                    let hit = core.hit.read(core.pixel_to_linear(full));
                    let pos = core.get_world_pos(full);
                    let roughness = core.world.get_roughness_metallic(hit).x;
                    let (value, reflector) = self.shade(core, rc, splat, full, frame, spp, cache_valid);
                    self.signal.write(i, value.extend(1.0));
                    self.write(
                        &self.positions,
                        phase,
                        i,
                        pos.extend((hit.inst + 1u32).cast_f32()),
                    );
                    self.write(
                        &self.normals,
                        phase,
                        i,
                        core.read_normal(full).extend(roughness),
                    );
                    self.write(&self.reflectors, phase, i, reflector);
                }))
            })
            .dispatch_async([self.size.x, self.size.y, 1], &frame, &cache_valid, &spp)
            .debug("Reflection bounded GGX")
    }
    #[tracked]
    fn compatible(
        &self,
        a: Expr<Vec4<f32>>,
        n: Expr<Vec4<f32>>,
        b: Expr<Vec4<f32>>,
        m: Expr<Vec4<f32>>,
        footprint: Expr<f32>,
    ) -> Expr<bool> {
        a.w > 0.0
            && a.w == b.w
            && n.xyz().dot(m.xyz()) > 0.98
            && (n.w - m.w).abs() < 0.10
            && (a.xyz() - b.xyz()).dot(n.xyz()).abs() < footprint * 0.2
    }
    pub fn reconstruct(
        &self,
        core: &CoreData,
        _rc: &RcData,
        _splat: &SplatData,
        frame: u32,
        previous: View,
        valid: bool,
        max_frames: f32,
        _cache_valid: bool,
    ) -> Vec<(&'static str, NodeConfigs<'static>)> {
        let phase = frame % 2;
        let spatial = self
            .spatial_kernel
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32)>(&track!(|phase| {
                    let p = dispatch_id().xy();
                    let i = self.index(p);
                    let pos = self.read(&self.positions, phase, i);
                    let n = self.read(&self.normals, phase, i);
                    let center = self.signal.read(i);
                    let sum = center.xyz().var();
                    let total = 1.0f32.var();
                    let footprint = keter::max(
                        0.01,
                        core.read_depth(p * self.scale)
                            * core.view.read().scaling
                            * (self.scale * 3) as f32,
                    );
                    for y in -2i32..=2 {
                        for x in -2i32..=2 {
                            if x == 0 && y == 0 {
                                continue;
                            }
                            let q = p.cast_i32() + Vec2::expr(x, y);
                            if !(q >= 0).all() || !(q < self.size.expr().cast_i32()).all() {
                                continue;
                            }
                            let j = self.index(q.cast_u32());
                            if !self.compatible(
                                pos,
                                n,
                                self.read(&self.positions, phase, j),
                                self.read(&self.normals, phase, j),
                                footprint,
                            ) {
                                continue;
                            }
                            let a = self.read(&self.reflectors, phase, i);
                            let b = self.read(&self.reflectors, phase, j);
                            if n.w < 0.12 && a.w != b.w {
                                continue;
                            }
                            let width = 0.35 + 2.0 * n.w;
                            let weight =
                                (-(x * x + y * y).cast_f32() / (2.0 * width * width)).exp();
                            *sum += self.signal.read(j).xyz() * weight;
                            *total += weight;
                        }
                    }
                    self.filtered.write(i, (sum / total).extend(center.w));
                }))
            })
            .dispatch_async([self.size.x, self.size.y, 1], &phase);
        let temporal = self
            .temporal_kernel
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32, View, bool, f32)>(&track!(
                    |phase, previous, valid, max_frames| {
                        let p = dispatch_id().xy();
                        let i = self.index(p);
                        let pos = self.read(&self.positions, phase, i);
                        let n = self.read(&self.normals, phase, i);
                        let current = self.filtered.read(i);
                        let history = Vec4::splat(0.0f32).var();
                        let support = 0.0f32.var();
                        let local = previous.transform.transpose() * (pos.xyz() - previous.pos);
                        let screen = (previous.project(pos.xyz()) - 0.5) / self.scale as f32;
                        let footprint =
                            keter::max(0.01, local.z * previous.scaling * (self.scale * 3) as f32);
                        if valid
                            && pos.w > 0.0
                            && local.z > 0.0
                            && (screen >= 0.0).all()
                            && (screen < self.size.expr().cast_f32() - 1.0).all()
                        {
                            let base = screen.floor().cast_u32();
                            let f = screen.fract();
                            for c in 0u32..4 {
                                let q = base + Vec2::expr(c & 1u32, c >> 1u32);
                                let j = self.index(q);
                                let pp = self.read(&self.positions, 1u32 - phase, j);
                                if !self.compatible(
                                    pos,
                                    n,
                                    pp,
                                    self.read(&self.normals, 1u32 - phase, j),
                                    footprint,
                                ) || (pp.xyz() - pos.xyz()).length() > footprint * 1.5
                                {
                                    continue;
                                }
                                let a = self.read(&self.reflectors, phase, i);
                                let b = self.read(&self.reflectors, 1u32 - phase, j);
                                if n.w < 0.08
                                    && (a.w != b.w
                                        || (a.xyz() - b.xyz()).length() > footprint * 4.0)
                                {
                                    continue;
                                }
                                let h = self.read(&self.history, 1u32 - phase, j);
                                if h.w < 0.5 {
                                    continue;
                                }
                                let w = (c & 1u32 != 0u32).select(f.x, 1.0 - f.x)
                                    * (c >> 1u32 != 0u32).select(f.y, 1.0 - f.y);
                                *history += h * w;
                                *support += w;
                            }
                        }
                        // Clip old lighting to current same-surface support, never to
                        // unrelated foreground/background colours. A highlight is a
                        // legitimate outlier, so a 3x3 min/max of a four-sample
                        // estimate must not bound it: that cuts the brightest pixels
                        // every frame and silently defeats accumulation. The bound is
                        // widened to half the local spread and needs real support.
                        let low = current.xyz().var();
                        let high = current.xyz().var();
                        let neighbours = 0u32.var();
                        for y in -1i32..=1 {
                            for x in -1i32..=1 {
                                let q = p.cast_i32() + Vec2::expr(x, y);
                                if !(q >= 0).all() || !(q < self.size.expr().cast_i32()).all() {
                                    continue;
                                }
                                let j = self.index(q.cast_u32());
                                if self.compatible(
                                    pos,
                                    n,
                                    self.read(&self.positions, phase, j),
                                    self.read(&self.normals, phase, j),
                                    footprint,
                                ) {
                                    *neighbours += 1u32;
                                    *low = keter::min(**low, self.filtered.read(j).xyz());
                                    *high = keter::max(**high, self.filtered.read(j).xyz());
                                }
                            }
                        }
                        if support > 0.5 {
                            *history /= support;
                        } else {
                            *history = Vec4::splat_expr(0.0);
                        }
                        let count = keter::min(history.w + 1.0, max_frames);
                        let spread = high - low;
                        // A rare bright reflection sample has no equally bright
                        // neighbour, so bounding the history by the neighbourhood
                        // maximum pins each highlight to its dimmer surroundings and
                        // re-randomises it whenever that maximum moves. Keep the
                        // history's own energy instead: the upper guard is relative to
                        // the pixel itself, while stale dark history is still cut.
                        let upper = keter::max(**high, current.xyz()) * 2.0 + 0.01;
                        let lower = low - spread * 0.5 - 0.002;
                        let retained = if neighbours >= 4u32 {
                            history.xyz().clamp(lower, upper)
                        } else {
                            history.xyz()
                        };
                        let value = (retained * (count - 1.0) + current.xyz()) / count;
                        self.write(&self.history, phase, i, value.extend(count * current.w));
                    }
                ))
            })
            .dispatch_async(
                [self.size.x, self.size.y, 1],
                &phase,
                &previous,
                &valid,
                &max_frames.max(1.0),
            );
        let upscale = self
            .upscale_kernel
            .get_or_init(|| {
                DEVICE.create_kernel::<fn(u32)>(&track!(|frame| {
                    let phase = frame % 2u32;
                    let p = dispatch_id().xy();
                    let i = core.pixel_to_linear(p);
                    let zero = Vec4::splat_expr(0.0f32);
                    self.write(&self.full_positions, phase, i, zero);
                    self.write(&self.full_normals, phase, i, zero);
                    if core.read_depth(p) == 0.0 {
                        self.raw.write(i, zero);
                        self.output.write(i, zero);
                        self.diagnostics.write(i, zero);
                        return;
                    }
                    let hit = core.hit.read(i);
                    let pos = core.get_world_pos(p);
                    let n = core.read_normal(p);
                    let (r, f0, _) = core.world.get_pbr_lobes(hit);
                    self.write(
                        &self.full_positions,
                        phase,
                        i,
                        pos.extend((hit.inst + 1u32).cast_f32()),
                    );
                    self.write(&self.full_normals, phase, i, n.extend(r));
                    let v = (core.view.read().pos - pos).normalize();
                    let footprint = keter::max(
                        0.01,
                        core.read_depth(p) * core.view.read().scaling * (self.scale * 3) as f32,
                    );
                    let base = (p / self.scale).cast_i32();
                    let sum = Vec3::splat(0.0f32).var();
                    let total = 0.0f32.var();
                    for y in -1i32..=1 {
                        for x in -1i32..=1 {
                            let q = base + Vec2::expr(x, y);
                            if !(q >= 0).all() || !(q < self.size.expr().cast_i32()).all() {
                                continue;
                            }
                            let j = self.index(q.cast_u32());
                            if !self.compatible(
                                pos.extend((hit.inst + 1u32).cast_f32()),
                                n.extend(r),
                                self.read(&self.positions, phase, j),
                                self.read(&self.normals, phase, j),
                                footprint,
                            ) {
                                continue;
                            }
                            let d = (q.cast_f32() * self.scale as f32 - p.cast_f32())
                                / self.scale as f32;
                            let weight = (-d.dot(d) * 1.5).exp();
                            *sum += self.read(&self.history, phase, j).xyz() * weight;
                            *total += weight;
                        }
                    }
                    // A thin foreground object may be absent from the low-resolution
                    // G-buffer. Resolve it from the accumulated history of its own tile
                    // instead of tracing a fresh full-resolution ray: a fresh trace
                    // would reintroduce four-sample noise at the exact pixels (mirrors,
                    // railings) that the temporal filter is supposed to stabilise.
                    let coverage = total > 0.00001;
                    let tile = self.read(&self.history, phase, self.index(p / self.scale));
                    let value = if coverage { sum / total } else { tile.xyz() };
                    let reflected = value * self.response(n, v, r, f0);
                    self.raw.write(i, reflected.extend(1.0));
                    self.output.write(i, reflected.extend(1.0));
                    self.diagnostics.write(
                        i,
                        Vec4::expr(
                            r,
                            coverage.select(0.0f32.expr(), 1.0f32.expr()),
                            1.0,
                            (r >= 0.08 && r < 0.6).select(4.0f32.expr(), 1.0f32.expr()),
                        ),
                    );
                }))
            })
            .dispatch_async(core.screen_dispatch_size(), &frame);
        vec![
            ("Reflection spatial", spatial.debug("Reflection spatial")),
            ("Reflection temporal", temporal.debug("Reflection temporal")),
            ("Reflection upsample", upscale.debug("Reflection upsample")),
        ]
    }
}
