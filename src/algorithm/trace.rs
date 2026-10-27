use yesod::direction::project_line;
use yesod::rand::{pcg3df, r2};
use keter::lang::functions::{warp_active_count_bits, warp_active_sum, warp_prefix_sum_exclusive, warp_is_first_active_lane, warp_prefix_count_bits, warp_read_first_active_lane};

use super::*;
use crate::algorithm::rc::{Ray, RayComps, RcData};
use crate::cache::LoadedNode;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
pub struct Hit {
    pub position: Vec3<f32>,
    pub normal: Vec3<f32>,
}
impl HitExpr {
    pub fn valid(&self) -> Expr<bool> {
        self.position.x.is_finite()
    }
}
impl Hit {
    pub fn invalid() -> Self {
        Self {
            position: Vec3::splat(f32::INFINITY),
            normal: Vec3::splat(0.0),
        }
    }
}

pub struct TraceData {
    // Number of rays traced through this probe
    pub ray_count: Buffer<u32>,
    // Next offset for the highest cascade's ray directions
    pub next_max_cascade_ray_direction: Singleton<u32>,
    // Offsets for ray directions for each probe
    pub ray_offset: Buffer<u32>,
    // Index of current ray direction within a base probe
    pub base_ray_index: Buffer<u32>,
    pub traced_hits: Buffer<Hit>,
    pub traced_rays: Buffer<Ray>,
    pub next_traced_ray: Singleton<u32>,
    pub fused_deposit: bool,
    pub ray_capacity: u32,
    pub stable_samples: Option<StableSamples>,
    frame_primary: Option<Buffer<FrameRay>>,
    pub min_probe_rays: u32,
    adaptive_work: Option<Buffer<Vec2<u32>>>,
    diffuse_work: Option<DiffuseWork>,
    primary_registers: u32,
}

struct DiffuseWork {
    items: Buffer<Vec2<u32>>,
    ranks: Buffer<u32>,
    count: Singleton<u32>,
}

#[repr(C)]
#[derive(Clone, Copy, Value)]
struct FrameRay {
    hit: crate::algorithm::Hit,
    direction: Vec3<f32>,
    direct: Vec3<f32>,
}

pub struct StableSamples {
    pixels: Buffer<u32>,
    ranks: Buffer<u32>,
    offsets: Buffer<u32>,
    pixel_counts: Buffer<u32>,
}

impl TraceData {
    pub fn set_primary_register_limit(&mut self, limit: u32) {
        assert!(limit <= 255);
        self.primary_registers = limit;
    }
    /// Split expensive diffuse pixels into eight-ray jobs. This preserves
    /// per-pixel normalization and the first ray used for secondary transport.
    /// Requires diffuse budgets in 1..=64 (the DiffuseData budget contract).
    pub fn enable_diffuse_work(&mut self, core: &CoreData) {
        self.diffuse_work=Some(DiffuseWork {
            items:DEVICE.create_buffer(core.num_pixels() as usize * 8),
            ranks:DEVICE.create_buffer(core.num_pixels() as usize),
            count:Singleton::new(0),
        });
    }
    fn prepare_diffuse_work(&self,core:&CoreData,rc:&RcData,budget:&Buffer<u32>)->NodeConfigs<'static> {
        let work=self.diffuse_work.as_ref().unwrap();
        let prepare=memo((), || DEVICE.create_kernel::<fn()>(&track!(|| {
            let pixel=dispatch_id().xy();
            let i=core.pixel_to_linear(pixel);
            if core.read_depth(pixel)==0.0 {self.traced_hits.write(i,Hit::invalid().expr());return;}
            let pos=core.get_world_pos(pixel);
            let node=rc.cache.get(rc.cache.quantize(pos,rc.cache.estimate_lod(pos),0u32.expr()));
            if node.invalid() {return;}
            let rank=if let Some(stable)=&self.stable_samples {stable.ranks.read(i)} else {self.base_ray_index.atomic_ref(node.index).fetch_add(1u32)};
            work.ranks.write(i,rank);
            let count=(budget.read(i)+7u32)/8u32;
            let total=warp_active_sum(count);
            let start=0u32.var();
            if warp_is_first_active_lane() {*start=work.count.atomic().fetch_add(total);}
            let start=warp_read_first_active_lane(**start)+warp_prefix_sum_exclusive(count);
            for batch in 0u32.expr()..count {work.items.write(start+batch,Vec2::expr(i,batch*8u32));}
        }))).dispatch_async(core.screen_dispatch_size());
        (work.count.write_host(0),prepare).chain()
    }
    pub fn new(core: &CoreData, rc: &RcData, max_rays: usize) -> Self {
        Self::create(core, rc, max_rays, false)
    }
    pub fn new_fused(core: &CoreData, rc: &RcData, max_rays: usize) -> Self {
        Self::create(core, rc, max_rays, true)
    }
    fn create(core: &CoreData, rc: &RcData, max_rays: usize, fused_deposit: bool) -> Self {
        assert!(max_rays <= u32::MAX as usize);
        Self {
            ray_count: DEVICE.create_buffer_from_fn::<u32>(rc.cache.total_probes() as usize, |_| 0),
            next_max_cascade_ray_direction: Singleton::new(0),
            ray_offset: DEVICE
                .create_buffer_from_fn::<u32>(rc.cache.total_probes() as usize, |_| 0),
            base_ray_index: DEVICE
                .create_buffer_from_fn::<u32>(rc.cache.base_probes as usize, |_| 0),
            traced_hits: DEVICE
                .create_buffer_from_fn::<Hit>(core.num_pixels() as usize, |_| Hit::invalid()),
            traced_rays: DEVICE.create_buffer::<Ray>(if fused_deposit { 1 } else { max_rays }),
            next_traced_ray: Singleton::new(0),
            fused_deposit,
            ray_capacity: max_rays as u32,
            stable_samples: None,
            frame_primary: None,
            min_probe_rays: 0,
            adaptive_work: None,
            diffuse_work: None,
            primary_registers: 0,
            // used for debug
        }
    }
    pub fn enable_stable_samples(&mut self, core: &CoreData, rc: &RcData) {
        self.stable_samples = Some(StableSamples {
            pixels: DEVICE.create_buffer(core.num_pixels() as usize),
            ranks: DEVICE.create_buffer(core.num_pixels() as usize),
            offsets: DEVICE.create_buffer(rc.cache.total_probes() as usize),
            pixel_counts: DEVICE.create_buffer(rc.cache.base_probes as usize),
        });
    }
    /// Redistribute extra rays only to undersampled probes. Every original
    /// surface pixel keeps at least one ray. Integer partitioning produces
    /// exactly max(pixel_count, minimum) rays, without rounding up per pixel.
    pub fn apply_probe_budget(&self, rc: &RcData) -> NodeConfigs<'static> {
        let stable = self.stable_samples.as_ref().expect("Stable samples required");
        dispatch0(track!(|| {
            let i = dispatch_id().x;
            if i == 0u32 { self.next_max_cascade_ray_direction.write(0u32); }
            if i < rc.cache.base_probes {
                let pixels = self.ray_count.read(i);
                stable.pixel_counts.write(i, pixels);
                self.ray_count.write(i, (pixels > 0u32).select(keter::max(pixels, self.min_probe_rays), 0u32.expr()));
            } else { self.ray_count.write(i, 0u32); }
        }), [rc.cache.total_probes(),1,1]).debug("Fill undersampled probe budgets")
    }
    pub fn enable_frame_cache(&mut self, core: &CoreData) {
        self.frame_primary = Some(DEVICE.create_buffer(core.num_pixels() as usize));
    }
    pub fn enable_probe_budget(&mut self, core: &CoreData, rc: &RcData, minimum: u32) {
        assert!(self.stable_samples.is_some() && minimum > 0 && minimum <= 256);
        self.min_probe_rays = minimum;
        self.adaptive_work = Some(DEVICE.create_buffer(core.num_pixels() as usize + rc.cache.base_probes as usize * minimum as usize));
    }
    pub fn prepare_probe_work(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        let work = self.adaptive_work.as_ref().expect("Probe work not enabled");
        let stable = self.stable_samples.as_ref().unwrap();
        dispatch0(track!(|| {
            set_block_size([8,8,1]);
            let pixel = dispatch_id().xy();
            let pos = core.get_world_pos(pixel);
            let node = rc.cache.get(rc.cache.quantize(pos, rc.cache.estimate_lod(pos), 0u32.expr()));
            if node.invalid() { return; }
            let rank = stable.ranks.read(core.pixel_to_linear(pixel));
            let pixels = stable.pixel_counts.read(node.index);
            let budget = self.ray_count.read(node.index);
            let (first,end) = if pixels == budget { (rank, rank + 1u32) } else {
                (rank * budget / pixels, (rank + 1u32) * budget / pixels)
            };
            let offset = self.ray_offset.read(node.index);
            let seed = stable.offsets.read(node.index);
            for i in first..end {
                work.write(offset + i, Vec2::expr(core.pixel_to_linear(pixel), seed + i));
            }
        }), core.screen_dispatch_size()).debug("Compact adaptive ray work")
    }
    pub fn clear_frame_hits(&self) -> NodeConfigs<'static> {
        dispatch0(track!(|| { self.traced_hits.write(dispatch_id().x, Hit::invalid()); }), [self.traced_hits.len() as u32,1,1]).debug("Clear frame hit points")
    }

    pub fn resolve_frame_primary(&self, core: &CoreData, rc: &RcData, c0_len: f32) -> NodeConfigs<'static> {
        let rays = self.frame_primary.as_ref().expect("Frame cache not enabled");
        dispatch1(track!(|c0_len| {
            set_block_size([8,8,1]);
            let pixel = dispatch_id().xy();
            let world_pos = core.get_world_pos(pixel);
            let ray = rays.read(core.pixel_to_linear(pixel));
            let radiance = ray.direct.var();
            if !ray.hit.miss() {
                let position = core.world.get_position(ray.hit);
                let normal = core.world.get_normal_flat(ray.hit, ray.direction);
                *radiance += core.world.get_diffuse(ray.hit) * rc.sample_secondary(position, normal);
            }
            rc.cache.iter_lods(world_pos, |lod,_,_| {
                self.compute_deposit_ray(rc, world_pos, lod, ray.direction, **radiance, ray.hit, c0_len);
            });
        }), core.screen_dispatch_size(), c0_len).debug("Resolve current-frame primary transport")
    }

    /// Atomic list insertion is only temporary storage. Rank by pixel ID so
    /// CUDA execution order cannot change the ray assigned to a surface pixel.
    /// Coarse-probe direction origins depend on spatial coordinates, not on
    /// freelist allocation order. The hierarchical R2 intervals stay contiguous.
    pub fn prepare_stable_samples(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        let stable = self.stable_samples.as_ref().expect("Stable samples not enabled");
        let scatter = dispatch0(track!(|| {
            set_block_size([8, 8, 1]);
            let pixel = dispatch_id().xy();
            let pos = core.get_world_pos(pixel);
            let node = rc.cache.get(rc.cache.quantize(pos, rc.cache.estimate_lod(pos), 0u32.expr()));
            if node.invalid() { return; }
            let slot = self.ray_offset.read(node.index) + self.base_ray_index.atomic_ref(node.index).fetch_add(1u32);
            stable.pixels.write(slot, core.pixel_to_linear(pixel));
        }), core.screen_dispatch_size());
        let rank = dispatch0(track!(|| {
            set_block_size([1, 64, 1]);
            let node = rc.cache.get_occupied(0u32.expr(), dispatch_id().x);
            if node.invalid() { return; }
            let offset = self.ray_offset.read(node.index);
            let count = self.ray_count.read(node.index);
            let lane = dispatch_id().y;
            let i = lane.var();
            while i < count {
                let pixel = stable.pixels.read(offset + i);
                let rank = 0u32.var();
                for j in 0u32.expr()..count {
                    *rank += (stable.pixels.read(offset + j) < pixel).cast_u32();
                }
                stable.ranks.write(pixel, **rank);
                *i += 64u32;
            }
        }), [rc.max_probes[0], 64, 1]);
        (scatter, rank).chain().debug("Stable surface sample ranks")
    }
    pub fn compute_ray_count_base(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([8, 8, 1]);
                let world_pos = core.get_world_pos(dispatch_id().xy());

                let lod = rc.cache.estimate_lod(world_pos);

                let probe = rc.cache.quantize(world_pos, lod, 0.expr());

                let node = rc.cache.get(probe);

                if node.invalid() {
                    device_println!("Should be impossible!");
                    return;
                }

                self.ray_count
                    .atomic_ref(node.index)
                    .fetch_add(1_u32.expr());
            }),
            core.screen_dispatch_size(),
        )
        .debug("Compute Ray Count Cascade 0")
    }
    pub fn compute_ray_count(&self, rc: &RcData, cascade: u32) -> NodeConfigs<'static> {
        dispatch1(
            track!(|cascade| {
                let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                if node.invalid() {
                    return;
                }
                let next = node.probe.next_nearest();
                let next = rc.cache.get(next); // must exist
                if next.invalid() { return; }

                self.ray_count
                    .atomic_ref(next.index)
                    .fetch_add(self.ray_count.read(node.index));
            }),
            [rc.cache.max_probes_at_cascade(cascade - 1), 1, 1],
            cascade - 1,
        )
        .debug(format!("Compute Ray Count Cascade {cascade}"))
    }
    pub fn compute_ray_offset(&self, rc: &RcData, cascade: u32) -> NodeConfigs<'static> {
        if cascade == rc.last_cascade() {
            dispatch0(
                track!(|| {
                    let node = rc.cache.get_occupied(cascade.expr(), dispatch_id().x);
                    if node.invalid() {
                        return;
                    }
                    let num_rays = self.ray_count.read(node.index);
                    self.ray_offset.write(
                        node.index,
                        self.next_max_cascade_ray_direction
                            .atomic()
                            .fetch_add(num_rays),
                    );
                    if let Some(stable) = &self.stable_samples {
                        let p = node.probe.position;
                        stable.offsets.write(node.index, yesod::rand::pcg3d(p.xyz() + p.w * 7919u32).x);
                    }
                }),
                [rc.cache.max_probes_at_cascade(cascade), 1, 1],
            )
        } else {
            dispatch1(
                track!(|cascade| {
                    let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                    if node.invalid() { return; }
                    let lower_nodes = <[LoadedNode; 8]>::var_zeroed();
                    let counts = <[u32; 8]>::var_zeroed();
                    let j = 0_u32.var();
                    for i in 0_u32..8_u32 {
                        let lower_probe = node.probe.last_nearest(i);
                        let lower_node = rc.cache.get(lower_probe);
                        if lower_node.loaded() {
                            lower_nodes.write(j, lower_node);
                            counts.write(j, self.ray_count.read(lower_node.index));
                            *j += 1;
                        }
                    }
                    let offset = self.ray_offset.read(node.index).var();
                    let direction_offset = self.stable_samples.as_ref().map(|s| s.offsets.read(node.index).var());
                    for i in 0.expr()..**j {
                        self.ray_offset.write(lower_nodes.read(i).index, offset);
                        *offset += counts.read(i);
                        if let (Some(stable), Some(direction_offset)) = (&self.stable_samples, &direction_offset) {
                            stable.offsets.write(lower_nodes.read(i).index, ***direction_offset);
                            **direction_offset += counts.read(i);
                        }
                    }
                }),
                [rc.cache.max_probes_at_cascade(cascade + 1), 1, 1],
                cascade + 1,
            )
        }
        .debug(format!("Compute Ray Offset Cascade {cascade}"))
    }
    pub fn compute_rays(
        &self,
        core: &CoreData,
        rc: &RcData,
        tick: u32,
        c0_len: f32,
        multibounce: bool,
    ) -> NodeConfigs<'static> {
        memo((multibounce, self.fused_deposit), || {
            DEVICE.create_kernel::<fn(u32, f32)>(&track!(|tick, c0_len| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();
                let world_pos = core.get_world_pos(pixel);
                let normal = core.read_normal(pixel);

                let lod = rc.cache.estimate_lod(world_pos);
                let probe = rc.cache.quantize(world_pos, lod, 0.expr());
                let node = rc.cache.get(probe);

                if node.invalid() {
                    device_println!("Should be impossible!");
                    return;
                }

                let offset = self.ray_offset.read(node.index);

                let offset_index = offset + self.base_ray_index.atomic_ref(node.index).fetch_add(1);

                let dir = project_line(
                    (r2(offset_index) + pcg3df(Vec3::expr(tick, 29, 968)).xy()).fract(),
                );
                let dir = dir * dir.dot(normal).signum();

                let start = offset_ray(world_pos, normal);
                let hit = core.world.trace_segment(
                    start,
                    dir,
                    Vec2::expr(rc.cm1_len * rc.cache.spacing(lod, 0), f32::INFINITY),
                );

                let radiance = core.get_radiance(multibounce.then_some(rc), start, dir, hit);

                self.traced_hits.write(
                    core.pixel_to_linear(pixel),
                    if hit.miss() {
                        Hit::invalid().expr()
                    } else {
                        Hit::from_comps_expr(HitComps {
                            position: core.hit_position(start, dir, hit),
                            normal: core.world.get_normal_flat(hit, dir),
                        })
                    },
                );

                rc.cache.iter_lods(world_pos, |lod, _, _| {
                    self.compute_deposit_ray(rc, world_pos, lod, dir, radiance, hit, c0_len);
                })
            }))
        })
        .dispatch_async(core.screen_dispatch_size(), &tick, &c0_len)
        .debug("Compute Rays")
    }
    /// Experimental frame-local path transport, still filtered by Split RC.
    /// `path_depth == 0` selects the original temporal secondary-probe path.
    pub fn compute_rays_mode(
        &self, core: &CoreData, rc: &RcData, tick: u32, c0_len: f32,
        multibounce: bool, path_depth: u32, samples: u32,
    ) -> NodeConfigs<'static> {
        self.compute_rays_adaptive(core,rc,tick,c0_len,multibounce,path_depth,samples,None)
    }
    pub fn compute_rays_adaptive(&self,core:&CoreData,rc:&RcData,tick:u32,c0_len:f32,
        multibounce:bool,path_depth:u32,samples:u32,budget:Option<&Buffer<u32>>)->NodeConfigs<'static> {
        self.compute_rays_adaptive_cached(core,rc,tick,c0_len,multibounce,path_depth,samples,budget,None)
    }
    pub fn compute_rays_adaptive_cached(&self,core:&CoreData,rc:&RcData,tick:u32,c0_len:f32,
        multibounce:bool,path_depth:u32,samples:u32,budget:Option<&Buffer<u32>>,irradiance:Option<&super::splat::SplatData>)->NodeConfigs<'static> {
        assert!(samples > 0);
        let diffuse_work=self.diffuse_work.as_ref().filter(|_|budget.is_some() && path_depth==0 && self.frame_primary.is_none() && self.min_probe_rays==0);
        let prepare=if diffuse_work.is_some() {self.prepare_diffuse_work(core,rc,budget.unwrap())} else {Default::default()};
        let trace=memo((multibounce, self.fused_deposit, path_depth, samples, self.frame_primary.is_some(), self.min_probe_rays,budget.is_some(),diffuse_work.is_some(),self.primary_registers,irradiance.is_some()), || {
            DEVICE.create_kernel_with_options::<fn(u32, f32)>(KernelBuildOptions {max_registers:self.primary_registers,..Default::default()}, &track!(|tick, c0_len| {
                let pixel = if let Some(work) = diffuse_work {
                    set_block_size([128,1,1]);
                    let i=dispatch_id().x;
                    if i>=work.count.read() {return;}
                    let pixel=work.items.read(i).x;
                    Vec2::expr(pixel % core.screen_size().x,pixel / core.screen_size().x)
                } else if let Some(work) = &self.adaptive_work {
                    set_block_size([128,1,1]);
                    let i = dispatch_id().x;
                    if i >= self.next_max_cascade_ray_direction.read() { return; }
                    let pixel = work.read(i).x;
                    Vec2::expr(pixel % core.screen_size().x, pixel / core.screen_size().x)
                } else {
                    set_block_size([8,8,1]);
                    dispatch_id().xy()
                };
                let world_pos = core.get_world_pos(pixel);
                let normal = core.read_normal(pixel);

                let lod = rc.cache.estimate_lod(world_pos);
                let probe = rc.cache.quantize(world_pos, lod, 0.expr());
                let node = rc.cache.get(probe);

                if node.invalid() {
                    device_println!("Should be impossible!");
                    return;
                }

                let offset = if let Some(stable) = &self.stable_samples { stable.offsets.read(node.index) }
                    else { self.ray_offset.read(node.index) };

                let pixel_index = if let Some(work)=diffuse_work {work.ranks.read(core.pixel_to_linear(pixel))}
                    else if let Some(stable) = &self.stable_samples { stable.ranks.read(core.pixel_to_linear(pixel)) }
                    else { self.base_ray_index.atomic_ref(node.index).fetch_add(1) };
                let sample_weight = if self.min_probe_rays > 0 {
                    let pixels = self.stable_samples.as_ref().unwrap().pixel_counts.read(node.index);
                    let budget = self.ray_count.read(node.index);
                    if budget == pixels { 1.0f32.expr() } else {
                        let rays = (pixel_index + 1u32) * budget / pixels - pixel_index * budget / pixels;
                        1.0 / rays.cast_f32()
                    }
                } else { 1.0f32.expr() };
                let (first, count) = if self.adaptive_work.is_some() {
                    (0u32.expr(), 1u32.expr())
                } else if self.min_probe_rays > 0 {
                    let pixels = self.stable_samples.as_ref().unwrap().pixel_counts.read(node.index);
                    let budget = self.ray_count.read(node.index);
                    if budget == pixels { (pixel_index, 1u32.expr()) } else {
                        let first = pixel_index * budget / pixels;
                        (first, (pixel_index + 1u32) * budget / pixels - first)
                    }
                } else { (pixel_index, 1u32.expr()) };
                let effective_samples=if let Some(budget)=budget {budget.read(core.pixel_to_linear(pixel))} else {samples.expr()};
                // A surface pixel is one transport observation regardless of
                // its adaptive ray budget. Otherwise edge/disocclusion pixels
                // dominate neighbouring probes and change their energy as the
                // camera changes the 2/16/32/64 sample mixture.
                let sample_weight=if path_depth==0 && self.frame_primary.is_none() {
                    sample_weight/effective_samples.cast_f32()
                } else {sample_weight};
                let sample_start=if let Some(work)=diffuse_work {work.items.read(dispatch_id().x).y} else {0u32.expr()};
                let sample_end=if diffuse_work.is_some() {keter::min(sample_start+8u32,effective_samples)} else {count*effective_samples};
                for sample in sample_start..sample_end {
                let offset_index = if let Some(work) = &self.adaptive_work { work.read(dispatch_id().x).y * samples + sample }
                    else if path_depth==0 && self.frame_primary.is_none() {offset+first+sample*104729u32}
                    else { (offset + first) * samples + sample };

                let dir = project_line(
                    (r2(offset_index) + pcg3df(Vec3::expr(tick, 29, 968)).xy()).fract(),
                );
                let dir = dir * dir.dot(normal).signum();

                let start = offset_ray(world_pos, normal);
                let hit = core.world.trace_segment(
                    start,
                    dir,
                    Vec2::expr(rc.cm1_len * rc.cache.spacing(lod, 0), f32::INFINITY),
                );

                let radiance = if self.frame_primary.is_some() {
                    core.get_radiance(None, start, dir, hit)
                } else if path_depth == 0 {
                    core.get_radiance_cached(multibounce.then_some(rc), irradiance, start, dir, hit)
                } else {
                    self.path_radiance(core, start, dir, hit, offset_index, path_depth)
                };

                if (path_depth == 0 || self.frame_primary.is_some()) && sample==0u32 { self.traced_hits.write(
                    core.pixel_to_linear(pixel),
                    if hit.miss() {
                        Hit::invalid().expr()
                    } else {
                        Hit::from_comps_expr(HitComps {
                            position: core.hit_position(start, dir, hit),
                            normal: core.world.get_normal_flat(hit, dir),
                        })
                    },
                ); }

                if let Some(rays) = &self.frame_primary {
                    assert_eq!(samples, 1);
                    rays.write(core.pixel_to_linear(pixel), FrameRay::from_comps_expr(FrameRayComps {
                        hit, direction: dir, direct: radiance,
                    }));
                } else {
                    rc.cache.iter_lods(world_pos, |lod, _, _| {
                        self.compute_deposit_ray_weighted(rc, world_pos, lod, dir, radiance, hit, c0_len, sample_weight);
                    });
                }
                }
            }))
        })
        .dispatch_async(if let Some(work)=diffuse_work {[work.items.len() as u32,1,1]} else if let Some(work) = &self.adaptive_work { [work.len() as u32,1,1] } else { core.screen_dispatch_size() }, &tick, &c0_len)
        .debug("Compute Rays");
        (prepare,trace).chain()
    }

    #[tracked]
    fn path_radiance(&self, core: &CoreData, start: Expr<Vec3<f32>>, direction: Expr<Vec3<f32>>,
        first_hit: Expr<crate::algorithm::Hit>, seed: Expr<u32>, depth: u32) -> Expr<Vec3<f32>> {
        let origin = start.var();
        let direction = direction.var();
        let hit = first_hit.var();
        let throughput = Vec3::splat(1.0f32).var();
        let radiance = Vec3::splat(0.0f32).var();
        for bounce in 0u32..depth {
            if hit.miss() {
                *radiance += throughput * core.get_environment(**direction);
                break;
            }
            let position = core.hit_position(**origin, **direction, **hit);
            *radiance += throughput * core.get_emission(None, position, **hit, **direction);
            if bounce + 1u32 == depth { break; }
            let normal = core.world.get_normal_flat(**hit, **direction);
            // Cosine-distributed deterministic hemisphere sample: diffuse/pdf
            // cancels, so throughput remains bounded by the material albedo.
            // Independent dimensions for later bounces. Reusing a translated
            // copy of the primary R2 pair correlates successive path vertices
            // and biases transport. This seed never depends on frame number.
            let u = pcg3df(Vec3::expr(seed, bounce, 941)).xy();
            let helper: Expr<Vec3<f32>> = (normal.z.abs() < 0.999).select(Vec3::new(0.0f32, 0., 1.).expr(), Vec3::new(0.0f32, 1., 0.).expr());
            let tangent = helper.cross(normal).normalize();
            let bitangent = normal.cross(tangent);
            let angle = u.y * (2.0 * std::f32::consts::PI);
            let radius = u.x.sqrt();
            *direction = tangent * (radius * angle.cos()) + bitangent * (radius * angle.sin()) + normal * (1.0 - u.x).sqrt();
            *throughput *= core.world.get_diffuse(**hit);
            *origin = offset_ray(position, normal);
            *hit = core.world.trace(**origin, **direction);
        }
        **radiance
    }
    pub fn compute_secondary_rays(
        &self,
        core: &CoreData,
        rc: &RcData,
        tick: u32,
        c0_len: f32,
        multibounce: bool,
    ) -> NodeConfigs<'static> {
        self.compute_secondary_mode(core, rc, tick, c0_len, multibounce, 0)
    }
    pub fn compute_secondary_mode(&self, core: &CoreData, rc: &RcData, tick: u32, c0_len: f32, multibounce: bool, path_depth: u32) -> NodeConfigs<'static> {
        self.compute_secondary_cached(core, rc, tick, c0_len, multibounce, path_depth, None)
    }
    pub fn compute_secondary_cached(&self, core: &CoreData, rc: &RcData, tick: u32, c0_len: f32, multibounce: bool, path_depth: u32, irradiance:Option<&super::splat::SplatData>) -> NodeConfigs<'static> {
        memo((multibounce, self.fused_deposit, path_depth, irradiance.is_some()), || {
            DEVICE.create_kernel::<fn(u32, f32)>(&track!(|tick, c0_len| {
                let hit = self.traced_hits.read(dispatch_id().x);
                if !hit.valid() {
                    return;
                }
                let world_pos = hit.position;
                let normal = hit.normal;

                let lod = rc.cache.estimate_lod_secondary(world_pos);
                let probe = rc.cache.quantize(world_pos, lod, 0.expr());
                let node = rc.cache.get(probe);

                if node.invalid() {
                    device_println!("Should be impossible!");
                    return;
                }
                let dir = project_line(pcg3df(Vec3::expr(dispatch_id().x, tick, 173)).xy());
                let dir = dir * dir.dot(normal).signum();
                let start = offset_ray(world_pos, normal);

                let hit = core.world.trace_segment(
                    start,
                    dir,
                    Vec2::expr(rc.cm1_len * rc.cache.spacing(lod, 0), f32::INFINITY),
                );

                let radiance = if path_depth == 0 { core.get_radiance_cached(multibounce.then_some(rc), irradiance, start, dir, hit) }
                    else { self.path_radiance(core, start, dir, hit, dispatch_id().x, path_depth) };

                self.compute_deposit_ray(rc, world_pos, lod, dir, radiance, hit, c0_len);
            }))
        })
        .dispatch_async([self.traced_hits.len() as u32, 1, 1], &tick, &c0_len)
        .debug("Secondary: Compute Rays")
    }
    #[tracked]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_deposit_ray(
        &self,
        rc: &RcData,
        world_pos: Expr<Vec3<f32>>,
        lod: Expr<u32>,
        dir: Expr<Vec3<f32>>,
        radiance: Expr<Vec3<f32>>,
        hit: Expr<crate::algorithm::Hit>,
        c0_len: Expr<f32>,
    ) {
        self.compute_deposit_ray_weighted(rc, world_pos, lod, dir, radiance, hit, c0_len, 1.0f32.expr());
    }
    #[tracked]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_deposit_ray_weighted(&self, rc: &RcData, world_pos: Expr<Vec3<f32>>, lod: Expr<u32>,
        dir: Expr<Vec3<f32>>, radiance: Expr<Vec3<f32>>, hit: Expr<crate::algorithm::Hit>, c0_len: Expr<f32>, weight: Expr<f32>) {
        let probe = rc.cache.quantize(world_pos, lod, 0.expr());

        // Reserve one contiguous range per active warp instead of contending
        // on the same global counter once per ray. Every active lane still
        // emits its original ray, including partially active LOD iterations.
        let count = warp_active_count_bits(true.expr());
        let base = 0u32.var();
        if warp_is_first_active_lane() {
            *base = self.next_traced_ray.atomic().fetch_add(count);
        }
        let ray_index = warp_read_first_active_lane(**base) + warp_prefix_count_bits(true.expr());

        if ray_index >= self.ray_capacity {
            return;
        }

        let cascade = if hit.miss() {
            rc.last_cascade().expr()
        } else {
            let cascade = keter::max(
                (hit.committed_ray_t / (c0_len * rc.cache.spacing(lod, 0))).log(rc.length_scale),
                0.0,
            )
            .ceil()
            .cast_u32();

            keter::min(cascade, rc.last_cascade())
        };

        let ray = Ray::from_comps_expr(RayComps {
            radiance,
            base_probe: probe,
            cascade,
            dir: SphericalEncoder::encode(dir),
        });
        if self.fused_deposit {
            rc.deposit_ray_linked_weighted(ray, weight);
        } else {
            self.traced_rays.write(ray_index, ray);
        }
    }
    pub fn reset(&self) -> NodeConfigs<'static> {
        (
            dispatch0(
                track!(|| {
                    self.ray_count.write(dispatch_id().x, 0_u32);
                }),
                [self.ray_count.len() as u32, 1, 1],
            )
            .debug("Reset Node Ray Count"),
            dispatch0(
                track!(|| {
                    self.base_ray_index.write(dispatch_id().x, 0_u32);
                }),
                [self.base_ray_index.len() as u32, 1, 1],
            )
            .debug("Reset Base Ray Offsets"),
            dispatch0(
                track!(|| {
                    self.next_traced_ray.write(0_u32);
                    self.next_max_cascade_ray_direction.write(0_u32);
                }),
                [1, 1, 1],
            )
            .debug("Reset Next Traced Ray"),
        )
            .chain()
    }
}
