use yesod::direction::project_line;
use yesod::rand::{pcg3df, r2};

use super::*;
use crate::algorithm::rc::RcData;
use crate::cache::LoadedNode;

pub struct TraceData {
    // Number of rays traced through this probe
    pub ray_count: Buffer<u32>,
    // Next offset for the highest cascade's ray directions
    pub next_max_cascade_ray_direction: Singleton<u32>,
    // Offsets for ray directions for each probe
    pub ray_offset: Buffer<u32>,
    // Index of current ray direction within a base probe
    pub base_ray_index: Buffer<u32>,
    pub traced_rays: Buffer<Ray>,
    pub next_traced_ray: Singleton<u32>,
    pub raw_radiance: Option<Tex2d<Vec3<f32>>>,
}

#[repr(C)]
#[derive(Value, Debug, Clone, Copy)]
pub struct Ray {
    // must be normalized
    pub dir: Vec3<f32>,
    pub start: Vec3<f32>,
    pub base_probe: Probe,
    pub hit: Hit,
}

impl TraceData {
    pub fn new(core: &CoreData, rc: &RcData, max_rays: usize, debug_raw_radiance: bool) -> Self {
        Self {
            ray_count: DEVICE.create_buffer_from_fn::<u32>(rc.cache.total_probes() as usize, |_| 0),
            next_max_cascade_ray_direction: Singleton::new(0),
            ray_offset: DEVICE
                .create_buffer_from_fn::<u32>(rc.cache.total_probes() as usize, |_| 0),
            base_ray_index: DEVICE
                .create_buffer_from_fn::<u32>(rc.cache.base_probes as usize, |_| 0),
            traced_rays: DEVICE.create_buffer::<Ray>(max_rays),
            next_traced_ray: Singleton::new(0),
            // used for debug
            raw_radiance: debug_raw_radiance.then(|| {
                DEVICE.create_tex2d::<Vec3<f32>>(
                    PixelStorage::Float4,
                    core.screen_size().x,
                    core.screen_size().y,
                    1,
                )
            }),
        }
    }
    pub fn compute_ray_count_base(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        dispatch0_async(
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
        dispatch1_async(
            track!(|cascade| {
                let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                if node.invalid() {
                    return;
                }
                let next = node.probe.next_nearest();
                let next = rc.cache.get(next); // must exist

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
            dispatch0_async(
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
                }),
                [rc.cache.max_probes_at_cascade(cascade), 1, 1],
            )
        } else {
            dispatch1_async(
                track!(|cascade| {
                    let node = rc.cache.get_occupied(cascade, dispatch_id().x);
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
                    for i in 0.expr()..**j {
                        self.ray_offset.write(lower_nodes.read(i).index, offset);
                        *offset += counts.read(i);
                    }
                }),
                [rc.cache.max_probes_at_cascade(cascade + 1), 1, 1],
                cascade + 1,
            )
        }
        .debug(format!("Compute Ray Offset Cascade {cascade}"))
    }
    pub fn compute_rays(&self, core: &CoreData, rc: &RcData, tick: u32) -> NodeConfigs<'static> {
        dispatch1_async(
            track!(|tick| {
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

                let bounce_ray = project_line(
                    (r2(offset_index) + pcg3df(Vec3::expr(tick, 29, 968)).xy()).fract(),
                );
                let bounce_ray = bounce_ray * bounce_ray.dot(normal).signum();

                let ray_index = self.next_traced_ray.atomic().fetch_add(1_u32.expr());

                if ray_index < self.traced_rays.len_expr_u32() {
                    let start = offset_ray(world_pos, normal);
                    let ray = Ray::from_comps_expr(RayComps {
                        dir: bounce_ray,
                        start,
                        base_probe: probe,
                        hit: core.world.trace(start, bounce_ray),
                    });
                    self.traced_rays.write(ray_index, ray);

                    if let Some(raw_radiance) = &self.raw_radiance {
                        let radiance = core.get_fluence(start, bounce_ray, ray.hit).xyz();
                        raw_radiance.write(pixel, radiance);
                    }
                }
            }),
            core.screen_dispatch_size(),
            tick,
        )
        .debug("Compute Rays")
    }
    // TODO: Can we remove the kernel options?
    // Also refactor tracing to previous step.
    pub fn compute_radiance(
        &self,
        core: &CoreData,
        rc: &RcData,
        c0_len: f32,
    ) -> NodeConfigs<'static> {
        memo(rc.terminate_deposit, || {
            DEVICE.create_kernel::<fn(f32)>(&track!(|c0_len| {
                set_block_size([64, 1, 1]);

                if dispatch_id().x >= self.next_traced_ray.read() {
                    return;
                }

                let ray = self.traced_rays.read(dispatch_id().x);

                rc.deposit_split(core, ray, c0_len)
            }))
        })
        .dispatch_async([self.traced_rays.len() as u32, 1, 1], &c0_len)
        .debug("Compute Radiance")
    }
    pub fn reset(&self) -> NodeConfigs<'static> {
        (
            dispatch0_async(
                track!(|| {
                    self.ray_count.write(dispatch_id().x, 0_u32);
                }),
                [self.ray_count.len() as u32, 1, 1],
            )
            .debug("Reset Node Ray Count"),
            dispatch0_async(
                track!(|| {
                    self.base_ray_index.write(dispatch_id().x, 0_u32);
                }),
                [self.base_ray_index.len() as u32, 1, 1],
            )
            .debug("Reset Base Ray Offsets"),
            dispatch0_async(
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
