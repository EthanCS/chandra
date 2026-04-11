use std::ops::Deref;
use std::sync::Arc;

use keter::lang::functions::{
    sync_block, thread_id, warp_active_count_bits, warp_prefix_count_bits, warp_read_lane_at,
};
use parking_lot::Mutex;
use yesod::utils::FetchAddVector;

use super::*;
use crate::algorithm::trace::Ray;
use crate::cache::{Fluence, HashCache, LoadedNode, Radiance};

pub struct RcData {
    pub cache: HashCache,
    pub radiance: Buffer<Radiance>,
    // only used if increased_base_resolution
    pub base_radiance: Buffer<Radiance>,
    pub probe_quality: Buffer<f32>,
    pub probe_lifetime: Buffer<u32>,
    // for base probes only, currently
    pub average_pos_normal: Buffer<AveragePosNormal>,
    // created from build_freelists
    // not really used. might be worth handling.
    pub num_probes: Vec<u32>,
    pub max_probes: Vec<u32>,

    pub config: RcConfig,
}
impl Deref for RcData {
    type Target = RcConfig;
    fn deref(&self) -> &Self::Target {
        &self.config
    }
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

#[derive(Debug, Clone, Copy)]
pub enum MergeLevel {
    // First,
    Last,
    Middle(Expr<u32>),
}

#[tracked]
fn over(fluence: Expr<Fluence>, radiance: Expr<Radiance>) -> Expr<Radiance> {
    fluence.xyz() + radiance * fluence.w
}

#[derive(Debug, Clone, Copy)]
pub struct RcConfig {
    pub base_grid_size: f32,
    pub base_radius: f32,
    pub base_angular_resolution: u32,
    pub base_probes: u32,
    pub num_cascades: u32,
    pub length_scale: f32,
    pub use_quality_weight: bool,
    pub quality_decay: f32,
    pub min_direction_weight: f32,
    pub max_lifetime: u32,
    pub merge_nearest: bool,
    pub increased_base_resolution: bool,
    pub terminate_deposit: bool,
    pub interpolation_exponent: f32,
}
impl Default for RcConfig {
    fn default() -> Self {
        Self {
            base_grid_size: 10.0,
            base_radius: 1000.0,
            base_angular_resolution: 4,
            base_probes: 1 << 19,
            num_cascades: 4,
            length_scale: 4.0,
            use_quality_weight: true,
            quality_decay: 0.99,
            min_direction_weight: 2.0,
            max_lifetime: 10,
            merge_nearest: false,
            increased_base_resolution: false,
            terminate_deposit: true,
            interpolation_exponent: 1.0,
        }
    }
}

impl RcData {
    #[allow(clippy::new_without_default)]
    pub fn new(config: RcConfig) -> Self {
        let cache = HashCache::new(
            config.base_grid_size,
            config.base_radius,
            config.base_angular_resolution,
            config.base_probes,
            config.num_cascades,
        );
        Self {
            radiance: DEVICE.create_buffer::<Radiance>(cache.total_directions() as usize / 4),
            base_radiance: DEVICE.create_buffer::<Radiance>(cache.total_directions() as usize),
            probe_quality: DEVICE
                .create_buffer_from_fn::<f32>(cache.total_probes() as usize, |_| 0.0),
            probe_lifetime: DEVICE
                .create_buffer_from_fn::<u32>(cache.total_probes() as usize, |_| 0),
            average_pos_normal: DEVICE
                .create_buffer_from_fn(cache.total_probes() as usize, |_| Default::default()),
            num_probes: vec![0; cache.num_cascades as usize],
            max_probes: vec![0; cache.num_cascades as usize],
            cache,
            config,
        }
    }
    pub fn num_cascades(&self) -> u32 {
        self.cache.num_cascades
    }
    pub fn last_cascade(&self) -> u32 {
        self.cache.num_cascades - 1
    }
    pub fn get_quality_weight(&self, node: Expr<LoadedNode>) -> Expr<f32> {
        if self.use_quality_weight {
            self.probe_quality.read(node.index)
        } else {
            1.0_f32.expr()
        }
        // let quality = node_quality.read(node.index);
        // let encoder = cache.get_encoder(node.probe.cascade);
        // let ratio = quality / encoder.linear_size().cast_f32();
        // keter::min(1.0, ratio * TARGET_QUALITY_MULTIPLIER)
    }
    #[tracked]
    pub fn build_freelists(&self, total: bool) -> NodeConfigs<'static> {
        memo(self.quality_decay, || {
            DEVICE.create_kernel_async::<fn(bool)>(&|total| {
                let data_index = dispatch_id().x;
                self.cache
                    .table
                    .build_freelists(data_index, |key, cascade| {
                        let lifetime = self.probe_lifetime.read(data_index).var();
                        let quality = self.probe_quality.read(data_index).var();
                        let quality = quality * self.quality_decay;
                        let invalid = lifetime == 0 || total;
                        *lifetime -= 1_u32;
                        if invalid {
                            // device_println!("Deleting: {}", host!(data_index));
                            let node = self
                                .cache
                                .load_index(self.cache.decode(key, cascade), data_index);
                            let encoder = self.cache.get_encoder(cascade);

                            for index in 0_u32.expr()..encoder.linear_size() {
                                self.cache
                                    .weighted_fluence
                                    .write(node.data_offset + index, Vec4::splat(0.0));
                                self.cache.weights.write(node.data_offset + index, 0.0);
                            }
                            self.probe_quality.write(data_index, 0.0);
                        } else {
                            self.probe_lifetime.write(data_index, **lifetime);
                            self.probe_quality.write(data_index, quality);
                        }
                        invalid
                    });
            })
        })
        .dispatch_async([self.cache.table.freelist.len() as u32, 1, 1], &total)
        .debug("Build Freelists")
    }
    pub fn merge_cache(&self, cascade: u32) -> NodeConfigs<'static> {
        let dispatch_size = [
            self.max_probes[cascade as usize],
            self.cache.directions_per_probe(cascade),
            1,
        ];
        let key = (
            self.merge_nearest,
            self.increased_base_resolution,
            self.quality_decay,
            self.min_direction_weight,
        );
        if cascade == self.last_cascade() {
            memo(key, || {
                DEVICE.create_kernel::<fn()>(&|| self.merge_cache_raw(MergeLevel::Last))
            })
            .dispatch_async(dispatch_size)
        } else {
            memo(key, || {
                DEVICE.create_kernel::<fn(u32)>(&|cascade| {
                    self.merge_cache_raw(MergeLevel::Middle(cascade))
                })
            })
            .dispatch_async(dispatch_size, &cascade)
        }
        .debug(format!("Merge Cascade {cascade}"))
    }
    // Note: can additionally load all lower probes at once as well for increased efficiency.
    #[tracked]
    fn merge_cache_raw(&self, merge_level: MergeLevel) {
        set_block_size([1, 32, 1]);
        let cascade = match merge_level {
            // MergeLevel::First => 0.expr(),
            MergeLevel::Last => self.last_cascade().expr(),
            MergeLevel::Middle(l) => l,
        };
        let node = self.cache.get_occupied(cascade, dispatch_id().x);
        if node.invalid() {
            return;
        }

        let probe = node.probe;

        let n = 0_u32.var();
        let next_offsets = Shared::<u32>::new(8);
        let next_weights = Shared::<f32>::new(8);

        if !matches!(merge_level, MergeLevel::Last) {
            if thread_id().y < if self.merge_nearest { 1 } else { 8 } {
                let i = thread_id().y;
                let (next_probe, weight) = if self.merge_nearest {
                    (probe.next_nearest(), 1.0.expr())
                } else {
                    probe.next_bilinear(i)
                };

                let next = self.cache.get(next_probe);
                if next.loaded() {
                    let index = warp_prefix_count_bits(true.expr());

                    next_offsets.write(index, next.data_offset);
                    next_weights.write(index, weight * self.get_quality_weight(next));
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
                let radiance = self.radiance.read(next_offsets.read(i) / 4 + index);
                // necessary?
                if radiance.is_finite().all() {
                    *total_radiance += next_weights.read(i)
                        * radiance.powf(self.interpolation_exponent).extend(1.0);
                }
            }
        }
        let data_index = node.data_offset + index;

        let far = total_radiance.xyz() / keter::max(total_radiance.w, 0.00001);
        let far = far.powf(self.interpolation_exponent.recip());

        let weight = self.cache.weights.read(data_index);
        let weighted_fluence = self.cache.weighted_fluence.read(data_index);
        let fluence = weighted_fluence / weight;

        // decay cache
        // TODO: Should this be here  or somewhere else?
        if self.quality_decay < 1.0 && weight > self.min_direction_weight {
            self.cache
                .weighted_fluence
                .write(data_index, weighted_fluence * self.quality_decay);
            self.cache
                .weights
                .write(data_index, weight * self.quality_decay);
        }

        let radiance = if weight < 0.0001 {
            Vec4::splat_expr(0.0)
        } else {
            over(fluence, far).extend(1.0)
        };

        let adj_radiance = warp_read_lane_at(radiance, (index + 1) % 32);
        if index % 2 == 0 {
            let radiance = radiance + adj_radiance;
            let adj_radiance = warp_read_lane_at(radiance, (index + 2) % 32);
            if index % 4 == 0 {
                let radiance = radiance + adj_radiance;
                self.radiance
                    .write(data_index / 4, radiance.xyz() / radiance.w);
            }
        }
        if self.increased_base_resolution && cascade == 0 {
            self.base_radiance
                .write(data_index, radiance.xyz() / radiance.w);
        }
    }

    // multiplicity: u32
    #[tracked]
    pub fn trace_deposit(
        &self,
        core: &CoreData,
        c0_len: f32,
        cascade: u32,
        multiplicity: u32,
        use_average_position: bool,
    ) -> NodeConfigs<'static> {
        memo((), || {
            DEVICE.create_kernel_async::<fn(f32, u32)>(&|c0_len, cascade| {
                set_block_size([1, 32, 1]);

                let node = self.cache.get_occupied(cascade, dispatch_id().x);
                if node.invalid() {
                    return;
                }
                let world_pos = if use_average_position {
                    let apn = self.average_pos_normal.read(node.index);
                    offset_ray(apn.position(), apn.normal())
                } else {
                    self.cache.dequantize(node.probe)
                };

                let dir = self
                    .cache
                    .get_encoder(cascade)
                    .linear_to_grid(dispatch_id().y)
                    .cast_f32();

                let fluence = Vec4::splat(0.0_f32).var();

                for i in 0..multiplicity * multiplicity {
                    let offset = Vec2::expr(i % multiplicity, i / multiplicity).cast_f32() + 0.5;
                    let dir = dir + offset / multiplicity as f32;
                    let dir = self.cache.get_encoder(cascade).grid_to_dir_f32(dir);

                    let start_len = c0_len
                        * self.base_grid_size
                        * (1_u32 << node.probe.position.w).cast_f32()
                        * self.length_scale.expr().powi(cascade.cast_i32() - 1);
                    let end_len = if cascade == self.last_cascade() {
                        f32::INFINITY.expr()
                    } else {
                        start_len * self.length_scale
                    };
                    // shadowing issues...
                    let start_len = if cascade == 0 { 0.0.expr() } else { start_len };

                    let hit =
                        core.world
                            .trace_segment(world_pos, dir, Vec2::expr(start_len, end_len));
                    *fluence += core.get_fluence(world_pos, dir, hit);
                }

                self.cache.deposit_once(
                    node.data_offset + dispatch_id().y,
                    fluence / (multiplicity * multiplicity) as f32,
                );
            })
        })
        .dispatch_async(
            [
                self.max_probes[cascade as usize],
                self.cache.directions_per_probe(cascade),
                1,
            ],
            &c0_len,
            &cascade,
        )
        .debug(format!("Trace and Deposit Cascade {cascade}"))
    }

    #[tracked]
    pub fn deposit_split(&self, core: &CoreData, ray: Expr<Ray>, c0_len: Expr<f32>) {
        let world_pos = ray.start;
        let bounce_ray = ray.dir;
        let hit = ray.hit;

        let probe = ray.base_probe;

        let (hit_level, fluence) = if hit.miss() {
            (self.num_cascades().expr() - 1, Vec4::splat_expr(0.0))
        } else {
            let cascade = keter::max(
                (hit.committed_ray_t
                    / (c0_len * self.base_grid_size * (1_u32 << probe.position.w).cast_f32()))
                .log(self.length_scale),
                0.0,
            )
            .ceil()
            .cast_u32();
            (
                keter::min(cascade, self.num_cascades().expr() - 1),
                core.get_fluence(world_pos, bounce_ray, hit),
            )
        };

        let deposit = |cascade, fluence| {
            let node = self.cache.get(probe.shift_level(cascade));
            if node.loaded() {
                let encoder = self.cache.get_encoder(cascade);
                self.cache.deposit(
                    node.data_offset + encoder.dir_to_linear(bounce_ray),
                    fluence,
                );
                self.probe_quality.atomic_ref(node.index).fetch_add(1.0);
            } // Otherwise should be impossible
        };

        if self.terminate_deposit {
            // TODO: If we invert transmittance then the lower-deposits are just count increments.
            // note: since lifetime is deposited, short rays can cause higher-cascades to vanish

            // NOTE: Perf here is very inconsistent; the second part seems much more expensive.
            for level in 0.expr()..hit_level {
                deposit(level, Vec3::splat_expr(0.0_f32).extend(1.0));
            }
            deposit(hit_level, fluence);
        } else {
            for level in 0..self.num_cascades() {
                deposit(
                    level,
                    if level < hit_level {
                        Vec3::splat_expr(0.0_f32).extend(1.0)
                    } else {
                        fluence
                    },
                );
            }
        }
    }

    pub fn reset_table(&mut self, commands: &mut CommandList<'static>, total: bool) {
        const ZEROS: [u32; 1024] = [0_u32; 1024];

        let table = &self.cache.table;
        let clear_entries = dispatch0(
            track!(|| {
                table
                    .entries
                    .write(dispatch_id().x, crate::table::Entry::EMPTY);
            }),
            [table.entries.len() as u32, 1, 1],
        );
        *commands += (
            table
                .table_data_start
                .copy_to_buffer_async(&table.freelist_index),
            table
                .table_data_start
                .copy_to_buffer_async(&table.last_occupied_index),
            table
                .total_occupied
                .copy_from_async(&ZEROS[..self.num_cascades() as usize]),
            clear_entries,
            self.build_freelists(total),
            table
                .table_data_start
                .copy_to_buffer_async(&table.freelist_index),
        )
            .chain();

        // find probe counts
        let total_occupied = Arc::new(Mutex::new(vec![0_u32; self.num_cascades() as usize]));
        let last_occupied_index = Arc::new(Mutex::new(vec![0_u32; self.num_cascades() as usize]));
        *commands += self
            .cache
            .table
            .total_occupied
            .copy_to_shared(&total_occupied);
        *commands += self
            .cache
            .table
            .last_occupied_index
            .copy_to_shared(&last_occupied_index);

        commands.execute();

        self.num_probes = total_occupied.lock().to_vec();

        let last_occupied_index = last_occupied_index.lock();
        self.max_probes = last_occupied_index
            .iter()
            .enumerate()
            .map(|(c, &x)| x - self.cache.get_index_offset_host(c as u32) + 1)
            .collect();
    }

    pub fn init_probes_base(&self, core: &CoreData) -> NodeConfigs<'static> {
        dispatch0(
            || {
                set_block_size([8, 8, 1]);
                let world_pos = core.get_world_pos(dispatch_id().xy());

                let lod = self.cache.estimate_lod(world_pos);
                let probe = self.cache.quantize(world_pos, lod, 0.expr());
                self.cache.insert(probe, |_| {});
            },
            core.screen_dispatch_size(),
        )
        .debug("Init Probes Cascade 0")
    }
    pub fn init_probes(&self, cascade: u32) -> NodeConfigs<'static> {
        dispatch1(
            track!(|cascade| {
                let index = dispatch_id().x + self.cache.get_index_offset(cascade);
                let key = self.cache.table.data_key.read(index);
                if key == crate::table::INVALID_KEY {
                    return;
                }
                let probe = self.cache.decode(key, cascade);
                let probe = probe.next_nearest();
                self.cache.insert(probe, |_| {});
            }),
            [self.cache.max_probes_at_cascade(cascade - 1), 1, 1],
            cascade - 1,
        )
        .debug(format!("Init Probes Cascade {cascade}"))
    }
    pub fn set_lifetime_base(&self, core: &CoreData) -> NodeConfigs<'static> {
        memo(self.max_lifetime, || {
            DEVICE.create_kernel::<fn()>(&track!(|| {
                set_block_size([8, 8, 1]);
                let world_pos = core.get_world_pos(dispatch_id().xy());

                let lod = self.cache.estimate_lod(world_pos);
                let probe = self.cache.quantize(world_pos, lod, 0.expr());

                let node = self.cache.get(probe);
                self.probe_lifetime.write(node.index, self.max_lifetime);
            }))
        })
        .dispatch_async(core.screen_dispatch_size())
        .debug("Init Probes Cascade 0: Lifetime")
    }
    pub fn set_lifetime(&self, cascade: u32) -> NodeConfigs<'static> {
        dispatch1(
            track!(|cascade| {
                let index = dispatch_id().x + self.cache.get_index_offset(cascade);
                let key = self.cache.table.data_key.read(index);
                if key == crate::table::INVALID_KEY {
                    return;
                }
                let probe = self.cache.decode(key, cascade);
                let node = self.cache.load_index(probe, index);
                let next = self.cache.get(probe.next_nearest());
                self.probe_lifetime
                    .atomic_ref(next.index)
                    .fetch_max(self.probe_lifetime.read(node.index));
            }),
            [self.cache.max_probes_at_cascade(cascade - 1), 1, 1],
            cascade - 1,
        )
        .debug(format!("Init Probes Cascade {cascade}: Lifetime"))
    }
    pub fn compute_average_pos_normal(&self, core: &CoreData) -> NodeConfigs<'static> {
        (
            dispatch0(
                track!(|| {
                    self.average_pos_normal
                        .write(dispatch_id().x, AveragePosNormal::default());
                }),
                [self.average_pos_normal.len() as u32, 1, 1],
            )
            .debug("Reset Average Position and Normal"),
            dispatch0(
                track!(|| {
                    set_block_size([8, 8, 1]);
                    let pixel = dispatch_id().xy();
                    let cascade = dispatch_id().z;
                    let world_pos = core.get_world_pos(pixel);
                    let normal = core.read_normal(pixel);

                    let lod = self.cache.estimate_lod(world_pos);
                    let probe = self.cache.quantize(world_pos, lod, cascade);
                    let node = self.cache.get(probe);

                    self.average_pos_normal
                        .atomic_ref(node.index)
                        .add_sample(world_pos, normal);
                }),
                [
                    core.screen_size().x,
                    core.screen_size().y,
                    self.num_cascades(),
                ],
            )
            .debug("Compute Average Position and Normal"),
        )
            .chain()
    }
}
