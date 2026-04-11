#![allow(unused_parens)]

use keter::graph::NodeConfigs;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::camera::View;
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::utils::{FetchAddVector, decode_morton2_16, encode_morton2_16, next_float_down_pos_2};

use crate::table::{HashTable, INVALID_KEY, Key};
use crate::utils::{corner_to_index, index_to_corner};

// Should be enough assuming LODs actually do their job.
#[cfg(feature = "small-key")]
const NUM_BITS: Key = 8;
#[cfg(not(feature = "small-key"))]
const NUM_BITS: Key = 18;
const MASK: Key = (1 << NUM_BITS) - 1;

const OFFSET: u32 = 1 << 31;

pub type Radiance = Vec3<f32>;
pub type Fluence = Vec4<f32>;

#[tracked]
fn encode_position(cell: Expr<Vec4<u32>>) -> Expr<Key> {
    #[cfg(not(feature = "small-key"))]
    let cell = cell.cast_u64();
    let cell = cell & MASK;
    cell.w << (NUM_BITS * 3) | (cell.z << (NUM_BITS * 2)) | (cell.y << NUM_BITS) | cell.x
}

#[tracked]
fn decode_position(pos: Expr<Key>) -> Expr<Vec4<u32>> {
    let result = Vec4::expr(
        pos & MASK,
        (pos >> NUM_BITS) & MASK,
        (pos >> (NUM_BITS * 2)) & MASK,
        pos >> (NUM_BITS * 3),
    );
    #[cfg(not(feature = "small-key"))]
    let result = result.cast_u32();
    result
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct RadianceElement {
    // actually pretty likely that the transmittance is 1 / the fluence is empty.
    pub weighted_fluence: Vec4<f32>, // xyz: unnormalized fluence, w: transmittance (multiplied by weight)
    pub weight: f32,                 // weight
}
impl Default for RadianceElement {
    fn default() -> Self {
        Self {
            weighted_fluence: Vec4::splat(0.0),
            weight: 0.0,
        }
    }
}
impl RadianceElementExpr {
    // pub fn radiance(&self) -> Expr<Radiance> {
    //     self.radiance_weight.xyz()
    // }
    #[tracked]
    pub fn fluence(&self) -> Expr<Fluence> {
        self.weighted_fluence / self.weight
    }

    #[tracked]
    pub fn merge(&self, far: Expr<Radiance>) -> Expr<Vec4<f32>> {
        // (self.weighted_fluence.xyz() + self.weighted_fluence.w * far).extend(self.weight)
        if self.weight < 0.0001 {
            Vec4::splat_expr(0.0)
        } else {
            let fluence = self.fluence();
            (fluence.xyz() + fluence.w * far).extend(1.0)
        }
    }
}
impl RadianceElementAtomicRef {
    #[tracked]
    pub fn deposit(&self, fluence: Expr<Fluence>) {
        self.weighted_fluence.fetch_add(fluence);
        self.weight.fetch_add(1.0);
    }
}
impl RadianceElementVar {
    #[tracked]
    pub fn deposit(&self, fluence: Expr<Fluence>) {
        *self.weighted_fluence += fluence;
        *self.weight += 1.0;
    }
    #[tracked]
    pub fn decay(&self, factor: Expr<f32>, min_count: Expr<f32>) {
        if self.weight > min_count {
            *self.weighted_fluence *= factor;
            *self.weight *= factor;
        }
    }
}

pub type Encoder = SphericalEncoder;
pub const INDEX_SCALING: Vec2<u32> = Vec2::new(2, 1);

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct DirectionIndexer {
    pub resolution: u32,
}
impl DirectionIndexerExpr {
    #[tracked]
    pub fn grid_to_dir_f32(&self, index: Expr<Vec2<f32>>) -> Expr<Vec3<f32>> {
        let v = index / self.grid_size().cast_f32();
        Encoder::decode(v)
    }
    #[tracked]
    pub fn grid_to_dir(&self, index: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        self.grid_to_dir_f32(index.cast_f32() + 0.5)
    }
    #[tracked]
    pub fn dir_to_grid(&self, dir: Expr<Vec3<f32>>) -> Expr<Vec2<u32>> {
        let v = Encoder::encode(dir);
        (v * next_float_down_pos_2(self.grid_size().cast_f32()))
            .floor()
            .cast_u32()
    }
    #[tracked]
    pub fn grid_to_linear(&self, index: Expr<Vec2<u32>>) -> Expr<u32> {
        let index = index % self.grid_size();
        encode_morton2_16(index)
        // index.x * self.grid_size().y + index.y
    }
    #[tracked]
    pub fn linear_to_grid(&self, index: Expr<u32>) -> Expr<Vec2<u32>> {
        decode_morton2_16(index)
        // Vec2::expr(index / self.grid_size().y, index % self.grid_size().y)
    }
    #[tracked]
    pub fn linear_to_dir(&self, index: Expr<u32>) -> Expr<Vec3<f32>> {
        self.grid_to_dir(self.linear_to_grid(index))
    }
    #[tracked]
    pub fn dir_to_linear(&self, dir: Expr<Vec3<f32>>) -> Expr<u32> {
        self.grid_to_linear(self.dir_to_grid(dir))
    }
    #[tracked]
    pub fn linear_size(&self) -> Expr<u32> {
        self.grid_size().reduce_prod()
    }
    #[tracked]
    pub fn grid_size(&self) -> Expr<Vec2<u32>> {
        self.resolution * INDEX_SCALING
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct LoadedNode {
    pub probe: Probe,
    // position in data buffer
    // 0 if not loaded so that things don't go out of bounds
    pub data_offset: u32,
    // index in the table
    // if u32::MAX, this isn't loaded
    pub index: u32,
}
impl LoadedNodeExpr {
    #[tracked]
    pub fn invalid(&self) -> Expr<bool> {
        self.index == u32::MAX.expr()
    }
    #[tracked]
    pub fn loaded(&self) -> Expr<bool> {
        self.index != u32::MAX.expr()
    }
}

/*
Values are offset by 1 << 31 to avoid negative offsets.
So a probe at 0 would have position 1<<31, regardless of the cascade.
*/
#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
pub struct Probe {
    pub position: Vec4<u32>,
    pub cascade: u32,
}
impl Probe {
    #[tracked]
    pub fn base(pos: Expr<Vec4<u32>>) -> Expr<Self> {
        Self::from_comps_expr(ProbeComps {
            position: pos,
            cascade: 0_u32.expr(),
        })
    }
}
impl ProbeExpr {
    #[tracked]
    pub fn next_bilinear(&self, i: Expr<u32>) -> (Expr<Probe>, Expr<f32>) {
        let base_pos = self.position.xyz() & !1_u32;
        let j = corner_to_index(self.position.xyz() & 1_u32);
        let base_probe = Probe::from_comps_expr(ProbeComps {
            position: base_pos.extend(self.position.w),
            cascade: self.cascade,
        })
        .next_nearest();
        let pos = base_probe.position.xyz() + index_to_corner(i);
        let next_probe = Probe::from_comps_expr(ProbeComps {
            position: pos.extend(base_probe.position.w),
            cascade: base_probe.cascade,
        });

        let corner_weights = std::array::from_fn::<_, 8, _>(|i| {
            let id = (i & 1) + ((i >> 1_u32) & 1) + ((i >> 2_u32) & 1);
            0.25_f32.powi(id as i32) * 0.75_f32.powi((3 - id) as i32)
        });

        (next_probe, corner_weights.expr().read(i ^ j))
    }
    #[tracked]
    pub fn next_nearest(&self) -> Expr<Probe> {
        self.shift_level(1.expr())
    }
    // All values that have `next_nearest` equal to this.
    #[tracked]
    pub fn last_nearest(&self, i: Expr<u32>) -> Expr<Probe> {
        let offset = index_to_corner(i);
        let bias = OFFSET / 2;
        let base = (self.position.xyz() - bias) << 1;
        let pos = base + offset;
        Probe::from_comps_expr(ProbeComps {
            position: pos.extend(self.position.w),
            cascade: self.cascade - 1,
        })
    }
    #[tracked]
    pub fn shift_level(&self, delta: Expr<u32>) -> Expr<Probe> {
        let bias = OFFSET - (OFFSET >> delta);
        Probe::from_comps_expr(ProbeComps {
            position: ((self.position.xyz() >> delta) + bias).extend(self.position.w),
            cascade: self.cascade + delta,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, Default)]
pub struct Projection {
    view_pos: Vec3<f32>,
}

#[derive(Debug)]
pub struct HashCache {
    pub base_grid_size: f32,
    pub base_radius: f32,
    pub base_angular_resolution: u32,
    pub base_probes: u32,
    pub num_cascades: u32,
    pub projection: Singleton<Projection>,
    pub table: HashTable,
    pub weighted_fluence: Buffer<Vec4<f32>>,
    pub weights: Buffer<f32>,
}
impl HashCache {
    pub fn new(
        base_grid_size: f32,
        base_radius: f32,
        base_angular_resolution: u32,
        base_probes: u32,
        num_cascades: u32,
    ) -> Self {
        assert!(base_probes.is_power_of_two());
        let num_data = (base_angular_resolution.pow(2)
            * INDEX_SCALING.x
            * INDEX_SCALING.y
            * base_probes
            * num_cascades) as usize;
        Self {
            base_grid_size,
            base_radius,
            base_angular_resolution,
            base_probes,
            num_cascades,
            projection: Singleton::new(Projection::default()),

            table: HashTable::new(
                num_cascades,
                base_probes * 4, // Can use an overly large amount to reduce collisions.
                (0..num_cascades).map(|level| base_probes >> (level * 2)),
            ),
            weighted_fluence: DEVICE.create_buffer_from_fn(num_data, |_| Vec4::splat(0.0)),
            weights: DEVICE.create_buffer_from_fn(num_data, |_| 0.0),
        }
    }
    pub fn set_view(&self, view: View) -> NodeConfigs<'static> {
        self.projection
            .write_host(Projection { view_pos: view.pos })
    }
    pub fn total_probes(&self) -> u32 {
        self.table.data_capacity()
    }
    pub fn max_probes_at_cascade(&self, cascade: u32) -> u32 {
        self.base_probes >> (cascade * 2)
    }
    pub fn directions_per_probe(&self, cascade: u32) -> u32 {
        self.base_angular_resolution.pow(2)
            * INDEX_SCALING.x
            * INDEX_SCALING.y
            * (1 << (cascade * 2))
    }
    pub fn total_directions_per_cascade(&self) -> u32 {
        self.base_probes * self.base_angular_resolution.pow(2) * INDEX_SCALING.x * INDEX_SCALING.y
    }
    pub fn total_directions(&self) -> u32 {
        self.total_directions_per_cascade() * self.num_cascades
    }
    pub fn projection(&self) -> Expr<Projection> {
        self.projection.read()
    }
    #[tracked]
    pub fn get_encoder(&self, cascade: Expr<u32>) -> Expr<DirectionIndexer> {
        DirectionIndexer::from_comps_expr(DirectionIndexerComps {
            resolution: self.base_angular_resolution << cascade,
        })
    }

    pub fn get_index_offset_host(&self, cascade: u32) -> u32 {
        // base_nodes + base_nodes / 4 + base_nodes / 16 + ... (level times)
        let pow4s = (0..16).map(|i| 1_u32 << (2 * i)).sum::<u32>();
        (pow4s & ((1 << (2 * cascade)) - 1))
            << (self.base_probes.trailing_zeros() - cascade * 2 + 2)
    }

    // Probably can precompute.
    #[tracked]
    pub fn get_index_offset(&self, cascade: Expr<u32>) -> Expr<u32> {
        // base_nodes + base_nodes / 4 + base_nodes / 16 + ... (level times)
        let pow4s = (0..16).map(|i| 1_u32 << (2 * i)).sum::<u32>();
        (pow4s & ((1 << (2 * cascade)) - 1))
            << (self.base_probes.trailing_zeros() - cascade * 2 + 2)
    }
    #[tracked]
    pub fn estimate_lod(&self, pos: Expr<Vec3<f32>>) -> Expr<u32> {
        let radius = (pos - self.projection().view_pos).length();
        keter::max((radius / self.base_radius).log2(), 0.0)
            .ceil()
            .cast_u32()
    }
    /*
    #[tracked]
    pub fn read_element(&self, node: Expr<LoadedNode>, index: Expr<u32>) -> Expr<RadianceElement> {
        self.data.read(node.data_offset + index)
    }
    #[tracked]
    pub fn atomic_element(
        &self,
        node: Expr<LoadedNode>,
        index: Expr<u32>,
    ) -> AtomicRef<RadianceElement> {
        self.data.atomic_ref(node.data_offset + index)
    }
     */

    #[tracked]
    pub fn deposit(&self, offset: Expr<u32>, fluence: Expr<Fluence>) {
        let fl = self.weighted_fluence.atomic_ref(offset);
        fl.fetch_add(fluence);
        self.weights.atomic_ref(offset).fetch_add(1.0);
    }
    #[tracked]
    pub fn deposit_once(&self, offset: Expr<u32>, fluence: Expr<Fluence>) {
        self.weighted_fluence.write(offset, fluence);
        self.weights.write(offset, 1.0);
    }

    #[tracked]
    pub fn quantize(
        &self,
        pos: Expr<Vec3<f32>>,
        lod: Expr<u32>,
        cascade: Expr<u32>,
    ) -> Expr<Probe> {
        let spacing = self.spacing(lod, cascade);

        let pos = (pos / spacing).floor().cast_i32();
        Probe::from_comps_expr(ProbeComps {
            position: (pos.cast_u32() + OFFSET).extend(lod),
            cascade,
        })
    }
    #[tracked]
    pub fn dequantize(&self, probe: Expr<Probe>) -> Expr<Vec3<f32>> {
        let local_pos = (probe.position.xyz() - OFFSET).cast_i32().cast_f32() + 0.5;
        let spacing = self.spacing(probe.position.w, probe.cascade);
        local_pos * spacing
    }
    #[tracked]
    pub fn probe_spacing(&self, probe: Expr<Probe>) -> Expr<f32> {
        self.spacing(probe.position.w, probe.cascade)
    }
    #[tracked]
    pub fn spacing(&self, lod: Expr<u32>, cascade: Expr<u32>) -> Expr<f32> {
        self.base_grid_size * (1_u32 << (cascade + lod)).cast_f32()
    }
    #[cfg(not(feature = "small-key"))]
    #[tracked]
    pub fn decode(&self, encoded_pos: Expr<Key>, cascade: Expr<u32>) -> Expr<Probe> {
        let pos = decode_position(encoded_pos);
        let lod = pos.w;
        let pos = pos.xyz();
        let sign = pos >> (NUM_BITS - 1) as u32;
        let sign_ext = sign * (u32::MAX - MASK as u32);
        let pos = pos | sign_ext;
        Probe::from_comps_expr(ProbeComps {
            position: (pos + OFFSET).extend(lod),
            cascade,
        })
    }
    #[cfg(feature = "small-key")]
    #[tracked]
    pub fn decode(&self, encoded_pos: Expr<Key>, cascade: Expr<u32>) -> Expr<Probe> {
        let pos = decode_position(encoded_pos);
        let view_pos = self.projection().view_pos;
        let view_pos = self.quantize(view_pos, pos.w, cascade).position.xyz();
        // let view_pos = (view_pos / self.spacing(pos.w, cascade))
        //     .floor()
        //     .cast_i32()
        //     .cast_u32()
        //     + OFFSET;
        let diff = pos.xyz() - view_pos;
        let diff = Vec3::expr(
            diff.x.cast_u8().cast_i8().cast_i32(),
            diff.y.cast_u8().cast_i8().cast_i32(),
            diff.z.cast_u8().cast_i8().cast_i32(),
        )
        .cast_u32();

        Probe::from_comps_expr(ProbeComps {
            position: (view_pos + diff).extend(pos.w),
            cascade,
        })
    }
    #[tracked]
    pub fn encode(&self, probe: Expr<Probe>) -> Expr<Key> {
        encode_position(probe.position)
    }
    #[tracked]
    pub fn load_index(&self, probe: Expr<Probe>, index: Expr<u32>) -> Expr<LoadedNode> {
        let encoder = self.get_encoder(probe.cascade);
        let data_offset = if index == u32::MAX {
            0_u32.expr()
        } else {
            probe.cascade * self.total_directions_per_cascade()
                + (index - self.get_index_offset(probe.cascade)) * encoder.linear_size()
        };
        LoadedNode::from_comps_expr(LoadedNodeComps {
            probe,
            data_offset,
            index,
        })
    }
    #[tracked]
    pub fn insert(&self, probe: Expr<Probe>, on_insert: impl Fn(Expr<u32>)) {
        self.table
            .insert(self.encode(probe), probe.cascade, on_insert);
    }
    #[tracked]
    pub fn get(&self, probe: Expr<Probe>) -> Expr<LoadedNode> {
        let index = self.table.get(self.encode(probe), probe.cascade);
        self.load_index(probe, index)
    }
    #[tracked]
    pub fn get_occupied(&self, cascade: Expr<u32>, index: Expr<u32>) -> Expr<LoadedNode> {
        let index = self.get_index_offset(cascade) + index;
        let key = self.table.data_key.read(index);
        let probe = self.decode(key, cascade);
        if key == INVALID_KEY {
            LoadedNode::from_comps_expr(LoadedNodeComps {
                probe,
                data_offset: 0.expr(),
                index: u32::MAX.expr(),
            })
        } else {
            self.load_index(probe, index)
        }
    }

    #[tracked]
    pub fn contains(&self, probe: Expr<Probe>) -> Expr<bool> {
        self.table.contains(self.encode(probe), probe.cascade)
    }
    #[tracked]
    pub fn cost(&self, probe: Expr<Probe>) -> Expr<u32> {
        self.table.cost(self.encode(probe), probe.cascade)
    }
}
