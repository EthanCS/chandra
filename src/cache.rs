#![allow(unused_parens)]

use keter::graph::NodeConfigs;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::camera::View;
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::utils::next_float_down_pos_2;

use crate::table::HashTable;

mod ext;

// Should be enough assuming LODs actually do their job.
const NUM_BITS: u32 = 8;
const MASK: u32 = (1 << NUM_BITS) - 1;

#[tracked]
pub fn encode_position(cell: Expr<Vec4<u32>>) -> Expr<u32> {
    cell.w << (NUM_BITS * 3) | (cell.x << (NUM_BITS * 2)) | (cell.y << NUM_BITS) | cell.z
}

#[tracked]
pub fn decode_position(pos: Expr<u32>) -> Expr<Vec4<u32>> {
    Vec4::expr(
        (pos >> (NUM_BITS * 2)) & MASK,
        (pos >> NUM_BITS) & MASK,
        pos & MASK,
        pos >> (NUM_BITS * 3),
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct RadianceDeposit {
    // actually pretty likely that the transmittance is 1 / the fluence is empty.
    pub fluence: Vec4<f32>, // xyz: partial fluence, w: transmittance (multiplied by weight = 1.0)
}
impl RadianceDeposit {
    #[tracked]
    pub fn expr(fluence: Expr<Vec4<f32>>) -> Expr<Self> {
        Self::from_comps_expr(RadianceDepositComps { fluence })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct RadianceElement {
    // actually pretty likely that the transmittance is 1 / the fluence is empty.
    pub fluence: Vec4<f32>, // xyz: partial fluence, w: transmittance (multiplied by weight)
    pub radiance_weight: Vec4<f32>, // xyz: radiance, w: weight
}
impl Default for RadianceElement {
    fn default() -> Self {
        Self {
            fluence: Vec4::splat(0.0),
            radiance_weight: Vec4::new(0.0, 0.0, 0.0, 0.0),
        }
    }
}
impl RadianceElementExpr {
    pub fn radiance(&self) -> Expr<Vec3<f32>> {
        self.radiance_weight.xyz()
    }
}
impl RadianceElementAtomicRef {
    #[tracked]
    pub fn deposit(&self, deposit: Expr<RadianceDeposit>) {
        self.fluence.x.fetch_add(deposit.fluence.x);
        self.fluence.y.fetch_add(deposit.fluence.y);
        self.fluence.z.fetch_add(deposit.fluence.z);
        self.fluence.w.fetch_add(deposit.fluence.w);
        self.radiance_weight.w.fetch_add(1.0);
    }
}
impl RadianceElementVar {
    #[tracked]
    pub fn deposit(&self, deposit: Expr<RadianceDeposit>) {
        *self.fluence += deposit.fluence;
        *self.radiance_weight.w += 1.0;
    }
    #[tracked]
    pub fn merge(&self, far: Expr<Vec3<f32>>) {
        let fluence = self.fluence / self.radiance_weight.w;
        let radiance = fluence.xyz() + fluence.w * far;
        *self.radiance_weight.x = radiance.x;
        *self.radiance_weight.y = radiance.y;
        *self.radiance_weight.z = radiance.z;
    }
    #[tracked]
    pub fn decay(&self, factor: Expr<f32>, min_count: Expr<f32>) {
        if self.radiance_weight.w > min_count {
            *self.fluence *= factor;
            *self.radiance_weight.w *= factor;
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
        index.x * self.grid_size().y + index.y
    }
    #[tracked]
    pub fn linear_to_grid(&self, index: Expr<u32>) -> Expr<Vec2<u32>> {
        Vec2::expr(index / self.grid_size().y, index % self.grid_size().y)
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
pub struct LoadedProbe {
    pub probe: Probe,
    pub data_offset: u32,
    pub index: u32,
}

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
    pub fn next_bilinear(&self) -> [(Expr<Probe>, Expr<f32>); 8] {
        let base_position = (self.position.xyz() - 1) >> 1;
        let offset = self.position.xyz() % 2;
        std::array::from_fn(|i| {
            let mask = Vec3::new(
                if (i & 4) != 0 { 1 } else { 0 },
                if (i & 2) != 0 { 1 } else { 0 },
                if (i & 1) != 0 { 1 } else { 0 },
            );
            let position = base_position + mask;
            let weight = (mask != offset)
                .select(Vec3::splat_expr(0.75_f32), Vec3::splat_expr(0.25))
                .reduce_prod();
            (
                Probe::from_comps_expr(ProbeComps {
                    position: position.extend(self.position.w),
                    cascade: self.cascade + 1,
                }),
                weight,
            )
        })
    }
    #[tracked]
    pub fn next_nearest(&self) -> Expr<Probe> {
        self.shift_level(1.expr())
    }
    #[tracked]
    pub fn shift_level(&self, delta: Expr<u32>) -> Expr<Probe> {
        Probe::from_comps_expr(ProbeComps {
            position: (self.position.xyz() >> delta).extend(self.position.w),
            cascade: self.cascade + delta,
        })
    }
}

const INTERP_RADIUS: f32 = 0.1;

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
    pub offset: Vec3<i32>,
    pub base_probes: u32,
    pub num_levels: u32,
    pub projection: Singleton<Projection>,
    pub table: HashTable,
    pub data: Buffer<RadianceElement>,
}
impl HashCache {
    pub fn new(
        base_grid_size: f32,
        base_radius: f32,
        base_angular_resolution: u32,
        base_probes: u32,
        num_levels: u32,
    ) -> Self {
        assert!(base_probes.is_power_of_two());
        Self {
            base_grid_size,
            base_radius,
            base_angular_resolution,
            offset: Vec3::splat(-(1 << (NUM_BITS - 1))),
            base_probes,
            num_levels,
            projection: Singleton::new(Projection::default()),

            table: HashTable::new(
                num_levels,
                base_probes * 4, // Can use an overly large amount to reduce collisions.
                (0..num_levels).map(|level| base_probes >> (level * 2)),
            ),
            data: DEVICE.create_buffer_from_fn(
                (base_angular_resolution.pow(2)
                    * INDEX_SCALING.x
                    * INDEX_SCALING.y
                    * base_probes
                    * num_levels) as usize,
                |_| RadianceElement::default(),
            ),
        }
    }
    pub fn set_view(&self, view: View) -> NodeConfigs<'static> {
        self.projection
            .write_host(Projection { view_pos: view.pos })
    }
    pub fn num_nodes_at_cascade(&self, cascade: u32) -> u32 {
        self.base_probes >> (cascade * 2)
    }
    pub fn num_data(&self) -> u32 {
        self.base_probes * self.base_angular_resolution.pow(2) * INDEX_SCALING.x * INDEX_SCALING.y
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
    // Probably can precompute.
    #[tracked]
    pub fn get_index_offset(&self, cascade: Expr<u32>) -> Expr<u32> {
        // base_nodes + base_nodes / 4 + base_nodes / 16 + ... (level times)
        let pow4s = (0..16).map(|i| 1_u32 << (2 * i)).sum::<u32>();
        (pow4s & ((1 << (2 * cascade)) - 1))
            << (self.base_probes.trailing_zeros() - cascade * 2 + 2)
    }
    #[tracked]
    pub fn get_lod(&self, pos: Expr<Vec3<f32>>) -> Expr<u32> {
        let radius = (pos - self.projection().view_pos).length();
        keter::max((radius / self.base_radius).log2(), 0.0)
            .ceil()
            .cast_u32()
    }
    #[tracked]
    pub fn read_grid(
        &self,
        node: Expr<LoadedProbe>,
        dir: Expr<Vec2<u32>>,
    ) -> Expr<RadianceElement> {
        self.data
            .read(node.data_offset + self.get_encoder(node.probe.cascade).grid_to_linear(dir))
    }
    #[tracked]
    pub fn quantize(&self, pos: Expr<Vec3<f32>>, lod: Expr<u32>) -> Expr<Probe> {
        let spacing = self.spacing(0.expr(), lod);

        let pos = (pos / spacing).floor().cast_i32();
        Probe::base((pos - self.offset).cast_u32().extend(lod))
    }
    #[tracked]
    pub fn dequantize(&self, probe: Expr<Probe>) -> Expr<Vec3<f32>> {
        let local_pos = ((probe.position.xyz() << probe.cascade).cast_i32() + self.offset)
            .cast_f32()
            + 0.5 * (1 << probe.cascade).cast_f32();
        let spacing = self.spacing(probe.cascade, probe.position.w);
        local_pos * spacing
    }
    #[tracked]
    pub fn spacing_probe(&self, probe: Expr<Probe>) -> Expr<f32> {
        self.spacing(probe.cascade, probe.position.w)
    }
    #[tracked]
    pub fn spacing(&self, cascade: Expr<u32>, lod: Expr<u32>) -> Expr<f32> {
        self.base_grid_size * (1_u32 << (cascade + lod)).cast_f32()
    }
    #[tracked]
    pub fn get_position(&self, encoded_pos: Expr<u32>, cascade: Expr<u32>) -> Expr<Vec4<u32>> {
        let pos = decode_position(encoded_pos);
        let view_pos = self.projection().view_pos;
        let view_pos = (view_pos / self.spacing(cascade, pos.w))
            .floor()
            .cast_i32()
            .cast_u32();
        let diff = (pos.xyz() - view_pos).cast_u8().cast_i8().cast_i32();
        (view_pos + diff.cast_u32()).extend(pos.w)
    }
    #[tracked]
    pub fn load_index(&self, probe: Expr<Probe>, index: Expr<u32>) -> Expr<LoadedProbe> {
        let encoder = self.get_encoder(probe.cascade);
        let data_offset = probe.cascade * self.num_data()
            + (index - self.get_index_offset(probe.cascade)) * encoder.linear_size();
        LoadedProbe::from_comps_expr(LoadedProbeComps {
            probe,
            data_offset,
            index,
        })
    }
    #[tracked]
    pub fn insert(&self, probe: Expr<Probe>) {
        self.table
            .insert(encode_position(probe.position), probe.cascade);
    }
    #[tracked]
    pub fn get(&self, probe: Expr<Probe>) -> Expr<LoadedProbe> {
        let index = self
            .table
            .get(encode_position(probe.position), probe.cascade);
        self.load_index(probe, index)
    }
}
