use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::normal::OctahedralVector;
use yesod::rand::pcg3d;
use yesod::utils::next_float_down_pos;

// type Fluence = fluence::Fluence<fluence::RgbF16>;
type Radiance = Vec3<f32>;
type Fluence = Vec4<f32>;

const NUM_BITS: u64 = 21;
const MASK: u32 = (1 << NUM_BITS) - 1;
pub const POSITION_INVALID: u64 = u64::MAX;

// TODO: This really isn't that good.
#[tracked]
fn hash64(key: Expr<u64>) -> Expr<u32> {
    pcg3d(Vec3::expr((key >> 32).cast_u32(), key.cast_u32(), 937)).x
}

#[tracked]
pub fn encode_position(cell: Expr<Vec3<u32>>) -> Expr<u64> {
    (cell.x.cast_u64() << (NUM_BITS * 2)) | (cell.y.cast_u64() << NUM_BITS) | cell.z.cast_u64()
}
#[tracked]
pub fn decode_position(pos: Expr<u64>) -> Expr<Vec3<u32>> {
    Vec3::expr(
        (pos >> (NUM_BITS * 2)).cast_u32(),
        (pos >> NUM_BITS).cast_u32() & MASK,
        pos.cast_u32() & MASK,
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, Default)]
pub struct RadianceDeposit {
    // actually pretty likely that the transmittance is 1 / the fluence is empty.
    pub fluence: Vec4<f32>, // xyz: partial fluence, w: transmittance (multiplied by count)
    pub radiance_count: Vec4<f32>, // xyz: radiance, w: count
}
impl RadianceDepositAtomicRef {
    #[tracked]
    pub fn deposit(&self, fluence: Expr<Vec4<f32>>) {
        self.fluence.x.fetch_add(fluence.x);
        self.fluence.y.fetch_add(fluence.y);
        self.fluence.z.fetch_add(fluence.z);
        self.fluence.w.fetch_add(fluence.w);
        self.radiance_count.w.fetch_add(1.0);
    }
}
impl RadianceDepositVar {
    #[tracked]
    pub fn deposit(&self, fluence: Expr<Vec4<f32>>) {
        *self.fluence += fluence;
        *self.radiance_count.w += 1.0;
    }
    #[tracked]
    pub fn merge(&self, far: Expr<Radiance>) {
        let fluence = self.fluence / self.radiance_count.w;
        let radiance = fluence.xyz() + fluence.w * far;
        *self.radiance_count.x = radiance.x;
        *self.radiance_count.y = radiance.y;
        *self.radiance_count.z = radiance.z;
    }
    #[tracked]
    pub fn reduce(&self, factor: Expr<f32>, min_count: Expr<f32>) {
        if self.radiance_count.w > min_count {
            *self.fluence *= factor;
            *self.radiance_count.w *= factor;
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct OctahedralEncoder {
    pub resolution: u32,
}
impl OctahedralEncoderExpr {
    #[tracked]
    pub fn get_grid_index(&self, dir: Expr<Vec3<f32>>) -> Expr<Vec2<u32>> {
        let v = OctahedralVector::encode(dir);
        ((v.value + 1.0) / 2.0 * next_float_down_pos((self.resolution).cast_f32()))
            .floor()
            .cast_u32()
    }
    #[tracked]
    pub fn grid_to_linear(&self, index: Expr<Vec2<u32>>) -> Expr<u32> {
        index.x * self.resolution + index.y
    }
    #[tracked]
    pub fn linear_to_grid(&self, index: Expr<u32>) -> Expr<Vec2<u32>> {
        Vec2::expr(index / self.resolution, index % self.resolution)
    }
    #[tracked]
    pub fn get_linear_index(&self, dir: Expr<Vec3<f32>>) -> Expr<u32> {
        let index = self.get_grid_index(dir);
        self.grid_to_linear(index)
    }
    #[tracked]
    pub fn linear_size(&self) -> Expr<u32> {
        self.resolution * self.resolution
    }
    #[tracked]
    pub fn foreach(&self, f: impl Fn(Expr<Vec3<f32>>, Expr<u32>)) {
        for index in 0_u32.expr()..(self.resolution * self.resolution) {
            let dir = self.linear_to_grid(index);
            let dir = (dir.cast_f32() + 0.5) / self.resolution.cast_f32();
            let dir = dir * 2.0 - 1.0;
            let dir = OctahedralVector::expr(dir).decode();
            f(dir, index);
        }
    }
}

// TODO: SoA.
#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct Node {
    pub position: u64,
    // can be used to decide when to remove nodes.
    // pub total_count: f32,
}
impl Default for Node {
    fn default() -> Self {
        Self {
            position: POSITION_INVALID,
            // total_count: 0.0,
        }
    }
}
// impl NodeVar {
//     #[tracked]
//     pub fn reduce(&self, factor: Expr<f32>, cutoff: Expr<f32>) {
//         if self.radiance_count.w > cutoff {
//             *self.radiance_count *= factor;
//         } else {
//             *self.radiance_count = Vec4::splat(0.0);
//             *self.position = POSITION_INVALID;
//         }
//     }
// }
// assert_eq_size!(Node, [f32; 8]);

#[derive(Debug, Clone, Copy, Value, PartialEq)]
#[repr(C)]
pub struct Probe {
    pub position: Vec3<u32>,
    pub level: u32,
}
impl Probe {
    #[tracked]
    pub fn base(pos: Expr<Vec3<u32>>) -> Expr<Self> {
        Self::from_comps_expr(ProbeComps {
            position: pos,
            level: 0_u32.expr(),
        })
    }
}
impl ProbeExpr {
    #[tracked]
    pub fn next_nearest(&self) -> Expr<Probe> {
        self.shift_level(1.expr())
    }
    #[tracked]
    pub fn shift_level(&self, delta: Expr<u32>) -> Expr<Probe> {
        Probe::from_comps_expr(ProbeComps {
            position: self.position >> delta,
            level: self.level + delta,
        })
    }
}

#[derive(Debug)]
pub struct HashCache {
    pub base_grid_size: f32,
    pub base_angular_resolution: u32,
    pub offset: Vec3<i32>,
    pub base_nodes: u32,
    pub num_levels: u32,
    pub nodes: Buffer<Node>,
    pub data: Buffer<RadianceDeposit>,
}
impl HashCache {
    pub fn new(
        base_grid_size: f32,
        base_angular_resolution: u32,
        base_nodes: u32,
        num_levels: u32,
    ) -> Self {
        assert!(base_nodes.is_power_of_two());
        Self {
            base_grid_size,
            base_angular_resolution,
            offset: Vec3::splat(-(1 << (NUM_BITS - 1))),
            base_nodes,
            num_levels,
            nodes: DEVICE.create_buffer_from_fn(
                (0..num_levels)
                    .map(|level| base_nodes >> (level * 2))
                    .sum::<u32>() as usize,
                |_| Node::default(),
            ),
            data: DEVICE.create_buffer_from_fn(
                (base_angular_resolution.pow(2) * base_nodes * num_levels) as usize,
                |_| RadianceDeposit::default(),
            ),
        }
    }
    pub fn num_nodes_at_level(&self, level: u32) -> u32 {
        self.base_nodes >> (level * 2)
    }
    #[tracked]
    pub fn get_encoder_by_level(&self, level: Expr<u32>) -> Expr<OctahedralEncoder> {
        OctahedralEncoder::from_comps_expr(OctahedralEncoderComps {
            resolution: self.base_angular_resolution << level,
        })
    }
    // Probably can precompute.
    #[tracked]
    pub fn get_node_offset(&self, level: Expr<u32>) -> Expr<u32> {
        // base_nodes + base_nodes / 4 + base_nodes / 16 + ... (level times)
        let pow4s = (0..16).map(|i| 1_u32 << (2 * i)).sum::<u32>();
        (pow4s & ((1 << (2 * level)) - 1)) << (self.base_nodes.trailing_zeros() - level * 2 + 2)
    }
    #[tracked]
    pub fn get_data_index_raw(
        &self,
        node_index: Expr<u32>,
        level: Expr<u32>,
        dir: Expr<u32>,
    ) -> Expr<u32> {
        let encoder = self.get_encoder_by_level(level);
        (node_index - self.get_node_offset(level)) * encoder.linear_size()
            + dir
            + level * self.base_nodes
    }
    #[tracked]
    pub fn get_data_index(
        &self,
        node_index: Expr<u32>,
        level: Expr<u32>,
        dir: Expr<Vec3<f32>>,
    ) -> Expr<u32> {
        let encoder = self.get_encoder_by_level(level);
        (node_index - self.get_node_offset(level)) * encoder.linear_size()
            + encoder.get_linear_index(dir)
            + level * self.base_nodes
    }
    #[tracked]
    pub fn get_node_index(&self, probe: Expr<Probe>) -> Expr<u32> {
        let node_offset = self.get_node_offset(probe.level);
        pcg3d(probe.position).x % (self.base_nodes >> (2 * probe.level)) + node_offset
    }
    #[tracked]
    pub fn quantize(&self, pos: Expr<Vec3<f32>>) -> Expr<Vec3<u32>> {
        ((pos / self.base_grid_size).floor().cast_i32() - self.offset).cast_u32()
    }
    #[tracked]
    pub fn query_node<T: Aggregate>(
        &self,
        probe: Expr<Probe>,
        success: impl Fn(Expr<u32>) -> T,
        fail: impl Fn() -> T,
    ) -> T {
        let node_index = self.get_node_index(probe);
        let node = self.nodes.read(node_index);
        if node.position == encode_position(probe.position) {
            success(node_index)
        } else {
            fail()
        }
    }
    // #[tracked]
    // pub fn query(&self, probe: Expr<Probe>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
    //     let node_index = self.get_node_index(probe);
    //     let node = self.nodes.read(node_index);
    //     if node.position != encode_position(probe.position) {
    //         Vec3::splat_expr(-1.0)
    //     } else {
    //         let data_index = self.get_data_index(node_index, probe.level, dir);
    //         self.data.read(data_index).radiance_count.xyz()
    //     }
    // }
    #[tracked]
    pub fn query_diffuse(&self, pos: Expr<Vec3<f32>>, normal: Expr<Vec3<f32>>) -> Expr<Radiance> {
        let probe = Probe::from_comps_expr(ProbeComps {
            position: self.quantize(pos),
            level: 0_u32.expr(),
        });
        self.query_node(
            probe,
            |node_index| {
                let data_offset = node_index * self.base_angular_resolution.pow(2);
                let encoder = self.get_encoder_by_level(0.expr());
                let total_radiance = Vec4::splat(0.0_f32).var();
                encoder.foreach(|dir, index| {
                    let weight = dir.dot(normal);
                    if weight > 0.0 {
                        let data_index = data_offset + index;
                        let deposit = self.data.read(data_index);
                        *total_radiance += deposit.radiance_count.xyz().extend(1.0) * weight;
                    }
                });
                total_radiance.xyz() / total_radiance.w
            },
            || Vec3::splat_expr(-1.0),
        )
    }
    // TODO: This doesn't really seem to balance out.
    #[tracked]
    pub fn try_deposit(
        &self,
        probe: Expr<Probe>,
        f: impl Fn() -> (Expr<Fluence>, Expr<Vec3<f32>>),
    ) -> Expr<bool> {
        let node_index = self.get_node_index(probe);
        let node = self.nodes.atomic_ref(node_index);
        let pos_enc = encode_position(probe.position);
        let prev = node.position.compare_exchange(POSITION_INVALID, pos_enc);
        if prev == POSITION_INVALID || prev == pos_enc {
            let (fluence, dir) = f();
            let data_index = self.get_data_index(node_index, probe.level, dir);
            let data = self.data.atomic_ref(data_index);
            data.deposit(fluence);
            true.expr()
        } else {
            false.expr()
        }
    }
    #[tracked]
    pub fn deposit(
        &self,
        probe: Expr<Probe>,
        fluence: Expr<Fluence>,
        dir: Expr<Vec3<f32>>,
    ) -> Expr<bool> {
        self.try_deposit(probe, || (fluence, dir))
    }
}
