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
    pub fn grid_to_dir(&self, index: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        let v = (index.cast_f32() + 0.5) / self.resolution.cast_f32() * 2.0 - 1.0;
        OctahedralVector::expr(v).decode()
    }
    #[tracked]
    pub fn dir_to_grid(&self, dir: Expr<Vec3<f32>>) -> Expr<Vec2<u32>> {
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
    pub fn linear_to_dir(&self, index: Expr<u32>) -> Expr<Vec3<f32>> {
        self.grid_to_dir(self.linear_to_grid(index))
    }
    #[tracked]
    pub fn dir_to_linear(&self, dir: Expr<Vec3<f32>>) -> Expr<u32> {
        self.grid_to_linear(self.dir_to_grid(dir))
    }
    #[tracked]
    pub fn linear_size(&self) -> Expr<u32> {
        self.resolution * self.resolution
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct LoadedNode {
    pub probe: Probe,
    // little bit redundant, but whatever.
    pub encoder: OctahedralEncoder,
    pub data_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct Node {
    pub position: u64,
    // Still no solution to do this without spinning.
    // pub data_index: u32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
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
    pub fn num_data(&self) -> u32 {
        self.base_nodes * self.base_angular_resolution.pow(2)
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
    pub fn read_grid(&self, node: Expr<LoadedNode>, dir: Expr<Vec2<u32>>) -> Expr<RadianceDeposit> {
        self.data
            .read(node.data_offset + node.encoder.grid_to_linear(dir))
    }
    #[tracked]
    pub fn read_node(&self, level: Expr<u32>, node_index: Expr<u32>) -> Expr<Node> {
        let node_offset = self.get_node_offset(level);
        self.nodes.read(node_offset + node_index)
    }
    #[tracked]
    pub fn quantize(&self, pos: Expr<Vec3<f32>>) -> Expr<Vec3<u32>> {
        ((pos / self.base_grid_size).floor().cast_i32() - self.offset).cast_u32()
    }
    #[tracked]
    pub fn load_index(&self, probe: Expr<Probe>, index: Expr<u32>) -> Expr<LoadedNode> {
        let encoder = self.get_encoder_by_level(probe.level);
        let data_offset = probe.level * self.num_data() + index * encoder.linear_size();
        LoadedNode::from_comps_expr(LoadedNodeComps {
            probe,
            encoder,
            data_offset,
        })
    }
    #[tracked]
    pub fn query<T: Aggregate>(
        &self,
        probe: Expr<Probe>,
        success: impl Fn(Expr<LoadedNode>) -> T,
        fail: impl Fn() -> T,
    ) -> T {
        let node_offset = self.get_node_offset(probe.level);
        let table_size = self.base_nodes >> (2 * probe.level);
        let node_index = (pcg3d(probe.position).x % table_size).var();
        let position = encode_position(probe.position);
        let loaded = false.var();
        for _i in 0..10 {
            let node = self.nodes.read(node_offset + node_index);
            if node.position == position {
                // Success.
                *loaded = true;
                break;
            } else if node.position == POSITION_INVALID {
                // Fail.
                break;
            }
            *node_index = (node_index + 1) % table_size;
        }
        if loaded {
            success(self.load_index(probe, **node_index))
        } else {
            fail()
        }
    }
    #[tracked]
    pub fn query_init<T: Aggregate>(
        &self,
        probe: Expr<Probe>,
        success: impl Fn(Expr<LoadedNode>) -> T,
        fail: impl Fn() -> T,
    ) -> T {
        let node_offset = self.get_node_offset(probe.level);
        let table_size = self.base_nodes >> (2 * probe.level);
        let node_index = (pcg3d(probe.position).x % table_size).var();
        let position = encode_position(probe.position);
        // TODO: This should probably always be true if the hashmap works but just in case.
        let loaded = false.var();
        for _i in 0..10 {
            let node = self.nodes.atomic_ref(node_offset + node_index);
            let prev = node.position.compare_exchange(POSITION_INVALID, position);
            if prev == POSITION_INVALID || prev == position {
                // Success.
                *loaded = true;
                break;
            }
            *node_index = (node_index + 1) % table_size;
        }
        if loaded {
            success(self.load_index(probe, **node_index))
        } else {
            fail()
        }
    }
    #[tracked]
    pub fn query_diffuse(&self, probe: Expr<Probe>, normal: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| {
                let total_radiance = Vec4::splat(0.0_f32).var();
                for i in 0_u32.expr()..node.encoder.linear_size() {
                    let dir = node.encoder.linear_to_dir(i);
                    let weight = dir.dot(normal);
                    if weight > 0.0 {
                        let data_index = node.data_offset + i;
                        let deposit = self.data.read(data_index);
                        *total_radiance += deposit.radiance_count.xyz().extend(1.0) * weight;
                    }
                }
                // TODO: Why the hell is this necessary??? Is a rust-analyzer issue.
                let out: Expr<Vec3<f32>> = total_radiance.xyz() / total_radiance.w;
                out
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
        self.query_init(
            probe,
            |node| {
                let (fluence, dir) = f();
                let data = self
                    .data
                    .atomic_ref(node.data_offset + node.encoder.dir_to_linear(dir));
                data.deposit(fluence);
                true.expr()
            },
            || false.expr(),
        )
    }
    #[tracked]
    pub fn init(&self, probe: Expr<Probe>) -> Expr<bool> {
        self.query_init(probe, |_| true.expr(), || false.expr())
    }
    #[tracked]
    pub fn contains(&self, probe: Expr<Probe>) -> Expr<bool> {
        self.query(probe, |_| true.expr(), || false.expr())
    }
    // TODO: Deduplicate.
    #[tracked]
    pub fn contains_cost(&self, probe: Expr<Probe>) -> (Expr<bool>, Expr<u32>) {
        let node_offset = self.get_node_offset(probe.level);
        let table_size = self.base_nodes >> (2 * probe.level);
        let node_index = (pcg3d(probe.position).x % table_size).var();
        let position = encode_position(probe.position);
        let loaded = false.var();
        let i = 0_u32.var();
        for j in 0..10_u32 {
            *i = j;
            let node = self.nodes.read(node_offset + node_index);
            if node.position == position {
                // Success.
                *loaded = true;
                break;
            } else if node.position == POSITION_INVALID {
                // Fail.
                break;
            }
            *node_index = (node_index + 1) % table_size;
        }
        (**loaded, **i)
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
