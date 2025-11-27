#![allow(unused_parens)]

use keter::graph::NodeConfigs;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::camera::View;
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::rand::pcg3d;
use yesod::utils::next_float_down_pos_2;

// type Fluence = fluence::Fluence<fluence::RgbF16>;
pub type Radiance = Vec3<f32>;
pub type Fluence = Vec4<f32>;

const NUM_BITS: u64 = 18;
const MASK: u32 = (1 << NUM_BITS) - 1;
pub const POSITION_INVALID: u64 = u64::MAX;
const NODE_LIFETIME: u32 = 10;

// TODO: This really isn't that good.
#[tracked]
fn hash64(key: Expr<u64>) -> Expr<u32> {
    pcg3d(Vec3::expr((key >> 32).cast_u32(), key.cast_u32(), 937)).x
}

#[tracked]
pub fn encode_position(cell: Expr<Vec4<u32>>) -> Expr<u64> {
    cell.w.cast_u64() << (NUM_BITS * 3)
        | (cell.x.cast_u64() << (NUM_BITS * 2))
        | (cell.y.cast_u64() << NUM_BITS)
        | cell.z.cast_u64()
}
#[tracked]
pub fn decode_position(pos: Expr<u64>) -> Expr<Vec4<u32>> {
    Vec4::expr(
        (pos >> (NUM_BITS * 2)).cast_u32() & MASK,
        (pos >> NUM_BITS).cast_u32() & MASK,
        pos.cast_u32() & MASK,
        (pos >> (NUM_BITS * 3)).cast_u32(),
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
    pub radiance_weight: Vec4<f32>, // xyz: radiance (multiplied by weight), w: weight
}
impl Default for RadianceElement {
    fn default() -> Self {
        Self {
            fluence: Vec4::splat(0.0),
            radiance_weight: Vec4::new(0.0, 0.0, 0.0, 0.001),
        }
    }
}
impl RadianceElementExpr {
    pub fn radiance(&self) -> Expr<Radiance> {
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
    pub fn merge(&self, far: Expr<Radiance>) {
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
    pub fn grid_to_dir(&self, index: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        let v = (index.cast_f32() + 0.5) / (self.resolution * INDEX_SCALING).cast_f32();
        Encoder::decode(v)
    }
    #[tracked]
    pub fn dir_to_grid(&self, dir: Expr<Vec3<f32>>) -> Expr<Vec2<u32>> {
        let v = Encoder::encode(dir);
        (v * next_float_down_pos_2((self.resolution * INDEX_SCALING).cast_f32()))
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
pub struct LoadedNode {
    pub probe: Probe,
    // little bit redundant, but whatever.
    pub encoder: DirectionIndexer,
    pub data_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct Node {
    pub position: u64,
    pub lifetime: u32,
    // Still no solution to do this without spinning.
    // pub data_index: u32,
    // can be used to decide when to remove nodes.
    // pub total_count: f32,
}
impl Default for Node {
    fn default() -> Self {
        Self {
            position: POSITION_INVALID,
            lifetime: 0,
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
    pub position: Vec4<u32>,
    pub level: u32,
}
impl Probe {
    #[tracked]
    pub fn base(pos: Expr<Vec4<u32>>) -> Expr<Self> {
        Self::from_comps_expr(ProbeComps {
            position: pos,
            level: 0_u32.expr(),
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
                    level: self.level + 1,
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
            level: self.level + delta,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct Projection {
    view: View,
    base_grid_size: f32,
    base_radius: f32,
}
impl ProjectionExpr {
    #[tracked]
    pub fn inv_project_fixed(&self, pos: Expr<Vec3<f32>>, cascde: Expr<u32>) -> Expr<Vec3<f32>> {
        let spacing = self.base_grid_size * (1_u32 << cascde).cast_f32();
        pos / spacing
    }
    #[tracked]
    pub fn inv_project(&self, pos: Expr<Vec3<f32>>) -> (Expr<Vec3<f32>>, Expr<u32>) {
        let approx_radius = (pos - self.view.pos).length();
        let approx_cascade = (keter::max((approx_radius / self.base_radius).log2(), 0.0) + 0.5)
            .ceil()
            .cast_u32();
        let approx_spacing = self.base_grid_size * (1_u32 << (approx_cascade + 2)).cast_f32();
        let discretized_pos = ((pos / approx_spacing).floor() + 0.5) * approx_spacing;
        let radius = (discretized_pos - self.view.pos).length();
        let cascade = keter::max((radius / self.base_radius).log2(), 0.0)
            .ceil()
            .cast_u32();
        let spacing = self.base_grid_size * (1_u32 << cascade).cast_f32();
        (pos / spacing, cascade)

        // let rotated = self.view.transform.transpose() * (pos - self.view.pos);
        // let pos = rotated / self.base_grid_size * Vec3::expr(2048.0, 2048.0, 1.0);
        // let xy = pos.xy() / pos.z * 32.0;
        // xy.extend(pos.z.log2() * 16.0)
    }
    #[tracked]
    pub fn project(&self, pos: Expr<Vec3<f32>>, cascade: Expr<u32>) -> Expr<Vec3<f32>> {
        let spacing = self.base_grid_size * (1_u32 << cascade).cast_f32();
        pos * spacing
    }
}

#[derive(Debug)]
pub struct HashCache {
    pub base_grid_size: f32,
    pub base_radius: f32,
    pub base_angular_resolution: u32,
    pub offset: Vec3<i32>,
    pub base_nodes: u32,
    pub num_levels: u32,
    pub nodes: Buffer<Node>,
    pub data: Buffer<RadianceElement>,
    pub projection: Buffer<Projection>,
}
impl HashCache {
    pub fn new(
        base_grid_size: f32,
        base_radius: f32,
        base_angular_resolution: u32,
        base_nodes: u32,
        num_levels: u32,
    ) -> Self {
        assert!(base_nodes.is_power_of_two());
        Self {
            base_grid_size,
            base_radius,
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
                (base_angular_resolution.pow(2)
                    * INDEX_SCALING.x
                    * INDEX_SCALING.y
                    * base_nodes
                    * num_levels) as usize,
                |_| RadianceElement::default(),
            ),
            projection: DEVICE.create_buffer(1),
        }
    }
    pub fn set_view(&self, view: View) -> NodeConfigs<'static> {
        self.projection.copy_from_vec(vec![Projection {
            view,
            base_grid_size: self.base_grid_size,
            base_radius: self.base_radius,
        }])
    }
    pub fn num_nodes_at_level(&self, level: u32) -> u32 {
        self.base_nodes >> (level * 2)
    }
    pub fn num_data(&self) -> u32 {
        self.base_nodes * self.base_angular_resolution.pow(2) * INDEX_SCALING.x * INDEX_SCALING.y
    }
    #[tracked]
    pub fn get_encoder(&self, level: Expr<u32>) -> Expr<DirectionIndexer> {
        DirectionIndexer::from_comps_expr(DirectionIndexerComps {
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
    pub fn read_grid(&self, node: Expr<LoadedNode>, dir: Expr<Vec2<u32>>) -> Expr<RadianceElement> {
        self.data
            .read(node.data_offset + node.encoder.grid_to_linear(dir))
    }
    #[tracked]
    pub fn read_node(&self, level: Expr<u32>, node_index: Expr<u32>) -> Expr<Node> {
        let node_offset = self.get_node_offset(level);
        self.nodes.read(node_offset + node_index)
    }

    #[tracked]
    pub fn write_node(&self, level: Expr<u32>, node_index: Expr<u32>, node: Expr<Node>) {
        let node_offset = self.get_node_offset(level);
        self.nodes.write(node_offset + node_index, node)
    }
    #[tracked]
    pub fn quantize_fixed(&self, pos: Expr<Vec3<f32>>, cascade: Expr<u32>) -> Expr<Probe> {
        let pos = self.projection.read(0).inv_project_fixed(pos, cascade);
        Probe::base(
            (pos.floor().cast_i32() - self.offset)
                .cast_u32()
                .extend(cascade),
        )
    }
    #[tracked]
    pub fn quantize(&self, pos: Expr<Vec3<f32>>) -> Expr<Probe> {
        let (pos, cascade) = self.projection.read(0).inv_project(pos);
        Probe::base(
            (pos.floor().cast_i32() - self.offset)
                .cast_u32()
                .extend(cascade),
        )
    }
    #[tracked]
    pub fn get_probe(
        &self,
        pos: Expr<Vec3<i32>>,
        cascade: Expr<u32>,
        level: Expr<u32>,
    ) -> Expr<Probe> {
        Probe::base(
            (pos * (1_u32 << level).cast_i32() - self.offset)
                .cast_u32()
                .extend(cascade),
        )
        .shift_level(level)
    }
    #[tracked]
    pub fn load_index(&self, probe: Expr<Probe>, index: Expr<u32>) -> Expr<LoadedNode> {
        let encoder = self.get_encoder(probe.level);
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
        let position = encode_position(probe.position);
        let node_index = (hash64(position) % table_size).var();
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
        let position = encode_position(probe.position);
        let node_index = (hash64(position) % table_size).var();
        // TODO: This should probably always be true if the hashmap works but just in case.
        let loaded = false.var();
        for _i in 0..10 {
            let node = self.nodes.atomic_ref(node_offset + node_index);
            let prev = node.position.compare_exchange(POSITION_INVALID, position);
            if prev == POSITION_INVALID || prev == position {
                // Success.
                *loaded = true;
                node.lifetime.exchange(NODE_LIFETIME);
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
    pub fn get_diffuse(&self, node: Expr<LoadedNode>, normal: Expr<Vec3<f32>>) -> Expr<Radiance> {
        let total_radiance = Vec4::splat(0.0_f32).var();
        for i in 0_u32.expr()..node.encoder.linear_size() {
            let dir = node.encoder.linear_to_dir(i);
            let weight = dir.dot(normal);
            if weight > 0.0 {
                let data_index = node.data_offset + i;
                let deposit = self.data.read(data_index);
                *total_radiance += deposit.radiance().extend(1.0) * weight;
            }
        }
        // TODO: Why the hell is this necessary??? Is a rust-analyzer issue.
        let out: Expr<Vec3<f32>> = total_radiance.xyz() / total_radiance.w;
        out
    }
    #[tracked]
    pub fn get_uniform(&self, node: Expr<LoadedNode>) -> Expr<Radiance> {
        let total_radiance = Vec3::splat(0.0_f32).var();
        for i in 0_u32.expr()..node.encoder.linear_size() {
            let data_index = node.data_offset + i;
            let deposit = self.data.read(data_index);
            *total_radiance += deposit.radiance();
        }
        // TODO: Why the hell is this necessary??? Is a rust-analyzer issue.
        let out: Expr<Vec3<f32>> = total_radiance / node.encoder.linear_size().cast_f32();
        out
    }
    #[tracked]
    pub fn get_dir(&self, node: Expr<LoadedNode>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.data
            .read(node.data_offset + node.encoder.dir_to_linear(dir))
            .radiance()
    }
    #[tracked]
    pub fn get_dir_bilinear(&self, node: Expr<LoadedNode>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        let dir = Encoder::encode(dir);
        let dir = ((dir + 1.0) / 2.0 * (node.encoder.resolution).cast_f32()) - 0.5;
        let base = dir.floor();
        let frac = dir - base;
        let base = base.cast_i32();
        let total_radiance = Vec3::splat(0.0_f32).var();
        for mask in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            let mask = Vec2::expr(mask[0] == 1, mask[1] == 1);
            let weight = mask.select(frac, 1.0 - frac).reduce_prod();
            let sample_index = node.encoder.grid_to_linear(
                (base + mask.cast_i32())
                    .rem_euclid(node.encoder.resolution.cast_i32())
                    .cast_u32(),
            );
            let deposit = self.data.read(node.data_offset + sample_index);
            *total_radiance += deposit.radiance() * weight;
        }
        **total_radiance
    }
    #[tracked]
    pub fn query_diffuse(&self, probe: Expr<Probe>, normal: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_diffuse(node, normal),
            || Vec3::splat_expr(-1.0),
        )
    }
    #[tracked]
    pub fn query_diffuse_bilinear(
        &self,
        pos: Expr<Vec3<f32>>,
        level: Expr<u32>,
        normal: Expr<Vec3<f32>>,
    ) -> Expr<Radiance> {
        let (pos, cascade) = self.projection.read(0).inv_project(pos);
        let spacing = (1_u32 << level).cast_f32();
        let base_probe = Probe::base(
            ((pos - spacing / 2.0).floor().cast_i32() - self.offset)
                .cast_u32()
                .extend(cascade),
        )
        .shift_level(level);

        let frac = (pos / spacing - 0.5).fract(); // or something like it
        let total_radiance = Vec4::splat(0.0_f32).var();
        for i in (0..8) {
            let mask = Vec3::new(
                if (i & 4) != 0 { 1 } else { 0 },
                if (i & 2) != 0 { 1 } else { 0 },
                if (i & 1) != 0 { 1 } else { 0 },
            );
            let weight = mask
                .map(|x| x == 1)
                .expr()
                .select(frac, 1.0 - frac)
                .reduce_prod();
            let probe = Probe::from_comps_expr(ProbeComps {
                position: (base_probe.position.xyz() + mask).extend(base_probe.position.w),
                level: base_probe.level,
            });
            *total_radiance += self.query(
                probe,
                |node| self.get_diffuse(node, normal).extend(1.0) * weight,
                || Vec4::splat_expr(0.0),
            );
        }
        total_radiance.xyz() / total_radiance.w
    }
    #[tracked]
    pub fn query_uniform(&self, probe: Expr<Probe>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_uniform(node),
            || Vec3::splat_expr(-1.0),
        )
    }
    #[tracked]
    pub fn query_dir(&self, probe: Expr<Probe>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_dir(node, dir),
            || Vec3::splat_expr(-1.0),
        )
    }
    #[tracked]
    pub fn query_dir_bilinear(&self, probe: Expr<Probe>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_dir_bilinear(node, dir),
            || Vec3::splat_expr(-1.0),
        )
    }
    // TODO: This doesn't really seem to balance out.
    #[tracked]
    pub fn try_deposit(
        &self,
        probe: Expr<Probe>,
        f: impl Fn() -> (Expr<Vec3<f32>>, Expr<RadianceDeposit>),
    ) -> Expr<bool> {
        self.query_init(
            probe,
            |node| {
                let (dir, deposit) = f();
                let data = self
                    .data
                    .atomic_ref(node.data_offset + node.encoder.dir_to_linear(dir));
                data.deposit(deposit);
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
        let position = encode_position(probe.position);
        let node_index = (hash64(position) % table_size).var();
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
        dir: Expr<Vec3<f32>>,
        deposit: Expr<RadianceDeposit>,
    ) -> Expr<bool> {
        self.try_deposit(probe, || (dir, deposit))
    }
    /*
    #[tracked]
    pub fn deposit_bilinear(
        &self,
        pos: Expr<Vec3<f32>>,
        level: Expr<u32>,
        dir: Expr<Vec3<f32>>,
        deposit: Expr<RadianceDeposit>,
    ) {
        let spacing = self.base_grid_size * (1_u32 << level).cast_f32();
        let base_probe = self.quantize(pos - spacing / 2.0).shift_level(level);
        let frac = (pos / spacing - 0.5).rem_euclid(1.0); // or something like it
        for i in (0..8) {
            let mask = Vec3::new(
                if (i & 4) != 0 { 1 } else { 0 },
                if (i & 2) != 0 { 1 } else { 0 },
                if (i & 1) != 0 { 1 } else { 0 },
            );
            let weight = mask
                .map(|x| x == 1)
                .expr()
                .select(frac, 1.0 - frac)
                .reduce_prod();
            let probe = Probe::from_comps_expr(ProbeComps {
                position: base_probe.position + mask,
                level: base_probe.level,
            });
            self.deposit(probe, dir, deposit.weighted(weight));
        }
    }
    */
}
