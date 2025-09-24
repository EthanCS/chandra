use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::utils::pcg3d;
use static_assertions::assert_eq_size;

// type Fluence = fluence::Fluence<fluence::RgbF16>;
type Radiance = Vec3<f32>;

const NUM_BITS: u64 = 21;
const MASK: u32 = (1 << NUM_BITS) - 1;
const POSITION_INVALID: u64 = u64::MAX;

#[tracked]
fn encode_position(cell: Expr<Vec3<i32>>) -> Expr<u64> {
    let cell = cell.cast_u32() & MASK;
    (cell.x.cast_u64() << (NUM_BITS * 2)) | (cell.y.cast_u64() << NUM_BITS) | cell.z.cast_u64()
}

// TODO: SoA.
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Node {
    pub radiance_count: Vec4<f32>,
    pub position: u64,
}
impl NodeVar {
    #[tracked]
    pub fn reduce(&self, factor: Expr<f32>, cutoff: Expr<f32>) {
        if self.radiance_count.w > cutoff {
            *self.radiance_count *= factor;
        } else {
            *self.radiance_count = Vec4::splat(0.0);
            *self.position = POSITION_INVALID;
        }
    }
}
assert_eq_size!(Node, [f32; 8]);

#[derive(Debug)]
pub struct HashCache {
    pub grid_size: f32,
    pub nodes: Buffer<Node>,
}
impl HashCache {
    pub fn new(grid_size: f32, cells: u32) -> Self {
        Self {
            grid_size,
            nodes: DEVICE.create_buffer(cells as usize),
        }
    }
    #[tracked]
    pub fn get_cell(&self, pos: Expr<Vec3<f32>>) -> Expr<Vec3<i32>> {
        (pos / self.grid_size).floor().cast_i32()
    }
    #[tracked]
    pub fn get_index(&self, cell_pos: Expr<Vec3<i32>>) -> Expr<u32> {
        pcg3d(cell_pos.cast_u32()).x % self.nodes.len() as u32
    }
    #[tracked]
    pub fn query(&self, pos: Expr<Vec3<f32>>) -> Expr<Radiance> {
        let cell = self.get_cell(pos);
        let node = self.nodes.read(self.get_index(cell));
        if node.position != encode_position(cell) || node.radiance_count.w == 0.0 {
            Vec3::splat_expr(-1.0)
        } else {
            let rc = node.radiance_count;
            rc.xyz() / rc.w
        }
    }
    #[tracked]
    pub fn try_deposit(&self, pos: Expr<Vec3<f32>>, f: impl Fn() -> Expr<Radiance>) {
        let cell = self.get_cell(pos);
        let index = self.get_index(cell);
        let node = self.nodes.atomic_ref(index);
        let pos_enc = encode_position(cell);
        let prev = node.position.compare_exchange(POSITION_INVALID, pos_enc);
        if prev == POSITION_INVALID || prev == pos_enc {
            let v = f();
            let rc = node.radiance_count;
            rc.x.fetch_add(v.x);
            rc.y.fetch_add(v.y);
            rc.z.fetch_add(v.z);
            rc.w.fetch_add(1.0);
        }
    }
    #[tracked]
    pub fn deposit(&self, pos: Expr<Vec3<f32>>, radiance: Expr<Radiance>) {
        self.try_deposit(pos, || radiance)
    }
}
