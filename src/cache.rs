use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::utils::pcg3d;
use static_assertions::assert_eq_size;

// type Fluence = fluence::Fluence<fluence::RgbF16>;
type Radiance = Vec3<f32>;

// TODO: SoA.
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Node {
    pub radiance_count: Vec4<f32>,
}
impl NodeVar {
    #[tracked]
    pub fn reduce(&self, factor: Expr<f32>, cutoff: Expr<f32>) {
        if self.radiance_count.w > cutoff {
            *self.radiance_count *= factor;
        }
    }
}
assert_eq_size!(Node, [f32; 4]);

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
    pub fn get_index(&self, pos: Expr<Vec3<f32>>) -> Expr<u32> {
        let cell_pos = (pos / self.grid_size).floor().cast_i32();
        pcg3d(cell_pos.cast_u32()).x % self.nodes.len() as u32
    }
    #[tracked]
    pub fn query(&self, pos: Expr<Vec3<f32>>) -> Expr<Radiance> {
        let rc = self.nodes.read(self.get_index(pos)).radiance_count;
        rc.xyz() / keter::max(rc.w, 0.01)
    }
    #[tracked]
    pub fn deposit(&self, pos: Expr<Vec3<f32>>, value: Expr<Radiance>) {
        let rc = self.nodes.atomic_ref(self.get_index(pos)).radiance_count;
        rc.x.fetch_add(value.r);
        rc.y.fetch_add(value.g);
        rc.z.fetch_add(value.b);
        rc.w.fetch_add(1.0);
    }
}
