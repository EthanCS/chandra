use glam::UVec3;
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;

#[tracked]
pub fn index_to_corner(i: Expr<u32>) -> Expr<Vec3<u32>> {
    Vec3::expr(i & 1, (i >> 1) & 1, (i >> 2) & 1)
}

#[tracked]
pub fn corner_to_index(corner: Expr<Vec3<u32>>) -> Expr<u32> {
    corner.x + (corner.y << 1) + (corner.z << 2)
}

pub fn to_linear(pos: UVec3, size: UVec3) -> usize {
    (pos.x + pos.y * size.x + pos.z * size.x * size.y) as usize
}

#[tracked]
pub fn to_linear_expr(pos: Expr<Vec3<u32>>, size: impl AsExpr<Value = Vec3<u32>>) -> Expr<u32> {
    let size = size.as_expr();
    pos.x + pos.y * size.x + pos.z * size.x * size.y
}

pub fn iter_vec(size: UVec3) -> impl Iterator<Item = UVec3> {
    (0..size.x)
        .flat_map(move |x| (0..size.y).map(move |y| (x, y)))
        .flat_map(move |(x, y)| (0..size.z).map(move |z| UVec3::new(x, y, z)))
}
