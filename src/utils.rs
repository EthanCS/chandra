use std::f32::consts::TAU;

use glam::UVec3;
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;

#[tracked]
pub fn index_to_corner(i: Expr<u32>) -> Expr<Vec3<u32>> {
    Vec3::expr(i & 1, (i >> 1) & 1, i >> 2)
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
#[tracked]
pub fn from_linear_expr(
    mut index: Expr<u32>,
    size: impl AsExpr<Value = Vec3<u32>>,
) -> Expr<Vec3<u32>> {
    let size = size.as_expr();
    let x = index % size.x;
    index = index / size.x;
    let y = index % size.y;
    index = index / size.y;
    let z = index;
    Vec3::expr(x, y, z)
}

pub fn iter_vec(size: UVec3) -> impl Iterator<Item = UVec3> {
    (0..size.x)
        .flat_map(move |x| (0..size.y).map(move |y| (x, y)))
        .flat_map(move |(x, y)| (0..size.z).map(move |z| UVec3::new(x, y, z)))
}

#[tracked]
pub fn rotate_hemi(normal: Expr<Vec3<f32>>, v: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    // Find perpendicular basis; due to hairy ball theorem this cannot be continuous.
    let perp1 = if normal.z.abs() < 0.9 {
        normal.cross(Vec3::expr(0.0_f32, 0.0, 1.0)).normalize()
    } else {
        normal.cross(Vec3::expr(0.0_f32, 1.0, 0.0)).normalize()
    };
    let perp2 = normal.cross(perp1);
    v.x * perp1 + v.y * perp2 + v.z * normal
}

// https://ameye.dev/notes/sampling-the-hemisphere/
#[tracked]
pub fn sample_bounce(normal: Expr<Vec3<f32>>, rng: Expr<Vec2<f32>>) -> Expr<Vec3<f32>> {
    let r = rng.x.sqrt();
    let phi = rng.y * TAU;
    let v = Vec3::expr(
        r * phi.cos(),
        r * phi.sin(),
        keter::max(0.0, 1.0 - r.sqr()).sqrt(),
    );
    rotate_hemi(normal, v)
}

#[tracked]
pub fn reflect(dir: Expr<Vec3<f32>>, normal: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    dir - normal * (2.0_f32 * dir.dot(normal))
}
