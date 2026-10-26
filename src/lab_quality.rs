//! Diagnostics follow fixed barycentric points on the current deformed mesh,
//! rather than subtracting unrelated screen pixels while geometry moves.
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;

use crate::algorithm::CoreData;

pub struct SurfaceSamples {
    values: Buffer<Vec4<f32>>,
    sample: Kernel<fn()>,
}

pub struct SurfaceBuffers {
    pub position: Buffer<Vec4<f32>>,
    pub normal: Buffer<Vec4<f32>>,
    capture: Kernel<fn()>,
}

impl SurfaceBuffers {
    pub fn new(core: &CoreData) -> Self {
        let position=DEVICE.create_buffer::<Vec4<f32>>(core.num_pixels() as usize);
        let normal=DEVICE.create_buffer::<Vec4<f32>>(core.num_pixels() as usize);
        let capture=DEVICE.create_kernel::<fn()>(&track!(|| {
            let pixel=dispatch_id().xy();
            let index=core.pixel_to_linear(pixel);
            if core.read_depth(pixel)==0.0 {
                position.write(index,Vec4::splat(0.0f32));
                normal.write(index,Vec4::splat(0.0f32));
            } else {
                let hit=core.hit.read(index);
                position.write(index,core.get_world_pos(pixel).extend((hit.inst+1u32).cast_f32()));
                normal.write(index,core.read_normal(pixel).extend(1.0));
            }
        }));
        Self {position,normal,capture}
    }
    pub fn capture(&self,size:[u32;3]) { self.capture.dispatch(size); }
}

impl SurfaceSamples {
    /// The procedural courtyard's moving box has twelve triangles.
    pub fn courtyard(core: &CoreData, linear_output: &Buffer<Vec4<f32>>) -> Self {
        let values = DEVICE.create_buffer::<Vec4<f32>>(12 * 64);
        let sample = DEVICE.create_kernel::<fn()>(&track!(|| {
            let i = dispatch_id().x;
            values.write(i, Vec4::splat(0.0f32));
            let offsets = core.world.offsets.read(4u32);
            let triangle = core.world.indices.read(offsets[0] + i / 64u32);
            let a = Expr::<Vec3<f32>>::from(core.world.positions.read(offsets[1] + triangle[0]));
            let b = Expr::<Vec3<f32>>::from(core.world.positions.read(offsets[1] + triangle[1]));
            let c = Expr::<Vec3<f32>>::from(core.world.positions.read(offsets[1] + triangle[2]));
            let u = ((i % 8u32).cast_f32() + 0.5) / 8.0;
            let v = (((i / 8u32) % 8u32).cast_f32() + 0.5) / 8.0;
            let uv = (u + v > 1.0).select(Vec2::expr(1.0 - u, 1.0 - v), Vec2::expr(u, v));
            let position = a + (b - a) * uv.x + (c - a) * uv.y;
            let normal = (b - a).cross(c - a).normalize();
            let view = core.view.read();
            if normal.dot(position - view.pos) >= 0.0 {
                return;
            }
            let pixel = view.project(position) - 0.5;
            if !(pixel >= 0.0).all() || !(pixel < core.screen_size().expr().cast_f32() - 1.0).all() {
                return;
            }
            let base = pixel.floor().cast_u32();
            let frac = pixel.fract();
            let color = Vec3::splat(0.0f32).var();
            for corner in 0u32..4 {
                let q = base + Vec2::expr(corner & 1u32, corner >> 1u32);
                if core.read_depth(q) == 0.0 {
                    return;
                }
                let hit = core.hit.read(core.pixel_to_linear(q));
                if hit.inst != 4u32 || (core.get_world_pos(q) - position).dot(normal).abs() > 0.05 {
                    return;
                }
                let weight = ((corner & 1u32) != 0u32).select(frac.x, 1.0 - frac.x)
                    * ((corner >> 1u32) != 0u32).select(frac.y, 1.0 - frac.y);
                *color += linear_output.read(core.pixel_to_linear(q)).xyz() * weight;
            }
            values.write(i, color.extend(1.0));
        }));
        Self { values, sample }
    }

    pub fn capture(&self) -> Vec<[f32; 4]> {
        self.sample.dispatch([self.values.len() as u32, 1, 1]);
        self.values
            .copy_to_vec()
            .into_iter()
            .map(|v| [v.x, v.y, v.z, v.w])
            .collect()
    }
}
