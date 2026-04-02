use std::ops::AddAssign;

use keter::bindless::{BINDLESS, Tex2dHandle};
use keter::graph::NodeConfigs;
use keter::lang::types::shared::Shared;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::camera::View;
use yesod::device_println;
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::utils::{dispatch0_async, dispatch1_async, memo, offset_ray};

use crate::cache::{Fluence, Probe, Radiance};

#[cfg(not(feature = "voxel"))]
type World = crate::world::TriangleWorld;
#[cfg(not(feature = "voxel"))]
type Hit = keter::rtx::SurfaceHit;
#[cfg(feature = "voxel")]
type World = crate::world::VoxelWorld;
#[cfg(feature = "voxel")]
type Hit = keter::rtx::CommittedHit;

pub struct CoreData {
    pub world: World,
    pub view: Singleton<View>,
    pub depth: Tex2dHandle<f32>,
    pub normal: Tex2d<Vec2<f32>>,
    pub hit: Buffer<Hit>,

    pub sunlight_dir: Vec3<f32>,
}
impl CoreData {
    pub fn new(world: World, [width, height]: [u32; 2], sunlight_dir: Vec3<f32>) -> Self {
        let depth = DEVICE.create_tex2d::<f32>(PixelStorage::Float1, width, height, 1);
        let depth = BINDLESS.push_tex2d(
            depth,
            Sampler {
                filter: SamplerFilter::LinearLinear,
                address: SamplerAddress::Edge,
            },
        );
        BINDLESS.update();
        // is the compression worth it?
        Self {
            world,
            view: Singleton::new_uninit(),
            depth,
            normal: DEVICE.create_tex2d::<Vec2<f32>>(PixelStorage::Short2, width, height, 1),
            hit: DEVICE.create_buffer::<Hit>(width as usize * height as usize),
            sunlight_dir,
        }
    }
    pub fn screen_size(&self) -> Vec2<u32> {
        Vec2::new(self.normal.width(), self.normal.height())
    }
    pub fn screen_dispatch_size(&self) -> [u32; 3] {
        [self.normal.width(), self.normal.height(), 1]
    }
    pub fn num_pixels(&self) -> u32 {
        self.normal.width() * self.normal.height()
    }
    #[tracked]
    pub fn get_emission(&self, pos: Expr<Vec3<f32>>, hit: Expr<Hit>) -> Expr<Radiance> {
        let normal = self.world.get_normal_flat(hit);
        let weight = normal.dot(self.sunlight_dir);
        let radiance = Vec3::splat(0.0_f32).var();

        if weight > 0.0 {
            *radiance += self
                .world
                .trace(offset_ray(pos, normal), self.sunlight_dir.expr())
                .miss()
                .select(Vec3::splat_expr(10.0_f32), Vec3::splat_expr(0.0_f32))
                * weight;
        }
        // if hit.inst == 1 {
        //     *radiance += Vec3::expr(10.0, 0.0, 0.0);
        // }
        **radiance
    }
    #[tracked]
    pub fn get_fluence(
        &self,
        start: Expr<Vec3<f32>>,
        dir: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
    ) -> Expr<Fluence> {
        if !hit.miss() {
            self.get_emission(start + dir * hit.committed_ray_t, hit)
                .extend(0.0)
        } else {
            Vec3::splat_expr(0.0_f32).extend(1.0)
        }
    }
    #[tracked]
    pub fn get_world_pos(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        let view = self.view.read();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();
        let depth = self.depth.read(pixel);
        if depth == 0.0 {
            return;
        }
        view.pos + ray_dir * depth
    }
    #[tracked]
    pub fn read_normal(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        SphericalEncoder::decode(self.normal.read(pixel))
    }
    #[tracked]
    pub fn read_depth(&self, pixel: Expr<Vec2<u32>>) -> Expr<f32> {
        // could use hit.committed_ray_t instead
        self.depth.read(pixel)
    }
    #[tracked]
    pub fn pixel_to_linear(&self, pixel: Expr<Vec2<u32>>) -> Expr<u32> {
        pixel.y * self.screen_size().x + pixel.x
    }
    pub fn compute_gbuffer(&self) -> NodeConfigs<'static> {
        dispatch0_async(
            track!(|| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();
                let view = self.view.read();
                let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

                let hit = self.world.trace(view.pos, ray_dir);
                if hit.miss() {
                    self.depth.write(pixel, 0.0);
                    return;
                }
                self.depth.write(pixel, hit.committed_ray_t);
                let normal = self.world.get_normal_flat(hit);
                self.normal.write(pixel, SphericalEncoder::encode(normal));
                self.hit.write(self.pixel_to_linear(pixel), hit);
            }),
            self.screen_dispatch_size(),
        )
        .debug("Compute G-Buffer")
    }
}

pub mod rc;
pub mod restir;
pub mod splat;
pub mod trace;

#[derive(Default)]
pub struct CommandList<'a> {
    unfinished: Vec<NodeConfigs<'a>>,
    pub timings: Vec<(String, f32)>,
}
impl<'a> AddAssign<NodeConfigs<'a>> for CommandList<'a> {
    fn add_assign(&mut self, rhs: NodeConfigs<'a>) {
        self.unfinished.push(rhs);
    }
}
impl<'a> CommandList<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn execute(&mut self) {
        self.timings
            .extend(std::mem::take(&mut self.unfinished).chain().execute_timed());
    }
    pub fn push(&mut self, node: impl AsNodes<'a>) {
        self.unfinished.push(node.into_node_configs());
    }
}
