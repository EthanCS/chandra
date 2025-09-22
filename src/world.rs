use std::mem::forget;

use dot_vox::{DotVoxData, SceneNode};
use glam::{IVec3, Mat3 as FMat3, UVec3, Vec3 as FVec3};
use keter::lang::types::vector::{Mat3, Mat4, Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter::rtx::{
    Aabb, Accel, AccelBuildRequest, AccelOption, AccelTraceOptions, AccelUsageHint, CommittedHit,
    Ray,
};

use crate::utils::*;

pub struct World {
    // bindless: BindlessArray,
    accel: Accel,
    rendered_models: Buffer<RenderedModel>,
    model_brick_offsets: Buffer<u32>,
    brick_masks: Buffer<BrickMask>,
    brick_positions: Buffer<Vec3<u32>>,
}

impl World {
    pub fn new(data: DotVoxData) -> Self {
        let bindless = DEVICE.create_bindless_array(data.models.len());

        let mut brick_masks = Vec::<BrickMask>::new();
        let mut brick_colors = Vec::<u8>::new();
        let mut brick_positions = Vec::<Vec3<u32>>::new();
        let mut model_bvhs = Vec::new();
        let mut model_sizes = Vec::<Vec3<u32>>::new();
        let mut model_brick_offsets = Vec::<u32>::new();
        for (index, model) in data.models.iter().enumerate() {
            let s = model.size;
            let s = UVec3::new(s.x, s.y, s.z);
            let bricks = s.map(|x| x.div_ceil(BRICK_SIZE));
            let s = bricks * BRICK_SIZE;
            model_brick_offsets.push(brick_masks.len() as u32);
            let mut aabbs = Vec::<Aabb>::new();
            // let mut brick_map = vec![u32::MAX; bricks.element_product() as usize];
            let mut voxels = vec![255; s.element_product() as usize];
            for voxel in &model.voxels {
                let pos = UVec3::new(voxel.x as u32, voxel.y as u32, voxel.z as u32);
                voxels[to_linear(pos, s)] = voxel.i;
            }
            if model.voxels.is_empty() {
                voxels[0] = 0;
            }
            for brick in iter_vec(bricks) {
                let mut mask = BrickMask::empty();
                let mut colors = [255; (BRICK_SIZE * BRICK_SIZE * BRICK_SIZE) as usize];
                let mut min = UVec3::splat(BRICK_SIZE);
                let mut max = UVec3::splat(0);
                for brick_pos in iter_vec(UVec3::splat(BRICK_SIZE)) {
                    let pos = brick * BRICK_SIZE + brick_pos;
                    let voxel = voxels[to_linear(pos, s)];
                    if voxel != 255 {
                        mask.set(brick_pos, true);
                        min = min.min(brick_pos);
                        max = max.max(brick_pos + 1);
                    }
                    colors[to_linear(brick_pos, UVec3::splat(BRICK_SIZE))] = voxel;
                }
                if mask != BrickMask::empty() {
                    // brick_map[to_linear(brick, bricks)] = brick_masks.len() as u32;
                    brick_masks.push(mask);
                    brick_colors.extend_from_slice(&colors);
                    brick_positions.push((brick * BRICK_SIZE).into());
                    aabbs.push(Aabb {
                        min: (brick * BRICK_SIZE + min).as_vec3().into(),
                        max: (brick * BRICK_SIZE + max).as_vec3().into(),
                    });
                }
            }
            // let brick_map = DEVICE.create_buffer_from_slice(&brick_map);
            // bindless.emplace_buffer(index, &brick_map);
            // forget(brick_map);
            if aabbs.is_empty() {
                println!("Voxels: {:?}", model.voxels);
                println!("Bricks: {:?}", bricks);
                println!("Size: {:?}", s);
                panic!("Shit");
            }
            let aabbs = DEVICE.create_buffer_from_slice(&aabbs);
            let bvh = DEVICE.create_procedural_primitive(
                aabbs.view(..),
                AccelOption {
                    hint: AccelUsageHint::FastTrace,
                    allow_compaction: true,
                    allow_update: false,
                },
            );
            bvh.build(AccelBuildRequest::ForceBuild);
            model_bvhs.push(bvh);
            forget(aabbs);
            model_sizes.push(bricks.into());
            if index % 20 == 0 {
                println!("Loading: {:?}", index);
            }
        }
        println!("Loaded");
        let model_brick_offsets = DEVICE.create_buffer_from_slice(&model_brick_offsets);
        let brick_masks = DEVICE.create_buffer_from_slice(&brick_masks);
        let brick_colors = DEVICE.create_buffer_from_slice(&brick_colors);
        let brick_positions = DEVICE.create_buffer_from_slice(&brick_positions);
        // let model_sizes = DEVICE.create_buffer_from_slice(&model_sizes);

        let mut rendered_models = Vec::new();
        visit(
            &data,
            &mut rendered_models,
            glam::IVec3::ZERO,
            dot_vox::Rotation::IDENTITY,
            0,
        );

        let accel = DEVICE.create_accel(AccelOption {
            hint: AccelUsageHint::FastTrace,
            allow_compaction: true,
            allow_update: false,
        });

        for model in &rendered_models {
            fn extend(a: Vec3<f32>, b: f32) -> Vec4<f32> {
                Vec4::new(a.x, a.y, a.z, b)
            }
            accel.push_procedural_primitive(
                &model_bvhs[model.index as usize],
                Mat4 {
                    cols: [
                        extend(model.rotation.cols[0], 0.0),
                        extend(model.rotation.cols[1], 0.0),
                        extend(model.rotation.cols[2], 0.0),
                        extend(model.position, 1.0),
                    ],
                },
                0xFF,
            );
        }

        let rendered_models = DEVICE.create_buffer_from_slice(&rendered_models);
        accel.build(AccelBuildRequest::ForceBuild);

        Self {
            // bindless,
            accel,
            rendered_models,
            model_brick_offsets,
            brick_masks,
            brick_positions,
        }
    }

    #[tracked]
    pub fn trace(
        &self,
        ray_start: Expr<Vec3<f32>>,
        ray_dir: Expr<Vec3<f32>>,
    ) -> Expr<CommittedHit> {
        self.accel
            .traverse(
                Ray::new_expr(
                    Expr::<[f32; 3]>::from(ray_start),
                    0.0,
                    Expr::<[f32; 3]>::from(ray_dir),
                    10000.0,
                ),
                AccelTraceOptions {
                    mask: 0xFF.expr(),
                    ..Default::default()
                },
            )
            .on_procedural_hit(|candidate| {
                let model = self.rendered_models.read(candidate.inst);
                let brick = self.model_brick_offsets.read(model.index) + candidate.prim;
                let mask = self.brick_masks.read(brick);

                let ray_start = model.rotation.transpose() * (ray_start - model.position)
                    - self.brick_positions.read(brick).cast_f32();
                let ray_dir = model.rotation.transpose() * ray_dir;

                dda(
                    ray_start,
                    ray_dir,
                    10000.0.expr(),
                    (
                        Vec3::splat_expr(0.0001),
                        Vec3::splat_expr(BRICK_SIZE as f32 - 0.0001),
                    ),
                    |pos, last_t, next_t| {
                        if mask.read(pos) {
                            candidate.commit((last_t + next_t) / 2.0);
                            true.expr()
                        } else {
                            false.expr()
                        }
                    },
                );
            })
            .trace()
    }
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
struct RenderedModel {
    position: Vec3<f32>,
    rotation: Mat3,
    index: u32,
}

fn to_transform(position: IVec3, rotation: dot_vox::Rotation, size: UVec3) -> (Vec3<f32>, Mat3) {
    let position = position.as_vec3();

    let rotation = FMat3::from_cols_array_2d(&rotation.to_cols_array_2d());

    let mut offset = FVec3::select(
        (size % 2).cmpeq(UVec3::ZERO),
        FVec3::ZERO,
        FVec3::splat(0.5),
    );
    offset = rotation.mul_vec3(offset); // If another seam shows up in the future, try multiplying this with `scale`
    let center = rotation * (size.as_vec3() / 2.0);
    ((position - center + offset).into(), rotation.into())
}

fn visit(
    file: &DotVoxData,
    rendered: &mut Vec<RenderedModel>,
    position: glam::IVec3,
    rotation: dot_vox::Rotation,
    index: u32,
) {
    if file.scenes.is_empty() && index == 0 {
        rendered.push(RenderedModel {
            index: 0,
            position: Vec3::splat(0.0),
            rotation: Mat3::identity(),
        });
        return;
    }
    match &file.scenes[index as usize] {
        SceneNode::Group { children, .. } => {
            for child in children {
                visit(file, rendered, position, rotation, *child);
            }
        }
        SceneNode::Shape { models, .. } => {
            for model in models {
                let size = file.models[model.model_id as usize].size;
                let (position, rotation) =
                    to_transform(position, rotation, glam::UVec3::new(size.x, size.y, size.z));
                rendered.push(RenderedModel {
                    index: model.model_id,
                    position,
                    rotation,
                });
            }
        }
        SceneNode::Transform { frames, child, .. } => {
            let frame = frames.first().unwrap();
            let position = frame
                .position()
                .map(|pos| glam::IVec3::new(pos.x, pos.y, pos.z))
                .unwrap_or(glam::IVec3::ZERO)
                + position;

            let rotation = frame.orientation().unwrap_or(dot_vox::Rotation::IDENTITY); // * rotation;

            visit(file, rendered, position, rotation, *child);
        }
    }
}

const BRICK_SIZE: u32 = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Value)]
struct BrickMask {
    values: u64,
}
impl BrickMask {
    fn empty() -> Self {
        Self { values: 0 }
    }
    fn set(&mut self, pos: UVec3, value: bool) {
        self.values |=
            (value as u64) << (pos.x + pos.y * BRICK_SIZE + pos.z * BRICK_SIZE * BRICK_SIZE);
    }
}
impl BrickMaskExpr {
    #[tracked]
    fn read(&self, pos: Expr<Vec3<u32>>) -> Expr<bool> {
        (self.values >> (pos.x + pos.y * BRICK_SIZE + pos.z * BRICK_SIZE * BRICK_SIZE).cast_u64())
            & 1
            != 0
    }
}
