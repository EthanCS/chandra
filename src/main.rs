use std::mem::forget;

use dot_vox::{DotVoxData, SceneNode};
use glam::{IVec3, Mat3 as FMat3, UVec3, Vec3 as FVec3};
use keter::lang::types::vector::{Mat3, Mat4, Vec2, Vec3};
use keter::prelude::*;
use keter::rtx::{Aabb, AccelBuildRequest, AccelOption, AccelTraceOptions, AccelUsageHint, Ray};
use keter_testbed::{App, KeyCode};

#[derive(Debug, Clone, Copy)]
struct Camera {
    screen_size: Vec2<f32>,
    yaw: f32,
    pitch: f32,
    pos: Vec3<f32>,
    fov: f32,
}
impl Camera {
    fn rotation(&self) -> FMat3 {
        let view = FMat3::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0).transpose();
        FMat3::from_cols(view.x_axis, view.z_axis, view.y_axis).transpose()
    }
    fn view(&self) -> View {
        let view = self.rotation();
        let ratio = (self.fov / 2.0).tan() / (self.screen_size.y / 2.0);
        let view = FMat3::from_cols(view.x_axis * ratio, -view.y_axis * ratio, view.z_axis);

        View {
            screen_size: self.screen_size,
            pos: self.pos,
            transform: view.into(),
        }
    }
    fn forward(&self) -> Vec3<f32> {
        self.rotation().z_axis.into()
    }
    fn right(&self) -> Vec3<f32> {
        self.rotation().x_axis.into()
    }
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
struct View {
    screen_size: Vec2<f32>,
    pos: Vec3<f32>,
    transform: Mat3,
}

#[tracked]
fn aabb_intersect(
    start: Expr<Vec3<f32>>,
    inv_dir: Expr<Vec3<f32>>,
    aabb_min: Expr<Vec3<f32>>,
    aabb_max: Expr<Vec3<f32>>,
) -> Expr<Vec2<f32>> {
    let t0 = (aabb_min - start) * inv_dir;
    let t1 = (aabb_max - start) * inv_dir;
    let tmin = keter::min(t0, t1).reduce_max();
    let tmax = keter::max(t0, t1).reduce_min();
    Vec2::expr(tmin, tmax)
}

#[tracked]
fn dda(
    start: Expr<Vec3<f32>>,
    ray_dir: Expr<Vec3<f32>>,
    length: Expr<f32>,
    bounds: (Expr<Vec3<f32>>, Expr<Vec3<f32>>),
    f: impl Fn(Expr<Vec3<u32>>, Expr<f32>, Expr<f32>) -> Expr<bool>,
) {
    let inv_dir = (ray_dir + f32::EPSILON).recip();
    let interval = aabb_intersect(start, inv_dir, bounds.0, bounds.1);
    let start_t = keter::max(interval.x, 0.0);
    let ray_start = start + start_t * ray_dir;
    let end_t = keter::min(interval.y, length) - start_t;
    if end_t >= 0.01 {
        let pos = ray_start.floor().cast_i32().cast_u32().var();

        let delta_dist = inv_dir.abs();

        let ray_step = ray_dir.signum().cast_i32().cast_u32();
        let side_dist = (ray_dir.signum() * (pos.cast_i32().cast_f32() - ray_start)
            + ray_dir.signum() * 0.5
            + 0.5)
            * delta_dist;
        let side_dist = side_dist.var();

        let last_t = start_t.var();

        loop {
            let next_t = side_dist.reduce_min();

            let next_adj_t = next_t + start_t;

            if f(**pos, **last_t, next_adj_t) {
                break;
            }
            *last_t = next_adj_t;

            if next_t >= end_t {
                break;
            }
            let mask = side_dist <= keter::min(side_dist.yzx(), side_dist.zxy());

            *side_dist += mask.select(delta_dist, Vec3::splat_expr(0.0));
            *pos += mask.select(ray_step, Vec3::splat_expr(0));
        }
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

fn main() {
    let app = App::new("Sunfall", [2048; 2]).agx().resize().init();

    let file = dot_vox::load("assets/castle.vox").unwrap();

    let palette = DEVICE.create_buffer_from_fn(file.palette.len(), |i| {
        let color = file.palette[i];
        Vec3::new(color.r as f32, color.g as f32, color.b as f32) * (1.0 / 255.0_f32)
    });

    let bindless = DEVICE.create_bindless_array(file.models.len());

    for (i, model) in file.models.iter().enumerate() {
        let s = model.size;
        let mut voxels = vec![255; (s.x * s.y * s.z) as usize];
        for voxel in &model.voxels {
            voxels[(voxel.x as u32 + voxel.y as u32 * s.x + voxel.z as u32 * s.x * s.y) as usize] =
                voxel.i;
        }
        let buffer = DEVICE.create_buffer_from_slice(&voxels);
        bindless.emplace_buffer(i, &buffer);
        forget(buffer);
    }
    let model_sizes = DEVICE.create_buffer_from_fn(file.models.len(), |i| {
        let s = file.models[i].size;
        Vec3::new(s.x, s.y, s.z)
    });

    let mut rendered_models = Vec::new();
    visit(
        &file,
        &mut rendered_models,
        glam::IVec3::ZERO,
        dot_vox::Rotation::IDENTITY,
        0,
    );
    let rendered_aabbs = rendered_models
        .iter()
        .map(|model| {
            let s = file.models[model.index as usize].size;
            let aabb_x = FVec3::new(model.position.x, model.position.y, model.position.z);
            let aabb_y = aabb_x
                + FMat3::from(model.rotation) * FVec3::new(s.x as f32, s.y as f32, s.z as f32);
            Aabb {
                min: aabb_x.min(aabb_y).into(),
                max: aabb_x.max(aabb_y).into(),
            }
        })
        .collect::<Vec<_>>();
    let rendered_aabbs = DEVICE.create_buffer_from_slice(&rendered_aabbs);

    let rendered_models = DEVICE.create_buffer_from_slice(&rendered_models);

    let model_accel = DEVICE.create_procedural_primitive(
        rendered_aabbs.view(..),
        AccelOption {
            hint: AccelUsageHint::FastTrace,
            allow_compaction: true,
            allow_update: false,
        },
    );
    model_accel.build(AccelBuildRequest::ForceBuild);

    let accel = DEVICE.create_accel(AccelOption {
        hint: AccelUsageHint::FastTrace,
        allow_compaction: true,
        allow_update: false,
    });
    accel.push_procedural_primitive(&model_accel, Mat4::identity(), 0xFF);
    accel.build(AccelBuildRequest::ForceBuild);

    let draw_kernel = DEVICE.create_kernel::<fn(View)>(&track!(|view| {
        let pixel = dispatch_id().xy();
        let ray_dir = view.transform * (pixel.cast_f32() - view.screen_size / 2.0).extend(1.0);
        let ray_dir = ray_dir.normalize();

        let hit = accel
            .traverse(
                Ray::new_expr(
                    Expr::<[f32; 3]>::from(view.pos),
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
                let index = candidate.prim;
                let model = rendered_models.read(index);

                let ray_start = model.rotation.transpose() * (view.pos - model.position);
                let ray_dir = model.rotation.transpose() * ray_dir;
                let size = model_sizes.read(model.index);
                let voxels = bindless.buffer::<u8>(model.index);

                dda(
                    ray_start,
                    ray_dir,
                    10000.0.expr(),
                    (Vec3::splat_expr(0.01), size.cast_f32() - 0.01),
                    |pos, last_t, next_t| {
                        let voxel_index = pos.x + pos.y * size.x + pos.z * size.x * size.y;
                        let voxel = voxels.read(voxel_index);
                        if voxel != 255 {
                            candidate.commit((last_t + next_t) / 2.0);
                            true.expr()
                        } else {
                            false.expr()
                        }
                    },
                );
            })
            .trace();

        if hit.procedural_hit() {
            let model = rendered_models.read(hit.prim);
            let ray_start = model.rotation.transpose() * (view.pos - model.position);
            let ray_dir = model.rotation.transpose() * ray_dir;
            let size = model_sizes.read(model.index);
            let voxels = bindless.buffer::<u8>(model.index);
            let pos = ray_start + ray_dir * hit.committed_ray_t;
            let pos = pos
                .floor()
                .cast_i32()
                .clamp(Vec3::splat(0), size.cast_i32() - 1)
                .cast_u32();
            let color = palette.read(
                voxels
                    .read(pos.x + pos.y * size.x + pos.z * size.x * size.y)
                    .cast_u32(),
            );
            app.set_pixel(pixel.cast_i32(), color);
        }
    }));

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 0.5,
        pitch: 0.6,
        pos: Vec3::new(-500.0, -500.0, 500.0),
        fov: 1.15,
    };

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = if rt.key_down(KeyCode::ControlLeft) {
            5.0
        } else {
            200.0
        } * rt.frame_time();
        rt.process_keys_down(|key| match key {
            KeyCode::ArrowLeft => camera.yaw -= 1.0 * rot_speed,
            KeyCode::ArrowRight => camera.yaw += 1.0 * rot_speed,
            KeyCode::ArrowUp => camera.pitch -= 1.0 * rot_speed,
            KeyCode::ArrowDown => camera.pitch += 1.0 * rot_speed,
            KeyCode::KeyW => camera.pos = camera.pos + camera.forward() * move_speed,
            KeyCode::KeyS => camera.pos = camera.pos - camera.forward() * move_speed,
            KeyCode::KeyA => camera.pos = camera.pos - camera.right() * move_speed,
            KeyCode::KeyD => camera.pos = camera.pos + camera.right() * move_speed,
            KeyCode::Space => camera.pos = camera.pos + Vec3::new(0.0, 0.0, 1.0) * move_speed,
            KeyCode::ShiftLeft => camera.pos = camera.pos - Vec3::new(0.0, 0.0, 1.0) * move_speed,
            _ => (),
        });

        rt.log_fps();

        draw_kernel.dispatch(rt.dispatch_size(), &camera.view());
    });
}
