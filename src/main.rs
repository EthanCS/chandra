use keter::lang::types::vector::{Mat3, Vec2, Vec3};
use keter::prelude::*;
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
    fn rotation(&self) -> glam::Mat3 {
        glam::Mat3::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }
    fn view(&self) -> View {
        let view = self.rotation();
        let ratio = (self.fov / 2.0).tan() / (self.screen_size.y / 2.0);
        let view = glam::Mat3::from_cols(view.x_axis * ratio, -view.y_axis * ratio, view.z_axis);

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
    f: impl Fn(Expr<Vec3<u32>>) -> Expr<bool>,
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

        loop {
            let next_t = side_dist.reduce_min();

            if f(**pos) {
                break;
            }

            if next_t >= end_t {
                break;
            }
            let mask = side_dist <= keter::min(side_dist.yzx(), side_dist.zxy());

            *side_dist += mask.select(delta_dist, Vec3::splat_expr(0.0));
            *pos += mask.select(ray_step, Vec3::splat_expr(0));
        }
    }
}

fn main() {
    let app = App::new("Amitayus", [2048; 2]).agx().resize().init();

    let file = dot_vox::load("assets/castle.vox").unwrap();

    let palette = DEVICE.create_buffer_from_fn(file.palette.len(), |i| {
        let color = file.palette[i];
        Vec3::new(color.r as f32, color.g as f32, color.b as f32) * (1.0 / 255.0_f32)
    });

    let voxels = DEVICE.create_buffer::<u8>(256 * 256 * 256);

    let draw_kernel = DEVICE.create_kernel::<fn(View)>(&track!(|view| {
        let pixel = dispatch_id().xy();
        let ray_dir = view.transform * (pixel.cast_f32() - view.screen_size / 2.0).extend(1.0);
        let ray_dir = ray_dir.normalize();

        let color_index = 255_u8.var();
        dda(
            view.pos,
            ray_dir,
            10000.0.expr(),
            (Vec3::splat_expr(0.01), Vec3::splat_expr(254.9)),
            |pos| {
                let voxel_index = pos.x + pos.y * 256 + pos.z * 256 * 256;
                *color_index = voxels.read(voxel_index);
                color_index != 255
            },
        );
        if color_index != 255 {
            app.set_pixel(pixel.cast_i32(), palette.read(color_index.cast_u32()));
        }
    }));

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 0.0,
        pitch: 0.0,
        pos: Vec3::new(128.0, 128.0, -256.0),
        fov: 1.15,
    };

    let mut model_index = 0;

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = 200.0 * rt.frame_time();
        rt.process_keys_down(|key| match key {
            KeyCode::ArrowLeft => camera.yaw -= 1.0 * rot_speed,
            KeyCode::ArrowRight => camera.yaw += 1.0 * rot_speed,
            KeyCode::ArrowUp => camera.pitch -= 1.0 * rot_speed,
            KeyCode::ArrowDown => camera.pitch += 1.0 * rot_speed,
            KeyCode::KeyW => camera.pos = camera.pos + camera.forward() * move_speed,
            KeyCode::KeyS => camera.pos = camera.pos - camera.forward() * move_speed,
            KeyCode::KeyA => camera.pos = camera.pos - camera.right() * move_speed,
            KeyCode::KeyD => camera.pos = camera.pos + camera.right() * move_speed,
            KeyCode::Space => camera.pos = camera.pos + Vec3::new(0.0, 1.0, 0.0) * move_speed,
            KeyCode::ShiftLeft => camera.pos = camera.pos - Vec3::new(0.0, 1.0, 0.0) * move_speed,
            _ => (),
        });

        let mut model_changed = rt.tick == 0;

        if rt.key_pressed(KeyCode::KeyQ) {
            model_index = (model_index + 1) % file.models.len();
            model_changed = true;
        }

        if model_changed {
            let model = &file.models[model_index];
            let mut host_voxels = vec![255; 256 * 256 * 256];
            for voxel in &model.voxels {
                host_voxels[(voxel.x as u32 + voxel.y as u32 * 256 + voxel.z as u32 * 256 * 256)
                    as usize] = voxel.i;
            }
            voxels.copy_from(&host_voxels);
        }

        draw_kernel.dispatch(rt.dispatch_size(), &camera.view());
    });
}
