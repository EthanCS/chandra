//! Deterministic trajectories shared by low-sample and reference captures.
use yesod::camera::Camera;

pub fn motion_time(frame: u32, dt: f32, stop: u32) -> f32 {
    let frame = frame.min(stop.saturating_sub(1));
    (frame.saturating_sub(119)) as f32 * dt
}

pub fn camera(base: Camera, track: &str, frame: u32, dt: f32, stop: u32) -> Camera {
    let stop = if track == "stop" { stop.min(208) } else { stop };
    let t = motion_time(frame, dt, stop);
    let mut camera = base;
    if matches!(track, "translate" | "combined" | "stop") {
        camera.pos += base.right() * 40. * (t * 1.3).sin() + base.forward() * 30. * (t * 0.9).sin();
    }
    if matches!(track, "rotate" | "combined" | "stop") {
        camera.yaw += 0.4 * (t * 1.3).sin();
        camera.pitch += 0.12 * (t * 0.7).sin();
    }
    if track == "orbit" {
        let target = base.pos + base.forward() * 500.;
        camera.pos = target + glam::Quat::from_rotation_z(t * 0.15) * (base.pos - target);
        let direction = (target - camera.pos).normalize();
        camera.yaw = direction.x.atan2(direction.y);
        camera.pitch = (-direction.z).asin();
    }
    if track == "return" {
        camera.yaw += 2.7 * (std::f32::consts::PI * (t / 1.8).clamp(0., 1.)).sin();
        camera.pos += base.forward() * 80. * (std::f32::consts::PI * (t / 1.8).clamp(0., 1.)).sin();
    }
    camera
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trajectories_share_time_across_frame_rates_and_stop_at_the_same_pose() {
        let base = crate::white_room::camera(1, [960, 600]);
        for track in ["orbit", "translate", "rotate", "combined", "stop", "return"] {
            let a = camera(base, track, 149, 1. / 60., u32::MAX);
            let b = camera(base, track, 179, 1. / 120., u32::MAX);
            assert!((a.pos - b.pos).length() < 1e-3);
            assert!((a.yaw - b.yaw).abs() < 1e-6);
            let before = camera(base, track, 119, 1. / 60., u32::MAX);
            assert!((before.pos - base.pos).length() < 1e-3);
            let a = camera(base, track, 207, 1. / 60., 208);
            let b = camera(base, track, 600, 1. / 60., 208);
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.yaw, b.yaw);
        }
    }
}
