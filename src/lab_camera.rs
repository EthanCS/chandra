//! Free camera input, kept independent of the window and renderer for testing.
#[derive(Default)]
pub struct Input {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fast: bool,
    pub slow: bool,
    pub focused: bool,
}

#[derive(Default)]
pub struct Pointer {
    previous: Option<glam::Vec2>,
}

#[derive(Default)]
pub struct KeyLatch {
    down: bool,
}
impl KeyLatch {
    pub fn update(&mut self, pressed: bool, down: bool, focused: bool) -> bool {
        let edge = focused && pressed && !self.down;
        self.down = focused && down;
        edge
    }
}

/// Own the native cursor capture; release on focus loss, scene switch and exit.
#[derive(Default)]
pub struct MouseCapture {
    active: bool,
    center: Option<(i32, i32)>,
}

#[cfg(windows)]
mod cursor {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    pub struct Point {
        pub x: i32,
        pub y: i32,
    }
    #[repr(C)]
    #[derive(Default)]
    pub struct Rect {
        pub left: i32,
        pub top: i32,
        pub right: i32,
        pub bottom: i32,
    }
    #[link(name = "user32")]
    unsafe extern "system" {
        pub fn GetForegroundWindow() -> *mut c_void;
        pub fn GetClientRect(window: *mut c_void, rect: *mut Rect) -> i32;
        pub fn ClientToScreen(window: *mut c_void, point: *mut Point) -> i32;
        pub fn GetCursorPos(point: *mut Point) -> i32;
        pub fn SetCursorPos(x: i32, y: i32) -> i32;
        pub fn ClipCursor(rect: *const Rect) -> i32;
        pub fn ShowCursor(show: i32) -> i32;
        pub fn SetCapture(window: *mut c_void) -> *mut c_void;
        pub fn ReleaseCapture() -> i32;
    }
}

impl MouseCapture {
    pub fn is_active(&self) -> bool {
        self.active
    }
    #[cfg(windows)]
    pub fn update(&mut self, active: bool) -> Option<glam::Vec2> {
        use cursor::*;
        if !active || !focused() {
            self.release();
            return None;
        }
        unsafe {
            let window = GetForegroundWindow();
            let mut rect = Rect::default();
            let mut origin = Point::default();
            let mut point = Point::default();
            if GetClientRect(window, &mut rect) == 0
                || ClientToScreen(window, &mut origin) == 0
                || GetCursorPos(&mut point) == 0
            {
                self.release();
                return None;
            }
            let center = (
                origin.x + (rect.right - rect.left) / 2,
                origin.y + (rect.bottom - rect.top) / 2,
            );
            let delta = if let Some(previous) = self.center {
                glam::Vec2::new((point.x - previous.0) as f32, (point.y - previous.1) as f32)
            } else {
                glam::Vec2::ZERO
            };
            if !self.active {
                let clip = Rect {
                    left: origin.x,
                    top: origin.y,
                    right: origin.x + rect.right,
                    bottom: origin.y + rect.bottom,
                };
                ClipCursor(&clip);
                ShowCursor(0);
                SetCapture(window);
                self.active = true;
            }
            if SetCursorPos(center.0, center.1) == 0 {
                self.release();
                return None;
            }
            // A partly off-screen window can clamp the requested centre.
            // Store the actual landing point so a stationary mouse cannot drift.
            let mut landed = Point::default();
            if GetCursorPos(&mut landed) == 0 {
                self.release();
                return None;
            }
            self.center = Some((landed.x, landed.y));
            Some(delta)
        }
    }
    #[cfg(not(windows))]
    pub fn update(&mut self, _active: bool) -> Option<glam::Vec2> {
        None
    }
    pub fn release(&mut self) {
        #[cfg(windows)]
        if self.active {
            unsafe {
                cursor::ClipCursor(std::ptr::null());
                cursor::ReleaseCapture();
                cursor::ShowCursor(1);
            }
        }
        self.active = false;
        self.center = None;
    }
}

pub fn wants_mouse_look(
    focused: bool,
    persistent: bool,
    right_down: bool,
    over_controls: bool,
    captured: bool,
) -> bool {
    focused && (persistent || (right_down && (!over_controls || captured)))
}
impl Drop for MouseCapture {
    fn drop(&mut self) {
        self.release();
    }
}

impl Pointer {
    /// The upstream cursor_velocity is the last event's delta, not this frame's
    /// delta. Consume positions once per frame so a stationary cursor cannot
    /// keep rotating an F-look camera. Re-entry starts a fresh baseline.
    pub fn update(&mut self, position: Option<glam::Vec2>) -> glam::Vec2 {
        let delta = match (position, self.previous) {
            (Some(current), Some(previous)) => current - previous,
            _ => glam::Vec2::ZERO,
        };
        self.previous = position;
        delta
    }
}

pub fn update(camera: &mut yesod::camera::Camera, input: Input, dt: f32) -> bool {
    if !input.focused {
        return false;
    }
    let delta =
        camera.forward() * input.forward + camera.right() * input.right + glam::Vec3::Z * input.up;
    let moving = delta.length_squared() > 0.0 || input.yaw != 0.0 || input.pitch != 0.0;
    let speed = if input.slow {
        60.
    } else if input.fast {
        1400.
    } else {
        350.
    };
    camera.pos += delta.normalize_or_zero() * speed * dt;
    camera.yaw += input.yaw;
    camera.pitch = (camera.pitch + input.pitch).clamp(-1.5, 1.5);
    moving
}

#[cfg(windows)]
pub fn focused() -> bool {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetForegroundWindow() -> *mut std::ffi::c_void;
        fn GetWindowThreadProcessId(window: *mut std::ffi::c_void, process: *mut u32) -> u32;
    }
    let mut process = 0;
    unsafe {
        GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
    }
    process == std::process::id()
}
#[cfg(not(windows))]
pub fn focused() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn viewport_rotation_ignores_root_ui_hover_but_respects_real_controls() {
        assert!(wants_mouse_look(true, false, true, false, false));
        assert!(!wants_mouse_look(true, false, true, true, false));
        assert!(wants_mouse_look(true, true, false, true, false));
        assert!(wants_mouse_look(true, false, true, true, true));
        assert!(!wants_mouse_look(true, false, false, false, true));
        assert!(!wants_mouse_look(false, true, true, false, true));
    }
    #[test]
    fn shortcut_edges_ignore_repeats_and_accept_short_taps() {
        let mut key = KeyLatch::default();
        assert!(key.update(true, true, true));
        assert!(!key.update(true, true, true));
        assert!(!key.update(false, false, true));
        assert!(key.update(true, false, true));
        assert!(!key.update(true, true, false));
        assert!(key.update(true, true, true));
    }
    #[test]
    fn pointer_motion_is_consumed_once_and_reentry_does_not_jump() {
        let mut pointer = Pointer::default();
        assert_eq!(
            pointer.update(Some(glam::Vec2::new(10., 20.))),
            glam::Vec2::ZERO
        );
        assert_eq!(
            pointer.update(Some(glam::Vec2::new(12., 23.))),
            glam::Vec2::new(2., 3.)
        );
        assert_eq!(
            pointer.update(Some(glam::Vec2::new(12., 23.))),
            glam::Vec2::ZERO
        );
        assert_eq!(pointer.update(None), glam::Vec2::ZERO);
        assert_eq!(
            pointer.update(Some(glam::Vec2::splat(10000.))),
            glam::Vec2::ZERO
        );
    }
    #[test]
    fn manual_motion_respects_focus_speed_and_diagonal_normalization() {
        let mut camera = crate::white_room::camera(0, [8, 8]);
        let origin = camera.pos;
        assert!(!update(
            &mut camera,
            Input {
                forward: 1.,
                focused: false,
                ..Default::default()
            },
            1.
        ));
        assert_eq!(camera.pos, origin);
        assert!(update(
            &mut camera,
            Input {
                forward: 1.,
                right: 1.,
                focused: true,
                ..Default::default()
            },
            1.
        ));
        assert!(((camera.pos - origin).length() - 350.).abs() < 0.001);
        let start = camera.pos;
        update(
            &mut camera,
            Input {
                forward: 1.,
                fast: true,
                focused: true,
                ..Default::default()
            },
            0.5,
        );
        assert!(((camera.pos - start).length() - 700.).abs() < 0.001);
    }
}
