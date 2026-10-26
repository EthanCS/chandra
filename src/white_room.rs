//! Asset-free, centimetre-scale diffuse GI and light-leak fixtures.
use keter::lang::types::vector::Vec3;

use crate::algorithm::WorldData;
use crate::world::TriangleWorld;

pub fn world() -> WorldData {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut offsets = Vec::new();
    let mut block = |min: [f32; 3], max: [f32; 3]| {
        let [x, y, z] = min;
        let [a, b, c] = max;
        offsets.push([triangles.len() as u32, vertices.len() as u32]);
        vertices.extend([
            [x, y, z],
            [a, y, z],
            [a, b, z],
            [x, b, z],
            [x, y, c],
            [a, y, c],
            [a, b, c],
            [x, b, c],
        ]);
        triangles.extend([
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]);
    };
    // The camera is inside a complete shell. The second shell has an actual
    // 8 cm roof slit; neither shell has a camera-shaped opening.
    for (x, slit) in [(2000., false), (2400., true)] {
        block([x - 4., -4., -4.], [x + 244., 304., 0.]);
        block([x - 4., -4., 0.], [x, 304., 220.]);
        block([x + 240., -4., 0.], [x + 244., 304., 220.]);
        block([x, -4., 0.], [x + 240., 0., 220.]);
        block([x, 300., 0.], [x + 240., 304., 220.]);
        if slit {
            block([x - 4., -4., 220.], [x + 244., 146., 224.]);
            block([x - 4., 154., 220.], [x + 244., 304., 224.]);
        } else {
            block([x - 4., -4., 220.], [x + 244., 304., 224.]);
        }
    }
    // Open cell with 2 / 12 cm partitions and a roof corner admitting real sky.
    block([2800., -4., -4.], [3440., 304., 0.]);
    block([2800., 300., 0.], [3440., 304., 220.]);
    block([2870., 0., 0.], [2872., 300., 180.]);
    block([2960., 0., 0.], [2972., 300., 180.]);
    block([3200., 0., 0.], [3204., 300., 220.]);
    block([3200., 150., 220.], [3440., 304., 224.]);
    WorldData {
        world: TriangleWorld::new(
            &vertices,
            &triangles,
            &offsets,
            &vec![0; offsets.len()],
            &[Vec3::splat(0.8)],
        ),
        envmap: image::Rgb32FImage::from_pixel(8, 4, image::Rgb([0.65, 0.78, 1.0])),
        envmap_scale: 0.18,
        sun_dir: Vec3::new(-0.5, -0.6, 0.7),
        sun_color: Vec3::splat(5.0),
        custom_emitters: vec![],
    }
}

pub fn camera(index: usize, size: [u32; 2]) -> yesod::camera::Camera {
    let (pos, target) = match index {
        1 => (
            glam::Vec3::new(2520., 45., 110.),
            glam::Vec3::new(2520., 215., 30.),
        ),
        2 => (
            glam::Vec3::new(2850., -180., 220.),
            glam::Vec3::new(2930., 150., 70.),
        ),
        3 => (
            glam::Vec3::new(3380., 35., 140.),
            glam::Vec3::new(3220., 250., 100.),
        ),
        _ => (
            glam::Vec3::new(2120., 65., 110.),
            glam::Vec3::new(2120., 240., 105.),
        ),
    };
    let direction = (target - pos).normalize();
    yesod::camera::Camera {
        pos,
        yaw: direction.x.atan2(direction.y),
        pitch: (-direction.z).asin(),
        screen_size: glam::Vec2::new(size[0] as f32, size[1] as f32),
        fov: 1.05,
    }
}
