use glam::UVec2;
use histo::Histogram;
use image::RgbImage;
use sunfall::noise::*;

fn main() {
    let cascades = 6;
    let base_angular_size = UVec2::new(4, 4);

    // let base_support = 5.0;
    // let noise = compute_noise(
    //     UVec2::new(128, 128),
    //     cascades,
    //     base_angular_size,
    //     base_support,
    // );

    for i in 0..64 {
        let noise = compute_noise_simple(cascades, base_angular_size);

        let max_dir = base_angular_size << (cascades - 1);
        let scaling = 256 / max_dir;
        let image = RgbImage::from_fn(noise.size.x, noise.size.y, |x, y| {
            let dir = noise[UVec2::new(x, y)] * scaling;
            image::Rgb([dir.x as u8, dir.y as u8, 0])
        });
        image.save(format!("assets/noise2/{i}.png")).unwrap();
    }

    // let mut arr = Array2::new(base_angular_size << (cascades - 1), 0.0);
    // let mut histo = Histogram::with_buckets(10);
    // for y in 0..noise.size.y {
    //     for x in 0..noise.size.x {
    //         let pos = UVec2::new(x, y);
    //         let min_dir = calc_min_support(
    //             &noise,
    //             &mut arr,
    //             pos,
    //             base_support * 2_f32.powi(cascades as i32 - 1),
    //         );
    //         histo.add((arr[min_dir] * 1000.0) as u64);
    //     }
    // }
    // println!("{histo}");
}
