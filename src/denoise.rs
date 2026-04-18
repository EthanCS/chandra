use keter::lang::types::vector::Vec3;
use keter::prelude::*;
use yesod::utils::once;

use crate::algorithm::CoreData;

pub struct Denoiser {
    denoiser: oidn::RayTracing<'static>,
    albedo: Vec<f32>,
    input: Buffer<Vec3<f32>>,
    normal: Buffer<Vec3<f32>>,
    pub output: Buffer<Vec3<f32>>,
}

impl Denoiser {
    pub fn new(core: &CoreData) -> Self {
        let device = oidn::Device::new();
        let device = Box::leak(Box::new(device));

        let mut denoiser = oidn::RayTracing::new(device);
        denoiser
            .hdr(true)
            .srgb(false)
            .clean_aux(true)
            .filter_quality(oidn::Quality::High)
            .image_dimensions(core.screen_size().x as usize, core.screen_size().y as usize);

        Self {
            denoiser,
            input: DEVICE.create_buffer::<Vec3<f32>>(core.num_pixels() as usize),
            normal: DEVICE.create_buffer::<Vec3<f32>>(core.num_pixels() as usize),
            albedo: vec![0.7; 3 * core.num_pixels() as usize],
            output: DEVICE.create_buffer::<Vec3<f32>>(core.num_pixels() as usize),
        }
    }
    pub fn denoise(&mut self, core: &CoreData, input: &Tex2d<Vec3<f32>>) {
        once(|| {
            DEVICE.create_kernel_async::<fn(Tex2d<Vec3<f32>>)>(&|input| {
                let pixel = dispatch_id().xy();
                let index = core.pixel_to_linear(pixel);
                self.input.write(index, input.read(pixel));
                self.normal.write(index, core.read_normal(pixel));
            })
        })
        .dispatch(core.screen_dispatch_size(), input);
        let mut colors = self
            .input
            .copy_to_vec()
            .iter()
            .flat_map(|c| [c.x, c.y, c.z])
            .collect::<Vec<f32>>();

        let normals = self
            .normal
            .copy_to_vec()
            .iter()
            .flat_map(|n| [n.x, n.y, n.z])
            .collect::<Vec<f32>>();

        self.denoiser
            .albedo_normal(&self.albedo, &normals)
            .filter_in_place(&mut colors)
            .unwrap();

        let output_data = colors
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect::<Vec<Vec3<f32>>>();
        self.output.copy_from(&output_data);
    }
}
