use std::f32::consts::TAU;

use yesod::color::luma;
use yesod::rand::pcg3df;
use yesod::shapes::intersect_sphere;

use super::*;

#[derive(Debug, Clone, Copy, Value, Default)]
#[repr(C)]
pub struct Sample {
    pub hit_pos: Vec3<f32>,
    pub emission: Vec3<f32>,
}

/*
Then, for each pixel the base sampler p(x) has uniform probability of giving a hit point in each direction (within the hemisphere).

The target distribution is contribution-weighted; so p^(x) = p(x) * f(x), unnormalized.
If we had a normalized p_2(x) = p^(x) / integral p^(x), then the MC estimator would be f(x) / p_2(x) = p(x) * integral p^(x) which is constant.
*/

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct WeightedSample {
    pub sample: Sample,
    pub contribution: f32,
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Reservior {
    pub sample: Sample,
    // evaluation of p^(x) at the sample, used at the end.
    pub sample_target_pdf: f32,
    pub weight_sum: f32,
    // Built-in MIS 1/M factor.
    pub total_samples: f32,
}
impl Default for Reservior {
    fn default() -> Self {
        Self {
            sample: Sample::default(),
            sample_target_pdf: 0.0,
            weight_sum: 0.0,
            total_samples: 0.0,
        }
    }
}
impl ReserviorVar {
    #[tracked]
    fn update(
        &self,
        new_sample: Expr<Sample>,
        sample_target_pdf: Expr<f32>,
        // unbiased contribution weight W_X
        contribution: Expr<f32>,
        // used for average-based MIS.
        count: Expr<f32>,
        rand: Expr<f32>,
    ) {
        let weight = sample_target_pdf * contribution;
        *self.total_samples += count;
        *self.weight_sum += weight * count;
        if rand < (weight / self.weight_sum) {
            *self.sample = new_sample;
            *self.sample_target_pdf = sample_target_pdf;
        }
    }
    #[tracked]
    fn _combine(&self, other: &Expr<Reservior>, rand: Expr<f32>) {
        let sample = other.finish();
        self.update(
            sample.sample,
            other.sample_target_pdf,
            sample.contribution,
            other.total_samples,
            rand,
        );
    }
    #[tracked]
    fn clamp_samples(&self, max: Expr<f32>) {
        if self.total_samples > max {
            let scale = max / self.total_samples;
            *self.total_samples = max;
            *self.weight_sum *= scale;
        }
    }
}
impl ReserviorExpr {
    #[tracked]
    fn finish(self) -> Expr<WeightedSample> {
        WeightedSample::from_comps_expr(WeightedSampleComps {
            sample: self.sample,
            contribution: self.sample_target_pdf.recip() * self.weight_sum / self.total_samples,
        })
    }
}

pub struct RisData {
    pub reserviors: Buffer<Reservior>,
    pub next_reserviors: Buffer<Reservior>,
    pub output: Tex2d<Vec3<f32>>,
    // pub search_radius: f32,
    // pub max_depth_change: f32,
    // pub max_normal_change: f32,
    // pub num_merge_samples: u32,
}

// const HEMI_INTEGRAL: f32 = TAU;

impl RisData {
    pub fn new(core: &CoreData) -> Self {
        Self {
            reserviors: DEVICE
                .create_buffer_from_fn(core.num_pixels() as usize, |_| Reservior::default()),
            next_reserviors: DEVICE
                .create_buffer_from_fn(core.num_pixels() as usize, |_| Reservior::default()),
            output: DEVICE.create_tex2d::<Vec3<f32>>(
                PixelStorage::Float4,
                core.screen_size().x,
                core.screen_size().y,
                1,
            ),
            // search_radius: 30.0,
            // max_depth_change: 0.1,
            // max_normal_change: 0.5,
            // num_merge_samples: 5,
        }
    }
    pub fn reset(&self) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| { self.reserviors.write(dispatch_id().x, Reservior::default()) }),
            [self.reserviors.len() as u32, 1, 1],
        )
        .debug("Reset RIS")
    }
    pub fn update(&self, core: &CoreData, tick: u32) -> NodeConfigs<'static> {
        dispatch1(
            track!(|tick| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();

                let pos = core.get_world_pos(pixel);

                let normal = core.read_normal(pixel);
                let rand = pcg3df(pixel.extend(tick * 2));
                let sample_dir = SphericalEncoder::decode(rand.xy());
                let sample_dir = sample_dir * sample_dir.dot(normal).signum();
                let pos = offset_ray(pos, normal);
                let hit = core.world.trace(pos, sample_dir);
                // 1/2pi isn't necessary??
                // p(x) = 1 / (2 * PI), W_x = 1 / p(x)
                // p^(x) = p(x) * f(x)
                // Also there's no cos factor here currently; hopefully shouldn't be a problem.

                let sample = if hit.miss() {
                    // use a fake skybox to avoid special casing
                    let t = intersect_sphere(pos, sample_dir, 1e5.expr()).0.y;
                    let hit_pos = pos + sample_dir * t;
                    let emission = Vec3::splat_expr(0.0);
                    Sample::from_comps_expr(SampleComps { hit_pos, emission })
                } else {
                    let hit_pos = pos + sample_dir * hit.committed_ray_t;
                    let emission = core.get_emission(None, hit_pos, hit);
                    Sample::from_comps_expr(SampleComps { hit_pos, emission })
                };

                let sample_target_pdf = luma(sample.emission);

                let pixel_index = core.pixel_to_linear(pixel);

                let reservior = self.reserviors.read(pixel_index).var();
                reservior.update(sample, sample_target_pdf, 1.0.expr(), 1.0.expr(), rand.z);

                reservior.clamp_samples(100.0.expr());

                self.next_reserviors.write(pixel_index, reservior);

                let weighted_sample = reservior.finish();

                let sample = weighted_sample.sample;

                let dir = (sample.hit_pos - pos).normalize();

                self.output.write(
                    pixel,
                    sample.emission
                        * weighted_sample.contribution
                        * keter::max(dir.dot(normal), 0.0)
                        * TAU,
                );
            }),
            core.screen_dispatch_size(),
            tick,
        )
        .debug("Update RIS")
    }
    pub fn copy_back(&self) -> NodeConfigs<'static> {
        self.next_reserviors
            .copy_to_buffer_async(&self.reserviors)
            .debug("Copy RIS Reserviors Back")
    }
}
