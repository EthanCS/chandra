/*
Design:
NxN (N=128ish) wrapping pattern.
Each ray is actually a projective line, so flipping if the normal is wrong works.

Need each area to fill up all of the cascade levels efficiently. For each pixel, sample areas from a gaussian distribution proportional to the cascade. (calculate average probe size?)

Each pixel needs to fill up all of the levels. So first choose c0 sampling, then choose c1 somehow? Or have c1-and-higher influence the lower ones as well.
Also each pixel needs to not repeat itself as much as possible so that small probes eventually get filled up. Probably already fulfilled by the gaussian thing.

Don't particularly bother with temporal stability since the camera is expected to be moving anyways.
*/

use core::f32;
use std::f32::consts::PI;
use std::ops::{Index, IndexMut};

use glam::{IVec2, UVec2, Vec2};
use rand::seq::SliceRandom;

fn normalizer(h: f32) -> f32 {
    (40.0 / 7.0) / (PI * h * h)
}

// Taken from https://docs.rs/salva2d/latest/src/salva2d/kernel/cubic_spline_kernel.rs.html.
fn cubic_spline(q: f32) -> f32 {
    if q < 0.5 {
        let q2 = q * q;
        1.0 + 6.0 * (q2 * q - q2)
    } else if q <= 1.0 {
        let nq = 1.0 - q;
        2.0 * nq * nq * nq
    } else {
        0.0
    }
}
fn cubic_spline_norm(r: f32, h: f32) -> f32 {
    normalizer(h) * cubic_spline(r / h)
}

#[derive(Debug, Clone)]
pub struct Array2<T> {
    pub data: Vec<T>,
    pub size: UVec2,
}
impl<T> Array2<T> {
    pub fn new(size: UVec2, value: T) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![value; size.element_product() as usize],
            size,
        }
    }
    pub fn from_fn(size: UVec2, mut f: impl FnMut(UVec2) -> T) -> Self {
        Self {
            data: (0..size.element_product())
                .map(|i| {
                    let index = UVec2::new(i % size.x, i / size.x);
                    f(index)
                })
                .collect(),
            size,
        }
    }
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }
    pub fn to_linear(&self, index: UVec2) -> usize {
        assert!(index.cmplt(self.size).all());
        (index.y * self.size.x + index.x) as usize
    }
    pub fn to_linear_repeating(&self, index: IVec2) -> usize {
        let index = index.rem_euclid(self.size.as_ivec2()).as_uvec2();
        (index.y * self.size.x + index.x) as usize
    }
    pub fn to_grid(&self, index: usize) -> UVec2 {
        assert!(index < self.linear_size());
        UVec2::new(index as u32 % self.size.x, index as u32 / self.size.x)
    }
    pub fn linear_size(&self) -> usize {
        self.size.element_product() as usize
    }
    pub fn enumerate(&self) -> impl Iterator<Item = (UVec2, &T)> {
        self.data
            .iter()
            .enumerate()
            .map(move |(i, v)| (self.to_grid(i), v))
    }
}
impl<T> Index<IVec2> for Array2<T> {
    type Output = T;

    fn index(&self, index: IVec2) -> &Self::Output {
        &self.data[self.to_linear_repeating(index)]
    }
}
impl<T> IndexMut<IVec2> for Array2<T> {
    fn index_mut(&mut self, index: IVec2) -> &mut Self::Output {
        let index = self.to_linear_repeating(index);
        &mut self.data[index]
    }
}
impl<T> Index<UVec2> for Array2<T> {
    type Output = T;

    fn index(&self, index: UVec2) -> &Self::Output {
        &self.data[self.to_linear(index)]
    }
}
impl<T> IndexMut<UVec2> for Array2<T> {
    fn index_mut(&mut self, index: UVec2) -> &mut Self::Output {
        let index = self.to_linear(index);
        &mut self.data[index]
    }
}

fn min_slice_rand(a: &[f32]) -> usize {
    let mut min = f32::INFINITY;
    let mut count = 0;
    for &v in a {
        if v < min {
            min = v;
            count = 1;
        } else if v == min {
            count += 1;
        }
    }
    let mut target = rand::random_range(0..count);
    for (i, &v) in a.iter().enumerate() {
        if v == min {
            if target == 0 {
                return i;
            } else {
                target -= 1;
            }
        }
    }
    unreachable!();
}
fn min_array_rand(a: &Array2<f32>) -> UVec2 {
    a.to_grid(min_slice_rand(&a.data))
}

pub fn compute_noise_simple(cascades: u32, base_angular_size: UVec2) -> Array2<UVec2> {
    let mut rays = Array2::from_fn(base_angular_size, |pos| pos);
    rays.data.shuffle(&mut rand::rng());
    for level in 1..cascades {
        let mut next_rays = Array2::new(base_angular_size << level, UVec2::MAX);

        let offsets = [UVec2::ZERO, UVec2::X, UVec2::Y, UVec2::ONE];
        for (pos, facing) in rays.enumerate() {
            let mut shfl_offsets = offsets;
            shfl_offsets.shuffle(&mut rand::rng());
            for (offset, shfl_offset) in offsets.iter().zip(shfl_offsets.iter()) {
                let next_pos = pos + offset * rays.size;
                let facing = facing * 2 + shfl_offset;
                next_rays[next_pos] = facing;
            }
        }
        rays = next_rays;
    }
    rays
}

// TODO: Add projective line.. actually can probably just cut off X axis by half and mirror..
// Also this is incredibly unoptimized.
pub fn compute_noise(
    size: UVec2,
    cascades: u32,
    base_angular_size: UVec2,
    base_support: f32,
) -> Array2<UVec2> {
    let mut rays = Array2::new(size, UVec2::MAX);
    // c0
    let mut pixels = (0..rays.linear_size())
        .map(|i| rays.to_grid(i))
        .collect::<Vec<_>>();
    pixels.shuffle(&mut rand::rng());
    let mut ray_dirs = Array2::new(base_angular_size, 0.0_f32);
    for &pos in &pixels {
        let min_dir = calc_min_support(&rays, &mut ray_dirs, pos, base_support);
        rays[pos] = min_dir;
    }

    // c1+
    for level in 1..cascades {
        let support = base_support * (2.0_f32).powi(level as i32);

        let mut next_rays = Array2::new(size, UVec2::MAX);
        pixels.shuffle(&mut rand::rng());
        let mut ray_dirs = Array2::new(UVec2::splat(2), 0.0_f32);
        for &pos in &pixels {
            let min_dir = calc_min_support_narrow(&rays, &next_rays, &mut ray_dirs, pos, support);
            next_rays[pos] = min_dir;
        }
        rays = next_rays;
    }

    rays
}

pub fn calc_min_support_narrow(
    last_rays: &Array2<UVec2>,
    rays: &Array2<UVec2>,
    ray_dirs: &mut Array2<f32>,
    pos: UVec2,
    support: f32,
) -> UVec2 {
    ray_dirs.fill(0.0);
    let pos = pos.as_ivec2();

    let facing = last_rays[pos];

    let s = support.ceil() as i32;
    for i in -s..=s {
        for j in -s..=s {
            let offset = IVec2::new(i, j);
            let r = offset.as_vec2().length();
            let weight = cubic_spline(r / support);
            if weight == 0.0 {
                continue;
            }
            let neighbor_pos = pos + offset;
            let neighbor_facing = last_rays[neighbor_pos];
            if neighbor_facing != facing {
                continue;
            }
            let neighbor_dir = rays[pos + offset];
            if neighbor_dir != UVec2::MAX {
                ray_dirs[neighbor_dir - facing * 2] += weight;
            }
        }
    }
    min_array_rand(ray_dirs) + facing * 2
}

pub fn calc_min_support(
    rays: &Array2<UVec2>,
    ray_dirs: &mut Array2<f32>,
    pos: UVec2,
    support: f32,
) -> UVec2 {
    ray_dirs.fill(0.0);
    let pos = pos.as_ivec2();
    let s = support.ceil() as i32;
    for i in -s..=s {
        for j in -s..=s {
            let offset = IVec2::new(i, j);
            let r = offset.as_vec2().length();
            let weight = cubic_spline(r / support);
            if weight == 0.0 {
                continue;
            }
            let neighbor_dir = rays[pos + offset];
            if neighbor_dir != UVec2::MAX {
                ray_dirs[neighbor_dir] += weight;
            }
        }
    }
    min_array_rand(ray_dirs)
}
