//! Conservative current-frame light lists. Culling changes only which lights
//! reach the existing exact distance/cosine/visibility tests, in ascending order.
use super::*;

#[derive(Clone, Copy, Value)]
#[repr(C)]
struct GridInfo {
    origin_scale: Vec4<f32>,
    dimensions: Vec3<u32>,
}

pub(super) struct LightGrid {
    info: Singleton<GridInfo>,
    masks: Buffer<u32>,
}

impl LightGrid {
    pub(super) fn new(lights: &[PointLight]) -> Self {
        let (info, masks) = build(lights);
        let grid = Self { info: Singleton::new(info), masks: DEVICE.create_buffer(32 * 32 * 32) };
        grid.masks.view(..masks.len()).copy_from(&masks);
        grid
    }
    pub(super) fn update(&self, lights: &[PointLight]) -> NodeConfigs<'static> {
        let (info, masks) = build(lights);
        (self.masks.view(..masks.len()).copy_from_vec(masks), self.info.write_host(info)).chain()
    }
    #[tracked]
    pub(super) fn candidates(&self, position: Expr<Vec3<f32>>) -> Expr<u32> {
        let info = self.info.read();
        let cell = ((position - info.origin_scale.xyz()) * info.origin_scale.w).floor().cast_i32();
        let mask = 0u32.var();
        if (cell >= 0i32).all() && (cell < info.dimensions.cast_i32()).all() {
            let cell = cell.cast_u32();
            let index = cell.x + info.dimensions.x * (cell.y + info.dimensions.y * cell.z);
            *mask = self.masks.read(index);
        }
        **mask
    }
}

fn build(lights: &[PointLight]) -> (GridInfo, Vec<u32>) {
    use glam::Vec3 as V;
    assert!(!lights.is_empty() && lights.len() <= 32);
    let mut min = V::splat(f32::INFINITY);
    let mut max = V::splat(f32::NEG_INFINITY);
    let mut cell_size = 1.0f32;
    for light in lights {
        let p = light.position_range;
        assert!([p.x,p.y,p.z,p.w].iter().all(|v| v.is_finite()) && p.w >= 0.);
        let position = V::new(p.x,p.y,p.z);
        let radius = V::splat(p.w + 0.01);
        min = min.min(position - radius);
        max = max.max(position + radius);
        cell_size = cell_size.max(p.w);
    }
    cell_size = cell_size.max((max - min).max_element() / 30.);
    let scale = cell_size.recip();
    let dimensions = ((max - min) * scale).floor().as_uvec3() + 1;
    assert!(dimensions.max_element() <= 32);
    let mut masks = vec![0u32; (dimensions.x * dimensions.y * dimensions.z) as usize];
    for (i, light) in lights.iter().enumerate() {
        let p = light.position_range;
        let center = V::new(p.x,p.y,p.z);
        let radius = V::splat(p.w + 0.01);
        let lo = ((center - radius - min) * scale).floor().max(V::ZERO).as_uvec3();
        let hi = ((center + radius - min) * scale).floor().as_uvec3().min(dimensions - 1);
        for z in lo.z..=hi.z { for y in lo.y..=hi.y { for x in lo.x..=hi.x {
            masks[(x + dimensions.x * (y + dimensions.y * z)) as usize] |= 1 << i;
        }}}
    }
    (GridInfo { origin_scale: Vec4::new(min.x,min.y,min.z,scale), dimensions: dimensions.to_array().into() }, masks)
}
