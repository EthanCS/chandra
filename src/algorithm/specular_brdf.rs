//! Isotropic GGX with visible-normal sampling and correlated Smith masking.
use super::*;
#[tracked]
pub fn fresnel(f0: Expr<Vec3<f32>>, cosine: Expr<f32>) -> Expr<Vec3<f32>> {
    f0 + (Vec3::splat_expr(1.) - f0) * (1.0 - cosine.clamp(0., 1.)).powf(5.0)
}
#[tracked]
pub fn smith_g1(nv: Expr<f32>, a2: Expr<f32>) -> Expr<f32> {
    2.0 * nv / (nv + (a2 + (1.0 - a2) * nv * nv).sqrt())
}
#[tracked]
pub fn smith_g2(nv: Expr<f32>, nl: Expr<f32>, a2: Expr<f32>) -> Expr<f32> {
    2.0 * nv * nl
        / keter::max(
            nl * (a2 + (1.0 - a2) * nv * nv).sqrt() + nv * (a2 + (1.0 - a2) * nl * nl).sqrt(),
            0.000001,
        )
}
#[tracked]
pub fn evaluate(
    n: Expr<Vec3<f32>>,
    v: Expr<Vec3<f32>>,
    l: Expr<Vec3<f32>>,
    roughness: Expr<f32>,
    f0: Expr<Vec3<f32>>,
) -> Expr<Vec3<f32>> {
    let nv = keter::max(n.dot(v), 0.0001);
    let nl = keter::max(n.dot(l), 0.0);
    let h = (v + l).normalize();
    let nh = keter::max(n.dot(h), 0.0);
    let a = keter::max(roughness * roughness, 0.001);
    let a2 = a * a;
    let d = a2 / (std::f32::consts::PI * (nh * nh * (a2 - 1.0) + 1.0).powf(2.0));
    fresnel(f0, keter::max(v.dot(h), 0.0)) * (d * smith_g2(nv, nl, a2) / (4.0 * nv))
}
#[tracked]
pub fn sample(
    n: Expr<Vec3<f32>>,
    v: Expr<Vec3<f32>>,
    roughness: Expr<f32>,
    f0: Expr<Vec3<f32>>,
    u: Expr<Vec2<f32>>,
) -> (Expr<Vec3<f32>>, Expr<Vec3<f32>>) {
    let axis: Expr<Vec3<f32>> = if n.z.abs() < 0.999 {
        Vec3::z().expr()
    } else {
        Vec3::x().expr()
    };
    let tx = axis.cross(n).normalize();
    let ty = n.cross(tx);
    let local = Vec3::expr(tx.dot(v), ty.dot(v), keter::max(n.dot(v), 0.0001));
    let a = keter::max(roughness * roughness, 0.001);
    let vh = Vec3::expr(a * local.x, a * local.y, local.z).normalize();
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 0.000001 {
        Vec3::expr(-vh.y, vh.x, 0.) / lensq.sqrt()
    } else {
        Vec3::x().expr()
    };
    let t2 = vh.cross(t1);
    let r = u.x.sqrt();
    let phi = u.y * std::f32::consts::TAU;
    let p1 = r * phi.cos();
    let s = (1.0 + vh.z) * 0.5;
    let p2 = (1.0 - s) * (1.0 - p1 * p1).sqrt() + s * r * phi.sin();
    let nh = t1 * p1 + t2 * p2 + vh * keter::max(0.0, 1.0 - p1 * p1 - p2 * p2).sqrt();
    let h = Vec3::expr(a * nh.x, a * nh.y, keter::max(0.0, nh.z)).normalize();
    let h = tx * h.x + ty * h.y + n * h.z;
    let l = if roughness <= 0.001 {
        n * (2.0 * n.dot(v)) - v
    } else {
        h * (2.0 * v.dot(h)) - v
    };
    let nl = keter::max(n.dot(l), 0.0);
    let nv = keter::max(n.dot(v), 0.0001);
    let weight = if roughness <= 0.001 {
        fresnel(f0, nv)
    } else {
        fresnel(f0, v.dot(h)) * (smith_g2(nv, nl, a * a) / smith_g1(nv, a * a))
    };
    (l, weight)
}

/// Digitally shifted 4D Sobol net: two dimensions per glossy bounce.
/// Power-of-two prefixes stay stratified as the adaptive budget grows.
/// The independent per-pixel/frame shift preserves progressive accumulation;
/// this is digital scrambling, not a claimed Owen/blue-noise sampler.
#[tracked]
pub fn path_samples(
    pixel: Expr<Vec2<u32>>,
    frame: Expr<u32>,
    sample: Expr<u32>,
) -> Expr<Vec4<f32>> {
    // Standard direction numbers for dimensions 1..4, generated from the
    // primitive-polynomial parameters (1,0,[1]), (2,1,[1,3]), (3,1,[1,3,1]).
    let directions = [
        [0x80000000u32, 0x80000000u32, 0x80000000u32, 0x80000000u32],
        [0x40000000u32, 0xc0000000u32, 0xc0000000u32, 0xc0000000u32],
        [0x20000000u32, 0xa0000000u32, 0x60000000u32, 0x20000000u32],
        [0x10000000u32, 0xf0000000u32, 0x90000000u32, 0x50000000u32],
        [0x08000000u32, 0x88000000u32, 0xe8000000u32, 0xf8000000u32],
        [0x04000000u32, 0xcc000000u32, 0x5c000000u32, 0x74000000u32],
        [0x02000000u32, 0xaa000000u32, 0x8e000000u32, 0xa2000000u32],
    ];
    let shift = yesod::rand::pcg3d(pixel.extend(frame));
    let fourth =
        yesod::rand::pcg(frame * 1013904223u32 + pixel.x * 1664525u32 + pixel.y * 747796405u32);
    let bits = shift.extend(fourth).var();
    for (j, v) in directions.into_iter().enumerate() {
        if (sample & (1u32 << j)) != 0u32 {
            *bits ^= Vec4::new(v[0], v[1], v[2], v[3]);
        }
    }
    (bits >> 8u32).cast_f32() * (1.0f32 / 16777216.0)
}
