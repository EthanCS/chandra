//! PICA PICA: author's static meshes, normals, textures and published materials.
use std::collections::HashMap;
use std::path::Path;

use chandra::algorithm::WorldData;
use chandra::world::{PbrMaterial, TriangleWorld};
use image::{Rgb, Rgb32FImage};
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;

pub fn load(path: &Path) -> WorldData {
    let gltf = gltf::Gltf::open(path).expect("PICA missing; run tools/setup-pica-pica.ps1");
    let parent = path.parent().unwrap();
    let buffers: Vec<Vec<u8>> = gltf
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Uri(uri) => std::fs::read(parent.join(uri)).unwrap(),
            gltf::buffer::Source::Bin => gltf.blob.clone().unwrap(),
        })
        .collect();
    let mut geometry = crate::gltf_scene::Geometry {
        include_mask: true,
        ..Default::default()
    };
    let basis = glam::Mat4::from_cols(
        glam::Vec4::new(100., 0., 0., 0.),
        glam::Vec4::new(0., 0., 100., 0.),
        glam::Vec4::new(0., -100., 0., 0.),
        glam::Vec4::W,
    );
    for node in gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .unwrap()
        .nodes()
    {
        geometry.visit(node, basis, &buffers, gltf.materials().len() as u32);
    }
    let mut textures = Vec::new();
    let mut uploaded = HashMap::new();
    let mut upload = |texture: gltf::Texture<'_>, srgb: bool| -> u32 {
        let source = texture.source();
        *uploaded.entry((source.index(), srgb)).or_insert_with(|| {
            let image = match source.source() {
                gltf::image::Source::Uri { uri, .. } => image::open(parent.join(uri)).unwrap(),
                gltf::image::Source::View { view, .. } => image::load_from_memory(
                    &buffers[view.buffer().index()][view.offset()..view.offset() + view.length()],
                )
                .unwrap(),
            }
            .to_rgba8();
            let (w, h) = image.dimensions();
            let levels = w.max(h).ilog2() + 1;
            let tex = DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Half4, w, h, levels);
            let mut pixels: Vec<[f32; 4]> = image
                .pixels()
                .map(|p| {
                    std::array::from_fn(|i| {
                        let v = p[i] as f32 / 255.;
                        if srgb && i < 3 {
                            if v <= 0.04045 {
                                v / 12.92
                            } else {
                                ((v + 0.055) / 1.055).powf(2.4)
                            }
                        } else {
                            v
                        }
                    })
                })
                .collect();
            let (mut width, mut height) = (w, h);
            for level in 0..levels {
                // Half4 upload is generated on GPU from float pixels.
                let input = DEVICE.create_buffer_from_slice(&pixels);
                let upload = DEVICE.create_kernel::<fn()>(&track!(|| {
                    let p = dispatch_id().xy();
                    tex.view(level)
                        .write(p, Expr::<Vec4<f32>>::from(input.read(p.y * width + p.x)));
                }));
                upload.dispatch([width, height, 1]);
                let (nw, nh) = ((width / 2).max(1), (height / 2).max(1));
                let old = &pixels;
                let mut next = Vec::with_capacity((nw * nh) as usize);
                for y in 0..nh {
                    for x in 0..nw {
                        let mut sum = [0.; 4];
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let p = old[(((y * 2 + dy).min(height - 1)) * width
                                    + (x * 2 + dx).min(width - 1))
                                    as usize];
                                for i in 0..4 {
                                    sum[i] += p[i] * 0.25;
                                }
                            }
                        }
                        next.push(sum);
                    }
                }
                pixels = next;
                width = nw;
                height = nh;
            }
            let index = textures.len() as u32;
            textures.push(tex);
            index
        })
    };
    let mut materials = Vec::new();
    let mut colors = Vec::new();
    for m in gltf.materials() {
        let p = m.pbr_metallic_roughness();
        let c = p.base_color_factor();
        colors.push(Vec3::new(c[0], c[1], c[2]));
        let mut mat = PbrMaterial::default();
        mat.parameters = Vec4::new(
            p.roughness_factor(),
            p.metallic_factor(),
            0.04,
            m.normal_texture().map_or(1., |t| t.scale()),
        );
        let e = m.emissive_factor();
        mat.emission_alpha = Vec4::new(e[0], e[1], e[2], c[3]);
        if m.alpha_mode() == gltf::material::AlphaMode::Mask {
            mat.alpha_cutoff = m.alpha_cutoff().unwrap_or(0.5);
        }
        if let Some(t) = p.base_color_texture() {
            assert_eq!(t.tex_coord(), 0);
            mat.textures[0] = upload(t.texture(), true);
        }
        if let Some(t) = p.metallic_roughness_texture() {
            assert_eq!(t.tex_coord(), 0);
            mat.textures[1] = upload(t.texture(), false);
        }
        if let Some(t) = m.normal_texture() {
            assert_eq!(t.tex_coord(), 0);
            mat.textures[2] = upload(t.texture(), false);
        }
        if let Some(t) = m.emissive_texture() {
            assert_eq!(t.tex_coord(), 0);
            mat.textures[3] = upload(t.texture(), true);
        }
        materials.push(mat);
    }
    materials.push(PbrMaterial::default());
    colors.push(Vec3::splat(0.7));
    let mut world = TriangleWorld::new(
        &geometry.positions,
        &geometry.indices,
        &geometry.offsets,
        &geometry.materials,
        &colors,
    );
    println!(
        "PICA: {} triangles, {} meshes, {} materials, {} textures, {} omitted blend primitives",
        geometry.indices.len(),
        geometry.offsets.len(),
        materials.len() - 1,
        textures.len(),
        geometry.skipped
    );
    world.set_pbr(
        &materials,
        &geometry.normals,
        &geometry.uv,
        &geometry.density,
        textures,
    );
    let envmap = Rgb32FImage::from_fn(256, 128, |_, y| {
        let t = (y as f32 / 127. * std::f32::consts::PI).cos().max(0.);
        Rgb([0.25 + 0.55 * t, 0.28 + 0.65 * t, 0.34 + 0.8 * t])
    });
    WorldData {
        world,
        envmap,
        envmap_scale: 0.6,
        sun_dir: Vec3::new(-0.4, -0.5, 0.77),
        sun_color: Vec3::new(3., 2.82, 2.46),
        custom_emitters: vec![],
    }
}
pub fn camera(index: usize, size: [u32; 2]) -> yesod::camera::Camera {
    let (pos, target) = match index {
        3 => (
            glam::Vec3::new(22., -31., 15.),
            glam::Vec3::new(6., -11., 4.5),
        ),
        1 => (
            glam::Vec3::new(36., -42., 26.),
            glam::Vec3::new(-4., 8., 7.),
        ),
        2 => (
            glam::Vec3::new(-36., -30., 20.),
            glam::Vec3::new(4., 8., 7.),
        ),
        _ => (
            glam::Vec3::new(52., -65., 55.),
            glam::Vec3::new(-4., 7., 5.),
        ),
    };
    let dir = (target - pos).normalize();
    yesod::camera::Camera {
        pos,
        yaw: dir.x.atan2(dir.y),
        pitch: (-dir.z).asin(),
        screen_size: glam::Vec2::new(size[0] as f32, size[1] as f32),
        fov: if index == 0 { 0.68 } else { 0.85 },
    }
}
