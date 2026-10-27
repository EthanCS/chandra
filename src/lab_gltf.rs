//! Local Intel Sponza import for the diffuse Chandra renderer.
//! Source files stay in Albor; transparent dirt decals are explicitly excluded.
use std::collections::HashMap;
use std::path::Path;

use chandra::world::TriangleWorld;
use glam::{Mat4, Vec2, Vec3};
use keter::lang::types::vector::{Vec3 as GpuVec3, Vec4};
use keter::prelude::*;

#[derive(Default)]
pub(crate) struct Geometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
    pub density: Vec<f32>,
    pub offsets: Vec<[u32; 2]>,
    pub materials: Vec<u32>,
    pub skipped: usize,
    pub include_mask: bool,
}

impl Geometry {
    pub fn visit(
        &mut self,
        node: gltf::Node<'_>,
        parent: Mat4,
        buffers: &[Vec<u8>],
        default_material: u32,
    ) {
        let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                if primitive.material().alpha_mode() != gltf::material::AlphaMode::Opaque
                    && !(self.include_mask && primitive.material().alpha_mode()==gltf::material::AlphaMode::Mask) {
                    self.skipped += 1;
                    continue;
                }
                assert_eq!(
                    primitive.mode(),
                    gltf::mesh::Mode::Triangles,
                    "Only triangle primitives are supported"
                );
                let reader = primitive.reader(|buffer| Some(buffers[buffer.index()].as_slice()));
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .expect("Missing positions")
                    .map(|p| transform.transform_point3(Vec3::from(p)).to_array())
                    .collect();
                let uv: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|uv| uv.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.; 2]; positions.len()]);
                let normal_transform=transform.inverse().transpose();
                let mut normals:Vec<[f32;3]>=reader.read_normals().map(|values|values.map(|n|
                    normal_transform.transform_vector3(Vec3::from(n)).normalize_or_zero().to_array()).collect())
                    .unwrap_or_else(||vec![[0.;3];positions.len()]);
                let missing_normals=reader.read_normals().is_none();
                let indices: Vec<u32> = reader
                    .read_indices()
                    .map(|v| v.into_u32().collect())
                    .unwrap_or_else(|| (0..positions.len() as u32).collect());
                assert_eq!(indices.len() % 3, 0);
                assert_eq!(uv.len(), positions.len());
                self.offsets
                    .push([self.indices.len() as u32, self.positions.len() as u32]);
                self.materials.push(
                    primitive
                        .material()
                        .index()
                        .map_or(default_material, |i| i as u32),
                );
                for t in indices.chunks_exact(3) {
                    let mut t = [t[0], t[1], t[2]];
                    if transform.determinant() < 0. {
                        t.swap(1, 2);
                    }
                    let p = t.map(|i| Vec3::from(positions[i as usize]));
                    let u = t.map(|i| Vec2::from(uv[i as usize]));
                    let area = (p[1] - p[0]).cross(p[2] - p[0]).length();
                    if missing_normals {
                        let n=(p[1]-p[0]).cross(p[2]-p[0]).normalize_or_zero().to_array();
                        for i in t {normals[i as usize]=n;}
                    }
                    let uv_area = (u[1] - u[0]).perp_dot(u[2] - u[0]).abs();
                    self.density.push(if area > 1e-12 {
                        (uv_area / area).sqrt()
                    } else {
                        0.
                    });
                    self.indices.push(t);
                }
                self.positions.extend(positions);
                self.normals.extend(normals);
                self.uv.extend(uv);
            }
        }
        for child in node.children() {
            self.visit(child, transform, buffers, default_material);
        }
    }

    fn cube(&mut self, material: u32, center: Vec3, edge: f32) -> usize {
        let index = self.offsets.len();
        self.offsets
            .push([self.indices.len() as u32, self.positions.len() as u32]);
        self.materials.push(material);
        let h = edge * 0.5;
        self.positions.extend(
            [
                [-h, -h, 5.],
                [h, -h, 5.],
                [h, h, 5.],
                [-h, h, 5.],
                [-h, -h, 5. + edge],
                [h, -h, 5. + edge],
                [h, h, 5. + edge],
                [-h, h, 5. + edge],
            ]
            .map(|p| (Vec3::from(p) + center).to_array()),
        );
        self.uv.extend([[0.; 2]; 8]);
        self.indices.extend([
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
        self.density.extend([0.; 12]);
        index
    }
}

/// 90 cm spacing leaves clearance for a rotating 45 cm cube and 12 cm motion.
pub fn cube_center(index: usize, count: usize) -> Vec3 {
    if count == 1 {
        return Vec3::ZERO;
    }
    let columns = count.min(20);
    let rows = count.div_ceil(columns);
    Vec3::new(
        (index % columns) as f32 - (columns - 1) as f32 * 0.5,
        (index / columns) as f32 - (rows - 1) as f32 * 0.5,
        0.,
    ) * 90.
}

fn cube_color(index: usize, count: usize) -> GpuVec3<f32> {
    if count == 1 {
        return GpuVec3::new(0.75, 0.018, 0.012);
    }
    // Golden-angle hue ordering distributes neighbouring hues across the grid.
    let hue = (index as f32 * 0.618_034).fract() * 6.;
    let x = 1. - (hue % 2. - 1.).abs();
    let c = match hue as u32 {
        0 => [1., x, 0.],
        1 => [x, 1., 0.],
        2 => [0., 1., x],
        3 => [0., x, 1.],
        4 => [x, 0., 1.],
        _ => [1., 0., x],
    };
    GpuVec3::new(
        c[0] * 0.70 + 0.035,
        c[1] * 0.70 + 0.035,
        c[2] * 0.70 + 0.035,
    )
}

pub fn load(path: &Path, cube_count: usize, emissive_cubes: bool) -> (TriangleWorld, Vec<usize>) {
    let start = std::time::Instant::now();
    println!("Loading Intel Sponza: {}", path.display());
    let gltf = gltf::Gltf::open(path).expect("Cannot read Intel Sponza glTF");
    let parent = path.parent().unwrap();
    let buffers: Vec<Vec<u8>> = gltf
        .buffers()
        .map(|buffer| match buffer.source() {
            gltf::buffer::Source::Uri(uri) => {
                std::fs::read(parent.join(uri)).expect("Cannot read glTF buffer")
            }
            gltf::buffer::Source::Bin => gltf.blob.clone().expect("Missing GLB buffer"),
        })
        .collect();
    let mut geometry = Geometry::default();
    // Source is Y-up, metres. Chandra uses Z-up, centimetres.
    let basis = Mat4::from_cols(
        glam::Vec4::new(100., 0., 0., 0.),
        glam::Vec4::new(0., 0., 100., 0.),
        glam::Vec4::new(0., -100., 0., 0.),
        glam::Vec4::W,
    );
    let default_material = gltf.materials().len() as u32;
    for node in gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .expect("Missing scene")
        .nodes()
    {
        geometry.visit(node, basis, &buffers, default_material);
    }
    let mut colors: Vec<GpuVec3<f32>> = gltf
        .materials()
        .map(|m| {
            let c = m.pbr_metallic_roughness().base_color_factor();
            GpuVec3::new(c[0], c[1], c[2])
        })
        .collect();
    colors.push(GpuVec3::splat(0.7));
    let cube_material_start = colors.len();
    let source_primitives = geometry.offsets.len();
    let mut rigid = Vec::with_capacity(cube_count);
    for i in 0..cube_count {
        let material = colors.len() as u32;
        colors.push(cube_color(i, cube_count));
        rigid.push(geometry.cube(
            material,
            cube_center(i, cube_count),
            if cube_count == 1 { 150. } else { 45. },
        ));
    }

    let decode = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>, Tex2d<Vec4<f32>>)>(&track!(
        |source, target| {
            let p = dispatch_id().xy();
            let srgb = source.read(p);
            let channel = |v: Expr<f32>| {
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            target.write(
                p,
                Vec4::expr(channel(srgb.x), channel(srgb.y), channel(srgb.z), srgb.w),
            );
        }
    ));
    let mip = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>, Tex2d<Vec4<f32>>)>(&track!(
        |source, target| {
            let p = dispatch_id().xy();
            let q = p * 2u32;
            let limit = source.size() - 1u32;
            target.write(
                p,
                (source.read(keter::min(q, limit))
                    + source.read(keter::min(
                        q + keter::lang::types::vector::Vec2::expr(1u32, 0u32),
                        limit,
                    ))
                    + source.read(keter::min(
                        q + keter::lang::types::vector::Vec2::expr(0u32, 1u32),
                        limit,
                    ))
                    + source.read(keter::min(q + 1u32, limit)))
                    * 0.25,
            );
        }
    ));
    let mut image_indices = HashMap::new();
    let mut textures = Vec::new();
    let mut material_texture = vec![u32::MAX; colors.len()];
    for material in gltf.materials() {
        if material.alpha_mode() != gltf::material::AlphaMode::Opaque {
            continue;
        }
        if let Some(info) = material.pbr_metallic_roughness().base_color_texture() {
            assert_eq!(info.tex_coord(), 0, "Only TEXCOORD_0 is supported");
            let source = info.texture().source();
            let image_index = source.index();
            let texture_index = *image_indices.entry(image_index).or_insert_with(|| {
                let image = match source.source() {
                    gltf::image::Source::Uri { uri, .. } => {
                        image::open(parent.join(uri)).expect("Cannot read base-color image")
                    }
                    gltf::image::Source::View { view, .. } => image::load_from_memory(
                        &buffers[view.buffer().index()]
                            [view.offset()..view.offset() + view.length()],
                    )
                    .unwrap(),
                }
                .to_rgba8();
                let (w, h) = image.dimensions();
                let input = DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Byte4, w, h, 1);
                input
                    .view(0)
                    .copy_from(bytemuck::cast_slice::<u8, [u8; 4]>(image.as_raw()));
                let levels = w.max(h).ilog2() + 1;
                let texture = DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Half4, w, h, levels);
                decode.dispatch([w, h, 1], &input, &texture.view(0));
                for level in 1..levels {
                    mip.dispatch(
                        [(w >> level).max(1), (h >> level).max(1), 1],
                        &texture.view(level - 1),
                        &texture.view(level),
                    );
                }
                println!(
                    "Base color {}: {}x{}, {} mip levels",
                    image_index, w, h, levels
                );
                let index = textures.len() as u32;
                textures.push(texture);
                index
            });
            material_texture[material.index().unwrap()] = texture_index;
        }
    }
    println!(
        "Intel Sponza: {} source primitives + {} independently animated cubes, {} triangles, {} vertices, {} base-color textures; {} transparent decal primitives excluded",
        source_primitives,
        cube_count,
        geometry.indices.len(),
        geometry.positions.len(),
        textures.len(),
        geometry.skipped
    );
    let mut world = TriangleWorld::new_dynamic(
        &geometry.positions,
        &geometry.indices,
        &geometry.offsets,
        &geometry.materials,
        &colors,
        &rigid,
    );
    world.set_diffuse_textures(&geometry.uv, &geometry.density, &material_texture, textures);
    if emissive_cubes {
        let emission: Vec<_> = colors
            .iter()
            .enumerate()
            .map(|(i, color)| {
                if i >= cube_material_start {
                    *color
                } else {
                    GpuVec3::splat(0.0)
                }
            })
            .collect();
        world.set_material_emission(&emission);
        println!(
            "Emissive materials: {cube_count} colored cube emitters; constant-time material lookup"
        );
    }
    println!(
        "Intel Sponza loaded in {:.2}s",
        start.elapsed().as_secs_f64()
    );
    (world, rigid)
}
