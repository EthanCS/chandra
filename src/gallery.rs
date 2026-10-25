//! Asset-free Chromatic Gallery: 30 local lights and three groups of 30 movers.
//! All dimensions are centimetres. No model, texture, HDR or random seed is required.
use glam::{Mat4, Vec3};
use image::{Rgb, Rgb32FImage};
use keter::graph::NodeConfigs;
use keter::lang::types::vector::{Vec3 as GpuVec3, Vec4 as GpuVec4};

use crate::algorithm::{CommandList, PointLight, WorldData};
use crate::dynamic::{Bounds, RigidBatch, SkinBinding, SkinWeights};
use crate::world::TriangleWorld;

pub const COUNT: usize = 30;
const JOINTS: usize = 5;
const SEGMENT: f32 = 38.;

pub fn station(i: usize) -> Vec3 {
    Vec3::new(
        (i % 6) as f32 * 330. - 825.,
        (i / 6) as f32 * 350. - 700.,
        0.,
    )
}

pub fn color(i: usize) -> Vec3 {
    let h = (i as f32 * 0.618_034).fract() * 6.;
    let f = h.fract();
    let c = match h as u32 {
        0 => Vec3::new(1., f, 0.),
        1 => Vec3::new(1. - f, 1., 0.),
        2 => Vec3::new(0., 1., f),
        3 => Vec3::new(0., 1. - f, 1.),
        4 => Vec3::new(f, 0., 1.),
        _ => Vec3::new(1., 0., 1. - f),
    };
    c * 0.65 + Vec3::splat(0.08)
}

pub fn lights(time: f32, strength: f32) -> Vec<PointLight> {
    (0..COUNT)
        .map(|i| {
            let phase = i as f32 * 2.399_963;
            let p = station(i)
                + Vec3::new(
                    110. * (time * 0.61 + phase).cos(),
                    95. * (time * 0.47 + phase).sin(),
                    230. + 50. * (time * 0.71 + phase).sin(),
                );
            let intensity =
                color(i + 7) * (65_000. * strength * (0.8 + 0.2 * (time * 0.9 + phase).sin()));
            PointLight {
                position_range: GpuVec4::from(p.extend(380.)),
                intensity: GpuVec4::from(intensity.extend(0.)),
            }
        })
        .collect()
}

#[derive(Default)]
struct Builder {
    positions: Vec<[f32; 3]>,
    indices: Vec<[u32; 3]>,
    offsets: Vec<[u32; 2]>,
    materials: Vec<u32>,
    colors: Vec<GpuVec3<f32>>,
    emission: Vec<GpuVec3<f32>>,
}

impl Builder {
    fn mesh(&mut self, vertices: Vec<[f32; 3]>, triangles: Vec<[u32; 3]>, color: Vec3) -> usize {
        let index = self.offsets.len();
        self.offsets
            .push([self.indices.len() as u32, self.positions.len() as u32]);
        self.materials.push(index as u32);
        self.colors.push(color.into());
        self.emission.push(GpuVec3::splat(0.));
        self.positions.extend(vertices);
        self.indices.extend(triangles);
        index
    }
    fn block(&mut self, min: Vec3, max: Vec3, color: Vec3) -> usize {
        let p = (0..8)
            .map(|i| {
                [
                    if i & 1 == 0 { min.x } else { max.x },
                    if i & 2 == 0 { min.y } else { max.y },
                    if i & 4 == 0 { min.z } else { max.z },
                ]
            })
            .collect();
        self.mesh(
            p,
            vec![
                [0, 2, 3],
                [0, 3, 1],
                [4, 5, 7],
                [4, 7, 6],
                [0, 1, 5],
                [0, 5, 4],
                [2, 6, 7],
                [2, 7, 3],
                [0, 4, 6],
                [0, 6, 2],
                [1, 3, 7],
                [1, 7, 5],
            ],
            color,
        )
    }
    fn column(&mut self, center: Vec3, height: f32, color: Vec3) {
        let profile = [
            (0., 38.),
            (12., 38.),
            (18., 27.),
            (34., 25.),
            (height - 34., 25.),
            (height - 18., 30.),
            (height - 12., 42.),
            (height, 42.),
        ];
        let mut p = Vec::new();
        let mut t = Vec::new();
        for (z, r) in profile {
            for s in 0..32 {
                let a = s as f32 * std::f32::consts::TAU / 32.;
                p.push((center + Vec3::new(r * a.cos(), r * a.sin(), z)).into());
            }
        }
        for j in 0..7u32 {
            for s in 0..32u32 {
                let a = j * 32 + s;
                let b = j * 32 + (s + 1) % 32;
                t.extend([[a, b, a + 32], [b, b + 32, a + 32]]);
            }
        }
        for s in 1..31 {
            t.push([0, s + 1, s]);
            t.push([224, 224 + s, 225 + s]);
        }
        self.mesh(p, t, color);
    }
    fn arch(&mut self, center: Vec3, turn: f32, color: Vec3) {
        let transform = Mat4::from_translation(center) * Mat4::from_rotation_z(turn);
        for s in 0..20 {
            let mut p = Vec::new();
            for depth in [-22., 22.] {
                for radius in [145., 180.] {
                    for k in [0, 1] {
                        let a = (s + k) as f32 * std::f32::consts::PI / 20.;
                        p.push(
                            transform
                                .transform_point3(Vec3::new(
                                    radius * a.cos(),
                                    depth,
                                    radius * a.sin(),
                                ))
                                .into(),
                        );
                    }
                }
            }
            let tone = color * if s % 2 == 0 { 0.96 } else { 1.04 };
            self.mesh(
                p,
                vec![
                    [0, 1, 3],
                    [0, 3, 2],
                    [4, 6, 7],
                    [4, 7, 5],
                    [0, 4, 5],
                    [0, 5, 1],
                    [2, 3, 7],
                    [2, 7, 6],
                    [0, 2, 6],
                    [0, 6, 4],
                    [1, 5, 7],
                    [1, 7, 3],
                ],
                tone,
            );
        }
    }
    fn lathe(
        &mut self,
        center: Vec3,
        radius: f32,
        height: f32,
        color: Vec3,
        rings: u32,
        slices: u32,
    ) -> usize {
        let mut p = Vec::new();
        let mut t = Vec::new();
        for j in 0..=rings {
            let a = j as f32 * std::f32::consts::PI / rings as f32;
            for i in 0..=slices {
                let b = i as f32 * std::f32::consts::TAU / slices as f32;
                p.push(
                    (center
                        + Vec3::new(
                            radius * a.sin() * b.cos(),
                            radius * a.sin() * b.sin(),
                            height * a.cos(),
                        ))
                    .into(),
                );
            }
        }
        for j in 0..rings {
            for i in 0..slices {
                let a = j * (slices + 1) + i;
                let b = a + slices + 1;
                if j > 0 {
                    t.push([a, b, a + 1]);
                }
                if j + 1 < rings {
                    t.push([a + 1, b, b + 1]);
                }
            }
        }
        self.mesh(p, t, color)
    }
}

struct SkinSpec {
    mesh: usize,
    root: Vec3,
    weights: Vec<SkinWeights>,
}
struct Layout {
    mesh: Builder,
    rigids: Vec<usize>,
    centers: Vec<Vec3>,
    skins: Vec<SkinSpec>,
}

fn layout() -> Layout {
    let mut b = Builder::default();
    let stone = Vec3::new(0.62, 0.64, 0.66);
    let pale = Vec3::new(0.76, 0.72, 0.64);
    b.block(
        Vec3::new(-1250., -1200., -30.),
        Vec3::new(1250., 1200., -3.),
        stone,
    );
    for y in 0..20 {
        for x in 0..20 {
            let p = Vec3::new(-1200. + x as f32 * 120., -1150. + y as f32 * 115., -3.);
            let tone = if (x + y) % 2 == 0 {
                Vec3::new(0.48, 0.51, 0.53)
            } else {
                Vec3::new(0.62, 0.60, 0.55)
            };
            b.block(p, p + Vec3::new(118., 113., 3.), tone);
        }
    }
    // Enclosing walls, raised side galleries, colonnades and masonry arches.
    for side in [-1., 1.] {
        let x = side * 1170.;
        b.block(
            Vec3::new(x - 18., -1060., 0.),
            Vec3::new(x + 18., 1130., 790.),
            stone * 0.82,
        );
        b.block(
            Vec3::new(x - 130., -900., 440.),
            Vec3::new(x + 130., 1100., 470.),
            pale,
        );
        for row in 0..7 {
            let y = -900. + row as f32 * 330.;
            b.column(Vec3::new(side * 1000., y, 0.), 420., pale);
            b.column(Vec3::new(side * 1000., y, 470.), 330., pale * 0.9);
            b.block(
                Vec3::new(side * 1000. - 32., y - 25., 795.),
                Vec3::new(side * 1000. + 32., y + 25., 835.),
                pale,
            );
            if row < 6 {
                b.arch(
                    Vec3::new(side * 1000., y + 165., 420.),
                    std::f32::consts::FRAC_PI_2,
                    pale,
                );
                b.block(
                    Vec3::new(x - 110., y, 525.),
                    Vec3::new(x + 110., y + 12., 545.),
                    stone,
                );
            }
        }
        // Two staircases ascend to the galleries; each step has its own occlusion.
        for step in 0..18 {
            let z = (step + 1) as f32 * 440. / 18.;
            let y = -1200. + step as f32 * 24.;
            b.block(
                Vec3::new(x - 105., y, 0.),
                Vec3::new(x + 105., y + 24., z),
                pale,
            );
        }
    }
    b.block(
        Vec3::new(-1200., 1120., 0.),
        Vec3::new(1200., 1150., 830.),
        stone * 0.9,
    );
    for col in 0..7 {
        let x = -990. + col as f32 * 330.;
        b.column(Vec3::new(x, 1000., 0.), 420., pale);
        if col < 6 {
            b.arch(Vec3::new(x + 165., 1000., 420.), 0., pale);
        }
    }
    // Roof beams and slats over the perimeter leave a broad central skylight.
    for y in 0..16 {
        let y = -1000. + y as f32 * 140.;
        for side in [-1., 1.] {
            let x = side * 880.;
            b.block(
                Vec3::new(x - 270., y, 840.),
                Vec3::new(x + 270., y + 24., 865.),
                Vec3::new(0.30, 0.25, 0.19),
            );
        }
    }
    for x in [-990., -660., -330., 0., 330., 660., 990.] {
        b.block(
            Vec3::new(x - 12., 850., 840.),
            Vec3::new(x + 12., 1150., 865.),
            pale,
        );
    }
    // Recessed color-bounce niches and plinths along both side aisles.
    for i in 0..12 {
        let side = if i % 2 == 0 { -1. } else { 1. };
        let y = -730. + (i / 2) as f32 * 300.;
        let p = Vec3::new(side * 1110., y, 70.);
        b.block(
            p - Vec3::new(45., 55., 70.),
            p + Vec3::new(45., 55., 0.),
            stone,
        );
        b.lathe(p + Vec3::Z * 75., 32., 75., color(i) * 0.65, 20, 32);
    }
    let mut rigids = Vec::new();
    let mut centers = Vec::new();
    for i in 0..COUNT {
        let center = station(i) + Vec3::new(-75., -30., 55.);
        let size = Vec3::new(34. + (i % 3) as f32 * 5., 32., 50.);
        rigids.push(b.block(center - size, center + size, color(i)));
        centers.push(center);
    }
    for i in 0..COUNT {
        let center = station(i) + Vec3::new(75., 0., 150.);
        let mesh = b.lathe(center, 25., 36., color(i + 3), 12, 20);
        b.emission[mesh] = GpuVec3::from(color(i + 3));
        rigids.push(mesh);
        centers.push(center);
    }
    let mut skins = Vec::new();
    for i in 0..COUNT {
        let root = station(i) + Vec3::new(0., 100., 15.);
        let (rings, slices) = (32u32, 16u32);
        let mut p = Vec::new();
        let mut t = Vec::new();
        let mut weights = Vec::new();
        for j in 0..=rings {
            let fraction = j as f32 / rings as f32;
            let joint = fraction * (JOINTS - 1) as f32;
            let lower = (joint.floor() as u32).min(JOINTS as u32 - 2);
            let blend = joint - lower as f32;
            let radius = 23. * (1. - 0.45 * fraction) * (1. + 0.12 * (fraction * 18.).sin());
            for s in 0..=slices {
                let a = s as f32 * std::f32::consts::TAU / slices as f32;
                p.push(
                    (root + Vec3::new(radius * a.cos(), radius * a.sin(), joint * SEGMENT)).into(),
                );
                weights.push(SkinWeights {
                    joints: [lower, lower + 1, 0, 0],
                    weights: [1. - blend, blend, 0., 0.],
                });
            }
        }
        for j in 0..rings {
            for s in 0..slices {
                let a = j * (slices + 1) + s;
                let c = a + slices + 1;
                t.extend([[a, a + 1, c], [a + 1, c + 1, c]]);
            }
        }
        for (ring, flip) in [(0, true), (rings, false)] {
            let c = p.len() as u32;
            p.push(
                (root + Vec3::Z * (ring as f32 / rings as f32) * (JOINTS - 1) as f32 * SEGMENT)
                    .into(),
            );
            weights.push(SkinWeights {
                joints: [if flip { 0 } else { JOINTS as u32 - 1 }; 4],
                weights: [1., 0., 0., 0.],
            });
            for s in 0..slices {
                let a = ring * (slices + 1) + s;
                t.push(if flip { [c, a + 1, a] } else { [c, a, a + 1] });
            }
        }
        let mesh = b.mesh(p, t, color(i + 17) * 0.8);
        skins.push(SkinSpec {
            mesh,
            root,
            weights,
        });
    }
    Layout {
        mesh: b,
        rigids,
        centers,
        skins,
    }
}

pub struct GalleryAnimation {
    rigid: RigidBatch,
    centers: Vec<Vec3>,
    skins: Vec<(SkinBinding, Vec3)>,
    previous: Vec<Bounds>,
    rigid_time: f32,
    skin_time: f32,
}
pub struct GalleryUpdate {
    pub vertices: CommandList<'static>,
    pub blas: NodeConfigs<'static>,
    pub tlas: NodeConfigs<'static>,
    pub swept: Vec<Bounds>,
}

pub fn create() -> (WorldData, GalleryAnimation) {
    let Layout {
        mesh: b,
        rigids,
        centers,
        skins,
    } = layout();
    let dynamic: Vec<_> = rigids
        .iter()
        .copied()
        .chain(skins.iter().map(|s| s.mesh))
        .collect();
    println!(
        "Chromatic Gallery: {} static meshes, {} triangles, {} vertices; 30 point lights / 30 rigid / 30 emissive / 30 skinned; no external assets",
        b.offsets.len() - dynamic.len(),
        b.indices.len(),
        b.positions.len()
    );
    let mut world = TriangleWorld::new_dynamic(
        &b.positions,
        &b.indices,
        &b.offsets,
        &b.materials,
        &b.colors,
        &dynamic,
    );
    world.set_material_emission(&b.emission);
    let rigid = RigidBatch::new(&mut world, &rigids).unwrap();
    let skins: Vec<_> = skins
        .into_iter()
        .map(|s| {
            let inverse: Vec<_> = (0..JOINTS)
                .map(|j| Mat4::from_translation(-s.root - Vec3::Z * (j as f32 * SEGMENT)))
                .collect();
            (
                SkinBinding::new(&world, s.mesh, &s.weights, &inverse).unwrap(),
                s.root,
            )
        })
        .collect();
    let previous = rigid
        .bind_bounds()
        .iter()
        .copied()
        .chain(skins.iter().map(|s| s.0.bounds()))
        .collect();
    (
        WorldData {
            world,
            envmap: Rgb32FImage::from_pixel(1, 1, Rgb([0.65, 0.78, 1.])),
            envmap_scale: 0.12,
            sun_dir: GpuVec3::new(-0.4, -0.6, 0.7),
            sun_color: GpuVec3::splat(1.8),
            custom_emitters: vec![],
        },
        GalleryAnimation {
            rigid,
            centers,
            skins,
            previous,
            rigid_time: 0.,
            skin_time: 0.,
        },
    )
}

impl GalleryAnimation {
    pub fn skin_vertices(&self) -> u32 {
        self.skins.iter().map(|s| s.0.vertex_count()).sum()
    }
    pub fn update(
        &mut self,
        world: &TriangleWorld,
        dt: f32,
        rigid: bool,
        skin: bool,
        frame: u32,
    ) -> GalleryUpdate {
        let mut vertices = CommandList::new();
        let mut swept = Vec::new();
        let mut changed = Vec::new();
        if rigid {
            self.rigid_time += dt;
            let t = self.rigid_time;
            let transforms: Vec<_> = self
                .centers
                .iter()
                .enumerate()
                .map(|(i, &center)| {
                    let phase = i as f32 * 2.399_963;
                    let shift = if i < COUNT {
                        Vec3::new(
                            42. * (t * 0.65 + phase).sin(),
                            32. * (t * 0.47 + phase).cos(),
                            10. * (1. + (t + phase).sin()),
                        )
                    } else {
                        Vec3::new(
                            38. * (t * 0.8 + phase).cos(),
                            50. * (t * 0.63 + phase).sin(),
                            48. * (t * 0.55 + phase).sin(),
                        )
                    };
                    Mat4::from_translation(center + shift)
                        * Mat4::from_rotation_z(t * (0.3 + (i % 5) as f32 * 0.08) + phase)
                        * Mat4::from_translation(-center)
                })
                .collect();
            let (update, bounds) = self.rigid.update(world, &transforms).unwrap();
            vertices += update;
            changed.extend_from_slice(self.rigid.meshes());
            for (i, b) in bounds.into_iter().enumerate() {
                swept.push(self.previous[i].union(b));
                self.previous[i] = b;
            }
        }
        if skin {
            self.skin_time += dt;
            let t = self.skin_time;
            for (i, (binding, root)) in self.skins.iter().enumerate() {
                let phase = i as f32 * 1.73;
                let shift = Vec3::new(
                    28. * (t * 0.55 + phase).sin(),
                    24. * (t * 0.7 + phase).cos(),
                    0.,
                );
                let mut joint = Mat4::from_translation(*root + shift);
                let mut palette = Vec::new();
                for j in 0..JOINTS {
                    if j > 0 {
                        joint *= Mat4::from_translation(Vec3::Z * SEGMENT);
                    }
                    joint *=
                        Mat4::from_rotation_y(0.23 * (t * 1.35 + phase + j as f32 * 0.7).sin())
                            * Mat4::from_rotation_x(
                                0.16 * (t * 1.1 + phase + j as f32 * 0.4).cos(),
                            );
                    palette.push(joint);
                }
                let (update, b) = binding.update(&palette).unwrap();
                vertices += update;
                let k = COUNT * 2 + i;
                swept.push(self.previous[k].union(b));
                self.previous[k] = b;
                changed.push(binding.mesh);
            }
        }
        GalleryUpdate {
            vertices,
            blas: world.update_bottom_level(&changed, frame.is_multiple_of(180)),
            tlas: world.update_top_level(!changed.is_empty(), frame.is_multiple_of(180)),
            swept,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_ninety_gallery_meshes_move_and_stay_inside_history_bounds() {
        let (data, mut animation) = create();
        let world = data.world;
        let original = world.positions.copy_to_vec();
        let update = animation.update(&world, 0.5, true, true, 1);
        assert_eq!(update.swept.len(), 90);
        let bounds = update.swept;
        let mut commands = update.vertices;
        commands += update.blas;
        commands += update.tlas;
        commands.execute();
        let current = world.positions.copy_to_vec();
        let indices: Vec<_> = animation
            .rigid
            .meshes()
            .iter()
            .copied()
            .chain(animation.skins.iter().map(|s| s.0.mesh))
            .collect();
        for (slot, index) in indices.into_iter().enumerate() {
            let range = world.mesh_vertex_ranges[index].clone();
            assert!(
                range
                    .clone()
                    .any(|i| Vec3::from(original[i]).distance(current[i].into()) > 0.1)
            );
            for i in range {
                assert!(bounds[slot].expanded(0.01).contains(current[i].into()));
            }
        }
    }
    #[test]
    fn gallery_has_exact_groups_valid_topology_and_animated_lights() {
        let layout = layout();
        assert_eq!(layout.rigids.len(), 60);
        assert_eq!(layout.skins.len(), 30);
        assert_eq!(
            layout
                .mesh
                .emission
                .iter()
                .filter(|c| c.x + c.y + c.z > 0.)
                .count(),
            30
        );
        assert!(layout.mesh.offsets.len() > 800);
        for (i, offset) in layout.mesh.offsets.iter().enumerate() {
            let end = layout.mesh.offsets.get(i + 1).copied().unwrap_or([
                layout.mesh.indices.len() as u32,
                layout.mesh.positions.len() as u32,
            ]);
            for tri in &layout.mesh.indices[offset[0] as usize..end[0] as usize] {
                assert!(tri.iter().all(|v| *v < end[1] - offset[1]));
                let p = tri.map(|v| Vec3::from(layout.mesh.positions[(offset[1] + v) as usize]));
                assert!((p[1] - p[0]).cross(p[2] - p[0]).length_squared() > 1e-8);
            }
        }
        let a = lights(0., 1.);
        let b = lights(1., 1.);
        assert_eq!(a.len(), 30);
        for (a, b) in a.iter().zip(&b) {
            assert_ne!(a.position_range, b.position_range);
        }
    }
}
