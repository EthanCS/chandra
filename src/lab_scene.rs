//! Small, asset-free scene for inspecting shadows and diffuse interreflection.
use chandra::algorithm::{CommandList, WorldData};
use chandra::dynamic::{Bounds, RigidBatch, RigidBinding, SkinBinding, SkinWeights};
use chandra::world::TriangleWorld;
use glam::{Mat4, Vec3};
use image::{Rgb, Rgb32FImage};
use keter::lang::types::vector::Vec3 as GpuVec3;

#[derive(Default)]
struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    indices: Vec<[u32; 3]>,
    offsets: Vec<[u32; 2]>,
    materials: Vec<u32>,
    colors: Vec<GpuVec3<f32>>,
}

impl MeshBuilder {
    fn mesh(&mut self, vertices: Vec<[f32; 3]>, triangles: Vec<[u32; 3]>, color: [f32; 3]) {
        self.offsets
            .push([self.indices.len() as u32, self.positions.len() as u32]);
        self.materials.push(self.colors.len() as u32);
        self.colors.push(color.into());
        self.positions.extend(vertices);
        self.indices.extend(triangles);
    }

    fn block(&mut self, min: [f32; 3], max: [f32; 3], color: [f32; 3]) {
        let [x, y, z] = min;
        let [a, b, c] = max;
        self.mesh(
            vec![
                [x, y, z],
                [a, y, z],
                [a, b, z],
                [x, b, z],
                [x, y, c],
                [a, y, c],
                [a, b, c],
                [x, b, c],
            ],
            vec![
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
            ],
            color,
        );
    }

    fn sphere(&mut self, center: Vec3, radius: f32, color: [f32; 3]) {
        let (rings, slices) = (32, 64);
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for j in 0..=rings {
            let theta = std::f32::consts::PI * j as f32 / rings as f32;
            for i in 0..=slices {
                let phi = std::f32::consts::TAU * i as f32 / slices as f32;
                vertices.push(
                    (center
                        + radius
                            * Vec3::new(
                                theta.sin() * phi.cos(),
                                theta.sin() * phi.sin(),
                                theta.cos(),
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
                    triangles.push([a, b, a + 1]);
                }
                if j < rings - 1 {
                    triangles.push([a + 1, b, b + 1]);
                }
            }
        }
        self.mesh(vertices, triangles, color);
    }

    fn finish(self, dynamic_meshes: &[usize]) -> TriangleWorld {
        TriangleWorld::new_dynamic(
            &self.positions,
            &self.indices,
            &self.offsets,
            &self.materials,
            &self.colors,
            dynamic_meshes,
        )
    }
}

pub struct LabScene {
    pub world: WorldData,
    pub animation: Option<Animation>,
}

pub struct Animation {
    gallery: Option<chandra::gallery::GalleryAnimation>,
    rigid: Vec<RigidBinding>,
    rigid_instances: Option<RigidBatch>,
    skin: Option<SkinBinding>,
    intel_courtyard: bool,
    previous: Vec<Bounds>,
    pub rigid_time: f32,
    pub skin_time: f32,
}

pub struct AnimationUpdates {
    pub vertices: CommandList<'static>,
    pub blas: keter::graph::NodeConfigs<'static>,
    pub tlas: keter::graph::NodeConfigs<'static>,
    pub swept: Vec<Bounds>,
}

impl Animation {
    pub fn update(
        &mut self,
        world: &TriangleWorld,
        dt: f32,
        rigid: bool,
        skin: bool,
        frame: u32,
    ) -> AnimationUpdates {
        if let Some(gallery) = &mut self.gallery {
            let update = gallery.update(world, dt, rigid, skin, frame);
            return AnimationUpdates {
                vertices: update.vertices,
                blas: update.blas,
                tlas: update.tlas,
                swept: update.swept,
            };
        }
        let mut commands = CommandList::new();
        let mut swept = Vec::new();
        let mut changed = Vec::new();
        if rigid {
            self.rigid_time += dt;
            let t = self.rigid_time;
            let rigid_count = self.rigid_count();
            let mut transforms = Vec::with_capacity(rigid_count);
            for i in 0..rigid_count {
                let pivot = Vec3::new(-87.5, 47.5, 57.5);
                let transform = if self.intel_courtyard && rigid_count > 1 {
                    let pivot = crate::gltf_scene::cube_center(i, rigid_count);
                    let phase = i as f32 * 2.399_963;
                    let shift = Vec3::new(
                        12. * (t * 0.8 + phase).sin(),
                        12. * (t * 0.9 + phase).cos(),
                        0.,
                    );
                    Mat4::from_translation(pivot + shift)
                        * Mat4::from_rotation_z(t * (0.3 + (i % 7) as f32 * 0.03) + phase)
                        * Mat4::from_translation(-pivot)
                } else if self.intel_courtyard {
                    // Remains on the courtyard floor; 11 m round trip, 12 s period.
                    Mat4::from_translation(Vec3::new(
                        550. * (t * std::f32::consts::TAU / 12.).sin(),
                        0.,
                        0.,
                    )) * Mat4::from_rotation_z(0.35 * (t * 0.7).sin())
                } else {
                    Mat4::from_translation(
                        pivot
                            + Vec3::new(
                                80. * (t * 0.7).sin(),
                                45. * (t * 0.55).sin(),
                                50. * (0.5 - 0.5 * t.cos()),
                            ),
                    ) * Mat4::from_rotation_z(t * 0.65)
                        * Mat4::from_translation(-pivot)
                };
                transforms.push(transform);
            }
            let bounds = if let Some(batch) = &self.rigid_instances {
                let (update, bounds) = batch.update(world, &transforms).unwrap();
                commands += update;
                changed.extend_from_slice(batch.meshes());
                bounds
            } else {
                self.rigid
                    .iter()
                    .zip(transforms)
                    .map(|(binding, transform)| {
                        let (update, bounds) = binding.update(transform).unwrap();
                        commands += update;
                        changed.push(binding.mesh);
                        bounds
                    })
                    .collect()
            };
            for (i, bounds) in bounds.into_iter().enumerate() {
                swept.push(self.previous[i].union(bounds));
                self.previous[i] = bounds;
            }
        }
        if skin && let Some(skin) = &self.skin {
            self.skin_time += dt;
            let t = self.skin_time;
            let mut joints = Vec::new();
            let mut joint = Mat4::from_translation(SKIN_ROOT);
            for i in 0..SKIN_JOINTS {
                if i > 0 {
                    joint *= Mat4::from_translation(Vec3::Z * SKIN_SEGMENT);
                }
                joint *= Mat4::from_rotation_y(0.24 * (t * 1.4 + i as f32 * 0.65).sin())
                    * Mat4::from_rotation_x(0.16 * (t * 1.1 + i as f32 * 0.4).cos());
                joints.push(joint);
            }
            let (update, bounds) = skin.update(&joints).unwrap();
            commands += update;
            let index = self.rigid_count();
            swept.push(self.previous[index].union(bounds));
            self.previous[index] = bounds;
            changed.push(skin.mesh);
        }
        AnimationUpdates {
            vertices: commands,
            blas: world.update_bottom_level(&changed, frame.is_multiple_of(180)),
            tlas: world.update_top_level(!changed.is_empty(), frame.is_multiple_of(180)),
            swept,
        }
    }
    pub fn skin_vertices(&self) -> u32 {
        if let Some(gallery) = &self.gallery {
            return gallery.skin_vertices();
        }
        self.skin.as_ref().map_or(0, SkinBinding::vertex_count)
    }
    pub fn moving_meshes(&self) -> usize {
        if self.gallery.is_some() {
            return chandra::gallery::COUNT * 3;
        }
        self.rigid_count() + usize::from(self.skin.is_some())
    }
    fn rigid_count(&self) -> usize {
        self.rigid_instances
            .as_ref()
            .map_or(self.rigid.len(), |batch| batch.meshes().len())
    }
}

const SKIN_ROOT: Vec3 = Vec3::new(-200., -190., 0.);
const SKIN_JOINTS: usize = 6;
const SKIN_SEGMENT: f32 = 44.;

pub fn world(
    sponza: bool,
    intel_path: Option<&std::path::Path>,
    cube_count: usize,
    rigid_refit: bool,
    emissive_cubes: bool,
    gallery: bool,
) -> LabScene {
    if gallery {
        let (world, animation) = chandra::gallery::create();
        return LabScene {
            world,
            animation: Some(Animation {
                gallery: Some(animation),
                rigid: vec![],
                rigid_instances: None,
                skin: None,
                intel_courtyard: false,
                previous: vec![],
                rigid_time: 0.,
                skin_time: 0.,
            }),
        };
    }
    let mut animation = None;
    let geometry = if let Some(path) = intel_path {
        let (mut geometry, rigid_indices) =
            crate::gltf_scene::load(path, cube_count, emissive_cubes);
        let rigid_instances =
            (!rigid_refit).then(|| RigidBatch::new(&mut geometry, &rigid_indices).unwrap());
        let rigid: Vec<_> = rigid_indices
            .into_iter()
            .filter(|_| rigid_refit)
            .map(|i| RigidBinding::new(&geometry, i).unwrap())
            .collect();
        let previous = rigid_instances.as_ref().map_or_else(
            || rigid.iter().map(RigidBinding::bounds).collect(),
            |b| b.bind_bounds().to_vec(),
        );
        animation = Some(Animation {
            gallery: None,
            rigid,
            rigid_instances,
            skin: None,
            intel_courtyard: true,
            previous,
            rigid_time: 0.,
            skin_time: 0.,
        });
        geometry
    } else if sponza {
        TriangleWorld::load(
            "assets/sponza.obj",
            glam::Mat3::from_cols_array(&[0., 1., 0., 0., 0., 1., 1., 0., 0.]),
            Some(GpuVec3::splat(0.7)),
        )
    } else {
        let mut mesh = MeshBuilder::default();
        let white = [0.72, 0.72, 0.69];
        mesh.block([-460., -380., -16.], [460., 440., 0.], [0.56, 0.58, 0.60]);
        mesh.block(
            [-370., -140., 0.],
            [-350., 390., 260.],
            [0.70, 0.075, 0.045],
        );
        mesh.block([350., -140., 0.], [370., 390., 260.], [0.055, 0.32, 0.68]);
        mesh.block([-350., 370., 0.], [350., 390., 260.], white);
        mesh.block([-150., -15., 0.], [-25., 110., 115.], white);
        mesh.block([35., 120., 0.], [140., 225., 205.], white);
        mesh.sphere(Vec3::new(175., -120., 70.), 70., [0.78, 0.78, 0.78]);
        // Low alcove: sunlight has to bounce from the red wall onto its white surfaces.
        mesh.block([-345., 170., 0.], [-180., 185., 145.], white);
        mesh.block([-345., 170., 145.], [-180., 365., 160.], white);
        for y in [-80., 65., 210.] {
            mesh.block([265., y, 0.], [285., y + 25., 285.], white);
        }
        // A slatted roof makes changes in sunlight direction very easy to see.
        for y in [-110., -40., 30., 100., 170., 240., 310.] {
            mesh.block([190., y, 275.], [365., y + 18., 290.], white);
        }
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        let mut influences = Vec::new();
        let (rings, slices) = (96u32, 32u32);
        for j in 0..=rings {
            let t = j as f32 / rings as f32;
            let bone = t * (SKIN_JOINTS - 1) as f32;
            let lower = (bone.floor() as u32).min(SKIN_JOINTS as u32 - 2);
            let blend = bone - lower as f32;
            let z = t * SKIN_SEGMENT * (SKIN_JOINTS - 1) as f32;
            let radius = 28. * (1.0 - 0.55 * t);
            for i in 0..=slices {
                let phi = std::f32::consts::TAU * i as f32 / slices as f32;
                vertices.push(
                    (SKIN_ROOT + Vec3::new(radius * phi.cos(), radius * phi.sin(), z)).into(),
                );
                influences.push(SkinWeights {
                    joints: [lower, lower + 1, 0, 0],
                    weights: [1. - blend, blend, 0., 0.],
                });
            }
        }
        for j in 0..rings {
            for i in 0..slices {
                let a = j * (slices + 1) + i;
                let b = a + slices + 1;
                triangles.push([a, a + 1, b]);
                triangles.push([a + 1, b + 1, b]);
            }
        }
        // Close both caps so the animated mesh remains an opaque occluder.
        for &(ring, flip) in &[(0, true), (rings, false)] {
            let center = vertices.len() as u32;
            vertices.push(
                (SKIN_ROOT
                    + Vec3::Z
                        * (ring as f32 / rings as f32 * SKIN_SEGMENT * (SKIN_JOINTS - 1) as f32))
                    .into(),
            );
            influences.push(if flip {
                SkinWeights {
                    joints: [0; 4],
                    weights: [1., 0., 0., 0.],
                }
            } else {
                SkinWeights {
                    joints: [SKIN_JOINTS as u32 - 1; 4],
                    weights: [1., 0., 0., 0.],
                }
            });
            for i in 0..slices {
                let a = ring * (slices + 1) + i;
                triangles.push(if flip {
                    [center, a + 1, a]
                } else {
                    [center, a, a + 1]
                });
            }
        }
        let skin_index = mesh.offsets.len();
        mesh.mesh(vertices, triangles, [0.62, 0.33, 0.07]);
        let mut geometry = mesh.finish(&[4, skin_index]);
        let inverse_bind: Vec<_> = (0..SKIN_JOINTS)
            .map(|i| Mat4::from_translation(-(SKIN_ROOT + Vec3::Z * i as f32 * SKIN_SEGMENT)))
            .collect();
        let rigid_instances = (!rigid_refit).then(|| RigidBatch::new(&mut geometry, &[4]).unwrap());
        let rigid: Vec<_> = if rigid_refit {
            vec![RigidBinding::new(&geometry, 4).unwrap()]
        } else {
            vec![]
        };
        let skin = SkinBinding::new(&geometry, skin_index, &influences, &inverse_bind).unwrap();
        let rigid_bounds = rigid_instances
            .as_ref()
            .map_or_else(|| rigid[0].bounds(), |b| b.bind_bounds()[0]);
        let previous = vec![rigid_bounds, skin.bounds()];
        animation = Some(Animation {
            gallery: None,
            rigid,
            rigid_instances,
            skin: Some(skin),
            intel_courtyard: false,
            previous,
            rigid_time: 0.,
            skin_time: 0.,
        });
        geometry
    };
    LabScene {
        world: WorldData {
            world: geometry,
            // Constant sky deliberately avoids dependence on an external HDR asset.
            envmap: Rgb32FImage::from_pixel(8, 4, Rgb([0.65, 0.78, 1.0])),
            envmap_scale: 0.18,
            sun_dir: Vec3::new(-0.5, -0.6, 0.7).normalize().into(),
            sun_color: GpuVec3::splat(5.0),
            custom_emitters: vec![],
        },
        animation,
    }
}

pub fn camera(
    index: usize,
    size: [u32; 2],
    sponza: bool,
    intel: bool,
    gallery: bool,
) -> yesod::camera::Camera {
    let (pos, target) = if gallery {
        match index {
            1 => (Vec3::new(650., -1050., 420.), Vec3::new(-100., 250., 130.)),
            2 => (Vec3::new(-820., -820., 260.), Vec3::new(-300., 100., 110.)),
            _ => (
                Vec3::new(1050., -1950., 1100.),
                Vec3::new(-100., 150., 180.),
            ),
        }
    } else if intel {
        match index {
            1 => (Vec3::new(650., -100., 320.), Vec3::new(-100., 0., 65.)),
            2 => (Vec3::new(300., -175., 140.), Vec3::new(0., 0., 45.)),
            _ => (Vec3::new(1050., 0., 330.), Vec3::new(-250., 0., 100.)),
        }
    } else if sponza {
        (Vec3::new(-94., -937., 613.), Vec3::new(0., 100., 80.))
    } else {
        match index {
            1 => (Vec3::new(430., -470., 260.), Vec3::new(125., 80., 60.)),
            2 => (Vec3::new(-70., -310., 160.), Vec3::new(-290., 240., 70.)),
            _ => (Vec3::new(520., -800., 510.), Vec3::new(0., 70., 80.)),
        }
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
