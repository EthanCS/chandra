//! Static glTF PBR attributes. All color textures are uploaded in linear space.
use super::*;

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct PbrMaterial {
    /// Perceptual roughness, metallic, dielectric F0, normal scale.
    pub parameters: Vec4<f32>,
    /// Base color, metallic-roughness, normal and emissive texture indices.
    pub textures: [u32; 4],
    /// Emissive RGB and base-color alpha factor.
    pub emission_alpha: Vec4<f32>,
    /// Negative disables alpha masking.
    pub alpha_cutoff: f32,
}
impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            parameters: Vec4::new(0.6, 0., 0.04, 1.),
            textures: [u32::MAX; 4],
            emission_alpha: Vec4::new(0., 0., 0., 1.),
            alpha_cutoff: -1.,
        }
    }
}
pub(super) struct PbrData {
    materials: Buffer<PbrMaterial>,
    normals: Buffer<[f32; 3]>,
    uv: Buffer<[f32; 2]>,
    density: Buffer<f32>,
    sizes: Buffer<Vec2<f32>>,
    array: BindlessArray,
    _textures: Vec<Tex2d<Vec4<f32>>>,
    pub has_mask: bool,
    pub hardware_alpha: bool,
    pub masked_meshes: Vec<bool>,
}
impl TriangleWorld {
    pub fn set_pbr(
        &mut self,
        materials: &[PbrMaterial],
        normals: &[[f32; 3]],
        uv: &[[f32; 2]],
        density: &[f32],
        textures: Vec<Tex2d<Vec4<f32>>>,
    ) {
        assert_eq!(materials.len(), self.material_diffuse.len());
        assert_eq!(normals.len(), self.positions.len());
        assert_eq!(uv.len(), normals.len());
        assert_eq!(density.len(), self.indices.len());
        assert!(
            materials
                .iter()
                .flat_map(|m| m.textures)
                .all(|i| i == u32::MAX || (i as usize) < textures.len())
        );
        let array = DEVICE.create_bindless_array(textures.len().max(1));
        for (i, t) in textures.iter().enumerate() {
            array.emplace_tex2d_async(
                i,
                t,
                Sampler {
                    filter: SamplerFilter::LinearLinear,
                    address: SamplerAddress::Repeat,
                },
            );
        }
        array.update();
        let mut sizes: Vec<_> = textures
            .iter()
            .map(|t| Vec2::new(t.width() as f32, t.height() as f32))
            .collect();
        if sizes.is_empty() {
            sizes.push(Vec2::new(1., 1.));
        }
        let masked_meshes: Vec<_> = self
            .material_indices
            .copy_to_vec()
            .iter()
            .map(|&i| materials[i as usize].alpha_cutoff >= 0.0)
            .collect();
        // Static/world-space meshes have identity instances. Preserve the
        // existing fallback if material setup follows rigid/group conversion.
        let hardware_alpha = self.group_hit_offsets.is_none()
            && self.rigid_instance_vertices.iter().all(Option::is_none);
        if hardware_alpha {
            for (i, mesh) in self.meshes.iter().enumerate() {
                self.accel
                    .set_mesh(i, mesh, Mat4::identity(), 0xFF, !masked_meshes[i]);
            }
            self.accel.build(AccelBuildRequest::ForceBuild);
        }
        self.pbr = Some(PbrData {
            materials: DEVICE.create_buffer_from_slice(materials),
            normals: DEVICE.create_buffer_from_slice(normals),
            uv: DEVICE.create_buffer_from_slice(uv),
            density: DEVICE.create_buffer_from_slice(density),
            sizes: DEVICE.create_buffer_from_slice(&sizes),
            array,
            _textures: textures,
            has_mask: materials.iter().any(|m| m.alpha_cutoff >= 0.),
            hardware_alpha,
            masked_meshes,
        });
    }
    pub fn has_pbr(&self) -> bool {
        self.pbr.is_some()
    }
    #[tracked]
    pub fn pbr_material(&self, hit: Expr<SurfaceHit>) -> Expr<PbrMaterial> {
        if let Some(data) = &self.pbr {
            data.materials.read(self.material_indices.read(hit.inst))
        } else {
            PbrMaterial::default().expr()
        }
    }
    #[tracked]
    fn pbr_uv(&self, hit: Expr<SurfaceHit>) -> Expr<Vec2<f32>> {
        if let Some(data) = &self.pbr {
            let o = self.offsets.read(hit.inst);
            let t = self.indices.read(o[0] + hit.prim);
            let v: [_; 3] =
                std::array::from_fn(|i| Expr::<Vec2<f32>>::from(data.uv.read(o[1] + t[i as u32])));
            hit.interpolate(v[0], v[1], v[2])
        } else {
            Vec2::splat_expr(0.)
        }
    }
    #[tracked]
    pub fn pbr_texture(
        &self,
        hit: Expr<SurfaceHit>,
        texture: Expr<u32>,
        fine: bool,
    ) -> Expr<Vec4<f32>> {
        let value = Vec4::splat(1.0f32).var();
        if let Some(data) = &self.pbr {
            if texture != u32::MAX {
                let uv = self.pbr_uv(hit);
                let size = data.sizes.read(texture);
                let o = self.offsets.read(hit.inst);
                let footprint = if fine {
                    1.0f32.expr()
                } else {
                    hit.committed_ray_t
                        * 0.0008
                        * data.density.read(o[0] + hit.prim)
                        * keter::max(size.x, size.y)
                };
                let lod = keter::max(footprint, 1.0f32).log2();
                let level = lod.floor().cast_u32();
                let f = lod.fract();
                let tex = data.array.tex2d(texture);
                *value = tex.sample_level(uv, level) * (1.0 - f)
                    + tex.sample_level(uv, level + 1u32) * f;
            }
        }
        **value
    }
    #[tracked]
    pub fn accepts_hit(&self, hit: Expr<SurfaceHit>) -> Expr<bool> {
        let m = self.pbr_material(hit);
        m.alpha_cutoff < 0.0
            || self.pbr_texture(hit, m.textures[0], true).w * m.emission_alpha.w >= m.alpha_cutoff
    }
    #[tracked]
    pub fn get_roughness_metallic(&self, hit: Expr<SurfaceHit>) -> Expr<Vec2<f32>> {
        let m = self.pbr_material(hit);
        let tex = self.pbr_texture(hit, m.textures[1], false);
        Vec2::expr(
            (m.parameters.x * tex.y).clamp(0., 1.),
            (m.parameters.y * tex.z).clamp(0., 1.),
        )
    }
    #[tracked]
    pub fn get_f0(&self, hit: Expr<SurfaceHit>) -> Expr<Vec3<f32>> {
        let metal = self.get_roughness_metallic(hit).y;
        Vec3::splat_expr(self.pbr_material(hit).parameters.z) * (1.0 - metal)
            + self.get_base_color(hit) * metal
    }
    /// Fetch the shared base-color and metallic/roughness inputs once for both lobes.
    #[tracked]
    pub fn get_pbr_lobes(
        &self,
        hit: Expr<SurfaceHit>,
    ) -> (Expr<f32>, Expr<Vec3<f32>>, Expr<Vec3<f32>>) {
        let rm = self.get_roughness_metallic(hit);
        let color = self.get_base_color(hit);
        let dielectric = self.pbr_material(hit).parameters.z;
        let f0 = Vec3::splat_expr(dielectric) * (1.0 - rm.y) + color * rm.y;
        let diffuse = if self.pbr.is_some() {
            color * (1.0 - rm.y) * (1.0 - dielectric)
        } else {
            color
        };
        (rm.x, f0, diffuse)
    }
    #[tracked]
    pub fn get_diffuse(&self, hit: Expr<SurfaceHit>) -> Expr<Vec3<f32>> {
        let color = self.get_base_color(hit);
        if self.pbr.is_some() {
            color
                * (1.0 - self.get_roughness_metallic(hit).y)
                * (1.0 - self.pbr_material(hit).parameters.z)
        } else {
            color
        }
    }
    #[tracked]
    pub fn get_shading_normal(
        &self,
        hit: Expr<SurfaceHit>,
        direction: Expr<Vec3<f32>>,
    ) -> Expr<Vec3<f32>> {
        let geometric = self.get_normal_flat(hit, direction);
        if let Some(data) = &self.pbr {
            let o = self.offsets.read(hit.inst);
            let t = self.indices.read(o[0] + hit.prim);
            let v: [_; 3] = std::array::from_fn(|i| {
                Expr::<Vec3<f32>>::from(data.normals.read(o[1] + t[i as u32]))
            });
            let n = hit.interpolate(v[0], v[1], v[2]).normalize().var();
            if n.dot(geometric) < 0.0 {
                *n = -n;
            }
            let m = self.pbr_material(hit);
            if m.textures[2] != u32::MAX {
                let p: [_; 3] = std::array::from_fn(|i| {
                    Expr::<Vec3<f32>>::from(self.positions.read(o[1] + t[i as u32]))
                });
                let u: [_; 3] = std::array::from_fn(|i| {
                    Expr::<Vec2<f32>>::from(data.uv.read(o[1] + t[i as u32]))
                });
                let du = u[1] - u[0];
                let dv = u[2] - u[0];
                let det = du.x * dv.y - du.y * dv.x;
                if det.abs() > 0.0000001 {
                    let tangent = ((p[1] - p[0]) * dv.y - (p[2] - p[0]) * du.y) / det;
                    let tangent = (tangent - n * n.dot(tangent)).normalize();
                    let bitangent = n.cross(tangent) * det.signum();
                    let map = self.pbr_texture(hit, m.textures[2], false).xyz() * 2.0 - 1.0;
                    *n = (tangent * map.x * m.parameters.w
                        + bitangent * map.y * m.parameters.w
                        + n * map.z)
                        .normalize();
                }
            }
            if n.dot(geometric) < 0.05 || n.dot(-direction) < 0.001 {
                *n = geometric;
            }
            **n
        } else {
            geometric
        }
    }
}
