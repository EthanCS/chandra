//! GPU rigid deformation, linear blend skinning, and conservative motion bounds.
//!
//! Mesh topology is fixed. Queue deformation, BLAS/TLAS updates, history
//! invalidation, then rendering on the same stream. No per-frame vertex readback
//! or CPU skinning is necessary.
use glam::{Mat4 as HostMat4, Vec3 as HostVec3};
use keter::graph::NodeConfigs;
use keter::lang::types::vector::{Mat4, Vec3, Vec4};
use keter::prelude::*;

use crate::world::TriangleWorld;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: HostVec3,
    pub max: HostVec3,
}

impl Bounds {
    pub fn from_points(points: impl IntoIterator<Item = HostVec3>) -> Self {
        let mut bounds = Self {
            min: HostVec3::splat(f32::INFINITY),
            max: HostVec3::splat(f32::NEG_INFINITY),
        };
        for p in points {
            bounds.min = bounds.min.min(p);
            bounds.max = bounds.max.max(p);
        }
        bounds
    }
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
    pub fn expanded(self, distance: f32) -> Self {
        Self {
            min: self.min - distance,
            max: self.max + distance,
        }
    }
    pub fn transformed(self, matrix: HostMat4) -> Self {
        Self::from_points((0..8).map(|i| {
            matrix.transform_point3(HostVec3::new(
                if i & 1 == 0 { self.min.x } else { self.max.x },
                if i & 2 == 0 { self.min.y } else { self.max.y },
                if i & 4 == 0 { self.min.z } else { self.max.z },
            ))
        }))
    }
    pub fn contains(self, p: HostVec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

fn bind_vertices(world: &TriangleWorld, mesh: usize) -> Result<&[[f32; 3]], String> {
    world
        .bind_positions
        .get(mesh)
        .filter(|p| !p.is_empty())
        .map(Vec::as_slice)
        .ok_or_else(|| format!("Mesh {mesh} was not created as a dynamic mesh"))
}

/// Reference path: transform bind vertices and refit the BLAS. RigidBatch uses
/// immutable BLASes and instance transforms for production rigid animation.
pub struct RigidBinding {
    pub mesh: usize,
    bind_bounds: Bounds,
    deform: Kernel<fn(Mat4)>,
    count: u32,
    _bind: Buffer<[f32; 3]>,
}

impl RigidBinding {
    pub fn new(world: &TriangleWorld, mesh: usize) -> Result<Self, String> {
        let vertices = bind_vertices(world, mesh)?;
        if world.is_rigid_instance(mesh) {
            return Err("Use the instance batch for this rigid mesh".into());
        }
        let bind_bounds = Bounds::from_points(vertices.iter().copied().map(HostVec3::from));
        let bind = DEVICE.create_buffer_from_slice(vertices);
        let start = world.mesh_vertex_ranges[mesh].start as u32;
        let deform = DEVICE.create_kernel::<fn(Mat4)>(&track!(|transform| {
            let i = dispatch_id().x;
            let p = Expr::<Vec3<f32>>::from(bind.read(i));
            let position = (transform * p.extend(1.0)).xyz();
            world
                .positions
                .write(start + i, Expr::<[f32; 3]>::from(position));
        }));
        Ok(Self {
            mesh,
            bind_bounds,
            deform,
            count: vertices.len() as u32,
            _bind: bind,
        })
    }
    pub fn update(&self, transform: HostMat4) -> Result<(NodeConfigs<'static>, Bounds), String> {
        if !transform.is_finite() || transform.determinant().abs() < 1e-8 {
            return Err("Rigid transform must be finite and nonsingular".into());
        }
        Ok((
            self.deform
                .dispatch_async([self.count, 1, 1], &Mat4::from(transform))
                .debug("Rigid deformation"),
            self.bind_bounds.transformed(transform),
        ))
    }
    pub fn bounds(&self) -> Bounds {
        self.bind_bounds
    }
}

/// Immutable rigid BLASes, one batched shading-vertex update, and transform-only
/// TLAS edits. Queue the returned vertex update before update_acceleration.
pub struct RigidBatch {
    meshes: Vec<usize>,
    bounds: Vec<Bounds>,
    palette: Buffer<Mat4>,
    deform: Kernel<fn()>,
    count: u32,
    _bind: Buffer<[f32; 3]>,
    _owners: Buffer<u32>,
    _destinations: Buffer<u32>,
}

impl RigidBatch {
    pub fn new(world: &mut TriangleWorld, meshes: &[usize]) -> Result<Self, String> {
        if meshes.is_empty() {
            return Err("Rigid batch cannot be empty".into());
        }
        let mut seen = std::collections::HashSet::new();
        if meshes.iter().any(|i| !seen.insert(*i)) {
            return Err("Duplicate rigid mesh".into());
        }
        let mut vertices = Vec::new();
        let mut owners = Vec::new();
        let mut destinations = Vec::new();
        let mut bounds = Vec::new();
        for (slot, &mesh) in meshes.iter().enumerate() {
            let bind = bind_vertices(world, mesh)?;
            bounds.push(Bounds::from_points(
                bind.iter().copied().map(HostVec3::from),
            ));
            vertices.extend_from_slice(bind);
            owners.extend(std::iter::repeat_n(slot as u32, bind.len()));
            destinations.extend(world.mesh_vertex_ranges[mesh].clone().map(|i| i as u32));
        }
        world.enable_rigid_instances(meshes)?;
        let count = vertices.len() as u32;
        let bind = DEVICE.create_buffer_from_slice(&vertices);
        let owners = DEVICE.create_buffer_from_slice(&owners);
        let destinations = DEVICE.create_buffer_from_slice(&destinations);
        let palette = DEVICE.create_buffer_from_fn(meshes.len(), |_| Mat4::identity());
        let deform = DEVICE.create_kernel::<fn()>(&track!(|| {
            let i = dispatch_id().x;
            let p = Expr::<Vec3<f32>>::from(bind.read(i));
            let position = (palette.read(owners.read(i)) * p.extend(1.0)).xyz();
            world
                .positions
                .write(destinations.read(i), Expr::<[f32; 3]>::from(position));
        }));
        Ok(Self {
            meshes: meshes.to_vec(),
            bounds,
            palette,
            deform,
            count,
            _bind: bind,
            _owners: owners,
            _destinations: destinations,
        })
    }
    pub fn meshes(&self) -> &[usize] {
        &self.meshes
    }
    pub fn bind_bounds(&self) -> &[Bounds] {
        &self.bounds
    }

    pub fn update(
        &self,
        world: &TriangleWorld,
        transforms: &[HostMat4],
    ) -> Result<(NodeConfigs<'static>, Vec<Bounds>), String> {
        if transforms.len() != self.meshes.len()
            || transforms.iter().any(|m| {
                !m.is_finite()
                    || m.determinant().abs() < 1e-8
                    || m.x_axis.w != 0.
                    || m.y_axis.w != 0.
                    || m.z_axis.w != 0.
                    || m.w_axis.w != 1.
            })
        {
            return Err(
                "Rigid batch needs one finite nonsingular affine transform per mesh".into(),
            );
        }
        let matrices: Vec<Mat4> = transforms.iter().map(|m| Mat4::from(*m)).collect();
        let instances: Vec<_> = self
            .meshes
            .iter()
            .copied()
            .zip(matrices.iter().copied())
            .collect();
        world.set_rigid_transforms(&instances);
        let bounds = self
            .bounds
            .iter()
            .zip(transforms)
            .map(|(b, m)| b.transformed(*m))
            .collect();
        Ok((
            (
                self.palette.copy_from_vec(matrices),
                self.deform.dispatch_async([self.count, 1, 1]),
            )
                .chain()
                .debug("Batched rigid shading vertices"),
            bounds,
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SkinWeights {
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

impl SkinWeights {
    pub fn normalized(self, joint_count: usize) -> Result<Self, String> {
        if self.joints.iter().any(|&j| j as usize >= joint_count) {
            return Err("Skin joint index is out of range".into());
        }
        if self.weights.iter().any(|&w| !w.is_finite() || w < 0.) {
            return Err("Skin weights must be finite and nonnegative".into());
        }
        let sum: f32 = self.weights.iter().sum();
        if !sum.is_finite() || sum <= 1e-8 {
            return Err("Skin weights must have a positive sum".into());
        }
        Ok(Self {
            joints: self.joints,
            weights: self.weights.map(|w| w / sum),
        })
    }
}

#[derive(Clone, Copy, Value)]
#[repr(C)]
struct GpuSkinWeights {
    joints: Vec4<u32>,
    weights: Vec4<f32>,
}

/// Four influences per vertex, arbitrary joint hierarchy evaluated by the host.
/// Supply current joint-to-world matrices; inverse bind matrices are applied
/// here. The same deformed position buffer drives shading and ray traversal.
pub struct SkinBinding {
    pub mesh: usize,
    inverse_bind: Vec<HostMat4>,
    joint_bounds: Vec<Option<Bounds>>,
    palette: Buffer<Mat4>,
    deform: Kernel<fn()>,
    count: u32,
    bind_bounds: Bounds,
    _bind: Buffer<[f32; 3]>,
    _weights: Buffer<GpuSkinWeights>,
}

impl SkinBinding {
    pub fn new(
        world: &TriangleWorld,
        mesh: usize,
        influences: &[SkinWeights],
        inverse_bind: &[HostMat4],
    ) -> Result<Self, String> {
        let vertices = bind_vertices(world, mesh)?;
        if world.is_rigid_instance(mesh) {
            return Err("A skinned mesh cannot use an immutable rigid BLAS".into());
        }
        if vertices.len() != influences.len() || inverse_bind.is_empty() {
            return Err("Skin needs one influence record per vertex and at least one joint".into());
        }
        if inverse_bind
            .iter()
            .any(|m| !m.is_finite() || m.determinant().abs() < 1e-8)
        {
            return Err("Inverse bind matrices must be finite and nonsingular".into());
        }
        let influences: Vec<_> = influences
            .iter()
            .map(|w| w.normalized(inverse_bind.len()))
            .collect::<Result<_, _>>()?;
        let mut joint_points = vec![Vec::new(); inverse_bind.len()];
        for (&p, w) in vertices.iter().zip(&influences) {
            for k in 0..4 {
                if w.weights[k] > 0. {
                    let j = w.joints[k] as usize;
                    joint_points[j].push(inverse_bind[j].transform_point3(HostVec3::from(p)));
                }
            }
        }
        let joint_bounds = joint_points
            .into_iter()
            .map(|p| {
                if p.is_empty() {
                    None
                } else {
                    Some(Bounds::from_points(p))
                }
            })
            .collect();
        let weights = DEVICE.create_buffer_from_slice(
            &influences
                .iter()
                .map(|w| GpuSkinWeights {
                    joints: w.joints.into(),
                    weights: w.weights.into(),
                })
                .collect::<Vec<_>>(),
        );
        let bind = DEVICE.create_buffer_from_slice(vertices);
        let palette = DEVICE.create_buffer_from_fn(inverse_bind.len(), |_| Mat4::identity());
        let start = world.mesh_vertex_ranges[mesh].start as u32;
        let deform = DEVICE.create_kernel::<fn()>(&track!(|| {
            let i = dispatch_id().x;
            let p = Expr::<Vec3<f32>>::from(bind.read(i)).extend(1.0);
            let skin = weights.read(i);
            let position = Vec3::splat(0.0f32).var();
            for k in 0..4u32 {
                *position += (palette.read(skin.joints[k]) * p).xyz() * skin.weights[k];
            }
            world
                .positions
                .write(start + i, Expr::<[f32; 3]>::from(**position));
        }));
        Ok(Self {
            mesh,
            inverse_bind: inverse_bind.to_vec(),
            joint_bounds,
            palette,
            deform,
            count: vertices.len() as u32,
            bind_bounds: Bounds::from_points(vertices.iter().copied().map(HostVec3::from)),
            _bind: bind,
            _weights: weights,
        })
    }
    pub fn update(
        &self,
        joint_world: &[HostMat4],
    ) -> Result<(NodeConfigs<'static>, Bounds), String> {
        if joint_world.len() != self.inverse_bind.len()
            || joint_world.iter().any(|m| !m.is_finite())
        {
            return Err("Joint palette has the wrong size or nonfinite transforms".into());
        }
        let matrices = joint_world
            .iter()
            .zip(&self.inverse_bind)
            .map(|(joint, inverse)| Mat4::from(*joint * *inverse))
            .collect();
        let bounds = self
            .joint_bounds
            .iter()
            .zip(joint_world)
            .filter_map(|(b, m)| b.map(|b| b.transformed(*m)))
            .reduce(Bounds::union)
            .unwrap();
        Ok((
            (
                self.palette.copy_from_vec(matrices),
                self.deform.dispatch_async([self.count, 1, 1]),
            )
                .chain()
                .debug("GPU skinning"),
            bounds,
        ))
    }
    pub fn bounds(&self) -> Bounds {
        self.bind_bounds
    }
    pub fn vertex_count(&self) -> u32 {
        self.count
    }
}

/// One upload and one dispatch for many independent skeletons. Bone transforms
/// and all four influences are identical to SkinBinding's reference path.
pub struct SkinBatch {
    palette: Buffer<Mat4>,
    deform: Kernel<fn()>,
    count: u32,
    inverse_bind: Vec<HostMat4>,
    _bind: Buffer<[f32; 3]>,
    _weights: Buffer<GpuSkinWeights>,
    _destinations: Buffer<u32>,
}

impl SkinBatch {
    pub fn new(world: &TriangleWorld, bindings: &[&SkinBinding]) -> Self {
        assert!(!bindings.is_empty());
        let mut vertices = Vec::new();
        let mut weights = Vec::new();
        let mut destinations = Vec::new();
        let mut inverse_bind = Vec::new();
        for binding in bindings {
            let base = inverse_bind.len() as u32;
            vertices.extend(binding._bind.copy_to_vec());
            weights.extend(binding._weights.copy_to_vec().into_iter().map(|mut w| {
                w.joints = Vec4::new(w.joints.x + base, w.joints.y + base, w.joints.z + base, w.joints.w + base);
                w
            }));
            destinations.extend(world.mesh_vertex_ranges[binding.mesh].clone().map(|i| i as u32));
            inverse_bind.extend_from_slice(&binding.inverse_bind);
        }
        let count = vertices.len() as u32;
        let bind = DEVICE.create_buffer_from_slice(&vertices);
        let weights = DEVICE.create_buffer_from_slice(&weights);
        let destinations = DEVICE.create_buffer_from_slice(&destinations);
        let palette = DEVICE.create_buffer_from_fn(inverse_bind.len(), |_| Mat4::identity());
        let deform = DEVICE.create_kernel::<fn()>(&track!(|| {
            let i = dispatch_id().x;
            let p = Expr::<Vec3<f32>>::from(bind.read(i)).extend(1.0);
            let skin = weights.read(i);
            let position = Vec3::splat(0.0f32).var();
            for k in 0..4u32 {
                *position += (palette.read(skin.joints[k]) * p).xyz() * skin.weights[k];
            }
            world.positions.write(destinations.read(i), Expr::<[f32; 3]>::from(**position));
        }));
        Self { palette, deform, count, inverse_bind, _bind: bind, _weights: weights, _destinations: destinations }
    }

    pub fn update(&self, joints: &[HostMat4]) -> Result<NodeConfigs<'static>, String> {
        if joints.len() != self.inverse_bind.len() || joints.iter().any(|m| !m.is_finite()) {
            return Err("Joint palette has the wrong size or nonfinite transforms".into());
        }
        let matrices = joints.iter().zip(&self.inverse_bind).map(|(j, inv)| Mat4::from(*j * *inv)).collect();
        Ok((self.palette.copy_from_vec(matrices), self.deform.dispatch_async([self.count, 1, 1])).chain().debug("Batched GPU skinning"))
    }
}

impl SkinBinding {
    pub fn transformed_bounds(&self, joints: &[HostMat4]) -> Bounds {
        assert_eq!(joints.len(), self.joint_bounds.len());
        self.joint_bounds.iter().zip(joints).filter_map(|(b, m)| b.map(|b| b.transformed(*m))).reduce(Bounds::union).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotated_swept_bounds_include_old_and_new_vertices() {
        let points = [HostVec3::new(-2., -1., 0.), HostVec3::new(2., 1., 5.)];
        let original = Bounds::from_points(points);
        let transform = HostMat4::from_translation(HostVec3::new(12., -4., 1.))
            * HostMat4::from_rotation_z(1.2);
        let swept = original.union(original.transformed(transform));
        for p in points {
            assert!(swept.contains(p));
            assert!(swept.contains(transform.transform_point3(p)));
        }
    }
    #[test]
    fn skin_weights_are_normalized_and_bad_data_rejected() {
        let normalized = SkinWeights {
            joints: [0, 1, 0, 0],
            weights: [3., 1., 0., 0.],
        }
        .normalized(2)
        .unwrap();
        assert_eq!(normalized.weights, [0.75, 0.25, 0., 0.]);
        assert!(
            SkinWeights {
                weights: [0.; 4],
                ..normalized
            }
            .normalized(2)
            .is_err()
        );
        assert!(
            SkinWeights {
                weights: [f32::NAN, 1., 0., 0.],
                ..normalized
            }
            .normalized(2)
            .is_err()
        );
        assert!(normalized.normalized(1).is_err());
    }
}
