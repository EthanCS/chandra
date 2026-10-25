    /// Update instance transforms without replacing their primitives. In
    /// particular, do not set PRIMITIVE: the CUDA backend treats that flag as
    /// requiring a full TLAS rebuild, even when the mesh handle is unchanged.
    pub fn set_transforms(&self, transforms: &[(usize, Mat4)]) {
        let mut modifications = self.modifications.write();
        let count = self.instance_handles.read().len();
        assert!(transforms.iter().all(|(index, _)| *index < count));
        for &(index, transform) in transforms {
            let modification = modifications.entry(index).or_insert(api::AccelBuildModification {
                index: index as u32,
                user_id: 0,
                flags: api::AccelBuildModificationFlags::EMPTY,
                visibility: 0,
                mesh: 0,
                affine: [0.0; 12],
            });
            modification.flags |= api::AccelBuildModificationFlags::TRANSFORM;
            modification.affine = transform.into_affine3x4();
        }
    }
