use super::*;

impl HashCache {
    #[tracked]
    pub fn get_diffuse(
        &self,
        node: Expr<LoadedNode>,
        normal: Expr<Vec3<f32>>,
        shadow_mask: Expr<u32>,
    ) -> Expr<Radiance> {
        let encoder = self.get_encoder(node.probe.cascade);
        let total_radiance = Vec4::splat(0.0_f32).var();
        for i in 0_u32.expr()..encoder.linear_size() {
            let dir = encoder.linear_to_dir(i);
            let weight = dir.dot(normal);
            if weight > 0.0 {
                let data_index = node.data_offset + i;
                let deposit = self.data.read(data_index);
                *total_radiance += (deposit.radiance()
                    * (shadow_mask & (1 << i) == 0).cast_u32().cast_f32())
                .extend(1.0)
                    * weight;
            }
        }
        // TODO: Why the hell is this necessary??? Is a rust-analyzer issue.
        let out: Expr<Vec3<f32>> = total_radiance.xyz() / total_radiance.w;
        out
    }
    /*
    #[tracked]
    pub fn get_dir(&self, node: Expr<LoadedNode>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.data
            .read(node.data_offset + node.encoder.dir_to_linear(dir))
            .radiance()
    }
    #[tracked]
    pub fn get_dir_bilinear(&self, node: Expr<LoadedNode>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        let dir = Encoder::encode(dir);
        let dir = ((dir + 1.0) / 2.0 * (node.encoder.resolution).cast_f32()) - 0.5;
        let base = dir.floor();
        let frac = dir - base;
        let base = base.cast_i32();
        let total_radiance = Vec3::splat(0.0_f32).var();
        for mask in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            let mask = Vec2::expr(mask[0] == 1, mask[1] == 1);
            let weight = mask.select(frac, 1.0 - frac).reduce_prod();
            let sample_index = node.encoder.grid_to_linear(
                (base + mask.cast_i32())
                    .rem_euclid(node.encoder.resolution.cast_i32())
                    .cast_u32(),
            );
            let deposit = self.data.read(node.data_offset + sample_index);
            *total_radiance += deposit.radiance() * weight;
        }
        **total_radiance
    }
    #[tracked]
    pub fn query_diffuse(
        &self,
        probe: Expr<Probe>,
        normal: Expr<Vec3<f32>>,
        shadow_mask: Expr<u32>,
    ) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_diffuse(node, normal, shadow_mask),
            || Vec3::splat_expr(-1.0),
        )
    }
    #[tracked]
    pub fn query_diffuse_bilinear(
        &self,
        pos: Expr<Vec3<f32>>,
        level: Expr<u32>,
        normal: Expr<Vec3<f32>>,
        interpolate: Expr<bool>,
        shadow_mask: Expr<u32>,
    ) -> Expr<Radiance> {
        let (pos, cascade) = self.projection().inv_project(pos, interpolate);
        let spacing = (1_u32 << level).cast_f32();
        let base_probe = Probe::base(
            ((pos - spacing / 2.0).floor().cast_i32() - self.offset)
                .cast_u32()
                .extend(cascade),
        )
        .shift_level(level);

        // TODO: Do queries and then sum them up ?
        let frac = (pos / spacing - 0.5).fract(); // or something like it
        let total_radiance = Vec4::splat(0.0_f32).var();
        for i in (0..8) {
            let mask = Vec3::new(
                if (i & 4) != 0 { 1 } else { 0 },
                if (i & 2) != 0 { 1 } else { 0 },
                if (i & 1) != 0 { 1 } else { 0 },
            );
            let weight = mask
                .map(|x| x == 1)
                .expr()
                .select(frac, 1.0 - frac)
                .reduce_prod();
            let probe = Probe::from_comps_expr(ProbeComps {
                position: (base_probe.position.xyz() + mask).extend(base_probe.position.w),
                cascade: base_probe.level,
            });
            *total_radiance += self.query(
                probe,
                |node| self.get_diffuse(node, normal, shadow_mask).extend(1.0) * weight,
                || Vec4::splat_expr(0.0),
            );
        }
        total_radiance.xyz() / total_radiance.w
    }
    */
    #[tracked]
    pub fn deposit(
        &self,
        probe: Expr<Probe>,
        dir: Expr<Vec3<f32>>,
        fluence: Expr<Fluence>,
    ) -> Expr<bool> {
        let node = self.get(probe);
        if node.loaded() {
            let encoder = self.get_encoder(probe.cascade);
            let index = encoder.dir_to_linear(dir);
            let data_index = node.data_offset + index;
            let deposit = self.data.atomic_ref(data_index);
            deposit.deposit(fluence);
        }
        node.loaded()
    }
    #[tracked]
    pub fn contains(&self, probe: Expr<Probe>) -> Expr<bool> {
        self.table.contains(self.encode(probe), probe.cascade)
    }
    #[tracked]
    pub fn cost(&self, probe: Expr<Probe>) -> Expr<u32> {
        self.table.cost(self.encode(probe), probe.cascade)
    }
}
