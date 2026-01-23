use super::*;

impl HashCache {
    #[tracked]
    pub fn get_diffuse(
        &self,
        node: Expr<LoadedProbe>,
        normal: Expr<Vec3<f32>>,
        shadow_mask: Expr<u32>,
    ) -> Expr<Radiance> {
        let total_radiance = Vec4::splat(0.0_f32).var();
        for i in 0_u32.expr()..node.encoder.linear_size() {
            let dir = node.encoder.linear_to_dir(i);
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
    #[tracked]
    pub fn get_uniform(&self, node: Expr<LoadedProbe>) -> Expr<Radiance> {
        let total_radiance = Vec3::splat(0.0_f32).var();
        for i in 0_u32.expr()..node.encoder.linear_size() {
            let data_index = node.data_offset + i;
            let deposit = self.data.read(data_index);
            *total_radiance += deposit.radiance();
        }
        // TODO: Why the hell is this necessary??? Is a rust-analyzer issue.
        let out: Expr<Vec3<f32>> = total_radiance / node.encoder.linear_size().cast_f32();
        out
    }
    #[tracked]
    pub fn get_dir(&self, node: Expr<LoadedProbe>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.data
            .read(node.data_offset + node.encoder.dir_to_linear(dir))
            .radiance()
    }
    #[tracked]
    pub fn get_dir_bilinear(
        &self,
        node: Expr<LoadedProbe>,
        dir: Expr<Vec3<f32>>,
    ) -> Expr<Radiance> {
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
    #[tracked]
    pub fn query_uniform(&self, probe: Expr<Probe>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_uniform(node),
            || Vec3::splat_expr(-1.0),
        )
    }
    #[tracked]
    pub fn query_dir(&self, probe: Expr<Probe>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_dir(node, dir),
            || Vec3::splat_expr(-1.0),
        )
    }
    #[tracked]
    pub fn query_dir_bilinear(&self, probe: Expr<Probe>, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        self.query(
            probe,
            |node| self.get_dir_bilinear(node, dir),
            || Vec3::splat_expr(-1.0),
        )
    }
    // TODO: This doesn't really seem to balance out.
    #[tracked]
    pub fn try_deposit(
        &self,
        probe: Expr<Probe>,
        f: impl Fn() -> (Expr<Vec3<f32>>, Expr<RadianceDeposit>),
    ) -> Expr<bool> {
        self.query_init(
            probe,
            |node| {
                let (dir, deposit) = f();
                let data = self
                    .data
                    .atomic_ref(node.data_offset + node.encoder.dir_to_linear(dir));
                data.deposit(deposit);
                true.expr()
            },
            || false.expr(),
        )
    }
    #[tracked]
    pub fn init(&self, probe: Expr<Probe>) -> Expr<bool> {
        self.query_init(probe, |_| true.expr(), || false.expr())
    }
    #[tracked]
    pub fn contains(&self, probe: Expr<Probe>) -> Expr<bool> {
        self.query(probe, |_| true.expr(), || false.expr())
    }
    // TODO: Deduplicate.
    #[tracked]
    pub fn contains_cost(&self, probe: Expr<Probe>) -> (Expr<bool>, Expr<u32>) {
        let node_offset = self.get_index_offset(probe.level);
        let table_size = self.base_probes >> (2 * probe.level);
        let position = encode_position(probe.position);
        let node_index = (hash64(position) % table_size).var();
        let loaded = false.var();
        let i = 0_u32.var();
        for j in 0..10_u32 {
            *i = j;
            let node = self.nodes.read(node_offset + node_index);
            if node.position == position {
                // Success.
                *loaded = true;
                break;
            }
            *node_index = (node_index + 1) % table_size;
        }
        (**loaded, **i)
    }
    #[tracked]
    pub fn deposit(
        &self,
        probe: Expr<Probe>,
        dir: Expr<Vec3<f32>>,
        deposit: Expr<RadianceDeposit>,
    ) -> Expr<bool> {
        self.try_deposit(probe, || (dir, deposit))
    }
    /*
    #[tracked]
    pub fn deposit_bilinear(
        &self,
        pos: Expr<Vec3<f32>>,
        level: Expr<u32>,
        dir: Expr<Vec3<f32>>,
        deposit: Expr<RadianceDeposit>,
    ) {
        let spacing = self.base_grid_size * (1_u32 << level).cast_f32();
        let base_probe = self.quantize(pos - spacing / 2.0).shift_level(level);
        let frac = (pos / spacing - 0.5).rem_euclid(1.0); // or something like it
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
                position: base_probe.position + mask,
                level: base_probe.level,
            });
            self.deposit(probe, dir, deposit.weighted(weight));
        }
    }
    */
}
