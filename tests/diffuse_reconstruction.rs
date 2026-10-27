use chandra::algorithm::CoreData;
use chandra::algorithm::diffuse::DiffuseData;
use keter::lang::types::vector::Vec4;
use keter::prelude::*;

#[test]
fn short_temporal_windows_do_not_permanently_raise_diffuse_ray_budgets() {
    let size = [32, 32];
    let core = CoreData::new(size, chandra::white_room::world());
    let camera = chandra::white_room::camera(0, size).view();
    core.view.write_host(camera).execute();
    core.compute_gbuffer().execute();
    let data = DiffuseData::new(&core);
    data.raw.copy_from(&vec![Vec4::new(2., 1., 0.5, 1.); 1024]);
    // 20/30 FPS and a short light-response window. Temporal blending must
    // stay short, while four valid observations still establish support.
    for limit in [1., 4. / 3., 2., 8. / 3., 4.] {
        for frame in 0..8 {
            data.prepare(&core, frame % 2, camera, frame > 0, 2, 0).execute();
            if frame == 0 {
                assert!(data.samples.copy_to_vec().iter().all(|&v| v == 64));
            }
            if frame >= 4 {
                assert!(data.samples.copy_to_vec().iter().filter(|&&v| v == 2).count() > 500,
                    "Short history limit {limit} kept stable pixels at 64 rays");
            }
            data.reconstruct(&core, frame % 2, camera, frame > 0, limit).execute();
            for v in data.output.copy_to_vec() {
                assert!(v.w <= limit + 1e-5 && (v.x - 2.).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn diffuse_history_preserves_energy_follows_energy_edits_and_rejects_disocclusion() {
    let size = [32, 32];
    let core = CoreData::new(size, chandra::white_room::world());
    let first = chandra::white_room::camera(0, size).view();
    core.view.write_host(first).execute();
    core.compute_gbuffer().execute();
    let data = DiffuseData::new(&core);
    data.prepare(&core, 0, first, false, 2, 0).execute();
    assert!(data.samples.copy_to_vec().iter().all(|&v| v == 64));
    data.raw.copy_from(&vec![Vec4::new(2., 1., 0.5, 1.); 1024]);
    data.reconstruct(&core, 0, first, false, 16.).execute();
    for value in data.output.copy_to_vec() {
        assert!((value.x - 2.).abs() < 1e-5 && (value.y - 1.).abs() < 1e-5);
    }
    for phase in [1, 0, 1] {
        data.prepare(&core, phase, first, true, 2, 0).execute();
        data.reconstruct(&core, phase, first, true, 16.).execute();
    }
    data.prepare(&core, 0, first, true, 2, 0).execute();
    assert!(
        data.samples
            .copy_to_vec()
            .iter()
            .filter(|&&v| v == 2)
            .count()
            > 500,
        "Stable surfaces never became eligible for normal sampling"
    );
    let noisy: Vec<_> = (0..1024)
        .map(|i| {
            Vec4::new(
                2. + if (i + i / 32) % 2 == 0 { 0.1 } else { -0.1 },
                1.,
                0.5,
                1.,
            )
        })
        .collect();
    data.raw.copy_from(&noisy);
    data.reconstruct(&core, 0, first, true, 16.).execute();
    let values = data.output.copy_to_vec();
    let error = values.iter().map(|v| (v.x - 2.).powi(2)).sum::<f32>() / 1024.;
    assert!(
        error < 0.001,
        "Geometry-guided filtering did not reduce alternating noise: {error}"
    );
    let mean = values.iter().map(|v| v.x).sum::<f32>() / 1024.;
    assert!((mean - 2.).abs() < 0.01, "Filtering changed mean energy");
    // A genuine energy edit must not be hidden by a long temporal trail.
    data.prepare(&core, 1, first, true, 2, 0).execute();
    data.raw.copy_from(&vec![Vec4::new(0., 0., 0., 1.); 1024]);
    data.reconstruct(&core, 1, first, true, 2.).execute();
    assert!(
        data.output
            .copy_to_vec()
            .iter()
            .all(|v| v.x == 0. && v.y == 0. && v.z == 0.)
    );
    let cut = chandra::white_room::camera(1, size).view();
    core.view.write_host(cut).execute();
    core.compute_gbuffer().execute();
    data.prepare(&core, 0, first, true, 2, 0).execute();
    assert!(
        data.samples
            .copy_to_vec()
            .iter()
            .filter(|&&v| v == 64)
            .count()
            > 1000
    );
    data.raw.copy_from(&vec![Vec4::new(3., 2., 1., 1.); 1024]);
    data.reconstruct(&core, 0, first, true, 16.).execute();
    for value in data.output.copy_to_vec() {
        if value.w > 0. {
            assert!(
                (value.x - 3.).abs() < 1e-5,
                "New surface inherited old screen colour"
            );
        }
    }
}
