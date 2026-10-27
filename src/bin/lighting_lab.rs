//! Interactive viewer using Chandra's actual Split Radiance Cascades pipeline.
#[path = "../lab_benchmark.rs"]
mod benchmark;
#[path = "../lab_gltf.rs"]
mod gltf_scene;
#[path = "../lab_pica.rs"]
mod pica_scene;
#[path = "../lab_profile.rs"]
mod gpu_profile;
#[path = "../lab_scene.rs"]
mod scene;

use std::time::Instant;

use chandra::algorithm::dynamic_history::{DynamicHistory, LightingResponse};
use chandra::algorithm::rc::{GiAvailability, RcConfig, RcData};
use chandra::algorithm::splat::SplatData;
use chandra::algorithm::trace::TraceData;
use chandra::algorithm::{CommandList, CoreData, Lighting};
use egui_software_backend::{BufferMutRef, ColorFieldOrder, EguiSoftwareRender};
use keter::lang::types::vector::{Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::{App, KeyCode, MouseButton};
use yesod::camera::View;

#[derive(Clone, PartialEq)]
struct Controls {
    azimuth: f32,
    elevation: f32,
    intensity: f32,
    sky: f32,
    emission: f32,
    point_intensity: f32,
    color: [f32; 3],
    exposure: f32,
    animate: bool,
    mode: u32,
    bounces: bool,
    rigid_motion: bool,
    skin_motion: bool,
    camera_motion: bool,
    reflections: bool,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            azimuth: 225.,
            elevation: 48.,
            intensity: 5.,
            sky: 0.18,
            emission: 0.0,
            point_intensity: 1.0,
            color: [1.0, 0.94, 0.82],
            exposure: 0.,
            animate: true,
            mode: 0,
            bounces: true,
            rigid_motion: true,
            skin_motion: true,
            camera_motion: true,
            reflections: true,
        }
    }
}

impl Controls {
    fn for_scene(intel: bool, gallery: bool) -> Self {
        let mut controls = Self::default();
        if intel {
            controls.skin_motion = false;
            controls.camera_motion = false;
            controls.animate = false;
            controls.azimuth = 160.;
            controls.elevation = 70.;
            controls.exposure = 0.7;
        }
        if gallery {
            controls.intensity = 1.8;
            controls.sky = 0.12;
            controls.emission = 2.;
            controls.exposure = 0.;
        }
        controls
    }
    fn lighting(&self) -> Lighting {
        let a = self.azimuth.to_radians();
        let e = self.elevation.to_radians();
        Lighting {
            sun_dir: Vec3::new(a.cos() * e.cos(), a.sin() * e.cos(), e.sin()),
            sun_color: Vec3::new(self.color[0], self.color[1], self.color[2]) * self.intensity,
            environment_scale: self.sky,
            emission_scale: self.emission,
        }
    }
    fn emissive_preset(&mut self, night: bool, emission: f32) {
        self.intensity = 0.;
        self.animate = false;
        self.sky = if night { 0. } else { 0.18 };
        self.emission = emission;
        self.exposure = 0.7;
    }
}

fn main() {
    chandra::lab_session::supervise();
    let args: Vec<_> = std::env::args().collect();
    let string_option = |name: &str| args.iter().position(|a| a == name)
        .map(|i| args.get(i+1).expect("Missing option value").as_str()).unwrap_or("");
    let camera_track=string_option("--quality-camera-track");
    let light_track=string_option("--quality-light-track");
    assert!(["","orbit","translate","rotate","combined","stop","return"].contains(&camera_track));
    assert!(["","sun","fast","drag","points","oscillate","step"].contains(&light_track));
    let split_single_frame = args.iter().any(|a| a == "--split-single-frame");
    let single_frame = split_single_frame || args.iter().any(|a| a == "--single-frame");
    let integer_option = |name: &str, default: u32| args.iter().position(|a| a == name)
        .map(|i| args.get(i + 1).expect("Missing option value").parse::<u32>().expect("Invalid integer option"))
        .unwrap_or(default);
    let path_depth = integer_option("--path-depth", if split_single_frame { 1 } else { 8 });
    let samples = integer_option("--samples", 1);
    let diffuse_samples=integer_option("--diffuse-samples",0);
    let use_diffuse_filter=!single_frame && !args.iter().any(|a|a=="--no-diffuse-filter");
    let quality_fps=integer_option("--quality-fps",60);
    assert!(diffuse_samples<=64 && [20,30,60,120].contains(&quality_fps));
    let min_probe_rays = integer_option("--min-probe-rays", 0);
    let roam_test = args.iter().any(|a| a == "--roam-test");
    let quality_check = roam_test || args.iter().any(|a| a == "--quality-check");
    let quality_frames = integer_option("--quality-frames", if roam_test { integer_option("--roam-frames", 2400) } else { 200 });
    let quality_tail = integer_option("--quality-tail", 0);
    let quality_stop = integer_option("--quality-stop", u32::MAX);
    let quality_light_off = integer_option("--quality-light-off", u32::MAX);
    let probe_budget = integer_option("--probe-budget", 0);
    assert!(probe_budget == 0 || (probe_budget >= 256 && probe_budget.is_power_of_two()));
    let camera_preset = integer_option("--view", 0) as usize;
    let camera_dolly = args.iter().position(|a| a == "--camera-dolly")
        .map(|i| args.get(i + 1).expect("--camera-dolly needs a distance").parse::<f32>().expect("Invalid camera distance"))
        .unwrap_or(0.0);
    assert!(camera_dolly.is_finite());
    let quality_motion = args.iter().any(|a| a == "--quality-motion");
    let frame_cache = args.iter().any(|a| a == "--frame-cache");
    assert!(!split_single_frame || (path_depth == 1 && samples == 1 && min_probe_rays == 0 && !frame_cache),
        "--split-single-frame uses the original one-ray, single-bounce transport; extra paths, samples and frame caches are disabled");
    assert!(min_probe_rays <= 256 && (min_probe_rays == 0 || (single_frame && !frame_cache)));
    assert!(min_probe_rays == 0 || !args.iter().any(|a| a == "--unfused"), "Weighted adaptive rays require fused deposit");
    assert!(!frame_cache || (single_frame && samples == 1 && path_depth >= 2 && !args.iter().any(|a| a == "--unfused")), "--frame-cache needs --single-frame, depth >= 2, samples=1 and fused deposit");
    assert!(camera_preset < if args.iter().any(|a| a == "--white-room" || a == "--pica-pica") { 4 } else { 3 });
    assert!(quality_frames <= 36000);
    assert!((1..=16).contains(&path_depth) && (1..=64).contains(&samples));
    assert!(single_frame || samples == 1, "--samples requires --single-frame");
    if args.iter().any(|a| a == "--help") {
        println!("Near-view diagnostics:\n  --camera-dolly N Move forward from a camera preset in scene units\n  --quality-fps 20|30|60|120 Set deterministic camera/light step and history clock\n  --profile-realtime Use measured dt for history, with identical replayed poses\n  --exact-diffuse-feedback Use per-ray angular quadrature instead of the PICA normal atlas\n  --diffuse-registers N Primary trace register limit (PICA default 64)");
        println!("PICA PICA / cached reflections:\n  --specular-quality Use the original full-resolution adaptive GGX path\n  --reflection-scale N Reflection downsample factor 1/2/4 (default 2)\n  --rc-update-samples N Maximum adaptive RC update rays per pixel (default 8 cached / 64 quality)\n  --pica-pica Import external/PicaPica/scene.gltf (tools/setup-pica-pica.ps1)\n  --specular-samples N Samples for --specular-quality (default 4; 128 selects quality)\n  --no-specular Disable indirect reflection display\n  --specular-only Show indirect reflections\n  --specular-reference Trace the first reflection segment without RC substitution\n  --no-specular-filter Disable reflection reconstruction for comparison");
        println!("Stable diffuse GI:\n  --quality-camera-track orbit|translate|rotate|combined|stop|return\n  --quality-light-track sun|drag|points|fast|step|oscillate\n  --quality-fps 20|30|60|120   Deterministic simulation rate\n  --diffuse-diagnostics Export raw/spatial/temporal irradiance and sample counts\n  --no-diffuse-filter   Raw RC comparison; disables adaptive reconstruction\n  --diffuse-samples N   Fixed primary sample budget for diagnostics\n  --interaction-log PATH Record real input and camera state as JSON Lines\n  Tab or F1: panel; F or held right mouse: unlimited look; Q/E: down/up; Shift/Ctrl: fast/slow");
        println!("Diffuse GI diagnostics:\n  --white-room   Sealed shell, 8 cm slit, 2/12 cm partitions and roof corner; --view 0..3\n  --geometry-only Keep camera and light motion off\n  --single-bounce Disable secondary diffuse bounces\n  --probe-budget N Power-of-two base capacity, minimum 256\n  --quality-tail N Export the last N consecutive linear frames and same-surface samples\n  --quality-stop N Stop all motion at frame N\n  --quality-stop-clear Clear transport once at that stopped pose\n  --quality-light-off N Turn off light energy at frame N\n  --roam-test --roam-frames N Deterministic camera/capacity diagnostic\n  --scene-worker Bypass the scene supervisor\n  F: mouse look; Q/E: down/up; Shift/Ctrl: fast/slow; F1: hide panel");
        println!("Split RC baseline:\n  --split-single-frame  Original primary trace/split path, one indirect bounce, cleared each frame; no temporal jitter or extra path tracing");
        println!("Experimental transport:\n  --single-frame  Clear GI transport each frame; no prior-frame hits or radiance\n  --path-depth N  Frame-local path depth (default 8, experimental)\n  --samples N     Samples per pixel within the same frame (default 1)\n  --min-probe-rays N  Fill undersampled probes within this frame (default 0)\n  --frame-cache   Experimental current-frame secondary spatial cache\n  --unbatched-skin  Reference separate skeleton dispatches and BLASes\n  --no-light-grid Reference full local-light loop\n  --view N        Camera preset 0..2\n  --indirect-only Show indirect light\n  --quality-check Export current-frame linear RGBA32F; alpha is surface mask\n  --quality-frames N End quality check at frame N (default 200)\n  --quality-motion Keep animations on during the quality check");
        println!(
            "Chandra lighting lab\n  --sponza       Load assets/sponza.obj\n  --intel-sponza Load an external Intel Sponza asset (--scene-path required)\n  --scene-path   Override the Intel Sponza glTF path\n  --smoke-test   Verify two light directions and save PNGs, then exit\n  --benchmark    Deterministic scenarios and convergence checks\n  --stress-test  3600 frames with rigid, skin, sun and camera motion\n  --profile      Collect CUDA event timings after 200 warmup frames\n  --output DIR   Benchmark, stress or profiler output directory\n  --unfused      Use the separate ray-deposit queue for comparison\n  --static       Start with all animations paused\n  --1080p        Render at 1920 x 1080\n  --small        Render at 960 x 600 (default: 1280 x 800)\n\nRendering is uncapped. Mouse sliders change sunlight live. WASD moves; arrows/right drag look; Q/E move vertically; Shift/Ctrl select fast/slow speed. R resets the camera."
        );
        println!(
            "\nAdditional scene/profile options:\n  --cubes N           Intel courtyard cube count, 1..100 (default 1)\n  --rigid-only        Keep sunlight and camera stationary\n  --rigid-refit       Reference per-object vertex and BLAS updates\n  --profile-frames N  Measured frames after warmup (default 600)"
        );
        println!(
            "\nProcedural scenes:\n  --gallery       Chromatic Gallery, 30 lights + 30 rigid + 30 emissive + 30 skinned (default)\n  --courtyard     Small legacy procedural scene"
        );
        println!(
            "\nEmissive options (gallery / Intel Sponza):\n  --emissive-day    Sun off, sky 0.18, colored cube emission\n  --emissive-night  Sun and sky off, colored cube emission\n  --emission N      Linear emission multiplier (default 4)"
        );
        return;
    }
    let sponza = args.iter().any(|a| a == "--sponza");
    let intel = args.iter().any(|a| a == "--intel-sponza");
    let courtyard = args.iter().any(|a| a == "--courtyard");
    let white_room = args.iter().any(|a| a == "--white-room");
    let pica = args.iter().any(|a| a == "--pica-pica");
    let specular_samples=integer_option("--specular-samples",4);
    assert!((1..=128).contains(&specular_samples));
    let specular_reference=args.iter().any(|a|a=="--specular-reference");
    let bounded_reflections=pica && !args.iter().any(|a|a=="--specular-quality") && specular_samples<128 && !specular_reference;
    let rc_update_samples=integer_option("--rc-update-samples",if bounded_reflections {8}else{64});
    assert!((1..=64).contains(&rc_update_samples));
    let pica_path=std::path::PathBuf::from("external/PicaPica/scene.gltf");
    assert!([sponza, intel, courtyard, white_room, pica, args.iter().any(|a| a == "--gallery")].into_iter().filter(|v| *v).count() <= 1, "Choose one scene");
    let gallery = !sponza && !intel && !courtyard && !white_room && !pica;
    use chandra::lab_session::Scene;
    let active_scene = if pica {Scene::PicaPica} else if white_room { Scene::WhiteRoom } else if courtyard { Scene::Courtyard }
        else if intel { Scene::IntelSponza } else if sponza { Scene::Sponza } else { Scene::Gallery };
    assert!(
        !(args.iter().any(|a| a == "--gallery") && !gallery),
        "Choose one scene"
    );
    let emissive_day = args.iter().any(|a| a == "--emissive-day");
    let emissive_night = args.iter().any(|a| a == "--emissive-night");
    assert!(!(emissive_day && emissive_night), "Choose day or night");
    let emissive_cubes = emissive_day || emissive_night;
    assert!(
        !emissive_cubes || intel || gallery,
        "Emissive presets require the gallery or Intel Sponza"
    );
    let emission_strength = args
        .iter()
        .position(|a| a == "--emission")
        .map(|i| {
            args.get(i + 1)
                .expect("--emission needs a value")
                .parse::<f32>()
                .expect("Invalid emission")
        })
        .unwrap_or(4.0);
    assert!(emission_strength.is_finite() && (0.0..=100.0).contains(&emission_strength));
    let cube_count = args
        .iter()
        .position(|a| a == "--cubes")
        .map(|i| {
            args.get(i + 1)
                .expect("--cubes needs a count")
                .parse::<usize>()
                .expect("Invalid cube count")
        })
        .unwrap_or(1);
    assert!((1..=100).contains(&cube_count), "Use 1 to 100 cubes");
    assert!(intel || cube_count == 1, "--cubes requires --intel-sponza");
    let rigid_only = args.iter().any(|a| a == "--rigid-only");
    let rigid_refit = args.iter().any(|a| a == "--rigid-refit");
    assert!(
        !(gallery && rigid_refit),
        "--rigid-refit is available for --courtyard or --intel-sponza"
    );
    let profile_frames = args
        .iter()
        .position(|a| a == "--profile-frames")
        .map(|i| {
            args.get(i + 1)
                .expect("--profile-frames needs a count")
                .parse::<u32>()
                .expect("Invalid frame count")
        })
        .unwrap_or(600);
    assert!(profile_frames > 0 && profile_frames <= 36000);
    let intel_path = args
        .iter()
        .position(|a| a == "--scene-path")
        .map(|i| std::path::PathBuf::from(args.get(i + 1).expect("--scene-path needs a glTF path")))
        .or_else(|| std::env::var_os("CHANDRA_INTEL_SPONZA").map(std::path::PathBuf::from))
        .unwrap_or_else(|| {
            std::path::PathBuf::from("external/IntelSponza/NewSponza_Main_glTF_003.gltf")
        });
    assert!(
        !(sponza && intel),
        "Select either legacy Sponza or Intel Sponza"
    );
    if intel {
        assert!(
            intel_path.exists(),
            "Intel Sponza missing: {}. Use --scene-path to select its glTF.",
            intel_path.display()
        );
    }
    let smoke = args.iter().any(|a| a == "--smoke-test");
    let benchmarking = args.iter().any(|a| a == "--benchmark");
    let stressing = args.iter().any(|a| a == "--stress-test");
    let profiling = args.iter().any(|a| a == "--profile");
    let mut output_dir = args
        .iter()
        .position(|a| a == "--output")
        .map(|i| args.get(i + 1).expect("--output needs a directory").clone());
    assert!(
        [benchmarking, stressing, profiling, smoke]
            .iter()
            .filter(|&&enabled| enabled)
            .count()
            <= 1,
        "Choose only one of --benchmark, --stress-test, --profile or --smoke-test"
    );
    let size = if args.iter().any(|a| a == "--quality-width") {
        [integer_option("--quality-width",480),integer_option("--quality-height",300)]
    } else if args.iter().any(|a| a == "--1080p") {
        [1920, 1080]
    } else if args.iter().any(|a| a == "--small") {
        [960, 600]
    } else {
        [1280, 800]
    };
    if (cube_count > 1 || emissive_cubes || gallery) && output_dir.is_none() {
        let mode = if benchmarking {
            "benchmark"
        } else if stressing {
            "stress"
        } else {
            "profile"
        };
        let preset = if emissive_night {
            "-emissive-night"
        } else if emissive_day {
            "-emissive-day"
        } else {
            ""
        };
        let scene_name = if gallery {
            "gallery-30x4".to_string()
        } else {
            format!("intel-{cube_count}-cubes")
        };
        output_dir = Some(format!(
            ".local/{scene_name}{preset}-{mode}-{}x{}",
            size[0], size[1]
        ));
    }
    if sponza && !std::path::Path::new("assets/sponza.obj").exists() {
        eprintln!("Missing assets/sponza.obj. Run without --sponza for the procedural gallery.");
        std::process::exit(1);
    }
    let app = App::new("Chandra | Interactive Lighting Lab", size)
        .agx()
        .init();
    yesod::printer::global::set_capacity(1024 * 1024);
    let lab_scene = if pica {scene::LabScene{world:pica_scene::load(&pica_path),animation:None}}
        else if white_room { scene::LabScene { world: chandra::white_room::world(), animation: None } } else { scene::world(
        sponza,
        intel.then_some(intel_path.as_path()),
        cube_count,
        rigid_refit,
        emissive_cubes,
        gallery,
        args.iter().any(|a| a == "--unbatched-skin"),
    ) };
    println!(
        "Rigid update path: {}",
        if rigid_refit {
            "per-mesh vertex transform + BLAS refit (reference)"
        } else {
            "batched vertices + transform-only TLAS update; immutable rigid BLASes"
        }
    );
    let mut animation = lab_scene.animation;
    let mut core = CoreData::new(size, lab_scene.world);
    if gallery {
        core.set_point_lights_grid(&chandra::gallery::lights(0., 1.), !args.iter().any(|a| a == "--no-light-grid"));
    }
    let mut rc = RcData::new(RcConfig {
        base_grid_size: if pica {0.5} else if sponza || intel || gallery { 10. } else { 6. },
        // The gallery exposes more coarse-cascade cells than the narrow Intel
        // courtyard view. Reserve headroom without reducing spatial resolution.
        base_probes: if probe_budget > 0 { probe_budget } else if gallery { 1 << 17 } else { 1 << 16 },
        quality_decay: if single_frame { 1.0 } else { 0.90 },
        ..Default::default()
    });
    let adaptive_diffuse=use_diffuse_filter && !args.iter().any(|a|a=="--unfused");
    let ray_capacity = ((size[0] as u64 * size[1] as u64 + rc.cache.base_probes as u64 * min_probe_rays as u64) * 3 * if single_frame { samples } else { diffuse_samples.max(if adaptive_diffuse {64}else{4}) } as u64) as usize;
    let mut trace = if args.iter().any(|a| a == "--unfused") {
        TraceData::new(&core, &rc, ray_capacity)
    } else {
        TraceData::new_fused(&core, &rc, ray_capacity)
    };
    trace.set_primary_register_limit(integer_option("--diffuse-registers", if pica {64} else {0}));
    if single_frame && !split_single_frame { trace.enable_stable_samples(&core, &rc); }
    if min_probe_rays > 0 { trace.enable_probe_budget(&core, &rc, min_probe_rays); }
    if frame_cache { trace.enable_frame_cache(&core); }
    if adaptive_diffuse && !single_frame && min_probe_rays==0 && !args.iter().any(|a|a=="--diffuse-serial") {
        trace.enable_diffuse_work_with_limit(&core,rc_update_samples.max(diffuse_samples));
    }
    let splat = SplatData::new(&core, &rc, 1.0);
    let cached_diffuse_feedback = pica && !single_frame && !args.iter().any(|a| a == "--exact-diffuse-feedback");
    let diffuse_reconstruction=use_diffuse_filter.then(||chandra::algorithm::diffuse::DiffuseData::with_sample_limit(&core,rc_update_samples));
    let reflection_scale=integer_option("--reflection-scale",2);
    let cached_specular=bounded_reflections.then(||chandra::algorithm::cached_specular::CachedSpecular::new(&core,&rc,reflection_scale));
    let specular=(core.world.has_pbr() && !bounded_reflections).then(||chandra::algorithm::specular::SpecularData::with_trace_options(&core,!args.iter().any(|a|a=="--specular-serial"),integer_option("--specular-batch",if core.num_pixels()<=1920*1080 {1} else {32}),integer_option("--specular-registers",96)));
    let reflection_buffers=if let Some(data)=&cached_specular {Some((&data.raw,&data.output,&data.diagnostics))}
        else {specular.as_ref().map(|data|(&data.raw,&data.output,&data.diagnostics))};
    let history_regions = animation.as_ref().map_or(16, |a| a.moving_meshes().max(16));
    let history = DynamicHistory::with_region_capacity(&rc, &trace, history_regions);
    let mut lighting_response = LightingResponse::default();
    let mut manual_lighting_response = LightingResponse::default();
    let mut gi_availability = GiAvailability::default();
    let mut capacity_recoveries = 0u32;
    let mut skipped_gi_frames = 0u32;
    let poison_transport = (quality_check && single_frame).then(|| DEVICE.create_kernel::<fn()>(&track!(|| {
        let i = dispatch_id().x;
        // A conspicuous finite poison also works with backends that reject
        // NaN literals during shader code generation.
        rc.cache.weights.write(i, 65536.0f32);
        rc.cache.weighted_fluence.write(i, Vec4::splat(1e12f32));
        rc.base_radiance.write(i, Vec3::splat(1e12f32));
        if i < rc.radiance.len() as u32 { rc.radiance.write(i, Vec3::splat(1e12f32)); }
        if i < rc.probe_quality.len() as u32 { rc.probe_quality.write(i, 65536.0f32); }
        if i < trace.traced_hits.len() as u32 { trace.traced_hits.write(i, chandra::algorithm::trace::Hit::invalid()); }
    })));
    println!("Dynamic history capacity: {history_regions} independent swept regions");
    let invalid_pixels = Singleton::new(0u32);
    let linear_output = quality_check.then(|| DEVICE.create_buffer::<Vec4<f32>>(core.num_pixels() as usize));
    let base_output=(quality_check && bounded_reflections).then(|| DEVICE.create_buffer::<Vec4<f32>>(core.num_pixels() as usize));
    let surface_samples = (quality_check && courtyard).then(|| chandra::lab_quality::SurfaceSamples::courtyard(&core, linear_output.as_ref().unwrap()));
    let surface_buffers = (quality_check && args.iter().any(|a| a=="--quality-surfaces"))
        .then(|| chandra::lab_quality::SurfaceBuffers::new(&core));
    let draw = DEVICE.create_kernel::<fn(View, u32, f32, bool, bool)>(&track!(|view, mode, exposure, gi_available, reflections| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let direction = view.ray_dir(pixel.cast_f32() + 0.5).normalize();
        let depth = core.read_depth(pixel);
        if depth == 0.0 {
            let color = core.get_environment(direction) * exposure;
            if let Some(output) = &linear_output { output.write(core.pixel_to_linear(pixel), color.extend(0.0)); }
            if let Some(data)=&cached_specular {data.composition.write(core.pixel_to_linear(pixel),(color/exposure).extend(0.0));}
            if let Some(output)=&base_output {output.write(core.pixel_to_linear(pixel),color.extend(0.0));}
            app.set_pixel(pixel.cast_i32(), color);
            return;
        }
        let index = core.pixel_to_linear(pixel);
        let hit = core.hit.read(index);
        let diffuse = core.world.get_diffuse(hit);
        let direct = core.get_surface_irradiance(
            None,
            core.hit_position(view.pos, direction, hit),
            hit,
            direction,
        ) * diffuse
            + core.get_surface_emission(hit)
            + if core.world.has_pbr(){core.direct_specular(core.get_world_pos(pixel),hit,-direction)}else{Vec3::splat_expr(0.0)};
        let indirect = if gi_available {
            let irradiance=if let Some(data)=&diffuse_reconstruction {data.output.read(index).xyz()} else {splat.get_radiance(index)};
            irradiance * diffuse
        } else { Vec3::splat_expr(0.0f32) };
        let reflected=if let Some((_,output,_))=&reflection_buffers {
            if reflections {output.read(index).xyz()}else{Vec3::splat_expr(0.0)}
        }else{Vec3::splat_expr(0.0)};
        let color = if mode == 1 {
            direct
        } else if mode == 2 {
            indirect+reflected
        } else if mode == 3 {
            core.read_normal(pixel) * 0.5 + 0.5
        } else if mode == 4 {reflected} else {
            direct + indirect + reflected
        };
        if !color.is_finite().all() {
            invalid_pixels.atomic().fetch_add(1u32);
        }
        if let Some(output) = &linear_output { output.write(index, (color * exposure).extend(1.0)); }
        if let Some(output)=&base_output {output.write(index,((direct+indirect)*exposure).extend(1.0));}
        if let Some(data)=&cached_specular {data.composition.write(index,color.extend(1.0));}
        app.set_pixel(pixel.cast_i32(), color * exposure);
    }));
    let present_cached=cached_specular.as_ref().map(|data|DEVICE.create_kernel::<fn(f32)>(&track!(|exposure| {
        let p=dispatch_id().xy();let i=core.pixel_to_linear(p);let value=data.composition.read(i);
        if !value.is_finite().all(){invalid_pixels.atomic().fetch_add(1u32);}
        if let Some(output)=&linear_output {output.write(i,(value.xyz()*exposure).extend(value.w));}
        app.set_pixel(p.cast_i32(),value.xyz()*exposure);
    })));

    let ui_context = egui::Context::default();
    ui_context.set_visuals(egui::Visuals::dark());
    let mut ui_renderer = EguiSoftwareRender::new(ColorFieldOrder::Rgba);
    let mut ui_pixels = vec![[0u8; 4]; (size[0] * size[1]) as usize];
    let ui_texture = DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Byte4, size[0], size[1], 1);
    let overlay = app.overlay();
    let composite_ui = DEVICE.create_kernel::<fn()>(&track!(|| {
        let p = dispatch_id().xy();
        let rgba = ui_texture.read(p);
        // egui colors are premultiplied; the testbed expects straight alpha.
        let rgb = rgba.xyz() / keter::max(rgba.w, 0.001);
        overlay.write(p, rgb.extend(rgba.w));
    }));
    let mut controls = Controls::for_scene(intel, gallery);
    if pica {controls.azimuth=230.;controls.elevation=55.;controls.intensity=3.;controls.sky=0.6;controls.emission=1.;controls.animate=false;controls.camera_motion=false;}
    controls.reflections=!args.iter().any(|a|a=="--no-specular");
    if args.iter().any(|a|a=="--specular-only"){controls.mode=4;}
    if split_single_frame { controls.bounces = false; }
    if args.iter().any(|a| a == "--single-bounce") { controls.bounces = false; }
    if args.iter().any(|a| a == "--indirect-only") { controls.mode = 2; }
    if emissive_cubes {
        controls.emissive_preset(emissive_night, emission_strength);
    }
    if smoke || (quality_check && !quality_motion) || args.iter().any(|a| a == "--static") {
        controls.animate = false;
        controls.rigid_motion = false;
        controls.skin_motion = false;
        controls.camera_motion = false;
    }
    if sponza {
        controls.rigid_motion = false;
        controls.skin_motion = false;
        controls.camera_motion = false;
    }
    let mut camera_time = 0.0f32;
    let mut light_time = 0.0f32;
    let make_camera = |index| {
        let mut camera = if pica {pica_scene::camera(index,size)}else if white_room { chandra::white_room::camera(index, size) }
            else { scene::camera(index, size, sponza, intel, gallery) };
        camera.pos += camera.forward() * camera_dolly;
        camera
    };
    let mut camera = make_camera(camera_preset);
    let mut previous_diffuse_camera=camera;
    let mut diffuse_history_valid=false;
    let mut specular_history_valid=false;
    let mut switch_scene = None;
    let mut show_ui = true;
    let mut look_mode = false;
    let mut camera_pointer = chandra::lab_camera::Pointer::default();
    let mut mouse_capture=chandra::lab_camera::MouseCapture::default();
    let mut panel_key=chandra::lab_camera::KeyLatch::default();
    let mut look_key=chandra::lab_camera::KeyLatch::default();
    let interaction_path=string_option("--interaction-log");
    let mut interaction_log=if interaction_path.is_empty() {None} else {
        use std::io::Write;
        let path=std::path::Path::new(interaction_path);
        if let Some(parent)=path.parent().filter(|p|!p.as_os_str().is_empty()){std::fs::create_dir_all(parent).unwrap();}
        let mut file=std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        writeln!(file,"{{\"event\":\"start\",\"pid\":{},\"scene\":\"{}\"}}",std::process::id(),active_scene.name()).unwrap();Some(file)
    };
    let mut last_pointer = egui::Pos2::ZERO;
    let mut last_left_down = false;
    let start = Instant::now();
    let mut first_test_image: Option<Vec<[u8; 4]>> = None;
    let mut needs_reset = true;
    let mut previous_interaction_state=(false,true,false);
    let mut benchmark = benchmarking.then(|| {
        benchmark::Benchmark::new(size, intel, emissive_cubes, gallery, output_dir.clone())
    });
    let mut profiler = gpu_profile::GpuProfiler::new(profiling);
    let debug_sync = std::env::var_os("CHANDRA_GPU_DEBUG_SYNC").is_some();
    let mut stress = stressing.then(|| {
        benchmark::StressTest::new(
            if gallery {
                "gallery-30x4"
            } else if intel {
                "intel-sponza"
            } else {
                "courtyard"
            },
            size,
            output_dir.clone(),
        )
    });
    if (stressing || profiling) && !quality_check && !args.iter().any(|a| a == "--static") {
        controls.rigid_motion = true;
        controls.skin_motion = !intel;
        controls.animate = true;
        controls.camera_motion = true;
    }
    if rigid_only {
        controls.rigid_motion = true;
        controls.skin_motion = false;
        controls.animate = false;
        controls.camera_motion = false;
    }
    if emissive_cubes {
        controls.animate = false;
    }
    let lighting_metadata = format!(
        "emissive-materials={}; preset={}; sun={}; sky={}; emission={}; exposure_ev={}; local_light_strength={}",
        emissive_cubes || gallery,
        if emissive_night {
            "night"
        } else if emissive_day {
            "day"
        } else {
            "default"
        },
        controls.intensity,
        controls.sky,
        controls.emission,
        controls.exposure,
        if gallery {
            controls.point_intensity
        } else {
            0.
        }
    );
    println!("Lighting setup: {lighting_metadata}");
    let transport_metadata = format!("single_frame={single_frame}; split_single_frame={split_single_frame}; frame_cache={frame_cache}; path_depth={path_depth}; samples={samples}; min_probe_rays={min_probe_rays}; batched_skin={}; light_grid={}; camera_preset={camera_preset}; quality_motion={quality_motion}; pica={pica}; specular_samples={specular_samples}; specular_reference={specular_reference}; bounded_reflections={bounded_reflections}; reflection_scale={reflection_scale}; rc_update_samples={rc_update_samples}", !args.iter().any(|a| a == "--unbatched-skin"), !args.iter().any(|a| a == "--no-light-grid"));
    println!("Transport: {transport_metadata}");
    let scene_counts = if pica {
        "PICA PICA: 145 source objects; 170 material meshes; 76300 triangles; 28 materials".to_string()
    } else if gallery {
        "30 point lights; 30 rigid; 30 emissive; 30 skinned".to_string()
    } else if intel {
        format!("{cube_count} cubes")
    } else {
        "courtyard: 1 rigid; 1 skinned".to_string()
    };
    if (benchmarking || stressing || profiling)
        && let Some(directory) = &output_dir
    {
        std::fs::create_dir_all(directory).unwrap();
        std::fs::write(format!("{directory}/configuration.txt"), format!(
            "{}x{}; {scene_counts}; rigid-only={rigid_only}; rigid-refit={rigid_refit}; {history_regions} history regions; multiple bounces={}; uncapped; simulation dt=1/60s; {lighting_metadata}; {transport_metadata}\n",
            size[0], size[1], controls.bounces)).unwrap();
    }
    println!(
        "Scene ready: {}. First frame compiles the GPU kernels.",
        if pica {"PICA PICA / 145 objects / 170 material meshes / 76300 triangles"}
        else if gallery {
            "Chromatic Gallery / 30 lights + 30 rigid + 30 emissive + 30 skinned"
        } else if intel {
            "Intel New Sponza / moving cubes"
        } else if sponza {
            "Sponza"
        } else if white_room {
            "White light-leak lab / four fixtures"
        } else {
            "built-in courtyard"
        }
    );

    if white_room {
        controls.animate = false; controls.camera_motion = false;
        controls.rigid_motion = false; controls.skin_motion = false;
    }
    let interactive=!(quality_check || benchmarking || stressing || profiling || smoke);
    app.run(|rt| {
        // Automated captures must never consume physical input while opening
        // a window. Otherwise a short key tap changes only one reference pose.
        let focused = interactive && chandra::lab_camera::focused();
        if !focused { rt.keys_down.clear(); rt.buttons_down.clear(); rt.keys_pressed.clear(); rt.buttons_pressed.clear(); look_mode = false; }
        if panel_key.update(rt.key_pressed(KeyCode::F1)||rt.key_pressed(KeyCode::Tab),rt.key_down(KeyCode::F1)||rt.key_down(KeyCode::Tab),focused)
            && !ui_context.egui_wants_keyboard_input() { show_ui = !show_ui; }
        if look_key.update(rt.key_pressed(KeyCode::KeyF),rt.key_down(KeyCode::KeyF),focused) { look_mode = !look_mode; }
        let dt = if (benchmarking || stressing || profiling || quality_check)
            && !args.iter().any(|a| a == "--profile-realtime") {
            1.0 / quality_fps as f32
        } else { rt.frame_time().clamp(0.001, 0.05) };
        if let Some(stress) = &mut stress { stress.begin_frame(rt.tick); }
        if let Some(benchmark) = &mut benchmark {
            let [rigid, skin, sun, orbit] = benchmark.begin_frame(rt.tick);
            controls.rigid_motion = rigid; controls.skin_motion = skin;
            controls.animate = sun && (!emissive_cubes || gallery); controls.camera_motion = orbit;
        }
        if args.iter().any(|a| a == "--geometry-only") {
            controls.animate = false;
            controls.camera_motion = false;
        }
        if args.iter().any(|a|a=="--quality-geometry-motion") {
            controls.rigid_motion=rt.tick>=120 && rt.tick<quality_stop;
            controls.skin_motion=controls.rigid_motion;
        }
        if rt.tick >= quality_stop {
            controls.animate = false; controls.rigid_motion = false;
            controls.skin_motion = false; controls.camera_motion = false;
            if rt.tick == quality_stop && args.iter().any(|a| a == "--quality-stop-clear") { needs_reset = true; }
        }
        let previous = controls.clone();
        let mut reset_camera = None;
        let mut events = Vec::new();
        let pos = egui::pos2(rt.cursor_position.x, rt.cursor_position.y);
        let valid_pointer = pos.x.is_finite() && pos.y.is_finite();
        let mouse_delta = camera_pointer.update((valid_pointer && focused).then_some(glam::Vec2::new(pos.x, pos.y)));
        let left_down = rt.button_down(MouseButton::Left);
        if valid_pointer {
            events.push(egui::Event::PointerMoved(pos));
        } else {
            events.push(egui::Event::PointerGone);
        }
        let left_pressed = rt.button_pressed(MouseButton::Left);
        if left_pressed || (left_down && !last_left_down) {
            events.push(egui::Event::PointerButton {
                pos: if valid_pointer { pos } else { last_pointer },
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::default(),
            });
        }
        if !left_down && (last_left_down || left_pressed) {
            events.push(egui::Event::PointerButton {
                pos: if valid_pointer { pos } else { last_pointer },
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            });
        }
        // The testbed exposes physical keys rather than text events. Forward the
        // numeric-editing subset so slider values can also be typed directly.
        for &(key, egui_key, text) in &[
            (KeyCode::Digit0, egui::Key::Num0, "0"), (KeyCode::Digit1, egui::Key::Num1, "1"),
            (KeyCode::Digit2, egui::Key::Num2, "2"), (KeyCode::Digit3, egui::Key::Num3, "3"),
            (KeyCode::Digit4, egui::Key::Num4, "4"), (KeyCode::Digit5, egui::Key::Num5, "5"),
            (KeyCode::Digit6, egui::Key::Num6, "6"), (KeyCode::Digit7, egui::Key::Num7, "7"),
            (KeyCode::Digit8, egui::Key::Num8, "8"), (KeyCode::Digit9, egui::Key::Num9, "9"),
            (KeyCode::Period, egui::Key::Period, "."), (KeyCode::Minus, egui::Key::Minus, "-"),
            (KeyCode::Enter, egui::Key::Enter, ""), (KeyCode::Backspace, egui::Key::Backspace, ""),
            (KeyCode::Escape, egui::Key::Escape, ""), (KeyCode::Tab, egui::Key::Tab, ""),
        ] {
            if rt.key_pressed(key) {
                events.push(egui::Event::Key { key: egui_key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::default() });
                events.push(egui::Event::Key { key: egui_key, physical_key: None, pressed: false, repeat: false, modifiers: egui::Modifiers::default() });
                if !text.is_empty() { events.push(egui::Event::Text(text.into())); }
            }
        }
        last_left_down = left_down;
        let mut controls_rect=egui::Rect::NOTHING;
        let output = ui_context.run_ui(egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0] as f32, size[1] as f32))),
            time: Some(start.elapsed().as_secs_f64()),
            predicted_dt: dt,
            events,
            ..Default::default()
        }, |ui| {
            if !show_ui {
                let area=egui::Area::new("show-controls".into()).fixed_pos(egui::pos2(6.,6.)).show(ui.ctx(),|ui| {
                    if ui.button("Show controls [Tab]").clicked(){show_ui=true;}
                });controls_rect=area.response.rect;return;
            }
            let panel=egui::Panel::left("lighting").exact_size(292.).resizable(false).show_inside(ui, |ui| {
              egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(8.);
                ui.heading("CHANDRA");
                if ui.small_button("Hide controls [Tab / F1]").clicked(){show_ui=false;}
                ui.label("Split Radiance Cascades / live lab");
                egui::ComboBox::from_id_salt("scene").selected_text(active_scene.name()).show_ui(ui, |ui| {
                    for next in Scene::ALL {
                        let available = match next {
                            Scene::Sponza => std::path::Path::new("assets/sponza.obj").exists(),
                            Scene::IntelSponza => intel_path.exists(), Scene::PicaPica=>pica_path.exists(), _ => true,
                        };
                        if ui.add_enabled(available, egui::Button::new(next.name()).selected(next == active_scene)).clicked() && next != active_scene {
                            switch_scene = Some(next); rt.exit();
                        }
                    }
                });
                if intel { ui.label(format!("Intel Sponza / {cube_count} moving cubes")); }
                if gallery { ui.label("Chromatic Gallery / procedural"); ui.small("30 lights / 30 rigid / 30 emissive / 30 skinned"); }
                ui.label(format!("{:.1} FPS  |  {:.1} ms", rt.fps(), rt.average_frame_time * 1000.));
                ui.separator();
                ui.strong("SUNLIGHT");
                ui.add(egui::Slider::new(&mut controls.azimuth, 0.0..=360.).text("Azimuth").suffix(" deg"));
                ui.add(egui::Slider::new(&mut controls.elevation, 2.0..=88.).text("Elevation").suffix(" deg"));
                ui.add(egui::Slider::new(&mut controls.intensity, 0.0..=15.).text("Intensity"));
                ui.add(egui::Slider::new(&mut controls.sky, 0.0..=1.5).text("Sky fill"));
                ui.horizontal(|ui| { ui.label("Sun color"); ui.color_edit_button_rgb(&mut controls.color); });
                ui.checkbox(&mut controls.animate, if gallery { "Animate sun / local lights" } else { "Animate sun direction" });
                if gallery { ui.add(egui::Slider::new(&mut controls.point_intensity, 0.0..=3.).text("Local lights")); }
                ui.horizontal(|ui| {
                    if ui.button("Daylight").clicked() { controls = Controls::for_scene(intel, gallery); needs_reset = true; }
                    if ui.button("Sunset").clicked() {
                        controls.azimuth = 195.; controls.elevation = 12.; controls.intensity = 7.;
                        controls.color = [1., 0.48, 0.18]; controls.sky = 0.08; needs_reset = true;
                    }
                    if ui.button("Sky only").clicked() { controls.intensity = 0.; controls.sky = 0.8; needs_reset = true; }
                });
                if emissive_cubes || gallery {
                    ui.separator();
                    ui.strong("EMISSIVE OBJECTS");
                    ui.add(egui::Slider::new(&mut controls.emission, 0.0..=12.).text("Emission"));
                    ui.horizontal(|ui| {
                        if ui.button("Emissive day").clicked() { controls.emissive_preset(false, emission_strength); needs_reset = true; }
                        if ui.button("Emissive night").clicked() { controls.emissive_preset(true, emission_strength); needs_reset = true; }
                    });
                    ui.small("Presets: sun off; day sky 0.18 / night sky 0");
                }
                ui.separator();
                ui.strong("DYNAMIC SCENE");
                ui.add_enabled(animation.is_some(), egui::Checkbox::new(&mut controls.rigid_motion, if gallery { "Move rigid / emissive objects" } else { "Move / rotate rigid box" }));
                ui.add_enabled(animation.as_ref().is_some_and(|a| a.skin_vertices() > 0), egui::Checkbox::new(&mut controls.skin_motion, "Animate skinned mesh"));
                ui.add_enabled(!white_room, egui::Checkbox::new(&mut controls.camera_motion, "Orbit camera"));
                if ui.button("Pause all motion").clicked() {
                    controls.rigid_motion = false; controls.skin_motion = false;
                    controls.camera_motion = false; controls.animate = false;
                }
                if intel && cube_count == 1 { ui.small("Red cube: 1.5 m / 11 m travel / 12 s cycle"); }
                else if intel { ui.small(format!("{cube_count} colors / 45 cm cubes / independent motion")); }
                else if let Some(animation) = &animation { ui.small(format!("{} skinned vertices", animation.skin_vertices())); }
                ui.separator();
                ui.strong("COMPARE LIGHTING");
                ui.radio_value(&mut controls.mode, 0, "Full global illumination");
                ui.radio_value(&mut controls.mode, 1, "Direct light only");
                ui.radio_value(&mut controls.mode, 2, "Indirect light only");
                ui.radio_value(&mut controls.mode, 3, "Surface normals");
                if reflection_buffers.is_some(){
                    ui.checkbox(&mut controls.reflections,"Indirect reflections");
                    ui.radio_value(&mut controls.mode,4,"Reflections only");
                }
                if split_single_frame {
                    ui.label("Split RC: single frame / one indirect bounce");
                    ui.small("No accumulation / no extra path tracing");
                } else if single_frame {
                    ui.label(format!("Experimental single frame: {path_depth} path vertices, {samples} spp"));
                } else { ui.checkbox(&mut controls.bounces, "Multiple indirect bounces"); }
                ui.add(egui::Slider::new(&mut controls.exposure, -3.0..=3.).text("Exposure").suffix(" EV"));
                if ui.button(if single_frame { "Rebuild GI allocation" } else { "Clear GI history" }).clicked() { needs_reset = true; }
                ui.separator();
                ui.strong("CAMERA / TEST VIEWS");
                ui.horizontal(|ui| {
                    let labels: &[&str] = if white_room { &["Sealed", "Slit", "Thin walls", "Corner"] }
                        else if pica { &["Overview", "Right", "Left", "Reflections"] }
                        else { &["Overview", "Shadows", "Color bleed"] };
                    for (i, label) in labels.iter().enumerate() {
                        if ui.button(*label).clicked() { reset_camera = Some(i); }
                    }
                });
                ui.label("WASD: move  |  Arrows / right drag: look");
                ui.label("Q / E: down / up | Shift: fast | Ctrl: slow");
                ui.label("F: mouse look | F1: panel | R: reset | Esc: close");
                ui.add_space(8.);
                ui.small(if split_single_frame { "Original primary trace/split path. GI is rebuilt each frame. Fixed directional jitter; no temporal accumulation or secondary bounce cache." }
                    else if single_frame { "Frame-local GI. Fixed sampling removes temporal random jitter; spatial sampling error remains experimental." }
                    else { "Drag the sun sliders to move shadows. Switch to Indirect light to inspect color bleeding on white surfaces. GI settles over several frames after a change." });
                ui.add_space(8.);
                ui.small(format!("{} probes  /  {} cascades", rc.num_probes.iter().sum::<u32>(), rc.num_cascades));
              });
            });
            controls_rect=panel.response.rect;
        });

        // Deterministic energy edits exercise the same response and extra
        // sampling window as edits made through the sliders.
        // Replay identical poses per frame even when profiling the interactive
        // history clock. frame * variable_dt would make the camera jump whenever
        // performance changes, creating artificial disocclusions and stalls.
        let diagnostic_dt = 1.0 / quality_fps as f32;
        let diagnostic_time=chandra::lab_diagnostics::motion_time(rt.tick,diagnostic_dt,quality_stop);
        if light_track=="drag" { controls.azimuth=225.+18.*diagnostic_time; controls.animate=false; }
        if rt.tick >= quality_light_off {
            controls.intensity = 0.; controls.sky = 0.; controls.emission = 0.; controls.point_intensity = 0.;
        }
        let changed = |a: f32, b: f32| (a-b).abs() > 1e-5 * a.abs().max(b.abs()).max(1.0);
        let edited_direction = changed(controls.azimuth, previous.azimuth) || changed(controls.elevation, previous.elevation);
        let edited_energy = changed(controls.intensity, previous.intensity) || changed(controls.sky, previous.sky)
            || controls.color.iter().zip(previous.color).any(|(&a,b)| changed(a,b))
            || changed(controls.emission, previous.emission) || changed(controls.point_intensity, previous.point_intensity);
        let manual_response = manual_lighting_response.update(dt, edited_direction, edited_energy, controls.bounces);
        // run_ui's full-screen root is hoverable even over transparent pixels.
        // Only the actual panel/popup may block a viewport right-drag. An
        // explicit look request owns the pointer and releases stale UI focus.
        let over_controls=valid_pointer && controls_rect.contains(pos);
        let looking=chandra::lab_camera::wants_mouse_look(focused,look_mode,
            rt.button_down(MouseButton::Right),over_controls,mouse_capture.is_active());
        if looking {
            ui_context.memory_mut(|memory| {if let Some(id)=memory.focused(){memory.surrender_focus(id);}});
            controls.camera_motion=false;
        }
        let mut camera_mouse=glam::Vec2::ZERO;
        if !ui_context.egui_wants_keyboard_input() || looking {
            // A down/up pair can arrive between redraws; preserve a short tap.
            let held=|key|rt.key_down(key)||rt.key_pressed(key);
            let axis = |positive, negative| (held(positive) as u8 as f32) - (held(negative) as u8 as f32);
            let mouse=if looking {mouse_capture.update(true).unwrap_or(mouse_delta)*0.003} else {mouse_capture.release();glam::Vec2::ZERO};
            camera_mouse=mouse;
            let old_position=camera.pos;
            let moved = chandra::lab_camera::update(&mut camera, chandra::lab_camera::Input {
                forward: axis(KeyCode::KeyW,KeyCode::KeyS), right: axis(KeyCode::KeyD,KeyCode::KeyA),
                up: axis(KeyCode::KeyE,KeyCode::KeyQ) + held(KeyCode::Space) as u8 as f32,
                yaw: axis(KeyCode::ArrowRight,KeyCode::ArrowLeft)*1.3*dt + mouse.x,
                pitch: axis(KeyCode::ArrowDown,KeyCode::ArrowUp)*1.3*dt + mouse.y,
                fast: held(KeyCode::ShiftLeft)||held(KeyCode::ShiftRight), slow: held(KeyCode::ControlLeft)||held(KeyCode::ControlRight), focused,
            }, dt);
            if pica {camera.pos=old_position+(camera.pos-old_position)*0.04;}
            if moved { controls.camera_motion = false; }
            if rt.key_pressed(KeyCode::KeyR) { reset_camera = Some(0); }
        } else {mouse_capture.release();}
        if rt.key_pressed(KeyCode::Escape) { rt.exit(); }
        if valid_pointer { last_pointer = pos; }
        if let Some(index) = reset_camera { camera = make_camera(index); controls.camera_motion = false; diffuse_history_valid=false; }
        if roam_test {
            camera = make_camera((rt.tick / 600) as usize % if white_room { 4 } else { 3 });
            camera.pos.x += (rt.tick as f32 * 0.017).sin() * 60.;
            camera.yaw += (rt.tick as f32 * 0.011).sin() * 0.5;
        }
        if controls.camera_motion {
            camera_time += dt;
            let angle = -0.58 + 0.25 * (camera_time * 0.35).sin();
            camera.pos = if pica {glam::Vec3::new(70.*(camera_time*0.2).sin(),-70.*(camera_time*0.2).cos(),40.)}
                else if gallery { glam::Vec3::new(1050. + 160. * (camera_time * 0.19).sin(), -1950. + 130. * (camera_time * 0.23).sin(), 1100. + 100. * (camera_time * 0.21).sin()) }
                else if intel { glam::Vec3::new(1050. + 40. * (camera_time * 0.23).sin(), 60. * (camera_time * 0.31).sin(), 330. + 30. * (camera_time * 0.27).sin()) }
                else { glam::Vec3::new(-angle.sin() * 960., -angle.cos() * 960., 450. + 65. * (camera_time * 0.27).sin()) };
            let target = if pica {glam::Vec3::new(-4.,7.,5.)}else if gallery { glam::Vec3::new(-100., 150., 180.) } else if intel { glam::Vec3::new(-250., 0., 100.) } else { glam::Vec3::new(-40., 70., 85.) };
            let direction = (target - camera.pos).normalize();
            camera.yaw = direction.x.atan2(direction.y);
            camera.pitch = (-direction.z).asin();
        }
        if controls.animate { controls.azimuth = (controls.azimuth + dt * 18.).rem_euclid(360.); }
        if gallery && controls.animate { light_time += dt; }
        if !camera_track.is_empty() {
            let base=make_camera(camera_preset);
            camera=chandra::lab_diagnostics::camera(base,camera_track,rt.tick,diagnostic_dt,quality_stop);
            // The diorama is roughly 60 cm wide; courtyard translations are meters.
            if pica { camera.pos=base.pos+(camera.pos-base.pos)*0.2; }
            controls.camera_motion=false;
        }
        match light_track {
            "sun" | "fast" => { controls.azimuth=225.+diagnostic_time*if light_track=="fast" {120.} else {18.}; controls.animate=false; }
            "oscillate" => { controls.azimuth=225.+30.*(diagnostic_time*1.7).sin(); controls.elevation=48.+15.*(diagnostic_time*0.8).sin(); controls.animate=false; }
            "step" => { controls.azimuth=if rt.tick>=160 {90.} else {225.}; controls.animate=false; }
            "points" => { light_time=diagnostic_time; controls.animate=false; }
            _ => {}
        }
        if smoke && rt.tick == 81 { controls.azimuth = 100.; controls.elevation = 22.; needs_reset = true; }
        if controls.bounces != previous.bounces { needs_reset = true; }
        if let Some(file)=&mut interaction_log {
            use std::io::Write;
            if !rt.keys_pressed.is_empty() || previous_interaction_state!=(focused,show_ui,look_mode) || (camera.pos-previous_diffuse_camera.pos).length()>1e-5
                || (camera.yaw-previous_diffuse_camera.yaw).abs()>1e-5 {
                let keys:Vec<_>=rt.keys_pressed.iter().map(|k|format!("{k:?}")).collect();
                writeln!(file,"{{\"frame\":{},\"dt\":{dt},\"focused\":{focused},\"position\":{:?},\"yaw\":{},\"pitch\":{},\"panel\":{show_ui},\"look\":{look_mode},\"mouse\":{:?},\"rmb\":{},\"over_controls\":{over_controls},\"orbit\":{},\"keys\":{:?}}}",
                    rt.tick,[camera.pos.x,camera.pos.y,camera.pos.z],camera.yaw,camera.pitch,[camera_mouse.x,camera_mouse.y],rt.button_down(MouseButton::Right),controls.camera_motion,keys).unwrap();
            }
            previous_interaction_state=(focused,show_ui,look_mode);
        }
        // Keep the cache while dragging so GI remains visible and adapts naturally.
        // Explicit presets / the reset button clear history for a clean comparison.
        let mut commands = CommandList::new();
        if needs_reset {
            rc.reset_table(&mut commands, true);
            commands += trace.clear_frame_hits();
            gi_availability.update(false);
            diffuse_history_valid=false;
            needs_reset = false;
        }
        let mut swept = Vec::new();
        if let Some(animation) = &mut animation {
            let updates = animation.update(&core.world, dt, controls.rigid_motion, controls.skin_motion, rt.tick);
            commands.append(updates.vertices);
            profiler.checkpoint("Geometry vertex updates", &mut commands);
            commands += updates.blas;
            profiler.checkpoint("Geometry BLAS updates", &mut commands);
            commands += updates.tlas;
            profiler.checkpoint("Geometry TLAS update", &mut commands);
            swept = updates.swept;
        }
        if gallery {
            commands += core.update_point_lights(chandra::gallery::lights(light_time, controls.point_intensity));
            profiler.checkpoint("Local light upload", &mut commands);
        }
        let direction_changed = changed(controls.azimuth, previous.azimuth) || changed(controls.elevation, previous.elevation);
        let energy_changed = changed(controls.intensity, previous.intensity) || changed(controls.sky, previous.sky)
            || controls.color.iter().zip(previous.color).any(|(&a,b)| changed(a,b))
            || changed(controls.emission, previous.emission) || changed(controls.point_intensity, previous.point_intensity);
        let local_light_motion=gallery && (controls.animate || (light_track=="points" && rt.tick>=120 && rt.tick<quality_stop));
        let response = lighting_response.update(dt, direction_changed || local_light_motion, energy_changed, controls.bounces);
        if !single_frame {
            commands += history.update_with_dt(&rc, &swept, response.retention, dt);
            profiler.checkpoint("Dynamic history", &mut commands);
        }
        commands += core.lighting.write_host(controls.lighting());
        commands += core.view.write_host(camera.view());
        commands += core.compute_gbuffer();
        profiler.checkpoint("GBuffer", &mut commands);
        let diffuse_phase=rt.tick%2;
        let translation=(camera.pos-previous_diffuse_camera.pos).length();
        let rotation=(camera.yaw-previous_diffuse_camera.yaw).abs()+(camera.pitch-previous_diffuse_camera.pitch).abs();
        let camera_moving=translation>1e-4 || rotation>1e-5;
        if let Some(data)=&diffuse_reconstruction {
            commands+=data.prepare(&core,diffuse_phase,previous_diffuse_camera.view(),diffuse_history_valid,
                // Diffuse lighting is view independent. Reprojection and local
                // support already give newly visible pixels 16/32/64 samples.
                if response.retention<1.0 || !swept.is_empty() {4}else{2},
                if diffuse_samples>0 {diffuse_samples} else if !adaptive_diffuse {4}else{0});
            profiler.checkpoint("Diffuse sample support",&mut commands);
        }
        commands += rc.cache.set_view(camera.view());
        commands += rc.init_probes_base(&core);
        // Secondary ray buffers have no valid data before the first trace.
        if !single_frame && controls.bounces && rt.tick > 0 { commands += rc.init_probes_secondary(&trace); }
        for c in 1..rc.num_cascades { commands += rc.init_probes(c); }
        commands += rc.set_lifetime_base(&core);
        if !single_frame && controls.bounces && rt.tick > 0 { commands += rc.set_lifetime_secondary(&trace); }
        for c in 1..rc.num_cascades { commands += rc.set_lifetime(c); }
        profiler.checkpoint("Probe init and lifetime", &mut commands);
        if debug_sync {
            println!("Probe allocation cursors {:?}; cascade starts {:?}",
                rc.cache.table.freelist_index.copy_to_vec(), rc.cache.table.table_data_start.copy_to_vec());
        }
        let timer = profiler.begin("Cache freelists and readback");
        let mut capacity_ok = rc.try_reset_table(&mut commands, false);
        if !capacity_ok {
            capacity_recoveries += 1;
            gi_availability.update(false);
            diffuse_history_valid=false;
            capacity_ok = rc.recover_probe_capacity(&core,
                (!single_frame && controls.bounces && rt.tick > 0).then_some(&trace), &mut commands);
            if capacity_ok && let Some(data)=&diffuse_reconstruction {
                commands+=data.prepare(&core,diffuse_phase,previous_diffuse_camera.view(),false,64,
                    if adaptive_diffuse {0}else{4});
            }
        }
        profiler.end(timer);
        if capacity_ok {
            if single_frame {
                if [1,16,64].contains(&rt.tick) && let Some(poison) = &poison_transport {
                    commands.push(poison.dispatch_async([rc.cache.weights.len() as u32,1,1]));
                }
                commands += rc.clear_transport();
                profiler.checkpoint("Clear frame transport", &mut commands);
            }
            // Split ray tracing starts at G-buffer surface points. Average probe
            // positions are used only by the alternate trace_deposit renderer.
            commands += trace.reset();
            commands += trace.compute_ray_count_base(&core, &rc);
            for c in 1..rc.num_cascades { commands += trace.compute_ray_count(&rc, c); }
            for c in (0..rc.num_cascades).rev() { commands += trace.compute_ray_offset(&rc, c); }
            if single_frame && !split_single_frame { commands += trace.prepare_stable_samples(&core, &rc); }
            if min_probe_rays > 0 {
                commands += trace.apply_probe_budget(&rc);
                for c in 1..rc.num_cascades { commands += trace.compute_ray_count(&rc, c); }
                for c in (0..rc.num_cascades).rev() { commands += trace.compute_ray_offset(&rc, c); }
                commands += trace.prepare_probe_work(&core, &rc);
            }
            profiler.checkpoint("Ray scheduling", &mut commands);
            // Allocation can recycle slots. Rebuild from the SAME immutable
            // base_radiance read by exact queries, never from a stale atlas.
            if cached_diffuse_feedback && controls.bounces {
                commands += splat.calculate_irradiance(&rc);
                commands += splat.copy_irradiance_borders(&rc);
            }
            profiler.checkpoint("Secondary irradiance", &mut commands);
            let c0_len = ((2 * rc.base_angular_resolution.pow(2)) as f32 / (4. * std::f32::consts::PI)).sqrt();
            if !single_frame && controls.bounces && rt.tick > 0 { commands += trace.compute_secondary_cached(&core, &rc, rt.tick, c0_len, true, 0, cached_diffuse_feedback.then_some(&splat)); }
            profiler.checkpoint("GI secondary rays", &mut commands);
            if frame_cache { commands += trace.clear_frame_hits(); }
            commands += if split_single_frame { trace.compute_rays(&core, &rc, 0, c0_len, false) }
                else if single_frame { trace.compute_rays_mode(&core, &rc, 0, c0_len, false, path_depth, samples) }
                else { trace.compute_rays_adaptive_cached(&core, &rc, rt.tick, c0_len, controls.bounces, 0,
                    if diffuse_samples>0 {diffuse_samples} else {manual_response.primary_samples},
                    diffuse_reconstruction.as_ref().map(|d|&d.samples), cached_diffuse_feedback.then_some(&splat)) };
            profiler.checkpoint("GI primary rays", &mut commands);
            if frame_cache {
                let timer = profiler.begin("Frame secondary probe init");
                commands += rc.init_probes_secondary(&trace);
                for c in 1..rc.num_cascades { commands += rc.init_probes(c); }
                commands += rc.set_lifetime_base(&core);
                commands += rc.set_lifetime_secondary(&trace);
                for c in 1..rc.num_cascades { commands += rc.set_lifetime(c); }
                capacity_ok = rc.try_reset_table(&mut commands, false);
                profiler.end(timer);
                if capacity_ok {
                    commands += trace.compute_secondary_mode(&core, &rc, 0, c0_len, false, path_depth - 1);
                    profiler.checkpoint("Frame secondary paths", &mut commands);
                    for c in (0..rc.num_cascades).rev() { commands += rc.merge_cache(c); }
                    profiler.checkpoint("Frame secondary merge", &mut commands);
                    commands += trace.resolve_frame_primary(&core, &rc, c0_len);
                    profiler.checkpoint("Frame primary resolve", &mut commands);
                }
            }
            if capacity_ok {
                if !trace.fused_deposit {
                    commands.execute();
                    let rays = trace.next_traced_ray.read_blocking();
                    assert!(rays <= trace.ray_capacity, "Ray buffer capacity exceeded");
                    commands += rc.deposit_split(trace.traced_rays.view(..rays as usize));
                }
                profiler.checkpoint("Radiance deposit", &mut commands);
                for c in (0..rc.num_cascades).rev() { commands += rc.merge_cache(c); }
                profiler.checkpoint("Cache merge", &mut commands);
                // shade_optimized_32 overwrites every visible pixel; sky pixels are
                // handled directly by draw. The other splat path's depth buffer is unused.
                commands += splat.calculate_irradiance(&rc);
                commands += splat.copy_irradiance_borders(&rc);
                profiler.checkpoint("Irradiance", &mut commands);
                commands += splat.shade_optimized_32(&core, &rc);
                profiler.checkpoint("Screen GI interpolation", &mut commands);
                if let Some(data)=&diffuse_reconstruction {
                    commands+=data.capture(&core,&splat);
                    let frames=if energy_changed {2.0} else if response.retention<1.0 || !swept.is_empty() {4.0}
                        else if camera_moving {8.0} else {16.0};
                    commands+=data.reconstruct(&core,diffuse_phase,previous_diffuse_camera.view(),diffuse_history_valid,
                        (frames*(1.0/60.0)/dt).max(1.0));
                    profiler.checkpoint("Diffuse reconstruction",&mut commands);
                }
            }
        }
        if !capacity_ok {
            skipped_gi_frames += 1;
            commands += trace.reset();
            commands += trace.clear_frame_hits();
        }
        let warmed = gi_availability.update(capacity_ok);
        let gi_available = capacity_ok && ((single_frame && capacity_recoveries == 0) || warmed);
        let previous_reflection_view=previous_diffuse_camera.view();
        let mut composition_history_valid=false;
        if controls.reflections && let Some(data)=&cached_specular {
            let light_now=controls.lighting().sun_dir;
            let light_before=previous.lighting().sun_dir;
            let light_jump=light_now.x*light_before.x+light_now.y*light_before.y+light_now.z*light_before.z<5_f32.to_radians().cos();
            let reset=energy_changed || light_jump || !swept.is_empty();
            let valid=specular_history_valid && diffuse_history_valid && !reset;
            composition_history_valid=valid && controls.mode==previous.mode;
            if gi_available { commands+=data.prefilter(&rc); }
            profiler.checkpoint("Reflection RC prefilter",&mut commands);
            commands+=data.trace(&core,&rc,&splat,rt.tick,gi_available);
            profiler.checkpoint("Reflection bounded GGX",&mut commands);
            let frames=if reset {1.0} else if camera_moving || response.retention<1.0 {6.0} else {16.0};
            for (stage,command) in data.reconstruct(&core,&rc,&splat,rt.tick,previous_diffuse_camera.view(),valid,(frames/60.0/dt).max(1.0),gi_available) {
                commands+=command; profiler.checkpoint(stage,&mut commands);
            }
            specular_history_valid=capacity_ok;
        } else if controls.reflections && let Some(data)=&specular {
            let light_now=controls.lighting().sun_dir;
            let light_before=previous.lighting().sun_dir;
            let light_jump=light_now.x*light_before.x+light_now.y*light_before.y+light_now.z*light_before.z<5_f32.to_radians().cos();
            let reset_reflections=energy_changed || light_jump || !swept.is_empty();
            let reflection_history_valid=specular_history_valid && diffuse_history_valid && !reset_reflections;
            use chandra::algorithm::specular::SpecularTraceStage;
            let reflection_stages=|stage|data.trace_stages(&core,&rc,if args.iter().any(|a|a=="--specular-exact-diffuse") {None} else {Some(&splat)},rt.tick,specular_samples,gi_available,specular_reference,!args.iter().any(|a| a == "--no-specular-regularization"),previous_diffuse_camera.view(),reflection_history_valid,stage);
            for (stage, command) in reflection_stages(SpecularTraceStage::Base) {
                commands+=command; profiler.checkpoint(stage,&mut commands);
            }
            if specular_samples<128 && !args.iter().any(|a|a=="--specular-serial") {
                // Include the compact count readback in event and wall-clock timings.
                commands.execute();
                let timer=profiler.begin("Reflection queue readback");
                let active=data.pending_refinement_count();
                profiler.end(timer);
                for (stage,command) in reflection_stages(SpecularTraceStage::Refinement(active)) {
                    commands+=command; profiler.checkpoint(stage,&mut commands);
                }
            }
            if args.iter().any(|a|a=="--no-specular-filter") {
                commands.push(data.raw.copy_to_buffer_async(&data.output));
            } else {
                let history_frames=if reset_reflections {1.0}
                    else if camera_moving || response.retention<1.0 {4.0}else{24.0};
                commands+=data.reconstruct(&core,rt.tick,previous_diffuse_camera.view(),reflection_history_valid,
                    (history_frames*(1.0/60.0)/dt).max(1.0));
            }
            profiler.checkpoint("Reflection reconstruction",&mut commands);
            specular_history_valid=capacity_ok;
        } else {specular_history_valid=false;}
        diffuse_history_valid=capacity_ok;
        previous_diffuse_camera=camera;
        commands += invalid_pixels.write_host(0);
        commands.push(draw.dispatch_async([size[0], size[1], 1], &camera.view(), &controls.mode, &2_f32.powf(controls.exposure), &gi_available,&controls.reflections));
        profiler.checkpoint("Direct and compose", &mut commands);
        if controls.reflections && controls.mode==0 && let Some(data)=&cached_specular {
            commands+=data.compose(&core,rt.tick,previous_reflection_view,composition_history_valid);
            commands.push(present_cached.as_ref().unwrap().dispatch_async(core.screen_dispatch_size(),&2_f32.powf(controls.exposure)));
            profiler.checkpoint("Reflection composition TAA",&mut commands);
        }
        commands.execute();

        if quality_check && ([0, 1, 2, 16, 64, 200].contains(&rt.tick) || rt.tick == quality_frames
            || (quality_tail > 0 && rt.tick > quality_frames.saturating_sub(quality_tail))) {
            let directory = output_dir.as_deref().unwrap_or(".local/quality-check");
            std::fs::create_dir_all(directory).unwrap();
            let pixels = linear_output.as_ref().unwrap().copy_to_vec();
            if let Some(buffer)=&base_output {
                let values:Vec<[f32;4]>=buffer.copy_to_vec().into_iter().map(|v|[v.x,v.y,v.z,v.w]).collect();
                std::fs::write(format!("{directory}/base-lighting-{:03}.rgba32f",rt.tick),bytemuck::cast_slice(&values)).unwrap();
            }

            if let Some((raw,output,diagnostics))=&reflection_buffers {
                if args.iter().any(|a|a=="--specular-diagnostics") {
                    let values:Vec<[f32;4]>=diagnostics.copy_to_vec().into_iter().map(|v|[v.x,v.y,v.z,v.w]).collect();
                    std::fs::write(format!("{directory}/specular-diagnostics-{:03}.rgba32f",rt.tick),bytemuck::cast_slice(&values)).unwrap();
                }
                for (name,buffer) in [("specular-raw",raw),("specular-filtered",output)]{
                    let values:Vec<[f32;4]>=buffer.copy_to_vec().into_iter().map(|v|[v.x,v.y,v.z,v.w]).collect();
                    std::fs::write(format!("{directory}/{name}-{:03}.rgba32f",rt.tick),bytemuck::cast_slice(&values)).unwrap();
                }
            }
            if rt.tick == 0 {
                std::fs::write(format!("{directory}/probe-pixel-counts.u32"), bytemuck::cast_slice(&trace.ray_count.view(..rc.cache.base_probes as usize).copy_to_vec())).unwrap();
            }
            let pixels: Vec<[f32;4]> = pixels.into_iter().map(|p| [p.x,p.y,p.z,p.w]).collect();
            std::fs::write(format!("{directory}/frame-{:03}.rgba32f", rt.tick), bytemuck::cast_slice(&pixels)).unwrap();
            if let Some(buffers)=&surface_buffers {
                buffers.capture(core.screen_dispatch_size());
                let positions:Vec<[f32;4]>=buffers.position.copy_to_vec().into_iter().map(|p|[p.x,p.y,p.z,p.w]).collect();
                let normals:Vec<[f32;4]>=buffers.normal.copy_to_vec().into_iter().map(|p|[p.x,p.y,p.z,p.w]).collect();
                std::fs::write(format!("{directory}/position-{:03}.rgba32f",rt.tick),bytemuck::cast_slice(&positions)).unwrap();
                std::fs::write(format!("{directory}/normal-{:03}.rgba32f",rt.tick),bytemuck::cast_slice(&normals)).unwrap();
            }
            if args.iter().any(|a|a=="--diffuse-diagnostics") && let Some(data)=&diffuse_reconstruction {
                for (name,buffer) in [("diffuse-raw",&data.raw),("diffuse-spatial",&data.filtered),("diffuse-temporal",&data.output)] {
                    let values:Vec<[f32;4]>=buffer.copy_to_vec().into_iter().map(|p|[p.x,p.y,p.z,p.w]).collect();
                    std::fs::write(format!("{directory}/{name}-{:03}.rgba32f",rt.tick),bytemuck::cast_slice(&values)).unwrap();
                }
                std::fs::write(format!("{directory}/diffuse-samples-{:03}.u32",rt.tick),bytemuck::cast_slice(&data.samples.copy_to_vec())).unwrap();
            }
            if let Some(samples) = &surface_samples {
                std::fs::write(format!("{directory}/surface-{:03}.rgba32f", rt.tick), bytemuck::cast_slice(&samples.capture())).unwrap();
            }
            std::fs::write(format!("{directory}/capacity.txt"), format!("recoveries={capacity_recoveries}; skipped_gi_frames={skipped_gi_frames}; lod_bias={}; gi_available={gi_available}\n", rc.cache.lod_bias.read_blocking())).unwrap();
            assert_eq!(invalid_pixels.read_blocking(), 0, "Nonfinite quality output");
            let motion_description = if quality_motion { "animation controlled by geometry-only/stop options" } else { "static scene, lights and camera" };
            std::fs::write(format!("{directory}/quality.txt"), format!("{size:?}; {transport_metadata}; {lighting_metadata}; linear RGBA32F little endian, current-frame output after exposure before presentation; {motion_description}; frame 0 is cold; single-frame mode poisons prior transport at frames 1/16/64; display_mode={}\n", controls.mode)).unwrap();
            std::fs::write(format!("{directory}/arguments.txt"), args.join("\n")).unwrap();
            let v = camera.view();
            let columns = v.transform.cols.map(|c| [c.x,c.y,c.z]);
            let animation_times = animation.as_ref().map_or([0.,0.], |a| [a.rigid_time,a.skin_time]);
            std::fs::write(format!("{directory}/camera-{:03}.json", rt.tick),
                format!("{{\"position\":{:?},\"transform_columns\":{:?},\"scaling\":{},\"screen_size\":{:?},\"animation_times\":{:?},\"dt\":{dt}}}\n",
                    [v.pos.x,v.pos.y,v.pos.z], columns, v.scaling, size, animation_times)).unwrap();
            if rt.tick == quality_frames { rt.exit(); }
        }

        ui_pixels.fill([0; 4]);
        let primitives = ui_context.tessellate(output.shapes, output.pixels_per_point);
        ui_renderer.render(&mut BufferMutRef::new(&mut ui_pixels, size[0] as usize, size[1] as usize),
            &primitives, &output.textures_delta, output.pixels_per_point);
        ui_texture.view(0).copy_from(&ui_pixels);
        composite_ui.dispatch([size[0], size[1], 1]);
        profiler.finish_frame(rt.tick >= 200 && rt.tick < 200 + profile_frames);
        // At frame N, the runtime interval covers the completed frame N-1,
        // including presentation. Pair intervals 200..319 with GPU samples.
        if profiling && rt.tick > 200 && rt.tick <= 200 + profile_frames {
            profiler.record_frame_interval(rt.frame_time_f64());
        }
        if profiling {
            assert_eq!(invalid_pixels.read_blocking(), 0, "Nonfinite rendered pixel during profiling");
        }
        if profiling && rt.tick == 200 + profile_frames {
            let directory = output_dir.as_deref().unwrap_or(".local/gpu-profile");
            profiler.save(directory);
            let image: Vec<[u8; 4]> = rt.final_display().view(0).copy_to_vec();
            image::save_buffer(format!("{directory}/view.png"), bytemuck::cast_slice(&image), size[0], size[1], image::ColorType::Rgba8).unwrap();
            std::fs::write(format!("{directory}/scene.txt"), format!("{}x{}; {scene_counts}; rigid-only={rigid_only}; rigid-refit={rigid_refit}; {history_regions} independent history regions; 200 warmup frames + {profile_frames} measured frames; simulation step=1/60s; uncapped; multiple bounces={}; no nonfinite output\n", size[0], size[1], controls.bounces)).unwrap();
            std::fs::write(format!("{directory}/lighting.txt"), format!("{lighting_metadata}\n")).unwrap();
            rt.exit();
        }
        let rays = if benchmarking || stressing || profiling || rt.tick == 1 || rt.tick.is_multiple_of(300) {
            let rays = trace.next_traced_ray.read_blocking();
            assert!(rays <= trace.ray_capacity, "Ray capacity exceeded");
            rays
        } else { 0 };
        yesod::printer::global::flush_printer();
        if rt.tick == 1 || rt.tick.is_multiple_of(300) {
            println!("Frame {}: {:.1} FPS, {} rays, {} probes", rt.tick, rt.fps(), rays, rc.num_probes.iter().sum::<u32>());
        }
        if smoke && (rt.tick == 80 || rt.tick == 160) {
        // The testbed presents the preceding frame before this callback.
            let image: Vec<[u8; 4]> = rt.final_display().view(0).copy_to_vec();
            let path = if rt.tick == 80 { ".local/daylight.png" } else { ".local/rotated-light.png" };
            std::fs::create_dir_all(".local").unwrap();
            image::save_buffer(path, bytemuck::cast_slice(&image), size[0], size[1], image::ColorType::Rgba8).unwrap();
            if let Some(first) = &first_test_image {
                let mut sum = 0u64;
                let mut count = 0u64;
                for y in 0..size[1] { for x in 320..size[0] {
                    let i = (y * size[0] + x) as usize;
                    sum += first[i][0].abs_diff(image[i][0]) as u64 + first[i][1].abs_diff(image[i][1]) as u64 + first[i][2].abs_diff(image[i][2]) as u64;
                    count += 3;
                }}
                let difference = sum as f64 / count as f64;
                assert!(difference > 1.0, "Lighting did not visibly change: {difference}");
                println!("SMOKE TEST PASS: mean RGB difference after rotating sunlight = {difference:.2}/255");
                rt.exit();
            } else { first_test_image = Some(image); }
        }
        if let Some(benchmark) = &mut benchmark {
            if benchmark.end_frame(rt.tick, rt, size, invalid_pixels.read_blocking()) { rt.exit(); }
        }
        if let Some(stress) = &mut stress {
            if stress.end_frame(rt.tick, rt, size, invalid_pixels.read_blocking()) { rt.exit(); }
        }
    });
    mouse_capture.release();
    if let Some(scene) = switch_scene { std::process::exit(scene.exit_code()); }
}
