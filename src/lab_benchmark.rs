//! Deterministic scenarios; timing uses complete wall-clock frame intervals.
use std::time::Instant;

use keter_testbed::Runtime;

pub const FRAMES: u32 = 240;
pub const CASES: [&str; 7] = [
    "static", "rigid", "skinned", "sun", "camera", "combined", "settled",
];

pub struct Benchmark {
    last_frame: Instant,
    timings: Vec<Vec<f64>>,
    cases: Vec<&'static str>,
    frames: u32,
    skip_interval: bool,
    moving: Option<Vec<[u8; 4]>>,
    after_20: Option<Vec<[u8; 4]>>,
    after_120: Option<Vec<[u8; 4]>>,
    directory: String,
}

impl Benchmark {
    pub fn new(
        size: [u32; 2],
        intel: bool,
        emissive: bool,
        gallery: bool,
        output: Option<String>,
    ) -> Self {
        let directory = output.unwrap_or_else(|| {
            format!(
                ".local/{}benchmark-{}x{}",
                if intel { "intel-sponza-" } else { "" },
                size[0],
                size[1]
            )
        });
        std::fs::create_dir_all(&directory).unwrap();
        let cases = if gallery {
            CASES.to_vec()
        } else if emissive {
            vec!["static", "rigid", "camera", "combined", "settled"]
        } else if intel {
            vec!["static", "rigid", "sun", "camera", "combined", "settled"]
        } else {
            CASES.to_vec()
        };
        Self {
            last_frame: Instant::now(),
            timings: vec![Vec::new(); cases.len()],
            cases,
            frames: if intel || gallery { 600 } else { FRAMES },
            skip_interval: false,
            moving: None,
            after_20: None,
            after_120: None,
            directory,
        }
    }
    pub fn begin_frame(&mut self, frame: u32) -> [bool; 4] {
        let elapsed = self.last_frame.elapsed().as_secs_f64() * 1000.;
        self.last_frame = Instant::now();
        let case = (frame / self.frames) as usize;
        if !self.skip_interval && frame % self.frames > self.frames / 4 && case < self.cases.len() {
            self.timings[case].push(elapsed);
        }
        self.skip_interval = false;
        // rigid, skin, sun, camera
        match self.cases[case] {
            "rigid" => [true, false, false, false],
            "skinned" => [false, true, false, false],
            "sun" => [false, false, true, false],
            "camera" => [false, false, false, true],
            "combined" => [true; 4],
            _ => [false; 4],
        }
    }
    pub fn end_frame(&mut self, frame: u32, rt: &Runtime, size: [u32; 2], nonfinite: u32) -> bool {
        let case = (frame / self.frames) as usize;
        assert_eq!(nonfinite, 0, "Nonfinite output in {}", self.cases[case]);
        let local = frame % self.frames;
        if self.cases[case] == "rigid" && self.frames == 600 && matches!(local, 180 | 360 | 540) {
            self.save(
                &format!("rigid-{local}"),
                &rt.final_display().view(0).copy_to_vec(),
                size,
            );
            // Exclude screenshot readback/file I/O from the following frame interval.
            self.skip_interval = true;
        }
        if self.cases[case] == "settled" && matches!(local, 0 | 20 | 120) {
            let image = rt.final_display().view(0).copy_to_vec();
            let name = match local {
                0 => {
                    self.moving = Some(image);
                    "moving-end"
                }
                20 => {
                    self.after_20 = Some(image);
                    "settle-20"
                }
                _ => {
                    self.after_120 = Some(image);
                    "settle-120"
                }
            };
            let pixels = match local {
                0 => self.moving.as_ref(),
                20 => self.after_20.as_ref(),
                _ => self.after_120.as_ref(),
            }
            .unwrap();
            self.save(name, pixels, size);
        }
        if local != self.frames - 1 {
            return false;
        }
        let pixels: Vec<[u8; 4]> = rt.final_display().view(0).copy_to_vec();
        self.save(self.cases[case], &pixels, size);
        let (mean, p95, worst) = statistics(&self.timings[case]);
        println!(
            "BENCH {}: mean={mean:.3} ms, p95={p95:.3} ms, max={worst:.3} ms, FPS={:.1}",
            self.cases[case],
            1000. / mean
        );
        if case != self.cases.len() - 1 {
            return false;
        }
        let mut csv = String::from("scenario,width,height,samples,mean_ms,p95_ms,max_ms,fps\n");
        for (i, name) in self.cases.iter().enumerate() {
            let (mean, p95, worst) = statistics(&self.timings[i]);
            csv.push_str(&format!(
                "{name},{},{},{},{mean:.4},{p95:.4},{worst:.4},{:.2}\n",
                size[0],
                size[1],
                self.timings[i].len(),
                1000. / mean
            ));
        }
        std::fs::write(format!("{}/timings.csv", self.directory), csv).unwrap();
        let moving_error = difference(self.moving.as_ref().unwrap(), &pixels, size);
        let error20 = difference(self.after_20.as_ref().unwrap(), &pixels, size);
        let error120 = difference(self.after_120.as_ref().unwrap(), &pixels, size);
        let results = format!(
            "Same-pose RGB MAE versus the image after {} stationary frames (0..255), excluding control panel:\nAt motion stop: {moving_error:.4}\nAfter 20 stationary frames: {error20:.4}\nAfter 120 stationary frames: {error120:.4}\nAll cases: no nonfinite rendered pixels.\n",
            self.frames
        );
        println!("{results}");
        std::fs::write(format!("{}/convergence.txt", self.directory), results).unwrap();
        assert!(
            error120 < 10.,
            "History failed to settle: {error120:.3}/255"
        );
        true
    }
    fn save(&self, name: &str, pixels: &[[u8; 4]], size: [u32; 2]) {
        image::save_buffer(
            format!("{}/{name}.png", self.directory),
            bytemuck::cast_slice(pixels),
            size[0],
            size[1],
            image::ColorType::Rgba8,
        )
        .unwrap();
    }
}

fn statistics(values: &[f64]) -> (f64, f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (
        sorted.iter().sum::<f64>() / sorted.len() as f64,
        sorted[(sorted.len() as f64 * 0.95).floor() as usize],
        *sorted.last().unwrap(),
    )
}

fn difference(a: &[[u8; 4]], b: &[[u8; 4]], size: [u32; 2]) -> f64 {
    let mut total = 0u64;
    let mut count = 0;
    for y in 0..size[1] {
        for x in 320..size[0] {
            let i = (y * size[0] + x) as usize;
            for c in 0..3 {
                total += a[i][c].abs_diff(b[i][c]) as u64;
                count += 1;
            }
        }
    }
    total as f64 / count as f64
}

pub struct StressTest {
    last_frame: Instant,
    timings: Vec<f64>,
    render_timings: Vec<f64>,
    intervals_csv: String,
    directory: String,
}

impl StressTest {
    pub fn new(scene: &str, size: [u32; 2], output: Option<String>) -> Self {
        Self {
            last_frame: Instant::now(),
            timings: Vec::new(),
            render_timings: Vec::new(),
            intervals_csv: "frame,interval_ms,includes_png_capture\n".into(),
            directory: output
                .unwrap_or_else(|| format!(".local/{scene}-stress-test-{}x{}", size[0], size[1])),
        }
    }
    pub fn begin_frame(&mut self, frame: u32) {
        let elapsed = self.last_frame.elapsed().as_secs_f64() * 1000.;
        self.last_frame = Instant::now();
        if frame > 120 {
            self.timings.push(elapsed);
            let capture = matches!(frame, 1201 | 2401);
            if !capture { self.render_timings.push(elapsed); }
            self.intervals_csv.push_str(&format!("{frame},{elapsed:.6},{capture}\n"));
        }
    }
    pub fn end_frame(&mut self, frame: u32, rt: &Runtime, size: [u32; 2], nonfinite: u32) -> bool {
        assert_eq!(
            nonfinite, 0,
            "Nonfinite output at stress-test frame {frame}"
        );
        if matches!(frame, 1200 | 2400 | 3599) {
            std::fs::create_dir_all(&self.directory).unwrap();
            let pixels: Vec<[u8; 4]> = rt.final_display().view(0).copy_to_vec();
            image::save_buffer(
                format!("{}/frame-{frame}.png", self.directory),
                bytemuck::cast_slice(&pixels),
                size[0],
                size[1],
                image::ColorType::Rgba8,
            )
            .unwrap();
        }
        if frame < 3599 {
            return false;
        }
        let (mean, p95, worst) = statistics(&self.timings);
        let mut report = format!(
            "Combined motion stress test PASS\n3600 frames / 60 seconds simulated time / {}x{}\nNo nonfinite rendered pixels or ray-buffer overflow.\nUncapped end-to-end frame timing (first 120 frames excluded):\nMean {mean:.3} ms, p95 {p95:.3} ms, max {worst:.3} ms, {:.1} FPS\n",
            size[0],
            size[1],
            1000. / mean
        );
        let (mean, p95, worst) = statistics(&self.render_timings);
        report.push_str(&format!("Excluding only the two known PNG readback/encoding intervals (all other outliers retained):\nMean {mean:.3} ms, p95 {p95:.3} ms, max {worst:.3} ms, {:.1} FPS\n", 1000. / mean));
        println!("{report}");
        std::fs::write(format!("{}/results.txt", self.directory), report).unwrap();
        std::fs::write(format!("{}/intervals.csv", self.directory), &self.intervals_csv).unwrap();
        true
    }
}
