//! Optional CUDA event ranges, without the upstream Linux-only bindgen setup.
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr;

use chandra::algorithm::CommandList;
use keter::prelude::*;

type Handle = *mut c_void;
type Push = unsafe extern "system" fn(Handle) -> i32;
type Pop = unsafe extern "system" fn(*mut Handle) -> i32;
type Create = unsafe extern "system" fn(*mut Handle, u32) -> i32;
type Record = unsafe extern "system" fn(Handle, Handle) -> i32;
type Event = unsafe extern "system" fn(Handle) -> i32;
type Elapsed = unsafe extern "system" fn(*mut f32, Handle, Handle) -> i32;

struct Driver {
    _library: libloading::Library,
    push: Push,
    pop: Pop,
    create: Create,
    record: Record,
    synchronize: Event,
    destroy: Event,
    elapsed: Elapsed,
}
impl Driver {
    fn new() -> Self {
        unsafe {
            let library = libloading::Library::new(if cfg!(windows) {
                "nvcuda.dll"
            } else {
                "libcuda.so.1"
            })
            .unwrap();
            Self {
                push: *library.get(b"cuCtxPushCurrent_v2\0").unwrap(),
                pop: *library.get(b"cuCtxPopCurrent_v2\0").unwrap(),
                create: *library.get(b"cuEventCreate\0").unwrap(),
                record: *library.get(b"cuEventRecord\0").unwrap(),
                synchronize: *library.get(b"cuEventSynchronize\0").unwrap(),
                destroy: *library.get(b"cuEventDestroy_v2\0").unwrap(),
                elapsed: *library.get(b"cuEventElapsedTime\0").unwrap(),
                _library: library,
            }
        }
    }
    fn push(&self) {
        unsafe {
            check((self.push)(DEVICE.native_handle()));
        }
    }
    fn pop(&self) {
        unsafe {
            let mut context = ptr::null_mut();
            check((self.pop)(&mut context));
        }
    }
}
fn check(result: i32) {
    assert_eq!(result, 0, "CUDA profiler error {result}");
}

pub struct GpuProfiler {
    debug_sync: bool,
    driver: Option<Driver>,
    ranges: Vec<(&'static str, Handle, Handle)>,
    samples: BTreeMap<&'static str, Vec<f32>>,
}
impl GpuProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            debug_sync: std::env::var_os("CHANDRA_GPU_DEBUG_SYNC").is_some(),
            driver: enabled.then(Driver::new),
            ranges: Vec::new(),
            samples: BTreeMap::new(),
        }
    }
    pub fn begin(&mut self, name: &'static str) -> Option<usize> {
        let driver = self.driver.as_ref()?;
        driver.push();
        unsafe {
            let (mut start, mut end) = (ptr::null_mut(), ptr::null_mut());
            check((driver.create)(&mut start, 0));
            check((driver.create)(&mut end, 0));
            check((driver.record)(
                start,
                DEVICE.default_stream().native_handle(),
            ));
            let index = self.ranges.len();
            self.ranges.push((name, start, end));
            driver.pop();
            Some(index)
        }
    }
    pub fn end(&self, index: Option<usize>) {
        if let Some(index) = index {
            let driver = self.driver.as_ref().unwrap();
            driver.push();
            unsafe {
                check((driver.record)(
                    self.ranges[index].2,
                    DEVICE.default_stream().native_handle(),
                ));
            }
            driver.pop();
        }
    }
    pub fn checkpoint(&mut self, name: &'static str, commands: &mut CommandList<'static>) {
        if self.driver.is_none() {
            return;
        }
        let index = self.begin(name);
        if self.debug_sync {
            println!("GPU stage begin: {name}");
        }
        commands.execute();
        self.end(index);
        if self.debug_sync
            && let Some(index) = index
        {
            let driver = self.driver.as_ref().unwrap();
            driver.push();
            unsafe {
                check((driver.synchronize)(self.ranges[index].2));
            }
            driver.pop();
            println!("GPU stage complete: {name}");
        }
    }
    pub fn finish_frame(&mut self, record: bool) {
        let Some(driver) = &self.driver else {
            return;
        };
        driver.push();
        if let Some(last) = self.ranges.last() {
            unsafe {
                check((driver.synchronize)(last.2));
            }
        }
        for (name, start, end) in self.ranges.drain(..) {
            unsafe {
                let mut ms = 0.;
                check((driver.elapsed)(&mut ms, start, end));
                if record {
                    self.samples.entry(name).or_default().push(ms);
                }
                check((driver.destroy)(start));
                check((driver.destroy)(end));
            }
        }
        driver.pop();
    }
    pub fn save(&self, directory: &str) {
        std::fs::create_dir_all(directory).unwrap();
        let mut rows: Vec<_> = self
            .samples
            .iter()
            .map(|(name, samples)| {
                (
                    *name,
                    samples.iter().sum::<f32>() / samples.len() as f32,
                    samples,
                )
            })
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut csv = String::from("pass,mean_gpu_ms,min_gpu_ms,max_gpu_ms,samples,p95_gpu_ms\n");
        for (name, mean, samples) in rows {
            let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
            let max = samples.iter().copied().fold(0., f32::max);
            let mut sorted = samples.clone();
            sorted.sort_by(f32::total_cmp);
            let p95 = sorted[(sorted.len() as f32 * 0.95) as usize];
            println!("GPU {name}: {mean:.4} ms");
            csv.push_str(&format!(
                "{name},{mean:.6},{min:.6},{max:.6},{},{p95:.6}\n",
                samples.len()
            ));
        }
        std::fs::write(format!("{directory}/passes.csv"), csv).unwrap();
        let mut summary = String::from("metric,mean_gpu_ms,p95_gpu_ms,max_gpu_ms,samples\n");
        for (name, include_history) in [
            ("GI main", false),
            ("GI including dynamic history", true),
            ("Geometry total", false),
        ] {
            let count = self.samples.values().next().unwrap().len();
            let mut totals = vec![0.0f32; count];
            for (stage, samples) in &self.samples {
                let geometry = stage.starts_with("Geometry ");
                if (name == "Geometry total" && !geometry)
                    || (name != "Geometry total" && geometry)
                    || matches!(*stage, "GBuffer" | "Direct and compose")
                    || (!include_history && *stage == "Dynamic history")
                {
                    continue;
                }
                assert_eq!(samples.len(), count);
                for (total, sample) in totals.iter_mut().zip(samples) {
                    *total += sample;
                }
            }
            let mean = totals.iter().sum::<f32>() / count as f32;
            totals.sort_by(f32::total_cmp);
            let p95 = totals[(count as f32 * 0.95) as usize];
            let max = totals[count - 1];
            summary.push_str(&format!("{name},{mean:.6},{p95:.6},{max:.6},{count}\n"));
            println!("GPU {name}: {mean:.4} ms, p95 {p95:.4} ms");
        }
        std::fs::write(format!("{directory}/summary.csv"), summary).unwrap();
    }
}
