# Chandra Dynamic GI Lab

A dynamic-scene extension of [entropylost/chandra](https://github.com/entropylost/chandra), the reference implementation of [Split Radiance Cascades](https://arxiv.org/abs/2607.20384). The original research renderer is retained. This fork adds an interactive lighting lab, rigid instancing, GPU skinning, emissive materials, moving local lights, dynamic history invalidation and GPU timing tools.

**2026-10-26 / 稳定间接漫反射与自由相机：** 在动态 GI 历史、RC 合并、容量恢复、薄墙偏移、白房间与独立场景 worker 的基础上，加入按像素归一化的自适应补采样、表面重投影和漫反射重建，修正移动光源及相机运动时的局部闪烁。自由相机使用实际面板范围判断右键输入，F 看向可释放控件焦点并停止自动环绕。用户已确认当前漫反射效果满意并要求收尾，最终版本不再补跑性能。见 [当前收尾记录](benchmarks/diffuse-motion/README.md)；[第一批基础验证](benchmarks/diffuse/README.md)及下面的性能数字均为历史结果。Lighting Lab 未接入间接高光或 ReSTIR，PICA/机器人资源接入仍在本轮范围外。

**Historical 1080p optimization, before diffuse reconstruction:** the experimental single-frame mode reached **150.3 FPS** in the combined scenario and **143.9 FPS** over continuous motion on an RTX 5070 Ti, with **4.87 ms GI**. It clears transport every frame, but residual spatial sampling error remains. **120 FPS + no accumulation + no noise + unchanged quality has not been achieved.** The temporal path remains the default. See the [historical measurements and quality comparisons](benchmarks/1080p-single-frame/README.md).

![Procedural Chromatic Gallery](images/chromatic-gallery.png)

## Included scene / 自带测试场景

**Chromatic Gallery** is generated entirely in [`src/gallery.rs`](src/gallery.rs), without external models, textures or environment maps. Tiled floors, raised galleries, staircases, profiled columns, masonry arches, roof slats, niches and sculptures form an 88,160-triangle scene with 49,628 vertices and 926 static meshes.

| Group | Count | Animation |
|---|---:|---|
| Local point lights | 30 | Moving positions and changing intensity; finite-range shadows and GI |
| Non-emissive rigid objects | 30 | Independent translation and rotation |
| Emissive objects | 30 | Moving colored lanterns that illuminate nearby surfaces through GI |
| Skinned objects | 30 | Five joints each, with root motion and continuous deformation |

The sun direction and camera also animate. All 90 moving meshes have independent swept history regions and are simulated even when occluded or outside the view. Rendering is uncapped; VSync is off.

## Run on Windows

Tested with Windows, an NVIDIA RTX 5070 Ti and CUDA 12.8.1. Install Git, Rustup, PowerShell 7, Visual Studio 2022 C++ Build Tools (including CMake/Ninja), and a compatible NVIDIA driver.

```powershell
git clone https://github.com/EthanCS/chandra.git
cd chandra
./setup-lighting-lab.ps1
./run-lighting-lab.ps1 -FullHD
```

Setup checks out `entropylost/sefirot` at `590f48f4d6cb329fd0833525348b621d43f4fd21` in sibling directory `../sefirot`, downloads checksum-verified CUDA components to ignored `.local/cuda`, and installs pinned Rust toolchains. Existing unrelated dependency directories are not overwritten. The launcher uses the `nightly-2026-03-02` Cargo driver with the `nightly-2026-10-03` compiler to support the pinned Luisa build scripts. An installed `CUDA_PATH` can also be used.

The pinned Luisa frontend lacks transform-only instance updates and needs a corrected instance index when appending after a build. The launcher prepares a project-local patch using `tools/prepare-luisa-frontend.ps1` and `tools/luisa-accel-transform.rs`; backend/IR revisions stay pinned by `Cargo.lock`. It never modifies the shared Cargo cache. Run setup before invoking Cargo directly. First builds and GPU kernel compilation take longer.

```powershell
# Default complex gallery; all four groups enabled.
./run-lighting-lab.ps1 -Gallery -FullHD
# Smaller render target, same geometry.
./run-lighting-lab.ps1 -Gallery -Small
# Original small procedural courtyard: one rigid box and one skinned object.
./run-lighting-lab.ps1 -Courtyard -FullHD
# Four asset-free diffuse/leak fixtures: sealed shell, 8 cm slit, thin walls, roof corner.
./run-lighting-lab.ps1 -WhiteRoom -View 0
# Geometry motion with stationary camera and lights; optional one-bounce GI.
./run-lighting-lab.ps1 -Courtyard -GeometryOnly -SingleBounce
# Day/night presets disable the sun; local lights remain independently adjustable.
./run-lighting-lab.ps1 -Gallery -FullHD -EmissiveDay
./run-lighting-lab.ps1 -Gallery -FullHD -EmissiveNight
# Experimental frame-local transport, 8 path vertices, 1 sample per surface pixel.
./run-lighting-lab.ps1 -Gallery -FullHD -SingleFrame
# Original primary Split RC trace/split path, single-frame single-bounce comparison.
# No radiance history, secondary bounce cache, extra path tracing or stable-rank sorting.
./run-lighting-lab.ps1 -Gallery -FullHD -SplitSingleFrame -Static -View 2
# More current-frame samples; this exceeds the 120 FPS budget on the tested GPU.
./run-lighting-lab.ps1 -Gallery -FullHD -SingleFrame -Samples 8
```

Default resolution is 1280×800; `-FullHD` is 1920×1080 and `-Small` is 960×600. The `lighting_lab` binary is separate from the original `chandra` research viewer.

`-SplitSingleFrame` / `--split-single-frame` explicitly selects the original primary trace/split function with multibounce disabled and fixed directional jitter, rebuilding GI transport each frame. It uses one primary GI ray per visible surface pixel and bypasses the experimental path tracer and stable pixel-rank sorting. Spatial probe allocation is retained, but radiance, directional weights and quality are cleared. This is a single-bounce algorithm comparison in the dynamic lab, not a claim of identical pixels to the upstream executable or elimination of all spatial interpolation/sampling artifacts. `-Static` starts with animations paused; the controls can re-enable them.

## Controls

- Sliders: sun direction/color/intensity, sky, local-light strength, emission and exposure.
- Toggles: sun/local-light motion, rigid/emissive motion, skinning and camera motion.
- Compare full GI, direct illumination, indirect illumination and surface normals.
- `WASD`: free movement; arrows, right-drag or `F` mouse-look: look; `Q`/`E`: down/up; `Space`: up; `Shift`: fast; `Ctrl`: slow. Manual movement stops orbit. On Windows, losing window focus releases held input.
- `R`: reset view; `Tab` or `F1`: show/hide panel; `Esc`: close. A visible button also restores a hidden panel. White Room has four camera presets; other scenes have three.
- Scene menu: the supervisor waits for the renderer worker to exit before starting the new scene. A brief close/reopen is expected. `--scene-worker` bypasses supervision for deterministic diagnostics.
- Default mode: `Clear GI history` resets temporal accumulation; indirect light needs several frames to settle. Single-frame mode: `Rebuild GI allocation` resets spatial allocation; it does not accumulate radiance across frames.

Day/night presets use the same emission and exposure. Day scales the constant sky by 0.18; night sets sky to zero. The gallery's 30 local lights remain adjustable with the Local lights slider. These are lighting presets, not a physical atmosphere simulation. Radiance is in relative linear units, not calibrated lumens/lux; no bloom is added.

## Implementation

- Rigid BLASes retain immutable bind geometry. One transform batch updates the TLAS; one GPU dispatch updates world-space shading vertices for all rigid meshes.
- GPU skinning supports four normalized weights per vertex. The gallery batches 30 joint palettes and deformation dispatches; groups of three neighboring skins share a BLAS, retaining original triangle/material identity. Only deformed BLASes are refitted, with periodic rebuilds to limit traversal degradation. `--unbatched-skin` retains the reference path. Static BLASes are retained. Empty graph stages preserve execution order without extra GPU waits.
- Material-indexed emitted radiance is independent of diffuse reflectance, for camera and GI ray shading. There is no per-ray emitter-list scan.
- A conservative current-frame light grid selects candidates without changing light order or the exact range/cosine/shadow tests. `--no-light-grid` retains the full loop.
- Moving geometry reduces confidence in affected swept directions while preserving normalized radiance. Illumination motion and energy edits have separate response windows. Continuous point-light motion does not repeatedly trigger the stronger energy-edit flush.
- The Lab reconstructs unexposed diffuse irradiance before albedo/direct-light composition. Adaptive rays retain a common sample prefix and divide their deposit weights by the per-pixel budget; extra rays cannot overweight thin edges or disocclusions. New surface history receives four high-sample warmup frames. Spatial/temporal support checks instance identity, normals, plane distance and projected coverage, with separate previous/current buffers. `--no-diffuse-filter` selects raw RC and `--diffuse-diagnostics` exports the stages.
- Fused tracing/deposit, warp-aggregated counters, cached parent links/quadrature, coalesced history updates and opaque any-hit shadows reduce work. `--unfused` retains the separate deposit queue.
- The gallery reserves 131,072 base probes with the same 10 cm spatial setting. Exhausted pools reclaim allocations and coarsen spatial LOD, then explicitly skip GI if bounded recovery still fails. Direct light and the camera remain available; GI display resumes after two successful frames.
- Irradiance uses shared-memory angular quadrature with the same 32 input directions, 6×6 output resolution and Half4 storage. Corrected octahedral border indices read current-frame interior texels instead of old border texels.
- Experimental `--single-frame` retains the Split RC hierarchy, ray splitting and spatial interpolation. Hierarchical R2 primary directions and fixed later-bounce samples feed finite-depth paths within one frame. Transport/quality are cleared, prior-frame hits are unused, and no temporal denoiser is applied. This changes the multibounce estimator and is not quality-equivalent to converged temporal rendering.

The default research renderer uses temporal, approximate GI; fast motion and disocclusion may show convergence noise. The experimental mode removes radiance history but retains spatial sampling error and finite-depth truncation. Neither is a complete PBR renderer or an unbiased path-traced reference. Fixed sample placement is not a guarantee of noise-free rendering or bit-identical floating-point accumulation.

## Tests and measurements

```powershell
./run-lighting-lab.ps1 -Test
# Seven 600-frame cases: static, rigid/emissive, skinned, lights, camera, combined, settled.
./run-lighting-lab.ps1 -Gallery -FullHD -Benchmark -Output .local/gallery-benchmark
# 3,600 continuously moving frames; simulation advances by 1/60 s without sleeping.
./run-lighting-lab.ps1 -Gallery -FullHD -StressTest -Output .local/gallery-stress
# Separate CUDA event ranges: 200 warmup frames and 600 measured frames.
./run-lighting-lab.ps1 -Gallery -FullHD -ProfileGpu -Output .local/gallery-profile
```

The suite checks counts/topology, motion of all 90 gallery meshes, skin deformation and bounds, mixed instancing, finite light visibility, emission, history invalidation, ray queues, transport accumulation and safe probe-pool exhaustion.

The current suite has 31 passing Rust CPU/GPU tests, plus `python tools/test-diffuse-motion.py` for two receiver-correspondence checks. Current interaction acceptance and older image experiments are distinguished in the [closeout record](benchmarks/diffuse-motion/README.md). Performance reruns above remain optional; none was claimed for the final reconstruction build.

`timings.csv` contains unprofiled frame intervals including CPU/UI/presentation, excluding warmup and screenshot I/O. `passes.csv` and `summary.csv` contain CUDA stream-event ranges including submission gaps, not isolated kernel execution times. GI excludes geometry, GBuffer and direct composition; history is reported separately and in the combined GI total. Configuration files, screenshots and convergence reports record the setup. Set `CHANDRA_GPU_DEBUG_SYNC=1` with profiling only for stage-by-stage GPU diagnostics; these are not performance runs.

The [original dynamic-lab baseline](benchmarks/chromatic-gallery-rtx5070ti/README.txt) recorded 108.5 FPS in the combined scenario, 6.33 ms GI including history and 1.89 ms geometry updates. The [optimization report](benchmarks/1080p-single-frame/README.md) records **17 passing tests**, new 4,200-frame scenario benchmarks, a 3,600-frame motion run, individual GPU ranges and linear-GI quality comparisons. The optimized default temporal path reaches 124.2 FPS in the combined scenario with 0.87 ms geometry updates. The faster experimental single-frame path does not meet the unchanged-quality requirement.

## Optional external scenes

**Intel New Sponza is not included:** no Intel model, textures or rendered captures are distributed in this fork. Its optional importer requires an explicit path or `CHANDRA_INTEL_SPONZA`. The public default scene does not depend on it.

```powershell
./run-lighting-lab.ps1 -IntelSponza -ScenePath 'D:/external/NewSponza_Main_glTF_003.gltf' -FullHD -Cubes 100
```

The optional importer uses base-color textures of opaque primitives and excludes transparent decals; it does not implement the source renderer's full material/postprocessing stack. `-RigidRefit` selects the older per-object BLAS path for the small courtyard or optional Intel scene. Legacy assets already in upstream Git history remain available through `-Sponza` and the original viewer. New external assets and local captures are ignored.

## Upstream research

Chandra implements sparse, hashmap-based Radiance Cascades and ray splitting. See the [paper](https://arxiv.org/abs/2607.20384) and [upstream repository](https://github.com/entropylost/chandra). Original DDGI, RIS and optional denoising experiments remain in `src/`. Join the [Radiance Cascades Discord](https://discord.gg/EF9JfcEJPd) for discussion.
