# Chandra Dynamic GI Lab

A dynamic-scene extension of [entropylost/chandra](https://github.com/entropylost/chandra), the reference implementation of [Split Radiance Cascades](https://arxiv.org/abs/2607.20384). The original research renderer is retained. This fork adds an interactive lighting lab, rigid instancing, GPU skinning, emissive materials, moving local lights, dynamic history invalidation and GPU timing tools.

**1080p optimization update:** the experimental single-frame mode reaches **150.3 FPS** in the combined scenario and **143.9 FPS** over continuous motion on an RTX 5070 Ti, with **4.87 ms GI**. It clears transport every frame, but residual spatial sampling error remains. **120 FPS + no accumulation + no noise + unchanged quality has not been achieved.** The existing temporal path remains the default. See the [measurements and quality comparisons](benchmarks/1080p-single-frame/README.md).

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
# Day/night presets disable the sun; local lights remain independently adjustable.
./run-lighting-lab.ps1 -Gallery -FullHD -EmissiveDay
./run-lighting-lab.ps1 -Gallery -FullHD -EmissiveNight
# Experimental frame-local transport, 8 path vertices, 1 sample per surface pixel.
./run-lighting-lab.ps1 -Gallery -FullHD -SingleFrame
# More current-frame samples; this exceeds the 120 FPS budget on the tested GPU.
./run-lighting-lab.ps1 -Gallery -FullHD -SingleFrame -Samples 8
```

Default resolution is 1280×800; `-FullHD` is 1920×1080 and `-Small` is 960×600. The `lighting_lab` binary is separate from the original `chandra` research viewer.

## Controls

- Sliders: sun direction/color/intensity, sky, local-light strength, emission and exposure.
- Toggles: sun/local-light motion, rigid/emissive motion, skinning and camera motion.
- Compare full GI, direct illumination, indirect illumination and surface normals.
- `WASD`: move; arrows or right-drag: look; `Space`/`Shift`: up/down; `Ctrl`: slow.
- `R`: reset view; `Esc`: close. Three camera presets include close-up views.
- Default mode: `Clear GI history` resets temporal accumulation; indirect light needs several frames to settle. Single-frame mode: `Rebuild GI allocation` resets spatial allocation; it does not accumulate radiance across frames.

Day/night presets use the same emission and exposure. Day scales the constant sky by 0.18; night sets sky to zero. The gallery's 30 local lights remain adjustable with the Local lights slider. These are lighting presets, not a physical atmosphere simulation. Radiance is in relative linear units, not calibrated lumens/lux; no bloom is added.

## Implementation

- Rigid BLASes retain immutable bind geometry. One transform batch updates the TLAS; one GPU dispatch updates world-space shading vertices for all rigid meshes.
- GPU skinning supports four normalized weights per vertex. The gallery batches 30 joint palettes and deformation dispatches; groups of three neighboring skins share a BLAS, retaining original triangle/material identity. Only deformed BLASes are refitted, with periodic rebuilds to limit traversal degradation. `--unbatched-skin` retains the reference path. Static BLASes are retained. Empty graph stages preserve execution order without extra GPU waits.
- Material-indexed emitted radiance is independent of diffuse reflectance, for camera and GI ray shading. There is no per-ray emitter-list scan.
- A conservative current-frame light grid selects candidates without changing light order or the exact range/cosine/shadow tests. `--no-light-grid` retains the full loop.
- Moving geometry invalidates swept regions; changing lights decay history. Radiance is held fixed during each frame's transport accumulation.
- Fused tracing/deposit, warp-aggregated counters, cached parent links/quadrature, coalesced history updates and opaque any-hit shadows reduce work. `--unfused` retains the separate deposit queue.
- The gallery reserves 131,072 base probes with the same 10 cm spatial setting. Exhausted cascade pools report a capacity error instead of accessing another pool's memory.
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
