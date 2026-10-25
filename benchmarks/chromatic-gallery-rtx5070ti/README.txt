Chromatic Gallery validation — 2026-10-26
NVIDIA RTX 5070 Ti 16 GB / driver 617.42 / AMD Ryzen 9 9950X
1920x1080; uncapped; VSync off; optimized dev build.
30 moving point lights, 30 moving rigid objects, 30 moving emitters, 30 moving skinned objects.
Animated sun and camera; 90 independent history regions; 131072 base probes; 10 cm base grid; 4 cascades; multiple bounces enabled.
No external models, textures or HDR maps. This folder contains only procedural-scene measurements.

15 CPU/CUDA tests passed. Tests include all 90 meshes moving within their history bounds, 100 mixed rigid instances, finite point-light visibility and safe probe-pool exhaustion.

frame-timings.csv: 4200 frames / seven 600-frame cases. Each case discards 150 warmup frames; screenshot I/O intervals are excluded. The sun case animates both sun and local lights. Combined motion mean 9.2152 ms (108.52 FPS), p95 9.6810 ms.
stress.txt: 3600 frames of continuously combined motion. Mean 9.620 ms (103.9 FPS), p95 10.221 ms. Early warmup excluded; includes screenshot stalls at frames 1200/2400, so it is not directly interchangeable with the scenario benchmark.
gpu-passes.csv / gpu-summary.csv: separate CUDA stream-event run, 200 warmup + 600 measured frames. Events include submission gaps, not just isolated kernel execution. GI main 4.3241 ms, GI+history 6.3311 ms, geometry 1.8949 ms.
convergence.txt: error after stopping motion decreases to 0.0660 / 255 after 120 stationary frames.

All benchmark/stress frames passed finite-output and ray-capacity checks. No probe-pool exhaustion at the tested views. This is a temporal GI implementation; these measurements do not establish path-traced ground-truth accuracy or performance on other GPUs.
