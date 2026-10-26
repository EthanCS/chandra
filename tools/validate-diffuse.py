"""Serial, binary-bound diffuse GI acceptance; raw frames stay in .local.

References here measure same-pose convergence, not path-tracing accuracy.
This suite deliberately does not certify arbitrary camera-motion flicker.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash():
    files = [ROOT / p for p in ("Cargo.toml", "Cargo.lock", "run-lighting-lab.ps1")]
    for directory in ("src", "tests", "tools"):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def command_output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def load(directory, frame):
    path = directory / f"frame-{frame:03}.rgba32f"
    data = np.fromfile(path, dtype="<f4").reshape(600, 960, 4)
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite pixels: {path}")
    return data


def display(data):
    return np.power(np.maximum(data, 0) / (1 + np.maximum(data, 0)), 1 / 2.2)


def required_artifacts(directory, options):
    files = []
    if "--quality-check" in options or "--roam-test" in options:
        end_option = "--quality-frames" if "--quality-frames" in options else "--roam-frames"
        end = int(options[options.index(end_option)+1])
        frames = {f for f in (0,1,2,16,64,200,end) if f <= end}
        if "--quality-tail" in options:
            frames.update(range(max(0,end-int(options[options.index("--quality-tail")+1])+1),end+1))
        files.extend([directory/"capacity.txt", directory/"arguments.txt", directory/"quality.txt"])
        width, height = (1920,1080) if "--1080p" in options else (960,600) if "--small" in options else (1280,800)
        for frame in sorted(frames):
            raw = directory/f"frame-{frame:03}.rgba32f"
            if not raw.exists() or raw.stat().st_size != width*height*16:
                raise RuntimeError(f"Missing or truncated capture: {raw}")
            files.extend([raw, directory/f"camera-{frame:03}.json"])
            if "--courtyard" in options:
                files.append(directory/f"surface-{frame:03}.rgba32f")
    elif "--benchmark" in options:
        files.extend([directory/"timings.csv", directory/"convergence.txt"])
    elif "--stress-test" in options:
        files.extend([directory/"results.txt", directory/"intervals.csv"])
    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing capture artifact: {path}")
    return {p.relative_to(directory).as_posix(): sha(p) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/diffuse-acceptance/final")
    parser.add_argument("--report", type=Path, default=ROOT / "benchmarks/diffuse/validation.json")
    parser.add_argument("--test-log", type=Path, default=ROOT / ".local/diffuse-acceptance/tests.log")
    args = parser.parse_args()
    args.output = args.output.resolve()
    exe = ROOT / "target/debug/lighting_lab.exe"
    binary_hash, sources = sha(exe), source_hash()
    hardware = command_output("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    env = os.environ.copy()
    env["PATH"] = str(ROOT / ".local/cuda/bin") + os.pathsep + env["PATH"]
    env.pop("CHANDRA_GPU_DEBUG_SYNC", None)
    cases = {
        "gallery-benchmark": ["--gallery", "--1080p", "--benchmark"],
        "gallery-stress": ["--gallery", "--1080p", "--stress-test"],
        "courtyard-benchmark": ["--courtyard", "--small", "--benchmark"],
        "capacity": ["--white-room", "--small", "--roam-test", "--roam-frames", "2400", "--probe-budget", "1024"],
        "capacity-skip": ["--gallery", "--small", "--quality-check", "--quality-frames", "16", "--probe-budget", "256"],
    }
    for view in range(4):
        cases[f"white-{view}"] = ["--white-room", "--small", "--quality-check", "--view", str(view), "--quality-frames", "200"]
    for mode in ("single", "multi"):
        common = ["--courtyard", "--small", "--quality-check", "--quality-motion", "--geometry-only", "--indirect-only"]
        if mode == "single":
            common += ["--single-bounce"]
        cases[f"{mode}-motion"] = common + ["--quality-frames", "240", "--quality-tail", "64"]
        cases[f"{mode}-settled"] = common + ["--quality-stop", "241", "--quality-frames", "600"]
        cases[f"{mode}-cleared"] = common + ["--quality-stop", "241", "--quality-stop-clear", "--quality-frames", "600"]
    cases["lights-off"] = ["--courtyard", "--small", "--quality-check", "--indirect-only", "--quality-light-off", "201", "--quality-frames", "241"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, options in cases.items():
        directory = args.output / name
        command = [str(exe), "--scene-worker", *options, "--output", str(directory)]
        expected = {"binary_sha256": binary_hash, "source_sha256": sources, "hardware": hardware, "command": command}
        manifest_path = args.output / f"{name}.manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        reuse = args.reuse and all(manifest.get(k) == v for k,v in expected.items())
        if reuse and manifest.get("files") != required_artifacts(directory, options):
            raise RuntimeError(f"Capture contents changed since collection: {name}")
        if args.capture and not reuse:
            if directory.exists():
                raise RuntimeError(f"Capture already exists; choose a fresh output directory: {directory}")
            print(f"Capturing {name}", flush=True)
            with (args.output / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if sha(exe) != binary_hash or source_hash() != sources:
                raise RuntimeError("Binary or rendering sources changed during serial capture")
            manifest = {**expected, "files": required_artifacts(directory, options)}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        elif not reuse:
            raise RuntimeError(f"Use --capture for a fresh collection or --reuse with matching manifests: {name}")
        manifests[name] = manifest

    checks = []

    def check(name, passed, **measurements):
        checks.append({"name": name, "passed": bool(passed), **measurements})

    white_images = []
    for view in range(4):
        data = load(args.output / f"white-{view}", 200)
        surface = data[..., 3] > 0.5
        energy = float(data[..., :3][surface].mean())
        maximum = float(data[..., :3][surface].max())
        check(f"white-{view}-energy", maximum <= 1e-6 if view == 0 else energy > 1e-4,
              mean_linear=energy, max_linear=maximum)
        white_images.append(Image.fromarray((display(data[..., :3]) * 255).clip(0,255).astype("uint8")).resize((480,300)))
    montage = Image.new("RGB", (960,600))
    for i, image in enumerate(white_images):
        montage.paste(image, ((i%2)*480, (i//2)*300))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.report.parent / "white-room.png")

    for mode in ("single", "multi"):
        moving = load(args.output / f"{mode}-motion", 240)
        settled = load(args.output / f"{mode}-settled", 600)
        cleared = load(args.output / f"{mode}-cleared", 600)
        mask = (settled[...,3] > 0.5) & (cleared[...,3] > 0.5)
        if not np.array_equal(settled[...,3], cleared[...,3]):
            raise RuntimeError("Convergence comparison changed the surface mask")
        ref = cleared[..., :3][mask]
        for label, data in (("at-stop", moving), ("settled", settled)):
            sample = data[..., :3][mask]
            mae = float(np.abs(display(sample) - display(ref)).mean())
            bias = float(sample.mean() / max(float(ref.mean()), 1e-12) - 1)
            check(f"{mode}-{label}-convergence", mae <= 0.02 and abs(bias) <= 0.10,
                  display_mae=mae, linear_energy_bias=bias,
                  reference="same pose, transport cleared at frame 241, converged through frame 600")
        frames = sorted((args.output / f"{mode}-motion").glob("frame-*.rgba32f"))
        check(f"{mode}-motion-finite", all(np.isfinite(np.fromfile(p,dtype="<f4")).all() for p in frames),
              captured_frames=len(frames), local_flicker_acceptance="not assessed by this finite-value check")
        tracked = np.stack([np.fromfile(args.output/f"{mode}-motion"/f"surface-{f:03}.rgba32f", dtype="<f4").reshape(-1,4)
                            for f in range(177,241)])
        pairs = (tracked[1:,:,3] > 0.5) & (tracked[:-1,:,3] > 0.5)
        delta = display(tracked[1:,:,:3])-display(tracked[:-1,:,:3])
        rms = float(np.sqrt(np.square(delta[pairs]).mean())) if pairs.any() else None
        check(f"{mode}-same-surface-capture", int(pairs.sum()) >= 1000,
              matched_point_pairs=int(pairs.sum()), display_frame_difference_rms=rms,
              interpretation="Fixed barycentric points; includes real illumination changes as the box rotates. No zero-flicker claim.")

    before = load(args.output / "lights-off", 200)
    after = load(args.output / "lights-off", 241)
    surface = before[...,3] > 0.5
    remainder = float(after[...,:3][surface].mean() / max(float(before[...,:3][surface].mean()), 1e-12))
    check("lights-off-40-frames", remainder <= 0.01, remaining_surface_energy_ratio=remainder)
    timings = {}
    for name in ("gallery-benchmark", "courtyard-benchmark"):
        with (args.output/name/"timings.csv").open() as stream:
            timings[name] = list(csv.DictReader(stream))
        check(name, len(timings[name]) == 7, scenarios=len(timings[name]))
    # StressTest asserts nonfinite output and ray capacity on every frame.
    log = (args.output / "gallery-stress.log").read_text()
    check("gallery-3600-frame-stress", "Combined motion stress test PASS" in log,
          log=str(args.output/"gallery-stress.log"))
    capacity = (args.output / "capacity/capacity.txt").read_text().strip()
    check("capacity-roam-2400", "gi_available=true" in capacity, state=capacity)
    skipped = (args.output / "capacity-skip/capacity.txt").read_text().strip()
    load(args.output / "capacity-skip", 16)
    check("capacity-safe-skip", "gi_available=false" in skipped and "skipped_gi_frames=17" in skipped,
          state=skipped, rendered="finite direct light while GI allocation remains exhausted")
    test_log = args.test_log.read_text()
    import re
    passed_tests = sum(int(n) for n in re.findall(r"test result: ok\. (\d+) passed", test_log))
    check("regression-suite", passed_tests >= 26 and "test result: FAILED" not in test_log,
          passed_tests=passed_tests, log=str(args.test_log), log_sha256=sha(args.test_log))
    report = {
        "scope": "Diffuse GI foundations; indirect specular and ReSTIR remain unimplemented",
        "git_head": command_output("git", "rev-parse", "HEAD"),
        "working_tree": command_output("git", "status", "--short"),
        "source_sha256": sources, "binary_sha256": binary_hash,
        "sefirot_revision": command_output("git", "-C", "../sefirot", "rev-parse", "HEAD"),
        "dependency_contract": {"cargo": "nightly-2026-03-02", "rustc": "nightly-2026-10-03",
                                "luisa": "d270b5296f600a3147efedc2823766ff989d1b9e", "cuda": "12.8.1",
                                "assets": "procedural Gallery, courtyard and White Room"},
        "hardware": hardware,
        "simulation_dt": "1/60 s", "tone_map_for_metrics": "Reinhard + gamma 2.2 at shared exposure",
        "captures": manifests, "checks": checks, "timings": timings,
        "failures": [c for c in checks if not c["passed"]],
        "unverified": ["Arbitrary camera motion and light dragging visual acceptance",
                       "Reference-supported same-surface temporal impulse acceptance", "Path-tracing ground-truth accuracy",
                       "PICA, RobotExpressive and external asset integration"],
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "failures": report["failures"], "report": str(args.report)}, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    sys.exit(main())
