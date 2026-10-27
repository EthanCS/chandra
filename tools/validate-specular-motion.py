"""Continuous PICA reflection stability captures, with same-frame dense samples.

The reference uses the same finite hybrid transport and 128 raw samples. It is
not path-tracing truth. Motion-compensated residuals and reference neighborhoods
distinguish unsupported impulses from actual moving highlights/disocclusions.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diffuse_checks", ROOT / "tools/validate-diffuse-motion.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
CASES = {
    "static": [],
    "camera": ["--quality-camera-track", "combined"],
    "sun": ["--quality-light-track", "drag"],
    "combined": ["--quality-camera-track", "combined", "--quality-light-track", "oscillate"],
    "fast-stop": ["--quality-light-track", "fast", "--quality-stop", "240"],
    "return": ["--quality-camera-track", "return"],
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(directory, kind, frame, width, height):
    value = np.fromfile(directory / f"{kind}-{frame:03}.rgba32f", dtype="<f4").reshape(height, width, 4)
    assert np.isfinite(value).all(), (directory, kind, frame)
    return value

def reprojection(position, normal, previous_position, previous_normal, previous_view):
    """Cache geometry once for all six radiance channels; same checks as diffuse."""
    matrix=np.array(previous_view['transform_columns'],dtype=np.float32)
    local=(position[...,:3]-np.array(previous_view['position'],np.float32)) @ matrix.T
    height,width=position.shape[:2]
    xy=local[...,:2]/np.maximum(local[...,2,None],1e-6)/previous_view['scaling']+np.array([width,height])*.5-.5
    xy=np.nan_to_num(xy,nan=-100,posinf=-100,neginf=-100)
    base=np.floor(np.clip(xy,-100,100000)).astype(np.int32)
    valid=(position[...,3]>0)&(local[...,2]>0)&(base[...,0]>=0)&(base[...,1]>=0)&(base[...,0]+1<width)&(base[...,1]+1<height)
    fraction=xy-base
    weight_sum=np.zeros((height,width),np.float32)
    footprint=np.maximum(.05,local[...,2]*previous_view['scaling']*3)
    taps=[]
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        x=np.clip(base[...,0]+dx,0,width-1);y=np.clip(base[...,1]+dy,0,height-1)
        p=previous_position[y,x];n=previous_normal[y,x]
        compatible=(p[...,3]==position[...,3])&(np.sum(n[...,:3]*normal[...,:3],axis=-1)>.95)
        compatible &= np.linalg.norm(p[...,:3]-position[...,:3],axis=-1)<footprint
        compatible &= np.abs(np.sum((p[...,:3]-position[...,:3])*normal[...,:3],axis=-1))<np.maximum(.03,footprint*.1)
        weight=(fraction[...,0] if dx else 1-fraction[...,0])*(fraction[...,1] if dy else 1-fraction[...,1])
        weight=weight*valid*compatible
        weight_sum+=weight
        taps.append((x,y,weight))
    def warp(value):
        output=np.zeros((height,width),np.float32)
        for x,y,weight in taps:output+=value[y,x]*weight
        return output/np.maximum(weight_sum,1e-6)
    return warp,valid&(weight_sum>.75)

def surface_support(position,normal):
    support=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if not(dx or dy):continue
            q=np.roll(position,(dy,dx),axis=(0,1));n=np.roll(normal,(dy,dx),axis=(0,1))
            valid=(position[...,3]>0)&(q[...,3]==position[...,3])&(np.sum(n[...,:3]*normal[...,:3],axis=-1)>.98)
            valid &= np.abs(np.sum((q[...,:3]-position[...,:3])*normal[...,:3],axis=-1))<.05
            if dy>0:valid[:dy]=False
            if dy<0:valid[dy:]=False
            if dx>0:valid[:,:dx]=False
            if dx<0:valid[:,dx:]=False
            support.append((dy,dx,valid))
    return support

def reference_bounds(value,support):
    low=value.copy();high=value.copy()
    for dy,dx,valid in support:
        v=np.roll(value,(dy,dx),axis=(0,1))
        low=np.where(valid,np.minimum(low,v),low);high=np.where(valid,np.maximum(high,v),high)
    return low,high

def analyze(directory, width, height, end, tail, reference_directory=None, candidate_label="Adaptive GGX + reconstruction"):
    previous = None
    metrics = {name: {"bright_impulses": 0, "dark_impulses": 0, "valid_pairs": 0, "squared_delta": 0., "display_error": [], "candidate_energy": 0., "reference_energy": 0., "peak_linear": 0., "events": []} for name in ("full", "specular", "base", "rough_specular")}
    previews = []
    first = end - tail + 1
    for frame in range(first, end + 1):
        cdir, rdir = directory / "candidate", reference_directory or directory / "reference"
        cfull, rfull = [load(d, "frame", frame, width, height) for d in (cdir, rdir)]
        cspec = load(cdir, "specular-filtered", frame, width, height)
        rspec = load(rdir, "specular-raw", frame, width, height)
        pos, normal = [load(cdir, kind, frame, width, height) for kind in ("position", "normal")]
        view = json.loads((cdir / f"camera-{frame:03}.json").read_text())
        assert view == json.loads((rdir / f"camera-{frame:03}.json").read_text())
        assert np.array_equal(cfull[..., 3], rfull[..., 3])
        # The final composition may have its own temporal stabilization.
        # Read the independently captured base layer instead of subtracting a
        # pre-composition reflection from a post-composition image.
        cbase = load(cdir, "base-lighting", frame, width, height) if (cdir / f"base-lighting-{frame:03}.rgba32f").exists() else cfull - cspec
        values = {"full": (cfull, rfull), "specular": (cspec, rspec), "base": (cbase, rfull - rspec), "rough_specular": (cspec, rspec)}
        rough = load(cdir,"specular-diagnostics",frame,width,height)[...,0]>=.6
        now = {}
        mask = pos[..., 3] > 0
        if previous is not None:
            pp,pn,pv,old=previous
            warp,valid=reprojection(pos,normal,pp,pn,pv)
            footprint_support=surface_support(pos,normal)
        for name, (candidate, reference) in values.items():
            c, r = checks.lum(checks.display(candidate)), checks.lum(checks.display(reference))
            residual = c - r
            mask = (pos[..., 3] > 0) & (rough if name=="rough_specular" else True)
            m = metrics[name]
            m["peak_linear"] = max(m["peak_linear"], float(candidate[..., :3].max()))
            if mask.any():
                m["display_error"].append(float(np.abs(residual[mask]).mean()))
            m["candidate_energy"] += float(checks.lum(candidate)[mask].sum())
            m["reference_energy"] += float(checks.lum(reference)[mask].sum())
            if previous is not None:
                warped=warp(old[name][1])
                old_c=warp(old[name][0])
                delta = residual - warped
                lo, hi = reference_bounds(r, footprint_support)
                # A >10% display-luminance jump, unsupported by dense same-frame
                # reference anywhere in the same-surface 3x3 pixel footprint.
                selected = valid & mask
                bright = selected & (delta > .10) & (c - old_c > .10) & (c - hi > .10)
                dark = selected & (delta < -.10) & (c - old_c < -.10) & (lo - c > .10)
                m["bright_impulses"] += int(bright.sum())
                m["dark_impulses"] += int(dark.sum())
                m["valid_pairs"] += int(selected.sum())
                m["squared_delta"] += float(np.square(delta[selected]).sum())
                ys, xs = np.where(bright | dark)
                if len(ys):
                    order = np.argsort(-np.abs(delta[ys, xs]))[:4]
                    m["events"].extend(dict(frame=frame, x=int(xs[i]), y=int(ys[i]), delta=float(delta[ys[i], xs[i]]), candidate=float(c[ys[i], xs[i]]), reference=float(r[ys[i], xs[i]])) for i in order)
            now[name] = (c, residual)
        previous = pos, normal, view, now
        if frame % 2 == 0:
            side = np.concatenate([checks.display(cfull[..., :3]), checks.display(rfull[..., :3])], axis=1)
            im = Image.fromarray(np.uint8(np.clip(side, 0, 1) * 255))
            canvas = Image.new("RGB", (2 * width, height + 24), (18, 20, 24))
            canvas.paste(im, (0, 24))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f"{candidate_label} / {frame}", fill="white")
            draw.text((width + 8, 6), "128 spp raw hybrid reference", fill="white")
            previews.append(canvas)
    for m in metrics.values():
        m["compensated_residual_rms"] = float(np.sqrt(m.pop("squared_delta") / max(m["valid_pairs"], 1)))
        m["display_mae"] = float(np.mean(m.pop("display_error")))
        m["relative_mean_energy"] = m.pop("candidate_energy") / max(m.pop("reference_energy"), 1e-12) - 1.
        m["events"] = sorted(m["events"], key=lambda e: -abs(e["delta"]))[:20]
    if previews:
        previews[0].save(directory / "comparison.png")
        encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width*2}x{height+24}", "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(directory / "comparison.mp4")], stdin=subprocess.PIPE)
        for preview in previews:
            encoder.stdin.write(preview.tobytes())
        encoder.stdin.close()
        assert encoder.wait() == 0, "Preview video encoding failed"
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--reference-root", type=Path, help="Reuse immutable dense captures when only reconstruction or low-sample allocation changes")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--tail", type=int, default=190)
    parser.add_argument("--camera-dolly", type=float, default=0)
    parser.add_argument("--view", type=int, choices=[0,1,2,3], default=1)
    parser.add_argument("--simulation-fps", type=int, choices=[20,30,60,120], default=60)
    parser.add_argument("--reference-exact-feedback", action="store_true",
                        help="Use exact angular integration for the reference's diffuse cache feedback")
    parser.add_argument("--diffuse-samples", type=int, help="Override candidate diffuse update budget")
    parser.add_argument("--reflection-scale", type=int, choices=[1,2,4], default=2)
    parser.add_argument("--specular-quality", action="store_true", help="Validate the original adaptive path")
    args = parser.parse_args()
    assert 1 <= args.samples <= 128 and 2 <= args.tail <= args.frames
    exe = ROOT / "target/debug/lighting_lab.exe"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(ROOT / ".local/cuda/bin") + os.pathsep + env["PATH"]
    result = {"scene_sha256": sha(ROOT / "external/PicaPica/scene.gltf"), "candidate_samples": args.samples, "width": args.width, "height": args.height, "frames": args.frames, "tail": args.tail, "camera_dolly": args.camera_dolly, "simulation_fps": args.simulation_fps, "analysis_sha256": sha(Path(__file__)), "reference": "128 spp raw, same finite hybrid transport; not full path tracing", "cases": {}}
    for case in args.cases:
        directory = output / case
        directory.mkdir(exist_ok=True)
        for variant, samples in (("candidate", args.samples), ("reference", 128)):
            dest = directory / variant
            command = [str(exe), "--scene-worker", "--pica-pica", "--static", "--view", str(args.view), "--quality-check", "--quality-surfaces", "--specular-diagnostics", "--quality-width", str(args.width), "--quality-height", str(args.height), "--quality-frames", str(args.frames), "--quality-tail", str(args.tail), "--specular-samples", str(samples), "--output", str(dest), *CASES[case]]
            command += ["--camera-dolly", str(args.camera_dolly), "--quality-fps", str(args.simulation_fps)]
            if variant == "reference":
                command.append("--no-specular-filter")
                if args.reference_exact_feedback:
                    command.append("--exact-diffuse-feedback")
            else:
                command += ["--reflection-scale", str(args.reflection_scale)]
                if args.diffuse_samples is not None:
                    command += ["--diffuse-samples", str(args.diffuse_samples)]
                if args.specular_quality:
                    command.append("--specular-quality")
            receipt = directory / (variant + "-receipt.json")
            if variant == "reference" and args.reference_root:
                saved = json.loads((args.reference_root.resolve() / case / "reference-receipt.json").read_text())
                assert saved["scene_sha256"] == result["scene_sha256"], "Reference asset differs; recapture after material/geometry changes"
                assert (args.reference_root.resolve() / case / "reference").is_dir(), "Reference must point to an original capture directory"
                saved["reused_from"] = str(args.reference_root.resolve() / case / "reference")
                receipt.write_text(json.dumps(saved, indent=2))
                continue
            if args.capture:
                binary = sha(exe)
                print(f"Capturing {case}/{variant}", flush=True)
                with (directory / (variant + ".log")).open("w") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                assert binary == sha(exe), "Binary changed during capture"
                receipt.write_text(json.dumps({"binary_sha256": binary, "arguments": command, "scene_sha256": sha(ROOT / "external/PicaPica/scene.gltf")}, indent=2))
            assert receipt.exists(), "Capture receipt missing; use --capture"
        reference_directory=args.reference_root.resolve() / case / "reference" if args.reference_root else None
        result["cases"][case] = analyze(directory, args.width, args.height, args.frames, args.tail, reference_directory, "Adaptive GGX" if args.specular_quality else "RC + bounded reflections")
        print(case, json.dumps({k: {n: v for n, v in m.items() if n != "events"} for k, m in result["cases"][case].items()}), flush=True)
        (output / "summary.json").write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
