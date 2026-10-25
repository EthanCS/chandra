"""Compare a candidate with both a reference and its unchanged-renderer repeat.

Usage: python tools/compare_lab_images.py BASELINE REPEAT CANDIDATE OUTPUT.json
The temporal Monte Carlo renderer is not bit deterministic. Bounds use its
measured repeatability, including local 64-pixel tiles, rather than exact hashes.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def pixels(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)[:, 320:]


def metrics(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Image dimensions differ: {a.shape} != {b.shape}")
    difference = np.abs(a - b)
    tiles = [float(difference[y:y+64, x:x+64].mean())
             for y in range(0, difference.shape[0], 64)
             for x in range(0, difference.shape[1], 64)]
    return {"mae": float(difference.mean()), "p99": float(np.percentile(difference, 99)),
            "max": int(difference.max()), "worst_tile_mae": max(tiles)}


baseline, repeat, candidate, output = map(Path, sys.argv[1:])
rows = []
for path in sorted(baseline.glob("*.png")):
    reference = pixels(path)
    noise = metrics(reference, pixels(repeat / path.name))
    delta = metrics(reference, pixels(candidate / path.name))
    passed = (delta["mae"] <= noise["mae"] * 1.5 + 0.05
              and delta["p99"] <= noise["p99"] + 1.0
              and delta["worst_tile_mae"] <= noise["worst_tile_mae"] * 1.5 + 0.25)
    rows.append({"image": path.name, "repeatability": noise, "candidate_difference": delta,
                 "within_repeatability_envelope": passed})
    print(f"{path.name}: MAE {delta['mae']:.4f} vs repeat {noise['mae']:.4f}; "
          f"worst tile {delta['worst_tile_mae']:.4f}; {'PASS' if passed else 'FAIL'}")
if not rows:
    raise ValueError("No reference PNGs found")
result = {"units": "RGB8 (0..255), excluding left 320 pixels",
          "baseline": str(baseline), "repeat": str(repeat), "candidate": str(candidate),
          "passed": all(row["within_repeatability_envelope"] for row in rows), "images": rows}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
sys.exit(0 if result["passed"] else 1)
