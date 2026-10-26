"""Compare current-frame linear GI captures, never UI screenshots.

Run lighting_lab --quality-check --indirect-only for each candidate and a
--single-frame --path-depth 12 --samples 32 reference at the identical pose.
The reference is densely sampled Split RC, not an unbiased path tracer.
Alpha in these RGBA32F files is 1 for surfaces and 0 for sky.
"""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("candidates", nargs="+", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

def load(path):
    values = np.fromfile(path, dtype="<f4")
    if values.size % 4:
        raise ValueError(f"Invalid RGBA32F capture: {path}")
    return values.reshape(-1, 4)

reference = load(args.reference)
mask = reference[:, 3] > 0
if not mask.any() or not np.isfinite(reference).all():
    raise ValueError("Reference has no surfaces or contains nonfinite values")
rows = []
for path in args.candidates:
    candidate = load(path)
    if candidate.shape != reference.shape or not np.array_equal(candidate[:, 3], reference[:, 3]):
        raise ValueError(f"Capture dimensions/surface mask differ: {path}")
    if not np.isfinite(candidate).all():
        raise ValueError(f"Nonfinite candidate: {path}")
    delta = candidate[mask, :3] - reference[mask, :3]
    rows.append(dict(candidate=str(path), surface_pixels=int(mask.sum()),
                     mae=float(np.abs(delta).mean()),
                     rmse=float(np.sqrt(np.square(delta).mean())),
                     p95_absolute_error=float(np.quantile(np.abs(delta), .95)),
                     mean_radiance=float(candidate[mask, :3].mean())))
result = json.dumps(dict(reference=str(args.reference), comparisons=rows), indent=2)
print(result)
if args.output:
    args.output.write_text(result + "\n", encoding="utf-8")
