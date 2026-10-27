"""Actual same-pose on/off renders; no lighting subtraction or exposure boost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=720)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
exe = ROOT / 'target/debug/lighting_lab.exe'
env = os.environ.copy()
env['PATH'] = str(ROOT / '.local/cuda/bin') + os.pathsep + env['PATH']
receipts = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def png(path):
    a = np.fromfile(path, dtype='<f4').reshape(args.height, args.width, 4)[..., :3]
    assert np.isfinite(a).all() and a.min() >= -1e-6
    a = np.maximum(a, 0)
    return Image.fromarray(np.uint8(np.clip((a / (1 + a)) ** (1 / 2.2), 0, 1) * 255))

for name, view, dolly in [('console-close', 3, 0), ('console-detail', 3, 9), ('console-left', 2, 15)]:
    pictures = []
    for enabled in [False, True]:
        label = 'on' if enabled else 'off'
        dest = out / f'{name}-{label}-raw'
        command = [str(exe), '--scene-worker', '--pica-pica', '--static', '--view', str(view),
                   '--camera-dolly', str(dolly), '--quality-check', '--specular-diagnostics',
                   '--quality-width', str(args.width), '--quality-height', str(args.height),
                   '--quality-frames', '240', '--quality-tail', '1', '--quality-fps', '60',
                   '--output', str(dest)]
        if not enabled:
            command.append('--no-specular')
        digest = sha(exe)
        print('Capture', name, label, flush=True)
        with (out / f'{name}-{label}.log').open('w') as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
        assert digest == sha(exe), 'Binary changed during capture'
        receipts.append({'arguments': command, 'binary_sha256': digest,
                         'scene_sha256': sha(ROOT / 'external/PicaPica/scene.gltf')})
        im = png(dest / 'frame-240.rgba32f')
        im.save(out / f'{name}-{label}.png')
        pictures.append(im)
        if enabled:
            png(dest / 'specular-filtered-240.rgba32f').save(out / f'{name}-reflections.png')
    panel = Image.new('RGB', (args.width * 2, args.height + 32), (18, 20, 24))
    draw = ImageDraw.Draw(panel)
    for i, im in enumerate(pictures):
        panel.paste(im, (i * args.width, 32))
        draw.text((i * args.width + 12, 9), 'Indirect reflections ' + ['OFF', 'ON'][i], fill='white')
    panel.save(out / f'{name}-comparison.png')
    (out / 'receipt.json').write_text(json.dumps(receipts, indent=2))
