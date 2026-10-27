"""Sequential PICA 1080p profiling with immutable binary/scene receipts.

Close competing GPU workloads before running. No process is closed by this tool.
Frame averages include presentation; stage times are CUDA event ranges.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'target/debug/lighting_lab.exe'
SCENE = ROOT / 'external/PicaPica/scene.gltf'
parser = argparse.ArgumentParser()
parser.add_argument('--cases', nargs='+')
parser.add_argument('--output', required=True)
parser.add_argument('--samples', type=int, default=4)
parser.add_argument('--specular-batch', type=int, choices=[1,2,4,8,16,32])
parser.add_argument('--specular-registers', type=int)
options = parser.parse_args()
assert 1 <= options.samples <= 128
selected = options.cases
OUT = ROOT / options.output
OUT.mkdir(exist_ok=bool(selected))
env = os.environ.copy()
env['PATH'] = str(ROOT / '.local/cuda/bin') + os.pathsep + env['PATH']
flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

def query():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.current.graphics', '--format=csv'], text=True, creationflags=flags).strip()

def processes():
    return subprocess.check_output(['powershell', '-NoProfile', '-Command', "Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('Albor.Runtime.exe','lighting_lab.exe') } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json"], text=True, creationflags=flags).strip()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

initial = {'gpu': query(), 'processes': processes(), 'binary_sha256': sha(EXE), 'scene_gltf_sha256': sha(SCENE)}
assert not initial['processes'], 'Another lighting_lab or Albor runtime is active; stop it before profiling'
if not (OUT / 'environment-before.json').exists():
    (OUT / 'environment-before.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
cases = {'static': [], 'camera': ['--quality-camera-track','combined'], 'sun': ['--quality-light-track','oscillate'], 'combined': ['--quality-camera-track', 'combined', '--quality-light-track', 'oscillate'], 'static-no-specular': ['--no-specular'], 'camera-no-specular':['--quality-camera-track','combined','--no-specular'], 'sun-no-specular':['--quality-light-track','oscillate','--no-specular'], 'combined-no-specular': ['--quality-camera-track', 'combined', '--quality-light-track', 'oscillate', '--no-specular']}
if selected:
    cases = {name: cases[name] for name in selected}
for name, extra in cases.items():
    dest = OUT / name
    dest.mkdir()
    args = [str(EXE), '--scene-worker', '--pica-pica', '--1080p', '--static', '--view', '1', '--profile', '--profile-frames', '120', '--specular-samples', str(options.samples), '--output', str(dest), *extra]
    if options.specular_batch is not None:
        args += ['--specular-batch',str(options.specular_batch)]
    if options.specular_registers is not None:
        args += ['--specular-registers',str(options.specular_registers)]
    receipt = {'case': name, 'arguments': args, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'binary_sha256': sha(EXE), 'scene_gltf_sha256': sha(SCENE), 'gpu_before': query(), 'processes_before': processes()}
    assert not receipt['processes_before'], 'Another renderer started between cases'
    start = time.perf_counter()
    telemetry = []
    print('START', name, flush=True)
    with (dest / 'run.log').open('w', encoding='utf-8') as log:
        run = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        receipt['renderer_pid'] = run.pid
        while run.poll() is None:
            telemetry.append({'elapsed_s': time.perf_counter() - start, 'gpu': query()})
            time.sleep(2)
    receipt.update(returncode=run.returncode, wall_seconds_including_startup=time.perf_counter()-start, binary_sha256_after=sha(EXE), gpu_after=query(), processes_after=processes())
    (dest / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (dest / 'gpu-telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')
    assert receipt['binary_sha256'] == receipt['binary_sha256_after'] == initial['binary_sha256']
    assert run.returncode == 0, name
    assert (dest / 'summary.csv').exists(), name
    print('DONE', name, (dest / 'summary.csv').read_text(), flush=True)
print('ALL DONE', query(), flush=True)
