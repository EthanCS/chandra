"""Sequential PICA profiling (1080p by default) with immutable binary/scene receipts.

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
parser.add_argument('--width', type=int, default=1920)
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--view', type=int, choices=[0, 1, 2, 3], default=1)
parser.add_argument('--camera-dolly', type=float, default=0,
                    help='Move forward from the camera preset in scene units')
parser.add_argument('--simulation-fps', type=int, choices=[20,30,60,120], default=60)
parser.add_argument('--realtime', action='store_true', help='Use measured frame time, like interactive rendering')
parser.add_argument('--executable', type=Path, default=EXE)
parser.add_argument('--specular-batch', type=int, choices=[1,2,4,8,16,32])
parser.add_argument('--specular-registers', type=int)
parser.add_argument('--diffuse-registers', type=int)
parser.add_argument('--diffuse-samples', type=int)
parser.add_argument('--reflection-scale', type=int, choices=[1,2,4])
parser.add_argument('--specular-quality', action='store_true')
options = parser.parse_args()
EXE = options.executable.resolve()
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
    return subprocess.check_output(['powershell', '-NoProfile', '-Command', "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'Albor*' -or $_.Name -like 'lighting_lab*' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json"], text=True, creationflags=flags).strip()

def gpu_consumers():
    # WDDM applications may be absent from nvidia-smi's compute process list.
    script = """Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -SampleInterval 1 -MaxSamples 2 |
      ForEach-Object { $_.CounterSamples | Where-Object CookedValue -gt 2 | ForEach-Object {
        $taskPid = [int]([regex]::Match($_.InstanceName, 'pid_(\\d+)_').Groups[1].Value)
        [pscustomobject]@{pid=$taskPid;name=(Get-Process -Id $taskPid -ErrorAction SilentlyContinue).ProcessName;engine=$_.InstanceName;utilization=$_.CookedValue}
      }} | ConvertTo-Json"""
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', script], text=True, creationflags=flags).strip()
    return json.loads(raw) if raw else []

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

initial = {'gpu': query(), 'processes': processes(), 'gpu_engine_processes': gpu_consumers(), 'binary_sha256': sha(EXE), 'scene_gltf_sha256': sha(SCENE)}
assert not initial['processes'], 'Another lighting_lab or Albor runtime is active; stop it before profiling'
if not (OUT / 'environment-before.json').exists():
    (OUT / 'environment-before.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
cases = {'static': [], 'camera': ['--quality-camera-track','combined'], 'sun': ['--quality-light-track','oscillate'], 'combined': ['--quality-camera-track', 'combined', '--quality-light-track', 'oscillate'], 'static-no-specular': ['--no-specular'], 'camera-no-specular':['--quality-camera-track','combined','--no-specular'], 'sun-no-specular':['--quality-light-track','oscillate','--no-specular'], 'combined-no-specular': ['--quality-camera-track', 'combined', '--quality-light-track', 'oscillate', '--no-specular']}
if selected:
    cases = {name: cases[name] for name in selected}
for name, extra in cases.items():
    dest = OUT / name
    dest.mkdir()
    args = [str(EXE), '--scene-worker', '--pica-pica', '--quality-width', str(options.width), '--quality-height', str(options.height), '--static', '--view', str(options.view), '--camera-dolly', str(options.camera_dolly), '--profile', '--profile-frames', '120', '--specular-samples', str(options.samples), '--output', str(dest), *extra]
    if options.specular_batch is not None:
        args += ['--specular-batch',str(options.specular_batch)]
    args += ['--quality-fps',str(options.simulation_fps)]
    if options.realtime:
        args += ['--profile-realtime']
    if options.specular_registers is not None:
        args += ['--specular-registers',str(options.specular_registers)]
    if options.diffuse_registers is not None:
        args += ['--diffuse-registers',str(options.diffuse_registers)]
    if options.diffuse_samples is not None:
        args += ['--diffuse-samples', str(options.diffuse_samples)]
    if options.reflection_scale is not None:
        args += ['--reflection-scale', str(options.reflection_scale)]
    if options.specular_quality:
        args += ['--specular-quality']
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
(OUT / 'environment-after.json').write_text(json.dumps({'gpu': query(), 'processes': processes(), 'gpu_engine_processes': gpu_consumers()}, indent=2), encoding='utf-8')
print('ALL DONE', query(), flush=True)
