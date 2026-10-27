"""Capture the hybrid's actual GPU output. Requires a built lighting_lab.exe.

The comparison is a higher-sample traced first segment with the same finite
secondary transport, not an unbiased full path-traced ground truth.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='.local/hybrid-acceptance')
parser.add_argument('--report',default='.local/hybrid-static-report')
parser.add_argument('--capture',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
os.chdir(root)
output=Path(args.output).resolve();report=Path(args.report).resolve()
output.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
exe=root/'target/debug/lighting_lab.exe'
env=os.environ.copy();env['PATH']=str(root/'.local/cuda/bin')+os.pathsep+env['PATH']
cases=[
    ('near',960,600,1,160,16,[]),
    ('overview',960,600,0,160,16,[]),
    ('static',480,300,1,176,16,[]),
    ('reference',480,300,1,176,128,['--specular-reference','--no-specular-filter']),
    ('rotate',480,300,1,176,16,['--quality-camera-track','rotate']),
    ('relight',480,300,1,176,16,['--quality-light-track','step']),
]
records={}
def load(directory,name,w,h):
    a=np.fromfile(directory/name,dtype='<f4').reshape(h,w,4)
    if not np.isfinite(a).all():raise AssertionError(f'Nonfinite output: {directory/name}')
    return a
def tone(rgb):
    rgb=np.maximum(rgb,0)
    return np.uint8(np.clip((rgb/(1+rgb))**(1/2.2)*255,0,255))
for name,w,h,view,frames,samples,extra in cases:
    directory=output/name
    command=[str(exe),'--scene-worker','--pica-pica','--static','--quality-check',
        '--quality-width',str(w),'--quality-height',str(h),'--view',str(view),
        '--quality-frames',str(frames),'--quality-tail','8','--specular-samples',str(samples),
        '--output',str(directory),*extra]
    if args.capture:
        print('Capturing',name,flush=True)
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    full=load(directory,f'frame-{frames:03}.rgba32f',w,h)
    spec=load(directory,f'specular-filtered-{frames:03}.rgba32f',w,h)
    raw=[load(directory,f'specular-raw-{i:03}.rgba32f',w,h)[...,:3] for i in range(frames-7,frames+1)]
    clean=[load(directory,f'specular-filtered-{i:03}.rgba32f',w,h)[...,:3] for i in range(frames-7,frames+1)]
    records[name]=dict(resolution=[w,h],frames=frames,samples=samples,
        finite=True,negative_specular_values=int((spec[...,:3]<-1e-6).sum()),
        mean_specular=float(spec[...,:3].mean()),max_specular=float(spec[...,:3].max()),
        raw_temporal_variance=float(np.var(raw,axis=0).mean()),
        filtered_temporal_variance=float(np.var(clean,axis=0).mean()),
        capacity=(directory/'capacity.txt').read_text().strip(),arguments=command[1:])
    Image.fromarray(tone(full[...,:3])).save(report/(name+'.png'))
    if name in ('near','overview'):
        # These are the exact direct+diffuse and full outputs from one frame.
        left=tone(full[...,:3]-spec[...,:3]);right=tone(full[...,:3])
        canvas=Image.new('RGB',(w*2,h+32),(20,23,28))
        canvas.paste(Image.fromarray(left),(0,32));canvas.paste(Image.fromarray(right),(w,32))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,10),'Indirect specular OFF',fill='white');draw.text((w+12,10),'RC + traced indirect specular ON',fill='white')
        canvas.save(report/(name+'-comparison.png'))
    if name=='reference':
        reference=np.mean(raw,axis=0)
    if name=='static':
        hybrid=np.mean(clean,axis=0)
    if name=='rotate':
        frames_rgb=[Image.fromarray(tone(load(directory,f'frame-{i:03}.rgba32f',w,h)[...,:3])) for i in range(frames-7,frames+1)]
        frames_rgb[0].save(report/'motion.gif',save_all=True,append_images=frames_rgb[1:],duration=100,loop=0)
records['comparison']=dict(label='128 spp x 8 raw frames, traced first segment; same finite secondary transport',
    scene_linear_specular_mae=float(np.abs(hybrid-reference).mean()),
    relative_mean_energy=float(hybrid.mean()/max(reference.mean(),1e-9)-1),
    tone_mapped_specular_mae=float(np.abs(tone(hybrid).astype(float)-tone(reference)).mean()/255))
summary=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
    scene_sha256=hashlib.sha256((root/'external/PicaPica/scene.gltf').read_bytes()).hexdigest(),
    source='SEED / Electronic Arts, CC BY-NC 4.0',tone_mapping='Reinhard + gamma 2.2, fixed exposure 0 EV',
    cases=records,limitations=['Finite secondary transport; not full path tracing.','Glass is opaque.','RC interpolation and directional discretization remain biased.','Motion and relighting captures are diagnostic evidence, not a zero-flicker certification.'])
(report/'validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(records['comparison'],indent=2))
