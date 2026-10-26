"""Compare moving-light/camera diffuse GI against same-frame dense RC captures.

No exit-code shortcut: release requires zero unsupported impulses, not merely
successful capture. References are this hybrid renderer, not path-tracing truth.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
CASES={
    "sun": ["--courtyard","--view","1","--quality-light-track","sun"],
    "drag": ["--courtyard","--view","1","--quality-light-track","drag"],
    "points": ["--gallery","--view","1","--quality-light-track","points"],
    "fast": ["--courtyard","--view","1","--quality-light-track","fast"],
    "step": ["--courtyard","--view","1","--quality-light-track","step"],
    "oscillate": ["--courtyard","--view","1","--quality-light-track","oscillate"],
    **{name:["--courtyard","--view", "0" if name=="orbit" else "1","--quality-camera-track",name]
       for name in ("orbit","translate","rotate","combined","stop","return")},
    "rotate120":["--courtyard","--view","1","--quality-camera-track","rotate","--quality-fps","120"],
    "camera-light":["--courtyard","--view","1","--quality-camera-track","combined","--quality-light-track","drag"],
    "light-stop":["--courtyard","--view","1","--quality-light-track","sun","--quality-stop","208"],
    "scene-motion":["--gallery","--view","1","--quality-light-track","points","--quality-camera-track","combined","--quality-geometry-motion"],
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def display(x):
    x=np.maximum(x,0)
    return (x/(1+x))**(1/2.2)
def lum(x): return x[...,:3] @ np.array([.2126,.7152,.0722],np.float32)
def load(directory,kind,frame,width,height):
    return np.fromfile(directory/f"{kind}-{frame:03}.rgba32f",dtype="<f4").reshape(height,width,4)

def reproject(position,normal,previous_position,previous_normal,previous_view,previous_value):
    matrix=np.array(previous_view["transform_columns"],dtype=np.float32)
    local=(position[...,:3]-np.array(previous_view["position"],np.float32)) @ matrix.T
    height,width=position.shape[:2]
    xy=local[...,:2]/np.maximum(local[...,2,None],1e-6)/previous_view["scaling"]+np.array([width,height])*.5-.5
    xy=np.nan_to_num(xy,nan=-100,posinf=-100,neginf=-100)
    base=np.floor(np.clip(xy,-100,100000)).astype(np.int32)
    valid=(position[...,3]>0)&(local[...,2]>0)&(base[...,0]>=0)&(base[...,1]>=0)&(base[...,0]+1<width)&(base[...,1]+1<height)
    fraction=xy-base
    output=np.zeros((height,width),np.float32)
    weight_sum=np.zeros_like(output)
    footprint=np.maximum(.05,local[...,2]*previous_view["scaling"]*3)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        x=np.clip(base[...,0]+dx,0,width-1); y=np.clip(base[...,1]+dy,0,height-1)
        p=previous_position[y,x]; n=previous_normal[y,x]
        compatible=(p[...,3]==position[...,3])&(np.sum(n[...,:3]*normal[...,:3],axis=-1)>.95)
        compatible &= np.linalg.norm(p[...,:3]-position[...,:3],axis=-1)<footprint
        compatible &= np.abs(np.sum((p[...,:3]-position[...,:3])*normal[...,:3],axis=-1))<np.maximum(.03,footprint*.1)
        weight=(fraction[...,0] if dx else 1-fraction[...,0])*(fraction[...,1] if dy else 1-fraction[...,1])
        weight=weight*valid*compatible
        output+=previous_value[y,x]*weight
        weight_sum+=weight
    return output/np.maximum(weight_sum,1e-6), valid&(weight_sum>.75)

def surface_reference_bounds(value,position,normal):
    """One-pixel reconstruction footprint, strictly on the same receiver plane.

    Raw RC references have sparse dark texels at mesh corners. Keep point-only
    events in the report, but do not mistake removal of those holes for a flash.
    This cannot borrow support across an object edge or a different normal.
    """
    height,width=value.shape
    low=value.copy();high=value.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if not(dx or dy):continue
            q=np.roll(position,(dy,dx),axis=(0,1));n=np.roll(normal,(dy,dx),axis=(0,1));v=np.roll(value,(dy,dx),axis=(0,1))
            valid=(position[...,3]>0)&(q[...,3]==position[...,3])&(np.sum(n[...,:3]*normal[...,:3],axis=-1)>.98)
            valid &= np.abs(np.sum((q[...,:3]-position[...,:3])*normal[...,:3],axis=-1))<.05
            if dy>0:valid[:dy]=False
            if dy<0:valid[dy:]=False
            if dx>0:valid[:,:dx]=False
            if dx<0:valid[:,dx:]=False
            low=np.where(valid,np.minimum(low,v),low);high=np.where(valid,np.maximum(high,v),high)
    return low,high

def analyze(directory,width,height,end,tail):
    first=end-tail+1
    metrics={"display_mae":[],"energy_bias":[],"compensated_rms":[]}
    counts={k:0 for k in ("screen_bright","screen_dark","point_compensated_bright","point_compensated_dark","compensated_bright","compensated_dark","severe_positive","valid_pairs")}
    events=[]
    previous=None
    squared_error_sum=0.
    previews=[]
    for frame in range(first,end+1):
        candidate=load(directory/"candidate","frame",frame,width,height)
        reference=load(directory/"reference","frame",frame,width,height)
        position=load(directory/"candidate","position",frame,width,height)
        normal=load(directory/"candidate","normal",frame,width,height)
        view=json.loads((directory/"candidate"/f"camera-{frame:03}.json").read_text())
        reference_view=json.loads((directory/"reference"/f"camera-{frame:03}.json").read_text())
        assert view==reference_view,"Capture camera or animation metadata differ"
        assert np.isfinite(candidate).all() and np.isfinite(reference).all()
        assert np.array_equal(candidate[...,3],reference[...,3]), "Camera/geometry mismatch"
        c=lum(display(candidate)); r=lum(display(reference)); residual=c-r
        mask=candidate[...,3]>.5
        if mask.any():
            metrics["display_mae"].append(float(np.abs(display(candidate[mask,:3])-display(reference[mask,:3])).mean()))
            metrics["energy_bias"].append(float(lum(candidate)[mask].mean()/max(float(lum(reference)[mask].mean()),1e-12)-1))
        if previous is not None:
            pc,pr,pp,pn,pv,presidual=previous
            next_r=lum(display(load(directory/"reference","frame",min(frame+1,end),width,height)))
            support=np.maximum.reduce([pr,r,next_r]); dark_support=np.minimum.reduce([pr,r,next_r])
            bright=(c-pc>.08)&(c-support>.08)&mask
            dark=(pc-c>.08)&(dark_support-c>.08)&mask
            counts["screen_bright"]+=int(bright.sum()); counts["screen_dark"]+=int(dark.sum())
            warped,valid=reproject(position,normal,pp,pn,pv,presidual)
            jump=residual-warped
            b=(jump>.08)&(c-support>.08)&valid; d=(jump<-.08)&(dark_support-c>.08)&valid
            counts["point_compensated_bright"]+=int(b.sum());counts["point_compensated_dark"]+=int(d.sum())
            low,high=surface_reference_bounds(r,position,normal)
            b &= c-high>.08;d &= low-c>.08
            severe=b&(lum(candidate)>4*lum(reference)+1)
            counts["compensated_bright"]+=int(b.sum()); counts["compensated_dark"]+=int(d.sum())
            counts["severe_positive"]+=int(severe.sum()); counts["valid_pairs"]+=int(valid.sum())
            if valid.any():
                metrics["compensated_rms"].append(float(np.sqrt(np.square(jump[valid]).mean())))
                squared_error_sum+=float(np.square(jump[valid]).sum())
            yy,xx=np.where(b|d)
            if len(yy):
                order=np.argsort(-np.abs(jump[yy,xx]))[:8]
                events.extend(dict(frame=frame,x=int(xx[i]),y=int(yy[i]),jump=float(jump[yy[i],xx[i]]),candidate=float(c[yy[i],xx[i]]),reference=float(r[yy[i],xx[i]])) for i in order)
        previous=(c,r,position,normal,view,residual)
        if frame%2==0:
            side=np.concatenate([display(candidate[...,:3]),display(reference[...,:3])],axis=1)
            previews.append(Image.fromarray((np.clip(side,0,1)*255).astype(np.uint8)))
    if previews:
        palette=previews[0].quantize(colors=128)
        frames=[p.quantize(palette=palette) for p in previews]
        frames[0].save(directory/"comparison.gif",save_all=True,append_images=frames[1:],duration=66,loop=0)
    summary={**counts,**{key:float(np.mean(values)) for key,values in metrics.items()},
             "max_absolute_energy_bias":float(np.max(np.abs(metrics["energy_bias"]))),
             "worst_events":sorted(events,key=lambda e:-abs(e["jump"]))[:24]}
    summary["compensated_rms"]=float(np.sqrt(squared_error_sum/max(counts["valid_pairs"],1)))
    summary["evaluated_surface_frames"]=len(metrics["display_mae"])
    summary["evaluated_pair_frames"]=len(metrics["compensated_rms"])
    summary["passed"]=(summary["compensated_bright"]==0 and summary["compensated_dark"]==0
                       and summary["severe_positive"]==0 and summary["display_mae"]<=.02
                       and summary["max_absolute_energy_bias"]<=.10)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",action="store_true")
    parser.add_argument("--reuse",action="store_true")
    parser.add_argument("--analyze",action="store_true",help="Verify and reanalyze existing manifests without changing capture commands")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    parser.add_argument("--width",type=int,default=480); parser.add_argument("--height",type=int,default=300)
    parser.add_argument("--frames",type=int,default=240); parser.add_argument("--tail",type=int,default=64)
    parser.add_argument("--reference-samples",type=int,default=64)
    parser.add_argument("--extra",nargs="*",default=[])
    parser.add_argument("--exe",type=Path,default=ROOT/"target/debug/lighting_lab.exe")
    args=parser.parse_args()
    exe=args.exe.resolve(); binary=sha(exe); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["PATH"]=str(ROOT/".local/cuda/bin")+os.pathsep+env["PATH"]
    env.pop("CHANDRA_GPU_DEBUG_SYNC",None)
    report={"binary_sha256":binary,"analysis_sha256":sha(Path(__file__)),"width":args.width,"height":args.height,
            "reference":"64 primary samples, same hybrid RC renderer, same simulation frame; not path-tracing ground truth",
            "hardware":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip(),"cases":{}}
    for case in args.cases:
        directory=output/case; directory.mkdir(exist_ok=True)
        for mode in ("candidate","reference"):
            target=directory/mode
            command=[str(exe),"--scene-worker","--static","--indirect-only","--quality-check","--quality-surfaces",
                     "--quality-width",str(args.width),"--quality-height",str(args.height),"--quality-frames",str(args.frames),
                     "--quality-tail",str(args.tail),*CASES[case],*args.extra,"--output",str(target)]
            if mode=="reference": command += ["--diffuse-samples",str(args.reference_samples),"--no-diffuse-filter"]
            key={"binary_sha256":binary,"command":command,"hardware":report["hardware"]}
            manifest=directory/f"{mode}.manifest.json"
            existing=json.loads(manifest.read_text()) if manifest.exists() else {}
            reuse=(args.reuse and all(existing.get(k)==v for k,v in key.items())) or (args.analyze and existing.get("binary_sha256")==binary)
            if not reuse:
                if not args.capture or target.exists(): raise RuntimeError(f"Fresh capture required: {target}")
                print(f"Capturing {case}/{mode}",flush=True)
                with (directory/f"{mode}.log").open("w") as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                assert sha(exe)==binary,"Binary changed during capture"
                files={p.name:sha(p) for p in target.iterdir() if p.is_file()}
                manifest.write_text(json.dumps({**key,"files":files},indent=2),encoding="utf-8")
            else:
                for name,digest in existing["files"].items():
                    if sha(target/name)!=digest: raise RuntimeError(f"Capture changed: {target/name}")
        report["cases"][case]=analyze(directory,args.width,args.height,args.frames,args.tail)
        print(case,json.dumps({k:v for k,v in report["cases"][case].items() if k!="worst_events"}),flush=True)
    report["failures"]=[name for name,value in report["cases"].items() if not value["passed"]]
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return int(bool(report["failures"]))

if __name__=="__main__": raise SystemExit(main())
