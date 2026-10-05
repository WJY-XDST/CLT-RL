"""Inspect authored joint drives, limits and friction in the closed-chain USD."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--usd',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();app=AppLauncher(args).app
try:
    from pxr import Usd,UsdPhysics
    stage=Usd.Stage.Open(str(args.usd.resolve()))
    data={}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            data[prim.GetName()]={'schemas':prim.GetAppliedSchemas(),
                'attributes':{a.GetName():str(a.Get()) for a in prim.GetAttributes()},
                'relationships':{r.GetName():[str(v) for v in r.GetTargets()] for r in prim.GetRelationships()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2),flush=True)
finally:
    app.close()
