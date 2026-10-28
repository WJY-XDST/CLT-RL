"""Offline collider-bound reconstruction for native pre-reset events.

Cross-check the exact evaluator's base-collider bound against its recorded
clearance before interpreting terminal snapshots. A final-snapshot bound
does not identify a contact partner or the earlier continuous-contact history.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
USD_LIB = next((ROOT/'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent
if str(USD_LIB/'bin') not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
    environment = dict(os.environ)
    environment['LD_LIBRARY_PATH'] = str(USD_LIB/'bin')+':'+environment.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,__file__,*sys.argv[1:]],environment)
sys.path.insert(0,str(USD_LIB))

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(training_output,output):
    training_output,output = Path(training_output).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve earlier terminal-contact evidence')
    cfg = json.loads((training_output/'experiment_plan.json').read_text())['training_config']
    asset = Path(cfg['robot']['spawn']['usd_path'])
    stage = Usd.Stage.Open(str(asset))
    candidates = [p for p in stage.Traverse() if p.GetName()=='base_link' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(candidates) != 1:
        raise ValueError('Require the one imported chassis rigid body')
    base = candidates[0]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
    bounds,shapes = Gf.Range3d(),[]
    for prim in Usd.PrimRange(base,Usd.TraverseInstanceProxies()):
        if (prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False):
            current = cache.ComputeRelativeBound(prim,base).ComputeAlignedRange()
            if not current.IsEmpty():
                bounds.UnionWith(current)
                shapes.append(dict(path=str(prim.GetPath()),
                    authored_contact_offset_m=prim.GetAttribute('physxCollision:contactOffset').Get(),
                    authored_rest_offset_m=prim.GetAttribute('physxCollision:restOffset').Get()))
    if bounds.IsEmpty():
        raise ValueError('No enabled chassis collider bound')
    low,high = np.asarray(bounds.GetMin()),np.asarray(bounds.GetMax())
    corners = np.array([[x,y,z] for x in (low[0],high[0])
                        for y in (low[1],high[1]) for z in (low[2],high[2])])
    replay = training_output/'replay'
    manifest = json.loads((replay/'manifest.json').read_text())
    complete = json.loads((replay/'completed.json').read_text())
    rows = np.genfromtxt(replay/'trace.csv',delimiter=',',names=True)
    if not complete['completed'] or len(rows) != 27000 or complete['samples'] != len(rows):
        raise ValueError('Cross-check requires the completed original full27 replay')
    euler = np.column_stack([rows[n] for n in ('roll_rad','pitch_rad','yaw_rad')])
    matrices = Rotation.from_euler('xyz',euler).as_matrix()
    calculated = (matrices[:,None,2,:]*corners[None]).sum(-1).min(-1)+rows['height_m']
    error = float(abs(calculated-rows['base_collision_min_z_m']).max())
    if error > 2.e-6:
        raise ValueError('Offline bound does not reproduce native clearance: '+str(error))
    source = training_output/'native_learning_signal/terminal_events_before_reset.jsonl'
    events = [json.loads(line) for line in source.read_text().splitlines()]
    report = []
    for event in events:
        state = event['state']
        q = state['root_quaternion']
        rotation = Rotation.from_quat([*q[1:],q[0]])
        minimum = float(rotation.apply(corners)[:,2].min()+state['root_position'][2])
        report.append(dict(control_step=event['control_step'],environment_id=event['environment_id'],
            reasons=event['reasons'],episode_time_s=state['episode_time_s'],
            command_age_s=state['command_age_s'],height_command_m=state['command'][2],
            base_net_force_n=state['base_net_force_n'],
            conservative_base_collider_bound_min_z_m=minimum))
    contact = [row for row in report if 'base_contact' in row['reasons']]
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=manifest['checkpoint_sha256'],
        chassis_collider_bounds_in_base_m=[low.tolist(),high.tolist()],colliders=shapes,
        native_full27_clearance_reconstruction_max_error_m=error,
        native_reference_samples=len(rows),terminal_events=len(events),
        base_contact_events=len(contact),
        base_contact_event_snapshot_bound_min_z_range_m=[min(row['conservative_base_collider_bound_min_z_m'] for row in contact),
            max(row['conservative_base_collider_bound_min_z_m'] for row in contact)] if contact else None,
        events=report,training_modified=False,physics_modified=False,termination_changed=False,
        simulator_started=False,PPO_updates_performed=0,policy_accepted=False,
        source_files={str(path):sha(path) for path in (Path(__file__),asset,source,
            replay/'manifest.json',replay/'completed.json',replay/'trace.csv')},
        limitation='Conservative bounding box plus one final pre-reset snapshot per event. '
            'Negative bound does not prove collider-ground penetration; positive bound alone '
            'does not rule out earlier sustained contact, speculative contacts or identify a '
            'self-contact partner. Do not waive the existing termination or acceptance rules.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.training_output,args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('events','source_files')},indent=2))
