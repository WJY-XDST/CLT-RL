"""Read composed wheel-joint attributes and recorded velocity envelopes.

No simulator, training hook, controller write or physical parameter edit runs.
The asset composition is not a substitute for a native PhysX getter.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

# Reuse the existing offline USD library bootstrap, not an Isaac application.
import audit_joint_target_geometric_exploration as usd_reader
from pxr import Usd, UsdPhysics
import numpy as np

from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()


def audit(replay_root, destination):
    replay_root, destination = Path(replay_root).resolve(), Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / 'mine_model/results'):
        raise ValueError('Require a fresh owned read-only report')
    cfg_path = replay_root / 'high/resolved_env.json'
    cfg = json.loads(cfg_path.read_text())
    asset = Path(cfg['robot']['spawn']['usd_path'])
    stage = Usd.Stage.Open(str(asset))
    if stage is None or cfg['wheel_control_mode'] != 'implicit_velocity':
        raise ValueError('Require the composed current saved implicit wheel drive')
    rows = []
    for name in cfg['wheel_joint_names']:
        matches = [p for p in stage.Traverse() if p.GetName() == name and p.IsA(UsdPhysics.Joint)]
        if len(matches) != 1:
            raise ValueError('Require one composed wheel joint: ' + name)
        prim = matches[0]
        attrs = {}
        for attr in prim.GetAttributes():
            key = attr.GetName()
            if any(token in key.lower() for token in ('velocity', 'friction', 'damping', 'stiffness', 'force')):
                value = attr.Get()
                attrs[key] = dict(value=value if isinstance(value, (str, float, int, bool, type(None))) else str(value),
                                  authored=attr.HasAuthoredValueOpinion())
        rows.append(dict(name=name, prim_path=str(prim.GetPath()), applied_schemas=prim.GetAppliedSchemas(), attributes=attrs))
    envelopes = {}
    files = [SCRIPT, Path(usd_reader.__file__), cfg_path]
    for level in ('low', 'nominal', 'high'):
        replay = replay_root / level
        manifest_path, complete_path, trace_path = [replay / f for f in ('manifest.json', 'completed.json', 'trace.csv')]
        manifest, complete = [json.loads(p.read_text()) for p in (manifest_path, complete_path)]
        if not complete['completed'] or complete['samples'] != 108000 or complete['resets']:
            raise ValueError('Require every completed108-trial record including its transients')
        columns = ('time_s', 'left_wheel_rad_s', 'right_wheel_rad_s', 'left_wheel_ref_rad_s', 'right_wheel_ref_rad_s')
        data = np.genfromtxt(trace_path, delimiter=',', names=True, usecols=columns)
        if len(data) != 108000 or any(not np.isfinite(data[k]).all() for k in columns):
            raise ValueError('Incomplete or nonfinite recorded wheel rates')
        envelopes[level] = dict(samples=len(data), steady_samples=int((data['time_s'] >= 17.).sum()),
            actual_abs_max_rad_s=[float(abs(data[k]).max()) for k in columns[1:3]],
            target_abs_max_rad_s=[float(abs(data[k]).max()) for k in columns[3:]],
            actual_frames_exceeding15rad_s=[int((abs(data[k]) > 15. + 1.e-6).sum()) for k in columns[1:3]])
        files.extend((manifest_path, complete_path, trace_path))
    files.extend(Path(layer.realPath) for layer in stage.GetUsedLayers() if not layer.identifier.startswith('anon:'))
    result = dict(observed_at=datetime.now().astimezone().isoformat(), asset=str(asset),
        source_checkpoint_sha256=manifest['checkpoint_sha256'],
        wheel_joint_attributes=rows, saved_wheel_actuator=cfg['robot']['actuators']['wheels'],
        saved_reference_limit_rad_s=cfg['wheel_action_limit'] * cfg['action_scale_vel'],
        recorded_native_envelopes=envelopes, source_pins={str(p): sha(p) for p in files},
        simulator_started=False, training_modified=False, controller_modified=False,
        physical_parameters_modified=False, policy_accepted=False,
        limitations=['Saved actuator configuration overlays the authored USD drive parameters at initialization.',
            'Composed USD defaults and recorded rates do not directly read the live PhysX maximum joint velocity.',
            'A wheel-following error is not by itself proof of a velocity clamp or a motor gain bug.'])
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--replay-root', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.replay_root, args.output)
    print(json.dumps(dict(output=str(args.output), limit=result['saved_reference_limit_rad_s'],
        recorded_native_envelopes=result['recorded_native_envelopes'], training_modified=False)))
