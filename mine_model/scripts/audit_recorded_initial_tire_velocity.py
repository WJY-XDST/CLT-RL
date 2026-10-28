"""Read initial tire contact velocities from recorded nominal reset observations.

This is a kinematic observation, not a change to reset sampling or a causal
failure test. The root velocity is the root link COM velocity in this SDK.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
USD_LIB = next((ROOT / 'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent
if str(USD_LIB / 'bin') not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(USD_LIB / 'bin') + ':' + env.get('LD_LIBRARY_PATH', '')
    os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)
sys.path.insert(0, str(USD_LIB))

import numpy as np
from pxr import Usd, UsdPhysics

SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(training, pose_path, destination):
    training, pose_path, destination = map(lambda p: Path(p).resolve(), (training, pose_path, destination))
    if destination.exists() or destination.parent != training:
        raise ValueError('Preserve a fresh owned initial-state diagnostic')
    plan_path = training / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = plan['training_config']
    pose = json.loads(pose_path.read_text())
    if (cfg['leg_control_mode'] != 'implicit_joint_position'
            or cfg['robot']['init_state']['rot'] != [1., 0., 0., 0.]
            or not pose['all_closed_loops_consistent'] or not pose['both_analytic_tires_on_ground']
            or any(sha(p) != h for p, h in pose['source_files'].items())):
        raise ValueError('Require the unchanged level nominal geometry and its actual native initial observations')
    sdk_data = ROOT / 'IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py'
    sdk_write = sdk_data.with_name('articulation.py')
    tree = ast.parse(sdk_data.read_text())
    aliases = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'root_lin_vel_w']
    if (len(aliases) != 1 or not any(isinstance(n, ast.Return) and isinstance(n.value, ast.Attribute)
            and n.value.attr == 'root_com_lin_vel_w' for n in ast.walk(aliases[0]))):
        raise ValueError('Recheck the installed SDK root linear-velocity reference point')
    writes = [n for n in ast.walk(ast.parse(sdk_write.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == 'write_root_velocity_to_sim']
    if (len(writes) != 1 or not any(isinstance(n, ast.Attribute) and n.attr == 'write_root_com_velocity_to_sim'
            for n in ast.walk(writes[0]))):
        raise ValueError('Recheck the installed SDK reset velocity write reference point')
    asset = Path(cfg['robot']['spawn']['usd_path'])
    stage = Usd.Stage.Open(str(asset))
    bodies = [p for p in stage.Traverse() if p.GetName() == 'base_link' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError('Require the exact root link')
    root_com = np.asarray(UsdPhysics.MassAPI(bodies[0]).GetCenterOfMassAttr().Get(), dtype=float)
    sample = training / 'native_learning_signal/rollout_0001.npz'
    proof_path = training / 'native_learning_signal/audit_update_0001.json'
    proof = json.loads(proof_path.read_text())
    if any(proof['all_environment_alignment_max_errors'].values()) or not proof['probe_RNG_unchanged']:
        raise ValueError('Resolve original native data alignment first')
    ids = np.array(proof['sampled_environment_ids'])
    with np.load(sample, allow_pickle=False) as data:
        obs = data['observations'][0].astype(np.float64)
    if (obs.shape != (32, 27) or not np.isfinite(obs).all()
            or abs(obs[:, 3:6] - [0., 0., -1.]).max() > 1.e-7
            or abs(obs[:, 13:17]).max() or abs(obs[:, 19:21]).max()):
        raise ValueError('This calculation requires level reset and zero active/wheel relative velocities')
    omega = obs[:, :3] / cfg['obs_scales']['ang_vel']
    base_velocity_xy = obs[:, 17:19] / cfg['obs_scales']['lin_vel']
    radius = cfg['wheel_radius']
    contacts = np.stack([np.array(pose['wheels'][name]['center_in_base_m']) - [0., 0., radius]
        for name in ('LW_link', 'RW_link')])
    arm = contacts - root_com
    # All relative joint velocities are reset to zero. Root orientation is
    # identity, so heading/world/root-link vector components agree initially.
    velocity_xy = base_velocity_xy[:, None, :] + np.cross(omega[:, None, :], arm[None, :, :])[..., :2]
    groups = {}
    for name, mask in dict(all32=np.ones(32, dtype=bool),
            dedicated_static=ids < plan['fixed_standing_environments']).items():
        v = velocity_xy[mask]
        groups[name] = dict(samples=int(mask.sum()),
            root_COM_velocity_xy_min_m_s=base_velocity_xy[mask].min(0).tolist(),
            root_COM_velocity_xy_max_m_s=base_velocity_xy[mask].max(0).tolist(),
            tire_contact_forward_velocity_min_m_s=v[..., 0].min(0).tolist(),
            tire_contact_forward_velocity_max_m_s=v[..., 0].max(0).tolist(),
            tire_contact_forward_velocity_RMS_m_s=np.sqrt(np.mean(v[..., 0]**2, axis=0)).tolist(),
            tire_contact_horizontal_speed_RMS_m_s=np.sqrt(np.mean(np.sum(v**2, axis=-1), axis=0)).tolist(),
            forward_contact_speed_above_005_fraction=np.mean(abs(v[..., 0]) > .05, axis=0).tolist(),
            wheel_relative_speed_initial_rad_s=[0., 0.])
    files = [SCRIPT, sdk_data, sdk_write, asset, plan_path, pose_path, sample, proof_path]
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        training_output=str(training), native_initial_pose_verified=True,
        root_velocity_reference_point='Root link center of mass, verified in the installed SDK',
        root_link_COM_in_link_m=root_com.tolist(), tire_contact_in_link_m=contacts.tolist(),
        formula='v_contact_xy = v_root_COM_xy + (omega_world cross (r_contact - r_root_COM))_xy; all relative joint velocities are initially zero',
        groups=groups, reset_velocity_initial=cfg['reset_velocity_initial'],
        reset_velocity_final=cfg['reset_velocity_final'], initial_commands=obs[:, 6:9].tolist(),
        initial_command_values_are_observation_scaled=True,
        current_angle_loop_and_contact_height_errors_not_detected=True,
        nonzero_contact_velocity_is_a_reset_transient_not_proof_of_a_code_bug=True,
        failed_environment_exact_initial_states_not_recorded=True,
        no_causal_reset_distribution_comparison_performed=True,
        source_pins={str(p): sha(p) for p in files}, simulator_started=False,
        policy_loaded=False, optimizer_executed=False, RNG_draws_added=0,
        training_modified=False, acceptance_changed=False, policy_accepted=False,
        limitations=['32 first native states, not exact initial states of every later failed environment.',
            'Vertical initial contact velocity is not observable from the saved27-vector and is not reconstructed.',
            'This audit does not select a new reset distribution, wheel controller or reference limit.'])
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-output', type=Path, required=True)
    cli.add_argument('--pose-audit', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.training_output, args.pose_audit, args.output)
    print(json.dumps(dict(groups=result['groups'], training_modified=False, policy_accepted=False)))
