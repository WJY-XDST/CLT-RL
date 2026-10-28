"""Compare direct-joint actors on identical recorded states without running dynamics."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_height_response import actor
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from qualify_source801_drift50_transfer import differences
from wheel_legged_gym_isaaclab import leg_control as mapping
from wheel_legged_gym_isaaclab import five_bar_vmc as geometry
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def compare(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'recorded_source_joint_targets_prefix_{through:04d}_readonly.json'
    if directory.parent != RESULTS or destination.exists() or not 1 <= through <= 200:
        raise ValueError('Preserve an owned committed plain27 diagnostic')
    plan_path = directory / 'experiment_plan.json'
    health_path = directory / f'policy_update_health_prefix_{through:04d}_readonly.json'
    geo_path = directory / f'joint_target_geometric_exploration_prefix_{through:04d}_readonly.json'
    plan, health, native_geo = [json.loads(p.read_text()) for p in (plan_path, health_path, geo_path)]
    if (not plan['ordinary_full_policy_PPO'] or plan['auxiliary_losses_enabled']
            or plan['training_config']['leg_control_mode'] != mapping.DIRECT_JOINT_POSITION_MODE
            or plan['training_config']['observation_space'] != 27
            or len(health['snapshots']) != through or health['committed_updates'] != through
            or native_geo['committed_updates'] != through
            or native_geo['original_native_motor_target_parity_max_error_rad']
            or native_geo['original_native_FK_reference_parity_max_error_m'] > 2.e-7):
        raise ValueError('Require qualified original plain27 motor mapping and committed policies')
    replay = Path(plan['source_replay'])
    manifest_path, complete_path = replay / 'manifest.json', replay / 'completed.json'
    manifest, complete = [json.loads(p.read_text()) for p in (manifest_path, complete_path)]
    baseline = Path(manifest['checkpoint'])
    candidate = Path(health['snapshots'][-1]['checkpoint_after'])
    if (not complete['completed'] or complete['samples'] != 27000 or complete['resets']
            or len(manifest['cases']) != 9 or manifest['seeds'] != [83, 84, 85]
            or manifest['seconds_per_case'] != 20. or manifest['policy_dt'] != .02
            or manifest['weights_updated'] or manifest['sampled_policy_actions']
            or manifest['policy_input_remapping'] or manifest['command_scale'] != 1.
            or baseline != Path(plan['source_checkpoint'])
            or sha(baseline) != manifest['checkpoint_sha256']
            or sha(baseline) != plan['source_sha256']):
        raise ValueError('Require this intact zero-reset canonical source record')
    cfgs = [validate_checkpoint_robot(p) for p in (baseline, candidate)]
    expected = {'/log_dir'}
    expected.update('/commands/' + key for key in plan['only_command_change'])
    expected.update('/rewards/' + key for key in plan['only_reward_change'])
    if set(differences(*cfgs)) != expected:
        raise ValueError('Physical/observation/motor contract changed beyond the declared trial')
    files = (SCRIPT, plan_path, health_path, geo_path, manifest_path, complete_path,
        replay / 'policy_inputs.npz', replay / 'trace.csv', baseline, candidate,
        Path(mapping.__file__), Path(geometry.__file__), Path(actor.__code__.co_filename))
    pins = {**health['source_pins'], **native_geo['source_pins'],
            **{str(p): sha(p) for p in files}}
    for checkpoint in (baseline, candidate):
        for path in (checkpoint.parent / 'params/env.yaml', checkpoint.parent / 'params/agent.yaml'):
            pins[str(path)] = sha(path)
    models = []
    for path, actual in ((baseline, plan['source_actual_ppo_updates']),
                         (candidate, plan['source_actual_ppo_updates'] + through)):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        states = saved['optimizer_state_dict']['state']
        if (not tensor_finite(saved) or saved_updates(path) != actual or len(states) != 17
                or any(float(s['step']) != 20 * actual for s in states.values())):
            raise ValueError('Require exact finite original weights and all17 continuous Adam clocks')
        models.append(saved['model_state_dict'])
    with np.load(replay / 'policy_inputs.npz', allow_pickle=False) as data:
        observations, recorded = [data[k][-150:].copy() for k in ('observations', 'means')]
    if observations.shape != (150, 27, 27) or recorded.shape != (150, 27, 6):
        raise ValueError('Require the full original last3s input record')
    trace = np.genfromtxt(replay / 'trace.csv', delimiter=',', names=True)
    frame = np.rint(trace['time_s'] / .02).astype(int) - 1
    rows = trace[frame >= 850]
    frame = frame[frame >= 850] - 850
    env_id = rows['case_id'].astype(int) * 3 + np.searchsorted(manifest['seeds'], rows['seed'].astype(int))
    seen = np.zeros((150, 27), dtype=int)
    np.add.at(seen, (frame, env_id), 1)
    if not np.all(seen == 1):
        raise ValueError('Recorded input and trace source identities do not align one to one')
    refs = np.zeros((150, 27, 2), dtype=np.float32)
    refs[frame, env_id, 0] = rows['left_length_ref_m']
    refs[frame, env_id, 1] = rows['right_length_ref_m']
    commands = observations[:, :, 8] / cfgs[0]['obs_scales']['height_measurements']
    trace_commands = np.zeros((150, 27), dtype=np.float32)
    trace_commands[frame, env_id] = rows['height_cmd_m']
    if np.max(abs(commands - trace_commands)) > 1.e-7:
        raise ValueError('Source height-command/input clocks differ')
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    inputs = torch.from_numpy(observations)
    limits = inputs.new_tensor(native_geo['USD_composed_joint_limits_rad']).expand(150 * 27, -1, -1)
    means, targets, lengths, angles = [], [], [], []
    with torch.inference_mode():
        for state, cfg in zip(models, cfgs):
            raw = actor(inputs, state)
            q, _ = mapping.joint_position_targets(raw[..., mapping.LEG_ACTION_COLUMNS].reshape(-1, 4),
                cfg['leg_joint_position_offset'], cfg['leg_joint_position_scale'], limits,
                cfg['leg_joint_target_limit_margin'])
            length, angle, _, valid = geometry.five_bar_state(q[:, ::2], q[:, 1::2], **cfg['five_bar_geometry'])
            if not torch.isfinite(raw).all() or not valid.all() or not torch.isfinite(length).all():
                raise ValueError('Invalid actual-source or candidate target geometry')
            means.append(raw.numpy())
            targets.append(q.reshape(150, 27, 4).numpy())
            lengths.append(length.reshape(150, 27, 2).numpy())
            angles.append(angle.reshape(150, 27, 2).numpy())
    mean_parity = float(np.max(abs(means[0] - recorded)))
    reference_parity = float(np.max(abs(lengths[0] - refs)))
    if mean_parity > 1.e-4 or reference_parity > 2.e-7:
        raise ValueError('Original CPU actor/motor/FK does not reproduce the independent source record')
    cases = []
    for index, case in enumerate(manifest['cases']):
        selection = slice(index * 3, (index + 1) * 3)
        summaries = []
        for label, raw, q, length, angle, cfg in zip(
                ('source', 'candidate'), means, targets, lengths, angles, cfgs):
            r, q, length, angle = [v[:, selection].astype(np.float64) for v in (raw, q, length, angle)]
            wheel = np.clip(r[..., (2, 5)], -1., 1.) * cfg['action_scale_vel']
            shortfall = commands[:, selection, None] + cfg['rewards']['leg_length_target_height_offset'] - length
            summaries.append(dict(label=label, raw_action_mean=r.mean((0, 1)).tolist(),
                raw_leg_mean_outside_action_bounds_fraction=(abs(r[..., mapping.LEG_ACTION_COLUMNS]) > 1.).mean((0, 1)).tolist(),
                motor_target_mean_rad=q.mean((0, 1)).tolist(),
                FK_leg_target_mean_m=length.mean((0, 1)).tolist(),
                FK_leg_angle_target_mean_rad=angle.mean((0, 1)).tolist(),
                commanded_upright_length_proxy_minus_target_mean_m=shortfall.mean((0, 1)).tolist(),
                wheel_forward_reference_mean_rad_s=wheel.mean((0, 1)).tolist(),
                wheel_rolling_reference_proxy_mean_m_s=float(wheel.mean(-1).mean() * cfg['wheel_radius'])))
        cases.append(dict(original_case=case, retained_source_states=450, policies=summaries,
            normalized_raw_mean_shift_RMS_by_head=np.sqrt(
                ((means[1][:, selection] - means[0][:, selection]).astype(np.float64)**2).mean((0, 1))).tolist()))
    if not torch.equal(rng, torch.random.get_rng_state()) or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Read-only comparison changed RNG or retained source evidence')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_prefix=through,
        fixed_input_source=str(replay), source_checkpoint=str(baseline), candidate_checkpoint=str(candidate),
        source_actual_updates=plan['source_actual_ppo_updates'],
        candidate_actual_updates=plan['source_actual_ppo_updates'] + through,
        source_actor_recorded_mean_parity_max_error=mean_parity,
        source_recorded_FK_length_parity_max_error_m=reference_parity,
        source_observation_input_columns_changed=[], retained_final_window_seconds=3.,
        all_original9_cases_and27_trials_retained=True, cases=cases, source_pins=pins,
        added_IK_or_VMC_feedback=False, RNG_draws_added=0, optimizer_executed=False,
        simulator_started=False, training_modified=False, acceptance_changed=False, policy_accepted=False,
        limitations=['Both actors receive identical last3s states and action histories from the older source policy; no new candidate trajectory is produced.',
            'FK length/angle and wheel rolling targets are control-output diagnostics, not predictions of actual candidate height, velocity or support contact.',
            'A changed target or smaller target shortfall does not prove stability, convergence or improved closed-loop acceptance. The original native200/replay and sustained gates remain required.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, result = compare(args.directory, args.through)
    print(json.dumps(dict(report=str(path), source_actor_parity=result['source_actor_recorded_mean_parity_max_error'],
        source_FK_parity=result['source_recorded_FK_length_parity_max_error_m'],
        original_reverse060=[case for case in result['cases'] if case['original_case']['name'] == 'reverse_0p6'],
        closed_loop_acceptance=False)))
