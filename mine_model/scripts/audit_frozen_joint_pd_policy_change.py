"""Compare saved policies on identical recorded states without any fitting.

Wheel-reference changes and exact clipping-reduced noise are conditional action
diagnostics. They are not new closed-loop errors or independent acceptance.
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diagnose_height_response import actor
from joint_pd_continuation import ROOT, sha, tensor_finite
from probe_clipped_gaussian_exploration import moments, numerical_check
from qualify_source801_drift50_transfer import differences
from train_joint_pd_standing_seed import saved_updates


def audit(output, baseline, through):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    destination = output / f'frozen_policy_change_prefix_{through:04d}_readonly.json'
    if destination.exists():
        raise ValueError('Preserve the existing immutable comparison')
    plan_path = output / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    if (not plan['ordinary_full_policy_PPO'] or plan['policy_class'] != 'ActorCritic'
            or plan['external_observation_size'] != 27 or plan['auxiliary_losses_enabled']
            or plan['only_command_change'] or plan['action_map_or_controller_changed']):
        raise ValueError('Require the same-input ordinary PPO reward-only trial')
    proofs = output / 'native_learning_signal'
    if not ((proofs / f'audit_update_{through + 1:04d}.json').is_file()
            or through == 200 and (proofs / 'completed_external.json').is_file()):
        raise ValueError('Wait for a later committed update or exact completed200')
    proof_path = proofs / f'audit_update_{through:04d}.json'
    native = json.loads(proof_path.read_text())
    if (native['update'] != through or len(native['all_environment_alignment_max_errors']) != 9
            or any(native['all_environment_alignment_max_errors'].values())
            or not native['probe_RNG_unchanged'] or native['added_loss_or_backward']):
        raise ValueError('Resolve original native alignment first')
    runtime_paths = [Path(path) for path, digest in plan['source_files'].items()
        if digest == plan['profile']['native_runtime_sha256']
        and Path(path).parent == ROOT / 'mine_model/experiments']
    if len(runtime_paths) != 1 or sha(runtime_paths[0]) != plan['profile']['native_runtime_sha256']:
        raise ValueError('Use the single pinned runtime belonging to the actual native trial')
    spec = importlib.util.spec_from_file_location('_frozen_comparison_native_runtime', runtime_paths[0])
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    source = Path(plan['source_checkpoint'])
    runs = list((ROOT / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat').glob('*_' + plan['run_name']))
    if len(runs) != 1:
        raise ValueError('Require the exact owned native run')
    actual = plan['source_actual_ppo_updates'] + through
    candidate = runs[0] / f'model_{actual - 1}.pt'
    rng = torch.random.get_rng_state().clone()
    try:
        before, cfg = runtime.require_source(source)
    except ValueError:
        before, cfg = runtime.require_source(source, transition=True)
    after, after_cfg = runtime.require_source(candidate)
    for path, saved, updates in ((source, before, plan['source_actual_ppo_updates']),
                                 (candidate, after, actual)):
        if (not tensor_finite(saved) or saved_updates(path) != updates
                or len(saved['optimizer_state_dict']['state']) != 17
                or any(float(state['step']) != 20 * updates
                       for state in saved['optimizer_state_dict']['state'].values())):
            raise ValueError('Require finite model/all17 Adam states at the exact actual update clock')
    expected_changes = {'/log_dir', *('/rewards/' + key for key in plan['only_reward_change'])}
    if (set(differences(cfg, after_cfg)) != expected_changes
            or cfg['leg_control_mode'] != 'implicit_joint_position'):
        raise ValueError('Input units, commands, controller or physical configuration changed')
    manifest_path, complete_path = baseline / 'manifest.json', baseline / 'completed.json'
    manifest, complete = [json.loads(path.read_text()) for path in (manifest_path, complete_path)]
    if (manifest['checkpoint_sha256'] != sha(source) or sha(source) != plan['source_sha256']
            or manifest['sampled_policy_actions'] or manifest['weights_updated']
            or manifest['policy_input_remapping'] or manifest['command_scale'] != 1.
            or not complete['completed'] or complete['samples'] != 51000 or complete['resets']
            or len(manifest['cases']) != 17 or manifest['seeds'] != [483, 484, 485]):
        raise ValueError('Require the intact full51 deterministic source-state record')
    agent_path = candidate.parent / 'params/agent.yaml'
    agent = yaml.safe_load(agent_path.read_text())
    policy = agent['policy']
    if (policy['activation'] != 'elu' or policy.get('actor_obs_normalization')
            or policy['actor_hidden_dims'] != [256, 128, 64]):
        raise ValueError('Use the original unnormalized ELU actor')
    with np.load(baseline / 'policy_inputs.npz', allow_pickle=False) as saved:
        inputs, recorded = saved['observations'], saved['means']
        if inputs.shape != (1000, 51, 27) or recorded.shape != (1000, 51, 6):
            raise ValueError('Incomplete fixed source-state record')
        inputs, recorded = inputs[-151:].copy(), recorded[-151:].copy()
    with torch.inference_mode():
        old_mean = actor(torch.from_numpy(inputs), before['model_state_dict']).numpy()
        new_mean = actor(torch.from_numpy(inputs), after['model_state_dict']).numpy()
    parity = float(abs(old_mean - recorded).max())
    if parity > 1.e-4 or not np.isfinite(new_mean).all():
        raise ValueError('The original saved actor does not reproduce the recorded source means')
    old_std, new_std = [state['model_state_dict']['std'].numpy() for state in (before, after)]
    quadrature_error = numerical_check()
    noise = [moments(mean, std) for mean, std in ((old_mean, old_std), (new_mean, new_std))]
    trace = np.genfromtxt(baseline / 'trace.csv', delimiter=',', names=True)
    if len(trace) != 51000:
        raise ValueError('Incomplete original source physical trace')
    radius, track, scale = [cfg[key] for key in ('wheel_radius', 'wheel_track_width', 'action_scale_vel')]
    leg_scale = np.asarray(cfg['leg_joint_position_scale'])
    rows = []
    for cid, case in enumerate(manifest['cases']):
        for sid, seed in enumerate(manifest['seeds']):
            index = 3 * cid + sid
            part = trace[(trace['case_id'] == cid) & (trace['seed'] == seed) & (trace['time_s'] >= 17.)]
            if len(part) != 151 or np.any(part['episode']) or np.any(part['terminated']):
                raise ValueError('Keep the original first-episode steady source window')
            old, new = [np.clip(mean[:, index], -1., 1.) for mean in (old_mean, new_mean)]
            wheel_delta = (new[:, (2, 5)] - old[:, (2, 5)]) * scale
            leg_delta = (new[:, (0, 1, 3, 4)] - old[:, (0, 1, 3, 4)]) * leg_scale
            rows.append(dict(case=case, seed=seed, frozen_source_samples=151,
                source_closed_loop_yaw_signed_error=float((part['yaw_rate_rad_s'] - part['yaw_cmd_rad_s']).mean()),
                source_closed_loop_speed_signed_error=float((part['speed_m_s'] - part['speed_cmd_m_s']).mean()),
                conditional_yaw_reference_proxy_change=float((radius * (wheel_delta[:, 1] - wheel_delta[:, 0]) / track).mean()),
                conditional_speed_reference_proxy_change=float((radius * wheel_delta.mean(-1)).mean()),
                conditional_motor_angle_target_change_RMS_rad_by_head=np.sqrt((leg_delta**2).mean(0)).tolist(),
                raw_mean_outside_action_box_fraction_before=(abs(old_mean[:, index]) > 1.).mean(0).tolist(),
                raw_mean_outside_action_box_fraction_after=(abs(new_mean[:, index]) > 1.).mean(0).tolist(),
                exact_clipped_action_noise_RMS_before=np.sqrt(noise[0]['noise_mse'][:, index].mean(0)).tolist(),
                exact_clipped_action_noise_RMS_after=np.sqrt(noise[1]['noise_mse'][:, index].mean(0)).tolist()))
    files = [Path(__file__).resolve(), plan_path, proof_path, runtime_paths[0], source, candidate,
        agent_path, *(baseline / name for name in ('manifest.json', 'completed.json', 'resolved_env.json',
            'trace.csv', 'policy_inputs.npz', 'motion025_full_height_diagnosis.json'))]
    pins = {str(path): sha(path) for path in files}
    if (not torch.equal(rng, torch.random.get_rng_state())
            or any(sha(path) != digest for path, digest in pins.items())):
        raise ValueError('Immutable inputs or the diagnostic RNG state changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_prefix=through,
        source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
        source_actual_updates=plan['source_actual_ppo_updates'], checkpoint=str(candidate),
        checkpoint_sha256=sha(candidate), actual_updates=actual, all17_Adam_step=20 * actual,
        source_actor_CPU_parity_max_error=parity, independent_clipped_moment_quadrature_max_error=quadrature_error,
        source_std=old_std.tolist(), current_std=new_std.tolist(), trials=rows, source_pins=pins,
        source_states_observations_and_commands_unmodified=True, simulator_started=False,
        optimizer_executed=False, added_training_loss=False, training_modified=False, RNG_draws_added=0,
        policy_accepted=False, local_optimum_proven=False,
        limitations=[
            'Both policies see identical old source observations, including its previous actions. These are conditional targets, not a new rollout.',
            'Wheel rolling proxies are not body-speed or yaw measurements. Target shifts do not predict how wheel loads, contact or body motion will respond.',
            'Exact clipped-Gaussian noise measures action-envelope exploration; it does not establish useful geometric exploration or stability.',
            'All17 low-speed cases are retained here. Original high-speed canonical failures remain part of full acceptance and are not tested by this diagnostic.',
            'No checkpoint is fitted or selected, and no acceptance threshold is changed. Complete200 and independent full27/full51/static review are still required.'])
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline-replay', type=Path, required=True)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path, result = audit(args.output, args.baseline_replay, args.through)
    print(json.dumps(dict(report=str(path), actual_updates=result['actual_updates'],
        source_parity=result['source_actor_CPU_parity_max_error'], simulation_started=False,
        optimizer_executed=False, training_modified=False), ensure_ascii=False))
