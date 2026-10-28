"""Account for saved training yaw residuals without changing training.

Clipped actor means give the deterministic reference; sampled actions give
the actual wheel targets. No post-step wheel velocities were recorded here,
so the body-versus-target residual must not be called a pure servo/slip error.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, through):
    directory = Path(directory).resolve()
    assert directory.parent == ROOT / 'mine_model/results' and through in (50, 100, 200)
    destination = directory / f'sampled_yaw_reference_components_prefix_{through:04d}_readonly.json'
    assert not destination.exists()
    plan_path = directory / 'experiment_plan.json'
    physical_path = directory / f'native_motion_training_saved_limits_prefix_{through:04d}_readonly.json'
    plan, physical = [json.loads(p.read_text()) for p in (plan_path, physical_path)]
    cfg = plan['training_config']
    assert cfg['leg_control_mode'] == 'implicit_joint_position' and cfg['wheel_control_mode'] == 'implicit_velocity'
    assert cfg['observation_space'] == 27 and physical['committed_updates'] == through
    assert physical['native_environment_count'] == plan['native_num_envs']
    assert len(physical['all_environment_alignment_max_errors']) == 9
    assert not any(physical['all_environment_alignment_max_errors'].values())
    assert physical['observer_RNG_unchanged'] and not physical['added_loss_or_backward']
    pins = dict(physical['source_sha256'])
    pins.update({str(p): sha(p) for p in (SCRIPT, plan_path, physical_path)})
    ids = physical['sampled_environment_ids']
    fields = ('native_commands', 'native_yaw', 'mu', 'actions', 'native_raw_actions', 'dones')
    parts = {key: [] for key in fields}
    for update in range(1, through + 1):
        path = directory / f'native_learning_signal/rollout_{update:04d}.npz'
        assert sha(path) == pins[str(path)]
        with np.load(path, allow_pickle=False) as archive:
            for key in fields:
                value = archive[key]
                assert value.shape[:2] == (48, len(ids)) and np.isfinite(value).all()
                parts[key].append(value.copy())
    data = {key: np.concatenate(value) for key, value in parts.items()}
    command = data['native_commands']
    done = data['dones'][..., 0].astype(bool)
    assert np.array_equal(data['actions'], data['native_raw_actions'])
    dt = cfg['sim']['dt'] * cfg['decimation']
    age = np.empty(done.shape, dtype=np.float64)
    stable_age = np.empty_like(age)
    age[0], stable_age[0] = dt, 0.
    for t in range(1, len(age)):
        age[t] = np.where(done[t - 1], dt, age[t - 1] + dt)
        changed = np.any(abs(command[t] - command[t - 1]) > 1.e-7, axis=-1)
        stable_age[t] = np.where(changed | done[t - 1], 0., stable_age[t - 1] + dt)
    settled = (age > 2.) & (stable_age >= 1.) & ~done
    static = np.broadcast_to(np.array(ids)[None, :] < plan['fixed_standing_environments'], done.shape)
    scale, limit = cfg['action_scale_vel'], cfg['wheel_action_limit']
    ratio = cfg['wheel_radius'] / cfg['wheel_track_width']
    assert scale == 15. and limit == 1.25 and ratio > 0.

    def proxy(key):
        wheels = np.clip(data[key][..., [2, 5]].astype(np.float64), -limit, limit) * scale
        return ratio * (wheels[..., 1] - wheels[..., 0])

    deterministic, sampled = proxy('mu'), proxy('actions')
    body, yaw_command = data['native_yaw'].astype(np.float64), command[..., 1].astype(np.float64)
    error = body - yaw_command
    components = dict(deterministic_reference_minus_command=deterministic - yaw_command,
        sampled_minus_deterministic_reference=sampled - deterministic,
        body_minus_sampled_reference=body - sampled)
    closure = float(abs(sum(components.values()) - error).max())
    assert closure < 1.e-12
    small = (abs(yaw_command) >= .05) & (abs(yaw_command) < .10)
    populations = dict(dedicated_static=static, motion_population=~static,
        small_yaw_positive=small & (yaw_command > 0.), small_yaw_negative=small & (yaw_command < 0.))
    heights = {'low': plan['full_height_range'][0], 'nominal': cfg['rewards']['base_height_target'],
               'high': plan['full_height_range'][1]}
    for hn, height in heights.items():
        for vn, velocity in [('reverse_fast', -.8), ('spin', 0.), ('forward_fast', .8)]:
            for yn, sign in [('positive', 1), ('negative', -1)]:
                populations[f'{hn}_{vn}_small_yaw_{yn}'] = (small & (yaw_command * sign > 0.)
                    & np.isclose(command[..., 0], velocity, rtol=0., atol=1.e-6)
                    & np.isclose(command[..., 2], height, rtol=0., atol=1.e-6))
    windows = {'entire_prefix': np.ones_like(done)}
    if through >= 40:
        first, last = np.zeros_like(done), np.zeros_like(done)
        first[:20 * 48], last[-20 * 48:] = True, True
        windows.update(first20_updates=first, last20_updates=last)
    rows, original_mae_error = {}, 0.
    for wn, window in windows.items():
        rows[wn] = {}
        for pn, population in populations.items():
            mask = settled & window & population
            count = int(mask.sum())
            metrics = None if not count else dict(
                body_yaw_MAE_rad_s=float(abs(error[mask]).mean()),
                signed_body_minus_command_mean_rad_s=float(error[mask].mean()),
                body_yaw_mean_rad_s=float(body[mask].mean()),
                command_yaw_mean_rad_s=float(yaw_command[mask].mean()),
                components={name: dict(signed_mean_rad_s=float(value[mask].mean()),
                    MAE_rad_s=float(abs(value[mask]).mean())) for name, value in components.items()})
            rows[wn][pn] = dict(recorded_settled_transitions=count, metrics=metrics)
            if pn in ('dedicated_static', 'motion_population'):
                original = physical['groups'][wn][pn]['settled_sampled_transitions']
                assert count == original['samples']
                if count:
                    original_mae_error = max(original_mae_error,
                        abs(metrics['body_yaw_MAE_rad_s'] - original['yaw_MAE_rad_s']))
    assert original_mae_error < 1.e-7
    assert all(sha(p) == digest for p, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        committed_prefix=through, native_population=plan['native_num_envs'], sampled_environment_ids=ids,
        source_actual_updates=plan['source_actual_ppo_updates'],
        formula='body-command=(deterministic_clipped_mean_reference-command)+(sampled_reference-deterministic_clipped_mean_reference)+(body-sampled_reference)',
        signed_component_closure_max_error_rad_s=closure,
        original_settled_masks_and_population_counts_exact=True,
        original_body_MAE_float64_reproduction_max_error_rad_s=original_mae_error,
        saved_wheel_reference_limit_rad_s=scale * limit, groups=rows, source_pins=pins,
        simulator_started=False, optimizer_executed=False, RNG_draws_added=0,
        training_modified=False, controller_changed=False, acceptance_changed=False,
        next_parameter_change_selected=False, policy_accepted=False,
        limitations=[
            'Only32 recorded environments from evolving Gaussian-policy training, including changing commands and resets; not independent acceptance.',
            'The body-versus-target residual includes dynamics, noise filtering, contact and velocity-servo response; no post-step wheel velocity was recorded.',
            'Clipped mean is the deterministic evaluation reference, not the expectation of a clipped Gaussian.',
            'Signed components add algebraically; MAEs do not. First/last windows are not matched-state causal comparisons.',
            'A zero sample count means this recorded subset contains no exposure, not that the8192 training population lacks that condition.'])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, report = audit(args.directory, args.through)
    print(json.dumps(dict(verified=True, report=str(path),
        closure_max_error=report['signed_component_closure_max_error_rad_s'],
        original_body_MAE_max_error=report['original_body_MAE_float64_reproduction_max_error_rad_s'],
        training_modified=False, policy_accepted=False)), flush=True)
