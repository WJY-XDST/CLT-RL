"""Recover post-step wheel sensors from aligned next observations, read-only.

Current plain27 has no sensor noise. Four separately recorded physical state
channels verify the next-observation timing before wheel columns are decoded.
This is an evolving-policy training diagnostic, never independent acceptance.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from audit_native_sampled_yaw_reference_components import ROOT, sha


SCRIPT = Path(__file__).resolve()
ENV = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
ENV_SHA = 'c1dd69e35d26d518f2d282c74c4fe081e789f3e8514c4865ea8604d2c3997a7e'


def audit(directory, through):
    directory = Path(directory).resolve()
    assert directory.parent == ROOT / 'mine_model/results' and through in (100, 200)
    destination = directory / f'recovered_wheel_yaw_components_prefix_{through:04d}_readonly.json'
    assert not destination.exists() and sha(ENV) == ENV_SHA
    parent_path = directory / f'sampled_yaw_reference_components_prefix_{through:04d}_readonly.json'
    parent = json.loads(parent_path.read_text())
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg, sc = plan['training_config'], plan['training_config']['obs_scales']
    assert plan['source_files'][str(ENV)] == ENV_SHA
    assert cfg['leg_control_mode'] == 'implicit_joint_position' and cfg['observation_space'] == 27
    assert cfg.get('observation_noise_model') is None and cfg.get('action_noise_model') is None
    assert parent['verified'] and parent['committed_prefix'] == through
    pins = dict(parent['source_pins'])
    pins.update({str(p): sha(p) for p in (SCRIPT, ENV, plan_path, parent_path)})
    fields = ('observations', 'native_commands', 'native_yaw', 'native_gravity', 'native_speed',
              'native_joint_positions', 'actions', 'mu', 'dones')
    parts = {key: [] for key in fields}
    for update in range(1, through + 1):
        path = directory / f'native_learning_signal/rollout_{update:04d}.npz'
        assert sha(path) == pins[str(path)]
        with np.load(path, allow_pickle=False) as archive:
            for key in fields:
                assert np.isfinite(archive[key]).all()
                parts[key].append(archive[key].copy())
    data = {key: np.concatenate(value) for key, value in parts.items()}
    command, done = data['native_commands'], data['dones'][..., 0].astype(bool)
    assert data['observations'].shape == (*done.shape, 27)
    obs, valid = data['observations'][1:], ~done[:-1]
    offset = np.asarray(cfg['leg_joint_position_offset'], dtype=np.float64)
    parity = {}
    for name, decoded, native in (
        ('body_yaw', obs[..., 2] / sc['ang_vel'], data['native_yaw'][:-1]),
        ('projected_gravity', obs[..., 3:6], data['native_gravity'][:-1]),
        ('heading_speed_x', obs[..., 17] / sc['lin_vel'], data['native_speed'][:-1]),
        ('active_joint_position', obs[..., 9:13] / sc['dof_pos'] + offset,
         data['native_joint_positions'][:-1]),
    ):
        parity[name] = float(abs(decoded - native)[valid].max())
    assert max(parity.values()) < 2.e-6, parity
    # Columns19/20 already use forward-positive coordinates; do not flip right again.
    wheels = obs[..., 19:21].astype(np.float64) / sc['dof_vel']
    assert abs(wheels[valid]).max() < cfg['obs_clip']['wheel_vel'] - 1.e-4
    dt = cfg['sim']['dt'] * cfg['decimation']
    age, stable = np.empty(done.shape), np.empty(done.shape)
    age[0], stable[0] = dt, 0.
    for t in range(1, len(age)):
        age[t] = np.where(done[t - 1], dt, age[t - 1] + dt)
        changed = np.any(abs(command[t] - command[t - 1]) > 1.e-7, axis=-1)
        stable[t] = np.where(changed | done[t - 1], 0., stable[t - 1] + dt)
    settled = (age > 2.) & (stable >= 1.) & ~done
    static = np.broadcast_to(np.asarray(parent['sampled_environment_ids'])[None, :]
        < plan['fixed_standing_environments'], done.shape)
    yaw = command[..., 1].astype(np.float64)
    small = (abs(yaw) >= .05) & (abs(yaw) < .10)
    populations = dict(dedicated_static=static, motion_population=~static,
        small_yaw_positive=small & (yaw > 0.), small_yaw_negative=small & (yaw < 0.))
    heights = {'low': plan['full_height_range'][0], 'nominal': cfg['rewards']['base_height_target'],
               'high': plan['full_height_range'][1]}
    for hn, height in heights.items():
        for vn, velocity in [('reverse_fast', -.8), ('spin', 0.), ('forward_fast', .8)]:
            for yn, sign in [('positive', 1), ('negative', -1)]:
                populations[f'{hn}_{vn}_small_yaw_{yn}'] = (small & (yaw * sign > 0.)
                    & np.isclose(command[..., 0], velocity, rtol=0., atol=1.e-6)
                    & np.isclose(command[..., 2], height, rtol=0., atol=1.e-6))
    windows = {'entire_prefix': np.ones_like(done)}
    first, last = np.zeros_like(done), np.zeros_like(done)
    first[:20 * 48], last[-20 * 48:] = True, True
    windows.update(first20_updates=first, last20_updates=last)
    ratio = cfg['wheel_radius'] / cfg['wheel_track_width']
    limit, scale = cfg['wheel_action_limit'], cfg['action_scale_vel']
    assert limit == 1.25 and scale == 15.

    def reference(key):
        velocity = np.clip(data[key][:-1, :, [2, 5]].astype(np.float64), -limit, limit) * scale
        return ratio * (velocity[..., 1] - velocity[..., 0])

    deterministic, sampled = reference('mu'), reference('actions')
    measured = ratio * (wheels[..., 1] - wheels[..., 0])
    body = data['native_yaw'][:-1].astype(np.float64)
    error = body - yaw[:-1]
    components = dict(deterministic_reference_minus_command=deterministic - yaw[:-1],
        sampled_minus_deterministic_reference=sampled - deterministic,
        measured_wheel_proxy_minus_sampled_reference=measured - sampled,
        body_minus_measured_wheel_proxy=body - measured)
    closure = float(abs(sum(components.values()) - error)[valid].max())
    assert closure < 1.e-12
    rows = {}
    for wn, window in windows.items():
        rows[wn] = {}
        for pn, population in populations.items():
            full_mask = settled & window & population
            original_count = parent['groups'][wn][pn]['recorded_settled_transitions']
            assert int(full_mask.sum()) == original_count
            mask = full_mask[:-1] & valid
            count = int(mask.sum())
            assert original_count - count == int(full_mask[-1].sum())
            rows[wn][pn] = dict(original_settled_count=original_count,
                recovered_sensor_settled_count=count, excluded_last_timestep_count=original_count - count,
                metrics=None if not count else dict(body_yaw_MAE_rad_s=float(abs(error[mask]).mean()),
                    signed_body_minus_command_mean_rad_s=float(error[mask].mean()),
                    components={name: dict(signed_mean_rad_s=float(value[mask].mean()),
                        MAE_rad_s=float(abs(value[mask]).mean())) for name, value in components.items()}))
    assert all(sha(p) == digest for p, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        committed_prefix=through, native_population=plan['native_num_envs'],
        source_actual_updates=plan['source_actual_ppo_updates'],
        reconstructed_nonterminal_poststep_states=int(valid.sum()),
        physical_state_timing_parity=parity,
        wheel_sensor_columns=[19, 20], wheel_sensor_scale=sc['dof_vel'],
        right_wheel_already_canonical_forward_positive=True,
        recovered_wheel_speed_range_rad_s=[float(wheels[valid].min()), float(wheels[valid].max())],
        signed_component_closure_max_error_rad_s=closure, groups=rows, source_pins=pins,
        simulator_started=False, optimizer_executed=False, RNG_draws_added=0,
        training_modified=False, controller_changed=False, acceptance_changed=False,
        next_parameter_change_selected=False, policy_accepted=False,
        limitations=[
            'All eligible nonterminal next observations pass four independently recorded state-channel checks before wheel decoding.',
            'Final32 states lack a next recorded observation and are explicitly excluded; no other settled-state filtering is added.',
            'Only32 sampled environments under evolving Gaussian policies, not independent trial-level acceptance.',
            'A wheel velocity-servo residual can supply load torque; it does not alone prove a gain bug.',
            'Body-versus-wheel differential kinematics is not a direct slip or tire-friction measurement.',
            'Signed terms add; MAEs do not. First/last windows have unmatched state and command distributions.'])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, choices=(100, 200), required=True)
    cli.add_argument('--wait-monitor-unit')
    args = cli.parse_args()
    if args.wait_monitor_unit:
        assert args.through == 200
        parent_path = args.directory / 'sampled_yaw_reference_components_prefix_0200_readonly.json'
        while True:
            status = subprocess.check_output(['systemctl', '--user', 'show', args.wait_monitor_unit,
                '--property=MainPID,SubState,Result'], text=True)
            owner = dict(line.split('=', 1) for line in status.splitlines() if '=' in line)
            try:
                ready = json.loads(parent_path.read_text()).get('verified') is True
            except (FileNotFoundError, json.JSONDecodeError):
                ready = False
            if ready:
                break
            assert owner.get('MainPID') != '0', ('Inspect the owned completed200 reader', owner)
            time.sleep(15.)
    path, result = audit(args.directory, args.through)
    print(json.dumps(dict(verified=True, report=str(path),
        physical_state_timing_parity=result['physical_state_timing_parity'],
        closure_max_error=result['signed_component_closure_max_error_rad_s'],
        training_modified=False, policy_accepted=False)), flush=True)
