"""Qualify one yaw-quadratic coefficient after restoring the complete6201 state."""
import argparse
import json
from pathlib import Path

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from qualify_source801_drift50_transfer import differences
from review_plain27_single_reward_block200 import rebound
import qualify_plain27_single_reward_transfer as original
import joint_pd_source6201_yawcost20_runtime as runtime

SCRIPT = Path(__file__).resolve()
OUTPUT = ROOT / 'mine_model/results/joint_pd_source6201_yawcost20_qualification_20261021'


def qualify():
    source, _, _ = runtime.anchor()
    result = rebound(original.qualify, SCRIPT=SCRIPT)(
        source, runtime.SCRIPT, runtime.DECISION, OUTPUT)
    result['source_files'].update({str(p): sha(p) for p in (SCRIPT, original.SCRIPT)})
    result.update(original_single_field_CPU_operations_preserved=True,
        expected_saved_wheel_action_limit=1.25,
        expected_saved_wheel_reference_cap_rad_s=18.75,
        only_reward_change=runtime.DELTA['rewards'])
    write_json(OUTPUT / 'qualification.json', result)
    return result


def penalty_formula(rows, cfg, dt):
    rewards = cfg['rewards']
    value = rewards['yaw_rate_error_sq'] * (
        rows['yaw_rate_rad_s'] - rows['yaw_cmd_rad_s']) ** 2 * dt
    return np.clip(value, -rewards['yaw_rate_penalty_clip'] * dt,
                   rewards['clip_single_reward'] * dt)


def frozen():
    runtime.repair_ready()
    before, after = runtime.TRAINING / 'replay', OUTPUT / 'native_reward_transfer'
    read = lambda directory, name: json.loads((directory / name).read_text())
    a, b = [read(p, 'manifest.json') for p in (before, after)]
    da, db = [read(p, 'completed.json') for p in (before, after)]
    ca, cb = [read(p, 'resolved_env.json') for p in (before, after)]
    if (a['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or b['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not da['completed'] or not db['completed']
            or da['samples'] != 27000 or db['samples'] != 27000
            or da['resets'] != db['resets']
            or differences(ca, cb) != ['/rewards/yaw_rate_error_sq']
            or ca['rewards']['yaw_rate_error_sq'] != -1.
            or cb['rewards']['yaw_rate_error_sq'] != -20.):
        raise ValueError('Require the exact frozen27 source and only the declared yaw penalty')
    old, new = [np.genfromtxt(p / 'trace.csv', delimiter=',', names=True)
                for p in (before, after)]
    changed = {'reward_total', 'reward_yaw_rate_error_sq'}
    if (old.dtype.names != new.dtype.names or len(new) != 27000
            or any(not np.isfinite(new[k]).all() for k in new.dtype.names)
            or any(not np.array_equal(old[k], new[k])
                   for k in old.dtype.names if k not in changed)):
        raise ValueError('Another reward, physical trajectory, action or observation changed')
    for rows, done in ((old, da), (new, db)):
        if np.count_nonzero((rows['terminated'] != 0) | (rows['timeout'] != 0)) != done['resets']:
            raise ValueError('Keep every raw OR reset and original failed episode')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(
            after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[k], y[k]) for k in x.files):
            raise ValueError('Frozen policy inputs or means changed')
    errors = [float(abs(penalty_formula(rows, cfg, manifest['policy_dt'])
                        - rows['reward_yaw_rate_error_sq']).max())
              for rows, cfg, manifest in ((old, ca, a), (new, cb, b))]
    residual = abs(new['reward_total'] - old['reward_total']
                   - new['reward_yaw_rate_error_sq'] + old['reward_yaw_rate_error_sq'])
    tolerance = 8 * np.finfo(np.float32).eps * np.maximum.reduce(
        (np.ones(len(new)), abs(old['reward_total']), abs(new['reward_total'])))
    evidence = read(after, 'yawcost20_runtime_evidence.json')
    if (max(errors) > 2.e-7 or np.any(residual > tolerance)
            or evidence['profile'] != runtime.profile()
            or evidence['authoritative_current_delta'] != runtime.DELTA):
        raise ValueError('Yaw penalty formula, total delta or actual runtime differs')
    result = dict(qualified=True, source_sha256=runtime.ANCHOR_SHA,
        only_reward_change=runtime.DELTA['rewards'], full27_samples=27000,
        identical_source_resets_including_failures=da['resets'],
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        independent_old_yaw_penalty_formula_max_error=errors[0],
        independent_new_yaw_penalty_formula_max_error=errors[1],
        total_reward_delta_max_error=float(residual.max()),
        positive_yaw_reward_and_width_bitwise_unchanged=True,
        height_posture_commands_std_optimizer_and_execution_unchanged=True,
        source_data_sha256={str(p / n): sha(p / n) for p in (before, after)
            for n in ('manifest.json', 'completed.json', 'resolved_env.json',
                      'trace.csv', 'policy_inputs.npz')},
        implementation_sha256=sha(runtime.SCRIPT), qualifier_sha256=sha(SCRIPT),
        PPO_updates_performed=0, policy_accepted=False,
        learning_benefit_unproven=True)
    for p in (SCRIPT, original.SCRIPT, runtime.SCRIPT):
        result['source_data_sha256'][str(p)] = sha(p)
    write_json(OUTPUT / 'native_reward_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'frozen'))
    args = parser.parse_args()
    result = qualify() if args.mode == 'qualify' else frozen()
    print(json.dumps({k: result[k] for k in ('qualified', 'source_actual_updates',
        'only_reward_change', 'all_model_and_Adam_tensors_bitwise_preserved', 'full27_samples')
        if k in result}))
