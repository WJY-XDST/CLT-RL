"""Qualify the prepared one-field yaw-cost200 transfer without PPO updates.

The ordinary retained-state CPU qualifier and original dense penalty reader
are reused. Native qualification additionally requires all frozen physics,
inputs, actions and every other reward to remain bitwise identical.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from qualify_source801_drift50_transfer import differences
from review_plain27_single_reward_block200 import rebound
import qualify_plain27_single_reward_transfer as original
import qualify_source6201_yawcost20_transfer as template
import joint_pd_preserved6801_yawcost200_runtime as runtime

SCRIPT = Path(__file__).resolve()
OUTPUT = runtime.QUALIFICATION
penalty_formula = template.penalty_formula


def qualify():
    source, _, _ = runtime.anchor()
    result = rebound(original.qualify, SCRIPT=SCRIPT)(
        source, runtime.SCRIPT, runtime.DECISION, OUTPUT)
    overrides = OUTPUT / 'native_reward_transfer_overrides.json'
    with overrides.open('x') as stream:
        json.dump({'rewards.' + runtime.REWARD_KEY: runtime.REWARD_VALUE}, stream)
        stream.write('\n')
    result['source_files'].update({str(p): sha(p) for p in
        (SCRIPT, original.SCRIPT, template.SCRIPT, overrides)})
    result.update(original_single_field_CPU_operations_preserved=True,
        expected_saved_wheel_action_limit=1.25,
        expected_saved_wheel_reference_cap_rad_s=18.75,
        all_source_exploration_floors_unchanged=.01,
        only_reward_change=runtime.DELTA['rewards'],
        native_frozen_transfer_complete=False)
    write_json(OUTPUT / 'qualification.json', result)
    return result


def frozen():
    runtime.repair_ready()
    before, after = runtime.TRAINING / 'replay', OUTPUT / 'native_reward_transfer'
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = [read(p, 'manifest.json') for p in (before, after)]
    da, db = [read(p, 'completed.json') for p in (before, after)]
    ca, cb = [read(p, 'resolved_env.json') for p in (before, after)]
    key = runtime.REWARD_KEY
    if (a['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or b['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not da['completed'] or not db['completed']
            or da['samples'] != 27000 or db['samples'] != 27000
            or da['resets'] != db['resets']
            or differences(ca, cb) != ['/rewards/' + key]
            or ca['rewards'][key] != runtime.OLD_VALUE
            or cb['rewards'][key] != runtime.REWARD_VALUE):
        raise ValueError('Require the exact frozen27 source and one declared reward coefficient')
    old, new = [np.genfromtxt(p / 'trace.csv', delimiter=',', names=True)
                for p in (before, after)]
    changed = {'reward_total', 'reward_yaw_rate_error_sq'}
    if (old.dtype.names != new.dtype.names or len(new) != 27000
            or any(not np.isfinite(new[k]).all() for k in new.dtype.names)
            or any(not np.array_equal(old[k], new[k])
                   for k in old.dtype.names if k not in changed)):
        raise ValueError('Another reward, physical trajectory or executed action changed')
    for rows, done in ((old, da), (new, db)):
        if np.count_nonzero((rows['terminated'] != 0) | (rows['timeout'] != 0)) != done['resets']:
            raise ValueError('Keep all raw OR resets and failed first episodes')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(
            after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[k], y[k]) for k in x.files):
            raise ValueError('Frozen observations, actor means or input metadata changed')
    errors = [float(abs(penalty_formula(rows, cfg, manifest['policy_dt'])
                        - rows['reward_yaw_rate_error_sq']).max())
              for rows, cfg, manifest in ((old, ca, a), (new, cb, b))]
    residual = abs(new['reward_total'] - old['reward_total']
                   - new['reward_yaw_rate_error_sq'] + old['reward_yaw_rate_error_sq'])
    tolerance = 8 * np.finfo(np.float32).eps * np.maximum.reduce(
        (np.ones(len(new)), abs(old['reward_total']), abs(new['reward_total'])))
    evidence = read(after, 'preserved6801_yawcost200_base_execution_evidence.json')
    sampler = read(after, 'retained_targeted030_yawcost200_runtime_evidence.json')
    if (max(errors) > 2.e-7 or np.any(residual > tolerance)
            or evidence['profile'] != runtime.profile()
            or evidence['authoritative_current_delta'] != runtime.DELTA
            or sampler['profile'] != runtime.profile()
            or sampler['authoritative_current_delta'] != runtime.DELTA):
        raise ValueError('Original yaw penalty formula, total delta or actual runtime differs')
    pins = {str(p / name): sha(p / name) for p in (before, after) for name in
        ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')}
    pins.update({str(p): sha(p) for p in (SCRIPT, original.SCRIPT, template.SCRIPT,
        runtime.SCRIPT, after / 'preserved6801_yawcost200_base_execution_evidence.json',
        after / 'retained_targeted030_yawcost200_runtime_evidence.json')})
    result = dict(qualified=True, source_sha256=runtime.ANCHOR_SHA,
        only_reward_change=runtime.DELTA['rewards'], full27_samples=27000,
        identical_source_resets_including_failures=da['resets'],
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        original_dense_yaw_penalty_formula_reader_reused=True,
        independent_old_yaw_penalty_formula_max_error=errors[0],
        independent_new_yaw_penalty_formula_max_error=errors[1],
        total_reward_delta_max_error=float(residual.max()),
        positive_yaw_reward_and_width_bitwise_unchanged=True,
        height_posture_commands_std_sampler_optimizer_and_execution_unchanged=True,
        source_data_sha256=pins, implementation_sha256=sha(runtime.SCRIPT),
        qualifier_sha256=sha(SCRIPT), PPO_updates_performed=0,
        optimizer_steps_executed=0, policy_accepted=False, learning_benefit_unproven=True)
    write_json(OUTPUT / 'native_reward_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'frozen'))
    result = {'qualify': qualify, 'frozen': frozen}[parser.parse_args().mode]()
    print(json.dumps({k: result[k] for k in ('qualified', 'source_actual_updates',
        'only_reward_change', 'all_model_and_Adam_tensors_bitwise_preserved', 'full27_samples')
        if k in result}))
