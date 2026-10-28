"""Pair immutable fixed-command replays across a declared global curriculum change.

Only sampler metadata can be normalized, after verifying identical commands,
seeds and initial27 observations. The original comparison/scoring code remains
unchanged. This reader has no training, policy inference or simulator entry.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import types

import numpy as np

import compare_uniform_joint_pd_replay_retention as original
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def metadata_changes(before, after, declared):
    allowed = {'training_motion_scale', 'yaw_min_abs_rate'}
    assert set(declared) == allowed
    changed = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys()
               if before.get(key) != after.get(key)}
    assert not changed or changed == declared, 'The coupled curriculum fields must change together'
    return changed


def compare_pair(source, candidate, declared, scope):
    a, b = [original.load(Path(path)) for path in (source, candidate)]
    before, after = a[2], b[2]
    changed = metadata_changes(before['commands'], after['commands'], declared)
    assert a[1]['command_probe'] and b[1]['command_probe']
    assert a[1]['cases'] == b[1]['cases'] and a[1]['seeds'] == b[1]['seeds']
    initial = []
    extra_pins = {}
    for path in (source, candidate):
        data = Path(path) / 'policy_inputs.npz'
        with np.load(data, allow_pickle=False) as arrays:
            value = arrays['observations'][0].copy()
        assert value.ndim == 2 and value.shape[-1] == 27 and np.isfinite(value).all()
        initial.append(value)
        extra_pins[str(data.resolve())] = sha(data)
    assert np.array_equal(*initial), 'Initial policy inputs must match exactly'

    def normalized_load(path):
        result = list(original.load(path))
        if Path(path).resolve() == Path(candidate).resolve():
            cfg = copy.deepcopy(result[2])
            for key in changed:
                cfg['commands'][key] = before['commands'][key]
            result[2] = cfg
        return tuple(result)

    numerical_compare = types.FunctionType(original.compare.__code__,
        dict(original.compare.__globals__, load=normalized_load), original.compare.__name__,
        original.compare.__defaults__, original.compare.__closure__)
    report, pins = numerical_compare(source, candidate, scope)
    report.update(declared_fixed_command_sampler_metadata_differences=changed,
        initial27_observations_bitwise_equal=True,
        original_numerical_comparison_code_unchanged=True)
    return report, {**pins, **extra_pins}


def wait_for_completion(candidate, unit):
    marker = candidate / 'complete_frozen_macro_envelope.json'
    while not marker.exists():
        assert unit, 'Incomplete candidate needs its live owner, never repeat replay'
        result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'SubState'],
                                text=True, capture_output=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['MainPID'] != '0', ('Inspect incomplete owned replay', state)
        time.sleep(15.)
    return marker


def run(args):
    marker = wait_for_completion(args.candidate, args.after_owned_unit)
    recipe = read(args.recipe)
    assert recipe['single_new_global_envelope_degree_of_freedom']
    assert not recipe['case_specific_training_logic'] and not recipe['acceptance_changed']
    assert not args.output.exists(), 'Do not overwrite a completed comparison'
    spec = [read(path / 'qualification.json') for path in (args.source, args.candidate)]
    reports = [read(path / 'complete_frozen_macro_envelope.json') for path in (args.source, args.candidate)]
    assert all(item['complete'] and not item['acceptance_changed'] for item in reports)
    assert all(item['trial_count'] == 189 for item in spec)
    assert len(spec[0]['batches']) == len(spec[1]['batches']) == 2
    for item in spec:
        assert all(sha(path) == digest for path, digest in item['source_pins'].items())
    comparisons, pairs = [], []
    pins = {str(SCRIPT): sha(SCRIPT), str(Path(original.__file__).resolve()): sha(original.__file__),
            str(args.recipe.resolve()): sha(args.recipe)}
    for i, (a, b) in enumerate(zip(spec[0]['batches'], spec[1]['batches'])):
        report, inputs = compare_pair(Path(a['replay']), Path(b['replay']), recipe['command_changes'],
                                      f'global_envelope189_batch{i}')
        comparisons.append(report)
        pairs.extend(report['all_pairs'])
        pins.update(inputs)
    assert len(pairs) == 189
    assert len({(row['case']['name'], row['seed']) for row in pairs}) == 189
    summary = original.summarize_pairs(pairs)
    for path in (args.source / 'complete_frozen_macro_envelope.json', marker):
        pins[str(path.resolve())] = sha(path)
    assert all(sha(path) == digest for path, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        source_checkpoint_sha256=reports[0]['checkpoint_sha256'],
        candidate_checkpoint_sha256=reports[1]['checkpoint_sha256'],
        source_summary=reports[0]['summary'], candidate_summary=reports[1]['summary'],
        paired_summary=summary, comparisons=comparisons, all189_pairs=pairs, source_pins=pins,
        every_requested_case_and_seed_retained=True, initial27_observations_bitwise_equal=True,
        condition_values_used_for_diagnosis_only=True, original_numeric_scoring_unchanged=True,
        acceptance_changed=False, training_modified=False, optimizer_executed=False,
        simulator_started=False, policy_accepted=False,
        limitation='The wider envelope189 is separate from old-scope525 retention and final full-speed sustained acceptance.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, paired_summary=summary)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-owned-unit')
    run(parser.parse_args())
