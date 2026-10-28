"""Combine complete, scoped frozen-drive comparisons without training or gating edits.

Separate scopes may deliberately reuse a case name and seed at different command
values. Keep the scope in the identity and retain every requested recorded trial.
The existing numerical comparison is reused; no physical test is started here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from compare_uniform_joint_pd_replay_retention import summarize_pairs
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def combine(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = read(plan_path)
    assert plan['qualified'] and not output.exists()
    assert plan['new_PPO_updates'] == 0 and not plan['training_configuration_selected']
    assert plan['groups'] and len({item['name'] for item in plan['groups']}) == len(plan['groups'])
    pins = dict(plan['source_pins'])
    pins.update({str(plan_path): sha(plan_path), str(SCRIPT): sha(SCRIPT)})
    scopes, pairs = [], []
    identities, physical = set(), set()
    for group in plan['groups']:
        directory = Path(group['output']).resolve()
        request = directory / 'qualification.json'
        report_path = directory / 'complete_frozen_uniform_wheel_drive.json'
        request_value, report = read(request), read(report_path)
        assert request_value['qualified'] and report['complete']
        assert not report['acceptance_changed'] and not report['training_modified']
        assert report['new_PPO_updates'] == 0 and not report['training_configuration_selected']
        assert request_value['source_checkpoint_sha256'] == report['source_checkpoint_sha256'] == plan['source_checkpoint_sha256']
        assert report['declared_global_damping'] == [plan['baseline_wheel_damping'], plan['probe_wheel_damping']]
        assert report['scoped_trial_count'] == request_value['trials'] == group['trials']
        assert len(report['comparisons']) == len(request_value['batches'])
        for batch, comparison in zip(request_value['batches'], report['comparisons']):
            label = group['name'] + '/' + batch['name']
            assert comparison['initial27_observations_bitwise_equal']
            assert comparison['first_unchanged_actor_mean_max_absolute_difference'] < 1.e-6
            assert len(comparison['all_pairs']) == batch['trials']
            scopes.append(dict(scope=label, source=batch['baseline'],
                               candidate=str(directory / ('probe_' + batch['name'])),
                               paired_summary=comparison['paired_summary'],
                               initial27_observations_bitwise_equal=True))
            for row in comparison['all_pairs']:
                identity = label, row['case']['name'], row['seed']
                assert identity not in identities
                identities.add(identity)
                physical.add(tuple(row['case'][axis] for axis in ('height', 'speed', 'yaw')) + (row['seed'],))
                pairs.append(dict(scope=label, **row))
        pins.update(report['source_pins'])
        pins.update({str(request): sha(request), str(report_path): sha(report_path)})
    assert len(pairs) == sum(group['trials'] for group in plan['groups']) == plan['scoped_trials']
    assert all(sha(path) == digest for path, digest in pins.items())
    result = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  source_checkpoint_sha256=plan['source_checkpoint_sha256'],
                  declared_global_damping=[plan['baseline_wheel_damping'], plan['probe_wheel_damping']],
                  scoped_trials=len(pairs), distinct_physical_case_seed_count=len(physical),
                  paired_summary=summarize_pairs(pairs), scopes=scopes, all_pairs=pairs,
                  all_requested_scopes_and_failures_retained=True,
                  all_initial27_observations_bitwise_equal=True,
                  original_numeric_comparison_unchanged=True, source_pins=pins,
                  new_PPO_updates=0, training_modified=False, optimizer_executed=False,
                  simulator_started=False, training_configuration_selected=False,
                  acceptance_changed=False, policy_accepted=False,
                  limitations=['Recorded scopes remain separate when identities or physical coordinates overlap.',
                               'These frozen interventions do not prove trained improvement, sustained acceptance or hardware calibration.'])
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(complete=True, scoped_trials=len(pairs),
                          distinct_physical_case_seed_count=len(physical),
                          paired_summary=result['paired_summary'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    combine(args.plan, args.output)
