"""Summarize and pair frozen replay evidence without selecting training cases.

The numerical acceptance reader supplies the per-trial flags. This file only
checks their stored relative errors, counts every trial (including failures),
and exposes where errors occur. It has no training or simulator entry point.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean


AXES = {'speed': 'speed_mae_m_s', 'yaw': 'yaw_mae_rad_s',
        'height': 'height_mae_m'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row):
    return row['case']['name'], row['seed']


def numeric_command(row):
    return tuple(float(row['case'][axis]) for axis in AXES)


def load(path):
    data = json.loads(path.read_text())
    assert data.get('complete') or data.get('verified'), path
    rows = data.get('all_trials', data.get('trials'))
    assert isinstance(rows, list) and rows
    assert len({identity(row) for row in rows}) == len(rows)
    for row in rows:
        assert row['survived'] == (row['metrics'] is not None)
        for axis, field in AXES.items():
            stored = row['nonzero_relative_percent'][axis]
            expected = (100. * row['metrics'][field] / abs(row['case'][axis])
                        if row['metrics'] is not None and row['case'][axis] else None)
            assert stored is None if expected is None else abs(stored - expected) < 1.e-9
        literal = row['original_passed'] and all(
            value < 10. for value in row['nonzero_relative_percent'].values()
            if value is not None)
        assert literal == row['original_AND_literal10_passed']
    # Read-only reports may include large trace pins. Check all of them so a
    # cached assessment is not silently combined with changed source evidence.
    for source, digest in data.get('source_pins', {}).items():
        assert sha(source) == digest, source
    return data, rows


def brief(row):
    return dict(case=row['case'], seed=row['seed'], survived=row['survived'],
                metrics=row['metrics'], limits=row['limits'], flags=row['flags'],
                original_passed=row['original_passed'],
                original_AND_literal10_passed=row['original_AND_literal10_passed'],
                nonzero_relative_percent=row['nonzero_relative_percent'])


def summary(rows):
    result = dict(trials=len(rows), survivors=sum(r['survived'] for r in rows),
                  original_passed=sum(r['original_passed'] for r in rows),
                  strict10_passed=sum(r['original_AND_literal10_passed'] for r in rows),
                  failed_metrics_retained_as_null=sum(r['metrics'] is None for r in rows))
    for axis in AXES:
        valid = [r for r in rows if r['nonzero_relative_percent'][axis] is not None]
        errors = [r['nonzero_relative_percent'][axis] for r in valid]
        worst = max(valid, key=lambda r: r['nonzero_relative_percent'][axis]) if valid else None
        result[axis] = dict(valid_nonzero_trials=len(valid),
                            mean_relative_MAE_percent=mean(errors) if errors else None,
                            worst_relative_MAE_percent=max(errors, default=None),
                            failed_relative10=sum(value >= 10. for value in errors),
                            worst_trial=brief(worst) if worst else None)
    valid = [r['metrics'] for r in rows if r['metrics'] is not None]
    result['worst_body_P95_deg'] = max((m['body_p95_deg'] for m in valid), default=None)
    result['zero_command_errors'] = {}
    for axis in ('speed', 'yaw'):
        values = [r['metrics'][AXES[axis]] for r in rows
                  if r['metrics'] is not None and r['case'][axis] == 0.]
        result['zero_command_errors'][axis] = dict(
            valid_trials=len(values), mean_MAE=mean(values) if values else None,
            worst_MAE=max(values, default=None))
    return result


def profile(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[numeric_command(row)].append(row)
    commands = [dict(speed=cmd[0], yaw=cmd[1], height=cmd[2],
                     labels=sorted({r['case']['name'] for r in group}),
                     seeds=sorted({r['seed'] for r in group}), summary=summary(group))
                for cmd, group in sorted(groups.items())]
    by_axis = {}
    for axis in AXES:
        grouped = defaultdict(list)
        for row in rows:
            grouped[float(row['case'][axis])].append(row)
        by_axis[axis] = [dict(command=value, summary=summary(group))
                         for value, group in sorted(grouped.items())]
    balanced = {}
    for axis in AXES:
        values = [r['summary'][axis]['mean_relative_MAE_percent'] for r in commands
                  if r['summary'][axis]['mean_relative_MAE_percent'] is not None]
        balanced[axis] = dict(valid_unique_commands=len(values),
                              equal_command_mean_relative_MAE_percent=mean(values) if values else None)
    return dict(summary=summary(rows), unique_numeric_commands=len(groups),
                unique_commands_failing_any_gate=sum(
                    any(not r['original_AND_literal10_passed'] for r in group)
                    for group in groups.values()),
                unique_commands_failing_relative10={axis: sum(
                    any(r['nonzero_relative_percent'][axis] is not None and
                        r['nonzero_relative_percent'][axis] >= 10. for r in group)
                    for group in groups.values()) for axis in AXES},
                equal_numeric_command_weighting=balanced,
                by_command_axis=by_axis, all_unique_numeric_commands=commands,
                all_failed_trials=[brief(r) for r in rows if not r['original_AND_literal10_passed']])


def run(after_path, output, before_path=None):
    assert not output.exists(), 'Preserve previous diagnostics'
    after_data, after = load(after_path)
    pins = {str(Path(__file__).resolve()): sha(__file__), str(after_path): sha(after_path)}
    result = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  source_report=str(after_path), checkpoint_sha256=after_data.get('checkpoint_sha256'),
                  after=profile(after), diagnostics_only=True, simulator_started=False,
                  optimizer_executed=False, training_modified=False, acceptance_changed=False,
                  case_specific_training_or_sampling=False, policy_accepted=False)
    if before_path:
        before_data, before = load(before_path)
        pins[str(before_path)] = sha(before_path)
        left = {identity(row): row for row in before}
        right = {identity(row): row for row in after}
        assert left.keys() == right.keys(), 'Paired comparison requires every original trial'
        assert all(numeric_command(left[k]) == numeric_command(right[k]) for k in left)
        result.update(before_report=str(before_path),
                      before_checkpoint_sha256=before_data.get('checkpoint_sha256'),
                      before=profile(before), pairs=len(left), all_commands_and_seeds_matched=True,
                      newly_failed=sum(left[k]['original_AND_literal10_passed'] and
                                       not right[k]['original_AND_literal10_passed'] for k in left),
                      recovered=sum(not left[k]['original_AND_literal10_passed'] and
                                    right[k]['original_AND_literal10_passed'] for k in left),
                      lost_survival=sum(left[k]['survived'] and not right[k]['survived'] for k in left),
                      lost_original_gates=sum(left[k]['original_passed'] and
                                              not right[k]['original_passed'] for k in left))
    assert all(sha(path) == digest for path, digest in pins.items())
    result['source_pins'] = pins
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--after', type=Path, required=True)
    cli.add_argument('--before', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = run(args.after.resolve(), args.output.resolve(),
                 args.before.resolve() if args.before else None)
    print(json.dumps(dict(output=str(args.output), trials=result['after']['summary']['trials'],
                          unique_commands=result['after']['unique_numeric_commands'],
                          strict10_passed=result['after']['summary']['strict10_passed'],
                          policy_accepted=False)))
