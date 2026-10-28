"""Versioned read-only recovery for saved wheel-bound diagnostics.

Keep old reports and logs. Use an explicit one-controller-field admission in
the fixed-input comparison, and the saved wheel bound in clipping counters.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from joint_pd_continuation import sha
import monitor_wheel125_learning_50_100_200 as base

SCRIPT = Path(__file__).resolve()
PHYSICAL_POINTER_ADAPTER = base.adapter.retarget(base.health, ('tick',),
    integers={'*': {40: 50}},
    strings={'native_motion_training_prefix_': 'native_motion_training_saved_limits_prefix_'})
base.health.CHECKS = (
    ('audit_effective_joint_pd_exploration_saved_limits.py', 'effective_exploration', []),
    ('audit_native_motion_training_saved_limits.py', 'native_motion_training_saved_limits', []),
    ('audit_joint_pd_policy_updates.py', 'policy_update_health', ['--interval', '1', '--device', 'cuda:0']),
)
base.extra.AUDITORS = tuple('compare_plain27_recorded_joint_targets_saved_limits.py'
    if name == 'compare_plain27_recorded_joint_targets.py' else name for name in base.extra.AUDITORS)


def tick():
    result = base.tick()
    if result.get('state') != 'observing':
        return result
    directory = Path(result['directory'])
    physical = {}
    for prefix in (50, 100, 200):
        combined = directory / f'learning_health_combined_prefix_{prefix:04d}_readonly.json'
        corrected = directory / f'native_motion_training_saved_limits_prefix_{prefix:04d}_readonly.json'
        if not combined.exists() or not corrected.exists():
            continue
        destination = directory / f'learning_health_saved_limits_combined_prefix_{prefix:04d}_readonly.json'
        if not destination.exists():
            value = json.loads(combined.read_text())
            value.update(physical_training_report=str(corrected), preserved_earlier_combined_report=str(combined),
                corrected_wheel_counter_uses_saved_limit=True,
                original_physical_errors_exploration_and_raw_KL_unchanged=True,
                source_pins={str(p): sha(p) for p in (combined, corrected, SCRIPT)})
            with destination.open('x') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        physical[str(prefix)] = str(corrected)
    result.update(reader_version='saved_wheel_bounds_v2', monitor_sha256=sha(SCRIPT),
        preserved_base_monitor_sha256=sha(base.SCRIPT),
        physical_report_pointer_metadata_adapter=PHYSICAL_POINTER_ADAPTER,
        physical_training_report_by_prefix=physical,
        legacy_unit_bound_reports_preserved=True,
        original_PPO_and_active_training_process_untouched=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            result = tick()
        except (FileNotFoundError, json.JSONDecodeError):
            result = dict(observed_at=datetime.now().astimezone().isoformat(), state='waiting_for_committed_data',
                training_modified=False, policy_accepted=False)
        except BaseException as error:
            result = dict(observed_at=datetime.now().astimezone().isoformat(), state='agent_inspection_required',
                error=repr(error), training_modified=False, policy_accepted=False)
        temporary = base.STATUS.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(base.STATUS)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.once:
            break
        time.sleep(15.)
