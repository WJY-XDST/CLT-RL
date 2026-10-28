"""Read-only50/100/200 monitoring for one explicitly owned8192 PPO trial."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from joint_pd_continuation import ROOT, sha
import joint_pd_AF7001_yawsigma0025_runtime as runtime

SCRIPT = Path(__file__).resolve()
CHECKS = (
    ('audit_effective_joint_pd_exploration_saved_limits.py', 'effective_exploration', []),
    ('audit_native_motion_training_saved_limits.py', 'native_motion_training_saved_limits', []),
    ('audit_joint_pd_policy_updates.py', 'policy_update_health', ['--interval', '1', '--device', 'cuda:0']),
    ('audit_joint_pd_critic_and_command_coverage.py', 'critic_and_command_coverage', []),
    ('audit_parallel8192_saved_action_limits_v2.py', 'clipped_action_policy_KL', []),
    ('audit_parallel8192_count_bound.py', 'gradient_allocation', ['--reader', 'gradient']),
    ('audit_parallel8192_count_bound.py', 'critic_fit_by_resampling_phase', ['--reader', 'phase']),
    ('audit_parallel8192_count_bound.py', 'joint_target_geometric_exploration', ['--reader', 'geometry']),
    ('audit_parallel8192_count_bound.py', 'terminal_event_ages', ['--reader', 'events']),
    ('compare_parallel8192_recorded_joint_targets.py', 'recorded_source_joint_targets', []),
    ('audit_parallel8192_count_bound.py', 'matched_clipping_atoms', ['--reader', 'atoms']),
)


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def tick(output, unit):
    plan = json.loads((output / 'experiment_plan.json').read_text())
    if (plan['native_num_envs'] != 8192 or plan['fixed_standing_environments'] != 2048
            or plan['bounded_training_updates'] != 200 or plan['profile'] != runtime.profile()
            or not plan['ordinary_full_policy_PPO'] or plan['auxiliary_losses_enabled']
            or plan['next_configuration_changes'] != runtime.DELTA):
        raise ValueError('Admit only the exact unchanged-recipe8192 ordinary PPO block')
    committed = len(list((output / 'native_learning_signal').glob('audit_update_*.json')))
    finished = (output / 'native_learning_signal/completed_external.json').exists()
    completed = []
    for prefix in (50, 100, 200):
        if not (committed > prefix or finished and committed == prefix):
            continue
        for name, stem, options in CHECKS:
            destination = output / (stem + '_prefix_' + str(prefix).zfill(4) + '_readonly.json')
            if destination.exists():
                continue
            if stem == 'clipped_action_policy_KL':
                options = ['--updates', *map(str, range(1,prefix+1))]
            command = [sys.executable, str(SCRIPT.with_name(name)), str(output), '--through', str(prefix), *options]
            with (output / ('learning_health_' + stem + '_' + str(prefix).zfill(4) + '.log')).open('a') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        health = json.loads((output / ('policy_update_health_prefix_' + str(prefix).zfill(4) + '_readonly.json')).read_text())
        physical = json.loads((output / ('native_motion_training_saved_limits_prefix_' + str(prefix).zfill(4) + '_readonly.json')).read_text())
        if (health['native_num_envs'] != 8192 or physical['native_environment_count'] != 8192
                or not health['diagnostic_RNG_states_unchanged']
                or any(physical['all_environment_alignment_max_errors'].values())):
            raise ValueError('Resolve8192 native count, retained update or RNG discrepancies')
        path = output / ('learning_health_AF7001_yawsigma0025_prefix_' + str(prefix).zfill(4) + '_readonly.json')
        if not path.exists():
            values = [row['exact_gaussian_KL_old_to_new']['mean'] for row in health['snapshots']]
            write(path, dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
                committed_prefix=prefix, native_num_envs=8192, standing_environments=2048,
                update_KL_mean=sum(values)/len(values), update_KL_max=max(values),
                original_Adam_minibatches=prefix * 20, numerical_readers_unchanged=True,
                original_saved_wheel_action_limit_used=True, all9_native_alignment_errors_zero=True,
                all17_Adam_and_original_gradients_audited=True, all_original_prefix_reports_preserved=True,
                source_pins={str(p):sha(p) for p in (SCRIPT,
                    output / ('policy_update_health_prefix_' + str(prefix).zfill(4) + '_readonly.json'),
                    output / ('native_motion_training_saved_limits_prefix_' + str(prefix).zfill(4) + '_readonly.json'))},
                training_modified=False, optimizer_executed=False, policy_accepted=False))
        completed.append(prefix)
    return dict(observed_at=datetime.now().astimezone().isoformat(), output=str(output), unit=unit,
        state='all_owned_prefix_diagnostics_completed' if completed == [50,100,200] else 'observing',
        committed_native_updates=committed, completed_prefixes=completed, native_num_envs=8192,
        training_modified=False, optimizer_executed=False, can_extend_training=False,
        can_accept_policy=False, policy_accepted=False)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--unit', required=True)
    cli.add_argument('--once', action='store_true')
    args = cli.parse_args()
    output = args.output.resolve()
    status = output.parent / (output.name + '_AF7001_yawsigma0025_learning_status.json')
    while True:
        try:
            record = tick(output, args.unit)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            record = dict(observed_at=datetime.now().astimezone().isoformat(),
                state='waiting_for_owned_committed_data', detail=str(error), training_modified=False)
        except Exception as error:
            record = dict(observed_at=datetime.now().astimezone().isoformat(),
                state='agent_inspection_required', error=repr(error), training_modified=False,
                can_extend_training=False, policy_accepted=False)
        write(status, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.once or record.get('state') == 'all_owned_prefix_diagnostics_completed':
            break
        time.sleep(15.)
