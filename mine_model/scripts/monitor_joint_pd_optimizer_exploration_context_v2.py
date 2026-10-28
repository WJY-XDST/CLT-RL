"""Follow qualified PPO prefixes with additional read-only exploration audits.

The existing native/learning/critic/geometry monitors own their original
reports. This monitor consumes those committed reports and adds context. It
cannot extend training, change a configuration, or accept a checkpoint.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
RESULTS = ROOT / 'mine_model/results'
SCRIPTS = ROOT / 'mine_model/scripts'
STATUS = RESULTS / 'joint_pd_optimizer_exploration_context_monitor_status_v2_20261019.json'
PREFIXES = (40, 100, 200)
AUDITORS = (
    'audit_plain27_gradient_allocation.py',
    'audit_critic_fit_by_resampling_phase.py',
    'compare_plain27_recorded_joint_targets.py',
    'audit_matched_plain27_clipping_atoms.py',
    'audit_clipped_joint_pd_policy_kl.py',
)
VALIDATED_REPORT_DIGESTS = set()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_status(value):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(STATUS)


def tick():
    pointer = read(RESULTS / 'joint_pd_active_joint_tracking.json')
    output = Path(pointer['directory']).resolve()
    if output.parent != RESULTS:
        raise ValueError('Only read the owned authoritative active output')
    plan = read(output / 'experiment_plan.json')
    if (not plan['ordinary_full_policy_PPO'] or plan['bounded_training_updates'] != 200
            or plan['native_num_envs'] != 4096 or plan['auxiliary_losses_enabled']
            or plan['training_config']['leg_control_mode'] != 'implicit_joint_position'
            or plan['training_config']['observation_space'] != 27):
        raise ValueError('Requalify the original plain27/direct-PD/ordinary200 protocol')
    implementation = {str(p): sha(p) for p in [SCRIPT, *(SCRIPTS / s for s in AUDITORS)]}
    completed, waiting = [], []
    for prefix in PREFIXES:
        prerequisites = [output / f'{stem}_prefix_{prefix:04d}_readonly.json' for stem in
            ('learning_health_combined', 'policy_update_health',
             'critic_and_command_coverage', 'joint_target_geometric_exploration')]
        if not all(path.is_file() for path in prerequisites):
            waiting.append(prefix)
            continue
        combined, health, critic, geometry = [read(path) for path in prerequisites]
        if (combined['committed_prefix'] != prefix or health['committed_updates'] != prefix
                or critic['committed_updates'] != prefix or geometry['committed_updates'] != prefix
                or len(health['snapshots']) != prefix
                or any(s['source_mean_parity_max_error'] or s['source_std_parity_max_error']
                       for s in health['snapshots'])
                or combined['training_modified'] or geometry['training_modified']
                or not geometry['RNG_state_unchanged']):
            raise ValueError('Require complete qualified native prefixes before extra analysis')
        peak = max(health['snapshots'],
            key=lambda row: row['exact_gaussian_KL_old_to_new']['mean'])['committed_update']
        selected = sorted({1, peak, prefix})
        jobs = (
            (AUDITORS[0], 'gradient_allocation', ()),
            (AUDITORS[1], 'critic_fit_by_resampling_phase', ()),
            (AUDITORS[2], 'recorded_source_joint_targets', ()),
            (AUDITORS[3], 'matched_clipping_atoms', ()),
            (AUDITORS[4], 'clipped_action_policy_KL', ('--updates', *map(str, selected))),
        )
        for entry, stem, extra in jobs:
            report_path = output / f'{stem}_prefix_{prefix:04d}_readonly.json'
            if not report_path.exists():
                argv = [str(ROOT / 'run_python.sh'), str(SCRIPTS / entry), str(output),
                    '--through', str(prefix), *extra]
                environment = dict(os.environ,
                    PYTHONPATH=':'.join(str(ROOT / p) for p in
                        ('wheel_legged_isaaclab', 'mine_model/scripts', 'mine_model/experiments')),
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
                write_status(dict(observed_at=datetime.now().astimezone().isoformat(),
                    monitor_pid=os.getpid(), directory=str(output), unit=pointer['unit'],
                    state='reading_committed_prefix', prefix=prefix, auditor=entry,
                    completed_prefixes=completed, training_modified=False,
                    simulator_started=False, optimizer_executed=False, policy_accepted=False))
                with (output / f'exploration_context_{stem}_{prefix:04d}.log').open('a') as log:
                    subprocess.run(argv, cwd=ROOT, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, check=True)
            report = read(report_path)
            if (report.get('committed_prefix', report.get('committed_updates')) != prefix
                    or report['training_modified'] or report['optimizer_executed']
                    or report.get('simulator_started', report.get('simulation_started'))):
                raise ValueError('Resolve changed inputs or non-read-only evidence: ' + entry)
            # The published inputs are immutable. Verify their hashes once for
            # each report digest, rather than rereading hundreds of checkpoints
            # every15s while the simulator is using host memory and storage.
            identity = (str(report_path), sha(report_path))
            if identity not in VALIDATED_REPORT_DIGESTS:
                if any(sha(path) != digest for path, digest in report['source_pins'].items()):
                    raise ValueError('An immutable auditor input changed: ' + entry)
                VALIDATED_REPORT_DIGESTS.add(identity)
            if stem == 'gradient_allocation':
                if (report['actual_ppo_updates'] != plan['source_actual_ppo_updates'] + prefix
                        or report['all17_Adam_step'] != 20 * report['actual_ppo_updates']
                        or sum(v['original_minibatches'] for k, v in report['groups'].items()
                               if k == 'entire_prefix') != prefix * 20):
                    raise ValueError('Require the exact retained17-state minibatch clock')
            elif stem == 'critic_fit_by_resampling_phase':
                if report['reference_metrics_reproduced_max_error'] > 1.e-12:
                    raise ValueError('Original critic statistics differ')
            elif stem == 'matched_clipping_atoms':
                if not report['RNG_state_unchanged'] or report['RNG_draws_added']:
                    raise ValueError('Matched-state analysis changed RNG')
            elif stem == 'clipped_action_policy_KL':
                if (report['selected_updates'] != selected or report['RNG_draws_added']
                        or report['independent_numeric_quadrature_max_error'] > 1.e-9
                        or not all(row['source_native_mean_and_std_parity_exact']
                                   for row in report['updates'])):
                    raise ValueError('Requalify selected clipped-KL updates and native parity')
        completed.append(prefix)
    if any(sha(path) != digest for path, digest in implementation.items()):
        raise ValueError('A read-only auditor changed during the observation tick')
    status = dict(observed_at=datetime.now().astimezone().isoformat(),
        monitor_pid=os.getpid(), directory=str(output), unit=pointer['unit'], state='observing',
        completed_prefixes=completed, waiting_for_original_prefixes=waiting,
        auditor_sha256=implementation, input_pins_verified_once_per_report_digest=True,
        training_modified=False, simulator_started=False,
        optimizer_executed=False, acceptance_changed=False, policy_accepted=False)
    write_status(status)
    return status


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--once', action='store_true')
    args = cli.parse_args()
    with (RESULTS / '.optimizer_exploration_context_monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = tick()
                if args.once:
                    print(json.dumps(result))
                    break
            except (FileNotFoundError, json.JSONDecodeError) as error:
                if args.once:
                    raise
                write_status(dict(observed_at=datetime.now().astimezone().isoformat(),
                    monitor_pid=os.getpid(), state='waiting_for_committed_files',
                    detail=str(error), training_modified=False, policy_accepted=False))
            except BaseException as error:
                write_status(dict(observed_at=datetime.now().astimezone().isoformat(),
                    monitor_pid=os.getpid(), state='audit_discrepancy_requires_agent_review',
                    detail=repr(error), training_modified=False, policy_accepted=False))
                raise
            time.sleep(15.)
