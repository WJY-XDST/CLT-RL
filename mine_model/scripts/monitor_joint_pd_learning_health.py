"""Follow reviewed PPO blocks and audit exploration/updates at40/100/200.

Read-only: no training restarts, optimizer steps, added loss, or gate changes.
Each report uses committed saved native data and preserves previous reports.
"""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'mine_model/results'
SCRIPTS = ROOT / 'mine_model/scripts'
CHECKS = (
    ('audit_effective_joint_pd_exploration.py', 'effective_exploration', []),
    ('audit_native_motion_training_prefix.py', 'native_motion_training', []),
    ('audit_joint_pd_policy_updates.py', 'policy_update_health', ['--interval', '1', '--device', 'cuda:0']),
)


def tick():
    pointer = json.loads((RESULTS / 'joint_pd_active_joint_tracking.json').read_text())
    output = Path(pointer['directory']).resolve()
    if output.parent != RESULTS:
        raise ValueError('Unexpected active output')
    plan = json.loads((output / 'experiment_plan.json').read_text())
    state = json.loads((output / 'status.json').read_text())
    if plan['bounded_training_updates'] != 200 or not plan['ordinary_full_policy_PPO']:
        raise ValueError('Only reviewed ordinary200-update PPO blocks')
    committed = len(list((output / 'native_learning_signal').glob('audit_update_*.json')))
    finished = (output / 'native_learning_signal/completed_external.json').is_file()
    for prefix in (40, 100, 200):
        # An additional committed update proves the earlier checkpoint save
        # finished; the last prefix instead requires external completion.
        if not (committed > prefix or finished and committed == prefix):
            continue
        for script, stem, extra in CHECKS:
            destination = output / f'{stem}_prefix_{prefix:04d}_readonly.json'
            if destination.exists():
                if json.loads(destination.read_text())['committed_updates'] != prefix:
                    raise ValueError('Existing diagnostic has another committed prefix')
                continue
            argv = [sys.executable, str(SCRIPTS / script), str(output), '--through', str(prefix), *extra]
            log = output / f'learning_health_{stem}_{prefix:04d}.log'
            with log.open('a') as stream:
                subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        exploration = json.loads((output / f'effective_exploration_prefix_{prefix:04d}_readonly.json').read_text())
        health = json.loads((output / f'policy_update_health_prefix_{prefix:04d}_readonly.json').read_text())
        values = [row['exact_gaussian_KL_old_to_new']['mean'] for row in health['snapshots']]
        record = dict(observed_at=datetime.now().astimezone().isoformat(), output=str(output),
            committed_prefix=prefix, update_KL_mean=sum(values)/len(values),
            update_KL_max=max(values), KL_sampled_update_count=len(values),
            KL_sampled_updates=[row['committed_update'] for row in health['snapshots']],
            sampled_transition_count_per_update=48*32,
            exploration_latest_window=exploration['groups'].get('last20', exploration['groups']['entire_prefix']),
            all17_Adam_and_positive_original_gradients_audited=True,
            physical_training_report=str(output / f'native_motion_training_prefix_{prefix:04d}_readonly.json'),
            added_RNG_draws=0, training_modified=False, acceptance_changed=False,
            local_optimum_proven=False, policy_accepted=False)
        target = output / f'learning_health_combined_prefix_{prefix:04d}_readonly.json'
        if not target.exists():
            with target.open('x') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            print(json.dumps(dict(output=str(output), committed_prefix=prefix,
                update_KL_mean=record['update_KL_mean'], read_only=True)), flush=True)


if __name__ == '__main__':
    while True:
        try:
            tick()
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        except BaseException as error:
            path = RESULTS / 'joint_pd_learning_health_monitor_error.json'
            path.write_text(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),
                error=repr(error), training_modified=False, requires_agent_inspection=True)) + '\n')
            raise
        time.sleep(15.)
