"""Follow completed learning-health prefixes; never control native training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'mine_model/results'
AUDITOR = ROOT / 'mine_model/scripts/audit_joint_pd_critic_and_command_coverage.py'


def tick():
    pointer = json.loads((RESULTS / 'joint_pd_active_joint_tracking.json').read_text())
    output = Path(pointer['directory']).resolve()
    if output.parent != RESULTS:
        raise ValueError('Unexpected active output')
    plan = json.loads((output / 'experiment_plan.json').read_text())
    if plan['bounded_training_updates'] != 200 or not plan['ordinary_full_policy_PPO']:
        raise ValueError('Only reviewed ordinary200-update PPO blocks')
    for prefix in (40, 100, 200):
        health = output / f'learning_health_combined_prefix_{prefix:04d}_readonly.json'
        if not health.is_file():
            continue
        if json.loads(health.read_text())['committed_prefix'] != prefix:
            raise ValueError('Wrong committed learning-health prefix')
        target = output / f'critic_and_command_coverage_prefix_{prefix:04d}_readonly.json'
        if target.is_file():
            if json.loads(target.read_text())['committed_updates'] != prefix:
                raise ValueError('Existing coverage diagnostic has another prefix')
            continue
        log = output / f'learning_health_critic_and_coverage_{prefix:04d}.log'
        with log.open('a') as stream:
            subprocess.run([sys.executable, str(AUDITOR), str(output), '--through', str(prefix)],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads(target.read_text())
        print(json.dumps(dict(output=str(output), committed_prefix=prefix,
            sampled_transitions=report['sampled_transitions'],
            bootstrap_error=report['independent_timeout_bootstrap_max_error'],
            training_modified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            tick()
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        except BaseException as error:
            path = RESULTS / 'joint_pd_critic_coverage_monitor_error.json'
            path.write_text(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),
                error=repr(error), training_modified=False, requires_agent_inspection=True)) + '\n')
            raise
        if args.once:
            break
        time.sleep(15.)
