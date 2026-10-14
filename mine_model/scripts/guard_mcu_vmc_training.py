"""Pause a scoped PPO service on physical/numerical regressions, not low reward.

This supervisor never changes gains/rewards, deletes models, starts another PPO
run or shuts down the host. Training can resume only after numerical screening.
Checkpoint replays preserve standing AND motion checks at curriculum milestones.
"""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
UNIT = 'clt-vmc8192-stand25-20261010.service'
RUN = ROOT / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-10_23-23-52_mcu_tgs_vmc_env8192_stand25_to1999'
LOG = ROOT / 'mine_model/results/mcu_standing_restart_20261010/motion_resume_8192_stand25/training.log'
OUTPUT = ROOT / 'mine_model/results/mcu_standing_restart_20261010/motion_resume_8192_stand25/guard'
SCREEN = ROOT / 'mine_model/results/tgs_numerical_convergence_20261011/assessment.json'
OVERRIDES = ROOT / 'mine_model/results/mcu_standing_restart_20261010/protected_stops_replay_overrides.json'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
ITERATION = re.compile(r'Learning iteration\s+(\d+)/(\d+)')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf)'
FATAL = re.compile(r'(CUDA error:|CUDA out of memory|Traceback \(most recent call last\)|'
                   r'\[Error\]\s+\[omni\.physx(?:\.tensors)?\.plugin\]|illegal memory access|'
                   r'The following assertion condition failed|Fatal Python error:|Recursion not allowed)', re.I)


def completed_updates(content):
    content = ANSI.sub('', content)
    matches = list(ITERATION.finditer(content))
    updates = []
    for index, match in enumerate(matches):
        block = content[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(content)]
        if 'ETA:' not in block:
            continue
        metrics = {}
        for key, value in re.findall(r'^\s*([^:\n]+):\s*(' + NUMBER + r')\s*$', block, re.M | re.I):
            metrics[key.strip()] = float(value)
        updates.append(dict(iteration=int(match[1]), target=int(match[2]), metrics=metrics))
    return updates


def anomaly(updates, content):
    if FATAL.search(content):
        return 'native_or_cuda_error'
    if not updates:
        return None
    recent = updates[-3:]
    if any(not math.isfinite(v) for update in recent for v in update['metrics'].values()):
        return 'nonfinite_training_metric'
    if any(update['metrics'].get('Episode_Termination/numerical', 0.) > 0.
           or update['metrics'].get('Episode_Count/numerical', 0.) > 0. for update in recent):
        return 'numerical_termination'
    if len(recent) < 3:
        return None
    if all(update['metrics'].get('Episode_Termination/standing_unsafe', 0.) >= .15 for update in recent):
        return 'standing_survival_regression_three_updates'
    if all(update['metrics'].get('Mean episode length', math.inf) < 250 and
           update['metrics'].get('Episode_Termination/unsafe_contact_or_orientation', 0.) >= .25
           for update in recent):
        return 'support_collapse_three_updates'
    for role in ('forward', 'reverse'):
        if all(update['metrics'].get(f'Tracking/{role}_fraction', 0.) >= .05
               and abs(update['metrics'].get(f'Tracking/{role}_command_m_s', 0.)) >= .10
               and update['metrics'].get(f'Tracking/{role}_opposite_direction_fraction', 0.) >= .60
               for update in recent):
            return f'{role}_opposite_motion_three_updates'
    if all(update['metrics'].get('Tracking/wheel_reference_second_difference_rms_rad_s', 0.) >= 10.
           and update['metrics'].get('Tracking/wheel_reference_delta_rms_rad_s', 0.) >= 6.
           for update in recent):
        return 'wheel_reference_chatter_three_updates'
    # Poor motion tracking at the start of a curriculum is not itself a physics
    # failure. It remains visible in checkpoint evaluation and needs diagnosis.
    return None


def replay_gate(assessment):
    stand = next(c for c in assessment['conditions'] if c['name'] == 'stand')
    if any(not t['survived'] for t in stand['trials']):
        return 'frozen_policy_cannot_stand'
    for trial in stand['trials']:
        m = trial['metrics']
        if not m['upright_support'] or m['body_p95_deg'] > 5.:
            return 'standing_posture_or_height_regression'
        if m['speed_mae_m_s'] > .05 or m['yaw_mae_rad_s'] > .10:
            return 'standing_drift_regression'
    if assessment['survivors'] < .75 * assessment['total_trials']:
        return 'replay_support_failures_over_25_percent'
    return None


def unit_state():
    output = subprocess.check_output(
        ['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'FreezerState', '-p', 'MainPID'], text=True)
    return dict(line.split('=', 1) for line in output.strip().splitlines())


def save(value):
    value = dict(value, updated_at=datetime.now().astimezone().isoformat(), unit=UNIT)
    path = OUTPUT / 'status.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def freeze(reason, updates):
    state = unit_state()
    if state['ActiveState'] == 'active' and state.get('FreezerState') != 'frozen':
        subprocess.run(['systemctl', '--user', 'freeze', UNIT], check=True)
    save(dict(state='paused_for_diagnosis', reason=reason,
              latest=updates[-1] if updates else None, progress_and_checkpoints_preserved=True))


def evaluate(iteration):
    checkpoint = RUN / f'model_{iteration}.pt'
    directory = OUTPUT / f'checkpoint_{iteration}'
    directory.mkdir(exist_ok=True)
    # At 1000 the movement curriculum is just starting. At 1500 its amplitude
    # is halfway to the final range; also test nonzero commands at that scale.
    scale = .25 if iteration <= 1000 else .5
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab'),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    cmd = [str(ROOT / 'run_python.sh'), str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
           '--checkpoint', str(checkpoint), '--overrides_json', str(OVERRIDES),
           '--output', str(directory), '--seconds', '20', '--seeds', '83', '84', '85',
           '--command-scale', str(scale), '--record_policy_inputs', '--headless', '--device', 'cuda:0']
    with (directory / 'evaluation.log').open('x') as stream:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=True)
    from summarize_mine_trial import assess
    assessment = assess(directory)
    return assessment, replay_gate(assessment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=15.)
    parser.add_argument('--once', action='store_true', help='Read-only status check; never pauses or resumes')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1.:
        parser.error('interval must be finite and at least one second')
    updates = completed_updates(LOG.read_text(errors='replace'))
    if args.once:
        print(json.dumps(dict(latest=updates[-1] if updates else None,
                              anomaly=anomaly(updates, LOG.read_text(errors='replace'))), indent=2))
        return
    if not SCREEN.exists() or not json.loads(SCREEN.read_text()).get('numerical_screen_passed'):
        parser.error('Numerical screening has not passed; do not keep training through this condition')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.guard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_iteration = None
        last_progress = time.monotonic()
        while True:
            if (OUTPUT / 'STOP_GUARD').exists():
                save(dict(state='cancelled_no_resume'))
                return
            content = LOG.read_text(errors='replace')
            updates = completed_updates(content)
            latest = updates[-1]['iteration'] if updates else None
            if latest != previous_iteration:
                last_progress = time.monotonic()
                previous_iteration = latest
            reason = anomaly(updates, content)
            if reason:
                freeze(reason, updates)
                return
            state = unit_state()
            if state['ActiveState'] != 'active':
                save(dict(state='training_service_finished', service=state,
                          latest=updates[-1] if updates else None,
                          final_evaluation=str(LOG.parent / 'final_evaluation/assessment.json')))
                return
            if state.get('FreezerState') == 'frozen':
                save(dict(state='externally_paused_no_resume', latest=updates[-1] if updates else None))
                return
            for milestone in (1000, 1500):
                result = OUTPUT / f'checkpoint_{milestone}/assessment.json'
                if latest is None or latest < milestone or result.exists() or not (RUN / f'model_{milestone}.pt').exists():
                    continue
                save(dict(state='checkpoint_evaluation', iteration=milestone))
                subprocess.run(['systemctl', '--user', 'freeze', UNIT], check=True)
                main_pid = state['MainPID']
                try:
                    assessment, reason = evaluate(milestone)
                except Exception as error:
                    freeze('checkpoint_evaluation_failed: ' + str(error), updates)
                    return
                if reason:
                    freeze(reason, updates)
                    return
                state_after = unit_state()
                if ((OUTPUT / 'STOP_GUARD').exists() or state_after['ActiveState'] != 'active'
                        or state_after['MainPID'] != main_pid):
                    save(dict(state='cancelled_or_service_replaced_no_resume'))
                    return
                subprocess.run(['systemctl', '--user', 'thaw', UNIT], check=True)
                last_progress = time.monotonic()
                save(dict(state='monitoring', last_replay_iteration=milestone,
                          replay_survivors=assessment['survivors'], replay_tracking_passed=assessment['passed'],
                          command_tracking_must_not_be_inferred_from_survival=True))
            save(dict(state='monitoring' if time.monotonic() - last_progress < 600 else 'stalled_needs_inspection',
                      latest=updates[-1] if updates else None,
                      seconds_without_progress=time.monotonic() - last_progress,
                      milestones=[1000, 1500], final_checkpoint=1998))
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
