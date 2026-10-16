"""Review each 100-update direct-joint PPO block against measured stability.

Start with the diagnosed zero-command drift cost comparison. Later changes
require a corresponding failure in a completed, checkpoint-matched replay.
All native checkpoints, full Adam states and original tolerances are retained.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time

from auto_train_mine import write_json
import continue_joint_pd_source200_height100 as handoff
from joint_pd_priority import STATIC_CASES, trials

guard = handoff.guard
ROOT = guard.ROOT
UPDATES = 100
TEST = ROOT/'wheel_legged_isaaclab/tests/test_stationary_drift100.py'


def deficits(report):
    """Use original per-trial limits, including failures without steady data."""
    result = dict(survival=[], speed=[], yaw=[], body=[], height=[])
    for (name, seed), (case, row) in trials(report).items():
        if not row['survived'] or row['resets']:
            result['survival'].append(dict(case=name, seed=seed, resets=row['resets']))
        if not row['survived']:
            continue
        for group, metric in [('body', 'body_p95_deg'), ('height', 'height_mae_m'),
                              ('speed', 'speed_mae_m_s'), ('yaw', 'yaw_mae_rad_s')]:
            if group in ('speed', 'yaw') and name not in STATIC_CASES:
                continue
            value, limit = row['metrics'][metric], row['limits'][metric]
            if value > limit:
                result[group].append(dict(case=name, seed=seed, metric=metric,
                                          value=value, original_limit=limit))
    return result


def choose(cfg, report, used, *, initial_reports=None, continue_improvement=False):
    """One evidenced coefficient, without repeating a failed source/seed pair."""
    rewards = cfg['rewards']
    failures = deficits(report)
    if initial_reports is not None:
        if (rewards['base_height_error_sq'] != -100. or rewards['standing_velocity'] != 0.
                or rewards['orientation'] != -20. or len(initial_reports) != 2
                or not all(deficits(value)['speed'] for value in initial_reports)):
            raise ValueError('Require the stable baseline and both observed drift comparisons')
        choices = [({'standing_velocity': -25.}, 'Repeated stationary drift at height cost -100 and -50')]
    elif continue_improvement:
        choices = [({}, 'Continue the measured stability-preserving improvement for exactly 100 updates')]
    else:
        choices = []
        # These are existing reward terms. Never add target-FK or auxiliary losses.
        for group, key, levels in [
            ('speed', 'standing_velocity', (25., 50., 100., 200.)),
            ('yaw', 'yaw_rate_error_sq', (2., 4., 8., 16.)),
            ('body', 'orientation', (40., 80., 160., 320.)),
        ]:
            if failures[group] or (group == 'body' and failures['survival']):
                choices.extend(({key: -value}, 'Completed replay failed '+group+' stability')
                               for value in levels if -value < rewards[key])
        # Height may only be strengthened after the trial preserves all stability.
        if not any(failures[k] for k in ('survival', 'speed', 'yaw', 'body')) and failures['height']:
            choices.extend(({'base_height_error_sq': -value},
                            'Completed replay retained stability but missed height')
                           for value in (150., 200., 300., 500.)
                           if -value < rewards['base_height_error_sq'])
    for delta, reason in choices:
        signature = (43, tuple(sorted(delta.items())))
        if signature not in used:
            return 43, delta, signature, dict(reason=reason, failures=failures,
                only_reward_change=delta, actual_PPO_updates=UPDATES,
                hypothesis_only=True, policy_accepted=False)
    raise RuntimeError('Measured bounded comparisons exhausted; inspect reward/control diagnostics before further training')


def memory_ready(values):
    # The failed native initializer used about 19 GiB of resident host memory.
    # Keep headroom without changing the 8192-env PPO batch or killing desktop apps.
    gib = 1024**3
    return values['MemAvailable'] >= 6*gib and values['MemAvailable']+values['SwapFree'] >= 26*gib


def wait_memory(base):
    while True:
        if (base/'STOP').exists():
            raise RuntimeError('Cooperative stop while waiting for native memory')
        values = {line.split(':')[0]: int(line.split()[1])*1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        ready = memory_ready(values)
        write_json(base/'native_memory_preflight.json', dict(
            updated_at=datetime.now().astimezone().isoformat(), ready=ready,
            available_bytes=values['MemAvailable'], swap_free_bytes=values['SwapFree'],
            allocator_arena_max=os.environ.get('MALLOC_ARENA_MAX'), num_envs=8192,
            state='native_memory_ready' if ready else 'waiting_for_host_memory'))
        if ready:
            return
        time.sleep(10)


def completed_report(directory):
    directory = Path(directory)
    state = json.loads((directory/'status.json').read_text())
    report = json.loads((directory/'replay/assessment.json').read_text())
    if (state['training_active'] or state['state'] != 'agent_review_required'
            or guard.core.sha(Path(state['checkpoint'])) != state['checkpoint_sha256']
            or report['checkpoint_sha256'] != state['checkpoint_sha256']):
        raise ValueError('Require the full replay of the exact saved checkpoint')
    trials(report)
    return report


def run(previous_cursor, candidate, base, unit):
    previous_cursor, candidate, base = map(lambda p: Path(p).resolve(), (previous_cursor, candidate, base))
    if base.parent != ROOT/'mine_model/results':
        raise ValueError('Use a fresh owned coordinator in the actual project')
    base.mkdir(exist_ok=True)
    with (previous_cursor.parent/'.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cursor = json.loads(previous_cursor.read_text())
        previous = handoff.selected_incumbent(cursor)
        pending = json.loads((Path(cursor['current_output'])/'status.json').read_text())
        if pending['training_active']:
            raise ValueError('Diagnose/reconcile the exited native owner before handoff')
        history = cursor['history'][-2:]
        if len(history) != 2 or any(row['promote'] for row in history):
            raise ValueError('Pin both completed rejected height-cost comparisons')
        initial = [completed_report(row['output']) for row in history]
        initial_paths = [Path(row['output'])/'replay/assessment.json' for row in history]
        baseline = guard.core.validate_checkpoint_robot(Path(json.loads((previous/'status.json').read_text())['checkpoint']))
        choose(baseline, completed_report(previous), set(), initial_reports=initial)
        decision_path = None
        original_prepare, original_updates = guard.prepare, guard.UPDATES

        def bootstrap(*unused):
            return handoff.adopt(previous, candidate, base, previous_cursor)

        def propose(cfg, report, used):
            nonlocal decision_path
            current = json.loads((base/'coordinator.json').read_text())
            reviews = current.get('history', [])
            paths = list(initial_paths)
            if not reviews:
                evidence, first, improved = report, initial, False
            else:
                latest = reviews[-1]
                evidence = completed_report(latest['output'])
                paths.append(Path(latest['output'])/'replay/assessment.json')
                first, improved = None, latest['promote']
            seed, delta, signature, diagnosis = choose(cfg, evidence, used,
                initial_reports=first, continue_improvement=improved)
            decision_path = base/f'reward_decision_{len(reviews):04d}.json'
            write_json(decision_path, dict(diagnosis, evidence_files=list(map(str, paths)),
                evidence_sha256={str(p): guard.core.sha(p) for p in paths}, seed=seed))
            return seed, delta, signature

        def prepare(previous_output, output, delta, seed):
            wait_memory(base)
            plan, expected = original_prepare(previous_output, output, delta, seed)
            decision = json.loads(decision_path.read_text())
            for path in (Path(__file__).resolve(), TEST, decision_path, *map(Path, decision['evidence_files'])):
                plan['source_files'][str(path)] = guard.core.sha(path)
            plan.update(diagnosed_reward_decision=decision,
                original_acceptance_unchanged=True, native_num_envs=8192,
                host_memory_preflight=str(base/'native_memory_preflight.json'),
                allocator_arena_max=os.environ.get('MALLOC_ARENA_MAX'))
            write_json(Path(output)/'experiment_plan.json', plan)
            return plan, expected

        try:
            guard.UPDATES, guard.prepare = UPDATES, prepare
            guard.run(candidate, previous, base, unit,
                      bootstrap_function=bootstrap, proposal_function=propose)
        finally:
            guard.UPDATES, guard.prepare = original_updates, original_prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--coordinator', required=True, type=Path)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    run(args.previous_coordinator, args.candidate, args.coordinator, args.unit)
