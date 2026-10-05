"""Apply a requested environment count after a completed optimization round.

The running supervisor owns status.json in memory, so stop it at the next
round boundary before editing its saved state and restarting it.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def at_boundary(state, after_round):
    return (int(state.get('round', -1)) > after_round
            and state.get('state') in ('planning_next_round', 'training_optimization'))


def systemctl(*arguments):
    return subprocess.run(['systemctl', '--user', *arguments], check=True,
                          text=True, capture_output=True).stdout.strip()


def apply_transition(status_path, request_path, service, after_round, count):
    # Freeze both supervisor and children to close the read/stop race. A newly
    # launched next-round child may be stopped; the completed round is preserved.
    systemctl('freeze', service)
    try:
        state = json.loads(status_path.read_text())
        if not at_boundary(state, after_round):
            return False
        checkpoint = Path(state['checkpoint'])
        if not checkpoint.is_file():
            raise RuntimeError(f'Missing continuation checkpoint: {checkpoint}')
        backup = status_path.parent / f'before_env{count}_after_round{after_round}.json'
        if not backup.exists():
            write_json(backup, state)
        # Stop before writing: SIGINT cleanup can otherwise overwrite our edit.
        systemctl('stop', service)
        previous = state['num_envs']
        suffix = state.get('run_name_suffix', '')
        suffix = re.sub(r'_env\d+$', '', suffix) + f'_env{count}'
        timestamp = datetime.now().astimezone().isoformat()
        state.update(num_envs=count, requested_num_envs=count,
                     run_name_suffix=suffix, state='env_scale_transition',
                     child_pid=None, updated_at=timestamp)
        state.pop('error', None)
        state.pop('child_command', None)
        state.setdefault('interventions', []).append({
            'time': timestamp, 'kind': 'user_requested_environment_count',
            'previous_num_envs': previous, 'num_envs': count,
            'after_round': after_round, 'effective_round': state['round'],
            'checkpoint': str(checkpoint), 'previous_state': str(backup),
            'capacity_testing_performed': False,
        })
        write_json(status_path, state)
        systemctl('start', service)
        write_json(request_path, {
            'state': 'applied', 'num_envs': count, 'after_round': after_round,
            'effective_round': state['round'], 'checkpoint': str(checkpoint),
            'applied_at': timestamp,
        })
        print(f'Applied {count} environments from round {state["round"]}', flush=True)
        return True
    finally:
        # A failed boundary check must not leave ongoing training frozen.
        if systemctl('show', service, '--property=ActiveState', '--value') == 'active':
            systemctl('thaw', service)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--service', required=True)
    parser.add_argument('--after-round', type=int, required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error('--num-envs must be positive')
    request = args.status.parent / f'pending_env{args.num_envs}.json'
    if request.exists() and json.loads(request.read_text()).get('state') == 'applied':
        return
    write_json(request, {'state': 'waiting_for_round_completion',
                        'num_envs': args.num_envs, 'after_round': args.after_round,
                        'requested_at': datetime.now().astimezone().isoformat()})
    while True:
        state = json.loads(args.status.read_text())
        if state.get('state') == 'accepted':
            write_json(request, {'state': 'training_already_accepted',
                                'num_envs': args.num_envs, 'after_round': args.after_round})
            return
        # Respect a user pause or stop; only transition a running service.
        properties = systemctl('show', args.service, '-p', 'ActiveState', '-p', 'FreezerState')
        if ('ActiveState=active' in properties and 'FreezerState=running' in properties
                and at_boundary(state, args.after_round)):
            if apply_transition(args.status, request, args.service,
                                args.after_round, args.num_envs):
                return
        time.sleep(1)


if __name__ == '__main__':
    main()
