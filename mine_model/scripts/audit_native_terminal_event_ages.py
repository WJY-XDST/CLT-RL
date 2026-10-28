"""Read-only physical-age audits of unsafe events at committed PPO prefixes.

Reuse the original RSL parser, expose its captured rows/text, and reconcile
all4096 unsafe counts with the pre-reset native event trace. Never control PPO.
"""
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import monitor_joint_pd_event_history as original

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
RESULTS = ROOT / 'mine_model/results'
PREFIXES = (40, 100, 200)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    tree = ast.parse(inspect.getsource(original.summarize))
    before = ast.dump(tree)
    statement = tree.body[0].body[-1]
    if not isinstance(statement, ast.Return) or not isinstance(statement.value, ast.Call):
        raise ValueError('Requalify the exact original parser return')
    extra = [ast.keyword(arg='all_complete_rows', value=ast.Name(id='rows', ctx=ast.Load())),
             ast.keyword(arg='parsed_text_snapshot', value=ast.Name(id='text', ctx=ast.Load()))]
    statement.value.keywords.extend(extra)
    changed = copy.deepcopy(tree)
    del statement.value.keywords[-2:]
    if ast.dump(tree) != before:
        raise ValueError('Original parsing or event-count predicates changed')
    scope = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(changed), original.__file__, 'exec'), scope)
    return scope['summarize']


def audit(directory, through, destination=None):
    directory = Path(directory).resolve()
    if directory.parent != RESULTS or through < 1 or through > 200:
        raise ValueError('Use an owned committed prefix of one200-update trial')
    destination = (Path(destination).resolve() if destination is not None else
        directory / f'terminal_event_ages_prefix_{through:04d}_readonly.json')
    if destination.exists() or not destination.is_relative_to(RESULTS):
        raise ValueError('Preserve previous immutable evidence')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    if (plan['native_num_envs'] != 4096 or plan['bounded_training_updates'] != 200
            or plan['native_transition_batch'] != 4096 * 48
            or plan['training_config']['leg_control_mode'] != 'implicit_joint_position'):
        raise ValueError('Requalify native4096/48 direct-joint event accounting')
    captured = parser()(directory)
    rows = captured['all_complete_rows'][:through]
    first = plan['source_actual_ppo_updates']
    if len(rows) != through:
        raise RuntimeError('The requested printed prefix is not committed yet')
    if [row['log_iteration'] for row in rows] != list(range(first, first + through)):
        raise ValueError('Printed update clocks are not contiguous from the retained source')
    text = captured['parsed_text_snapshot']
    clocks = re.findall(r'Initial command curriculum step:\s*(\d+)', text)
    if len(clocks) != 1:
        raise ValueError('Require one explicit initial native control clock')
    initial, dt = int(clocks[0]), .02
    if initial != first * 48:
        raise ValueError('Requalify the initial native clock before inferring physical ages')
    final = initial + through * 48
    headers = list(re.finditer(r'Learning iteration\s+(\d+)/(\d+)', text))
    normalized_prefix = text[:headers[through].start()] if len(headers) > through else text
    event_path = directory / 'native_learning_signal/terminal_events_before_reset.jsonl'
    raw = event_path.read_bytes() if event_path.exists() else b''
    prefix, events, seen = bytearray(), [], set()
    previous_clock = initial
    fixed = plan['fixed_standing_environments']
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break  # A concurrently written trailing event is not committed data.
        event = json.loads(line)
        clock, env_id = event['control_step'], event['environment_id']
        if clock < previous_clock or clock <= initial or not 0 <= env_id < 4096:
            raise ValueError('Native events are not ordered in this fresh trial')
        previous_clock = clock
        if clock > final:
            break
        key = (clock, env_id)
        if key in seen:
            raise ValueError('Duplicate native unsafe event')
        seen.add(key)
        prefix.extend(line)
        state = event['state']
        elapsed = (clock - initial) * dt
        age = min(elapsed, float(state['episode_time_s']))
        if not math.isfinite(age) or age < 0.:
            raise ValueError('Invalid physical episode-age bound')
        events.append(dict(update=(clock-initial-1)//48+1, control_step=clock, environment_id=env_id,
            physical_age_upper_bound_s=age, initial_native_elapsed_s=elapsed,
            logged_episode_time_s=state['episode_time_s'], command_age_s=state['command_age_s'],
            command=state['command'], reasons=event['reasons'],
            dedicated_static=env_id < fixed,
            exact_static=state['command'][0] == state['command'][1] == 0.,
            state_before_reset=state))
    expected = sum(row['unsafe'] for row in rows)
    if len(events) != expected:
        raise ValueError(f'Native/RSL unsafe mismatch at prefix{through}: {len(events)} versus {expected}')
    records = []
    for index in range(1, through + 1):
        path = directory / f'native_learning_signal/audit_update_{index:04d}.json'
        record = json.loads(path.read_text())
        if (record['update'] != index or record['total_environments'] != 4096
                or len(record['all_environment_alignment_max_errors']) != 9
                or any(record['all_environment_alignment_max_errors'].values())
                or not record['probe_RNG_unchanged'] or record['added_loss_or_backward']):
            raise ValueError('Resolve native data alignment before interpreting events')
        records.append(path)
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        committed_prefix=through, initial_logged_control_clock=initial, final_included_control_clock=final,
        all4096_ended_episodes=sum(row['ended'] for row in rows), all4096_unsafe_events=expected,
        all4096_numerical_events=sum(row['numerical'] for row in rows),
        raw_and_RSL_unsafe_counts_identical=True,
        physical_age_upper_bound_le1s=sum(row['physical_age_upper_bound_s'] <= 1. for row in events),
        physical_age_upper_bound_le2s=sum(row['physical_age_upper_bound_s'] <= 2. for row in events),
        dedicated_static_events=sum(row['dedicated_static'] for row in events),
        exact_static_events=sum(row['exact_static'] for row in events),
        reason_counts_can_overlap=dict(Counter(reason for row in events for reason in row['reasons'])),
        events=events, complete_printed_rows=rows,
        raw_trace=str(event_path), captured_raw_prefix_bytes=len(prefix),
        captured_raw_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        normalized_log_prefix_characters=len(normalized_prefix),
        normalized_log_prefix_sha256=hashlib.sha256(normalized_prefix.encode()).hexdigest(),
        live_trace_and_log_not_pinned_as_completed_files=True,
        original_RSL_parser_AST_restored_identical=True, only_parser_row_and_text_exposure_added=True,
        all_prefix_native4096_alignment_errors_zero=True, native_RNG_states_unchanged=True,
        source_pins={str(p): sha(p) for p in [SCRIPT, Path(original.__file__), plan_path, *records]},
        training_modified=False, simulator_started=False, optimizer_executed=False,
        acceptance_changed=False, policy_accepted=False,
        limitations=['Random initial episode counters are not physical ages. The reported age is an upper bound; above2s is not proof of a lower bound.',
            'All native unsafe events are retained, including dedicated static failures. Event reasons may overlap.',
            'Prefixes capture evolving-policy Gaussian training. These are not independent current-policy failure probabilities or acceptance.',
            'The original parser counts printed RSL rows. Native event IDs, commands and state are reconciled separately.',
            'Only the captured trace/log prefixes are hashed while training runs; appending later data is expected.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, result


def monitor():
    status = RESULTS / 'joint_pd_terminal_age_monitor_status_20261019.json'
    last = None
    while True:
        row = dict(observed_at=datetime.now().astimezone().isoformat(),
            monitor_pid=os.getpid(), auditor_sha256=sha(SCRIPT),
            training_modified=False, policy_accepted=False)
        try:
            pointer = json.loads((RESULTS / 'joint_pd_active_joint_tracking.json').read_text())
            directory = Path(pointer['directory']).resolve()
            if directory.parent != RESULTS:
                raise ValueError('Only observe an owned training output')
            owner = subprocess.run(['systemctl', '--user', 'show', pointer['unit'], '-p', 'MainPID',
                '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result'],
                capture_output=True, text=True, check=True).stdout.strip()
            available = len(list((directory / 'native_learning_signal').glob('audit_update_*.json')))
            row.update(directory=str(directory), unit=pointer['unit'], owner=owner,
                available_native_records=available, state='observing', completed_prefixes=[])
            for count in PREFIXES:
                target = directory / f'terminal_event_ages_prefix_{count:04d}_readonly.json'
                if available >= count and not target.exists():
                    audit(directory, count)
                if target.exists():
                    row['completed_prefixes'].append(count)
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError) as error:
            row.update(state='waiting_for_committed_data', detail=str(error))
        except Exception as error:
            row.update(state='audit_discrepancy_requires_agent_review', error=repr(error))
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(row, indent=2)+'\n')
        temporary.replace(status)
        key = (row.get('directory'), row['state'], tuple(row.get('completed_prefixes', [])), row.get('error'))
        if key != last:
            print(json.dumps(row), flush=True)
            last = key
        time.sleep(10.)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path, nargs='?')
    cli.add_argument('--through', type=int)
    cli.add_argument('--destination', type=Path)
    cli.add_argument('--monitor', action='store_true')
    args = cli.parse_args()
    if args.monitor:
        if args.directory is not None or args.through is not None or args.destination is not None:
            cli.error('The read-only monitor follows the authoritative active pointer')
        monitor()
    else:
        if args.directory is None or args.through is None:
            cli.error('Require one owned directory and committed prefix')
        path, report = audit(args.directory, args.through, args.destination)
        print(json.dumps(dict(report=str(path), committed_prefix=report['committed_prefix'],
            unsafe=report['all4096_unsafe_events'], dedicated_static=report['dedicated_static_events'],
            numerical=report['all4096_numerical_events'], training_modified=False)))
