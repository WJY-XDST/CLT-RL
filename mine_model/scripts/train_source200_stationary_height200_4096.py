"""Same reviewed200 source200 branch with4096 envs for measured host memory.

The completed frozen27-case transfer is reused without repeating PPO. Only
environment parallelism changes; rollout length48 and20 Adam steps/update,
direct control, rewards, architecture and all retained state are unchanged.
"""
import argparse
import ast
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import textwrap
import time
from types import SimpleNamespace

from auto_train_mine import write_json
import train_source200_stationary_height200 as base

ROOT, RESULTS = base.ROOT, base.RESULTS
ENVS = 4096
TRANSFER = RESULTS / 'joint_pd_optimization_20261017_112420_601471_source200_stationary_broad200'
ORIGINAL_PREPARE = base.prepare
ORIGINAL_COMMAND = base.command


def prepare(output, previous=None):
    plan, expected = ORIGINAL_PREPARE(output, previous)
    if previous is None:
        original_plan = json.loads((TRANSFER / 'experiment_plan.json').read_text())
        proof = json.loads((TRANSFER / 'source200_transfer_verification.json').read_text())
        before = json.loads((TRANSFER / 'initial_replay/assessment.json').read_text())
        if (proof['source_checkpoint_sha256'] != base.branch.SOURCE_SHA
                or not proof['nominal_standing_preserved'] or before['survivors'] != 27 or before['resets']
                or original_plan['source_sha256'] != plan['source_sha256']
                or original_plan['training_config'] != expected
                or proof['profile'] != base.branch.profile()):
            raise ValueError('Reuse only the intact completed source200 transfer of the identical physical recipe')
        for name, digest in original_plan['source_files'].items():
            if base.sha(Path(name)) != digest:
                raise ValueError('Frozen pre-PPO transfer source changed: ' + name)
        plan['source_replay'] = str(TRANSFER / 'initial_replay')
        plan['protected_reference'] = str(TRANSFER / 'initial_replay/assessment.json')
        plan['source200_transfer_reused'] = str(TRANSFER)
        for name in ('assessment.json', 'manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json', 'source200_environment.json'):
            path = TRANSFER / 'initial_replay' / name
            plan['source_files'][str(path)] = base.sha(path)
    plan.update(native_num_envs=ENVS, environment_parallelism_change={'previous':8192, 'current':ENVS},
        native_transition_batch=ENVS * 48, optimizer_minibatches_per_actual_update=20,
        actual_source_is_original_model200=base.sha(Path(plan['source_checkpoint'])) == base.branch.SOURCE_SHA,
        resource_reason='Host editor about12GiB RSS plus7GiB swap;8192-env initializer measured19GiB. '
                        'Use4096 environments; do not terminate desktop apps or weaken tracking acceptance.',
        source200_requested_rollback_once=True)
    for key in ('scene', 'terrain'):
        expected[key]['num_envs'] = ENVS
    plan['training_config'] = expected
    plan['source_files'][str(Path(__file__).resolve())] = base.sha(__file__)
    write_json(Path(output) / 'experiment_plan.json', plan)
    return plan, expected


def command(*args, **kwargs):
    argv, final, saved = ORIGINAL_COMMAND(*args, **kwargs)
    if argv[argv.index('--num_envs') + 1] != '8192':
        raise ValueError('Apply only the declared parallelism revision')
    argv[argv.index('--num_envs') + 1] = str(ENVS)
    return argv, final, saved


def install_count_validator():
    """Change only the old diagnostic environment-count assertion."""
    function = base.observed.summarize_pair
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) is int and n.value == 8192]
    if len(matches) != 1:
        raise ValueError('Requalify the native record environment-count assertion')
    matches[0].value = ENVS
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), 'exec'), namespace)
    base.observed.summarize_pair = namespace[function.__name__]


def wait_memory(directory):
    while True:
        if (directory / 'STOP').exists():
            raise RuntimeError('Explicit stop requested during resource wait')
        values = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        gib = 1024**3
        ready = values['MemAvailable'] >= 6*gib and values['MemAvailable'] + values['SwapFree'] >= 16*gib
        write_json(directory / 'native_memory_preflight_4096.json', dict(
            updated_at=datetime.now().astimezone().isoformat(), ready=ready, num_envs=ENVS,
            available_bytes=values['MemAvailable'], swap_free_bytes=values['SwapFree'],
            memory_requirement_scaled_to_half_environments=True))
        if ready:
            return
        time.sleep(10)


def run(output, unit, previous=None):
    output = Path(output).resolve()
    install_count_validator()
    original_child = base.block.runtime.run_child

    def child(*args, **kwargs):
        original_reporter = kwargs.get('reporter')
        if original_reporter is not None:
            def report(phase, **values):
                if 'num_envs' in values:
                    values['num_envs'] = ENVS
                return original_reporter(phase, **values)
            kwargs['reporter'] = report
        return original_child(*args, **kwargs)

    base.block.runtime.run_child = child
    plan, expected = prepare(output, previous)
    before = json.loads((Path(plan['source_replay']) / 'assessment.json').read_text())
    base.block.continuation_command = command
    base.block.replay_command = base.replay_command
    base.block.assess = base.acceptance.assess_current
    base.block.assess_directory = base.acceptance.assess_current_legs
    base.interval.install_finalizer(base.observed)

    def prepared(*unused):
        base.observed.record_resource_wait(output, plan)
        wait_memory(output)
        return plan, expected

    pointer = dict(unit=unit, directory=str(output), source_checkpoint=plan['source_checkpoint'],
        controller_kind='reviewed_analytic_tire_physics100', actual_updates_per_review=200,
        native_num_envs=ENVS, training_motion_scale=0., optimizer_reset=False,
        weights_reinitialized=False, no_total_training_update_limit=True, policy_accepted=False)
    for name in ('joint_pd_active_joint_tracking.json', 'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
        write_json(RESULTS / name, pointer)
    base.block.run(SimpleNamespace(previous=Path(plan['previous_output']), output=output,
        reward_key='unused', value=0., updates=200, prepare_only=False), prepare_function=prepared)
    base.observed.finalize_observations(output)
    after = json.loads((output / 'replay/assessment.json').read_text())
    base.acceptance.assessor.assess(output / 'replay', 'assessment_historical10pct.json')
    write_json(output / 'stationary_review.json', base.stationary_decision(after, before))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.unit, args.previous)
