"""Qualify, continue and review an immutable saved plain-PPO controller bundle.

This workflow reuses the original runner, native transition observer, eleven
learning readers, fixed-command metrics and sustained-window checks. Evaluation
matrices are separate files and are never passed into training.
"""
import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import joint_pd_saved_bundle_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def qualify(recipe, output):
    import torch
    import yaml
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    import rsl_rl.runners as runners
    import wheel_legged_gym_isaaclab.exploration as exploration

    runtime.configure(recipe)
    source, saved, cfg = runtime.baseline()
    output.mkdir(exist_ok=False)
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    options = dict(agent['policy'])
    assert options.pop('class_name') == 'ActorCritic'
    assert agent['algorithm']['class_name'] == 'PPO'
    observations = TensorDict({'policy': torch.randn(32, 27)}, [32])

    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(observations, agent['obs_groups'], 6, **options)
        result = object.__new__(cls)
        result.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
        result.current_learning_iteration = 0
        result.logger_type, result.disable_logs = 'tensorboard', True
        return result

    for folder in ('load', 'exploration', 'reload'):
        (output / folder).mkdir()
    upstream, cls = runtime.private_for_count(8192).install_runner(output / 'load')
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'loaded_NN')
        runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'loaded_17Adam')
        means = runner.alg.policy.act_inference(observations).detach().clone()
        previous = runtime.install_leg_exploration(output / 'exploration')
        try:
            floors = runtime.recipe()['exploration']
            handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
                floors['global_minimum_std'], minimum_wheel_std=floors['minimum_wheel_std'])
            runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'guard_NN')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'guard_Adam')
            handle.remove()
        finally:
            exploration.enforce_positive_exploration = previous
        offline = output / 'roundtrip/offline_metadata_not_trained.pt'
        shutil.copytree(source.parent / 'params', offline.parent / 'params')
        expected = runtime.expected_configuration(cfg, 8192)
        (offline.parent / 'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))
        runner.save(str(offline))
        runners.OnPolicyRunner = upstream
        up2, cls2 = runtime.private_for_count(8192).install_runner(output / 'reload')
        try:
            restored = bare(cls2)
            restored.load(str(offline), load_optimizer=True, map_location='cpu')
            runtime.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_NN')
            runtime.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_17Adam')
            assert restored.alg.learning_rate == saved['optimizer_state_dict']['param_groups'][0]['lr']
            assert torch.equal(means, restored.alg.policy.act_inference(observations))
        finally:
            runners.OnPolicyRunner = up2
        argv, final, _ = runtime.command(source, cfg, 'saved_bundle_CPU_fixture', 200,
                                         cfg['commands'], output / 'NOT_A_STOP_REQUEST')
        assert final == runtime.saved_updates(source) + 199
        assert argv[argv.index('--num_envs') + 1] == '8192'
        assert 'agent.save_interval=1' in argv
        assert not any('targeted_turn_command_fraction' in item for item in argv)
        assert all(name not in sys.modules for name in (
            'historical_yaw_focus_sampler', 'historical_yaw_focus_sampler_v2', 'historical_yaw_error_focus'))
        write_json(output / 'native_arguments.json', dict(command=argv, simulator_started=False))
        result = dict(qualified=True, source_checkpoint=str(source), checkpoint_sha256=sha(source),
            actual_ppo_updates=runtime.saved_updates(source), actual_PPO_updates_executed=0,
            selected_num_envs=8192, capacity_test_performed=False,
            all_saved_NN_17Adam_sixstd_actualLR_and_inference_bitwise_preserved=True,
            source_yaml_unchanged=True, ordinary_class_sampler_selected=True,
            retired_sampler_modules_loaded=False, saved_controller_and_reward_values_retained=True,
            source_pins={str(p): sha(p) for p in (SCRIPT, runtime.SCRIPT, recipe, source,
                source.parent / 'params/env.yaml', source.parent / 'params/agent.yaml')},
            optimizer_executed=False, simulator_started=False, policy_accepted=False)
        write_json(output / 'qualification.json', result)
        return result
    finally:
        runners.OnPolicyRunner = upstream


def compare(output, through):
    """Keep original mapping/FK math; admit only the removed sampler metadata."""
    import compare_plain27_recorded_joint_targets as original
    from audit_saved_action_limits import transform
    from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit

    def expected_difference(configs):
        before, after = configs
        assert runtime.normalize_saved_bundle(before)['commands'] == after['commands']
        assert checkpoint_wheel_action_limit(before) == checkpoint_wheel_action_limit(after)
        result = {'/log_dir'}
        if 'targeted_turn_command_fraction' in before['commands']:
            result.add('/commands/targeted_turn_command_fraction')
        if 'wheel_action_limit' not in before:
            result.add('/wheel_action_limit')
        return result

    tree = transform(original.compare, [("{'/log_dir'}", 'expected_difference(cfgs)'),
        ('np.clip(r[..., (2, 5)], -1., 1.)',
         'np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))')])
    scope = dict(original.__dict__, SCRIPT=SCRIPT, expected_difference=expected_difference,
                 saved_limit=checkpoint_wheel_action_limit)
    exec(compile(ast.fix_missing_locations(tree), str(original.SCRIPT), 'exec'), scope)
    return scope['compare'](output, through)


def processes(delegate):
    from train_uniform_control_joint_pd_block200_v5 import ForwardSubprocess

    class Routed(ForwardSubprocess):
        def routed(self, command):
            if isinstance(command, (list, tuple)) and len(command) > 1:
                modes = {'monitor_uniform_control_joint_pd_learning_v3.py': 'monitor',
                         'review_uniform_control_joint_pd_block_v3.py': 'review',
                         'compare_uniform_control_recorded_joint_targets_v3.py': 'compare'}
                mode = modes.get(Path(command[1]).name)
                if mode:
                    command = [command[0], str(SCRIPT), mode, *command[2:]]
            return command

    return Routed(delegate)


def train(pointer):
    import train_uniform_control_joint_pd_block200_v3 as original
    import train_uniform_control_direct8192_block200 as allocation
    import train_AF7001_wheelD075_wheel005_block200 as resources

    spec = read(pointer)
    runtime.configure(spec['uniform_recipe'])
    source = Path(read(spec['request'])['source_checkpoint'])
    selected = Path(spec['allocation'])
    allocation.require_explicit_allocation(selected, block_id=spec['block_id'], source=source, num_envs=8192)
    output = Path(spec['planned_production_output'])
    historical_plan = Path(runtime.recipe()['source_final_verification']).parent / 'experiment_plan.json'
    previous = (original.read, original.SCRIPT, original.subprocess, original.write_json,
                original.require_capacity_selection, resources.wait_resources)

    def scoped_read(path):
        # Do not inherit stale targeted-sampler banners and old experiment deltas.
        return {} if Path(path).resolve() == historical_plan.resolve() else previous[0](path)

    def write(path, value):
        if Path(path) == output / 'experiment_plan.json':
            commands = value['training_config']['commands']
            scale = commands['training_motion_scale']
            value.pop('capacity_admission', None)
            value.update(environment_allocation=str(selected),
                environment_allocation_mode='explicit_user_direct8192',
                capacity_tests_performed=False, actual_capacity_probe_updates=0,
                automatic_4096_fallback=False, scheduled_readonly_milestones=[50, 100, 150, 200],
                saved_bundle_ordinary_sampler_transfer=True, retired_targeted_sampler_enabled=False,
                removed_training_extension='commands.targeted_turn_command_fraction',
                saved_controller_and_reward_values_retained=True,
                training_motion_scale=scale,
                applied_speed_range_m_s=[min(commands['ranges_reverse_lin_vel_x']) * scale,
                                         max(commands['ranges_lin_vel_x']) * scale],
                applied_yaw_range_rad_s=[v * scale for v in commands['ranges_ang_vel_yaw']],
                full_height_range=commands['ranges_height'],
                evaluation_cases_enter_training=False)
        return previous[3](path, value)

    original.read, original.SCRIPT = scoped_read, SCRIPT
    original.subprocess, original.write_json = processes(previous[2]), write
    original.require_capacity_selection = allocation.require_explicit_allocation
    resources.wait_resources = allocation.record_direct_start
    try:
        return original.run(pointer)
    finally:
        (original.read, original.SCRIPT, original.subprocess, original.write_json,
         original.require_capacity_selection, resources.wait_resources) = previous


def monitor(output, recipe):
    import monitor_uniform_control_joint_pd_learning_v3 as original
    import plain27_sampler_metadata_adapter as adapter
    from audit_uniform_motion_milestone import audit as milestone_audit
    import time

    runtime.configure(recipe)
    plan = read(output / 'experiment_plan.json')
    assert plan['native_num_envs'] == 8192 and plan['fixed_standing_environments'] == 2048
    api = adapter.private_module(original.original.SCRIPT, '_saved_bundle_original11_monitor')
    api.SCRIPT, api.subprocess = SCRIPT, processes(api.subprocess)
    api.runtime = SimpleNamespace(profile=runtime.profile, DELTA=plan['next_configuration_changes'])
    checks = []
    for name, stem, options in original.original.CHECKS:
        if name == 'compare_parallel8192_recorded_joint_targets.py':
            name = 'compare_uniform_control_recorded_joint_targets_v3.py'
        elif name == 'audit_parallel8192_count_bound.py':
            name = 'audit_historical_source3001_count_bound.py'
        checks.append((name, stem, options))
    assert len(checks) == 11
    api.CHECKS = tuple(checks)
    while True:
        record = api.tick(output, 'owned_saved_bundle_controller')
        for through in (50, 100, 150, 200):
            ready = (record['committed_native_updates'] > through or
                     through == 200 and record['state'] == 'all_owned_prefix_diagnostics_completed')
            if ready and not (output / f'agent_review_milestone_{through:04d}_readonly.json').exists():
                milestone_audit(output, through)
        for through in record['completed_prefixes']:
            path = output / f'sampled_yaw_reference_components_prefix_{through:04d}_readonly.json'
            if not path.exists():
                command = [sys.executable, str(SCRIPT.with_name('audit_historical_source3001_sampled_yaw_components.py')),
                           str(output), '--through', str(through)]
                with (output / f'extra_yaw_components_{through:04d}.log').open('a') as log:
                    api.subprocess.run(command, cwd=ROOT, stdout=log, stderr=api.subprocess.STDOUT, check=True)
        write_json(output / 'uniform_learning_monitor_status.json', dict(record,
            original11_numeric_readers_unchanged=True, training_modified=False))
        if record['state'] == 'all_owned_prefix_diagnostics_completed':
            return
        time.sleep(15.)


def review(output, recipe):
    import review_uniform_control_joint_pd_block_v3 as original
    original.run(output, recipe)
    # Replay every retained source scope with exactly its recorded cases/seeds.
    # This list is evaluation metadata; it cannot affect native PPO sampling.
    import review_frozen_joint_pd_command_matrix as matrix
    checkpoint = Path(read(output / 'final_checkpoint_verification_readonly.json')['checkpoint'])
    rows, baseline_rows, scopes = [], [], []
    for index, entry in enumerate(runtime.recipe()['retained_evaluation_scopes']):
        before = Path(entry['replay'])
        manifest = read(before / 'manifest.json')
        assert manifest['checkpoint_sha256'] == runtime.recipe()['source_checkpoint_sha256']
        destination = output / 'independent_review' / ('retained_' + entry['name'])
        destination.mkdir(exist_ok=False)
        cases, seeds = manifest['cases'], manifest['seeds']
        assert len(seeds) == 3 and len(cases) <= 36
        batch_file = destination / 'commands.json'
        write_json(batch_file, cases)
        target = destination / 'replay'
        argv = runtime.replay_command(checkpoint, target)
        argv[argv.index('--seconds') + 1] = str(manifest['seconds_per_case'])
        at = argv.index('--seeds') + 1
        argv[at:at + 3] = list(map(str, seeds))
        argv += ['--command-probe-json', str(batch_file)]
        reader, previous_reader = matrix.verify_reader()
        paths = (SCRIPT, runtime.SCRIPT, recipe, checkpoint, before / 'manifest.json',
                 batch_file, reader, previous_reader)
        write_json(destination / 'qualification.json', dict(qualified=True,
            checkpoint_sha256=sha(checkpoint), actual_updates=runtime.saved_updates(checkpoint),
            trial_count=len(cases) * 3, seconds_per_case=manifest['seconds_per_case'],
            numerical_reader_extraction_AST_identical=True,
            batches=[dict(name='replay', replay=str(target), command=argv, cases=cases, seeds=seeds)],
            source_pins={str(p): sha(p) for p in paths}, policy_accepted=False))
        result = matrix.run(destination)
        source_report = matrix.read(before / 'original_and_literal10_conditional_assessment.json')
        assert len(source_report['trials']) == len(result['all_trials'])
        baseline_rows.extend(source_report['trials'])
        rows.extend(result['all_trials'])
        scopes.append(dict(name=entry['name'], summary=result['summary']))
        write_json(output / 'independent_review/retained_scope_progress.json', dict(
            completed_scopes=scopes, rows=len(rows), policy_accepted=False))
    result = dict(complete=True, source_checkpoint_sha256=runtime.recipe()['source_checkpoint_sha256'],
        candidate_checkpoint=str(checkpoint), source_trials=baseline_rows, candidate_trials=rows,
        scope_summaries=scopes, all_failures_retained=True, original_numeric_rules_unchanged=True,
        universal_training_sampler=True, case_specific_training_code=False,
        continuation_requires_agent_review=True, policy_accepted=False)
    write_json(output / 'independent_review/complete_retained_scope_comparison.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'train', 'monitor', 'compare', 'review'))
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pointer', type=Path)
    parser.add_argument('--through', type=int)
    parser.add_argument('directory', type=Path, nargs='?')
    args = parser.parse_args()
    try:
        if args.mode == 'qualify':
            assert args.recipe and args.output
            result = qualify(args.recipe.resolve(), args.output.resolve())
            print(json.dumps(result, ensure_ascii=False))
        elif args.mode == 'train':
            assert args.pointer
            train(args.pointer.resolve())
        elif args.mode == 'compare':
            assert args.directory and args.through
            print(compare(args.directory.resolve(), args.through)[0])
        else:
            assert args.recipe and args.output
            globals()[args.mode](args.output.resolve(), args.recipe.resolve())
    except BaseException as error:
        destination = (Path(read(args.pointer)['planned_production_output'])
                       if args.pointer else args.output)
        if destination and destination.exists():
            write_json(destination / ('saved_bundle_' + args.mode + '_inspection_required.json'),
                dict(error=repr(error), checkpoints_preserved=True, goal_remains_active=True,
                     no_automatic_blind_training_restart=True, policy_accepted=False))
        raise
