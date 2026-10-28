"""Retained-state, frozen27 parity, and full entry checks for the leg005 trial."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step
from qualify_source801_drift50_transfer import differences
from review_plain27_single_reward_block200 import rebound
import qualify_plain27_exploration_transfer as original
import joint_pd_preserved6801_leg005_runtime as runtime

SCRIPT = Path(__file__).resolve()
OUTPUT = ROOT / 'mine_model/results/joint_pd_preserved6801_leg005_qualification_20261021'


def qualify():
    source, _, _ = runtime.anchor()
    result = rebound(original.qualify, SCRIPT=SCRIPT)(source, runtime.SCRIPT, runtime.DECISION, OUTPUT)
    # Exercise below-floor wheel entries as well as legs; the original source
    # roundtrip separately verifies every saved NN/Adam tensor and adaptive LR.
    import wheel_legged_gym_isaaclab.exploration as exploration
    policy = torch.nn.Module()
    policy.std = torch.nn.Parameter(torch.tensor([.01] * 6))
    optimizer = torch.optim.Adam(policy.parameters())
    old = runtime.install_leg_exploration(OUTPUT / 'extra_per_head_projection')
    handle = None
    fixtures = []
    try:
        handle = exploration.enforce_positive_exploration(policy, optimizer, .005, minimum_wheel_std=.01)
        identity = policy.std
        state = copy.deepcopy(optimizer.state_dict())
        for values in [[-.02, .002, -.001, .004, .007, .002], [.02, .03, .02, .04, .05, .03]]:
            fixture = torch.tensor(values)
            with torch.no_grad():
                policy.std.copy_(fixture)
            for hook in optimizer._optimizer_step_post_hooks.values():
                hook(optimizer, (), {})
            expected = fixture.clamp_min(.005)
            expected[[2, 5]] = expected[[2, 5]].clamp_min(.01)
            if not torch.equal(expected, policy.std) or policy.std is not identity or not identity.requires_grad:
                raise ValueError('The actual registered six-head floor or parameter identity changed')
            runtime.base.equal_state(state, optimizer.state_dict(), 'extra_floor_fixture_Adam')
            fixtures.append(dict(before=values, expected=expected.tolist(), actual=policy.std.tolist()))
    finally:
        if handle is not None:
            handle.remove()
        exploration.enforce_positive_exploration = old
    result.update(only_exploration_change=runtime.DELTA['exploration'],
        only_reward_change={}, only_command_change={}, only_controller_change={},
        extra_per_head_floor_fixtures=fixtures,
        all_six_std_entries_trainable_and_parameter_identity_preserved=True,
        actual_optimizer_steps_in_projection_fixtures=0,
        frozen_native_transfer_required=True, native_transfer_qualified=False)
    for p in (SCRIPT, original.SCRIPT, runtime.SCRIPT):
        result['source_files'][str(p)] = sha(p)
    write_json(OUTPUT / 'qualification.json', result)
    return result


def frozen():
    runtime.repair_ready()
    before, after = runtime.TRAINING / 'replay', OUTPUT / 'native_exploration_transfer'
    read = lambda directory, name: json.loads((directory / name).read_text())
    a, b = [read(p, 'manifest.json') for p in (before, after)]
    da, db = [read(p, 'completed.json') for p in (before, after)]
    ca, cb = [read(p, 'resolved_env.json') for p in (before, after)]
    if (a['checkpoint_sha256'] != runtime.ANCHOR_SHA or b['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not da['completed'] or not db['completed'] or da != db
            or da['samples'] != 27000 or a['sampled_policy_actions'] or b['sampled_policy_actions']
            or differences(ca, cb)):
        raise ValueError('Frozen27 must retain the actual source, environment and every episode')
    old, new = [np.genfromtxt(p / 'trace.csv', delimiter=',', names=True) for p in (before, after)]
    if (old.dtype.names != new.dtype.names or len(new) != 27000
            or any(not np.array_equal(old[k], new[k]) for k in old.dtype.names)):
        raise ValueError('A physical trajectory, action, observation or reward changed')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(
            after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[k], y[k]) for k in x.files):
            raise ValueError('Frozen policy inputs or deterministic means changed')
    evidence = read(after, 'preserved6801_leg005_base_execution_evidence.json')
    if evidence['profile'] != runtime.profile() or evidence['authoritative_current_delta'] != runtime.DELTA:
        raise ValueError('The actual frozen runtime differs from the selected exploration-only trial')
    result = dict(qualified=True, source_sha256=runtime.ANCHOR_SHA,
        only_exploration_change=runtime.DELTA['exploration'], only_reward_change={},
        full27_samples=27000, physical_outputs_observations_actions_and_all_rewards_bitwise_equal=True,
        initial_saved_std_and_model_unchanged=True, saved_environment_configuration_unchanged=True,
        source_data_sha256={str(p / name): sha(p / name) for p in (before, after)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')},
        PPO_updates_performed=0, optimizer_steps_performed=0, policy_accepted=False,
        learning_benefit_unverified=True)
    for p in (SCRIPT, runtime.SCRIPT):
        result['source_data_sha256'][str(p)] = sha(p)
    write_json(OUTPUT / 'native_exploration_transfer_qualification.json', result)
    return result


def preflight():
    import train_preserved6801_leg005_200 as controller
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    output = controller.RESULTS / 'joint_pd_optimization_20261021_preflight_preserved6801_leg005_200'
    plan, expected = controller.prepare(output, runtime.TRAINING)
    tree = ast.parse(inspect.getsource(controller.helper.base.block.run))
    required = sorted({n.slice.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
        and isinstance(n.value, ast.Name) and n.value.id == 'plan'
        and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)})
    source = Path(plan['source_checkpoint'])
    cfg = validate_checkpoint_robot(source)
    if (not set(required) <= set(plan) or expected != cfg
            or plan['only_exploration_change'] != runtime.DELTA['exploration']
            or any(plan[k] for k in ('only_reward_change', 'only_command_change', 'only_controller_change'))):
        raise ValueError('The full original block plan changed beyond exploration metadata')
    args = source, cfg, plan['run_name'], 200, cfg['commands'], output / 'STOP'
    options = dict(minimum_wheel_std=.03, joint_mean_guard_coef=0., joint_height_feedback_coef=0.)
    argv, final, saved = controller.command(*args, **options)
    old, old_final, _ = runtime.previous.command(*args, **options)
    predicted = list(old)
    predicted[predicted.index(str(runtime.previous.SCRIPT))] = str(runtime.SCRIPT)
    predicted[predicted.index('--min_exploration_std') + 1] = '.005'
    # Command builders use a canonical decimal spelling.
    predicted[predicted.index('--min_exploration_std') + 1] = str(runtime.GLOBAL_STD_FLOOR)
    predicted.append(runtime.FLAG)
    if (argv != predicted or final != old_final or final != 7000
            or argv[argv.index('--target_iteration') + 1] != '7001'
            or optimizer_step(saved) != 136020 or len(saved['optimizer_state_dict']['state']) != 17
            or any(float(v['step']) != 136020 for v in saved['optimizer_state_dict']['state'].values())
            or argv.count('agent.policy.class_name=ActorCritic') != 1
            or argv.count('env.rewards.yaw_rate_error_sq=-1.0') != 1
            or argv.count('env.wheel_action_limit=1.25') != 1
            or argv[argv.index('--min_wheel_exploration_std') + 1] != '0.01'
            or any(x in argv for x in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'))):
        raise ValueError('The original ordinary200 entry changed outside its declared leg floor')
    result = dict(qualified=True, controller_sha256=sha(controller.SCRIPT), preflight_sha256=sha(SCRIPT),
        source_actual_updates=6801, target_actual_updates=7001, source_all17_Adam_step=136020,
        target_all17_Adam_step=140020, required_original_plan_keys=required,
        actual_block_command=argv, only_exploration_change=runtime.DELTA['exploration'],
        only_reward_change={}, only_command_change={}, only_controller_change={},
        saved_actual_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        initial_saved_std=saved['model_state_dict']['std'].tolist(),
        early_learning_inspection_actual_updates=[50, 100], PPO_updates_performed=0,
        simulator_started=False, policy_accepted=False)
    with (OUTPUT / 'full_original_block_entry_preflight.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'frozen', 'preflight'))
    result = globals()[parser.parse_args().mode]()
    print(json.dumps({k: result[k] for k in ('qualified', 'source_actual_updates',
        'target_actual_updates', 'only_exploration_change', 'full27_samples') if k in result}))
