"""One linear-only continuous reward refinement over the verified source runtime.

All original source-load, ordinary PPO, observation, action, command, physical
and numerical-review implementations come from the pinned parent. No failed
evaluation command or height selects a sampling, control or reward branch.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

import joint_pd_uniform_source_refinement_runtime as base
from linear_command_relative_reward import TAG, PROOF, contract, install_class_reward

SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = base.SCRIPT
ORIGINAL_CONFIGURE = base.configure
ORIGINAL_PROFILE = base.profile_for_count
ORIGINAL_IMMEDIATE = base.immediate_changes
ORIGINAL_EVALUATOR = base.evaluator_tree
ORIGINAL_PRIVATE = base.private_for_count
ORIGINAL_NATIVE = base.native
ORIGINAL_QUALIFY = base.qualify
_path = None
_kernel = None


def __getattr__(name):
    return getattr(base, name)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def configure(path):
    global _path, _kernel
    path = Path(path).resolve()
    if _path is not None:
        assert path == _path, 'One immutable recipe per process'
        return base.recipe()
    spec = read(path)['uniform_source_refinement']['global_linear_command_relative_kernel']
    assert spec == contract(spec['speed_cap_m_s'], spec['floor_fraction'])
    ORIGINAL_CONFIGURE(path)
    cfg = base.baseline()[2]
    bounds = [*cfg['commands']['ranges_lin_vel_x'],
              *cfg['commands']['ranges_reverse_lin_vel_x']]
    assert max(abs(float(x)) for x in bounds) == spec['speed_cap_m_s']
    _path, _kernel = path, spec
    namespaces = list({id(v): v for v in [base.__dict__, *base.NAMESPACES]}.values())
    for namespace in namespaces:
        namespace.update(SCRIPT=SCRIPT, profile_for_count=profile_for_count,
                         private_for_count=private_for_count, evaluator_tree=evaluator_tree)
    base.configure = configure
    base.profile_for_count = profile_for_count
    base.immediate_changes = immediate_changes
    base.private_for_count = private_for_count
    base.evaluator_tree = evaluator_tree
    base.parent.configure = configure
    base.parent.private_for_count = private_for_count
    base.parent.evaluator_tree = evaluator_tree
    return base.recipe()


def immediate_changes(cfg, source_floors=None):
    result = ORIGINAL_IMMEDIATE(cfg, source_floors)
    result['rewards']['positive_linear_variance_rule'] = ['constant_variance', copy.deepcopy(_kernel)]
    return result


def profile_for_count(count):
    result = ORIGINAL_PROFILE(count)
    result.update(profile='plain27_linear_only_command_relative_precision_v1',
        native_runtime_sha256=base.sha(SCRIPT),
        uniform_recipe_sha256=base.sha(_path),
        global_linear_command_relative_kernel=copy.deepcopy(_kernel),
        global_reward_changes=copy.deepcopy(immediate_changes(base.baseline()[2])['rewards']),
        existing_reward_equations_retained=False,
        all_reward_equations_except_positive_linear_denominator_retained=True,
        original_yaw_positive_and_negative_rewards_retained=True,
        source_physics_reward_commands_leg_PD_and_action_maps_retained=False,
        source_physics_commands_leg_PD_and_action_maps_retained=True,
        saved_initial_std_reset=False, saved_initial_distribution_retained=True,
        no_postload_std_pulse=True,
        base_runtime_sha256=base.sha(BASE_SCRIPT))
    return result


def private_for_count(count, selected_profile=None, *, is_replay=False):
    module = ORIGINAL_PRIVATE(count, selected_profile, is_replay=is_replay)
    original = module.observe_environment
    def observe(env, output):
        result = original(env, output)
        raw = env.unwrapped
        method = type(raw)._get_rewards
        assert getattr(method, TAG, None) == _kernel
        proof = getattr(method, PROOF)
        assert proof['all_other_application_reward_AST_identical']
        write(Path(output) / 'linear_command_relative_reward_contract.json', dict(
            verified=True, actual_num_envs=raw.num_envs,
            global_linear_command_relative_kernel=_kernel, builder_proof=proof,
            yaw_kernel_and_negative_costs_retained=True,
            original_PPO_loss_and_action_physics_sampler_retained=True,
            policy_accepted=False))
        path = Path(output) / 'uniform_recipe_environment.json'
        evidence = read(path)
        evidence.update(original_uniform_reward_structure=False,
                        declared_linear_only_reward_contract=_kernel)
        evidence['verified_reward_call_chain'].update(original_class_reward_reached=False,
            declared_class_linear_only_reward_reached=True,
            all_other_application_reward_AST_identical=True)
        write(path, evidence)
        return result
    module.observe_environment = observe
    return module


def native(output, arguments):
    import gymnasium as gym
    original_make = gym.make
    def make(*args, **kwargs):
        assert kwargs['cfg'].leg_control_mode == 'implicit_joint_position'
        from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv
        install_class_reward(WheelLeggedVMCFlatEnv, _kernel['speed_cap_m_s'], _kernel['floor_fraction'])
        return original_make(*args, **kwargs)
    gym.make = make
    try:
        return ORIGINAL_NATIVE(output, arguments)
    finally:
        gym.make = original_make


def evaluator_tree():
    tree, source, proof = ORIGINAL_EVALUATOR()
    initial = ast.dump(tree)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    indices = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'height' for t in n.targets)]
    assert len(indices) == 1
    inserted = ast.parse('from linear_command_relative_reward import install_class_reward as _install_linear_relative\n'
        + '_install_linear_relative(WheelLeggedVMCFlatEnv, '
        + repr(_kernel['speed_cap_m_s']) + ', ' + repr(_kernel['floor_fraction']) + ')').body
    at = indices[0]
    main.body[at:at] = inserted
    nodes = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'manifest' for t in n.targets)]
    assert len(nodes) == 1 and isinstance(nodes[0].value, ast.Dict)
    manifest = nodes[0].value
    manifest.keys.append(ast.Constant('application_linear_reward_kernel'))
    manifest.values.append(ast.parse(repr(_kernel), mode='eval').body)
    revised = copy.deepcopy(tree)
    manifest.keys.pop()
    manifest.values.pop()
    del main.body[at:at + len(inserted)]
    assert ast.dump(tree) == initial
    proof.update(declared_linear_only_reward_kernel=copy.deepcopy(_kernel),
        all_original_metrics_limits_windows_and_failure_logic_unchanged=True,
        only_application_reward_install_and_manifest_metadata_added=True)
    return ast.fix_missing_locations(revised), source, proof


def qualify(path, output):
    configure(path)
    result = ORIGINAL_QUALIFY(path, output)
    _, _, replay_proof = evaluator_tree()
    result.update(global_linear_only_reward_kernel=copy.deepcopy(_kernel),
        original_yaw_kernel_and_costs_unchanged=True,
        original_saved_NN_sixstd_17Adam_actualLR_retained=True,
        frozen_replay_same_kernel_binding=replay_proof,
        parameter_change_selected=False, training_launched=False)
    for p in (SCRIPT, BASE_SCRIPT, Path(__file__).with_name('linear_command_relative_reward.py'),
              Path(__file__).with_name('command_relative_tracking_kernel.py')):
        result['source_pins'][str(p)] = base.sha(p)
    write(output / 'qualification.json', result)
    return result


def train(request_path):
    import train_saved_joint_pd_review50 as workflow
    request = read(request_path)
    configure(request['recipe'])
    previous = workflow.write_json
    def scoped_write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            value.update(saved_leg_PD_action_maps_and_reward_equations_retained=False,
                all_reward_equations_except_positive_linear_denominator_retained=True,
                original_yaw_kernel_and_costs_unchanged=True,
                declared_linear_only_reward_contract=copy.deepcopy(_kernel),
                saved_initial_std_reset=False, no_postload_std_pulse=True)
        return previous(path, value)
    workflow.write_json = scoped_write
    try:
        return base.train(request_path)
    finally:
        workflow.write_json = previous


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        at = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:at]); configure(args.recipe)
        native(args.observation_output, sys.argv[at + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:]); configure(args.recipe)
        base.replay(remaining)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:]); result = qualify(args.recipe.resolve(), args.output.resolve())
        print(json.dumps(dict(qualified=result['qualified'], checkpoint_sha256=result['checkpoint_sha256'],
                              actual_updates=result['actual_ppo_updates'], training_launched=False)))
    elif mode == 'compare':
        parser.add_argument('directory', type=Path); parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:]); configure(args.recipe); base.compare(args.directory.resolve(), args.through)
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:]); configure(args.recipe); train(args.request.resolve())
    else:
        raise ValueError(mode)
