"""One continuous two-axis tracking-kernel revision with full-state plain PPO.

Keep the reviewed lower-noise source's architecture, all17 Adam, actual LR,
six learned std, physical model, action maps and ordinary command sampler.
Only the two existing positive tracking-kernel denominators change, using the
same continuous command-relative rule for every environment and task.
The preserved7426 controller remains the independent fallback reference.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

import joint_pd_calibrated_reward_runtime as implementation
from command_relative_tracking_reward import (TAG, PROOF, contract as kernel_contract,
                                               install_class_reward)


SCRIPT = Path(__file__).resolve()
reward_namespace = implementation.recipe.__globals__
wheel_namespace = reward_namespace['parent_namespace']
saved_namespace = wheel_namespace['saved_namespace']
read_recipe = reward_namespace['base_recipe']
parent_profile = implementation.profile_for_count
parent_train = reward_namespace['train']
parent_qualify = reward_namespace['qualify']
parent_compare = reward_namespace['parent_compare']
parent_command = implementation.command
parent_private = implementation.private_for_count
parent_native = implementation.native
parent_evaluator = implementation.evaluator_tree
parent_require_source = implementation.require_source


def recipe():
    value = read_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer']
    assert not value['case_specific_training_logic']
    assert not value['command_overrides'] and not value['command_changes']
    physics = value['physics_changes']
    assert set(physics) == {'robot.spawn.usd_path', 'feedforward_force'}
    target = value['physics_target']
    assert physics['robot.spawn.usd_path'][1] == target['asset']
    assert implementation.sha(target['asset']) == target['asset_sha256']
    assert physics['feedforward_force'][1] == target['vehicle_mass_kg'] * 9.81 / 2.
    controls = value['control_changes']
    assert set(controls) == {'wheel_damping', 'robot.actuators.wheels.damping'}
    assert controls['wheel_damping'] == controls['robot.actuators.wheels.damping']
    assert controls['wheel_damping'][1] == value['control_overrides']['wheel_damping']
    rewards = value['reward_changes']
    assert set(rewards) == set(value['reward_overrides']) == {'tracking_sigma_ang'}
    before_sigma, after_sigma = rewards['tracking_sigma_ang']
    assert 0. < after_sigma < before_sigma
    assert after_sigma == value['reward_overrides']['tracking_sigma_ang']
    before, after = value['source_exploration'], value['exploration']
    keys = {'global_minimum_std', 'minimum_leg_std', 'minimum_wheel_std'}
    assert set(before) == set(after) == keys
    assert 0. < after['minimum_leg_std'] < before['minimum_leg_std']
    assert 0. < after['minimum_wheel_std'] < before['minimum_wheel_std']
    assert after['global_minimum_std'] == after['minimum_leg_std'] == after['minimum_wheel_std']
    assert max(after['global_minimum_std'], after['minimum_leg_std']) < max(
        before['global_minimum_std'], before['minimum_leg_std'])
    changes = {k: [before[k], after[k]] for k in keys if before[k] != after[k]}
    assert set(changes) == keys
    assert value['exploration_changes'] == changes and value['initial_std_reset'] is False
    kernel = value['global_tracking_kernel']
    assert kernel == kernel_contract(kernel['command_caps'],kernel['common_floor_fraction'])
    assert set(kernel['command_caps']) == {'speed','yaw'}
    boundary = value['continuation_source_boundary']
    assert boundary['actual_updates'] > value['source_actual_updates']
    assert boundary['all17_Adam_step'] == boundary['actual_updates'] * 20
    assert implementation.sha(boundary['checkpoint']) == boundary['checkpoint_sha256']
    assert value['effective_change_relative_to_continuation_source'] == {'reward_kernel':kernel}
    assert value['effective_change_relative_to_source'] == dict(
        physics=physics, controller=controls, rewards=rewards, exploration=changes,
        reward_kernel=kernel)
    return value


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, 'Never reload model200'
    path = Path(path).resolve()
    boundary = recipe()['continuation_source_boundary']
    if path != Path(boundary['checkpoint']).resolve():
        return parent_require_source(path, transition=transition, initial_transfer=False)
    import torch
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    assert implementation.sha(path) == boundary['checkpoint_sha256']
    saved = torch.load(path,map_location='cpu',weights_only=True)
    assert implementation.tensor_finite(saved)
    assert implementation.saved_updates(path) == boundary['actual_updates']
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert {int(v['step']) for v in saved['optimizer_state_dict']['state'].values()} == {boundary['all17_Adam_step']}
    profile = json.loads(json.dumps(saved['infos'][implementation.KEY]))
    assert profile == boundary['original_profile'], 'Require this exact reviewed source profile'
    _, original, anchor_cfg = implementation.baseline()
    cfg = validate_checkpoint_robot(path)
    expected = implementation.expected_configuration(anchor_cfg,8192)
    expected['log_dir'] = cfg['log_dir']
    assert cfg == expected
    assert saved['model_state_dict'].keys() == original['model_state_dict'].keys()
    assert all(v.shape == original['model_state_dict'][k].shape for k,v in saved['model_state_dict'].items())
    return saved,cfg


def profile_for_count(count):
    result = parent_profile(count)
    result.update(profile='plain27_global_command_relative_tracking_kernels_v1',
        native_runtime_sha256=implementation.sha(SCRIPT),
        precision_noise_runtime_sha256=implementation.sha(SCRIPT),
        uniform_exploration_changes={},
        changed_noise_heads=list(range(6)), saved_initial_std_reset=False,
        all_six_std_entries_trainable=True, leg_exploration_floor_unchanged=False,
        saved_leg_PD_action_maps_and_exploration_retained=False,
        saved_leg_PD_and_action_maps_retained=True, saved_initial_distribution_retained=True,
        same_common_future_floor_for_every_command_and_head=True,
        same_future_wheel_floor_for_every_command=True,
        original_calibration_anchor_sha256=recipe()['source_checkpoint_sha256'],
        continuation_source_sha256=recipe()['continuation_source_boundary']['checkpoint_sha256'],
        continuation_source_actual_updates=recipe()['continuation_source_boundary']['actual_updates'],
        global_relative_tracking_kernel=recipe()['global_tracking_kernel'],
        current_source_exploration_floor_unchanged=True,
        existing_reward_equations_retained=False,
        all_reward_equations_except_two_positive_kernel_denominators_retained=True,
        original_PPO_loss_equations_and_adaptive_LR_rule_retained=True)
    return result


def command(*args, **kwargs):
    original = wheel_namespace['old_command']
    def unique_flag(*aa, **kk):
        argv,final,saved = original(*aa,**kk)
        assert argv.count('--allow_physics_contract_change') <= 1
        return [a for a in argv if a != '--allow_physics_contract_change'],final,saved
    wheel_namespace['old_command'] = unique_flag
    try:
        return parent_command(*args,**kwargs)
    finally:
        wheel_namespace['old_command'] = original


def private_for_count(count, selected_profile=None, *, is_replay=False):
    module = parent_private(count,selected_profile,is_replay=is_replay)
    original = module.observe_environment
    def observe(env,output):
        result = original(env,output)
        raw = env.unwrapped
        method = type(raw)._get_rewards
        spec = recipe()['global_tracking_kernel']
        assert getattr(method,TAG,None) == spec
        proof = getattr(method,PROOF)
        assert proof['all_other_application_reward_AST_identical']
        from auto_train_mine import write_json
        write_json(Path(output)/'uniform_relative_tracking_reward_contract.json',dict(
            verified=True,actual_num_envs=raw.num_envs,global_tracking_kernel=spec,
            builder_proof=proof,source_class_reward_revised_explicitly=True,
            all_other_reward_terms_clips_and_terminations_retained=True,
            original_action_maps_physics_and_sampler_retained=True,
            ordinary_PPO_loss_retained=True,policy_accepted=False))
        path = Path(output)/'uniform_recipe_environment.json'
        evidence = json.loads(path.read_text())
        evidence['existing_positive_kernel_equations_retained'] = False
        evidence['declared_global_tracking_kernel'] = spec
        evidence['verified_reward_call_chain'].update(
            original_class_reward_reached=False,declared_class_reward_kernel_reached=True,
            all_other_application_reward_AST_identical=True)
        write_json(path,evidence)
        return result
    module.observe_environment = observe
    return module


def native(output,arguments):
    import gymnasium as gym
    source=Path(arguments[arguments.index('--checkpoint_path')+1])
    assert implementation.saved_updates(source)>=recipe()['continuation_source_boundary']['actual_updates']
    original_make = gym.make
    def make(*args,**kwargs):
        cfg = kwargs['cfg']
        assert cfg.leg_control_mode == 'implicit_joint_position'
        from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv
        spec = recipe()['global_tracking_kernel']
        install_class_reward(WheelLeggedVMCFlatEnv,spec['command_caps'],spec['common_floor_fraction'])
        return original_make(*args,**kwargs)
    gym.make = make
    try:
        return parent_native(output,arguments)
    finally:
        gym.make = original_make


def evaluator_tree():
    tree,path,proof = parent_evaluator()
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    at = [i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
          and any(isinstance(t,ast.Name) and t.id=='height' for t in n.targets)]
    assert len(at)==1
    before = ast.dump(tree)
    spec = recipe()['global_tracking_kernel']
    inserted = ast.parse('from command_relative_tracking_reward import install_class_reward as _install_command_relative_reward\n'
        + '_install_command_relative_reward(WheelLeggedVMCFlatEnv, '
        + repr(spec['command_caps'])+', '+repr(spec['common_floor_fraction'])+')').body
    point=at[0]
    main.body[point:point]=inserted
    manifests=[n for n in ast.walk(main) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='manifest' for t in n.targets)]
    assert len(manifests)==1 and isinstance(manifests[0].value,ast.Dict)
    manifest = manifests[0].value
    manifest.keys.append(ast.Constant('application_reward_kernel'))
    manifest.values.append(ast.parse(repr(spec),mode='eval').body)
    revised=copy.deepcopy(tree)
    manifest.keys.pop();manifest.values.pop()
    del main.body[point:point+len(inserted)]
    assert ast.dump(tree)==before
    proof.update(declared_global_relative_tracking_kernel=spec,
        original_evaluator_metrics_limits_windows_and_failure_logic_unchanged=True,
        only_application_reward_install_and_manifest_metadata_added=True)
    return ast.fix_missing_locations(revised),path,proof


for namespace in (reward_namespace, wheel_namespace, saved_namespace):
    namespace['recipe'] = recipe
    namespace['SCRIPT'] = SCRIPT
reward_namespace['profile_for_count'] = profile_for_count
wheel_namespace['profile_for_count'] = profile_for_count
implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.profile_for_count = profile_for_count
implementation.require_source = require_source
implementation.command = command
implementation.private_for_count = private_for_count
implementation.native = native
implementation.evaluator_tree = evaluator_tree
sys.modules['joint_pd_command_relative_runtime'] = implementation


def qualify(recipe_path, output):
    import json
    result = parent_qualify(recipe_path, output)
    before, after = recipe()['source_exploration'], recipe()['exploration']
    anchor_command = result['fifty_update_native_command']
    assert anchor_command[anchor_command.index('--min_exploration_std') + 1] == str(after['global_minimum_std'])
    assert anchor_command[anchor_command.index('--min_wheel_exploration_std') + 1] == str(after['minimum_wheel_std'])
    result.update(saved_leg_PD_action_maps_and_exploration_retained=False,
        saved_leg_PD_and_action_maps_retained=True,
        source_exploration=before, future_projection_floors=after,
        source_saved_sixstd_not_reset=True,
        effective_leg_floor_before_after=[max(before['global_minimum_std'], before['minimum_leg_std']),
                                         max(after['global_minimum_std'], after['minimum_leg_std'])],
        effective_wheel_floor_before_after=[max(before['global_minimum_std'], before['minimum_wheel_std']),
                                           max(after['global_minimum_std'], after['minimum_wheel_std'])],
        future_projection_change_is_not_a_saved_std_reset=True,
        global_relative_tracking_kernel=recipe()['global_tracking_kernel'],
        exact_continuation_source_boundary=recipe()['continuation_source_boundary'])
    source = Path(recipe()['continuation_source_boundary']['checkpoint'])
    saved,cfg = require_source(source)
    argv,final,_ = command(source,cfg,'relative_kernel_CPU_actual_source',50,
                           cfg['commands'],output/'NO_STOP_REQUEST')
    assert final+1 == recipe()['continuation_source_boundary']['actual_updates']+50
    assert saved['model_state_dict']['std'].min() >= recipe()['exploration']['global_minimum_std']
    _,_,replay_proof=evaluator_tree()
    result.update(actual_continuation_source_verified=True,
        actual_continuation_checkpoint=str(source),actual_continuation_checkpoint_sha256=implementation.sha(source),
        actual_continuation_native_command=argv,declared_replay_reward_install_proof=replay_proof,
        calibration_anchor_qualification_does_not_replace_actual_source_load_test=True)
    (output / 'qualification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def train(request):
    import train_saved_joint_pd_review50 as workflow
    previous = workflow.write_json

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            boundary=recipe()['continuation_source_boundary']
            assert value['source_actual_ppo_updates']>=boundary['actual_updates']
            kernel=recipe()['global_tracking_kernel']
            value['next_configuration_changes']=dict(rewards={},commands={},exploration={},
                controller={},physics={},reward_kernel=kernel)
            value.update(only_exploration_change={},only_reward_change={},only_controller_change={},
                only_physics_change={},only_runtime_sampling_change={},
                only_reward_kernel_change=kernel,
                source_exploration=recipe()['source_exploration'],
                actual_future_projection_floors=recipe()['exploration'],
                saved_initial_std_reset=False, leg_exploration_floor_unchanged=True,
                saved_leg_PD_action_maps_exploration_and_reward_structure_retained=False,
                saved_leg_PD_action_maps_and_reward_structure_retained=True,
                same_common_future_floor_for_every_command_and_head=True,
                actual_continuation_source_all17Adam_sixstd_and_LR_retained=True,
                existing_reward_names_coefficients_and_clips_retained=True,
                all_reward_math_except_two_positive_kernel_denominators_retained=True,
                original_PPO_loss_and_observation_action_maps_retained=True,
                no_condition_specific_noise_rule=True)
        return previous(path, value)

    workflow.write_json = write
    try:
        return parent_train(request)
    finally:
        workflow.write_json = previous


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        at = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:at])
        implementation.configure(args.recipe)
        implementation.native(args.observation_output, sys.argv[at + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        implementation.configure(args.recipe)
        implementation.replay(remaining)
    elif mode == 'compare':
        parser.add_argument('directory', type=Path)
        parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        parent_compare(args.directory.resolve(), args.through)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        qualify(args.recipe.resolve(), args.output.resolve())
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        train(args.request.resolve())
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
