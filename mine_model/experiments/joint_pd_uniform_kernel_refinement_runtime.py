"""Refine existing constant tracking-kernel widths from a full-state checkpoint.

The pinned parent runtime still supplies PPO, observations, actions, physics,
sampling, exploration and numerical readers. This adapter changes declared
global config scalars only. No command, height, sign or evaluation case is an
input to a control or reward rule here. Preparing/qualifying a recipe cannot
start training; train additionally requires a completed-review decision.
"""
import argparse
import ast
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

import torch

import joint_pd_calibrated_common_noise_runtime as parent


SCRIPT = Path(__file__).resolve()
PARENT_CONFIGURE = parent.configure
PARENT_BASELINE = parent.baseline
PARENT_EXPECTED = parent.expected_configuration
PARENT_PROFILE = parent.profile_for_count
PARENT_COMMAND = parent.command
PARENT_EVALUATOR = parent.evaluator_tree
PARENT_TRAIN = PARENT_PROFILE.__globals__['train']
PARENT_COMPARE = PARENT_PROFILE.__globals__['parent_compare']
NAMESPACES = [parent.__dict__, PARENT_PROFILE.__globals__]
NAMESPACES += [PARENT_PROFILE.__globals__[name] for name in
               ('reward_namespace', 'wheel_namespace', 'saved_namespace')]
_path = None
_refinement = None
_selected = None


def __getattr__(name):
    return getattr(parent, name)


def configure(path):
    global _path, _refinement, _selected
    path = Path(path).resolve()
    if _path is not None:
        if path != _path:
            raise ValueError('One immutable refinement recipe per process')
        return parent.recipe()
    value = json.loads(path.read_text())
    refinement = value['uniform_kernel_refinement']
    assert refinement['schema'] == 'uniform_existing_kernel_refinement_v1'
    source_recipe = Path(refinement['source_recipe'])
    assert parent.sha(source_recipe) == refinement['source_recipe_sha256']
    # Keep the already qualified parent contract intact. Only the nested
    # refinement specification is new; it cannot silently rewrite its base.
    ancestor_recipe = json.loads(source_recipe.read_text())
    assert {k: v for k, v in value.items() if k != 'uniform_kernel_refinement'} == ancestor_recipe
    PARENT_CONFIGURE(path)
    _, _, ancestor_cfg = PARENT_BASELINE()
    source = Path(refinement['source_checkpoint']).resolve()
    assert parent.sha(source) == refinement['source_checkpoint_sha256']
    saved = torch.load(source, map_location='cpu', weights_only=True)
    cfg = parent.validate_checkpoint_robot(source)
    expected = PARENT_EXPECTED(ancestor_cfg, 8192)
    expected['log_dir'] = cfg['log_dir']
    assert cfg == expected and parent.tensor_finite(saved)
    old_profile = PARENT_PROFILE(8192)
    old_profile['uniform_recipe_sha256'] = refinement['source_recipe_sha256']
    assert saved['infos'][parent.KEY] == old_profile
    actual = parent.saved_updates(source)
    proof = json.loads(Path(refinement['source_final_verification']).read_text())
    assert proof['verified'] and proof['checkpoint_sha256'] == parent.sha(source)
    assert actual == refinement['source_actual_updates'] == proof['actual_ppo_updates']
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert {float(v['step']) for v in saved['optimizer_state_dict']['state'].values()} == {actual * 20}
    assert saved['model_state_dict']['actor.0.weight'].shape[1] == 27
    assert saved['model_state_dict']['critic.0.weight'].shape[1] == 27
    assert saved['model_state_dict']['std'].shape == (6,)
    changes = refinement['reward_changes']
    assert changes and set(changes) <= {'tracking_sigma', 'tracking_sigma_ang'}
    for key, pair in changes.items():
        assert len(pair) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in pair)
        assert 0. < pair[1] < pair[0] == cfg['rewards'][key]
    assert not refinement['optimizer_reset'] and not refinement['std_reset']
    assert not refinement['actual_LR_reset'] and not refinement['case_specific_training_logic']
    _path, _refinement, _selected = path, refinement, (source, saved, cfg)
    # Install before any environment/runner exists. The pinned implementations
    # call these globals; their numerical methods and source files stay intact.
    for namespace in NAMESPACES:
        namespace.update(SCRIPT=SCRIPT, baseline=baseline,
                         expected_configuration=expected_configuration,
                         profile_for_count=profile_for_count)
    parent.configure = configure
    parent.require_source = require_source
    parent.command = command
    parent.evaluator_tree = evaluator_tree
    return parent.recipe()


def baseline():
    assert _selected is not None
    return _selected


def expected_configuration(cfg, count):
    normalized = copy.deepcopy(cfg)
    # The parent's angular-width validator knows its original transition.
    # Restore each declared width to this recipe's source value before asking
    # it to preserve every other field; then apply the new constant target.
    for key, pair in _refinement['reward_changes'].items():
        assert normalized['rewards'][key] in pair
        normalized['rewards'][key] = pair[0]
    result = PARENT_EXPECTED(normalized, count)
    for key, pair in _refinement['reward_changes'].items():
        assert result['rewards'][key] in pair
        result['rewards'][key] = pair[1]
    return result


def profile_for_count(count):
    result = PARENT_PROFILE(count)
    ancestor_changes = copy.deepcopy(result['global_reward_changes'])
    ancestor_noise_changes = copy.deepcopy(result['uniform_exploration_changes'])
    result.update(
        profile='plain27_uniform_existing_kernel_refinement_v1',
        native_runtime_sha256=parent.sha(SCRIPT),
        uniform_recipe_sha256=parent.sha(_path),
        continuation_source_sha256=_refinement['source_checkpoint_sha256'],
        retained_ancestor_source_sha256=parent.recipe()['source_checkpoint_sha256'],
        inherited_ancestor_reward_changes=ancestor_changes,
        inherited_ancestor_exploration_changes=ancestor_noise_changes,
        global_reward_changes=copy.deepcopy(_refinement['reward_changes']),
        declared_physics_changes={}, global_control_changes={},
        uniform_exploration_changes={}, changed_noise_heads=[],
        leg_exploration_floor_unchanged=True,
        saved_leg_PD_action_maps_and_exploration_retained=True,
        uniform_reward_overrides={**result['uniform_reward_overrides'],
                                  **{k: v[1] for k, v in _refinement['reward_changes'].items()}},
        source_physics_controller_commands_and_future_noise_floors_retained=True,
        same_constant_kernel_for_every_height_speed_sign_and_group=True,
        source_NN_sixstd_17Adam_actualLR_reset=False)
    return result


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, 'No original model200 reload'
    path = Path(path).resolve()
    source, initial, cfg = baseline()
    if path == source:
        assert parent.sha(path) == _refinement['source_checkpoint_sha256']
        return initial, cfg
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = parent.validate_checkpoint_robot(path)
    expected = expected_configuration(cfg, 8192)
    expected['log_dir'] = current['log_dir']
    assert current == expected and parent.tensor_finite(saved)
    assert saved['infos'][parent.KEY] == profile_for_count(8192)
    assert not saved['infos'][parent.KEY].get('capacity_only_probe', False)
    assert parent.saved_updates(path) >= parent.saved_updates(source)
    assert saved['model_state_dict'].keys() == initial['model_state_dict'].keys()
    assert all(saved['model_state_dict'][k].shape == v.shape for k, v in initial['model_state_dict'].items())
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert {float(v['step']) for v in saved['optimizer_state_dict']['state'].values()} == {parent.saved_updates(path) * 20}
    return saved, current


@contextmanager
def unique_inherited_physics_flag():
    namespace = PARENT_COMMAND.__globals__
    original = namespace['old_command']

    def without_inherited_flag(*args, **kwargs):
        argv, final, saved = original(*args, **kwargs)
        assert argv.count('--allow_physics_contract_change') <= 1
        return [v for v in argv if v != '--allow_physics_contract_change'], final, saved

    namespace['old_command'] = without_inherited_flag
    try:
        yield
    finally:
        namespace['old_command'] = original


def command(source, cfg, label, updates, commands, stop_file):
    with unique_inherited_physics_flag():
        argv, final, saved = PARENT_COMMAND(source, cfg, label, updates, commands, stop_file)
    for key, pair in _refinement['reward_changes'].items():
        prefix = 'env.rewards.' + key + '='
        indices = [i for i, v in enumerate(argv) if v.startswith(prefix)]
        assert len(indices) == 1
        argv[indices[0]] = prefix + str(pair[1])
    assert argv[1] == str(SCRIPT) and argv[argv.index('--recipe') + 1] == str(_path)
    assert argv.count('--allow_physics_contract_change') == 1
    return argv, final, saved


def evaluator_tree():
    tree, path, proof = PARENT_EVALUATOR()
    original = ast.dump(tree)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    points = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'height' for t in n.targets)]
    assert len(points) == 1
    inserted = ast.parse('\n'.join('cfg.rewards.' + k + ' = ' + repr(v[1])
                                   for k, v in _refinement['reward_changes'].items())).body
    at = points[0]
    main.body[at:at] = inserted
    modified = copy.deepcopy(tree)
    del main.body[at:at + len(inserted)]
    assert ast.dump(tree) == original
    proof.update(uniform_existing_kernel_refinements=copy.deepcopy(_refinement['reward_changes']),
                 only_additional_replay_changes_are_constant_config_assignments=True,
                 original_metrics_limits_failure_logic_unchanged=True)
    return ast.fix_missing_locations(modified), path, proof


def qualify(path, output):
    import continue_saved_joint_pd_bundle as bundle
    configure(path)
    result = bundle.qualify(path, output)
    source, _, cfg = baseline()
    argv, final, _ = command(source, cfg, 'CPU_uniform_kernel_refinement50', 50,
                             cfg['commands'], output / 'NOT_A_STOP_REQUEST')
    assert final == parent.saved_updates(source) + 49
    result.update(global_existing_kernel_changes=copy.deepcopy(_refinement['reward_changes']),
                  saved_controller_and_reward_values_retained=False,
                  saved_physics_controller_commands_and_noise_retained=True,
                  source_initial_NN_std_17Adam_actualLR_preserved=True,
                  fifty_update_native_command=argv,
                  parameter_change_selected=False, training_launched=False)
    (output / 'qualification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def compare(output, through):
    return PARENT_COMPARE(output, through)


def train(request_path):
    import train_saved_joint_pd_review50 as workflow
    request = json.loads(Path(request_path).read_text())
    configure(request['recipe'])
    decision = json.loads(Path(request['selected_by_completed_parent_review']).read_text())
    assert decision['selected_for_one_native50']
    assert decision['source_checkpoint_sha256'] == parent.sha(request['source_checkpoint'])
    review = Path(decision['completed_full735_report'])
    assert parent.sha(review) == decision['completed_full735_report_sha256']
    reviewed = json.loads(review.read_text())
    assert reviewed['complete'] and len(reviewed['all_trials']) == 735
    assert reviewed['all_failures_retained']
    _, cfg = require_source(request['source_checkpoint'])
    expected = expected_configuration(cfg, 8192)
    changes = {k: [v, expected['rewards'][k]] for k, v in cfg['rewards'].items()
               if v != expected['rewards'][k]}
    assert set(changes) <= set(_refinement['reward_changes'])
    assert decision['next_uniform_reward_changes'] == changes
    original_write = workflow.write_json

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            value['inherited_ancestor_changes'] = value['next_configuration_changes']
            value['next_configuration_changes'] = dict(physics={}, controller={}, commands={}, exploration={}, rewards=changes)
            value.update(only_physics_change={}, only_controller_change={}, only_command_change={},
                         only_exploration_change={}, only_reward_change=changes, only_runtime_sampling_change={},
                         source_exploration=parent.recipe()['exploration'],
                         original_observation_and_action_maps_and_saved_controller_retained=True,
                         same_physics_controller_commands_noise_as_immediate_source=True,
                         saved_initial_std_reset=False,
                         metadata_delta_scope='immediate source; inherited ancestor changes recorded separately')
        elif Path(path).name == 'resume_verification.json':
            value = copy.deepcopy(value)
            value.update(calibrated_asset_and_full_height_unchanged=True,
                         physical_control_contract_unchanged=True,
                         declared_mass_gravity_and_common_wheel_gain_transition_verified=False,
                         declared_physics_changes={}, declared_control_changes={})
        return original_write(path, value)

    workflow.write_json = write
    try:
        return PARENT_TRAIN(request_path)
    finally:
        workflow.write_json = original_write


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        at = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:at])
        configure(args.recipe)
        parent.native(args.observation_output, sys.argv[at + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        configure(args.recipe)
        parent.replay(remaining)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        result = qualify(args.recipe.resolve(), args.output.resolve())
        print(json.dumps(dict(qualified=result['qualified'], checkpoint_sha256=result['checkpoint_sha256'],
                              actual_ppo_updates=result['actual_ppo_updates'], training_launched=False)))
    elif mode == 'compare':
        parser.add_argument('directory', type=Path)
        parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:])
        configure(args.recipe)
        compare(args.directory.resolve(), args.through)
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        configure(args.recipe)
        train(args.request.resolve())
    else:
        raise ValueError(mode)
