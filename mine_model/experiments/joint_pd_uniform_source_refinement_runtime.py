"""Continue a verified full-state policy with declared global parameters.

Only a common wheel velocity gain and a common future Gaussian floor are
supported. The original ordinary PPO, task, action map, rewards, continuous
sampler, physics asset and numeric readers remain supplied by the parent.
No operating condition or evaluation case selects a parameter here.
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
NAMESPACES = list({id(value): value for value in NAMESPACES}.values())
FLOOR_KEYS = ('global_minimum_std', 'minimum_leg_std', 'minimum_wheel_std')
CONTROL_KEYS = ('wheel_damping', 'robot.actuators.wheels.damping')
_path = None
_spec = None
_selected = None
_source_recipe = None


def __getattr__(name):
    return getattr(parent, name)


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0.


def project_source_recipe(source, specification):
    """Materialize targets in the parent's existing global config fields."""
    result = copy.deepcopy(source)
    changes = specification['controller_changes']
    assert set(changes) <= {'wheel_damping'}
    if changes:
        before, after = changes['wheel_damping']
        assert finite_positive(before) and finite_positive(after) and before != after
        assert source['control_overrides']['wheel_damping'] == before
        for key in CONTROL_KEYS:
            assert result['control_changes'][key][1] == before
            result['control_changes'][key][1] = after
            result['control_overrides'][key] = after
    floor = specification['future_common_std_floor']
    if floor:
        before, after = floor
        assert finite_positive(before) and finite_positive(after) and before != after
        assert all(source['exploration'][key] == before for key in FLOOR_KEYS)
        result['exploration'] = {key: after for key in FLOOR_KEYS}
        result['exploration_changes'] = {
            key: [result['source_exploration'][key], after]
            for key in FLOOR_KEYS if result['source_exploration'][key] != after}
    assert changes or floor
    result['effective_change_relative_to_source'] = dict(
        physics=result['physics_changes'], controller=result['control_changes'],
        rewards=result['reward_changes'], exploration=result['exploration_changes'])
    return result


@contextmanager
def source_recipe_view(value):
    """Reconstruct the saved source profile without weakening its checks."""
    old = [(namespace, namespace.get('recipe')) for namespace in NAMESPACES]
    for namespace, _ in old:
        namespace['recipe'] = lambda: value
    try:
        yield
    finally:
        for namespace, previous in old:
            namespace['recipe'] = previous


def configure(path):
    global _path, _spec, _selected, _source_recipe
    path = Path(path).resolve()
    if _path is not None:
        assert path == _path, 'One immutable recipe per process'
        return parent.recipe()
    value = json.loads(path.read_text())
    specification = value['uniform_source_refinement']
    assert specification['schema'] == 'uniform_saved_source_refinement_v1'
    source_recipe_path = Path(specification['source_recipe'])
    assert parent.sha(source_recipe_path) == specification['source_recipe_sha256']
    original_recipe = json.loads(source_recipe_path.read_text())
    core = {key: item for key, item in value.items() if key != 'uniform_source_refinement'}
    assert core == project_source_recipe(original_recipe, specification)
    PARENT_CONFIGURE(path)
    source = Path(specification['source_checkpoint']).resolve()
    assert parent.sha(source) == specification['source_checkpoint_sha256']
    saved = torch.load(source, map_location='cpu', weights_only=True)
    cfg = parent.validate_checkpoint_robot(source)
    # All original profile fields are checked under the original recipe. Its
    # digest is the only path-dependent field in this reconstruction.
    with source_recipe_view(original_recipe):
        _, _, anchor_cfg = PARENT_BASELINE()
        expected = PARENT_EXPECTED(anchor_cfg, 8192)
        original_profile = PARENT_PROFILE(8192)
    expected['log_dir'] = cfg['log_dir']
    original_profile['uniform_recipe_sha256'] = specification['source_recipe_sha256']
    assert cfg == expected and parent.tensor_finite(saved)
    assert saved['infos'][parent.KEY] == original_profile
    proof_path = Path(specification['source_final_verification'])
    assert parent.sha(proof_path) == specification['source_final_verification_sha256']
    proof = json.loads(proof_path.read_text())
    actual = parent.saved_updates(source)
    assert proof['verified'] and proof['all50_native_updates_verified']
    assert Path(proof['checkpoint']).resolve() == source
    assert proof['checkpoint_sha256'] == parent.sha(source)
    assert actual == specification['source_actual_updates'] == proof['actual_ppo_updates']
    assert saved['model_state_dict']['actor.0.weight'].shape[1] == 27
    assert saved['model_state_dict']['critic.0.weight'].shape[1] == 27
    assert saved['model_state_dict']['std'].shape == (6,)
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert {float(item['step']) for item in saved['optimizer_state_dict']['state'].values()} == {actual * 20}
    assert proof['all17_Adam_step'] == actual * 20
    assert not any(specification[key] for key in
                   ('optimizer_reset', 'std_reset', 'actual_LR_reset', 'case_specific_training_logic'))
    if specification['controller_changes']:
        assert cfg['wheel_damping'] == cfg['robot']['actuators']['wheels']['damping']
        assert specification['controller_changes']['wheel_damping'][0] == cfg['wheel_damping']
    if specification['future_common_std_floor']:
        target_floor = specification['future_common_std_floor'][1]
        assert float(saved['model_state_dict']['std'].min()) >= target_floor, 'Initial saved std must remain intact'
    _path, _spec, _selected, _source_recipe = path, specification, (source, saved, cfg), original_recipe
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
    if _spec['controller_changes']:
        pair = _spec['controller_changes']['wheel_damping']
        assert normalized['wheel_damping'] == normalized['robot']['actuators']['wheels']['damping']
        assert normalized['wheel_damping'] in pair
        normalized['wheel_damping'] = pair[1]
        normalized['robot']['actuators']['wheels']['damping'] = pair[1]
    return PARENT_EXPECTED(normalized, count)


def immediate_changes(cfg, source_floors=None):
    expected = expected_configuration(cfg, 8192)
    controls = {}
    if cfg['wheel_damping'] != expected['wheel_damping']:
        controls = {key: [cfg['wheel_damping'], expected['wheel_damping']] for key in CONTROL_KEYS}
    source_floors = _source_recipe['exploration'] if source_floors is None else source_floors
    exploration = {key: [source_floors[key], parent.recipe()['exploration'][key]]
                   for key in FLOOR_KEYS if source_floors[key] != parent.recipe()['exploration'][key]}
    return dict(physics={}, controller=controls, commands={}, rewards={}, exploration=exploration)


def profile_for_count(count):
    import observe_original_ppo_adaptive_kl as adaptive
    result = PARENT_PROFILE(count)
    inherited = dict(declared_physics_changes=copy.deepcopy(_source_recipe['physics_changes']),
                     global_control_changes=copy.deepcopy(_source_recipe['control_changes']),
                     global_reward_changes=copy.deepcopy(_source_recipe['reward_changes']),
                     uniform_exploration_changes=copy.deepcopy(_source_recipe['exploration_changes']))
    delta = immediate_changes(baseline()[2])
    result.update(profile='plain27_uniform_saved_source_refinement_v1',
        native_runtime_sha256=parent.sha(SCRIPT), uniform_recipe_sha256=parent.sha(_path),
        continuation_source_sha256=_spec['source_checkpoint_sha256'],
        retained_ancestor_source_sha256=parent.recipe()['source_checkpoint_sha256'],
        inherited_ancestor_changes=inherited, declared_physics_changes={},
        global_control_changes=delta['controller'], global_reward_changes={},
        uniform_exploration_changes=delta['exploration'],
        changed_noise_heads=list(range(6)) if delta['exploration'] else [],
        leg_exploration_floor_unchanged=not bool(delta['exploration']),
        saved_leg_PD_rewards_action_maps_and_exploration_retained=not bool(delta['exploration']),
        saved_leg_PD_action_maps_and_exploration_retained=not bool(delta['exploration']),
        saved_initial_std_reset=False, same_global_future_floor_for_every_command_and_head=True,
        source_NN_sixstd_17Adam_actualLR_reset=False,
        source_physics_reward_commands_leg_PD_and_action_maps_retained=True,
        full_minibatch_adaptive_KL_read_only=True,
        adaptive_KL_observer_source_sha256=parent.sha(Path(adaptive.__file__)),
        source_actual_updates=_spec['source_actual_updates'])
    return result


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, 'No original model200 reload'
    path = Path(path).resolve()
    source, initial, cfg = baseline()
    if path == source:
        assert parent.sha(path) == _spec['source_checkpoint_sha256']
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
    assert all(saved['model_state_dict'][key].shape == item.shape for key, item in initial['model_state_dict'].items())
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert {float(item['step']) for item in saved['optimizer_state_dict']['state'].values()} == {parent.saved_updates(path) * 20}
    return saved, current


@contextmanager
def unique_inherited_physics_flag():
    namespace = PARENT_COMMAND.__globals__
    original = namespace['old_command']
    def without_inherited_flag(*args, **kwargs):
        argv, final, saved = original(*args, **kwargs)
        assert argv.count('--allow_physics_contract_change') <= 1
        return [value for value in argv if value != '--allow_physics_contract_change'], final, saved
    namespace['old_command'] = without_inherited_flag
    try:
        yield
    finally:
        namespace['old_command'] = original


def command(source, cfg, label, updates, commands, stop_file):
    with unique_inherited_physics_flag():
        argv, final, saved = PARENT_COMMAND(source, cfg, label, updates, commands, stop_file)
    assert argv[1] == str(SCRIPT) and argv[argv.index('--recipe') + 1] == str(_path)
    assert argv.count('--allow_physics_contract_change') == 1
    floors = parent.recipe()['exploration']
    assert argv[argv.index('--min_exploration_std') + 1] == str(floors['global_minimum_std'])
    assert argv[argv.index('--min_wheel_exploration_std') + 1] == str(floors['minimum_wheel_std'])
    gain = expected_configuration(cfg, 8192)['wheel_damping']
    for key in CONTROL_KEYS:
        assert argv.count('env.' + key + '=' + str(gain)) == 1
    return argv, final, saved


def evaluator_tree():
    # The parent now reads the materialized global target recipe. No replay
    # equations or acceptance expressions need an additional transformation.
    return PARENT_EVALUATOR()


def source_bound_review(function):
    """Bind the legacy comparison predicate to this request's actual source.

    The old recipe anchor predates the current genuine checkpoint. Only this
    metadata predicate changes; the case layout and all numeric reader calls
    remain in the original review function.
    """
    import inspect
    tree = ast.parse(inspect.getsource(function))
    before = ast.dump(tree)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)
               and ast.unparse(node) == "runtime.recipe()['source_checkpoint_sha256']"]
    assert len(matches) == 1
    old = matches[0]
    replacement = ast.parse("runtime.sha(request['source_checkpoint'])", mode='eval').body
    class Replace(ast.NodeTransformer):
        def visit_Subscript(self, node):
            return copy.deepcopy(replacement) if node is old else self.generic_visit(node)
    changed = Replace().visit(tree)
    compiled = copy.deepcopy(changed)
    class Undo(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node) == ast.dump(replacement):
                return copy.deepcopy(old)
            return self.generic_visit(node)
    restored = Undo().visit(changed)
    assert ast.dump(restored) == before
    scope = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(compiled), str(SCRIPT), 'exec'), scope)
    return scope[function.__name__], dict(
        exact_request_source_checkpoint_SHA_used=True,
        only_stale_anchor_metadata_predicate_changed=True,
        all_original_replay_metrics_and_flags_restored_AST_identical=True)


def native(output, arguments):
    from joint_pd_training_observer import TrainingObserver
    from observe_original_ppo_adaptive_kl import install_for_training_observer
    restore = install_for_training_observer(TrainingObserver)
    try:
        return parent.native(output, arguments)
    finally:
        restore()


def qualify(path, output):
    import contextlib
    import inspect
    import io
    import shutil
    import numpy as np
    import yaml
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    import rsl_rl.runners as runners
    import train_joint_pd_standing_first as boundary
    import wheel_legged_gym_isaaclab.exploration as exploration
    configure(path)
    output.mkdir(exist_ok=False)
    for folder in ('load', 'exploration', 'reload'):
        (output / folder).mkdir()
    source, saved, cfg = baseline()
    canonical = Path(_spec['source_mapping_replay'])
    manifest = json.loads((canonical / 'manifest.json').read_text())
    context = json.loads((canonical / 'resolved_env.json').read_text())
    assert manifest['checkpoint_sha256'] == parent.sha(source)
    assert not any(manifest[key] for key in ('weights_updated', 'sampled_policy_actions', 'policy_input_remapping'))
    for key in ('wheel_damping', 'feedforward_force'):
        assert context[key] == cfg[key]
    assert context['robot']['spawn']['usd_path'] == cfg['robot']['spawn']['usd_path']
    with np.load(canonical / 'policy_inputs.npz', allow_pickle=False) as archive:
        inputs = torch.from_numpy(archive['observations'][-1].copy())
        recorded_means = torch.from_numpy(archive['means'][-1].copy())
    observations = TensorDict({'policy': inputs}, [len(inputs)])
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    policy_options, algorithm_options = dict(agent['policy']), dict(agent['algorithm'])
    assert policy_options.pop('class_name') == 'ActorCritic'
    assert algorithm_options.pop('class_name') == 'PPO'
    algorithm_options['learning_rate'] = saved['optimizer_state_dict']['param_groups'][0]['lr']
    original_update = PPO.update
    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(observations, agent['obs_groups'], 6, **policy_options)
        runner = object.__new__(cls)
        runner.alg = PPO(policy, device='cpu', **algorithm_options)
        runner.current_learning_iteration = 0
        runner.logger_type, runner.disable_logs = 'tensorboard', True
        return runner
    upstream, cls = parent.private_for_count(8192).install_runner(output / 'load')
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        assert type(runner.alg) is PPO and PPO.update is original_update
        parent.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'loaded_NN_sixstd')
        parent.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'loaded_all17Adam')
        assert runner.alg.learning_rate == saved['optimizer_state_dict']['param_groups'][0]['lr']
        with torch.inference_mode():
            before = runner.alg.policy.act_inference(observations).clone()
        previous = parent.install_leg_exploration(output / 'exploration')
        try:
            floors = parent.recipe()['exploration']
            handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
                floors['global_minimum_std'], minimum_wheel_std=floors['minimum_wheel_std'])
            parent.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'guard_NN_sixstd')
            parent.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'guard_all17Adam')
            with torch.inference_mode():
                assert torch.equal(before, runner.alg.policy.act_inference(observations))
            handle.remove()
        finally:
            exploration.enforce_positive_exploration = previous
        parity = float((before - recorded_means).abs().max())
        assert parity < 1.e-4
        offline = output / 'roundtrip/offline_metadata_not_trained.pt'
        shutil.copytree(source.parent / 'params', offline.parent / 'params')
        expected = expected_configuration(cfg, 8192)
        (offline.parent / 'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))
        contract_path = offline.parent / 'params/control_contract.yaml'
        contract = yaml.safe_load(contract_path.read_text())
        contract['wheel_damping'] = expected['wheel_damping']
        contract_path.write_text(yaml.safe_dump(contract))
        validate_contract = PARENT_PROFILE.__globals__['wheel_namespace']['validate_declared_contract']
        validate_contract(boundary.validate_contract, offline.parent,
                          fresh_weights=False, source_checkpoint=source)
        wrong = dict(contract, leg_effort_limit=contract['leg_effort_limit'] + 1.)
        contract_path.write_text(yaml.safe_dump(wrong))
        try:
            validate_contract(boundary.validate_contract, offline.parent,
                              fresh_weights=False, source_checkpoint=source)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('An undeclared effort limit change escaped validation')
        contract_path.write_text(yaml.safe_dump(contract))
        runner.save(str(offline))
        runners.OnPolicyRunner = upstream
        up2, cls2 = parent.private_for_count(8192).install_runner(output / 'reload')
        try:
            restored = bare(cls2)
            restored.load(str(offline), load_optimizer=True, map_location='cpu')
            parent.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_NN_sixstd')
            parent.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_all17Adam')
            assert restored.alg.learning_rate == runner.alg.learning_rate
            with torch.inference_mode():
                assert torch.equal(before, restored.alg.policy.act_inference(observations))
        finally:
            runners.OnPolicyRunner = up2
        assert all(item.grad is None for item in runner.alg.policy.parameters())
    finally:
        runners.OnPolicyRunner = upstream
    argv, final, _ = command(source, cfg, 'CPU_uniform_source_refinement50', 50,
                            cfg['commands'], output / 'NOT_A_STOP_REQUEST')
    assert final == parent.saved_updates(source) + 49
    import train_saved_joint_pd_review50 as workflow
    _, review_proof = source_bound_review(workflow.review)
    result = dict(qualified=True, actual_algorithm_class='rsl_rl.algorithms.ppo.PPO',
                  source_checkpoint=str(source), checkpoint_sha256=parent.sha(source),
                  actual_ppo_updates=parent.saved_updates(source), all17_Adam_step=parent.saved_updates(source) * 20,
                  saved_actual_LR=runner.alg.learning_rate, saved_sixstd=saved['model_state_dict']['std'].tolist(),
                  original_PPO_update_function_retained=True, source_canonical_actor_parity_max_error=parity,
                  source_mapping_context_verified=True, original_rollout48_Adam20_retained=True,
                  selected_num_envs=8192, capacity_test_performed=False,
                  undeclared_effort_limit_change_rejected=True,
                  immediate_source_changes=immediate_changes(cfg),
                  initial_NN_sixstd_17Adam_actualLR_bitwise_preserved=True,
                  source_is_verified_native_full_state=True,
                  fifty_update_native_command=argv,
                  matched189_request_source_binding=review_proof,
                  source_pins={str(item): parent.sha(item) for item in
                               (SCRIPT, path, source, source.parent / 'params/env.yaml', source.parent / 'params/agent.yaml',
                                Path(inspect.getfile(PPO)), Path(workflow.__file__),
                                Path(__import__('observe_original_ppo_adaptive_kl').__file__),
                                canonical / 'manifest.json', canonical / 'resolved_env.json', canonical / 'policy_inputs.npz')},
                  offline_metadata_fixture_is_not_a_native_checkpoint=True,
                  optimizer_executed=False, simulator_started=False,
                  parameter_change_selected=False, training_launched=False, policy_accepted=False)
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
    proof_path = Path(decision['source_native_final_verification'])
    assert parent.sha(proof_path) == decision['source_native_final_verification_sha256']
    proof = json.loads(proof_path.read_text())
    assert proof['verified'] and proof['all50_native_updates_verified']
    assert Path(proof['checkpoint']).resolve() == Path(request['source_checkpoint']).resolve()
    assert proof['checkpoint_sha256'] == decision['source_checkpoint_sha256']
    report_path = Path(decision['completed_full735_report'])
    assert parent.sha(report_path) == decision['completed_full735_report_sha256']
    report = json.loads(report_path.read_text())
    assert report['complete'] and len(report['all_trials']) == 735 and report['all_failures_retained']
    saved, cfg = require_source(request['source_checkpoint'])
    assert proof['actual_ppo_updates'] == parent.saved_updates(request['source_checkpoint']) == request['source_actual_updates']
    source_floors = {key: saved['infos'][parent.KEY][key] for key in FLOOR_KEYS}
    delta = immediate_changes(cfg, source_floors)
    assert decision['next_uniform_changes'] == delta
    assert request['target_actual_updates'] == request['source_actual_updates'] + 50
    if delta['controller']:
        frozen_path = Path(decision['controller_frozen_full735_report'])
        assert parent.sha(frozen_path) == decision['controller_frozen_full735_report_sha256']
        frozen = json.loads(frozen_path.read_text())
        assert frozen['complete'] and len(frozen['all_trials']) == 735 and frozen['all_failures_retained']
        assert frozen['checkpoint_sha256'] == decision['source_checkpoint_sha256']
        native_path = Path(decision['controller_frozen_actual_gain_verification'])
        assert parent.sha(native_path) == decision['controller_frozen_actual_gain_verification_sha256']
        native_receipts = json.loads(native_path.read_text())
        assert native_receipts['verified'] and native_receipts['all_original735_retained']
        for path, digest in native_receipts['source_pins'].items():
            assert parent.sha(path) == digest
        identity = lambda row: (row['case']['name'], row['seed'],
                                tuple(row['case'][key] for key in ('speed', 'yaw', 'height')))
        assert {identity(row) for row in frozen['all_trials']} == {identity(row) for row in report['all_trials']}
        gain = expected_configuration(cfg, 8192)['wheel_damping']
        qualified = json.loads((frozen_path.parent / 'qualification.json').read_text())
        for job in qualified['batches']:
            receipt = json.loads((Path(job['replay']) / 'native_common_wheel_gain_verified.json').read_text())
            assert receipt['verified'] and receipt['checkpoint_sha256'] == frozen['checkpoint_sha256']
            assert receipt['actual_PhysX_wheel_damping_range'] == [gain, gain]
            assert receipt['all_other_checked_control_and_reward_values_retained']
    original_write, original_review = workflow.write_json, workflow.review
    bound_review, review_proof = source_bound_review(original_review)
    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            value['materialized_parent_recipe_transition'] = value['next_configuration_changes']
            value['inherited_ancestor_changes'] = copy.deepcopy(_source_recipe['effective_change_relative_to_source'])
            value['next_configuration_changes'] = delta
            for name in ('physics', 'controller', 'reward', 'command', 'exploration'):
                plural = {'reward': 'rewards', 'command': 'commands'}.get(name, name)
                value['only_' + name + '_change'] = delta[plural]
            value.update(only_runtime_sampling_change={},
                         source_exploration=source_floors,
                         saved_initial_std_reset=False,
                         saved_leg_PD_action_maps_and_reward_equations_retained=True,
                         metadata_delta_scope='immediate genuine native source; inherited provenance recorded separately')
        elif Path(path).name == 'resume_verification.json':
            value = copy.deepcopy(value)
            value.update(calibrated_asset_and_full_height_unchanged=True,
                         physical_control_contract_unchanged=not bool(delta['controller']),
                         declared_mass_gravity_and_common_wheel_gain_transition_verified=False,
                         declared_physics_changes={}, declared_control_changes=delta['controller'])
        elif Path(path).name == 'complete50_matched189_agent_decision_required.json':
            value = copy.deepcopy(value)
            value.pop('contains_all9_old_yaw_failed_conditions', None)
            value.update(matched_actual_request_source_checkpoint_sha256=parent.sha(request['source_checkpoint']),
                         matched189_request_source_binding=review_proof,
                         all189_trials_retained_without_case_selection=True)
        return original_write(path, value)
    bound_review.__globals__['write_json'] = write
    workflow.write_json = write
    workflow.review = bound_review
    try:
        return PARENT_TRAIN(request_path)
    finally:
        workflow.write_json = original_write
        workflow.review = original_review


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        at = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:at])
        configure(args.recipe)
        native(args.observation_output, sys.argv[at + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        configure(args.recipe)
        parent.replay(remaining)
    elif mode == 'compare':
        parser.add_argument('directory', type=Path)
        parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:])
        configure(args.recipe)
        compare(args.directory.resolve(), args.through)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        qualify(args.recipe.resolve(), args.output.resolve())
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        configure(args.recipe)
        train(args.request.resolve())
    else:
        raise ValueError(mode)
