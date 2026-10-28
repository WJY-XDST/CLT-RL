"""Retain AF7001 ordinary PPO while testing one wheel velocity-drive gain.

The .5 -> .75 factor has a completed paired318 frozen diagnostic. Production
still needs a fresh exact-recipe capacity admission and retained-state CPU
qualification. Rewards, six saved std entries, Adam, LR and leg PD stay fixed.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import plain27_sampler_metadata_adapter as adapter
import joint_pd_preserved6801_yawcost200_runtime as family


SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
DECISION = RESULTS / 'joint_pd_AF7001_wheelD075_trial_decision_20261023.json'
ADMISSION = RESULTS / 'joint_pd_AF7001_wheelD075_admission_20261023.json'
SOURCE_REPORT = RESULTS / 'joint_pd_optimization_20261022_135230_239804_preserved6801_yawcost200_200/final_checkpoint_verification_readonly_20261019.json'
ANCHOR_SHA = 'af47077587044230d5814f5e0e76a633366260698471da6162f3411494e64f49'
FLAG = '--repair_AF7001_wheel_driveD075'
KEY, base, criteria_fix, bind = family.KEY, family.base, family.criteria_fix, family.bind
DAMPING = .75
DELTA = dict(rewards={}, commands={}, exploration={}, physics={}, controller={
    'wheel_damping': [.5, DAMPING], 'robot.actuators.wheels.damping': [.5, DAMPING]})
NUM_ENVS = json.loads(ADMISSION.read_text())['selected_num_envs'] if ADMISSION.exists() else 4096
assert NUM_ENVS in (4096, 8192)
recipe = family.recipe


@lru_cache(maxsize=1)
def baseline():
    final = json.loads(SOURCE_REPORT.read_text())
    source = Path(final['checkpoint'])
    saved, cfg = family.require_source(source)
    assert final['verified'] and final['checkpoint_sha256'] == sha(source) == ANCHOR_SHA
    assert final['actual_ppo_updates'] == 7001 and final['all17_Adam_step'] == 140020
    assert cfg['wheel_damping'] == cfg['robot']['actuators']['wheels']['damping'] == .5
    assert cfg['rewards']['tracking_sigma_ang'] == .01 and cfg['rewards']['yaw_rate_error_sq'] == -200.
    return source, saved, cfg


def expected_configuration(cfg, count):
    expected = copy.deepcopy(cfg)
    expected['wheel_damping'] = expected['robot']['actuators']['wheels']['damping'] = DAMPING
    for key in ('scene', 'terrain'):
        expected[key]['num_envs'] = count
    return expected


def profile_for_count(count):
    assert count in (4096, 8192)
    value = copy.deepcopy(family.profile())
    value.update(profile='plain_AF7001_wheel_driveD075_v1',
        wheel_velocity_drive_damping=DAMPING,
        minimum_leg_std=.01, minimum_wheel_std=.01, global_minimum_std=.01,
        native_runtime_sha256=sha(SCRIPT), parent_runtime_sha256=sha(family.SCRIPT),
        native_training_environment_count=count, fixed_standing_environment_count=count // 4,
        samples_per_original_PPO_update=count * 48, original_Adam_minibatches_per_update=20,
        capacity_only_probe=False, authority='completed318_frozen_factor_and_fresh_exact_recipe_capacity')
    return value


def profile():
    return profile_for_count(NUM_ENVS)


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, 'Originalmodel200 and optimizer resets remain forbidden'
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return family.require_source(path)
    assert not transition, 'Only exact AF7001 begins this declared controller transfer'
    _, before, cfg = baseline()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = expected_configuration(cfg, NUM_ENVS)
    expected['log_dir'] = current['log_dir']
    actual, states = saved_updates(path), saved['optimizer_state_dict']['state']
    assert tensor_finite(saved) and current == expected and actual >= 7001
    assert saved.get('infos', {}).get(KEY) == profile()
    assert len(states) == 17 and all(float(v['step']) == actual * 20 for v in states.values())
    assert saved['model_state_dict'].keys() == before['model_state_dict'].keys()
    assert all(saved['model_state_dict'][k].shape == v.shape for k, v in before['model_state_dict'].items())
    return saved, current


def repair_ready():
    decision = json.loads(DECISION.read_text())
    assert decision['all_required_review_scopes_complete'] and decision['selected_single_wheel_drive_trial']
    assert decision['source_checkpoint_sha256'] == ANCHOR_SHA
    assert decision['source_actual_updates'] == 7001 and decision['source_all17_Adam_step'] == 140020
    assert decision['planned_actual_updates'] == 200 and decision['next_configuration_changes'] == DELTA
    assert decision['relative_error_target'] == .1
    assert not any(decision[k] for k in ('optimizer_reset', 'std_reset', 'actual_LR_reset',
                                        'original_model200_reload', 'acceptance_changed'))
    assert all(sha(path) == digest for path, digest in decision['source_pins'].items())
    for item in decision['required_complete_review_receipts']:
        assert json.loads(Path(item['path']).read_text())[item['completion_key']] is True
    return decision


def observe_drive(env, output, selected_profile):
    raw = env.unwrapped
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all()
    assert raw.cfg.wheel_damping == raw.cfg.robot.actuators['wheels'].damping == DAMPING
    assert raw.cfg.leg_control_mode == 'implicit_joint_position'
    assert raw.cfg.leg_joint_stiffness == 300. and raw.cfg.leg_joint_damping == 3.
    assert raw.cfg.rewards.to_dict() == baseline()[2]['rewards']
    assert raw.cfg.observation_space == 27 and raw.cfg.action_space == 6
    actual = raw._robot.root_physx_view.get_dof_dampings()[:, raw._wheel_joint_ids]
    assert torch.allclose(actual, torch.full_like(actual, DAMPING), rtol=0., atol=1.e-6)
    from pxr import UsdPhysics
    usd = []
    for name in raw.cfg.wheel_joint_names:
        prim = raw.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/joints/' + name)
        drive = UsdPhysics.DriveAPI(prim, 'angular')
        usd.append(drive.GetDampingAttr().Get())
        assert drive.GetTypeAttr().Get() == 'force' and abs(usd[-1] - DAMPING) < 1.e-6
        assert abs(drive.GetMaxForceAttr().Get() - raw.cfg.wheel_effort_limit) < 1.e-5
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    write_json(Path(output) / 'wheelD075_native_drive_evidence.json', dict(
        profile=selected_profile(), actual_environments=raw.num_envs,
        actual_wheel_damping_min=float(actual.min()), actual_wheel_damping_max=float(actual.max()),
        actual_USD_drive_damping=usd, authoritative_current_delta=DELTA,
        rewards_and_four_leg_PD_feedback_unchanged=True, RNG_states_unchanged=True,
        observer_changes_actions_or_rewards=False, auxiliary_loss_enabled=False, policy_accepted=False))


def private_for_count(count, selected_profile=profile):
    sampler = adapter.private_module(family.previous.SCRIPT, '_AF7001_wheelD075_retained_sampler')
    for name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'DECISION',
                 'recipe', 'require_source', 'repair_ready'):
        setattr(sampler, name, globals()[name])
    sampler.TRAINING = SOURCE_REPORT.parent
    sampler.profile = selected_profile
    def plant(source=None):
        if source is not None:
            require_source(source)
        return bind(family._plant.private_runtime, recipe=recipe, profile=selected_profile)()
    sampler._provider = SimpleNamespace(private_runtime=plant)
    evidence = adapter.retarget(sampler, ('private_runtime',),
        integers={'*': {6601: 7001, 1024: count // 4, 4096: count}},
        strings={'targeted_turn_runtime_evidence.json': 'wheelD075_retained_sampler_evidence.json'},
        keyword_names={
            'reward_exploration_physics_reset_and_control_unchanged':
                'reward_exploration_physics_reset_leg_control_and_wheel_limits_unchanged',
            'sampler_RNG_draws_intentionally_added_only_for_mature_nonfixed_training':
                'retained_sampler_RNG_draws_only_for_mature_nonfixed_training'})
    sampler.FACTORY_METADATA = family.PLANT_METADATA + evidence
    sampler.install_sampling_field_restore()
    module = sampler.private_runtime()
    original_observe = module.observe_environment
    def observe(env, output):
        result = original_observe(env, output)
        observe_drive(env, output, selected_profile)
        return result
    module.profile, module.require_source, module.observe_environment = selected_profile, require_source, observe
    module.population_assertion_evidence = evidence
    return module


def private_runtime(source=None):
    if source is not None:
        require_source(source)
    return private_for_count(NUM_ENVS)


def install_leg_exploration(output):
    return bind(family.install_leg_exploration, profile=profile)(output)


def review_ready(source):
    repair_ready()
    decision = json.loads(ADMISSION.read_text())
    assert decision['source_checkpoint_sha256'] == sha(source) == ANCHOR_SHA
    assert decision['selected_num_envs'] == NUM_ENVS and decision['next_configuration_changes'] == DELTA
    assert decision['CPU_qualified'] and decision['frozen27_qualified']
    assert decision['proposal_decision_sha256'] == sha(DECISION)
    assert all(sha(path) == digest for path, digest in decision['source_pins'].items())
    from check_plain27_wheelD075_environment_capacity import require_capacity_selection
    require_capacity_selection(decision['capacity_verified_summary'],
        block_id=decision['capacity_block_id'], source=source, num_envs=NUM_ENVS)
    return decision


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    source = Path(source).resolve()
    saved, current = require_source(source)
    assert updates == 200 and cfg == current and commands == current['commands']
    assert not kwargs.pop('reward_overrides', {})
    assert kwargs.pop('minimum_wheel_std', .01) in (.01, .03)
    assert kwargs.pop('joint_mean_guard_coef', 0.) == 0.
    assert kwargs.pop('joint_height_feedback_coef', 0.) == 0. and not kwargs
    anchor, _, before = baseline()
    argv, _, _ = family.command(anchor, before, label, 200, before['commands'], Path(stop_file))
    argv[1] = str(SCRIPT)
    for flag, value in {'--checkpoint_path': source, '--target_iteration': saved_updates(source) + 200,
                        '--num_envs': NUM_ENVS, '--observation-output': Path(stop_file).parent / 'native_learning_signal',
                        '--run_name': label, '--stop_file': Path(stop_file)}.items():
        argv[argv.index(flag) + 1] = str(value)
    assert not any(arg.startswith('env.wheel_damping=') or
                   arg.startswith('env.robot.actuators.wheels.damping=') for arg in argv)
    argv += ['env.wheel_damping=0.75', 'env.robot.actuators.wheels.damping=0.75']
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    assert not any(x in argv for x in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'))
    return argv, saved_updates(source) + 199, saved


def native(output, arguments):
    import joint_pd_source2201_yaw025_runtime as template
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    review_ready(source)
    assert int(arguments[arguments.index('--num_envs') + 1]) == NUM_ENVS
    assert int(arguments[arguments.index('--target_iteration') + 1]) == saved_updates(source) + 200
    assert (FLAG in arguments) == (sha(source) == ANCHOR_SHA)
    family._sampler_api.install_sampling_field_restore()
    return bind(template.native, SCRIPT=SCRIPT, FLAG=FLAG, repair_ready=repair_ready,
                require_source=require_source, private_runtime=private_runtime,
                install_leg_exploration=install_leg_exploration)(output, arguments)


evaluator_tree = family.evaluator_tree


def replay_command(source, output):
    require_source(source)
    if sha(source) == ANCHOR_SHA:
        # A source comparison always uses its actual saved .5 recipe.
        return family.replay_command(source, output)
    argv = family.replay_command(baseline()[0], output)
    argv[argv.index(str(family.SCRIPT))] = str(SCRIPT)
    argv[argv.index('--checkpoint') + 1] = str(Path(source).resolve())
    return argv


def frozen_transfer_command(source, output):
    assert sha(source) == ANCHOR_SHA
    argv = family.replay_command(source, output)
    argv[argv.index(str(family.SCRIPT))] = str(SCRIPT)
    return argv + [FLAG, '--overrides_json', str(DECISION.parent / 'joint_pd_AF7001_wheelD075_frozen_overrides.json')]


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if sha(source) == ANCHOR_SHA and not transition:
        return family.replay(arguments)
    if transition:
        assert json.loads(Path(arguments[arguments.index('--overrides_json') + 1]).read_text()) == {'wheel_damping': DAMPING}
    else:
        assert '--overrides_json' not in arguments
    family._sampler_api.install_sampling_field_restore()
    import gymnasium as gym
    original = gym.make
    gym.make = family._sampler_api._wrap_make(original)
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
            _install_quality_criteria=criteria_fix.install,
            _configure_sampling=family._sampler_api._configure,
            _observe_source200=private_runtime().observe_environment))
    finally:
        gym.make = original


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'native':
        boundary = sys.argv.index('--')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    elif mode == 'replay':
        replay(sys.argv[1:])
    else:
        raise ValueError('Only retained-state ordinary PPO and frozen replay')
