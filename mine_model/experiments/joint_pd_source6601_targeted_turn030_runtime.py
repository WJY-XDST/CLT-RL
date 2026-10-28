"""Gated source6601 sampler trial; no admission from CPU-only evidence.

Preserve the ordinary27-input PPO, every Adam state, actual LR, trainable std,
control and full command envelope. A separately selected trial may supplement
30% of spin/turn resamples with joint command targets. Fixed replays bypass it.
"""
import ast
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import plain27_sampler_metadata_adapter as adapter
import joint_pd_source6401_yaw200_runtime as previous
import targeted_turn_command_sampler as sampler

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261021_010831_406206_source6401_yaw200_200'
DECISION = RESULTS / 'joint_pd_source6601_targeted_turn030_trial_decision_v3_20261021.json'
ANCHOR_SHA = '53b4d9c374ac514014d5a43e84cc364f2ca30dac38b110f53549fd73b7e87d6a'
FRACTION = .3
FLAG = '--repair_targeted_turn030'
KEY, base, criteria_fix, bind = previous.KEY, previous.base, previous.criteria_fix, previous.bind
DELTA = dict(rewards={}, commands={sampler.CONFIG_KEY: [None, FRACTION]},
             exploration={}, controller={}, physics={})
PROTOTYPE = TRAINING / 'launch.json'


@lru_cache(maxsize=1)
def anchor():
    final = json.loads((TRAINING / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    source = Path(final['checkpoint'])
    if (not final['verified'] or final['actual_ppo_updates'] != 6601
            or final['all17_Adam_step'] != 132020 or sha(source) != ANCHOR_SHA):
        raise ValueError('Require exact6601 and all17 retained Adam clocks')
    saved, cfg = previous.require_source(source)
    if (sampler.CONFIG_KEY in cfg['commands'] or cfg['commands']['training_motion_scale'] != 1.
            or cfg['rewards']['tracking_ang_vel'] != 2. or cfg['rewards']['tracking_lin_vel'] != 3.
            or cfg['rewards']['orientation'] != -60. or cfg['wheel_action_limit'] != 1.25):
        raise ValueError('Keep the exact reviewed full-envelope angular2 source')
    return source, saved, cfg


def profile():
    value = copy.deepcopy(previous.profile())
    value.update(profile='plain_source6601_targeted_turn030_v1',
        native_runtime_sha256=sha(SCRIPT), parent_yaw200_runtime_sha256=sha(previous.SCRIPT),
        targeted_turn_sampler_sha256=sha(sampler.__file__),
        targeted_turn_command_fraction=FRACTION, targeted_sampler_trial_anchor_sha256=ANCHOR_SHA,
        authority='own6601_complete735_static_120s_signedfullspeed_height_reviews_and_selected_trial')
    return value


def recipe(rewards):
    return previous.recipe(rewards)


def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Originalmodel200 and optimizer resets are forbidden')
    _, before, cfg = anchor()
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return previous.require_source(path)
    if transition:
        raise ValueError('Only exact6601 may start this sampler transfer')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = copy.deepcopy(cfg)
    expected['commands'][sampler.CONFIG_KEY] = FRACTION
    expected['log_dir'] = current.get('log_dir')
    states = saved['optimizer_state_dict']['state']
    infos = saved.get('infos')
    declared = infos.get(KEY) if isinstance(infos, dict) else None
    actual = saved_updates(path)
    if (not tensor_finite(saved) or current != expected or declared != profile()
            or actual < 6601 or len(states) != 17
            or any(float(state['step']) != actual * 20 for state in states.values())
            or saved['model_state_dict'].keys() != before['model_state_dict'].keys()
            or any(saved['model_state_dict'][key].shape != value.shape
                   for key, value in before['model_state_dict'].items())):
        raise ValueError('Require the exact one-command-field descendant and complete17 Adam states')
    return saved, current


def repair_ready():
    decision = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(decision[key]).read_text())
    full = read('source_full_review')
    static = read('source_static_raw_manifest')
    sustained = read('source_original_sustained120')
    extra = read('source_fullspeed_height_transition_review')
    cpu = read('sampler_CPU_qualification')
    modes = extra['results']['candidate6601']['modes']
    if (not decision['selected_targeted_turn_sampler_trial']
            or decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 6601 or decision['source_all17_Adam_step'] != 132020
            or decision['planned_actual_updates'] != 200 or decision['target_actual_updates'] != 6801
            or decision['next_configuration_changes'] != DELTA
            or not full['all_original_and_retained_reviews_complete'] or full['actual_updates'] != 6601
            or not static['raw_integrity_verified'] or not static['static_retention_verified']
            or not sustained['raw_integrity_verified'] or sustained['survivors'] != 54
            or sustained['raw_OR_resets'] or not extra['completed'] or len(modes) != 2
            or any(mode['survivors'] != 27 or mode['resets'] for mode in modes)
            or not cpu['qualified_cpu_only'] or cpu['provisional_fraction'] != FRACTION
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or not decision['all_failures_and_regressions_retained']
            or any(sha(path) != digest for path, digest in decision['source_pins'].items())):
        raise ValueError('Complete all own6601 reviews and explicitly select one sampler field first')
    return decision


# Keep every existing plain-PPO/physics/wheel guard. Build from the actual
# uncompiled factory source, then retarget only clock and evidence metadata.
_provider = adapter.private_module(previous.previous.SCRIPT, '_source6601_targeted_sampler_guard_provider')
FACTORY_METADATA = adapter.retarget(_provider, ('private_runtime',),
    integers={'*': {6201: 6601}},
    strings={'lin300_runtime_evidence.json': 'targeted_turn_base_invariants.json'},
    keyword_names={'source_actual_updates': 'trial_anchor_actual_updates',
        'height_yaw_command_exploration_physics_reset_and_direct_joint_execution_unchanged':
        'inherited_range_exploration_physics_reset_and_direct_joint_checks_before_sampler'})
for _name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'profile', 'recipe', 'require_source', 'repair_ready'):
    setattr(_provider, _name, globals()[_name])


def _configure(cfg):
    old = getattr(cfg.commands, sampler.CONFIG_KEY, None)
    if old is not None and old != FRACTION:
        raise ValueError('Never silently overwrite another sampling fraction')
    setattr(cfg.commands, sampler.CONFIG_KEY, FRACTION)


def install_sampling_field_restore():
    """Keep strict saved-config restoration for this declared training field."""
    from wheel_legged_gym_isaaclab import config_overrides
    original = config_overrides.apply_config_overrides
    if getattr(original, '_targeted_turn_field_restore', False):
        return

    def apply(config, overrides):
        key = 'commands.' + sampler.CONFIG_KEY
        if key in overrides:
            if overrides[key] != FRACTION:
                raise ValueError('Saved sampler fraction differs from this qualified runtime')
            _configure(config)
        return original(config, overrides)
    apply._targeted_turn_field_restore = True
    config_overrides.apply_config_overrides = apply


def _wrap_make(original):
    def make(*args, **kwargs):
        task = args[0] if args else kwargs.get('id')
        if task == 'WheelLeggedJointPD-Flat-v0':
            if 'cfg' not in kwargs:
                raise ValueError('Require the explicit qualified environment configuration')
            _configure(kwargs['cfg'])
        return original(*args, **kwargs)
    return make


def private_runtime(source=None):
    module = _provider.private_runtime(source)
    inherited_observe = module.observe_environment
    inherited_install = module.install_environment_observer

    def observe(env, output):
        result = inherited_observe(env, output)
        raw = env.unwrapped
        if getattr(raw.cfg.commands, sampler.CONFIG_KEY, None) != FRACTION:
            raise ValueError('Sampling configuration must be installed before environment construction')
        sampler.install(raw)
        original_step = raw._pre_physics_step
        count = 0

        def pre_step(actions):
            nonlocal count
            original_step(actions)
            count += 1
            if count % 48 == 0:
                fixed = float(raw._command_ranges[0, 0]) == float(raw._command_ranges[0, 1])
                if not fixed:
                    standing = raw._dedicated_standing_count
                    low, high = raw.cfg.commands.ranges_height
                    if (standing != 1024 or raw.num_envs != 4096
                            or torch.any(raw._target_lin_vel_x[:standing])
                            or torch.any(raw._commands[:standing, 1])
                            or torch.any(raw._target_lin_vel_x.abs() > .8 + 1.e-6)
                            or torch.any(raw._commands[:, 1].abs() > .3 + 1.e-6)
                            or torch.any(raw._commands[:, 2] < low - 1.e-6)
                            or torch.any(raw._commands[:, 2] > high + 1.e-6)):
                        raise ValueError('Original fixed standing IDs or full velocity/yaw bounds changed')
                write_json(Path(output) / 'targeted_turn_sampler_live_status.json', dict(
                    policy_steps=count, training_sampler_active=not fixed,
                    resample_calls=raw._targeted_turn_sampler_calls,
                    selected_target_rows=raw._targeted_turn_sampler_selected,
                    fraction=FRACTION, sampler_sha256=sha(sampler.__file__),
                    applied_speed_slew_buffer_preserved=True, auxiliary_loss_enabled=False,
                    observer_changes_actions_or_rewards=False, policy_accepted=False))
        raw._pre_physics_step = pre_step
        write_json(Path(output) / 'targeted_turn_runtime_evidence.json', dict(
            profile=profile(), authoritative_current_delta=DELTA,
            trial_anchor_actual_updates=6601, sampling_fraction=FRACTION,
            sampler_sha256=sha(sampler.__file__), factory_metadata_adapter=FACTORY_METADATA,
            fixed_replay_sampler_bypassed=True, original_actor_observation_count=27,
            direct_joint_angle_targets=True, IK_or_VMC_feedback_added=False,
            reward_exploration_physics_reset_and_control_unchanged=True,
            sampler_RNG_draws_intentionally_added_only_for_mature_nonfixed_training=True,
            saved_std_and_all17_Adam_not_reset=True, policy_accepted=False))
        return result

    def install_environment_observer(output):
        inherited_install(output)
        import gymnasium as gym
        gym.make = _wrap_make(gym.make)

    module.profile, module.require_source, module.observe_environment = profile, require_source, observe
    module.install_environment_observer = install_environment_observer
    return module


install_leg_exploration = _provider.install_leg_exploration


def evaluator_tree():
    tree, path = previous.evaluator_tree()
    expected = ast.dump(ast.parse('env = EvaluationEnv(cfg)').body[0])
    matches = 0
    for node in list(ast.walk(tree)):
        for field, value in list(ast.iter_fields(node)):
            if isinstance(value, list):
                indices = [i for i, item in enumerate(value)
                           if isinstance(item, ast.Assign) and ast.dump(item) == expected]
                if indices:
                    if len(indices) != 1 or matches:
                        raise ValueError('Requalify the direct frozen environment constructor')
                    revised = list(value)
                    revised.insert(indices[0], ast.parse('_configure_sampling(cfg)').body[0])
                    setattr(node, field, revised)
                    matches += 1
    if matches != 1:
        raise ValueError('Require exactly one pre-construction sampler configuration')
    return ast.fix_missing_locations(tree), path


def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    if (sha(source) == ANCHOR_SHA) != (FLAG in arguments):
        raise ValueError('Declare the first6601 sampler transfer exactly once')
    install_sampling_field_restore()
    template = previous.previous.previous.previous.native_template
    return bind(template.native, SCRIPT=SCRIPT, FLAG=FLAG, repair_ready=repair_ready,
        require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)(output, arguments)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    source, stop_file = Path(source).resolve(), Path(stop_file).resolve()
    saved, actual_cfg = require_source(source)
    target_commands = copy.deepcopy(actual_cfg['commands'])
    target_commands[sampler.CONFIG_KEY] = FRACTION
    if updates != 200 or cfg != actual_cfg or commands not in (actual_cfg['commands'], target_commands):
        raise ValueError('Keep exactly200 ordinary updates and the declared complete command contract')
    overrides = kwargs.pop('reward_overrides', {})
    if any(not key.startswith('rewards.') or actual_cfg['rewards'].get(key[8:]) != value
           for key, value in overrides.items()):
        raise ValueError('This sampler trial cannot change any reward')
    if (kwargs.pop('minimum_wheel_std', .01) not in (.01, .03)
            or kwargs.pop('joint_mean_guard_coef', 0.) != 0.
            or kwargs.pop('joint_height_feedback_coef', 0.) != 0. or kwargs):
        raise ValueError('Keep the original saved exploration and no auxiliary losses')
    prototype = json.loads(PROTOTYPE.read_text())
    argv = list(prototype['command'])
    if (prototype['source_sha256'] != previous.ANCHOR_SHA
            or argv[1] != str(previous.SCRIPT) or argv.count(previous.FLAG) != 1
            or argv.count('env.rewards.tracking_ang_vel=2.0') != 1
            or argv.count('agent.save_interval=1') != 1):
        raise ValueError('Requalify the exact successful ordinary200 CLI prototype')
    argv[1] = str(SCRIPT)
    argv.remove(previous.FLAG)
    actual = saved_updates(source)
    for flag, value in {'--observation-output': stop_file.parent / 'native_learning_signal',
            '--checkpoint_path': source, '--target_iteration': actual + 200,
            '--run_name': label, '--stop_file': stop_file}.items():
        argv[argv.index(flag) + 1] = str(value)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    if any(flag in argv for flag in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Never reset the retained policy, Adam or std')
    return argv, actual + 199, saved


def replay_command(source, output):
    source = Path(source).resolve()
    require_source(source)
    anchor_path, _, _ = anchor()
    argv = previous.replay_command(anchor_path, output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    argv[argv.index('--checkpoint') + 1] = str(source)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if '--overrides_json' in arguments or (sha(source) == ANCHOR_SHA) != transition:
        raise ValueError('Use only the declared sampler transfer or an exact saved descendant')
    install_sampling_field_restore()
    import gymnasium as gym
    original_make = gym.make
    gym.make = _wrap_make(original_make)
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
            _install_quality_criteria=criteria_fix.install,
            _configure_sampling=_configure,
            _observe_source200=private_runtime().observe_environment))
    finally:
        gym.make = original_make


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output', sys.argv[2]]:
            raise ValueError('Keep the existing native observation output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only frozen replay or retained-state ordinary PPO is supported')
