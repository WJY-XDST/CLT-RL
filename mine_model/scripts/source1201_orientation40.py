"""Repair measured steady posture bias without discarding source1201 height gains.

Only the existing orientation cost changes from -20 to -40. Plain27 PPO,
all17 Adam states, saved adaptive LR, exploration, physics, reset sampling,
height rewards and the completed reference-smoothing repair stay unchanged.
"""
import argparse
import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha, optimizer_step
from train_joint_pd_standing_seed import saved_updates
import source1001_reference_smooth10 as previous
import joint_pd_acceptance_10pct_20261017 as acceptance

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_source1201_orientation40_qualification_20261017'
ANCHOR_SHA = '3d6cb7844575ac1810c99e498df2cc825f4de4a34b0436a580663c7dbfb8962b'
ORIENTATION = -40.
FLAG = '--repair_steady_posture_orientation40'
base = previous.base
KEY = base.KEY


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source1201_stationary_leg05_smooth10_orientation40_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(previous.SCRIPT),
        orientation=ORIENTATION, orientation_roll_multiplier=2.,
        authority='user_standing_stability_first_preserve_height_gains_targeted_repair')
    return result


def recipe(rewards):
    return (rewards['standing_velocity'] == -50.
        and rewards['standing_wheel_tracking'] == -.1
        and rewards['action_smooth'] == -.1
        and rewards['action_smooth_wheel_multiplier'] == 5.
        and rewards['orientation'] == ORIENTATION
        and rewards['orientation_roll_multiplier'] == 2.)


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_source1201_orientation40_private', base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_require, original_observe = module.require_source, module.observe_environment
    transition_sha = None
    if source is not None:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 1201
                or cfg['rewards']['orientation'] != -20.
                or not recipe(dict(cfg['rewards'], orientation=ORIENTATION))):
            raise ValueError('Only the independently diagnosed exact1201 begins this posture repair')
        transition_sha = ANCHOR_SHA

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload original200 or another historical source for this repair')
        if transition_sha is not None and sha(path) == transition_sha:
            return previous.require_source(path)
        saved, cfg = original_require(path, initial_transfer=False)
        if not recipe(cfg['rewards']):
            raise ValueError('Preserve the qualified orientation40 recipe')
        return saved, cfg

    def observe(env, output):
        rewards = env.unwrapped.cfg.rewards
        if not recipe({name:getattr(rewards, name) for name in (
                'standing_velocity', 'standing_wheel_tracking', 'action_smooth',
                'action_smooth_wheel_multiplier', 'orientation', 'orientation_roll_multiplier')}):
            raise ValueError('Native rewards differ from the single declared posture cost change')
        result = original_observe(env, output)
        previous.install_failure_capture(env.unwrapped, output)
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    spec = importlib.util.spec_from_file_location('_source1201_unchanged_exploration', previous.previous.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.profile = profile
    return module.install_leg_exploration(output)


def native(output, arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
    source = Path(arguments[arguments.index('--checkpoint_path')+1])
    require_source(source, transition=transition)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Keep the original plain27 ActorCritic')
    private_runtime(source if transition else None).install_runner(output)
    private_runtime().install_environment_observer(output)
    install_leg_exploration(output)
    import train_joint_pd_observed as observed
    from joint_pd_review_interval200 import install_observer
    write_json(Path(output).parent/'review_interval_runtime.json', install_observer(observed))
    observed.run(output, previous.previous.tires.arguments(arguments))


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
    source = Path(arguments[arguments.index('--checkpoint')+1])
    require_source(source, transition=transition)
    sys.argv = [str(SCRIPT), *previous.previous.tires.arguments(arguments)]
    tree, original = previous.evaluator_tree()
    exec(compile(tree, str(original), 'exec'), dict(__name__='__main__', __file__=str(original),
        _observe_source200=private_runtime().observe_environment))


def command(source, *args, **kwargs):
    argv, final, saved = previous.command(source, *args, **kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


def replay_command(checkpoint, output):
    argv = previous.replay_command(checkpoint, output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    return argv


def prepare(output, previous_output):
    output, previous_output = Path(output).resolve(), Path(previous_output).resolve()
    if (output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_optimization_')
            or previous_output.parent != RESULTS):
        raise ValueError('Use a fresh owned output and preserve all prior evidence')
    state = json.loads((previous_output/'status.json').read_text())
    decision = json.loads((previous_output/'stationary_review.json').read_text())
    native_done = json.loads((previous_output/'native_learning_signal/completed_external.json').read_text())
    report = json.loads((previous_output/'replay/assessment.json').read_text())
    complete = json.loads((previous_output/'replay/completed.json').read_text())
    source = Path(state['checkpoint']).resolve()
    transition = sha(source) == ANCHOR_SHA
    saved, cfg = require_source(source, transition=transition)
    if (state['state'] != 'agent_review_required' or state['training_active']
            or not source.is_relative_to(RUNS.resolve())
            or state['checkpoint_sha256'] != sha(source) or report['checkpoint_sha256'] != sha(source)
            or native_done['checkpoint_sha256'] != sha(source)
            or native_done['actual_observed_updates'] != 200 or native_done['optimizer_minibatches'] != 4000
            or native_done['PPO_updates_repeated'] or report['total_trials'] != 27
            or not complete['completed'] or complete['samples'] != 27000
            or report['acceptance_revision'] != acceptance.profile()):
        raise ValueError('Require the exact completed200-update full27 boundary and unchanged10% criteria')
    diagnosis_file = previous_output/'stationary_orientation40_repair_diagnosis.json'
    if transition:
        diagnosis = json.loads(diagnosis_file.read_text())
        if (not diagnosis['qualified_for_single_orientation40_repair']
                or diagnosis['checkpoint_sha256'] != ANCHOR_SHA
                or diagnosis['proposed_reward_change'] != {'orientation':[-20., ORIENTATION]}
                or not diagnosis['all27_survive'] or diagnosis['resets']
                or not diagnosis['all_static_body_height_translation_yaw_and_wheel_limits_pass10']
                or not diagnosis['measured_steady_posture_bias']
                or decision['stationary_height_regressions']
                or set(decision['stationary_stability_regressions']) != {
                    'stand/83/body_p95_deg', 'stand/84/body_p95_deg', 'stand/85/body_p95_deg'}):
            raise ValueError('Require the measured surviving steady-posture and low-leg repair diagnosis')
        if any(sha(path) != digest for path,digest in diagnosis['source_files'].items()):
            raise ValueError('The measured repair evidence changed')
    elif not decision['continue_from_candidate']:
        raise ValueError('Review and diagnose a failed descendant before further training')
    qualification = json.loads((QUALIFICATION/'qualification.json').read_text())
    frozen = json.loads((QUALIFICATION/'native_reward_transfer_qualification.json').read_text())
    if (not qualification['qualified'] or qualification['profile'] != profile()
            or qualification['implementation_sha256'] != sha(SCRIPT)
            or not frozen['qualified'] or frozen['source_sha256'] != ANCHOR_SHA
            or frozen['implementation_sha256'] != sha(SCRIPT)
            or frozen['only_reward_change'] != {'orientation':[-20., ORIENTATION]}):
        raise ValueError('Require real retained-state and frozen native reward qualifications')
    old = json.loads((previous_output/'experiment_plan.json').read_text())
    pins = dict(old['source_files'])
    if any(sha(name) != digest for name,digest in pins.items()):
        raise ValueError('A frozen source or prior evidence changed')
    expected = copy.deepcopy(cfg)
    expected['rewards']['orientation'] = ORIENTATION
    previous.previous_controller.helper.base.validate_continuation_config(cfg, expected, cfg['commands'],
        reward_overrides={'rewards.orientation':ORIENTATION})
    if cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096:
        raise ValueError('Keep the actual4096 sampling configuration')
    watched = [SCRIPT, source, QUALIFICATION/'qualification.json',
        QUALIFICATION/'native_reward_transfer_qualification.json', acceptance.CURRENT, acceptance.POLICY,
        Path(acceptance.__file__).resolve(), previous_output/'status.json', previous_output/'stationary_review.json',
        previous_output/'native_learning_signal/completed_external.json',
        *sorted((source.parent/'params').glob('*.yaml')),
        *(previous_output/'replay'/name for name in ('assessment.json','manifest.json','completed.json','trace.csv'))]
    if transition:
        watched.extend([diagnosis_file, *(Path(p) for p in diagnosis['source_files'])])
    pins.update({str(path):sha(path) for path in watched})
    peak, requirement = previous.previous_controller.memory.memory_requirement()
    actual = saved_updates(source)
    plan = dict(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous_output),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=actual,
        source_Adam_step=optimizer_step(saved), source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous_output/'replay'),
        protected_reference=str(previous_output/'replay/assessment.json'), source_files=pins,
        target_actual_ppo_updates=actual+200, bounded_training_updates=200,
        expected_terminal_Adam_step=(actual+200)*20,
        run_name='plain_leg05_smooth10_orientation40_stationary_'+output.name+'_u'+str(actual),
        reward_key='orientation', reward_value=ORIENTATION,
        only_reward_change={'orientation':[-20., ORIENTATION]} if transition else {},
        only_exploration_change={}, training_motion_scale=0., native_num_envs=4096,
        native_transition_batch=4096*48, optimizer_minibatches_per_actual_update=20,
        full_height_range=cfg['commands']['ranges_height'],
        exploration_guard=dict(minimum_leg_std=.05, minimum_wheel_std=.03, global_minimum_std=.02,
                               saved_initial_std_reset=False),
        original_source200_loaded_only_at_first_block=False, external_observation_size=27,
        policy_class='ActorCritic', architecture_migrated=False, ordinary_full_policy_PPO=True,
        auxiliary_losses_enabled=False, optimizer_reset=False, learned_std_reset=False,
        acceptance_revision=acceptance.profile(), profile=profile(), training_config=expected,
        no_total_training_update_limit=True, next_block_requires_review=True,
        repair_from_unaccepted_surviving_source=transition, prior_review_retained_unmodified=True,
        acceptance_waived=False, resource_reservation_method='Measured4096 highwater plus4GiB',
        measured_native_RSS_highwater_bytes=peak, combined_memory_required_bytes=requirement,
        physical_available_memory_required_bytes=6*1024**3,
        motion_requires_static_and_independent_sustained_pass=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    output.mkdir()
    write_json(output/'experiment_plan.json', plan)
    return plan, expected


def train(output, previous_output, unit):
    helper = previous.previous_controller.helper
    helper.prepare, helper.command = prepare, command
    helper.wait_memory = previous.previous_controller.memory.wait_memory
    helper.base.acceptance = acceptance
    helper.base.replay_command = replay_command
    helper.run(output, unit, previous_output)
    proof = json.loads((Path(output)/'native_learning_signal/leg_exploration_floor_runtime.json').read_text())
    if proof['profile'] != profile() or not proof['all17_Adam_bitwise_preserved']:
        raise ValueError('The unchanged exploration guard or Adam load differed')
    previous.previous_controller.previous_acceptance.assess_current(Path(output)/'replay', 'assessment_previous20pct.json')


def main():
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        parser = argparse.ArgumentParser()
        parser.add_argument('--observation-output', type=Path, required=True)
        options = parser.parse_args(sys.argv[1:boundary])
        native(options.observation_output, sys.argv[boundary+1:])
    elif mode == 'train':
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--previous', type=Path, required=True)
        parser.add_argument('--unit')
        parser.add_argument('--prepare-only', action='store_true')
        options = parser.parse_args()
        if options.prepare_only:
            prepare(options.output, options.previous)
        else:
            train(options.output, options.previous, options.unit)
    else:
        raise ValueError('Use native, replay or train')


if __name__ == '__main__':
    main()
