"""One measured reference-chatter repair, retaining the complete source1001 PPO.

The only reward delta is action_smooth0->-0.1 with the existing wheel weight5.
All physical control, height reward, exploration, model and Adam state remain.
Every native block ends at200 actual updates for independent full27 review.
"""
import argparse
import copy
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha, optimizer_step
from train_joint_pd_standing_seed import saved_updates
import train_source801_drift50_200 as previous_controller
import joint_pd_acceptance_10pct_20261017 as acceptance
from evaluate_source200_stationary_height import evaluator_tree

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_source801_drift50_runtime as previous
import joint_pd_source200_stationary_runtime as base

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_source1001_smooth10_qualification_20261017'
KEY = base.KEY
ANCHOR_SHA = 'da47356f3dc65582818f718e2a7d474f41ff1e615fbcae574c9d286493adaf11'
ACTION_SMOOTH = -.1
FLAG = '--repair_reference_chatter'


def install_failure_capture(env, output):
    """Record every unsafe environment BEFORE reset; never alter returned reward."""
    output = Path(output)
    original = env._get_rewards
    counts = dict(observed_control_steps=0, unsafe_events=0, captured_environment_ids=[],
                  read_only=True, before_automatic_reset=True, policy_accepted=False,
                  base_contact_partner_not_identified=True)

    def finite(value):
        if isinstance(value, list):
            return [finite(v) for v in value]
        return value if not isinstance(value, float) or math.isfinite(value) else None

    def reward():
        result = original()
        counts['observed_control_steps'] += 1
        ids = env.reset_terminated.nonzero(as_tuple=False).flatten()
        if ids.numel():
            data = env._robot.data
            state = dict(command=env._commands[ids,:3], height=data.root_pos_w[ids,2],
                root_position=data.root_pos_w[ids], root_quaternion=data.root_quat_w[ids],
                gravity=data.projected_gravity_b[ids], root_angular_velocity=data.root_ang_vel_b[ids],
                root_linear_velocity=data.root_lin_vel_b[ids],
                episode_time_s=env.episode_length_buf[ids]*env.step_dt,
                command_age_s=(env.common_step_counter-env._command_resample_step[ids])*env.step_dt,
                raw_action=env._raw_actions[ids], applied_action=env._actions[ids],
                previous_actions=env._last_actions[ids],
                joint_position=data.joint_pos[ids][:,env._leg_joint_ids],
                joint_velocity=data.joint_vel[ids][:,env._leg_joint_ids],
                joint_target=env._joint_position_ref_applied[ids], leg_length=env._L0[ids],
                leg_angle=env._theta0[ids],
                base_net_force_n=env._contact_sensor.data.net_forces_w[ids][:,env._base_id].norm(dim=-1).max(-1).values)
            values = {key:value.detach().cpu().tolist() for key,value in state.items()}
            reasons = {key:value[ids].detach().cpu().tolist() for key,value in env._termination_reasons.items()}
            id_list = ids.cpu().tolist()
            with (output/'terminal_events_before_reset.jsonl').open('a') as stream:
                for offset,eid in enumerate(id_list):
                    row = dict(control_step=int(env.common_step_counter), environment_id=eid,
                        reasons=[key for key,value in reasons.items() if value[offset]],
                        state={key:finite(value[offset]) for key,value in values.items()})
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
            counts['unsafe_events'] += len(id_list)
            counts['captured_environment_ids'] = sorted(set(counts['captured_environment_ids']+id_list))
        if ids.numel() or counts['observed_control_steps'] % 48 == 0:
            write_json(output/'terminal_event_capture.json', counts)
        return result

    write_json(output/'terminal_event_capture.json', counts)
    env._get_rewards = reward
    return reward


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source1001_stationary_leg05_drift50_wheel01_smooth10_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(previous.__file__),
        action_smooth=ACTION_SMOOTH, action_smooth_wheel_multiplier=5.,
        authority='user_preserve_height_gains_repair_stability_before_gradual_motion')
    return result


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_source1001_smooth10_private', base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_require, original_observe = module.require_source, module.observe_environment
    transition_sha = None
    if source is not None:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 1001
                or cfg['rewards']['action_smooth'] != 0.
                or cfg['rewards']['action_smooth_wheel_multiplier'] != 5.):
            raise ValueError('Only the independently diagnosed exact1001 begins this repair')
        transition_sha = ANCHOR_SHA

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload the original model200 for this repair')
        if transition_sha is not None and sha(path) == transition_sha:
            return previous.require_source(path)
        saved, cfg = original_require(path, initial_transfer=False)
        if (cfg['rewards']['standing_velocity'] != -50.
                or cfg['rewards']['standing_wheel_tracking'] != -.1
                or cfg['rewards']['action_smooth'] != ACTION_SMOOTH
                or cfg['rewards']['action_smooth_wheel_multiplier'] != 5.):
            raise ValueError('Preserve the qualified single-cost recipe')
        return saved, cfg

    def observe(env, output):
        rewards = env.unwrapped.cfg.rewards
        if (rewards.standing_velocity != -50. or rewards.standing_wheel_tracking != -.1
                or rewards.action_smooth != ACTION_SMOOTH or rewards.action_smooth_wheel_multiplier != 5.):
            raise ValueError('Native reward settings differ from the one declared cost')
        result = original_observe(env, output)
        install_failure_capture(env.unwrapped, output)
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    spec = importlib.util.spec_from_file_location('_source1001_unchanged_exploration', previous.__file__)
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
    observed.run(output, previous.tires.arguments(arguments))


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
    source = Path(arguments[arguments.index('--checkpoint')+1])
    require_source(source, transition=transition)
    sys.argv = [str(SCRIPT), *previous.tires.arguments(arguments)]
    tree, original = evaluator_tree()
    exec(compile(tree, str(original), 'exec'), dict(__name__='__main__', __file__=str(original),
        _observe_source200=private_runtime().observe_environment))


def command(source, *args, **kwargs):
    argv, final, saved = previous_controller.command(source, *args, **kwargs)
    index = argv.index(str(previous_controller.ENTRY))
    argv[index:index+1] = [str(SCRIPT), 'native']
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


def replay_command(checkpoint, output):
    argv = previous_controller.replay_command(checkpoint, output)
    index = argv.index(str(previous_controller.EVALUATOR))
    argv[index:index+1] = [str(SCRIPT), 'replay']
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
            or not complete['completed'] or complete['samples'] != 27000):
        raise ValueError('Require the exact completed200-update full27 boundary')
    diagnosis_file = previous_output/'stationary_reference_chatter_repair_diagnosis.json'
    if transition:
        diagnosis = json.loads(diagnosis_file.read_text())
        if (not diagnosis['qualified_for_single_smooth10_repair']
                or diagnosis['checkpoint_sha256'] != ANCHOR_SHA
                or diagnosis['proposed_reward_change'] != {'action_smooth':[0., ACTION_SMOOTH]}
                or not diagnosis['all27_survive'] or diagnosis['resets']
                or not diagnosis['all_static_height_and_actual_leg_commands_pass10']
                or not diagnosis['all_static_translation_drifts_pass']
                or decision['stationary_height_regressions']
                or any(not label.startswith('height_high/') or not label.endswith('/yaw_mae_rad_s')
                       for label in decision['stationary_stability_regressions'])):
            raise ValueError('Require the specific surviving high-position chatter diagnosis')
    elif not decision['continue_from_candidate']:
        raise ValueError('Review and diagnose a failed descendant before further training')
    qualification = json.loads((QUALIFICATION/'qualification.json').read_text())
    frozen = json.loads((QUALIFICATION/'native_reward_transfer_qualification.json').read_text())
    if (not qualification['qualified'] or qualification['profile'] != profile()
            or qualification['implementation_sha256'] != sha(SCRIPT)
            or not frozen['qualified'] or frozen['source_sha256'] != ANCHOR_SHA
            or frozen['only_reward_change'] != {'action_smooth':[0., ACTION_SMOOTH]}):
        raise ValueError('Require real retained-state and frozen native reward qualifications')
    old = json.loads((previous_output/'experiment_plan.json').read_text())
    pins = dict(old['source_files'])
    if any(sha(name) != digest for name,digest in pins.items()):
        raise ValueError('A frozen source or prior evidence changed')
    expected = copy.deepcopy(cfg)
    expected['rewards']['action_smooth'] = ACTION_SMOOTH
    previous_controller.helper.base.validate_continuation_config(cfg, expected, cfg['commands'],
        reward_overrides={'rewards.action_smooth':ACTION_SMOOTH})
    if cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096:
        raise ValueError('Keep the actual4096 sampling configuration')
    watched = [SCRIPT, source, QUALIFICATION/'qualification.json',
        QUALIFICATION/'native_reward_transfer_qualification.json', acceptance.CURRENT, acceptance.POLICY,
        Path(acceptance.__file__).resolve(), previous_output/'status.json', previous_output/'stationary_review.json',
        previous_output/'native_learning_signal/completed_external.json',
        *sorted((source.parent/'params').glob('*.yaml')),
        *(previous_output/'replay'/name for name in ('assessment.json','manifest.json','completed.json','trace.csv'))]
    if transition:
        watched.append(diagnosis_file)
    pins.update({str(path):sha(path) for path in watched})
    peak, requirement = previous_controller.memory.memory_requirement()
    actual = saved_updates(source)
    plan = dict(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous_output),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=actual,
        source_Adam_step=optimizer_step(saved), source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous_output/'replay'),
        protected_reference=str(previous_output/'replay/assessment.json'), source_files=pins,
        target_actual_ppo_updates=actual+200, bounded_training_updates=200,
        expected_terminal_Adam_step=(actual+200)*20,
        run_name='plain_leg05_smooth10_stationary_'+output.name+'_u'+str(actual),
        reward_key='action_smooth', reward_value=ACTION_SMOOTH,
        only_reward_change={'action_smooth':[0.,ACTION_SMOOTH]} if transition else {},
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
    helper = previous_controller.helper
    helper.prepare, helper.command = prepare, command
    helper.wait_memory = previous_controller.memory.wait_memory
    helper.base.acceptance = acceptance
    helper.base.replay_command = replay_command
    helper.run(output, unit, previous_output)
    proof = json.loads((Path(output)/'native_learning_signal/leg_exploration_floor_runtime.json').read_text())
    if proof['profile'] != profile() or not proof['all17_Adam_bitwise_preserved']:
        raise ValueError('The unchanged exploration guard or Adam load differed')
    previous_controller.previous_acceptance.assess_current(Path(output)/'replay', 'assessment_previous20pct.json')


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
