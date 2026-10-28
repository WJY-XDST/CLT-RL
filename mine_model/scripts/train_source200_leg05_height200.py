"""Review before200 more PPO updates, with only leg exploration increased.

This does not extend the live preceding block. Its completed independent
replay and retained-state review must first authorize the selected source.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha, optimizer_step
from train_joint_pd_standing_seed import saved_updates
import train_source200_stationary_height200_4096 as helper

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_source200_leg05_runtime as runtime

RESULTS = ROOT / 'mine_model/results'
ENTRY = ROOT / 'mine_model/scripts/train_source200_leg05_observed.py'
EVALUATOR = ROOT / 'mine_model/scripts/evaluate_source200_leg05.py'
QUALIFICATION = RESULTS / 'joint_pd_source200_leg05_drift25_qualification_20261017/qualification.json'
FROZEN_QUALIFICATION = QUALIFICATION.parent / 'native_reward_transfer_qualification.json'
ORIGINAL_COMMAND = helper.command


def transition_source(source):
    saved = torch.load(source, map_location='cpu', weights_only=True)
    return saved.get('infos', {}).get(runtime.KEY) == runtime.prior.profile()


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    if (output.parent != RESULTS or not output.name.startswith('joint_pd_optimization_')
            or output.exists() or previous.parent != RESULTS):
        raise ValueError('Use a fresh owned output and preserve all previous blocks')
    state = json.loads((previous / 'status.json').read_text())
    decision = json.loads((previous / 'stationary_review.json').read_text())
    if state['training_active'] or state['state'] != 'agent_review_required':
        raise ValueError('Complete independent review before selecting another200-update source')
    source = Path(state['checkpoint']).resolve()
    if not source.is_relative_to(RUNS.resolve()):
        raise ValueError('Use an actual trained checkpoint, never an offline qualification roundtrip')
    transition = transition_source(source)
    saved, cfg = runtime.require_source(source, transition=transition)
    old = json.loads((previous / 'experiment_plan.json').read_text())
    native = json.loads((previous / 'native_learning_signal/completed_external.json').read_text())
    report = json.loads((previous / 'replay/assessment.json').read_text())
    completed = json.loads((previous / 'replay/completed.json').read_text())
    if (sha(source) != state['checkpoint_sha256'] or native['checkpoint_sha256'] != sha(source)
            or native['actual_observed_updates'] != 200 or native['optimizer_minibatches'] != 4000
            or native['PPO_updates_repeated'] != 0 or report['checkpoint_sha256'] != sha(source)
            or report['total_trials'] != 27 or not completed['completed'] or completed['samples'] != 27000):
        raise ValueError('Require the exact independently reviewed200/4000 boundary')
    repair = not decision['continue_from_candidate']
    diagnostic = previous / 'stationary_drift_repair_diagnosis.json'
    if repair:
        diagnosis = json.loads(diagnostic.read_text())
        static = [t for c in report['conditions'] if c['name'] in ('stand', 'height_low', 'height_high')
                  for t in c['trials']]
        if (not transition or not diagnosis['qualified_for_targeted_drift_repair']
                or diagnosis['checkpoint_sha256'] != sha(source)
                or diagnosis['proposed_reward_change'] != {'standing_velocity':[0., -25.]}
                or report['survivors'] != 27 or report['resets'] or len(static) != 9
                or decision['stationary_height_regressions']
                or any(not label.endswith('/speed_mae_m_s') for label in decision['stationary_stability_regressions'])
                or any(not t['survived'] or t['resets'] or
                    t['metrics']['body_p95_deg'] >= t['limits']['body_p95_deg'] or
                    t['metrics']['yaw_mae_rad_s'] >= t['limits']['yaw_mae_rad_s'] for t in static)):
            raise ValueError('Require the explicit surviving translation-drift repair, without waiving acceptance')
    frozen = json.loads(FROZEN_QUALIFICATION.read_text())
    if (not frozen['qualified'] or (transition and frozen['source_sha256'] != sha(source))
            or frozen['only_reward_change'] != {'standing_velocity':[0., -25.]}):
        raise ValueError('Qualify the exact frozen reward transfer before this initial repair')
    qualification = json.loads(QUALIFICATION.read_text())
    files = [Path(__file__).resolve(), ENTRY, EVALUATOR, Path(runtime.__file__).resolve()]
    if (not qualification['qualified'] or qualification['profile'] != runtime.profile()
            or any(qualification['implementation_sha256'].get(str(p)) != sha(p) for p in files)):
        raise ValueError('Qualify this exact leg-only floor implementation before native use')
    pins = dict(old['source_files'])
    if any(sha(Path(name)) != digest for name, digest in pins.items()):
        raise ValueError('Preserve the prior frozen implementation and evidence')
    expected = copy.deepcopy(cfg)
    expected['rewards']['standing_velocity'] = runtime.STANDING_VELOCITY_SCALE
    helper.base.validate_continuation_config(cfg, expected, cfg['commands'],
        reward_overrides={'rewards.standing_velocity':runtime.STANDING_VELOCITY_SCALE})
    if cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096:
        raise ValueError('Requalify resource/sampling changes separately from exploration')
    watched = [*files, QUALIFICATION, FROZEN_QUALIFICATION, source, *sorted((source.parent / 'params').glob('*.yaml')),
        previous / 'status.json', previous / 'stationary_review.json',
        previous / 'native_learning_signal/completed_external.json',
        *(previous / 'replay' / name for name in ('assessment.json', 'manifest.json', 'completed.json', 'trace.csv'))]
    if repair:
        watched.append(diagnostic)
    pins.update({str(p): sha(p) for p in watched})
    actual = saved_updates(source)
    plan = dict(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=actual,
        source_Adam_step=optimizer_step(saved), source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        protected_reference=str(previous / 'replay/assessment.json'), source_files=pins,
        target_actual_ppo_updates=actual + 200, bounded_training_updates=200,
        expected_terminal_Adam_step=(actual + 200) * 20,
        run_name='plain_leg05_stationary_' + output.name + '_u' + str(actual),
        reward_key='standing_velocity', reward_value=runtime.STANDING_VELOCITY_SCALE,
        only_reward_change={'standing_velocity':[cfg['rewards']['standing_velocity'], -25.]} if transition else {},
        only_exploration_change={'leg_std_floor':[.02, .05]} if transition else {},
        exploration_guard=dict(minimum_leg_std=.05, minimum_wheel_std=.03, global_minimum_std=.02,
            saved_initial_std_reset=False, explicit_leg_floor_projection_after_state_load=True),
        training_motion_scale=0., native_num_envs=4096, native_transition_batch=4096 * 48,
        optimizer_minibatches_per_actual_update=20, full_height_range=cfg['commands']['ranges_height'],
        original_source200_loaded_only_at_first_block=False, external_observation_size=27,
        policy_class='ActorCritic', architecture_migrated=False, ordinary_full_policy_PPO=True,
        auxiliary_losses_enabled=False, optimizer_reset=False, learned_std_reset=False,
        acceptance_revision=helper.base.acceptance.profile(), profile=runtime.profile(),
        training_config=expected, no_total_training_update_limit=True, next_block_requires_review=True,
        repair_from_unaccepted_drift_seed=repair,
        prior_review_retained_unmodified=True, acceptance_waived=False,
        user_authority='调大探索；不能因为稳定性有退步就回退；要做到稳定性和高度都收敛',
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    output.mkdir()
    write_json(output / 'experiment_plan.json', plan)
    return plan, expected


def command(source, *args, **kwargs):
    argv, final, saved = ORIGINAL_COMMAND(source, *args, **kwargs)
    old_entry = str(ROOT / 'mine_model/scripts/train_source200_stationary_observed.py')
    argv[argv.index(old_entry)] = str(ENTRY)
    if transition_source(source):
        argv.append('--increase_leg_exploration')
    return argv, final, saved


def replay_command(checkpoint, replay):
    argv = helper.base.BASE_REPLAY(checkpoint, replay)
    old_entry = str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py')
    argv[argv.index(old_entry)] = str(EVALUATOR)
    return runtime.tires.arguments(argv)


def run(output, previous, unit):
    helper.prepare = prepare
    helper.command = command
    helper.base.replay_command = replay_command
    helper.run(output, unit, previous)
    proof = json.loads((Path(output) / 'native_learning_signal/leg_exploration_floor_runtime.json').read_text())
    if proof['profile'] != runtime.profile() or not proof['all17_Adam_bitwise_preserved']:
        raise ValueError('Verify the native explicit leg-only floor before retaining this block')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.previous, args.unit)
