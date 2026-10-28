"""Reviewed ordinary-PPO feature trial; preserve the physical/control contract.

Compose the already exercised 100-update/stationary/full-replay coordinator.
Only process-local entry/architecture selection changes. This is not an
unlimited cost ladder: a stagnant representation trial requires diagnosis.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from auto_train_mine import write_json
import continue_joint_pd_stationary_height_credit100 as framework
from stationary_entropy_zero import canonical_command, qualify_control

ROOT, core, observed = framework.ROOT, framework.core, framework.observed
sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
from joint_pd_height_features_candidate import feature_model_from_config
from joint_pd_height_feature_runtime import saved_feature_model
from wheel_legged_gym_isaaclab.robot_model import MODEL

ENTRY = ROOT / 'mine_model/scripts/train_joint_pd_height_features_observed.py'
EVALUATOR = ROOT / 'mine_model/scripts/evaluate_joint_pd_height_features.py'
MODULE = ROOT / 'mine_model/experiments/joint_pd_height_features_candidate.py'
RUNTIME = ROOT / 'mine_model/experiments/joint_pd_height_feature_runtime.py'
CONTROL = ROOT / 'mine_model/results/joint_pd_optimization_20261017_015812_140338_stationary100'
QUALIFICATION = ROOT / 'mine_model/results/joint_pd_height_feature_launch_qualification_20261017.json'
PROFILE = 'encoder_imu_support_height_error_v1'
OLD_PREPARE = framework.prepare
OLD_TRAIN = framework.train
OLD_COMMAND = framework.native.entropy.observed_command
OLD_REPLAY = core.block.replay_command


def feature_command(argv):
    argv = list(argv)
    if argv.count(str(observed.ENTRY)) != 1 or argv.count('agent.policy.class_name=ActorCritic') != 1:
        raise ValueError('Transform exactly the qualified ordinary observed command')
    if any(value in argv for value in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Never reset a feature trial source')
    argv[argv.index(str(observed.ENTRY))] = str(ENTRY)
    argv[argv.index('agent.policy.class_name=ActorCritic')] = 'agent.policy.class_name=JointPdHeightFeatureActorCritic'
    return argv


def prepare(previous, output, delta, scale, migration_path):
    qualification = json.loads(QUALIFICATION.read_text())
    for name, digest in qualification['pinned_files'].items():
        if core.sha(Path(name)) != digest:
            raise ValueError('Qualified feature implementation changed: ' + name)
    plan, expected = OLD_PREPARE(previous, output, delta, scale, migration_path)
    source = Path(plan['source_checkpoint'])
    saved = torch.load(source, map_location='cpu', weights_only=True)
    model = feature_model_from_config(expected, MODEL['hip_z'])
    width = saved['model_state_dict']['actor.0.weight'].shape[1]
    matched = None
    if width == 27:
        if scale != 0. or plan['entropy_coefficient'] != .001:
            raise ValueError('The isolated first feature trial retains matched stationary entropy0.001')
        matched = qualify_control(CONTROL, source, expected, core.validate_checkpoint_robot, core.sha)
        matched.pop('only_agent_difference')  # this is a representation comparison
        matched['only_policy_difference'] = PROFILE
    elif width != 29 or saved_feature_model(saved) != model:
        raise ValueError('Continue only the exact declared feature representation')
    watched = [Path(__file__).resolve(), ENTRY, EVALUATOR, MODULE, RUNTIME, QUALIFICATION,
        ROOT / 'mine_model/scripts/stationary_entropy_zero.py']
    watched.extend(Path(p) for p in qualification['reports'])
    if matched:
        watched.extend(CONTROL / p for p in (
            'experiment_plan.json', 'status.json', 'launch.json', 'resume_verification.json',
            'replay/assessment.json', 'native_learning_signal/completed_external.json'))
    plan['source_files'].update({str(p): core.sha(p) for p in watched})
    plan.update(height_feature_model=model, height_feature_profile=PROFILE,
        height_feature_migration_required=width == 27, matched_representation_control=matched,
        only_policy_change=('zero-initialized measured support-height-error features at first actor/critic layer'
                            if width == 27 else 'none; continue retained feature policy'),
        only_difference_vs_matched_control=({'internal_actor_critic_features': PROFILE} if matched else None),
        external_observation_size=27, external_action_size=6, policy_parameter_and_Adam_state_count=17,
        policy_class='JointPdHeightFeatureActorCritic', auxiliary_losses_enabled=False,
        only_training_change='Explicit encoder/IMU height information; ordinary PPO loss, physical control and full initial retained state',
        qualification=str(QUALIFICATION), initial_function_preserving_migration=True)
    write_json(Path(output) / 'experiment_plan.json', plan)
    return plan, expected


def command(*args, **kwargs):
    original, final, saved = OLD_COMMAND(*args, **kwargs)
    argv = feature_command(original)
    output = Path(argv[argv.index('--observation-output') + 1]).parent
    plan = json.loads((output / 'experiment_plan.json').read_text())
    if plan['matched_representation_control']:
        control = json.loads((CONTROL / 'launch.json').read_text())['command']
        if canonical_command(argv) != canonical_command(feature_command(control)):
            raise ValueError('Unexpected argument change outside the declared representation entry')
    return argv, final, saved


def replay_command(*args, **kwargs):
    argv = OLD_REPLAY(*args, **kwargs)
    original = str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py')
    if argv.count(original) != 1:
        raise ValueError('Require the original full canonical evaluator arguments')
    argv[argv.index(original)] = str(EVALUATOR)
    return argv


def next_reward(cfg, evidence, history, used):
    feature_history = [row for row in history if json.loads(
        (Path(row['output']) / 'experiment_plan.json').read_text()).get('height_feature_profile') == PROFILE]
    if not feature_history:
        # Legacy reward/std comparisons are different methods; preserve their
        # history but do not misclassify this isolated feature trial as a repeat.
        used.clear()
        return {'joint_target_leg_length_error_sq': -500.} if cfg['rewards'].get('joint_target_leg_length_error_sq', 0.) == 0. else {}
    last = feature_history[-1]
    candidate = last['candidate_stationary']
    before = last['source_stationary']
    if not last['promote']:
        raise RuntimeError('Feature trial did not retain per-condition height/stability progress; inspect full native and replay data before a different trial')
    if candidate['stationary_all_original_metrics_passed']:
        return {}
    if (candidate['stationary_pose_passed']
            and candidate['worst_stationary_height_relative_mae'] < .99 * before['worst_stationary_height_relative_mae']):
        return {}
    raise RuntimeError('Measured feature learning is stagnant or remaining original reference metrics need diagnosis; do not blindly extend')


def install():
    framework.prepare = prepare
    framework.native.entropy.observed_command = command
    core.block.replay_command = replay_command
    framework.height_credit.next_reward = next_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', type=Path)
    parser.add_argument('--coordinator', type=Path)
    parser.add_argument('--environment-revision', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    install()
    if args.prepare_only:
        prepare(args.previous, args.output, {'joint_target_leg_length_error_sq': -500.}, 0., args.environment_revision)
    else:
        framework.run(args.previous_coordinator, args.coordinator, args.unit, args.environment_revision)
