"""Reviewed200-update posture repair; do not advance the motion curriculum.

Only orientation -40->-80 changes. Keep full17 Adam/LR/std and complete native
full27 replay; moving posture improvements outrank height and velocity gains.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step, validate_continuation_config
from train_joint_pd_standing_seed import saved_updates
import source1201_orientation40
import joint_pd_source1601_motion025_orientation80_runtime as repair
import joint_pd_acceptance_10pct_20261017 as acceptance
import joint_pd_stationary_stage as stage
from joint_pd_priority import compare_priority
import train_source200_stationary_height200_4096 as helper
import train_source200_drift25_measured_memory200 as memory
import run_static_reset_pair_diagnosis as boundary
import qualify_source1601_motion025_orientation80 as qualifier

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_source1601_motion025_orientation80_qualification_20261017'


def qualifications():
    cpu = json.loads((QUALIFICATION / 'qualification.json').read_text())
    frozen = json.loads((QUALIFICATION / 'native_reward_transfer_qualification.json').read_text())
    if (not cpu['qualified'] or cpu['profile'] != repair.profile()
            or cpu['implementation_sha256'] != sha(repair.SCRIPT)
            or not cpu['all17_Adam_bitwise_preserved'] or not cpu['actual_saved_adaptive_LR_preserved']
            or not cpu['saved_std_bitwise_preserved'] or not frozen['qualified']
            or frozen['source_sha256'] != repair.ANCHOR_SHA
            or frozen['implementation_sha256'] != sha(repair.SCRIPT)
            or frozen['only_reward_change'] != {'orientation': [-40., repair.ORIENTATION]}
            or not frozen['physical_outputs_observations_actions_and_other_rewards_bitwise_equal']
            or any(sha(path) != digest for path, digest in frozen['source_data_sha256'].items())):
        raise ValueError('Require intact full17 CPU and single-reward frozen native equivalence')
    for kind in ('cpu', 'frozen'):
        proof = json.loads((QUALIFICATION / ('qualification_adapter_' + kind + '.json')).read_text())
        if (not proof['qualified'] or proof['adapter_sha256'] != sha(qualifier.SCRIPT)
                or proof['runtime_sha256'] != sha(repair.SCRIPT)
                or proof['original_template_sha256'] != sha(qualifier.template.__file__)
                or proof['original_thresholds_changed']):
            raise ValueError('Qualification adapter proof changed')
    return cpu, frozen


def selected_source(previous):
    state = json.loads((previous / 'status.json').read_text())
    source = Path(state['checkpoint']).resolve()
    transition = sha(source) == repair.ANCHOR_SHA
    module = repair.previous if transition else repair
    old = boundary.smooth
    boundary.smooth = module
    try:
        source, entry = boundary.reviewed_source(previous)
    finally:
        boundary.smooth = old
    if entry != module.SCRIPT:
        raise ValueError('Continue only the exact reviewed native motion lineage')
    if transition:
        repair.repair_ready()
    elif not json.loads((previous / 'stationary_review.json').read_text())['continue_from_candidate']:
        raise ValueError('Diagnose the completed repair before another200 updates')
    repair.require_source(source, transition=transition)
    return source, transition


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_optimization_'):
        raise ValueError('Use a fresh owned output and preserve earlier trials')
    qualifications()
    source, transition = selected_source(previous)
    saved, cfg = repair.require_source(source, transition=transition)
    expected = copy.deepcopy(cfg)
    expected['rewards']['orientation'] = repair.ORIENTATION
    validate_continuation_config(cfg, expected, expected['commands'],
        reward_overrides={'rewards.orientation': repair.ORIENTATION})
    if cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096:
        raise ValueError('Keep measured4096 sampling; do not silently resize this repair')
    old = json.loads((previous / 'experiment_plan.json').read_text())
    pins = dict(old['source_files'])
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('A pinned previous source or proof changed')
    watched = [SCRIPT, repair.SCRIPT, repair.DECISION, qualifier.SCRIPT,
        Path(qualifier.template.__file__), source, *sorted((source.parent / 'params').glob('*.yaml')),
        *(QUALIFICATION / name for name in ('qualification.json', 'native_reward_transfer_qualification.json',
            'qualification_adapter_cpu.json', 'qualification_adapter_frozen.json')),
        acceptance.CURRENT, acceptance.POLICY, Path(acceptance.__file__),
        *(previous / name for name in ('status.json', 'stationary_review.json', 'experiment_plan.json',
            'native_learning_signal/completed_external.json')),
        *(previous / 'replay' / name for name in ('assessment.json', 'height_leg_command_assessment.json',
            'manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json'))]
    pins.update({str(path.resolve()): sha(path) for path in watched})
    if transition:
        decision = repair.repair_ready()
        pins.update(decision['source_pins'])
        paired = RESULTS / 'joint_pd_motion025_pair_20261017_source1601_full_height'
        for label in ('baseline1401', 'candidate'):
            data = json.loads((paired / label / 'motion025_full_height_diagnosis.json').read_text())
            if any(sha(path) != digest for path, digest in data['source_files'].items()):
                raise ValueError('Raw full-height diagnosis changed')
            pins.update(data['source_files'])
    peak, requirement = memory.memory_requirement()
    actual = saved_updates(source)
    plan = dict(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=actual,
        source_Adam_step=optimizer_step(saved), source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        protected_reference=str(RESULTS / 'joint_pd_optimization_20261017_162240_531855_source1201_orientation40_200/replay/assessment.json'),
        source_files=pins, target_actual_ppo_updates=actual + 200, bounded_training_updates=200,
        expected_terminal_Adam_step=(actual + 200) * 20,
        run_name='plain_motion025_orientation80_' + output.name + '_u' + str(actual),
        reward_key='orientation', reward_value=repair.ORIENTATION,
        only_reward_change={'orientation': [-40., repair.ORIENTATION]} if transition else {},
        only_command_change={}, only_exploration_change={},
        training_motion_scale=.25, native_num_envs=4096, fixed_standing_environments=1024,
        native_transition_batch=4096 * 48, optimizer_minibatches_per_actual_update=20,
        full_height_range=cfg['commands']['ranges_height'],
        exploration_guard=dict(minimum_leg_std=.05, minimum_wheel_std=.03,
                               global_minimum_std=.02, saved_initial_std_reset=False),
        source_model_and_optimizer_reloaded_from_original200=False,
        original_source200_loaded_only_at_first_block=False,
        stationary_checkpoint_preserved=True,
        verified_static_milestone=str(repair.previous.STATIC),
        motion_requires_static_and_independent_sustained_pass=True,
        policy_class='ActorCritic', external_observation_size=27, architecture_migrated=False,
        ordinary_full_policy_PPO=True, auxiliary_losses_enabled=False,
        optimizer_reset=False, learned_std_reset=False, acceptance_revision=acceptance.profile(),
        profile=repair.profile(), training_config=expected, no_total_training_update_limit=True,
        next_block_requires_review=True, acceptance_waived=False,
        resource_reservation_method='Measured4096 highwater plus4GiB',
        measured_native_RSS_highwater_bytes=peak, combined_memory_required_bytes=requirement,
        physical_available_memory_required_bytes=6 * 1024**3,
        post_block_full27_and_supplemental_low_speed_review_required=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    output.mkdir()
    write_json(output / 'experiment_plan.json', plan)
    return plan, expected


def quality(report):
    posture, height, motion = [], [], []
    for case in report['conditions']:
        for trial in case['trials']:
            if not trial['survived'] or not trial['metrics']:
                return [27 - report['survivors'], report['resets'], 1.e9]
            metrics, limits = trial['metrics'], trial['limits']
            posture.append(metrics['body_p95_deg'] / limits['body_p95_deg'])
            height.append(metrics['height_mae_m'] / limits['height_mae_m'])
            if case['speed'] == 0. and case['yaw'] == 0.:
                posture.extend(metrics[key] / limits[key] for key in ('speed_mae_m_s', 'yaw_mae_rad_s'))
            else:
                motion.extend(metrics[key] / limits[key] for key in ('speed_mae_m_s', 'yaw_mae_rad_s'))
    def violation(values):
        return [sum(value >= 1. for value in values), max([1., *values]),
                sum(max(0., value - 1.) for value in values)]
    return [0, report['resets'], *violation(posture), *violation(height), sum(motion) / len(motion)]


def review(output, after, before):
    result = stage.review(after, before)
    legs = json.loads((Path(output) / 'replay/height_leg_command_assessment.json').read_text())
    retained = bool(stage.summary(after)['stationary_all_original_metrics_passed']
        and legs['static_stage_candidate_passed'] and after['survivors'] == 27 and not after['resets'])
    comparison = compare_priority(after, before)
    improvement = quality(after) < quality(before)
    eligible = retained and not comparison['regressions']['stability'] and improvement
    result.update(continue_from_candidate=bool(eligible), promote=bool(eligible),
        candidate_priority_quality=quality(after), source_priority_quality=quality(before),
        static_all_original_metrics_and_actual_leg_commands_retained=retained,
        full_scope_comparison=comparison, posture_then_height_then_velocity_priority=True,
        training_motion_scale=.25, only_reward_repair={'orientation': [-40., repair.ORIENTATION]},
        original_acceptance_unchanged=True, acceptance_revision=acceptance.profile(),
        curriculum_selection_only=True, final_full27_gate_passed=bool(after['passed']),
        independent_sustained_verified=False, next_block_requires_review=True, policy_accepted=False)
    return result


def run(output, previous, unit):
    helper.prepare, helper.command = prepare, repair.command
    helper.wait_memory = memory.wait_memory
    helper.base.acceptance = acceptance
    helper.base.replay_command = repair.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original_write = helper.write_json
    def pointer_write(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=.25, fixed_standing_environments=1024,
                training_stage='motion025_orientation80')
        return original_write(path, value)
    helper.write_json = pointer_write
    try:
        helper.run(output, unit, previous)
    finally:
        helper.write_json = original_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.previous, args.unit)
