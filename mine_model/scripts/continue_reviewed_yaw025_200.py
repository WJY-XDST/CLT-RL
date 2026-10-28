"""Continue one reviewed yaw025 descendant without changing the PPO recipe.

A checkpoint-bound decision and its own completed static retention are
required. Full-speed failures stay in the original review and are never
relabelled accepted. This is one200-update repair block, not an open loop.
"""
import argparse
import json
from pathlib import Path

import train_source2201_yaw025_200 as template
import validate_yaw025_static_retention10 as retention
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def decision_source(previous, decision_path):
    previous, decision_path = Path(previous).resolve(), Path(decision_path).resolve()
    if previous.parent != RESULTS or decision_path.parent != RESULTS:
        raise ValueError('Use an owned completed trial and checkpoint-bound review decision')
    decision = json.loads(decision_path.read_text())
    expected_changes = dict(rewards={}, commands={}, exploration={}, controller={}, physics={})
    if (Path(decision['training_output']) != previous
            or decision['planned_actual_updates'] != 200
            or decision['next_configuration_changes'] != expected_changes
            or decision['relative_error_target'] != .1
            or not decision['requires_current_static_retention_verified']
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['full_speed_failures_remain_reported']
            or decision['faster_curriculum_authorized'] or decision['optimizer_reset']
            or decision['std_reset'] or decision['original_model200_reload']
            or decision['policy_accepted']
            or any(sha(p) != digest for p, digest in decision['source_pins'].items())):
        raise ValueError('Require the explicit unchanged-configuration200-update review decision')
    module = retention.adapted()
    source = module.reviewed_source(previous)
    full51 = module.require_full51(decision['full51_output'], source)
    digest, actual = sha(source), saved_updates(source)
    if (digest != decision['source_checkpoint_sha256']
            or actual != decision['source_actual_updates']
            or decision['source_all17_Adam_step'] != 20 * actual):
        raise ValueError('The decision selects another checkpoint or optimizer clock')
    directory = Path(decision['static_retention_output']).resolve()
    report_path = directory / 'static_retention.json'
    if not report_path.is_file():
        raise ValueError('Finish this checkpoint static retention before another200 updates')
    report = json.loads(report_path.read_text())
    status = json.loads((directory / 'status.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if (report['checkpoint_sha256'] != digest
            or status['checkpoint_sha256'] != digest
            or status['state'] != 'agent_review_required' or status['validation_active']
            or not status['static_retention_verified'] or not report['static_retention_verified']
            or not report['independent_current_checkpoint']
            or not report['same_original_thresholds_and_windows']
            or not report['deterministic_full_height_steps_passed']
            or not report['sampled_exploration_survived']
            or not all(stage['static_passed'] for stage in report['stages'].values())
            or report['training_modified'] or report['optimizer_updated']
            or report['weights_updated'] or report['policy_accepted']
            or report['faster_curriculum_authorized']
            or report['acceptance_revision'] != module.acceptance.profile()
            or plan['dependencies'] != module.dependencies()
            or Path(report['full51_review']) != full51):
        raise ValueError('This checkpoint has not passed the unchanged independent static retention')
    assessed = json.loads((full51 / 'candidate/motion025_full_height_diagnosis.json').read_text())
    baseline = json.loads(Path(decision['baseline_full51_report']).read_text())
    if (sum(x['stage_constraints_passed'] for x in assessed['trials'])
            <= sum(x['stage_constraints_passed'] for x in baseline['trials'])
            or assessed['survivors'] != 51 or assessed['resets']
            or not all(t['metrics']['height_mae_m'] < t['limits']['height_mae_m']
                       and t['metrics']['body_p95_deg'] < t['limits']['body_p95_deg']
                       for t in assessed['trials'])):
        raise ValueError('Review current stage progress and standing/height before unchanged continuation')
    if any((old['case'], old['seed']) != (new['case'], new['seed'])
           for old, new in zip(baseline['trials'], assessed['trials'])):
        raise ValueError('Use the matched original full51 stage comparison')
    # Do not overwrite the original full27 decision. This separate decision
    # permits further repair while preserving its failed acceptance and regressions.
    original = json.loads((previous / 'stationary_review.json').read_text())
    if original['continue_from_candidate'] != decision['original_full_scope_continue_flag']:
        raise ValueError('Preserve the original review result separately from repair selection')
    return source, decision, directory


def adapted(decision_path):
    decision_path = Path(decision_path).resolve()
    module = template.adapted()
    module.SCRIPT = SCRIPT

    def selected_source(previous):
        source, _, _ = decision_source(previous, decision_path)
        return source, False  # No first-profile transfer or configuration change.

    module.selected_source = selected_source
    original_prepare = module.prepare

    def prepare(output, previous):
        source, decision, static = decision_source(previous, decision_path)
        plan, cfg = original_prepare(output, previous)
        _, saved_cfg = module.repair.require_source(source)
        if (cfg != saved_cfg or plan['only_reward_change'] or plan['only_command_change']
                or plan['only_exploration_change'] or plan['optimizer_reset']
                or plan['learned_std_reset'] or plan['source_actual_ppo_updates'] != saved_updates(source)):
            raise ValueError('Unchanged continuation changed the saved training recipe')
        evidence = [decision_path, SCRIPT, Path(template.__file__), Path(retention.__file__),
            *(p for p in static.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.npz'))]
        plan['source_files'].update({str(p.resolve()): sha(p) for p in evidence})
        plan['source_files'].update(decision['source_pins'])
        plan.update(source_is_reviewed_unaccepted_repair_branch=True,
            selected_for_another200_by_explicit_review=str(decision_path),
            original_full_scope_continue_flag=decision['original_full_scope_continue_flag'],
            source_current_model_static_retention_verified=True,
            source_static_retention_report=str(static / 'static_retention.json'),
            unchanged_recipe_continuation=True, current_exploration_review=decision['effective_exploration_review'],
            current_policy_update_health_review=decision['policy_update_health_review'],
            current_wheel_error_budget=decision['wheel_error_budget'],
            effective_exploration_and_optimizer_review_required_per_block=True,
            acceptance_and_review_predicates_unchanged=True,
            full_speed_failures_preserved=True, faster_curriculum_authorized=False)
        module.write_json(Path(output) / 'experiment_plan.json', plan)
        return plan, cfg

    module.prepare = prepare
    return module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    module = adapted(args.decision)
    if args.prepare_only:
        module.prepare(args.output, args.previous)
    else:
        module.run(args.output, args.previous, args.unit)
