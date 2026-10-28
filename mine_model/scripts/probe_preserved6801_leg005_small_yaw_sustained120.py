"""Pair long small-yaw responses after all original reviews have completed.

The original54 sustained cases only request yaw0 or+/-.3. This supplemental
54-case/seed grid tests+/-.075 at three heights and speeds-.8,0,+.8, using
fresh seeds, the original98 windows, seven metrics, limits and failedNone
semantics. It cannot replace an original review or select/continue training.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from joint_pd_continuation import ROOT, sha
from review_plain27_single_reward_block200 import rebound
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_plan
import joint_pd_preserved6801_leg005_runtime as runtime
import qualify_preserved6801_leg005_transfer as qualifier
import review_preserved6801_leg005_block200_v2 as post
import review_preserved6801_leg005_fullspeed_height_transitions_v2 as signed
import verify_independent_fullspeed_sustained as original

SCRIPT = Path(__file__).resolve()
QUALIFICATION = qualifier.OUTPUT / 'small_yaw_sustained120_qualification.json'
OUTPUT = runtime.RESULTS / 'joint_pd_preserved6801_leg005_small_yaw_sustained120_20261021'
AFTER_UNIT = 'clt-joint-pd-preserved6801-leg005-v2-signed-height-review-20261021.service'
SEEDS = [402683, 402684, 402685]
read, core = original.read, post.core


def cases_for(cfg):
    return [dict(name=height_label + '_' + direction + '_' + turn,
            speed=speed, yaw=yaw, height=height)
        for height_label, height in zip(('low', 'nominal', 'high'),
            (cfg['commands']['ranges_height'][0], .3, cfg['commands']['ranges_height'][1]))
        for direction, speed in (('reverse08', -.8), ('spin', 0.), ('forward08', .8))
        for turn, yaw in (('right0075', -.075), ('left0075', .075))]


def qualify():
    if QUALIFICATION.exists() or OUTPUT.exists():
        raise ValueError('Preserve each previous qualification and native trial')
    decision = runtime.repair_ready()
    launch_path = qualifier.OUTPUT / 'training_launch_v2.json'
    launch = read(launch_path)
    plan_path = Path(launch['output']) / 'experiment_plan.json'
    plan = read(plan_path)
    source, _, cfg = runtime.anchor()
    old = read(post.QUALIFICATION)
    source120 = read(decision['source_original_sustained120'])
    cases = cases_for(cfg)
    parsed, sequence, period = command_probe_plan(cases, SimpleNamespace(**cfg['commands']))
    if (not old['qualified'] or launch['source_checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['source_actual_updates'] != 6801 or launch['target_actual_updates'] != 7001
            or launch['planned_actual_updates'] != 200 or launch['target_all17_Adam_step'] != 140020
            or plan['next_configuration_changes'] != runtime.DELTA
            or sorted({c['yaw'] for c in old['sustained_cases']}) != [-.3, 0., .3]
            or len(parsed) != 18 or sequence is not None or period is not None
            or not source120['raw_integrity_verified'] or source120['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or not source120['original98_overlapping_150frame_windows_per_trial_preserved']):
        raise ValueError('Require the exact selected200 trial, original long coverage and explicit supplemental grid')
    known = set()
    for path in plan['source_files']:
        if Path(path).name == 'manifest.json':
            known.update(read(path).get('seeds', []))
    if known.intersection(SEEDS):
        raise ValueError('Choose fresh seeds absent from the source manifest pins')
    pins = {**plan['source_files'], **old['source_pins']}
    for p in (SCRIPT, runtime.SCRIPT, runtime.previous.SCRIPT, qualifier.SCRIPT,
            post.SCRIPT, post.QUALIFICATION, signed.SCRIPT, signed.QUALIFICATION,
            original.SCRIPT, core.SCRIPT, Path(original.numeric.__file__),
            Path(original.numeric.acceptance.assessor.__file__), original.numeric.acceptance.CURRENT,
            original.numeric.acceptance.POLICY, launch_path, plan_path, source,
            Path(decision['source_original_sustained120']), runtime.TRAINING / 'replay/resolved_env.json',
            Path(command_probe_plan.__code__.co_filename)):
        pins[str(p)] = sha(p)
    proof = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        after_unit=AFTER_UNIT, training=launch['output'], output=str(OUTPUT),
        source_checkpoint=str(source), source_checkpoint_sha256=runtime.ANCHOR_SHA,
        source_actual_updates=6801, target_actual_updates=7001, target_all17_Adam_step=140020,
        cases=cases, seeds=SEEDS, seconds_per_case=120., stable_seconds=100.,
        original_windows_per_trial=98, episodes_per_policy=54, total_paired_episodes=108,
        fresh_seeds_absent_from_all_source_pinned_manifests=True,
        source_original120_raw=decision['source_original_sustained120'],
        source_reference_resolved_env=str(runtime.TRAINING / 'replay/resolved_env.json'),
        complete_original_review_required=str(post.OUTPUT / 'complete_with_retained_interior_reviews.json'),
        signed_height_review_required=str(signed.OUTPUT / 'complete_paired_fullspeed_height_transitions.json'),
        purpose='Distinguish slow small-yaw response from persistent steady bias at full signed speed and all heights.',
        original735_static810_120s_and_signed_reviews_not_replaced=True,
        original_numeric_score_function_unmodified=True, diagnostic_prefix_ends_s=[20., 60., 120.],
        no_new_training_or_policy_selection=True, source_pins=pins,
        training_modified=False, optimizer_executed=False, simulator_started=False,
        acceptance_changed=False, policy_accepted=False)
    core.intact(pins)
    core.save(QUALIFICATION, proof)
    return proof


def assess(directory, checkpoint, expected_cfg, proof):
    manifest, finished, cfg = [read(directory / n) for n in ('manifest.json', 'completed.json', 'resolved_env.json')]
    expected = dict(expected_cfg)
    expected['log_dir'] = cfg['log_dir']
    if (cfg != expected or not finished['completed'] or finished['samples'] != 324000
            or manifest['checkpoint_sha256'] != sha(checkpoint) or manifest['cases'] != proof['cases']
            or manifest['seeds'] != SEEDS or manifest['seconds_per_case'] != 120.
            or manifest['sampled_policy_actions'] or manifest['weights_updated']
            or manifest['policy_input_remapping'] or manifest['policy_acceptance_test']
            or manifest['command_sequence'] is not None or manifest['reset_velocity_uniform_amplitude'] != .05
            or manifest['physics_dt'] != .005 or manifest['policy_dt'] != .02
            or manifest['solver_position_iterations'] != 128 or manifest['solver_velocity_iterations'] != 8
            or manifest['enable_external_forces_every_iteration'] is not True
            or manifest['leg_control_mode'] != 'implicit_joint_position'
            or manifest['wheel_probe'] or manifest['action_bias_probes']
            or any(manifest.get(key) is not None for key in ('wheel_reference_limit_probe',
                'joint_saturation_reward_probe', 'tracking_goal_component_reward_probe',
                'tracking_goal_reward_probe', 'wheel_action_saturation_reward_probe'))):
        raise ValueError('The saved controller, full timeline or independent grid changed')
    raw = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    if len(raw) != 324000 or any(not np.isfinite(raw[k]).all() for k in raw.dtype.names):
        raise ValueError('Retain every complete finite raw frame, including failed episodes')
    data = raw.reshape(6000, 54)
    resets = int(np.count_nonzero((data['terminated'] != 0) | (data['timeout'] != 0)))
    if resets != finished['resets']:
        raise ValueError('Raw OR reset count and native completion differ')
    with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as z:
        obs, means = z['observations'], z['means']
        if (obs.shape != (6000, 54, 27) or means.shape != (6000, 54, 6)
                or not np.isfinite(obs).all() or not np.isfinite(means).all()
                or list(z['seeds']) != SEEDS or float(z['policy_dt']) != .02):
            raise ValueError('Require the complete ordinary27-input/6-output native timeline')
        initial = obs[0].copy()
        references = np.stack([data[k] for k in ('left_wheel_ref_rad_s', 'right_wheel_ref_rad_s')], -1)
        mapping = float(abs(references - 15. * np.clip(means[..., [2, 5]], -1.25, 1.25)).max())
    if mapping > 1.e-7:
        raise ValueError('Saved wheel actions no longer reproduce native references')
    trials = original.score(data, proof['cases'], SEEDS, 120., 100.)

    def with_literal(items):
        for t in items:
            relative = None if t['metrics'] is None else {metric: (
                100. * t['metrics'][metric] / abs(t['case'][command]) if t['case'][command] else None)
                for metric, command in (('speed_mae_m_s', 'speed'), ('yaw_mae_rad_s', 'yaw'), ('height_mae_m', 'height'))}
            t['nonzero_command_relative_error_percent'] = relative
            t['original_AND_literal10_passed'] = bool(t['passed'] and relative is not None
                and all(v is None or v < 10. for v in relative.values()))
        return items

    with_literal(trials)
    prefixes = {str(seconds): with_literal(original.score(data[:round(seconds / .02)],
        proof['cases'], SEEDS, seconds, 3.)) for seconds in (20., 60., 120.)}
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        total_trials=54, survivors=sum(t['survived'] for t in trials),
        passed=sum(t['passed'] for t in trials), original_AND_literal10_passed=sum(
            t['original_AND_literal10_passed'] for t in trials),
        raw_OR_resets=resets,
        original98_overlapping_windows_preserved=True, failed_first_episode_metrics_remain_None=True,
        native_wheel_reference_reproduction_max_error_rad_s=mapping, trials=trials,
        diagnostic_prefix_last3s_original_score=prefixes, original_scoring_function_unmodified=True,
        diagnostic_prefix_windows_are_not_the_100s_sustained_acceptance=True,
        source_pins={str(directory / name): sha(directory / name) for name in
            ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')},
        training_modified=False, optimizer_executed=False, policy_accepted=False)
    return result, initial


def run():
    proof = read(QUALIFICATION)
    core.intact(proof['source_pins'])
    complete, extra = read(proof['complete_original_review_required']), read(proof['signed_height_review_required'])
    final = read(Path(proof['training']) / 'final_checkpoint_verification_readonly_20261019.json')
    if (OUTPUT.exists() or not complete['all_original_and_retained_reviews_complete']
            or not extra['completed'] or not final['verified'] or final['actual_ppo_updates'] != 7001
            or final['all17_Adam_step'] != 140020 or complete['checkpoint_sha256'] != final['checkpoint_sha256']):
        raise ValueError('Complete all original frozen reviews before this long small-yaw supplement')
    candidate = Path(final['checkpoint'])
    runtime.require_source(candidate)
    OUTPUT.mkdir()
    core.save(OUTPUT / 'plan.json', proof)
    reference_cfg = read(proof['source_reference_resolved_env'])
    reference_cfg['episode_length_s'] = 122.
    reference_cfg['scene']['num_envs'] = reference_cfg['terrain']['num_envs'] = 54
    results, initial = {}, None
    for label, checkpoint, api in [('source6801', Path(proof['source_checkpoint']), runtime.previous),
            ('candidate7001', candidate, runtime)]:
        native = rebound(core.native, runtime=api, QUALIFICATION=QUALIFICATION)
        directory = native(checkpoint, OUTPUT / label / 'candidate', 120., SEEDS, proof['cases'])
        report, current_initial = assess(directory, checkpoint, reference_cfg, proof)
        if initial is None:
            initial = current_initial
        elif not np.array_equal(initial, current_initial):
            raise ValueError('Source/candidate initial observations are not exactly paired')
        core.save(OUTPUT / label / 'all54_raw_and_windows_verified.json', report)
        results[label] = {k: report[k] for k in ('checkpoint', 'checkpoint_sha256',
            'total_trials', 'survivors', 'passed', 'original_AND_literal10_passed', 'raw_OR_resets')}
        core.intact(proof['source_pins'])
    result = dict(completed=True, observed_at=datetime.now().astimezone().isoformat(), results=results,
        source_candidate_initial_observations_bitwise_paired=True,
        original735_static810_original120_and_signed_reviews_not_replaced=True,
        original98_windows_all7_metrics_limits_and_failed_None_preserved=True,
        only_supplemental_cases_and_fresh_seeds_added=True, source_pins=proof['source_pins'],
        training_modified=False, optimizer_executed=False, acceptance_changed=False,
        can_select_or_extend_training=False, policy_accepted=False)
    core.save(OUTPUT / 'complete_paired_small_yaw_sustained120.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    result = qualify() if parser.parse_args().qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'completed', 'total_paired_episodes', 'policy_accepted') if k in result}))
