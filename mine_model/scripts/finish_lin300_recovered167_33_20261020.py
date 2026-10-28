"""Verify167+33 exact saved updates, retain raw files, then canonical27 replay.

This collects evidence. It cannot repeat PPO, accept a policy or relax gates.
All original full/static/moving/steering/120s and retained78 reviews follow.
"""
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import recover_lin300_native33_20261020 as recovery
import verify_lin300_recovered167_33 as verifier

SCRIPT = Path(__file__).resolve()
ROOT, RESULTS = recovery.ROOT, recovery.RESULTS
OUTPUT = verifier.VIEW
QUALIFICATION = recovery.QUALIFICATION / 'finalizer/qualification.json'
REVIEW_QUALIFICATION = recovery.QUALIFICATION / 'reviews/post200_original_review_qualification.json'
REVIEW_SCRIPT = SCRIPT.with_name('review_lin300_recovered_block200_20261020.py')
UNIT = 'clt-joint-pd-lin300-recovered167-33-finalize-20261020.service'
POST_UNIT = 'clt-joint-pd-lin300-recovered6401-all-original-reviews-20261020.service'
ORIGINAL_QUALIFICATION = RESULTS / 'joint_pd_source6201_lin300_qualification_20261020/post200_original_review_qualification.json'


def read(path):
    return recovery.read(path)


def save(path, data):
    recovery.save(path, data, exclusive=True)


def resumed_load():
    runtime, saved_updates = recovery.api()
    plan = read(recovery.OUTPUT / 'recovery_plan.json')
    source = Path(plan['resume_checkpoint'])
    saved, cfg = runtime.require_source(source)
    signal = recovery.OUTPUT / 'native_learning_signal'
    load = read(signal / 'source200_runtime_load.json')
    guard = read(signal / 'leg_exploration_floor_runtime.json')
    std = saved['model_state_dict']['std'].tolist()
    if (not load['all_model_and_optimizer_tensors_bitwise_preserved']
            or load['checkpoint_sha256'] != plan['resume_checkpoint_sha256']
            or load['source_actual_updates'] != 6368 or load['all17_Adam_step'] != 127360
            or load['optimizer_reset'] or load['initial_std'] != std
            or load['optimizer_LR'] != saved['optimizer_state_dict']['param_groups'][0]['lr']
            or guard['before_std'] != std or guard['after_std'] != std
            or not guard['all17_Adam_bitwise_preserved'] or not guard['actor_and_critic_bitwise_preserved']
            or not guard['parameter_identity_preserved'] or guard['added_loss_or_backward']):
        raise ValueError('Require the exact real native6368 model/all17 Adam/LR/std load')
    return load


def qualify():
    if QUALIFICATION.exists() or REVIEW_QUALIFICATION.exists() or OUTPUT.exists():
        raise ValueError('Preserve existing recovered review qualifications/data')
    plan = read(recovery.OUTPUT / 'recovery_plan.json')
    recovery.intact(plan['source_pins'])
    load = resumed_load()
    parent = QUALIFICATION.parent
    parent.mkdir()
    fixture = verifier.qualify_fixture(parent)
    original = read(ORIGINAL_QUALIFICATION)
    recovery.intact(original['source_pins'])
    if (not original['qualified'] or original['target_actual_updates'] != 6401
            or original['target_all17_Adam_step'] != 128020
            or not original['known54_and_left24_both_required']
            or not original['original_metrics_thresholds_absolute_floors_windows_and_failed_None_preserved']
            or Path(original['output']).exists() or Path(original['static_output']).exists()):
        raise ValueError('Require all untouched original and retained review suites')
    pins = dict(plan['source_pins'])
    for p in (SCRIPT, REVIEW_SCRIPT, verifier.SCRIPT, verifier.original.SCRIPT,
            ORIGINAL_QUALIFICATION, recovery.OUTPUT / 'recovery_plan.json',
            recovery.OUTPUT / 'native_learning_signal/source200_runtime_load.json',
            recovery.OUTPUT / 'native_learning_signal/leg_exploration_floor_runtime.json'):
        pins[str(p)] = recovery.digest(p)
    proof = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training=str(recovery.OUTPUT), after_unit=recovery.UNIT, output=str(OUTPUT),
        target_actual_updates=6401, target_all17_Adam_step=128020,
        retained_original_updates=167, remaining_actual_updates=33,
        real_native6368_all17_load_verified=True, original_verifier_fixture=fixture,
        raw_audits_rewritten_or_renumbered=False, simulator_and_RNG_state_restored=False,
        original200_is_exact167_plus33_updates=True, PPO_updates_repeated=0,
        original_all657_and_retained78_static_motion_steering_and120s_still_required=True,
        source_pins=pins, acceptance_changed=False, policy_accepted=False)
    save(QUALIFICATION, proof)
    REVIEW_QUALIFICATION.parent.mkdir()
    launch_path = REVIEW_QUALIFICATION.parent / 'recovered_full_block_launch.json'
    launch = read(original['training_launch'])
    launch.update(output=str(OUTPUT), unit=UNIT, recovery_finalizer=str(SCRIPT),
        actual_updates_are_retained167_plus_resumed33=True)
    save(launch_path, launch)
    review = copy.deepcopy(original)
    review.update(observed_at=proof['observed_at'], training_launch=str(launch_path),
        training=str(OUTPUT), after_unit=UNIT, recovery_finalizer_qualification=str(QUALIFICATION),
        raw_segments_retained=True)
    review['source_pins'].update(pins)
    review['source_pins'].update({str(p): recovery.digest(p) for p in (QUALIFICATION, launch_path)})
    allowed = {'observed_at', 'training_launch', 'training', 'after_unit', 'source_pins',
        'recovery_finalizer_qualification', 'raw_segments_retained'}
    if {k: v for k, v in original.items() if k not in allowed} != {
            k: v for k, v in review.items() if k not in allowed}:
        raise ValueError('A review case, seed, threshold, window or original failure rule changed')
    save(REVIEW_QUALIFICATION, review)
    print(json.dumps(dict(qualified=True, finalizer=str(QUALIFICATION),
        original_post_review=str(REVIEW_QUALIFICATION), verifier_fixture_equal=True)), flush=True)
    return proof


def finish():
    runtime, saved_updates = recovery.api()
    proof = read(QUALIFICATION)
    recovery.intact(proof['source_pins'])
    status = read(recovery.OUTPUT / 'status.json')
    native = read(recovery.OUTPUT / 'native_learning_signal/plan.json')
    if (not proof['qualified'] or OUTPUT.exists() or status['state'] != 'segment_native_exited'
            or status['training_active'] or status['child_exit_code'] != 0
            or native['requested_updates'] != 33 or native['source_actual_ppo_updates'] != 6368
            or native['target_actual_ppo_updates'] != 6401):
        raise ValueError('Require the completed owned33 native segment before any replay')
    recovery.intact(native['source_files_sha256'])
    resumed_load()
    old_rows, old_pins, _ = recovery.records(recovery.OLD, 167, 6201)
    new_rows, new_pins, run = recovery.records(recovery.OUTPUT, 33, 6368)
    rows = old_rows + new_rows
    if [r['actual_update'] for r in rows] != list(range(6202, 6402)):
        raise ValueError('No completed update may be repeated or omitted')
    final = Path(rows[-1]['checkpoint'])
    saved, cfg = runtime.require_source(final)
    original = read(recovery.OLD / 'experiment_plan.json')
    expected = copy.deepcopy(original['training_config'])
    expected['log_dir'] = cfg['log_dir']
    if json.loads(json.dumps(expected)) != json.loads(json.dumps(cfg)) or saved_updates(final) != 6401:
        raise ValueError('The recovered saved configuration differs beyond log_dir')
    pins = {**proof['source_pins'], **old_pins, **new_pins}
    for p in (QUALIFICATION, recovery.OUTPUT / 'native_learning_signal/plan.json',
            *(run / 'params').glob('*.yaml')):
        pins[str(p)] = recovery.digest(p)
    OUTPUT.mkdir()
    signal = OUTPUT / 'native_learning_signal'
    signal.mkdir()
    save(OUTPUT / 'experiment_plan.json', dict(original, source_files=pins,
        recovery_original_segment=str(recovery.OLD), recovery_remaining_segment=str(recovery.OUTPUT),
        raw_audits_are_immutable_segment_files=True, simulator_restart_before_block_update=168,
        RNG_and_episode_state_restored=False))
    save(OUTPUT / 'recovery_integrity.json', dict(retained200_verified=True,
        old_retained_updates=167, new_observed_updates=33, all_executed_updates=200,
        PPO_updates_repeated=0, all17_resume_load_bitwise_preserved=True,
        all200_checkpoint_clocks_finite_and_continuous=True, checkpoint_rows=rows,
        raw_records_rewritten_or_renumbered=False, checkpoint=str(final),
        checkpoint_sha256=recovery.digest(final), source_pins=pins,
        simulator_restart_before_block_update=168, RNG_and_episode_state_restored=False,
        policy_accepted=False))
    for name in ('source200_runtime_load.json', 'leg_exploration_floor_runtime.json'):
        (signal / name).symlink_to(recovery.OLD / 'native_learning_signal' / name)
    save(signal / 'completed_external.json', dict(checkpoint=str(final),
        checkpoint_sha256=recovery.digest(final), actual_observed_updates=200,
        actual_ppo_updates=6401, optimizer_minibatches=4000, PPO_updates_repeated=0,
        verified_outside_Kit=True, records_sha256={r['audit']: recovery.digest(r['audit']) for r in rows},
        recovery_integrity=str(OUTPUT / 'recovery_integrity.json'), policy_accepted=False,
        limitation='Exact167+33 retained model/Adam updates; simulator episodes and RNG restarted after OS shutdown, not an uninterrupted physical trajectory.'))
    state = dict(state='independent_replay', training_active=False, replay_active=True,
        checkpoint=str(final), checkpoint_sha256=recovery.digest(final),
        actual_ppo_updates=6401, live_ppo_updates=6401, target_actual_ppo_updates=6401,
        goal_remains_active=True, policy_accepted=False)
    recovery.save(OUTPUT / 'status.json', state)
    import joint_pd_acceptance_10pct_20261017 as acceptance
    import train_source6201_lin300_200 as controller
    from train_source200_stationary_height200 import environment
    argv = runtime.replay_command(final, OUTPUT / 'replay')
    save(OUTPUT / 'replay_launch.json', dict(command=argv))
    with (OUTPUT / 'replay.log').open('x') as stream:
        subprocess.run(argv, cwd=ROOT, env=environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)
    after = acceptance.assess_current(OUTPUT / 'replay')
    save(OUTPUT / 'replay/height_leg_command_assessment.json', acceptance.assess_current_legs(OUTPUT / 'replay'))
    before = read(Path(original['source_replay']) / 'assessment.json')
    save(OUTPUT / 'stationary_review.json', controller.review(OUTPUT, after, before))
    state.update(state='agent_review_required', replay_active=False)
    recovery.save(OUTPUT / 'status.json', state)
    final_proof = verifier.verify(OUTPUT)
    print(json.dumps(dict(verified=final_proof['verified'], actual_updates=6401,
        all17_Adam_step=128020, canonical27_complete=True, original_full_review_pending=True)), flush=True)
    return final_proof


if __name__ == '__main__':
    if sys.argv[1:] == ['--qualify-only']:
        qualify()
    elif not sys.argv[1:]:
        finish()
    else:
        raise ValueError('Only qualification or completed-block finalization')
