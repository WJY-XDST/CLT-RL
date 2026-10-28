"""Pair frozen6201/6401 full-speed height changes after all original reviews.

This supplements the retained tests; it changes no training, controller,
threshold or original report. Saved-Gaussian numeric results are descriptive.
"""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from joint_pd_continuation import ROOT, sha
from audit_saved_action_limits import transform
from review_plain27_single_reward_block200 import rebound
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_plan
import review_wheel125_training_block as core
import joint_pd_source6201_lin300_runtime as runtime
import audit_completed_motion_phase_errors as phase_raw
import audit_original_and_literal_relative_errors as literal

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261020_195952_083563_source6201_lin300_200'
QUALIFICATION = RESULTS / 'joint_pd_source6201_lin300_qualification_20261020/fullspeed_height_transitions/qualification.json'
ORIGINAL_QUALIFICATION = QUALIFICATION.parent.parent / 'post200_original_review_qualification.json'
ORIGINAL_COMPLETE = RESULTS / 'joint_pd_lin300_review6401_20261020/complete_with_retained_interior_reviews.json'
OUTPUT = RESULTS / 'joint_pd_lin300_fullspeed_height_transitions_source6201_candidate6401_20261020'
SEEDS = [321483, 321484, 321485]
AFTER_UNIT = 'clt-joint-pd-source6401-lin300-original-and-retained-reviews-after200-20261020.service'
DECLARATIONS = [
    ("np.any(rows['speed_cmd_m_s'] < -1.e-7)",
     "np.any(rows['speed_cmd_m_s'] < min(0., initial['speed']) - 1.e-7)"),
    ("np.any(rows['speed_cmd_m_s'] > initial['speed'] + 1.e-7)",
     "np.any(rows['speed_cmd_m_s'] > max(0., initial['speed']) + 1.e-7)"),
]


def read(path):
    return json.loads(Path(path).read_text())


def plan():
    base = core.static_steps.step_plan()
    heights = sorted({case['height'] for phase in base['phases'] for case in phase})
    if len(heights) != 3:
        raise ValueError('Requalify the original three safe height values')
    motions = [('forward08_right03', .8, -.3), ('forward08_straight', .8, 0.),
        ('forward08_left03', .8, .3), ('reverse08_right03', -.8, -.3),
        ('reverse08_straight', -.8, 0.), ('reverse08_left03', -.8, .3),
        ('reverse08_left0075', -.8, .075), ('forward08_right0075', .8, -.075),
        ('static_control', 0., 0.)]
    return dict(phase_seconds=base['phase_seconds'], phases=[[
        dict(name=name, speed=speed, yaw=yaw, height=heights[(index + i) % 3])
        for i, (name, speed, yaw) in enumerate(motions)] for index in range(len(base['phases']))])


def reader(specification, seeds, capture=None):
    # Reverse speeds were outside this historical reader's positive-only
    # command assertion. Its seven metrics, limits,151-frame windows, survival
    # and failed-None rules are retained byte-for-AST after undoing these two
    # signed command-envelope expressions.
    tree = transform(core.numeric.assess, DECLARATIONS)
    scope = dict(core.numeric.__dict__, plan=lambda: specification, SEEDS=seeds)
    if capture is not None:
        scope['write_json'] = lambda path, value: capture.append((str(path), value))
    exec(compile(ast.fix_missing_locations(tree), str(core.numeric.SCRIPT), 'exec'), scope)
    return scope['assess']


def qualify():
    if QUALIFICATION.exists() or OUTPUT.exists():
        raise ValueError('Preserve every published qualification and native result')
    original = read(ORIGINAL_QUALIFICATION)
    final = read(runtime.TRAINING / 'final_checkpoint_verification_readonly_20261019.json')
    source = Path(final['checkpoint'])
    _, cfg = runtime.previous.require_source(source)
    if (not original['qualified'] or not final['verified'] or final['actual_ppo_updates'] != 6201
            or final['all17_Adam_step'] != 124020 or sha(source) != runtime.ANCHOR_SHA
            or original['target_actual_updates'] != 6401 or original['target_all17_Adam_step'] != 128020
            or cfg['commands']['training_motion_scale'] != 1. or cfg['wheel_action_limit'] != 1.25
            or cfg['min_root_height'] != .22 or cfg['low_height_terminal_time_s'] != .15):
        raise ValueError('Require the exact current full-speed source and already-qualified200 review')
    core.intact(original['source_pins'])
    checked, sequence, period = command_probe_plan(plan(), SimpleNamespace(**cfg['commands']))
    if len(checked) != 9 or len(sequence) != 10 or period != 5.:
        raise ValueError('Keep the existing9x3x10,50-second native phase contract')
    expected_heights = {case['height'] for phase in core.static_steps.step_plan()['phases'] for case in phase}
    for index in range(9):
        if {phase[index]['height'] for phase in plan()['phases']} != expected_heights:
            raise ValueError('Every signed trajectory must visit every original height')
    # Exercise the extended assertion on the full original positive-speed
    # dataset without rewriting any earlier report or trace.
    reference = RESULTS / 'joint_pd_orientation060_review6201_20261020/moving_height/deterministic'
    capture = []
    rebuilt = reader(original['moving_plan'], original['moving_seeds'], capture)(reference, source, False)
    if len(capture) != 1 or rebuilt != read(reference / 'motion_height_steps_diagnosis.json'):
        raise ValueError('Historical27-episode/270-phase numeric report changed')
    valid_bounds = []
    for speed in (-.8, 0., .8):
        for value in (min(0., speed), .5 * speed, max(0., speed)):
            if value < min(0., speed) - 1.e-7 or value > max(0., speed) + 1.e-7:
                raise ValueError('A valid signed ramp value was rejected')
        for value in (min(0., speed) - .01, max(0., speed) + .01):
            if not (value < min(0., speed) - 1.e-7 or value > max(0., speed) + 1.e-7):
                raise ValueError('A signed ramp value outside the envelope was admitted')
        valid_bounds.append([min(0., speed), max(0., speed)])
    paths = [SCRIPT, ORIGINAL_QUALIFICATION, runtime.SCRIPT, runtime.previous.SCRIPT,
        core.SCRIPT, core.numeric.SCRIPT, phase_raw.SCRIPT, literal.SCRIPT, literal.GATE,
        Path(transform.__code__.co_filename), Path(command_probe_plan.__code__.co_filename),
        runtime.TRAINING / 'final_checkpoint_verification_readonly_20261019.json', source,
        TRAINING / 'agent_learning_exploration_stability_review_prefix0050_20261020.json',
        TRAINING / 'terminal_event_ages_prefix_0050_readonly.json',
        TRAINING / 'terminal_joint_height_budget_prefix_0050_readonly.json',
        reference / 'manifest.json', reference / 'completed.json', reference / 'trace.csv',
        reference / 'resolved_env.json', reference / 'motion_height_steps_diagnosis.json']
    pins = {str(p): sha(p) for p in paths}
    proof = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        after_unit=AFTER_UNIT, training=str(TRAINING), output=str(OUTPUT),
        original_complete_required=str(ORIGINAL_COMPLETE), source_checkpoint=str(source),
        source_checkpoint_sha256=runtime.ANCHOR_SHA, source_actual_updates=6201,
        candidate_actual_updates=6401, candidate_all17_Adam_step=128020,
        command_plan=plan(), seeds=SEEDS, episodes_per_run=27, phases_per_run=270,
        samples_per_run=67500, modes=['deterministic', 'saved_Gaussian'],
        original270_phase_report_reproduced_exactly=True, numeric_reader_signed_assertion=DECLARATIONS,
        all_other_numeric_reader_AST_restored_identical=True, signed_ramp_bounds=valid_bounds,
        reset_velocity_amplitude_unchanged=.05, physical_min_root_height_unchanged=.22,
        low_height_terminal_time_unchanged=.15, original_all_scope_reviews_not_replaced=True,
        sampled_numeric_gates_descriptive_original_stress_gate_survival_only=True,
        exact_native_event_reproduction_or_initiating_cause_not_claimed=True,
        source_pins=pins, simulator_started=False, optimizer_executed=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False,
        can_select_or_extend_training=False)
    core.intact(pins)
    QUALIFICATION.parent.mkdir(parents=True, exist_ok=True)
    core.save(QUALIFICATION, proof)
    return proof


def run():
    proof = read(QUALIFICATION)
    core.intact(proof['source_pins'])
    complete = read(ORIGINAL_COMPLETE)
    final = read(TRAINING / 'final_checkpoint_verification_readonly_20261019.json')
    if (not proof['qualified'] or OUTPUT.exists() or not complete['all_original_and_retained_reviews_complete']
            or not final['verified'] or final['actual_ppo_updates'] != 6401
            or final['all17_Adam_step'] != 128020 or complete['checkpoint_sha256'] != final['checkpoint_sha256']):
        raise ValueError('Require complete original/retained6401 evidence before this supplement')
    sources = {'source6201': Path(proof['source_checkpoint']), 'candidate6401': Path(final['checkpoint'])}
    runtime.previous.require_source(sources['source6201'])
    runtime.require_source(sources['candidate6401'])
    OUTPUT.mkdir()
    core.save(OUTPUT / 'plan.json', proof)
    results = {}
    for label, source in sources.items():
        owner = runtime.previous if label == 'source6201' else runtime
        native = rebound(core.native, runtime=owner, QUALIFICATION=QUALIFICATION)
        entries = []
        for mode, sampled in [('deterministic', False), ('saved_Gaussian', True)]:
            core.save(OUTPUT / (label + '_' + mode + '_started.json'), dict(
                checkpoint=str(source), checkpoint_sha256=sha(source), mode=mode,
                training_modified=False, policy_accepted=False))
            directory = native(source, OUTPUT / label / mode, 50., SEEDS, proof['command_plan'], sampled)
            assessed = reader(proof['command_plan'], SEEDS)(directory, source, sampled)
            if label == 'candidate6401':
                with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as a, np.load(
                        OUTPUT / 'source6201' / mode / 'policy_inputs.npz', allow_pickle=False) as b:
                    if not np.array_equal(a['observations'][0], b['observations'][0]):
                        raise ValueError('Paired initial policy inputs changed')
            entries.append(dict(mode=mode, episodes=27, survivors=assessed['survivors'],
                resets=assessed['resets'], all_numeric_phase_gates_passed=assessed['all_numeric_phase_gates_passed']))
            core.intact(proof['source_pins'])
        raw_path = OUTPUT / label / 'all540_phases_raw_verified.json'
        raw = phase_raw.audit([OUTPUT / label / mode for mode in proof['modes']], raw_path)
        raw.update(actual_ppo_updates=6201 if label == 'source6201' else 6401,
            actual_update_clock_verified_from_completed_checkpoint=True)
        # Publish the clock before its file digest becomes a literal-audit input.
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        literal_path = OUTPUT / label / 'all540_phases_literal10.json'
        relative = literal.audit(raw_path, literal_path)
        results[label] = dict(checkpoint=str(source), checkpoint_sha256=sha(source), modes=entries,
            raw_report=str(raw_path), literal_report=str(literal_path),
            original_phase_passed=raw['total_passed'],
            original_and_literal_phase_passed=relative['original_gates_and_nonzero_relative10_both_passed'])
    core.intact(proof['source_pins'])
    result = dict(observed_at=datetime.now().astimezone().isoformat(), completed=True,
        results=results, candidate_initial_observations_bitwise_paired=True,
        original_scope_not_replaced=True, training_modified=False, optimizer_executed=False,
        acceptance_changed=False, policy_accepted=False, can_select_or_extend_training=False,
        sampled_numeric_results_descriptive_survival_stress_criterion_unchanged=True)
    core.save(OUTPUT / 'complete_paired_fullspeed_height_transitions.json', result)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    result = qualify() if args.qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'completed', 'policy_accepted') if k in result}))
