"""Compare the posture repair with the retained source1601 frozen traces.

Run only after the exact200 native updates and original full27 replay finish.
Reuse all51 source1601 trials; preserve the original strict criteria and every
height endpoint. Waiting observes one exact systemd unit and never restarts it.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
RESULTS = ROOT / 'mine_model/results'
BASELINE = RESULTS / 'joint_pd_motion025_pair_20261017_source1601_full_height/candidate'
QUALIFICATION = RESULTS / 'joint_pd_motion025_orientation80_review_qualification_20261017.json'
EXPECTED_UNIT = 'clt-joint-pd-source1601-motion025-orientation80-200-20261017-184021-781538.service'
EXPECTED_TRAINING = RESULTS / 'joint_pd_optimization_20261017_184021_781538_source1601_motion025_orientation80_200'


def dependencies():
    # Keep the unit waiter small while the existing simulator owns memory.
    import source1201_orientation40
    import diagnose_motion025_full_height as original
    import resume_motion025_full_height_diagnosis as corrected
    import joint_pd_source1601_motion025_orientation80_runtime as repair
    import run_static_reset_pair_diagnosis as boundary
    from auto_train_mine import write_json
    from joint_pd_continuation import sha
    from train_joint_pd_standing_seed import saved_updates
    return original, corrected, repair, boundary, write_json, sha, saved_updates


def baseline_readonly(api):
    original, corrected, repair, _, _, sha, _ = api
    report_path = BASELINE / 'motion025_full_height_diagnosis.json'
    report = json.loads(report_path.read_text())
    source = Path(report['checkpoint'])
    if sha(source) != repair.ANCHOR_SHA or report['checkpoint_sha256'] != repair.ANCHOR_SHA:
        raise ValueError('Use only the intact completed source1601 paired baseline')
    _, cfg = original.motion.require_source(source)
    if cfg['commands']['ranges_height'] != list(original.SAFE_BASE_HEIGHT_RANGE_M):
        raise ValueError('Saved baseline must retain the full safe height range')
    pins = {**report['source_files'], str(report_path): sha(report_path),
            str(original.SCRIPT): sha(original.SCRIPT), str(corrected.SCRIPT): sha(corrected.SCRIPT)}
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('Retained native baseline evidence changed')
    reader = corrected.corrected_reader()
    # Recompute every trial directly from CSV without rewriting pinned evidence.
    writes = []
    reader.__globals__['write_json'] = lambda path, value: writes.append((Path(path), value))
    recomputed = reader(BASELINE, source)
    if (len(writes) != 1 or writes[0][0] != report_path
            or writes[0][1] != recomputed or recomputed != report
            or any(sha(path) != digest for path, digest in pins.items())):
        raise ValueError('Read-only baseline reconstruction differs from its retained report')
    return source, cfg, report, pins


def command(api, source, output, commands):
    original, _, repair, _, _, _, _ = api
    argv = repair.replay_command(source, output)
    argv[argv.index('--seconds') + 1] = str(original.SECONDS)
    index = argv.index('--seeds') + 1
    argv[index:index + 3] = [str(seed) for seed in original.SEEDS]
    argv.extend(['--command-probe-json', str(commands)])
    banned = ('--command-scale', '--overrides_json', '--sample-policy-actions',
              '--heading-hold', '--export_dir', repair.FLAG)
    if any(flag in argv for flag in banned) or argv.count(str(repair.SCRIPT)) != 1:
        raise ValueError('Require the exact saved descendant and unscaled deterministic commands')
    return argv


def qualify(checkpoint, output):
    api = dependencies()
    original, corrected, repair, _, write_json, sha, saved_updates = api
    _, _, baseline, pins = baseline_readonly(api)
    _, cfg = repair.require_source(checkpoint)
    if cfg['commands']['ranges_height'] != list(original.SAFE_BASE_HEIGHT_RANGE_M):
        raise ValueError('Qualification descendant lost the full height contract')
    from types import SimpleNamespace
    from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases
    checked = command_probe_cases(original.cases(), SimpleNamespace(**cfg['commands']))
    argv = command(api, checkpoint, RESULTS / 'orientation80_review_cpu_not_native',
                   RESULTS / 'orientation80_review_cpu_not_native/commands.json')
    result = dict(qualified=True, created_at=datetime.now().astimezone().isoformat(),
        script_sha256=sha(SCRIPT), runtime_sha256=sha(repair.SCRIPT),
        original_driver_sha256=sha(original.SCRIPT), corrected_reader_sha256=sha(corrected.SCRIPT),
        baseline_native_samples_reused=51000, baseline_recomputed_from_raw_csv=True,
        baseline_report_identical=True, baseline_data_unchanged=True,
        baseline_source_pins=pins, baseline_checkpoint_sha256=baseline['checkpoint_sha256'],
        qualification_checkpoint=str(Path(checkpoint).resolve()),
        qualification_checkpoint_actual_updates=saved_updates(checkpoint),
        cases=checked, seeds=original.SEEDS, seconds=original.SECONDS,
        full_height_range=list(original.SAFE_BASE_HEIGHT_RANGE_M),
        expected_training_output=str(EXPECTED_TRAINING), expected_unit=EXPECTED_UNIT,
        reviewed_candidate_actual_updates=1801, reviewed_candidate_all17_Adam_step=36020,
        frozen_command=argv, PPO_updates_performed=0, native_replay_started=False,
        original_thresholds_changed=False, policy_accepted=False)
    output = Path(output).resolve()
    if output != QUALIFICATION or output.exists():
        raise ValueError('Preserve the one owned qualification proof')
    write_json(output, result)
    return result


def unchanged(api, pins):
    sha = api[5]
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('A pinned implementation, checkpoint, criterion or trace changed')


def wait_training(unit, output):
    if unit != EXPECTED_UNIT:
        raise ValueError('Observe only the exact already-running orientation80 unit')
    output = Path(output).resolve()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_motion025_orientation80_full_height_'):
        raise ValueError('Use a fresh owned supplemental output')
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
            '--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus'],
            text=True, capture_output=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if state.get('LoadState') != 'loaded':
            raise ValueError('The observed unit handle is missing; inspect without restarting')
        active = state.get('ActiveState')
        if active in ('active', 'activating', 'deactivating', 'reloading'):
            print(json.dumps(dict(time=datetime.now().astimezone().isoformat(),
                state='waiting_exact_training_unit', unit=unit, observed=state,
                training_restarted=False, native_replay_started=False)), flush=True)
            time.sleep(30.)
            continue
        if active != 'inactive' or int(state.get('ExecMainStatus', '-1')) != 0:
            raise ValueError('Training unit is terminal without successful completion; inspect its evidence')
        return


def run(output):
    api = dependencies()
    original, corrected, repair, boundary, write_json, sha, saved_updates = api
    output = Path(output).resolve()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_motion025_orientation80_full_height_'):
        raise ValueError('Use a fresh owned supplemental output and preserve prior evidence')
    proof = json.loads(QUALIFICATION.read_text())
    if (not proof['qualified'] or proof['script_sha256'] != sha(SCRIPT)
            or proof['runtime_sha256'] != sha(repair.SCRIPT)
            or proof['original_driver_sha256'] != sha(original.SCRIPT)
            or proof['corrected_reader_sha256'] != sha(corrected.SCRIPT)
            or proof['original_thresholds_changed']):
        raise ValueError('Require the intact CPU qualification before any new replay')
    baseline_source, before_cfg, before, pins = baseline_readonly(api)
    unchanged(api, proof['baseline_source_pins'])
    prior = boundary.smooth
    boundary.smooth = repair
    try:
        candidate, entry = boundary.reviewed_source(EXPECTED_TRAINING)
    finally:
        boundary.smooth = prior
    if entry != repair.SCRIPT or saved_updates(candidate) != 1801:
        raise ValueError('Require the completed200/full27/full17 source1801 repair')
    _, candidate_cfg = repair.require_source(candidate)
    comparable = copy.deepcopy(before_cfg)
    comparable['log_dir'] = candidate_cfg['log_dir']
    comparable['rewards']['orientation'] = -80.
    if comparable != candidate_cfg or before_cfg['rewards']['orientation'] != -40.:
        raise ValueError('Saved baseline and candidate differ beyond the single orientation cost')
    plan = json.loads((EXPECTED_TRAINING / 'experiment_plan.json').read_text())
    pins.update(plan['source_files'])
    pins.update({str(p): sha(p) for p in (SCRIPT, repair.SCRIPT, candidate, QUALIFICATION,
        EXPECTED_TRAINING / 'status.json', EXPECTED_TRAINING / 'stationary_review.json',
        EXPECTED_TRAINING / 'native_learning_signal/completed_external.json',
        EXPECTED_TRAINING / 'replay/assessment.json')})
    unchanged(api, pins)
    output.mkdir()
    commands = output / 'command_plan.json'
    write_json(commands, original.cases())
    pins[str(commands)] = sha(commands)
    write_json(output / 'plan.json', dict(created_at=datetime.now().astimezone().isoformat(),
        training_output=str(EXPECTED_TRAINING), checkpoint=str(candidate),
        baseline_checkpoint=str(baseline_source), baseline_raw_directory=str(BASELINE),
        baseline_samples_reused=51000, baseline_native_repeated=False, source_pins=pins,
        cases=original.cases(), seeds=original.SEEDS, seconds=original.SECONDS,
        full_height_range=list(original.SAFE_BASE_HEIGHT_RANGE_M),
        canonical_full27_already_completed=True, thresholds_changed=False,
        training_modified=False, policy_accepted=False, hardware_enabled=False))
    destination = output / 'candidate'
    argv = command(api, candidate, destination, commands)
    write_json(output / 'launch.json', dict(command=argv, checkpoint_sha256=sha(candidate)))
    write_json(output / 'status.json', dict(state='candidate_native_replay',
        diagnostic_active=True, training_active=False, policy_accepted=False))
    with (output / 'candidate.log').open('w') as log:
        subprocess.run(argv, cwd=ROOT, env=original.environment(),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    after = corrected.corrected_reader()(destination, candidate)
    unchanged(api, pins)
    native_before = json.loads((BASELINE / 'resolved_env.json').read_text())
    native_after = json.loads((destination / 'resolved_env.json').read_text())
    comparable = copy.deepcopy(native_before)
    comparable['log_dir'] = native_after['log_dir']
    comparable['rewards']['orientation'] = -80.
    if comparable != native_after or native_before['rewards']['orientation'] != -40.:
        raise ValueError('Native replay configurations differ beyond the declared reward cost')
    paired = []
    for left, right in zip(before['trials'], after['trials']):
        if (left['case'], left['seed']) != (right['case'], right['seed']):
            raise ValueError('Paired trial order, commands or seeds differ')
        changes = ({key: right['metrics'][key] - left['metrics'][key] for key in right['limits']}
                   if left['survived'] and right['survived'] else None)
        paired.append(dict(case=right['case']['name'], seed=right['seed'], metric_changes=changes,
            baseline_survived=left['survived'], candidate_survived=right['survived']))
    summary_keys = ('total_trials', 'survivors', 'resets', 'stage_all_constraints_passed')
    result = dict(baseline_checkpoint_sha256=sha(baseline_source), checkpoint_sha256=sha(candidate),
        results={label: {key: report[key] for key in summary_keys}
                 for label, report in (('source1601', before), ('orientation80_candidate1801', after))},
        paired_metric_changes=paired, source_pins=pins, baseline_native_samples_reused=51000,
        baseline_native_repeated=False, only_saved_configuration_change={'rewards.orientation': [-40., -80.]},
        paired_native_physics_and_control_equal=True, thresholds_changed=False,
        canonical_acceptance_replaced=False, automatic_source_selection=False,
        training_modified=False, policy_accepted=False)
    write_json(output / 'paired_motion025_orientation80_full_height.json', result)
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-checkpoint', type=Path)
    parser.add_argument('--qualification-output', type=Path, default=QUALIFICATION)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wait-training-unit')
    args = parser.parse_args()
    if args.qualify_checkpoint:
        result = qualify(args.qualify_checkpoint, args.qualification_output)
    else:
        if args.output is None:
            parser.error('--output is required for a native review')
        if args.wait_training_unit:
            wait_training(args.wait_training_unit, args.output)
        result = run(args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
