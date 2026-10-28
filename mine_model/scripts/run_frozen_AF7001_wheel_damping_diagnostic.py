"""Prepare/queue one frozen .5 versus .75 wheel-drive diagnostic.

The owned full sigma0025 review finishes first. A fresh native .5 replay
must reproduce the cached source51 before any .75 probe starts. All four
subsequent probe scopes retain their original cases, seeds, numerical gates
and failure semantics. This process cannot start PPO or select a model.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
RESULTS = ROOT / 'mine_model/results'
OUTPUT = RESULTS / 'joint_pd_frozen_AF7001_wheelD075_diagnostic_20261023'
PREPARATION = OUTPUT / 'CPU_preflight.json'
AF = RESULTS / 'joint_pd_preserved6801_yawcost200_review7001_20261022'
AF_TRAINING = RESULTS / 'joint_pd_optimization_20261022_135230_239804_preserved6801_yawcost200_200'
CURRENT = RESULTS / 'joint_pd_AF7001_yawsigma0025_review7201_20261023'
AFTER_UNIT = 'clt-joint-pd-af7001-yawsigma0025-full-reviews-20261023.service'
SOURCE_SHA = 'af47077587044230d5814f5e0e76a633366260698471da6162f3411494e64f49'
SCOPE_SPEC = (
    ('small51', 'full_height_0.25/candidate', 'motion025_full_height_diagnosis.json'),
    ('fullspeed51', 'full_height_1.0/candidate', 'motion025_full_height_diagnosis.json'),
    ('low_turn108', 'combined/candidate/low', 'combined_turn_tracking_diagnosis.json'),
    ('high_turn108', 'combined/candidate/high', 'combined_turn_tracking_diagnosis.json'),
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def modules():
    sys.path[:0] = [str(ROOT / 'mine_model/experiments'), str(ROOT / 'wheel_legged_isaaclab')]
    import frozen_AF7001_wheel_damping_probe as runtime
    import review_source5201_motion075_combined_turns as combined
    return runtime, combined


def prepare():
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    runtime, combined = modules()
    final_path = AF_TRAINING / 'final_checkpoint_verification_readonly_20261019.json'
    final = read(final_path)
    source = Path(final['checkpoint'])
    assert final['verified'] and final['checkpoint_sha256'] == sha(source) == SOURCE_SHA
    assert final['actual_ppo_updates'] == 7001 and final['all17_Adam_step'] == 140020
    saved, cfg = runtime.family.require_source(source)
    assert cfg['wheel_damping'] == .5
    assert len(saved['optimizer_state_dict']['state']) == 17
    _, evaluator, tree_proof = runtime.evaluator_tree()
    source_paths = [SCRIPT, runtime.SCRIPT, combined.SCRIPT, combined.original.SCRIPT,
        runtime.family.SCRIPT, Path(evaluator), final_path, source,
        source.parent / 'params/env.yaml', source.parent / 'params/agent.yaml',
        ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/leg_control.py',
        ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py',
        ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py']
    scopes = []
    for name, relative, report_name in SCOPE_SPEC:
        fixture = AF / relative
        manifest = read(fixture / 'manifest.json')
        prior = read(fixture / report_name)
        assert manifest['checkpoint_sha256'] == prior['checkpoint_sha256'] == SOURCE_SHA
        fn, writes, metadata = combined.reader(manifest['cases'], manifest['seeds'])
        rebuilt = fn(fixture, source)
        assert len(writes) == 1 and rebuilt['trials'] == prior['trials']
        assert rebuilt['survivors'] == rebuilt['total_trials'] == len(manifest['cases']) * 3
        command = OUTPUT / (name + '_commands.json')
        save(command, manifest['cases'])
        scopes.append(dict(name=name, fixture=str(fixture), original_report=str(fixture / report_name),
            commands=str(command), seeds=manifest['seeds'], count=rebuilt['total_trials'],
            original_numeric_metrics_flags_reproduced_exactly=True, reader_metadata=metadata))
        source_paths += [fixture / x for x in ('manifest.json', 'completed.json', 'resolved_env.json',
            'trace.csv', 'policy_inputs.npz', report_name)] + [command]
        print(json.dumps(dict(CPU_fixture_qualified=name, trials=rebuilt['total_trials'])), flush=True)
    override = OUTPUT / 'wheel_damping075_only.json'
    save(override, {'wheel_damping': .75})
    source_paths.append(override)
    pins = {str(path): sha(path) for path in source_paths}
    assert sum(scope['count'] for scope in scopes) == 318
    value = dict(qualified_CPU_only=True, observed_at=datetime.now().astimezone().isoformat(),
        checkpoint=str(source), checkpoint_sha256=SOURCE_SHA, source_actual_updates=7001,
        all17_Adam_step=140020, saved_std=saved['model_state_dict']['std'].tolist(),
        saved_actual_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        runtime=str(runtime.SCRIPT), runtime_sha256=sha(runtime.SCRIPT), scopes=scopes,
        evaluator_AST_qualification=tree_proof,
        declared_controller_factor=dict(wheel_damping=[.5, .75],
            synchronized_actuator_field='robot.actuators.wheels.damping'),
        baseline_native51_must_reproduce_cached_metrics_before_probe=True,
        baseline_metric_reproduction_tolerance=1.e-6,
        original_acceptance_gate_or_failure_semantics_changed=False,
        actual_PPO_updates=0, saved_std_or_optimizer_changed=False,
        simulation_started=False, native_damping_probe_not_yet_qualified=True,
        after_unit=AFTER_UNIT, source_pins=pins, can_extend_training=False, policy_accepted=False)
    save(PREPARATION, value)
    return value


def read_and_check_preparation():
    proof = read(PREPARATION)
    assert proof['qualified_CPU_only'] and proof['checkpoint_sha256'] == SOURCE_SHA
    assert all(sha(path) == digest for path, digest in proof['source_pins'].items())
    return proof


def native_scope(proof, scope, damping, name):
    runtime, combined = modules()
    output = OUTPUT / name
    assert not output.exists()
    request = OUTPUT / (name + '_request.json')
    command = runtime.family.replay_command(Path(proof['checkpoint']), output)
    assert command[2] == 'replay' and '--overrides_json' not in command
    values = command[3:]
    values[values.index('--seconds') + 1] = '20'
    index = values.index('--seeds') + 1
    values[index:index + 3] = list(map(str, scope['seeds']))
    values += ['--command-probe-json', scope['commands']]
    if damping != .5:
        values += ['--overrides_json', str(OUTPUT / 'wheel_damping075_only.json')]
    save(request, dict(checkpoint=proof['checkpoint'], output=str(output), wheel_damping=damping,
        frozen_diagnostic_only=True, runtime_sha256=proof['runtime_sha256'],
        command_file=scope['commands'], evaluator_AST_qualification=proof['evaluator_AST_qualification'],
        source_pins={**proof['source_pins'], str(PREPARATION): sha(PREPARATION)}))
    argv = [str(ROOT / 'run_python.sh'), proof['runtime'], '--request', str(request), '--', *values]
    save(OUTPUT / (name + '_launch.json'), dict(command=argv, wheel_damping=damping,
        checkpoint_sha256=SOURCE_SHA, frozen_diagnostic_only=True, policy_accepted=False))
    with (OUTPUT / (name + '.log')).open('x') as log:
        subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    evidence = read(output / 'frozen_wheel_damping_native_observation.json')
    assert evidence['checkpoint_sha256'] == SOURCE_SHA
    assert evidence['actual_wheel_damping_min'] == evidence['actual_wheel_damping_max'] == damping
    manifest = read(output / 'manifest.json')
    reader, captured, metadata = combined.reader(manifest['cases'], manifest['seeds'])
    result = reader(output, Path(proof['checkpoint']))
    assert len(captured) == 1 and result['total_trials'] == scope['count']
    result.update(frozen_controller_diagnostic_only=True, declared_wheel_damping=damping,
                  baseline_source_fixture=scope['fixture'], canonical_acceptance_replaced=False)
    save(output / 'metrics_diagnostic_only.json', result)
    return output, result


def run():
    state = OUTPUT / 'status.json'
    read_and_check_preparation()
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', AFTER_UNIT,
            '-p', 'MainPID', '-p', 'SubState', '-p', 'Result'], text=True, capture_output=True, check=True)
        unit = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if unit.get('MainPID') == '0':
            assert unit.get('Result') == 'success' and unit.get('SubState') == 'exited', unit
            break
        save(state, dict(observed_at=datetime.now().astimezone().isoformat(),
            state='waiting_for_verified_live_owned_full_review', unit=AFTER_UNIT, unit_state=unit,
            simulator_started=False, PPO_updates=0, policy_accepted=False))
        time.sleep(15.)
    current = read(CURRENT / 'complete_all_scopes_AF7001_yawsigma0025.json')
    assert current['completed'] and current['checkpoint_sha256'] == '47c6072c19facb03681f5187f93e1bc66669362264de8c50cb5ac8e891214c11'
    assert all(current[key] for key in ('original735_static810_original120_complete',
        'signed_fullspeed_height_complete', 'small_yaw120_complete'))
    proof = read_and_check_preparation()
    save(state, dict(state='running_fresh_frozen_baseline05_native51', PPO_updates=0, policy_accepted=False))
    output, baseline = native_scope(proof, proof['scopes'][0], .5, 'baseline05_small51')
    original = read(proof['scopes'][0]['original_report'])
    assert [t['case'] for t in baseline['trials']] == [t['case'] for t in original['trials']]
    maximum = 0.
    for new, old in zip(baseline['trials'], original['trials']):
        for key in ('seed', 'survived', 'stage_constraints_passed', 'failed_metrics', 'failed_prerequisites'):
            assert new[key] == old[key], key
        assert (new['metrics'] is None) == (old['metrics'] is None)
        if old['metrics'] is not None:
            for key, value in old['metrics'].items():
                if type(value) is bool:
                    assert new['metrics'][key] == value
                else:
                    maximum = max(maximum, abs(new['metrics'][key] - value))
    assert maximum <= proof['baseline_metric_reproduction_tolerance'], maximum
    import numpy as np
    with np.load(output / 'policy_inputs.npz', allow_pickle=False) as new, np.load(
            Path(proof['scopes'][0]['fixture']) / 'policy_inputs.npz', allow_pickle=False) as old:
        assert np.array_equal(new['observations'][0], old['observations'][0])
        assert np.array_equal(new['means'][0], old['means'][0])
    save(OUTPUT / 'baseline05_native_qualification.json', dict(qualified=True,
        original51_initial27_inputs_and_means_bitwise_equal=True, all_original_flags_identical=True,
        max_original_metric_reproduction_error=maximum, tolerance_is_reproduction_only=True,
        performance_gates_relaxed=False, PPO_updates=0, policy_accepted=False))
    completed = []
    for scope in proof['scopes']:
        save(state, dict(state='frozen_damping075_' + scope['name'], completed_scopes=completed,
                         PPO_updates=0, policy_accepted=False))
        _, report = native_scope(proof, scope, .75, 'damping075_' + scope['name'])
        completed.append(dict(scope=scope['name'], total_trials=report['total_trials'],
            survivors=report['survivors'], original_gate_passed=report['stage_all_constraints_passed']))
    read_and_check_preparation()
    save(OUTPUT / 'completed.json', dict(completed=True,
        observed_at=datetime.now().astimezone().isoformat(), baseline51_native_reproduced=True,
        probe_trials=318, probe_scopes=completed, checkpoint_sha256=SOURCE_SHA,
        controller_gain_is_diagnostic_not_selected=True, source_files_unchanged=True,
        PPO_updates=0, can_extend_training=False, can_select_source=False, policy_accepted=False))
    save(state, dict(state='frozen_controller_diagnostic_complete_agent_decision_required',
                     PPO_updates=0, policy_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assert args.prepare != args.run
    if args.prepare:
        prepare()
    else:
        try:
            run()
        except BaseException as error:
            save(OUTPUT / 'agent_inspection_required.json', dict(
                observed_at=datetime.now().astimezone().isoformat(), error=repr(error),
                PPO_restarted=False, PPO_updates=0, policy_accepted=False))
            raise
