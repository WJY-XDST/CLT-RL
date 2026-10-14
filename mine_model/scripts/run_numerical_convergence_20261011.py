"""Run a bounded fixed-controller sweep; never tune gains or training weights.

Physics steps vary while the VMC stays at 200 Hz and actions stay at 50 Hz.
Each case uses a fresh simulator process. Output names cannot overwrite a run.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'mine_model/results/tgs_numerical_convergence_20261011'
OVERRIDES = ROOT / 'mine_model/results/mcu_standing_restart_20261010/protected_stops_replay_overrides.json'
CRITERIA = {
    'controller_hz': 200, 'policy_hz': 50,
    'static_seconds': 5., 'step_seconds': 5., 'step_time_s': 2.,
    'fixed_length_m': .270, 'length_step_m': .005, 'angle_step_rad': .020,
    'static_height_spread_m': .001,
    'static_torque_spread_nm': .5,
    'waveform_height_rmse_m': .002,
    'waveform_length_rmse_m': .001,
    'waveform_angle_rmse_rad': .01,
    'waveform_torque_rmse_nm': 1.,
    'hinge_gap_max_m': .0005,
    'hinge_axis_max_deg': .2,
    'steady_joint_velocity_disagreement_p95_rad_s': .05,
    'steady_contact_weight_relative_error': .05,
    'freefall_velocity_error_m_s': .005,
    'freefall_position_error_m': .002,
}


def cases():
    result = []
    for hz in (200, 400, 800):
        result.append(dict(name=f'true_static_{hz}hz_p64', hz=hz, iterations=64, kind='static', external=True))
    for iterations in (32, 128):
        result.append(dict(name=f'true_static_200hz_p{iterations}', hz=200, iterations=iterations, kind='static', external=True))
    result.append(dict(name='false_static_200hz_p64', hz=200, iterations=64, kind='static', external=False))
    for kind in ('length_step', 'angle_step'):
        for hz in (200, 400, 800):
            result.append(dict(name=f'true_{kind}_{hz}hz_p64', hz=hz, iterations=64, kind=kind, external=True))
    return result


def command(case):
    cmd = [str(ROOT / 'run_python.sh'), str(ROOT / 'wheel_legged_isaaclab/scripts/check_mcu_vmc_support.py'),
           '--overrides', str(OVERRIDES), '--output', str(OUTPUT / (case['name'] + '.json')),
           '--mode', 'slider_guide', '--seconds', '5', '--physics-hz', str(case['hz']),
           '--control-hz', '200', '--position-iterations', str(case['iterations']), '--velocity-iterations', '8',
           '--length-reference', '.270', '--step-time', '2', '--freefall-probe', '--stop-on-failure',
           '--external-forces-every-iteration' if case['external'] else '--no-external-forces-every-iteration',
           '--headless', '--device', 'cuda:0']
    if case['kind'] == 'length_step':
        cmd += ['--length-step', '.005']
    if case['kind'] == 'angle_step':
        cmd += ['--angle-step', '.020']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[case['name'] for case in cases()])
    parser.add_argument('--remaining', action='store_true', help='Skip only successfully completed case outputs')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / 'manifest.json'
    if not manifest.exists():
        manifest.write_text(json.dumps(dict(
            started_at=datetime.now().astimezone().isoformat(), criteria=CRITERIA,
            overrides=json.loads(OVERRIDES.read_text()),
            cases=[{**case, 'command': command(case)} for case in cases()],
            controller='fixed VMC; no policy; explicit leg effort; physical frictionless Z slider',
            training_parameters_modified=False, robot_assets_modified=False,
        ), ensure_ascii=False, indent=2) + '\n')
    else:
        saved = json.loads(manifest.read_text())
        if saved['criteria'] != CRITERIA or saved['overrides'] != json.loads(OVERRIDES.read_text()):
            raise RuntimeError('Existing sweep has a different experiment contract')
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab'),
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    for case in cases():
        if args.case and case['name'] != args.case:
            continue
        output = OUTPUT / (case['name'] + '.json')
        if output.exists():
            json.loads(output.read_text())  # Incomplete JSON is not a successful case.
            if args.remaining:
                continue
            raise RuntimeError('Refusing to overwrite completed diagnostic: ' + str(output))
        log = OUTPUT / (case['name'] + '.log')
        if log.exists():
            raise RuntimeError('Inspect the existing failed/incomplete log before retrying: ' + str(log))
        print('START ' + case['name'], flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command(case), cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode or not output.exists():
            raise RuntimeError(f'Diagnostic failed: {case["name"]}, exit={result.returncode}; see {log}')
        report = json.loads(output.read_text())
        print(json.dumps(dict(case=case['name'], survived=report['survived_without_reset'],
                              first_failure_s=report['first_failure_s']), ensure_ascii=False), flush=True)
    print('SWEEP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
