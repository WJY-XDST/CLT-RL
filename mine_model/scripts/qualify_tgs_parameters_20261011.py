"""Qualify a more resolved solver; also isolate contact hysteresis from VMC.

The initial 64-iteration sweep remains failed and immutable. This is a new,
explicitly narrower experiment, not a relaxed re-labeling of that evidence.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

from run_numerical_convergence_20261011 import ROOT, OUTPUT as ORIGINAL, OVERRIDES, CRITERIA, command
from check_numerical_convergence import assess_case, compare_waveforms

PRIOR = ORIGINAL / 'qualification_p128'
OUTPUT = ORIGINAL / 'qualification_p128_v2'


def cases():
    result = [dict(name='true_static_200hz_p128', hz=200, iterations=128, kind='static', external=True, reuse=True)]
    for hz, iterations in ((400, 128), (800, 128), (200, 255)):
        result.append(dict(name=f'true_static_{hz}hz_p{iterations}', hz=hz, iterations=iterations, kind='static', external=True))
    for kind in ('length_step', 'angle_step'):
        for iterations in (128, 255):
            result.append(dict(name=f'true_{kind}_200hz_p{iterations}', hz=200, iterations=iterations, kind=kind, external=True))
    for kind in ('length_step', 'angle_step'):
        for hz in (200, 400, 800):
            result.append(dict(name=f'air_{kind}_{hz}hz_p128', hz=hz, iterations=128, kind=kind, external=True, air=True))
    for case in result:
        if not case.get('air'):
            case['reuse_path'] = str((ORIGINAL if case.get('reuse') else PRIOR) / (case['name'] + '.json'))
    return result


def case_path(case):
    return Path(case['reuse_path']) if case.get('reuse_path') else OUTPUT / (case['name'] + '.json')


def build_command(case):
    cmd = command(case)
    cmd[cmd.index('--output') + 1] = str(OUTPUT / (case['name'] + '.json'))
    if case.get('air'):
        cmd[cmd.index('--mode') + 1] = 'hanging'
        cmd[cmd.index('--length-reference') + 1] = '.2691946125'
        cmd += ['--angle-reference', '.0522195', '--zero-gravity']
        cmd.remove('--freefall-probe')
    return cmd


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / 'manifest.json'
    if not manifest.exists():
        manifest.write_text(json.dumps(dict(
            started_at=datetime.now().astimezone().isoformat(), criteria=CRITERIA,
            candidate=dict(physics_hz=200, control_hz=200, policy_hz=50, position_iterations=128,
                           velocity_iterations=8, external_forces_every_iteration=True),
            hypothesis='Finer TGS solving reduces sensitivity; contact tests and no-contact tests stay distinct',
            cases=cases(), overrides=json.loads(OVERRIDES.read_text()),
        ), indent=2) + '\n')
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab'),
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    reports, quality = {}, {}
    for case in cases():
        path = case_path(case)
        if not path.exists():
            log = OUTPUT / (case['name'] + '.log')
            if log.exists():
                raise RuntimeError('Inspect the incomplete log before retrying: ' + str(log))
            print('QUALIFY ' + case['name'], flush=True)
            with log.open('x') as stream:
                subprocess.run(build_command(case), cwd=ROOT, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=True)
        reports[case['name']] = json.loads(path.read_text())
        quality[case['name']] = assess_case(reports[case['name']], CRITERIA)
        if case.get('air'):
            # This deliberately unloaded test cannot validate gravity/contact.
            for key in ('contact_weight_balance', 'gravity_velocity', 'gravity_position'):
                quality[case['name']]['checks'].pop(key)
            quality[case['name']]['passed'] = all(quality[case['name']]['checks'].values())
        print(json.dumps(dict(case=case['name'], passed=quality[case['name']]['passed'])), flush=True)
    pairs = []
    for coarse, fine in ((200, 400), (400, 800)):
        pairs.append((f'true_static_{coarse}hz_p128', f'true_static_{fine}hz_p128'))
        for kind in ('length_step', 'angle_step'):
            pairs.append((f'air_{kind}_{coarse}hz_p128', f'air_{kind}_{fine}hz_p128'))
    for kind in ('static', 'length_step', 'angle_step'):
        pairs.append((f'true_{kind}_200hz_p128', f'true_{kind}_200hz_p255'))
    comparisons = []
    for left, right in pairs:
        pair = dict(left=left, right=right, **compare_waveforms(reports[left], reports[right]))
        pair['passed'] = pair['comparable'] and all(pair[key] <= CRITERIA[limit] for key, limit in (
            ('height_rmse_m', 'waveform_height_rmse_m'), ('length_rmse_m', 'waveform_length_rmse_m'),
            ('angle_rmse_rad', 'waveform_angle_rmse_rad'), ('torque_rmse_nm', 'waveform_torque_rmse_nm')))
        comparisons.append(pair)
    passed = all(v['passed'] for v in quality.values()) and all(p['passed'] for p in comparisons)
    output = dict(numerical_screen_passed=passed, candidate=json.loads(manifest.read_text())['candidate'],
                  criteria=CRITERIA, per_case=quality, comparisons=comparisons,
                  original_64_iteration_sweep_passed=False,
                  boundary='Qualified fixed controller only; policy and real hardware are not accepted by this gate')
    (OUTPUT / 'assessment.json').write_text(json.dumps(output, indent=2) + '\n')
    print('QUALIFICATION_COMPLETE ' + str(passed), flush=True)


if __name__ == '__main__':
    main()
