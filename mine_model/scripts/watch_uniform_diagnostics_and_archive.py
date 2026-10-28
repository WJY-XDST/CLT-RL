"""Run unchanged CPU readers and archive after one owned native PPO block.

This lightweight waiter imports no Torch/Kit and never starts, stops, selects,
or restarts PPO. The existing original11 readers remain owned by the training
controller. Failures require inspection instead of repeating training.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
STAGE = SCRIPT.with_name('audit_stage_normalized_training_yaw.py')
STD = SCRIPT.with_name('audit_uniform_std_floor_pressure.py')
ARCHIVE = SCRIPT.with_name('preserve_uniform_joint_pd_candidate.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wait_for(path, unit, status, phase):
    while not path.exists():
        result = subprocess.run(['systemctl', '--user', 'show', unit,
            '--property=MainPID,SubState,Result'], capture_output=True, text=True, check=True)
        state = dict(row.split('=', 1) for row in result.stdout.splitlines() if '=' in row)
        assert state['MainPID'] != '0' and state['Result'] == 'success', (
            'Owned controller ended before required evidence; inspect without repeating PPO', state)
        write(status, dict(observed_at=datetime.now().astimezone().isoformat(),
            phase=phase, waiting_for=str(path), owned_unit=unit, live_owner=state,
            training_modified=False, PPO_not_automatically_restarted=True, policy_accepted=False))
        time.sleep(15.)


def invoke(script, arguments, log):
    with log.open('x') as stream:
        subprocess.run([str(ROOT / 'run_python.sh'), str(script), *map(str, arguments)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run(args):
    output, recipe, archive = args.output.resolve(), args.recipe.resolve(), args.archive.resolve()
    assert output.parent == ROOT / 'mine_model/results'
    assert archive.parent == recipe.parent and not archive.exists()
    status = recipe.parent / 'owned_diagnostics_and_archive_status.json'
    pins = {str(p): sha(p) for p in (SCRIPT, STAGE, STD, ARCHIVE, recipe)}
    completed, reports = [], []
    plan = output / 'experiment_plan.json'
    wait_for(plan, args.unit, status, 'wait_for_formal_plan_after_capacity')
    spec = read(plan)
    assert spec['bounded_training_updates'] == 200 and spec['ordinary_full_policy_PPO']
    assert not spec['auxiliary_losses_enabled']
    pins[str(plan)] = sha(plan)
    for prefix in (50, 100, 200):
        health = output / f'policy_update_health_prefix_{prefix:04d}_readonly.json'
        original = output / f'sampled_yaw_reference_components_prefix_{prefix:04d}_readonly.json'
        for path in (health, original):
            wait_for(path, args.unit, status, f'wait_for_original_prefix{prefix}')
        assert read(health)['committed_updates'] == prefix
        stage = output / f'stage_relative_training_yaw_prefix_{prefix:04d}_readonly.json'
        pressure = output / f'uniform_std_floor_pressure_prefix_{prefix:04d}_readonly.json'
        assert not stage.exists() and not pressure.exists()
        invoke(STAGE, [output, '--through', prefix, '--output', stage],
               output / f'stage_relative_training_yaw_prefix_{prefix:04d}.log')
        invoke(STD, [output, '--through', prefix],
               output / f'uniform_std_floor_pressure_prefix_{prefix:04d}.log')
        for path in (stage, pressure):
            report = read(path)
            assert report['verified'] and not report['training_modified']
            assert not report['optimizer_executed'] and not report['policy_accepted']
            assert all(sha(p) == digest for p, digest in report['source_pins'].items())
            pins[str(path)] = sha(path)
            reports.append(str(path))
        completed.append(prefix)
        write(status, dict(phase='original_prefix_diagnostics_complete', completed_prefixes=completed,
            reports=reports, training_modified=False, policy_accepted=False))
    proof = output / 'final_checkpoint_verification_readonly.json'
    wait_for(proof, args.unit, status, 'wait_for_verified_complete200')
    assert read(proof)['verified'] and read(proof)['all200_native_updates_and_original_alignment_verified']
    invoke(ARCHIVE, ['--recipe', recipe, '--proof', proof, '--output', archive],
           recipe.parent / 'candidate_preservation.log')
    manifest = archive / 'preservation_manifest.json'
    assert read(manifest)['verified']
    pins[str(manifest)] = sha(manifest)
    assert all(sha(p) == digest for p, digest in pins.items())
    write(recipe.parent / 'owned_diagnostics_and_archive_complete.json', dict(verified=True,
        observed_at=datetime.now().astimezone().isoformat(), completed_prefixes=completed,
        archive_manifest=str(manifest), reports=reports, source_pins=pins,
        original_readers_unchanged=True, masks_enter_training=False,
        training_modified=False, PPO_not_automatically_restarted=True, policy_accepted=False))
    write(status, dict(phase='complete200_diagnostics_and_archive_finished',
        completed_prefixes=completed, archive_manifest=str(manifest),
        training_modified=False, policy_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        write(args.recipe.resolve().parent / 'owned_diagnostics_and_archive_inspection_required.json',
            dict(observed_at=datetime.now().astimezone().isoformat(), error=repr(error),
                 training_modified=False, PPO_not_automatically_restarted=True, policy_accepted=False))
        raise
