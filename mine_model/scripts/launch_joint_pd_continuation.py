"""Launch one owned cumulative PD run from a verified saved checkpoint.

Recover from actual checkpoint files, never stale status PIDs or iteration logs.
Each launch preserves historical directories and creates a new continuation.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
from zoneinfo import ZoneInfo

import torch

from joint_pd_continuation import qualified_wheel_mass_asset, optimizer_step, tensor_finite
from watch_joint_pd_scratch import RUNS, validate_contract
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot

ROOT = Path(__file__).resolve().parents[2]


def newest_owned_checkpoint(directory):
    directory = Path(directory).resolve()
    if directory.parent != ROOT/'mine_model/results' or not directory.name.startswith('joint_pd_optimization_'):
        raise ValueError('Recovery requires an owned cumulative output directory')
    candidates = list(RUNS.glob('*_joint_pd_cumulative_' + directory.name + '_c*/model_*.pt'))
    status = json.loads((directory/'status.json').read_text())
    if status.get('checkpoint'):
        candidates.append(Path(status['checkpoint']))
    valid = []
    rejected = []
    for path in set(candidates):
        try:
            if not path.resolve().is_relative_to(RUNS.resolve()):
                raise ValueError('Checkpoint is outside the owned run root')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if (not saved.get('model_state_dict') or not tensor_finite(saved)
                    or len(saved['optimizer_state_dict']['state']) != 17):
                raise ValueError('Checkpoint or its complete Adam state is nonfinite/incomplete')
            if validate_checkpoint_robot(path)['leg_control_mode'] != 'implicit_joint_position':
                raise ValueError('Checkpoint control mode changed')
            valid.append((optimizer_step(saved), int(saved['iter']), str(path.resolve())))
        except (OSError, RuntimeError, EOFError, ValueError, KeyError, pickle.UnpicklingError) as error:
            rejected.append(dict(checkpoint=str(path), reason=repr(error)))
    if not valid:
        raise ValueError('No intact policy and Adam checkpoint found')
    return Path(max(valid)[2]), rejected


def launch(previous_directory, *, checkpoint=None, reward_overrides=None, block_updates=100,
           height_feedback_coefficient=None):
    if subprocess.run(['pgrep', '-f', '^[^ ]*python.*scripts/rsl_rl/train[.]py'],
                      stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('Training already active; do not launch a duplicate')
    previous_directory = Path(previous_directory).resolve()
    recovered, rejected = newest_owned_checkpoint(previous_directory)
    checkpoint = Path(checkpoint).resolve() if checkpoint else recovered
    if checkpoint != recovered:
        raise ValueError('Use the latest intact owned checkpoint; no silent rollback')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_contract(checkpoint.parent, fresh_weights=False)
    profile = ASSETS / MANIFEST['wheel_mass_calibration']
    asset, calibration = qualified_wheel_mass_asset(profile)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT/'mine_model/results'/f'joint_pd_optimization_{stamp}_mass_recovery_cumulative'
    output.mkdir()
    unit = f'clt-joint-pd-continuation-{stamp}.service'
    command = [str(ROOT/'run_python.sh'), str(ROOT/'mine_model/scripts/auto_train_mine.py'),
        '--checkpoint', str(checkpoint), '--output', str(output),
        '--leg-control-mode', 'implicit_joint_position', '--cumulative-learning',
        '--round_iterations', str(block_updates), '--num_envs', '8192', '--full-policy-training',
        '--min_wheel_exploration_std', '.03', '--joint_mean_guard_coef', '10.0',
        '--wheel-mass-calibration', str(profile)]
    if reward_overrides:
        command.extend(['--initial-overrides', str(Path(reward_overrides).resolve())])
    if height_feedback_coefficient is not None:
        from wheel_legged_gym_isaaclab.joint_height_feedback import validate_coefficient
        height_feedback_coefficient=validate_coefficient(height_feedback_coefficient)
        command.extend(['--joint_height_feedback_coef',str(height_feedback_coefficient)])
    stop_command = [str(ROOT/'run_python.sh'), str(ROOT/'mine_model/scripts/stop_joint_pd_continuation_service.py'),
                    '--output', str(output)]
    request = ['systemd-run', '--user', '--unit', unit,
        '--property', f'WorkingDirectory={ROOT}', '--property', 'Restart=no',
        '--property', 'TimeoutStopSec=180', '--property', 'KillMode=mixed',
        '--property', 'ExecStop='+shlex.join(stop_command),
        '--setenv', f'PYTHONPATH={ROOT}/wheel_legged_isaaclab:{ROOT}/mine_model/scripts',
        '--setenv', 'OMP_NUM_THREADS=1', '--setenv', 'MKL_NUM_THREADS=1',
        '--setenv', 'PYTHONDONTWRITEBYTECODE=1', *command]
    completed = subprocess.run(request, capture_output=True, text=True, check=True)
    record = dict(directory=str(output), previous_directory=str(previous_directory), unit=unit,
        command=command, source_checkpoint=str(checkpoint),
        source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_iteration=int(saved['iter']), optimizer_step_before=optimizer_step(saved), optimizer_states=17,
        rejected_checkpoint_candidates=rejected, wheel_mass_calibration=calibration,
        weights_reset=False, optimizer_reset=False, policy_accepted=False,
        acceptance_unchanged=True, shutdown_after_success=False,
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        launcher_stderr=completed.stderr, rewards_changed=bool(reward_overrides),
        reward_overrides_path=str(Path(reward_overrides).resolve()) if reward_overrides else None,
        height_feedback_coefficient=height_feedback_coefficient)
    (output/'launch_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    for pointer in (ROOT/'mine_model/results/joint_pd_active_resume.json', Path('/tmp/clt_joint_pd_active_resume.json')):
        temporary = pointer.with_suffix('.new')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        os.replace(temporary, pointer)
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-directory',type=Path,required=True)
    parser.add_argument('--reward-overrides',type=Path)
    args=parser.parse_args()
    record=launch(args.previous_directory,reward_overrides=args.reward_overrides)
    print(json.dumps({key:record[key] for key in ('directory','unit','source_checkpoint',
                                                'source_iteration','optimizer_step_before')},indent=2))
