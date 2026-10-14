"""Launch interactive replay with a matched checkpoint + training overrides.

By default choose the latest completed evaluation, not a half-written training
checkpoint or the old 'best' policy. This script does not start training.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'wheel_legged_isaaclab'))
from wheel_legged_gym_isaaclab.leg_control import require_leg_control_mode
from wheel_legged_gym_isaaclab.config_overrides import strip_retired_config_fields
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
from wheel_legged_gym_isaaclab.reward_signals import DIRECT_JOINT_REWARD_PROFILE


def checkpoint_config(checkpoint):
    saved = validate_checkpoint_robot(checkpoint)
    require_leg_control_mode(saved.get('leg_control_mode'))
    return saved


def latest_completed_assessment(root, output=None):
    """Never select an in-flight checkpoint or a retired IK result."""
    if output is None:
        # 优化任务只回放当前保留模型，不选最近生成但已被淘汰的对照分支。
        states = []
        for folder in (root / 'mine_model/results').glob('joint_pd_optimization_*'):
            status = folder / 'status.json'
            if not status.is_file():
                continue
            state = json.loads(status.read_text())
            assessment = Path(state.get('best_assessment', ''))
            if assessment.is_file() and state.get('best_checkpoint'):
                report = json.loads(assessment.read_text())
                if str(report.get('checkpoint')) == state['best_checkpoint']:
                    states.append((status.stat().st_mtime_ns, assessment))
        if states:
            return max(states, key=lambda item: item[0])[1]
    folders = [Path(output)] if output is not None else list((root / 'mine_model/results').glob('joint_pd_scratch_*'))
    choices = []
    for folder in folders:
        status = folder / 'status.json'
        if status.is_file():
            state = json.loads(status.read_text())
            if state.get('last_assessment'):
                choices.append(Path(state['last_assessment']))
        choices.extend(p / 'assessment.json' for p in folder.glob('checkpoint_*')
                       if (p / 'completed.json').is_file() and (p / 'assessment.json').is_file())
    if not choices:
        raise FileNotFoundError('No completed supported-policy evaluation; supply an explicit checkpoint and overrides')
    return max(choices, key=lambda p: p.stat().st_mtime_ns)


def replay_overrides(saved):
    """Freeze controller/observation settings without replacing SDK objects."""
    saved = strip_retired_config_fields(saved)
    result = {key: value for key, value in saved.items()
              if isinstance(value, (bool, int, float, list, tuple))
              or key in ('leg_model', 'leg_control_mode', 'wheel_control_mode', 'feedforward_mode')}
    for group in ('commands', 'rewards', 'obs_scales', 'obs_clip'):
        result.update({f'{group}.{key}': value for key, value in saved.get(group, {}).items()
                       if value is None or isinstance(value, (bool, int, float, str, list, tuple))})
    result['sim.dt'] = saved['sim']['dt']
    result['robot.spawn.usd_path'] = saved['robot']['spawn']['usd_path']
    for key in ('solver_type',):
        if key in saved['sim'].get('physx', {}):
            result[f'sim.physx.{key}'] = saved['sim']['physx'][key]
    for key in ('solver_position_iteration_count', 'solver_velocity_iteration_count'):
        props = saved['robot']['spawn'].get('articulation_props', {})
        if key in props:
            result[f'robot.spawn.articulation_props.{key}'] = props[key]
    for group in ('pos', 'rot', 'joint_pos', 'joint_vel'):
        if group in saved['robot']['init_state']:
            result[f'robot.init_state.{group}'] = saved['robot']['init_state'][group]
    return result


def replay_command(root, output=None, checkpoint=None, overrides=None, extra=()):
    if (checkpoint is None) != (overrides is None):
        raise ValueError('Supply --checkpoint_path and --overrides_json together to preserve control settings')
    if checkpoint is None:
        assessment = latest_completed_assessment(root, output).expanduser().resolve()
        report = json.loads(assessment.read_text())
        checkpoint = Path(report['checkpoint'])
        overrides = assessment.parent.parent / 'overrides.json'
        if not overrides.is_file():
            saved = checkpoint_config(checkpoint)
            overrides = assessment.parent / 'keyboard_replay_overrides.json'
            overrides.write_text(json.dumps(replay_overrides(saved), ensure_ascii=False, indent=2) + '\n')
    checkpoint = Path(checkpoint).expanduser().resolve()
    overrides = Path(overrides).expanduser().resolve()
    for path in (checkpoint, overrides):
        if not path.is_file():
            raise FileNotFoundError(f'Replay file missing: {path}')
    saved = checkpoint_config(checkpoint)
    requested = json.loads(overrides.read_text()).get('leg_control_mode', saved['leg_control_mode'])
    require_leg_control_mode(requested, saved['leg_control_mode'])
    task = ('WheelLeggedJointPD-Flat-v0'
            if saved.get('rewards', {}).get('profile') == DIRECT_JOINT_REWARD_PROFILE
            else 'WheelLeggedVMC-Flat-v0')
    # Do not start a second simulator just to validate paths or select weights.
    return [str(root / 'IsaacLab/_isaac_sim/python.sh'),
            str(root / 'wheel_legged_isaaclab/scripts/rsl_rl/play.py'),
            '--task', task, '--num_envs', '1', '--keyboard',
            '--live_stats', '--real-time', '--checkpoint_path', str(checkpoint),
            '--overrides_json', str(overrides), *extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training_output', type=Path,
                        help='Optional completed evaluation folder; default is the latest joint-PD watcher result')
    parser.add_argument('--checkpoint_path', type=Path)
    parser.add_argument('--overrides_json', type=Path)
    parser.add_argument('--dry_run', action='store_true', help='Print launch command without starting a GUI')
    args, extra = parser.parse_known_args()
    # This convenience entry is exclusively for the own-model interactive GUI.
    # Fixed/headless diagnostics remain available through play_wheel.sh.
    forbidden = {'--task', '--headless', '--video', '--no_keyboard', '--use_pretrained_checkpoint',
                 '--fixed_leg_length', '--fixed_wheel_action', '--zero_actions', '--startup_focus'}
    if any(option.split('=')[0] in forbidden for option in extra):
        parser.error('For other tasks or automated/open-loop diagnostics use play_wheel.sh instead')
    try:
        command = replay_command(ROOT, args.training_output.expanduser().resolve() if args.training_output else None,
                                 args.checkpoint_path, args.overrides_json, extra)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print('[KEYBOARD REPLAY] ' + shlex.join(command), flush=True)
    if not args.dry_run:
        os.chdir(ROOT / 'IsaacLab')
        os.execv(command[0], command)


if __name__ == '__main__':
    main()
