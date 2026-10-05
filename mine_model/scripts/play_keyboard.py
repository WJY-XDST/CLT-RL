"""Launch interactive replay with a matched checkpoint + training overrides.

By default choose the latest completed evaluation, not a half-written training
checkpoint or the old 'best' policy. This script does not start training.
"""
import argparse
import json
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'mine_model/results/auto_training_implicit_20pct_20261004'


def replay_command(root, output, checkpoint=None, overrides=None, extra=()):
    if (checkpoint is None) != (overrides is None):
        raise ValueError('Supply --checkpoint_path and --overrides_json together to preserve control settings')
    if checkpoint is None:
        state = json.loads((output / 'status.json').read_text())
        assessment = Path(state['last_assessment']).expanduser().resolve()
        report = json.loads(assessment.read_text())
        checkpoint = Path(report['checkpoint'])
        overrides = assessment.parent.parent / 'overrides.json'
    checkpoint = Path(checkpoint).expanduser().resolve()
    overrides = Path(overrides).expanduser().resolve()
    for path in (checkpoint, overrides):
        if not path.is_file():
            raise FileNotFoundError(f'Replay file missing: {path}')
    # Do not start a second simulator just to validate paths or select weights.
    return [str(root / 'IsaacLab/_isaac_sim/python.sh'),
            str(root / 'wheel_legged_isaaclab/scripts/rsl_rl/play.py'),
            '--task', 'WheelLeggedVMC-Flat-v0', '--num_envs', '1', '--keyboard',
            '--live_stats', '--real-time', '--checkpoint_path', str(checkpoint),
            '--overrides_json', str(overrides), *extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training_output', type=Path, default=DEFAULT_OUTPUT,
                        help='Automatic training output folder containing status.json')
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
        command = replay_command(ROOT, args.training_output.expanduser().resolve(),
                                 args.checkpoint_path, args.overrides_json, extra)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print('[KEYBOARD REPLAY] ' + shlex.join(command), flush=True)
    if not args.dry_run:
        os.chdir(ROOT / 'IsaacLab')
        os.execv(command[0], command)


if __name__ == '__main__':
    main()
