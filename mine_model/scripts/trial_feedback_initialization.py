"""Fixed-checkpoint feedback initialization trials; never promote automatically."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'wheel_legged_isaaclab'))
from wheel_legged_gym_isaaclab.checkpoint_initialization import scale_actor_feedback_initialization,scale_wheel_difference_initialization
from summarize_mine_trial import assess, plot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-replay', type=Path, required=True)
    p.add_argument('--overrides', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--columns', type=int, nargs='+')
    mode.add_argument('--wheel-differential', action='store_true')
    p.add_argument('--fractions', type=float, nargs='+', default=[.75, .5])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.source_replay / 'manifest.json').read_text())
    source = Path(manifest['checkpoint'])
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != manifest['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed after replay')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab'))
    for k in ('ROS_DISTRO','ROS_VERSION','ROS_PYTHON_VERSION','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH'):
        env.pop(k, None)
    for fraction in args.fractions:
        folder = args.output / ('gain_' + str(fraction).replace('.', 'p'))
        checkpoint = folder / 'checkpoint' / source.name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source.parent / 'params', checkpoint.parent / 'params', dirs_exist_ok=True)
        initialized=(scale_wheel_difference_initialization(saved, fraction) if args.wheel_differential else
                     scale_actor_feedback_initialization(saved, args.columns, fraction))
        torch.save(initialized, checkpoint)
        provenance = {'source_checkpoint': str(source), 'source_sha256': source_sha,
            'source_replay': str(args.source_replay.resolve()), 'actor_input_columns': args.columns,
            'mode': 'wheel_differential_output' if args.wheel_differential else 'actor_input_gain',
            'fraction': fraction, 'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'critic_unchanged': True, 'runtime_observations_unchanged': True,
            'optimizer': 'clear only edited actor columns or wheel output rows; retain all other moments and exploration std',
            'status': 'diagnostic candidate, not accepted or promoted'}
        (folder / 'initialization.json').write_text(json.dumps(provenance, indent=2) + '\n')
        shutil.copy2(args.overrides, folder / 'overrides.json')
        cmd = [str(ROOT / 'run_python.sh'), str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
            '--headless', '--robot', 'mine', '--checkpoint', str(checkpoint.resolve()), '--seconds', '12',
            '--overrides_json', str((folder / 'overrides.json').resolve()),
            '--output', str((folder / 'replay').resolve())]
        print('Evaluating', folder, flush=True)
        with (folder / 'evaluation.log').open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = assess(folder / 'replay')
        plot(folder / 'replay')
        print('Result', folder, report['survivors'], report['passed'], flush=True)


if __name__ == '__main__':
    main()
