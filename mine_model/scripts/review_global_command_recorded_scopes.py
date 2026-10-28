"""Retain fixed source scopes across a declared global motion-envelope step.

Use the compatible command runtime and existing two-field metadata comparator.
Every fixed case, seed, initial observation, physical field, failure and numeric
gate is checked by the retained workflow. This process cannot start PPO.
"""
import argparse
from pathlib import Path

import joint_pd_uniform_command_runtime as runtime
import review_recorded_fixed_scopes_after_uniform_block_v2 as retained

SCRIPT = Path(__file__).resolve()


def install():
    assert retained.runtime is runtime
    retained.SCRIPT = SCRIPT
    retained.unchanged_command_metadata = lambda: runtime.recipe()['command_changes']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-only', action='store_true')
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--pointer', type=Path)
    parser.add_argument('--source', type=Path, nargs='+')
    parser.add_argument('--after-owned-unit')
    args = parser.parse_args()
    install()
    if args.qualify_only:
        assert args.recipe and args.pointer and args.source and args.after_owned_unit
        assert set(runtime.configure(args.recipe)['command_changes']) == {
            'training_motion_scale', 'yaw_min_abs_rate'}
        retained.qualify(args)
    else:
        retained.run(args.output.resolve())
