"""Reuse the completed-block retention workflow for one global standing share.

All recorded source scopes are qualified before waiting for the owned block.
The compatible runtime and disabled-sampler metadata comparator are selected;
PPO, fixed replay, numerical gates and failure retention are unchanged.
"""
import argparse
from pathlib import Path

import joint_pd_uniform_command_runtime as runtime
import review_recorded_fixed_scopes_after_uniform_block_v2 as retained
import compare_fixed_command_sampler_share as paired

SCRIPT = Path(__file__).resolve()


def install():
    assert retained.runtime is runtime
    retained.SCRIPT = SCRIPT
    retained.paired.compare_pair = paired.compare_pair
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
        assert set(runtime.configure(args.recipe)['command_changes']) == {'standing_env_fraction'}
        retained.qualify(args)
    else:
        retained.run(args.output.resolve())
