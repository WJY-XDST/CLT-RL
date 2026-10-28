"""Bind frozen replay to the explicit recipe without changing native provenance.

The native training/runtime file and its profile SHA remain immutable. The
parent command builder inherited a parent-recipe path; only that argv value is
replaced here. No dynamics, observations, actions, rewards or scoring change.
There is no new training entry point.
"""
import argparse
from pathlib import Path

import joint_pd_explicit_resume_exploration_runtime as native


SCRIPT = native.SCRIPT  # Native checkpoint identity stays bound to that file.
ADAPTER_SCRIPT = Path(__file__).resolve()
ORIGINAL_REPLAY_COMMAND = native.replay_command
_recipe = None


def __getattr__(name):
    return getattr(native, name)


def configure(path):
    global _recipe
    path = Path(path).resolve()
    if _recipe is not None:
        assert path == _recipe
        return native.recipe()
    result = native.configure(path)
    _recipe = path
    return result


def replay_command(checkpoint, output):
    argv = ORIGINAL_REPLAY_COMMAND(checkpoint, output)
    before = list(argv)
    assert argv[1] == str(SCRIPT)
    at = argv.index('--recipe') + 1
    assert Path(argv[at]).resolve() == Path(native._spec['parent_recipe']).resolve()
    argv[at] = str(_recipe)
    assert [i for i, (left, right) in enumerate(zip(before, argv)) if left != right] == [at]
    return argv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay',))
    parser.add_argument('--recipe', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    configure(args.recipe)
    native.replay(remaining)
