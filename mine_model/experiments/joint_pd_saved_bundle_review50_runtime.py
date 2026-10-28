"""The saved ordinary-PPO bundle, with a fifty-update review boundary only."""
import argparse
from pathlib import Path
import sys

import joint_pd_saved_bundle_runtime as implementation


SCRIPT = Path(__file__).resolve()
old_command = implementation.command
old_profile = implementation.profile_for_count
old_native = implementation.native


def profile_for_count(count):
    result = old_profile(count)
    result.update(review_boundary_runtime_sha256=implementation.sha(SCRIPT),
                  independent_review_interval=50, original_rollout_and_PPO_minibatches_retained=True)
    return result


def command(source, cfg, label, updates, commands, stop_file):
    assert updates in (50, 200)
    argv, _, saved = old_command(source, cfg, label, 200, commands, stop_file)
    target = implementation.saved_updates(source) + updates
    argv[argv.index('--target_iteration') + 1] = str(target)
    return argv, target - 1, saved


def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    assert int(arguments[arguments.index('--target_iteration') + 1]) - implementation.saved_updates(source) == 50
    return old_native(output, arguments, selected_count=8192, updates=50)


implementation.SCRIPT = SCRIPT
implementation.profile_for_count = profile_for_count
implementation.command = command
implementation.native = native
sys.modules['joint_pd_saved_bundle_review50_runtime'] = implementation


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        boundary = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:boundary])
        implementation.configure(args.recipe)
        native(args.observation_output, sys.argv[boundary + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        implementation.configure(args.recipe)
        implementation.replay(remaining)
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
