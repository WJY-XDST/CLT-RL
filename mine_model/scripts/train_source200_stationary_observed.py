"""Original ordinary PPO observer, with200-update reviews and retained model200."""
import argparse
from pathlib import Path
import sys

from auto_train_mine import write_json
from joint_pd_continuation import ROOT
from joint_pd_review_interval200 import install_observer

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_source200_stationary_runtime as runtime


def run(output, arguments):
    arguments = list(arguments)
    flag = '--initial_source200_transfer'
    initial = flag in arguments
    if initial:
        arguments.remove(flag)
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    runtime.require_source(source, initial_transfer=initial)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Keep the original27-input actor/critic')
    runtime.install_runner(output)
    runtime.install_environment_observer(output)
    import train_joint_pd_observed as observed
    proof = install_observer(observed)
    write_json(Path(output).parent / 'review_interval_runtime.json', proof)
    observed.run(output, runtime.tires.arguments(arguments))


if __name__ == '__main__':
    boundary = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-output', type=Path, required=True)
    args = parser.parse_args(sys.argv[1:boundary])
    run(args.observation_output, sys.argv[boundary + 1:])
