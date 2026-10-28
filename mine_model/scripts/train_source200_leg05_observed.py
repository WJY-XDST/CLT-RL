"""Ordinary200-update PPO with the explicit leg-only exploration increase."""
import argparse
from pathlib import Path
import sys

from auto_train_mine import write_json
from joint_pd_continuation import ROOT
from joint_pd_review_interval200 import install_observer

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_source200_leg05_runtime as runtime


def run(output, arguments):
    arguments = list(arguments)
    flag = '--increase_leg_exploration'
    transition = flag in arguments
    if transition:
        arguments.remove(flag)
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    runtime.require_source(source, transition=transition)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Retain the plain27 actor/critic')
    runtime.install_runner(output, transition_source=source if transition else None)
    runtime.install_environment_observer(output)
    runtime.install_leg_exploration(output)
    import train_joint_pd_observed as observed
    write_json(Path(output).parent / 'review_interval_runtime.json', install_observer(observed))
    observed.run(output, runtime.tires.arguments(arguments))


if __name__ == '__main__':
    boundary = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-output', type=Path, required=True)
    args = parser.parse_args(sys.argv[1:boundary])
    run(args.observation_output, sys.argv[boundary + 1:])
