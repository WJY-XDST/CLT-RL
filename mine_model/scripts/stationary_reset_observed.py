"""Entry points for a reviewed reset-only revision, never auto-selecting a source."""
import argparse
from pathlib import Path
import sys

from auto_train_mine import write_json
from evaluate_source200_stationary_height import evaluator_tree
from joint_pd_review_interval200 import install_observer
import joint_pd_stationary_reset_runtime as runtime


def source_arguments(arguments, checkpoint_option):
    arguments = list(arguments)
    transition = runtime.FLAG in arguments
    if transition:
        arguments.remove(runtime.FLAG)
    source = Path(arguments[arguments.index(checkpoint_option)+1]).resolve()
    runtime.require_source(source, transition=transition)
    return arguments, source, transition


def native(output, arguments):
    arguments,source,transition = source_arguments(arguments,'--checkpoint_path')
    if ([x for x in arguments if x.startswith('env.reset_velocity_final=')] != ['env.reset_velocity_final=0.05']
            or [x for x in arguments if x.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']):
        raise ValueError('Declare only the static reset-final revision and retain plain ActorCritic')
    runtime.private_runtime(source if transition else None).install_runner(output)
    runtime.private_runtime().install_environment_observer(output)
    runtime.install_leg_exploration(output)
    import train_joint_pd_observed as observed
    write_json(Path(output).parent/'review_interval_runtime.json', install_observer(observed))
    observed.run(output,runtime.parent.previous.tires.arguments(arguments))


def replay(arguments):
    arguments,source,_ = source_arguments(arguments,'--checkpoint')
    sys.argv = [__file__,*runtime.parent.previous.tires.arguments(arguments)]
    tree,original = evaluator_tree()
    exec(compile(tree,str(original),'exec'),dict(__name__='__main__',__file__=str(original),
        _observe_source200=runtime.private_runtime(evaluation=True).observe_environment))


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        parser = argparse.ArgumentParser()
        parser.add_argument('--observation-output',type=Path,required=True)
        options = parser.parse_args(sys.argv[1:boundary])
        native(options.observation_output,sys.argv[boundary+1:])
    else:
        raise ValueError('Use native or replay; a controller must select a reviewed checkpoint')
