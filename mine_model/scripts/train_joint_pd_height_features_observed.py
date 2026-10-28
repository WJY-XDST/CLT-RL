"""Feature candidate entry; explicit migration, unchanged native PPO observer."""
import argparse
from pathlib import Path
import sys

from joint_pd_continuation import ROOT
from wheel_legged_gym_isaaclab.robot_model import MODEL, validate_checkpoint_robot

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
from joint_pd_height_features_candidate import feature_model_from_config
from joint_pd_height_feature_runtime import install_training


def run(output, arguments):
    declared = [value for value in arguments if value.startswith('agent.policy.class_name=')]
    wanted = 'agent.policy.class_name=JointPdHeightFeatureActorCritic'
    if declared and declared != [wanted]:
        raise ValueError('The feature entry requires its explicitly declared policy architecture')
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    cfg = validate_checkpoint_robot(source)
    model = feature_model_from_config(cfg, MODEL['hip_z'])
    install_training(model, output)
    # Import only after process-local runner registration, so the observer
    # attaches to the same upstream runner subclass used by the native trainer.
    from train_joint_pd_observed import run as original_observed_run
    original_observed_run(output, [*arguments, *([] if declared else [wanted])])


if __name__ == '__main__':
    if '--' not in sys.argv:
        raise ValueError('Separate observer options from native training arguments with --')
    boundary = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-output', type=Path, required=True)
    args = parser.parse_args(sys.argv[1:boundary])
    run(args.observation_output, sys.argv[boundary + 1:])
