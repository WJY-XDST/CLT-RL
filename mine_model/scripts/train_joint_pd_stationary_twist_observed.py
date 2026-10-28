"""Ordinary retained-state PPO with one diagnosed stationary reward change."""
import argparse
from pathlib import Path
import sys

from joint_pd_continuation import ROOT
from wheel_legged_gym_isaaclab.robot_model import MODEL

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_analytic_tire_runtime as tires
import joint_pd_stationary_twist_runtime as twist
from joint_pd_height_features_candidate import feature_model_from_config
from joint_pd_height_feature_runtime import install_training


def run(output, values):
    values = tires.arguments(values)
    flag = '--allow_stationary_reward_transfer'
    transfer = flag in values
    if transfer:
        values.remove(flag)
    source = Path(values[values.index('--checkpoint_path') + 1]).resolve()
    saved, cfg = twist.require_source(source, initial_transfer=transfer)
    if [v for v in values if v.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=JointPdHeightFeatureActorCritic']:
        raise ValueError('Retain the exact feature policy and original six actions')
    install_training(feature_model_from_config(cfg, MODEL['hip_z']), output)
    tires.install_runner_metadata()
    twist.install_metadata()
    tires.install_environment_observer(output)
    twist.install_environment_observer(output)
    from train_joint_pd_observed import run as original
    original(output, values)


if __name__ == '__main__':
    boundary = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-output', type=Path, required=True)
    args = parser.parse_args(sys.argv[1:boundary])
    run(args.observation_output, sys.argv[boundary + 1:])
