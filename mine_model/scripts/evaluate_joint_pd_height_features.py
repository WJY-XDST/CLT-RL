"""Use the original canonical evaluator with its declared feature policy."""
from functools import partial
from pathlib import Path
import runpy
import sys

import torch
import yaml
import rsl_rl.modules

from joint_pd_continuation import ROOT
from wheel_legged_gym_isaaclab.robot_model import MODEL, validate_checkpoint_robot

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
from joint_pd_height_features_candidate import JointPdHeightFeatureActorCritic
from joint_pd_height_feature_runtime import require_paired_feature_model


def main():
    source = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
    saved = torch.load(source, map_location='cpu', weights_only=True)
    cfg = validate_checkpoint_robot(source)
    model = require_paired_feature_model(saved, cfg, MODEL['hip_z'])
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    if agent['policy']['class_name'] != 'JointPdHeightFeatureActorCritic':
        raise ValueError('Checkpoint metadata and paired agent policy class differ')
    rsl_rl.modules.ActorCritic = partial(JointPdHeightFeatureActorCritic, height_feature_model=model)
    runpy.run_path(str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'), run_name='__main__')


if __name__ == '__main__':
    main()
