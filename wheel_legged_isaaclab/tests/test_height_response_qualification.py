"""Diagnostic batch-size declarations cannot hide changed physics or commands."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path('/home/aaa/studyRL/src/CLT-RL')
sys.path.insert(0,str(ROOT/'mine_model/scripts'))
from qualify_height_response_span import validate_diagnostic_config


def configs():
    replay=ROOT/'mine_model/results/joint_pd_optimization_20261015_131443_666255_mass_recovery_cumulative/block_0002/replay'
    # Replay fixes resampling commands; use the paired training configuration.
    manifest=json.loads((replay/'manifest.json').read_text())
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    source=validate_checkpoint_robot(manifest['checkpoint'])
    candidate=copy.deepcopy(source)
    candidate['scene']['num_envs']=512
    candidate['terrain']['num_envs']=512
    return source,candidate


def test_only_declared_scene_and_terrain_batch_counts_are_allowed():
    source,candidate=configs()
    validate_diagnostic_config(source,candidate,source['commands'])


@pytest.mark.parametrize('change',['friction','terrain','physics_dt','height','undeclared_count'])
def test_unrelated_differences_still_fail(change):
    source,candidate=configs()
    if change=='friction':candidate['terrain']['physics_material']['static_friction']+=.1
    if change=='terrain':candidate['terrain']['terrain_type']='usd'
    if change=='physics_dt':candidate['sim']['dt']*=2
    if change=='height':candidate['commands']['ranges_height']=[.28,.32]
    if change=='undeclared_count':candidate['terrain']['num_envs']=256
    with pytest.raises(ValueError):validate_diagnostic_config(source,candidate,source['commands'])
