"""Full height coverage must not advance the speed/yaw startup curriculum."""
from types import SimpleNamespace

import pytest
import torch

from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M, training_height_bounds
from wheel_legged_gym_isaaclab.yaw_training import sample_grouped_motion


def commands(**changes):
    result = dict(ranges_height=SAFE_BASE_HEIGHT_RANGE_M, full_height_range_from_start=True,
        ranges_lin_vel_x=(.2, .8), ranges_reverse_lin_vel_x=(-.8, -.2),
        ranges_transition_lin_vel_x=(-.2, .2), speed_boundary_fraction=.3,
        grouped_low_speed_fraction=1/3, ranges_ang_vel_yaw=(-.3, .3),
        yaw_min_abs_rate=.15, yaw_boundary_fraction=.2)
    return SimpleNamespace(**dict(result, **changes))


@pytest.mark.parametrize('step', [0, 48, 4800, 9600, 48000])
def test_full_height_does_not_need_a_late_motion_clock(step):
    cfg = commands()
    assert training_height_bounds(cfg, .3, step, 0.) == SAFE_BASE_HEIGHT_RANGE_M
    speed, yaw = sample_grouped_motion(torch.arange(8192), cfg, 0., 0., 0.,
                                      dedicated_standing_count=2048)
    assert not speed.any() and not yaw.any()


def test_default_remains_the_legacy_coupled_height_schedule():
    cfg = commands(full_height_range_from_start=False)
    assert training_height_bounds(cfg, .3, 0, 0.) == (.3, .3)
    assert training_height_bounds(cfg, .3, 0, 1.) == SAFE_BASE_HEIGHT_RANGE_M


def test_full_height_switch_requires_an_actual_boolean():
    with pytest.raises(ValueError, match='boolean'):
        training_height_bounds(commands(full_height_range_from_start=1), .3, 0, 0.)


def test_full_height_switch_cannot_accept_invalid_envelopes():
    with pytest.raises(ValueError, match='contain'):
        training_height_bounds(commands(ranges_height=(.4, .2)), .3, 0, 0.)
