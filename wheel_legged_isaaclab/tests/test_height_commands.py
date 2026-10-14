"""Command reachability, stop clearance and resume curriculum boundaries."""
from types import SimpleNamespace

import pytest
import torch

from wheel_legged_gym_isaaclab.height_commands import (
    MEASURED_LEG_STOP_RANGE_M, LEG_STOP_MARGIN_M, SAFE_LEG_LENGTH_RANGE_M,
    SAFE_BASE_HEIGHT_RANGE_M, UPRIGHT_HEIGHT_OFFSET_M, training_height_bounds,
)
from wheel_legged_gym_isaaclab.robot_model import MAPPED_FIVE_BAR
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state
from wheel_legged_gym_isaaclab.yaw_training import curriculum_replay_cases


def test_safe_range_has_clearance_from_both_physical_stops():
    assert SAFE_LEG_LENGTH_RANGE_M[0] - MEASURED_LEG_STOP_RANGE_M[0] >= LEG_STOP_MARGIN_M
    assert MEASURED_LEG_STOP_RANGE_M[1] - SAFE_LEG_LENGTH_RANGE_M[1] >= LEG_STOP_MARGIN_M
    assert SAFE_BASE_HEIGHT_RANGE_M == tuple(v + UPRIGHT_HEIGHT_OFFSET_M for v in SAFE_LEG_LENGTH_RANGE_M)


def test_extreme_height_targets_fit_the_unchanged_joint_action_contract():
    # Independently solved motor coordinates; the forward calculation is the
    # production FK. Nothing supplies IK during policy action execution.
    q = torch.tensor([[.1887639535941677, -.2932028789746996, -.1887639535941677, .2932028789746996],
                      [-.3362512618952766, .23181233129105105, .3362512618952766, -.23181233129105105]], dtype=torch.float64)
    offset = q.new_tensor([-.0707452938, -.0336936327, .0707452938, .0336936327])
    assert ((q-offset)/.3).abs().max() < .9
    length, angle, _, valid = five_bar_state(q[:, ::2], q[:, 1::2], **MAPPED_FIVE_BAR)
    assert valid.all()
    torch.testing.assert_close(length[:, 0], q.new_tensor(SAFE_LEG_LENGTH_RANGE_M), atol=1e-8, rtol=0)
    assert angle.abs().max() < 1e-8


def test_height_expansion_resumes_at_old_range_and_preserves_motion():
    c = SimpleNamespace(ranges_height=SAFE_BASE_HEIGHT_RANGE_M, height_expansion_from=(.28, .32),
                        height_expansion_start_steps=117024, height_expansion_ramp_steps=48000)
    assert training_height_bounds(c, .3, 117024, 1) == (.28, .32)
    assert training_height_bounds(c, .3, 165024, 1) == SAFE_BASE_HEIGHT_RANGE_M
    midpoint = training_height_bounds(c, .3, 141024, 1)
    assert midpoint == pytest.approx(tuple((a+b)/2 for a,b in zip((.28,.32), SAFE_BASE_HEIGHT_RANGE_M)))
    assert training_height_bounds(c, .3, 117024, 0) == (.3, .3)


def test_canonical_replay_covers_new_extrema():
    c = SimpleNamespace(ranges_height=SAFE_BASE_HEIGHT_RANGE_M,
                        ranges_lin_vel_x=(.2,.8), ranges_reverse_lin_vel_x=(-.8,-.2))
    cases = curriculum_replay_cases(.3, c, 1)
    assert (cases[-2][3], cases[-1][3]) == SAFE_BASE_HEIGHT_RANGE_M
    assert [c[1] for c in cases[1:5]] == [.3,.6,-.3,-.6]


def test_invalid_expansion_cannot_exceed_the_final_range():
    c = SimpleNamespace(ranges_height=SAFE_BASE_HEIGHT_RANGE_M, height_expansion_from=(.1,.4),
                        height_expansion_start_steps=0, height_expansion_ramp_steps=48000)
    with pytest.raises(ValueError, match='inside'):
        training_height_bounds(c,.3,0,1)
