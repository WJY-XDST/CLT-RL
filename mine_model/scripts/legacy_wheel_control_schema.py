"""Compare old/new saved configs with the one documented historical default."""
from qualify_source801_drift50_transfer import differences
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit


def differences_with_wheel_default(before, after):
    # Work on copies. A missing historical field meant clamp1, while all
    # unrelated missing/extra keys retain the original strict /keys failure.
    before, after = dict(before), dict(after)
    before['wheel_action_limit'] = checkpoint_wheel_action_limit(before)
    after['wheel_action_limit'] = checkpoint_wheel_action_limit(after)
    return differences(before, after)
