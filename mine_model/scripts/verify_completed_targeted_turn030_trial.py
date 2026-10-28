"""Original full200 numeric verifier with exact added-field diff metadata."""
import copy

from review_plain27_single_reward_block200 import rebound
from qualify_source801_drift50_transfer import differences as original_differences
import verify_completed_plain27_joint_pd_trial as original
import joint_pd_source6601_targeted_turn030_runtime as runtime

from pathlib import Path

SCRIPT = Path(__file__).resolve()


def differences(before, after, prefix=''):
    key = runtime.sampler.CONFIG_KEY
    if (prefix == '' and isinstance(before, dict) and isinstance(after, dict)
            and key not in before.get('commands', {}) and key in after.get('commands', {})):
        if after['commands'][key] != runtime.FRACTION:
            raise ValueError('Require the declared added command field value')
        normalized = copy.deepcopy(after)
        normalized['commands'].pop(key)
        return original_differences(before, normalized) + ['/commands/' + key]
    return original_differences(before, after, prefix)


verify = rebound(original.verify, SCRIPT=SCRIPT, differences=differences)
