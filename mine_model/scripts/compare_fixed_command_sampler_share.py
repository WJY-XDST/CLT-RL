"""Admit a declared standing-share change only in fixed-command replay.

The evaluation sampler must be disabled in both recordings. Cases, seeds,
initial27 observations, physical configuration and every numeric gate retain
the original comparison checks. No recorded config or metric is rewritten.
"""
from pathlib import Path
from types import FunctionType

import compare_fixed_command_global_motion_replays as retained
from joint_pd_continuation import sha

SCRIPT = Path(__file__).resolve()
ORIGINAL_COMPARE_PAIR = retained.compare_pair


def metadata_changes(before, after, declared):
    assert set(declared) == {'standing_env_fraction'}
    assert before['grouped_training'] is False and after['grouped_training'] is False
    assert before['resampling_time'] == after['resampling_time'] == 1000.
    assert before['heading_command'] is False and after['heading_command'] is False
    pair = declared['standing_env_fraction']
    assert len(pair) == 2 and all(type(v) in (float, int) and 0. < v < 1. for v in pair)
    changed = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys()
               if before.get(key) != after.get(key)}
    assert not changed or changed == declared, 'Reject undeclared command differences'
    return changed


def compare_pair(source, candidate, declared, scope):
    original = ORIGINAL_COMPARE_PAIR
    compare = FunctionType(original.__code__,
        dict(original.__globals__, metadata_changes=metadata_changes),
        original.__name__, original.__defaults__, original.__closure__)
    result, pins = compare(source, candidate, declared, scope)
    result.update(only_disabled_sampler_share_metadata_admitted=True,
                  original_numeric_comparison_and_initial27_pairing_unchanged=True)
    pins[str(SCRIPT)] = sha(SCRIPT)
    return result, pins
