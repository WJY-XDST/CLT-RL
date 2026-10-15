"""Regression coverage for actual evaluation cases versus fixed sampler bounds."""
import copy

import pytest

from diagnose_height_response import height_cases


def manifest():
    return dict(cases=[dict(name='stand',height=.3,speed=0.,yaw=0.),
        dict(name='height_low',height=.22534645531,speed=0.,yaw=0.),
        dict(name='height_high',height=.37534645531,speed=0.,yaw=0.)],
        commands=dict(ranges_height=[.3,.3]))


def test_height_commands_come_from_actual_cases_despite_fixed_evaluator_sampler():
    values=height_cases(manifest())
    assert values['height_low']==(1,.22534645531)
    assert values['stand']==(0,.3)
    assert values['height_high']==(2,.37534645531)


@pytest.mark.parametrize('kind',['missing','duplicate','unordered','nonfinite','moving'])
def test_misidentified_height_cases_fail_before_any_response_claim(kind):
    value=copy.deepcopy(manifest())
    if kind=='missing':value['cases'].pop()
    elif kind=='duplicate':value['cases'].append(copy.deepcopy(value['cases'][2]))
    elif kind=='unordered':value['cases'][2]['height']=.2
    elif kind=='nonfinite':value['cases'][2]['height']=float('nan')
    elif kind=='moving':value['cases'][2]['speed']=.3
    with pytest.raises(ValueError):height_cases(value)
