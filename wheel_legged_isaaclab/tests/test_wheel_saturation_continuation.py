import copy
import json

import pytest
import torch

import continue_joint_pd_wheel_saturation as controller
from test_joint_pd_balance_rewards import reward_fixture
from test_priority_continuation import config


def source_config():
    saved = config()
    saved['leg_control_mode'] = 'implicit_joint_position'
    saved['rewards']['wheel_action_saturation'] = 0.
    saved['rewards']['joint_target_leg_length_error_sq'] = -100.
    return saved


def test_single_raw_wheel_cost_preserves_all_height_pose_commands_and_physics():
    source = source_config(); before = copy.deepcopy(source)
    expected = controller.revised_config(source)
    assert source == before
    expected['rewards']['wheel_action_saturation'] = 0.
    assert expected == source


def test_identical_clipped_references_have_distinguishable_rewards_for_raw_overshoot():
    env, _, calculate = reward_fixture(2)
    env._raw_actions[:, (2,5)] = torch.tensor([[-2.,-2.],[-1.9,-1.9]])
    env._actions[:, (2,5)] = -1.
    before = calculate().clone()
    assert before[0] == before[1]
    env.cfg.rewards.wheel_action_saturation = controller.COEFFICIENT
    after = calculate()
    assert after[1] > after[0]
    torch.testing.assert_close(env._last_reward_terms['wheel_action_saturation'], torch.tensor([-.04,-.0324]))


def test_valid_balancing_wheel_references_are_not_penalized():
    env, _, calculate = reward_fixture(3)
    env._raw_actions[:, (2,5)] = torch.tensor([[-1.,1.],[.8,-.7],[.2,.3]])
    env._actions.copy_(env._raw_actions)
    before = calculate().clone()
    env.cfg.rewards.wheel_action_saturation = controller.COEFFICIENT
    after = calculate()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert env._last_reward_terms['wheel_action_saturation'].count_nonzero() == 0


def test_prepare_passes_only_one_real_reward_override_to_the_existing_native_block(tmp_path, monkeypatch):
    previous = tmp_path/'source'; previous.mkdir(); output = tmp_path/'output'; output.mkdir()
    saved = source_config()
    def prepare(*unused):
        return dict(source_files={}, source_checkpoint='latest.pt'), copy.deepcopy(saved)
    monkeypatch.setattr(controller.priority.core, 'prepare', prepare)
    monkeypatch.setattr(controller.priority.core, 'sha', lambda path:'digest')
    plan, expected = controller.prepare(previous, output, -2., 25)
    assert plan['reward_key'] == 'wheel_action_saturation'
    assert plan['only_reward_change'] == {'wheel_action_saturation':[0.,-2.]}
    assert expected['rewards']['joint_target_leg_length_error_sq'] == -100.
    assert expected['commands'] == saved['commands']
    assert not plan['physical_reference_limit_changed']


def test_owner_pause_cannot_be_turned_into_a_wheel_reward_handoff(tmp_path):
    previous = tmp_path/'source'; previous.mkdir(); coordinator = tmp_path/'old'; coordinator.mkdir()
    (coordinator/'STOP').write_text(json.dumps(dict(reason='owner_pause')))
    (coordinator/'coordinator.json').write_text(json.dumps(dict(state='explicit_stop_respected')))
    (previous/'status.json').write_text(json.dumps(dict(state='agent_review_required')))
    with pytest.raises(ValueError,match='owner pause'):
        controller.reviewed_handoff(previous,coordinator)


def test_wrapper_reuses_continuous_priority_reviews_and_preserves_global_state(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'reviewed_handoff',lambda *unused:'latest.pt')
    monkeypatch.setattr(controller,'diagnose_source',lambda *unused:dict(requires_raw_cost=True))
    monkeypatch.setattr(controller.priority.core,'validate_checkpoint_robot',lambda *unused:source_config())
    original = controller.priority.prepare, controller.priority.REWARD_KEY, controller.priority.next_shaping_cost
    calls=[]
    def run(previous,base,unit):
        calls.append((previous,base,unit))
        assert controller.priority.prepare is controller.prepare
        assert controller.priority.REWARD_KEY == 'wheel_action_saturation'
        assert controller.priority.next_shaping_cost(-250.,[]) == -2.
    monkeypatch.setattr(controller.priority,'run',run)
    controller.run(tmp_path/'source',tmp_path/'new','new.service',tmp_path/'old')
    assert len(calls) == 1
    assert (controller.priority.prepare,controller.priority.REWARD_KEY,controller.priority.next_shaping_cost) == original


@pytest.mark.parametrize('deep', [False, True])
def test_raw_cost_requires_evidence_from_the_latest_frozen_checkpoint(tmp_path,monkeypatch,deep):
    import numpy as np
    previous = tmp_path/'source'; (previous/'replay').mkdir(parents=True)
    checkpoint = str(previous/'model.pt')
    (previous/'status.json').write_text(json.dumps(dict(checkpoint=checkpoint,checkpoint_sha256='exact')))
    cases=[dict(name=name) for name in ('stand','forward_0p3','forward_0p6','reverse_0p3','reverse_0p6',
                                      'spin_left','spin_right','height_low','height_high')]
    (previous/'replay/manifest.json').write_text(json.dumps(dict(checkpoint_sha256='exact',cases=cases,seeds=[83,84,85])))
    means=np.zeros((20,27,6),dtype=np.float32)
    means[:,9:15,2] = -2. if deep else -.9
    np.savez(previous/'replay/policy_inputs.npz',means=means)
    np.savetxt(previous/'replay/trace.csv',np.zeros((540,2)),delimiter=',',header='episode,dummy',comments='')
    monkeypatch.setattr(torch,'load',lambda *a,**kw:dict(model_state_dict=dict(std=torch.full((6,),.03))))
    result=controller.diagnose_source(previous)
    assert result['requires_raw_cost'] is deep
    assert result['checkpoint_sha256']=='exact'


def test_absent_deep_clipping_keeps_all_rewards_unchanged(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'reviewed_handoff',lambda *unused:'latest.pt')
    monkeypatch.setattr(controller,'diagnose_source',lambda *unused:dict(requires_raw_cost=False))
    monkeypatch.setattr(controller.priority.core,'validate_checkpoint_robot',lambda *unused:source_config())
    def run(*unused):
        coefficient=controller.priority.next_shaping_cost(-250.,[])
        assert coefficient == 0.
        assert controller.revised_config(source_config(),coefficient) == source_config()
    monkeypatch.setattr(controller.priority,'run',run)
    controller.run(tmp_path/'source',tmp_path/'new','new.service',tmp_path/'old')
