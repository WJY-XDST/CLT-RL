import ast
from pathlib import Path

import pytest
import torch

from test_joint_pd_balance_rewards import reward_fixture


@pytest.mark.parametrize('coefficient', [0., -50.])
def test_actual_reward_telemetry_schema_handles_optional_fk_cost(coefficient):
    env, _, calculate = reward_fixture(3)
    env.cfg.rewards.joint_target_leg_length_error_sq = coefficient
    env._l0_ref_applied = torch.full((3,2),.27)
    total = calculate()
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_mine_policy.py'
    tree = ast.parse(path.read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='diagnostic_reward_names' for t in node.targets))
    scope = dict(env=env,cfg=env.cfg,record_rewards=True)
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(path),'exec'),scope)
    names = scope['diagnostic_reward_names']
    assert ('joint_target_leg_length_error_sq' in names) is (coefficient != 0.)
    assert set(names) == set(env._last_reward_terms)
    recorded = torch.stack([env._last_reward_terms[name] for name in names],dim=-1)
    torch.testing.assert_close(recorded.sum(-1), total)
    # Enabling a term must never turn missing active telemetry into zero.
    if coefficient:
        del env._last_reward_terms['joint_target_leg_length_error_sq']
        with pytest.raises(KeyError):
            torch.stack([env._last_reward_terms[name] for name in names],dim=-1)
