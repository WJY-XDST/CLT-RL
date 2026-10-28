"""Freeze a direct-joint policy and test one common wheel velocity gain.

The same scalar applies to both wheels and every operating condition. Saved
mass, gravity feedforward, leg PD, rewards and observations are restored. No
weights, optimizer, reference map, case-dependent feedback or gate is changed.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / part) for part in
               ('mine_model/scripts', 'mine_model/experiments', 'wheel_legged_isaaclab')]


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--wheel-damping', type=float, required=True)
    parsed, arguments = own.parse_known_args()
    gain = parsed.wheel_damping
    assert math.isfinite(gain) and .5 <= gain <= 2.
    assert not any(name in arguments for name in ('--overrides_json', '--sample-policy-actions',
                   '--wheel-probe', '--bias-probe-json', '--wheel-reference-limit-probe'))
    checkpoint = Path(arguments[arguments.index('--checkpoint') + 1]).resolve()
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    from joint_pd_continuation import sha
    saved = validate_checkpoint_robot(checkpoint)
    assert saved['leg_control_mode'] == 'implicit_joint_position'
    assert saved['wheel_control_mode'] == 'implicit_velocity'

    def observe(env, output):
        import torch
        raw = env.unwrapped
        cfg = raw.cfg.to_dict()
        for key in ('leg_control_mode', 'wheel_control_mode', 'leg_joint_stiffness',
                    'leg_joint_damping', 'leg_effort_limit', 'wheel_effort_limit',
                    'feedforward_force', 'feedforward_mode', 'feedforward_angle_bias',
                    'feedforward_force_limit', 'leg_joint_gravity_feedforward',
                    'enable_external_forces_every_iteration', 'action_scale_vel', 'wheel_action_limit'):
            assert cfg[key] == saved[key], key
        assert cfg['robot']['spawn']['usd_path'] == saved['robot']['spawn']['usd_path']
        assert cfg['rewards'] == saved['rewards']
        assert cfg['wheel_damping'] == cfg['robot']['actuators']['wheels']['damping'] == gain
        actual = raw._robot.root_physx_view.get_dof_dampings()[:, raw._wheel_joint_ids]
        assert torch.allclose(actual, torch.full_like(actual, gain), atol=1.e-6, rtol=0.)
        masses = raw._robot.root_physx_view.get_masses().sum(-1)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        report = dict(verified=True, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      source_wheel_damping=saved['wheel_damping'], requested_common_wheel_damping=gain,
                      actual_PhysX_wheel_damping_range=[float(actual.min()), float(actual.max())],
                      actual_vehicle_mass_range_kg=[float(masses.min()), float(masses.max())],
                      all_other_checked_control_and_reward_values_retained=True,
                      same_scalar_for_every_command_and_both_wheels=True,
                      weights_updated=False, condition_specific_feedback=False,
                      policy_accepted=False)
        (output / 'native_common_wheel_gain_verified.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    from joint_pd_source2001_quality50_runtime import evaluator_tree, criteria_fix
    tree, path = evaluator_tree()
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'height' for t in node.targets)]
    assert len(indices) == 1
    original = ast.dump(tree)
    additions = ast.parse('cfg.wheel_damping = _common_wheel_gain\n'
                          "cfg.robot.actuators['wheels'].damping = _common_wheel_gain").body
    at = indices[0]
    function.body[at:at] = additions
    changed = copy.deepcopy(tree)
    del function.body[at:at + len(additions)]
    assert ast.dump(tree) == original
    sys.argv = [str(path), *arguments]
    exec(compile(ast.fix_missing_locations(changed), str(path), 'exec'),
         dict(__name__='__main__', __file__=str(path), _common_wheel_gain=gain,
              _observe_source200=observe, _install_quality_criteria=criteria_fix.install))


if __name__ == '__main__':
    main()
