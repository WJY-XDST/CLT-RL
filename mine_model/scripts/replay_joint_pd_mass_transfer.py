"""Frozen direct-joint policy on one explicitly declared mass calibration.

Reuse the original evaluator and its telemetry. Only robot USD and the nominal
half-vehicle gravity force change, identically for every command/environment.
No weights, feedback rule, command sampler or acceptance threshold is changed.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / part) for part in
               ('mine_model/scripts', 'mine_model/experiments', 'wheel_legged_isaaclab')]
from joint_pd_continuation import sha


def main():
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument('--mass-profile', type=Path, required=True)
    own, remaining = cli.parse_known_args()
    assert '--overrides_json' not in remaining and '--sample-policy-actions' not in remaining
    spec = json.loads(own.mass_profile.read_text())
    asset = Path(spec['asset']).resolve()
    checkpoint = Path(remaining[remaining.index('--checkpoint') + 1]).resolve()
    assert sha(asset) == spec['asset_sha256']
    assert sha(checkpoint) == spec['checkpoint_sha256']
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    saved = validate_checkpoint_robot(checkpoint)
    assert saved['leg_control_mode'] == 'implicit_joint_position'
    assert spec['old_vehicle_mass_kg'] == 12.34 and spec['new_vehicle_mass_kg'] == 13.64
    assert abs(spec['new_feedforward_force_n'] - 13.64 * 9.81 / 2.) < 1.e-12
    assert saved['feedforward_force'] == spec['old_feedforward_force_n']
    expected = json.loads(Path(spec['composition_report']).read_text())
    assert expected['verified'] and expected['all_nonbase_mass_COM_inertia_unchanged']

    def observe(env, output):
        import numpy as np
        import torch
        from carb.settings import get_settings
        raw = env.unwrapped
        actual = raw.cfg.to_dict()
        controls = ('leg_control_mode', 'wheel_control_mode', 'wheel_damping',
                    'leg_joint_stiffness', 'leg_joint_damping', 'leg_joint_position_offset',
                    'leg_joint_position_scale', 'leg_joint_target_limit_margin',
                    'leg_joint_gravity_feedforward', 'leg_effort_limit', 'wheel_effort_limit',
                    'feedforward_mode', 'feedforward_angle_bias', 'feedforward_force_limit',
                    'enable_external_forces_every_iteration', 'action_scale_vel', 'wheel_action_limit')
        canonical = lambda value: json.loads(json.dumps(value))
        for name in controls:
            assert canonical(actual[name]) == canonical(saved[name]), name
        assert Path(actual['robot']['spawn']['usd_path']).resolve() == asset
        assert actual['feedforward_force'] == spec['new_feedforward_force_n']
        assert actual['rewards'] == saved['rewards']
        assert get_settings().get('/physics/collisionApproximateCylinders') is False
        masses = raw._robot.root_physx_view.get_masses().detach().cpu().numpy()
        inertias = raw._robot.root_physx_view.get_inertias().detach().cpu().numpy()
        centres = raw._robot.root_physx_view.get_coms().detach().cpu().numpy()
        names = raw._robot.body_names
        assert len(names) == 15 and len(masses) == raw.num_envs
        errors = dict(mass_kg=0., inertia_kg_m2=0., COM_m=0.)
        for index, name in enumerate(names):
            reference = expected['body_mass_properties'][name]
            tensor = np.asarray(reference['tensor_link_frame'])
            errors['mass_kg'] = max(errors['mass_kg'], float(abs(masses[:, index] - reference['mass_kg']).max()))
            errors['inertia_kg_m2'] = max(errors['inertia_kg_m2'], float(abs(inertias[:, index].reshape(-1, 3, 3) - tensor).max()))
            errors['COM_m'] = max(errors['COM_m'], float(abs(centres[:, index, :3] - reference['com_local']).max()))
        assert errors['mass_kg'] < 1.e-5 and errors['inertia_kg_m2'] < 1.e-6 and errors['COM_m'] < 1.e-6, errors
        np.testing.assert_allclose(masses.sum(-1), spec['new_vehicle_mass_kg'], atol=2.e-5, rtol=0.)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        report = dict(verified=True, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      asset=str(asset), asset_sha256=sha(asset), actual_num_envs=raw.num_envs,
                      actual_mass_range_kg=[float(masses.sum(-1).min()), float(masses.sum(-1).max())],
                      body_masses_kg=dict(zip(names, masses[0].astype(float).tolist())),
                      body_COM_local_m={name: centres[0, i, :3].astype(float).tolist() for i, name in enumerate(names)},
                      body_inertias_link_frame={name: inertias[0, i].reshape(3, 3).astype(float).tolist() for i, name in enumerate(names)},
                      composed_USD_vs_native_max_errors=errors,
                      feedforward_force_n=actual['feedforward_force'],
                      old_feedforward_force_n=saved['feedforward_force'],
                      all_other_checked_controller_values_equal=True, original_rewards_equal=True,
                      actor_and_optimizer_modified=False, observer_changes_control=False,
                      policy_accepted=False)
        (output / 'native_mass_transfer_verified.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    from joint_pd_source2001_quality50_runtime import evaluator_tree, criteria_fix
    tree, original = evaluator_tree()
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    points = [i for i, node in enumerate(main_node.body) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'height' for target in node.targets)]
    assert len(points) == 1
    before = ast.dump(tree)
    at = points[0]
    changes = ast.parse('cfg.robot.spawn.usd_path = _mass_asset\ncfg.feedforward_force = _mass_force').body
    main_node.body[at:at] = changes
    modified = copy.deepcopy(tree)
    del main_node.body[at:at + len(changes)]
    assert ast.dump(tree) == before
    sys.argv = [str(original), *remaining]
    exec(compile(ast.fix_missing_locations(modified), str(original), 'exec'),
         dict(__name__='__main__', __file__=str(original), _observe_source200=observe,
              _install_quality_criteria=criteria_fix.install,
              _mass_asset=str(asset), _mass_force=spec['new_feedforward_force_n']))


if __name__ == '__main__':
    main()
