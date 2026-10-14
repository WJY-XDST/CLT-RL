"""Read back calibrated wheel inertia from native PhysX, without training.

This loading smoke test is diagnostic evidence, not tracking acceptance.
The supplied USD composition report defines the expected tensors in each
wheel's own rigid-body frame; equal wheels may have different local frames.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'wheel_legged_isaaclab'))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--composition-report', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
composition = json.loads(args.composition_report.read_text())
expected = composition['after']
asset = Path(expected['path'])
assert hashlib.sha256(asset.read_bytes()).hexdigest() == expected['sha256']
app = AppLauncher(headless=True, device='cuda:0').app

import gymnasium as gym
import numpy as np
import torch
import wheel_legged_gym_isaaclab.tasks  # task registration after AppLauncher
from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


env = None
try:
    cfg = WheelLeggedVMCFlatEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 83
    cfg.sim.device = 'cuda:0'
    cfg.robot.spawn.usd_path = str(asset)
    assert cfg.leg_control_mode == 'implicit_joint_position'
    assert cfg.feedforward_force == 60.5277
    env = gym.make('WheelLeggedJointPD-Flat-v0', cfg=cfg)
    observations, _ = env.reset()
    robot = env.unwrapped._robot
    masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    inertias = robot.root_physx_view.get_inertias()[0].cpu().numpy()
    centers = robot.root_physx_view.get_coms()[0].cpu().numpy()
    report = dict(asset=str(asset), asset_sha256=expected['sha256'],
                  environments=1, control_steps=10, training_started=False,
                  policy_acceptance_test=False, wheels={}, body_inertia_relative_errors={},
                  total_mass_kg=float(masses.sum()),
                  wheel_masses_kg={}, finite_observations=finite(observations),
                  finite_rewards=True, resets=0,
                  control=dict(leg_control_mode=cfg.leg_control_mode,
                               feedforward_force_n=cfg.feedforward_force,
                               feedforward_mode=cfg.feedforward_mode,
                               feedforward_angle_bias_rad=cfg.feedforward_angle_bias,
                               nominal_vehicle_mass_kg=2 * 6.17,
                               mass_per_leg_kg=6.17))
    for name, mass in expected['body_masses_kg'].items():
        index = robot.body_names.index(name)
        np.testing.assert_allclose(masses[index], mass, rtol=1.e-6)
        properties = expected.get('body_mass_properties', {}).get(name)
        if properties:
            native_tensor = inertias[index].reshape(3, 3).T
            expected_tensor = np.asarray(properties['tensor_link_frame'])
            # Rotating a float32 principal frame introduces cancellation in
            # small off-diagonal entries. Compare the full tensor norm rather
            # than dividing by an arbitrarily small individual component.
            relative_error = float(np.linalg.norm(native_tensor - expected_tensor)
                                   / np.linalg.norm(expected_tensor))
            assert relative_error < 2.e-6, (name, relative_error)
            assert np.linalg.eigvalsh(native_tensor).min() > 0
            report['body_inertia_relative_errors'][name] = relative_error
            np.testing.assert_allclose(centers[index, :3], properties['com_local'], rtol=1.e-6, atol=1.e-9)
    report['body_masses_kg'] = {name: float(masses[i]) for i, name in enumerate(robot.body_names)}
    np.testing.assert_allclose(report['total_mass_kg'], expected['total_mass_kg'], rtol=1.e-6)
    tensors_base = {}
    for name in ('LW_link', 'RW_link'):
        index = robot.body_names.index(name)
        tensor = inertias[index].reshape(3, 3).T  # PhysX column-major, body frame
        reference = expected['wheels'][name]
        np.testing.assert_allclose(tensor, reference['tensor_link_frame'], rtol=2.e-5, atol=1.e-9)
        np.testing.assert_allclose(centers[index, :3], reference['com_local'], rtol=1.e-6, atol=1.e-9)
        eigenvalues = np.linalg.eigvalsh(tensor)
        np.testing.assert_allclose(eigenvalues, sorted(reference['principal_inertia_kg_m2']), rtol=2.e-5)
        rotation = np.asarray(reference['rotation_link_to_base'])
        tensors_base[name] = rotation @ tensor @ rotation.T
        report['wheel_masses_kg'][name] = float(masses[index])
        report['wheels'][name] = dict(mass_kg=float(masses[index]),
                                    tensor_link_frame=tensor.tolist(),
                                    principal_inertia_sorted_kg_m2=eigenvalues.tolist(),
                                    spin_inertia_base_y_kg_m2=float(tensors_base[name][1, 1]),
                                    com_and_principal_axes_xyzw=centers[index].tolist())
    reflection = np.diag([1., -1., 1.])
    np.testing.assert_allclose(tensors_base['LW_link'], reflection @ tensors_base['RW_link'] @ reflection,
                               rtol=2.e-5, atol=1.e-9)
    np.testing.assert_allclose(report['wheels']['LW_link']['principal_inertia_sorted_kg_m2'],
                               report['wheels']['RW_link']['principal_inertia_sorted_kg_m2'], rtol=2.e-5)
    with torch.inference_mode():
        for _ in range(report['control_steps']):
            observations, rewards, terminated, truncated, _ = env.step(
                torch.zeros((1, 6), device=env.unwrapped.device))
            report['finite_observations'] &= finite(observations)
            report['finite_rewards'] &= finite(rewards)
            report['resets'] += int((terminated | truncated).sum())
    assert report['finite_observations'] and report['finite_rewards']
    report.update(native_wheel_tensor_matches_usd=True, same_principal_moments=True,
                  base_frame_mirror_tensor_verified=True, all_body_masses_match_composed_usd=True,
                  all_body_masses_unchanged=expected['body_masses_kg'] == composition['before']['body_masses_kg'],
                  all_body_inertias_match_composed_usd=(set(report['body_inertia_relative_errors'])
                      == set(expected['body_masses_kg'])),
                  qualification='Native loading and finite zero-action steps only; no policy acceptance claim')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
except Exception as error:
    # Kit shutdown can conceal a Python exception behind exit status 0.
    # Persist the exception before app.close(); a missing complete report is
    # always a failed qualification regardless of the process exit code.
    import traceback
    args.report.parent.mkdir(parents=True, exist_ok=True)
    failure = dict(exception_type=type(error).__name__, error=str(error),
                   traceback=traceback.format_exc(), policy_acceptance_test=False,
                   asset=str(asset), body_being_verified=locals().get('name'))
    args.report.with_name('native_physx_failure.json').write_text(
        json.dumps(failure, indent=2) + '\n')
    print(json.dumps(failure, indent=2), flush=True)
    raise
finally:
    if env is not None:
        env.close()
    app.close()
