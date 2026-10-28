"""Frozen-policy wheel damping diagnosis; no training entry point.

Reuse the qualified evaluator and criteria installation. A read-only native
observer verifies the explicitly declared damping, the remaining controller,
mass, tire geometry and both actual wheel drive properties. Probe outputs
cannot select a policy, replace canonical acceptance, or extend PPO.
"""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

from joint_pd_continuation import ROOT, sha
import joint_pd_AF7001_wheelD075_wheel005_runtime as source_runtime
from types import SimpleNamespace

# Keep the saved-D restoration of the current qualified evaluator.
family = SimpleNamespace(SCRIPT=source_runtime.SCRIPT,
    require_source=source_runtime.require_source,
    evaluator_tree=source_runtime.evaluator_tree,
    replay_command=source_runtime.replay_command,
    _sampler_api=source_runtime.family._sampler_api,
    base=source_runtime.base, criteria_fix=source_runtime.family.criteria_fix)


SCRIPT = Path(__file__).resolve()
SOURCE_SHA = '3964b3a21906dad8160f08d8ca4e1d2310a6bcc2d179277729a63e8a92756785'
RESULTS = ROOT / 'mine_model/results'


def read(path):
    return json.loads(Path(path).read_text())


def normalized(value):
    if isinstance(value, (tuple, list)):
        return [normalized(x) for x in value]
    if isinstance(value, dict):
        return {key: normalized(x) for key, x in value.items()}
    return value


def evaluator_tree():
    tree, original = family.evaluator_tree()
    before = ast.dump(tree)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.unparse(node.test) == 'args.command_probe_json']
    assert len(matches) == 1
    guards = [node for node in matches[0].body if isinstance(node, ast.If)]
    assert len(guards) == 1
    guard = guards[0]
    original_test = copy.deepcopy(guard.test)

    class DeclaredDampingGuard(ast.NodeTransformer):
        count = 0

        def visit_Attribute(self, node):
            if (isinstance(node.value, ast.Name) and node.value.id == 'args'
                    and node.attr == 'overrides_json'):
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(
                    id='_reject_other_control_overrides', ctx=ast.Load()),
                    args=[ast.Name(id='args', ctx=ast.Load())], keywords=[]), node)
            return node

    change = DeclaredDampingGuard()
    guard.test = change.visit(guard.test)
    assert change.count == 1
    modified_test = guard.test
    guard.test = original_test
    assert ast.dump(tree) == before
    guard.test = modified_test
    proof = dict(original_evaluator_AST_restored_identical=True,
        changed_guard_nodes=1,
        permission_extension_only='command diagnosis may declare exactly wheel_damping=1.0',
        original_metric_threshold_window_failure_and_execution_AST_unchanged=True,
        original_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        declared_probe_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest())
    return ast.fix_missing_locations(tree), original, proof


def run(request_path, arguments):
    request_path = Path(request_path).resolve()
    request = read(request_path)
    source, output = Path(request['checkpoint']), Path(request['output'])
    damping = request['wheel_damping']
    assert damping in (.75, 1.) and request['frozen_diagnostic_only']
    assert request['runtime_sha256'] == sha(SCRIPT)
    assert source.resolve() == Path(arguments[arguments.index('--checkpoint') + 1]).resolve()
    assert output.resolve() == Path(arguments[arguments.index('--output') + 1]).resolve()
    assert output.resolve().is_relative_to(RESULTS) and not output.exists()
    assert sha(source) == SOURCE_SHA and '--headless' in arguments
    assert all(sha(path) == digest for path, digest in request['source_pins'].items())
    forbidden = ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint',
        '--target_iteration', '--sample-policy-actions', '--bias-probe-json', '--wheel-probe',
        '--wheel-reference-limit-probe', '--export_dir', '--heading-hold')
    assert not any(flag in arguments for flag in forbidden)
    assert arguments[arguments.index('--command-probe-json') + 1] == request['command_file']
    if damping == .75:
        assert '--overrides_json' not in arguments
    else:
        assert read(arguments[arguments.index('--overrides_json') + 1]) == {'wheel_damping': 1.}
    saved, cfg = family.require_source(source)
    assert cfg['wheel_damping'] == cfg['robot']['actuators']['wheels']['damping'] == .75
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert all(float(value['step']) == 144020 for value in saved['optimizer_state_dict']['state'].values())
    assert saved['model_state_dict']['actor.0.weight'].shape[1] == 27
    del saved

    def observe(env, directory):
        from carb.settings import get_settings
        from pxr import UsdGeom, UsdPhysics
        raw = env.unwrapped
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all()
        for key in ('leg_control_mode', 'leg_joint_stiffness', 'leg_joint_damping', 'leg_effort_limit',
            'wheel_control_mode', 'wheel_effort_limit', 'action_scale_vel', 'wheel_action_limit',
            'leg_joint_position_offset', 'leg_joint_position_scale', 'leg_joint_target_limit_margin',
            'leg_joint_gravity_feedforward', 'feedforward_force', 'feedforward_mode',
            'feedforward_angle_bias', 'feedforward_force_limit', 'five_bar_geometry',
            'observation_space', 'action_space', 'joint_reset_profile',
            'enable_external_forces_every_iteration'):
            assert normalized(getattr(raw.cfg, key)) == normalized(cfg[key]), key
        assert raw.cfg.wheel_damping == raw.cfg.robot.actuators['wheels'].damping == damping
        assert raw.cfg.sim.dt == cfg['sim']['dt'] == .005 and raw.cfg.decimation == cfg['decimation'] == 4
        assert raw.cfg.robot.spawn.articulation_props.solver_position_iteration_count == 128
        assert raw.cfg.robot.spawn.articulation_props.solver_velocity_iteration_count == 8
        assert raw.cfg.rewards.to_dict() == cfg['rewards']
        assert Path(raw.cfg.robot.spawn.usd_path).resolve() == Path(cfg['robot']['spawn']['usd_path']).resolve()
        assert get_settings().get('/physics/collisionApproximateCylinders') is False
        masses = raw._robot.root_physx_view.get_masses().sum(-1)
        assert torch.allclose(masses, torch.full_like(masses, 12.34), rtol=0., atol=1.e-4)
        actual = raw._robot.root_physx_view.get_dof_dampings()[:, raw._wheel_joint_ids]
        assert torch.allclose(actual, torch.full_like(actual, damping), rtol=0., atol=1.e-6)
        usd_damping = []
        for joint, link in zip(raw.cfg.wheel_joint_names, ('LW_link', 'RW_link')):
            prim = raw.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/joints/' + joint)
            drive = UsdPhysics.DriveAPI(prim, 'angular')
            assert drive.GetTypeAttr().Get() == 'force'
            usd_damping.append(drive.GetDampingAttr().Get())
            assert abs(usd_damping[-1] - damping) < 1.e-6
            assert abs(drive.GetMaxForceAttr().Get() - cfg['wheel_effort_limit']) < 1.e-5
            tire = raw.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/' + link + '/analytic_tire_collision')
            assert tire.IsA(UsdGeom.Cylinder) and UsdPhysics.CollisionAPI(tire).GetCollisionEnabledAttr().Get()
            old_tire = raw.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/' + link + '/smooth_tire_collision')
            assert UsdPhysics.CollisionAPI(old_tire).GetCollisionEnabledAttr().Get() is False
        assert torch.equal(before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        path = Path(directory) / 'frozen_wheel_damping_native_observation.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),
            frozen_diagnostic_only=True, checkpoint_sha256=SOURCE_SHA,
            requested_wheel_damping=damping, actual_wheel_damping_min=float(actual.min()),
            actual_wheel_damping_max=float(actual.max()), actual_USD_drive_damping=usd_damping,
            native_environments=raw.num_envs, all_other_declared_control_fields_retained=True,
            mass_min_kg=float(masses.min()), mass_max_kg=float(masses.max()),
            RNG_states_unchanged=True, observer_changes_actions_or_rewards=False,
            optimizer_loaded_into_runner=False, PPO_updates_performed=0,
            can_select_source=False, policy_accepted=False), indent=2) + '\n')

    family._sampler_api.install_sampling_field_restore()
    tree, original, proof = evaluator_tree()
    assert proof == request['evaluator_AST_qualification']
    def reject_other_control_overrides(args):
        return bool(args.overrides_json) and not (
            damping == 1. and read(args.overrides_json) == {'wheel_damping': 1.})
    sys.argv = [str(SCRIPT), *family.base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(original), 'exec'), dict(__name__='__main__', __file__=str(original),
            _install_quality_criteria=family.criteria_fix.install,
            _configure_sampling=family._sampler_api._configure,
            _reject_other_control_overrides=reject_other_control_overrides,
            _observe_source200=observe))
    finally:
        assert sha(source) == SOURCE_SHA
        assert all(sha(path) == digest for path, digest in request['source_pins'].items())


if __name__ == '__main__':
    boundary = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(sys.argv[1:boundary])
    run(args.request, sys.argv[boundary + 1:])
