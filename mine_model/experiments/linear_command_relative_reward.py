"""Explicitly revise only the existing positive forward-speed reward width.

One continuous formula applies to every command, sign, height and task. The
original yaw kernel, all costs, action mapping and PPO remain unchanged.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap

from command_relative_tracking_kernel import command_relative_variance, validate_parameters

SCRIPT = Path(__file__).resolve()
TAG = '_declared_linear_command_relative_kernel'
PROOF = '_declared_linear_command_relative_proof'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(speed_cap, floor_fraction):
    cap, fraction = float(speed_cap), float(floor_fraction)
    validate_parameters(1., cap, fraction)
    return dict(version='positive_linear_only_command_relative_variance_v1',
        formula='sigma_eff = sigma * (speed_command^2 + (speed_cap*f)^2) / (speed_cap^2 + (speed_cap*f)^2)',
        speed_cap_m_s=cap, floor_fraction=fraction,
        same_formula_for_all_environments_heights_signs_and_tasks=True,
        nominal_sigma_retained_at_speed_cap=True,
        zero_command_has_finite_absolute_width=True,
        original_yaw_kernel_and_all_negative_costs_retained=True,
        acceptance_threshold_used_by_reward=False,
        helper_sha256=sha(inspect.getfile(command_relative_variance)),
        builder_sha256=sha(SCRIPT))


def build_reward(original, speed_cap, floor_fraction):
    spec = contract(speed_cap, floor_fraction)
    assert original.__name__ == '_get_rewards' and not original.__closure__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert not tree.body[0].decorator_list
    initial = ast.dump(tree)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == 'r_track_lin']
    assert len(matches) == 1
    node = matches[0]
    expected = ast.parse('torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)', mode='eval').body
    assert ast.dump(node.value) == ast.dump(expected)
    owner = node.value.args[0]
    old = owner.right
    owner.right = ast.parse('_linear_command_relative_variance(self._commands[:, 0], '
        'self.cfg.rewards.tracking_sigma, ' + repr(spec['speed_cap_m_s']) + ', '
        + repr(spec['floor_fraction']) + ')', mode='eval').body
    revised_tree = copy.deepcopy(tree)
    owner.right = old
    assert ast.dump(tree) == initial, 'A change escaped the single linear denominator'
    namespace = dict(original.__globals__, _linear_command_relative_variance=command_relative_variance)
    exec(compile(ast.fix_missing_locations(revised_tree), str(SCRIPT), 'exec'), namespace)
    revised = namespace['_get_rewards']
    revised.__qualname__ = original.__qualname__
    proof = dict(verified=True, contract=spec,
        changed_assignments=['r_track_lin'], all_other_application_reward_AST_identical=True,
        original_AST_sha256=hashlib.sha256(initial.encode()).hexdigest(),
        revised_AST_sha256=hashlib.sha256(ast.dump(revised_tree).encode()).hexdigest(),
        original_application_source=inspect.getfile(original),
        original_application_source_sha256=sha(inspect.getfile(original)),
        new_reward_terms_added=0, observation_callbacks_added=0,
        yaw_kernel_modified=False, negative_costs_modified=False,
        action_or_physics_or_sampler_changed=False, ordinary_PPO_loss_changed=False,
        acceptance_changed=False, policy_accepted=False)
    setattr(revised, TAG, spec)
    setattr(revised, PROOF, proof)
    return revised, proof, revised_tree


def install_class_reward(env_class, speed_cap, floor_fraction):
    spec = contract(speed_cap, floor_fraction)
    current = env_class._get_rewards
    previous = getattr(current, TAG, None)
    if previous is not None:
        assert previous == spec, 'Do not mix reward contracts within one process'
        return getattr(current, PROOF)
    revised, proof, _ = build_reward(current, speed_cap, floor_fraction)
    env_class._get_rewards = revised
    return proof
