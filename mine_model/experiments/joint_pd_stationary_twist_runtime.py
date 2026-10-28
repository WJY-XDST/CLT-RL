"""One reward-component experiment; direct PD actions and ordinary PPO stay intact."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import torch

from joint_pd_continuation import ROOT, sha
import joint_pd_analytic_tire_runtime as tires
from summarize_mine_trial import acceptance_criteria

KEY = 'joint_pd_stationary_twist_reward'
GATE = ROOT / 'mine_model/config/tracking_acceptance.json'
GATE_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'
INITIAL_SOURCE_SHA = '1e1a06722998513177edefd11041b37e16d8f000f1e44c489c2e58f30a06f767'
ENVIRONMENT = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
_installed_proof = None


def profile():
    if sha(GATE) != GATE_SHA:
        raise ValueError('Never change the original gate or its normalization constants')
    criteria = acceptance_criteria()
    speed, yaw = criteria['speed_absolute_m_s'], criteria['yaw_absolute_rad_s']
    if speed != .025 or yaw != .025:
        raise ValueError('Use the original measured zero-command absolute tolerances')
    return dict(profile='stationary_twist_huber_v1', existing_component='standing_velocity',
        coefficient=-1., speed_tolerance_m_s=speed, yaw_tolerance_rad_s=yaw,
        error_coordinates=['heading_vx', 'heading_vy', 'body_yaw_rate_minus_command'],
        Huber_transition_normalized_error=1., reduction='sum',
        original_standing_mask_retained=True, original_single_reward_clip_per_s=5.,
        original_acceptance_sha256=GATE_SHA, environment_source_sha256=sha(ENVIRONMENT),
        reward_runtime_source_sha256=sha(Path(__file__)))


def cost(velocity, yaw_rate, yaw_command):
    """Dimensionless stationary twist error; produces no physical reference."""
    if velocity.ndim != 2 or velocity.shape[1] != 2 or yaw_rate.shape != velocity.shape[:1] or yaw_command.shape != yaw_rate.shape:
        raise ValueError('Require heading vx/vy and the matching body yaw signal/command')
    # profile() validates these ORIGINAL tolerances when installing the method;
    # constants here avoid filesystem IO in the native50Hz reward callback.
    error = torch.cat((velocity / .025, ((yaw_rate - yaw_command) / .025)[:, None]), dim=-1)
    absolute = error.abs()
    return torch.where(absolute < 1., .5 * error.square(), absolute - .5).sum(-1)


def replace_assignment(tree):
    """Replace exactly one RHS and prove the rest of the original method equal."""
    original = copy.deepcopy(tree)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == 'r_standing_velocity']
    if len(matches) != 1:
        raise ValueError('Require exactly one original standing-velocity assignment')
    expected = ast.parse('torch.sum(torch.square(heading_velocity), dim=1) * standing', mode='eval').body
    if ast.dump(matches[0].value) != ast.dump(expected):
        raise ValueError('Requalify the original standing mask/term before replacing it')
    matches[0].value = ast.parse('_stationary_twist_cost(heading_velocity, root_ang_vel_b[:, 2], self._commands[:, 1]) * standing', mode='eval').body
    restored = copy.deepcopy(tree)
    target = [n for n in ast.walk(restored) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == 'r_standing_velocity'][0]
    target.value = expected
    if ast.dump(restored) != ast.dump(original):
        raise ValueError('Reward override changed code outside its one declared component')
    ast.fix_missing_locations(tree)
    return tree


def require_source(source, *, initial_transfer=False):
    saved, cfg = tires.require_source(source)
    infos = saved.get('infos')
    declared = infos.get(KEY) if isinstance(infos, dict) else None
    if declared is None:
        if not initial_transfer or sha(source) != INITIAL_SOURCE_SHA or cfg['rewards']['standing_velocity'] != -100.:
            raise ValueError('Only preserved midpoint576 may start this explicit reward transfer')
    elif declared != profile() or cfg['rewards']['standing_velocity'] != -1.:
        raise ValueError('Saved stationary-twist reward profile and paired config differ')
    return saved, cfg


def install_metadata():
    import rsl_rl.runners as runners
    upstream = runners.OnPolicyRunner

    class StationaryTwistRunner(upstream):
        def save(self, path, infos=None):
            result = copy.deepcopy(infos) if isinstance(infos, dict) else {'upstream_infos': infos}
            if result.get(KEY, profile()) != profile():
                raise ValueError('Cannot replace saved stationary reward semantics silently')
            result[KEY] = profile()
            return super().save(path, infos=result)

    runners.OnPolicyRunner = StationaryTwistRunner


def install_reward_method():
    """Call after Kit initialization for both gym and direct-constructor entry points."""
    global _installed_proof
    if _installed_proof is not None:
        if _installed_proof['profile'] != profile():
            raise ValueError('Reward source changed after installation')
        return copy.deepcopy(_installed_proof)
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat import wheel_legged_vmc_flat_env as module
    cls = module.WheelLeggedVMCFlatEnv
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state_all()
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls._get_rewards)))
    original_ast_sha = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    tree = replace_assignment(tree)
    namespace = dict(vars(module), _stationary_twist_cost=cost)
    exec(compile(tree, str(Path(__file__)) + ':private_one_component_reward', 'exec'), namespace)
    cls._get_rewards = namespace['_get_rewards']
    if (not torch.equal(before_cpu, torch.get_rng_state()) or any(
            not torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all()))):
        raise ValueError('Reward installation changed the policy random state')
    _installed_proof = dict(profile=profile(), original_reward_method_AST_sha256=original_ast_sha,
        exactly_one_existing_reward_assignment_changed=True,
        original_method_equal_after_restoring_that_RHS=True,
        install_policy_RNG_unchanged=True, actions_or_physics_or_PPO_changed=False,
        policy_accepted=False)
    return copy.deepcopy(_installed_proof)


def observe_environment(env, output):
    from joint_pd_height_feature_runtime import write
    env = env.unwrapped
    if (env.cfg.leg_control_mode != 'implicit_joint_position'
            or env.cfg.rewards.standing_velocity != -1. or env.cfg.rewards.clip_single_reward != 5.):
        raise ValueError('New reward semantics require explicit standing coefficient-1 and the original clip5')
    proof = dict(**install_reward_method(), native_environments=env.num_envs)
    if output is not None:
        write(Path(output) / 'stationary_twist_runtime.json', proof)
    if getattr(env.cfg, 'log_dir', None):
        write(Path(env.cfg.log_dir) / 'params/stationary_twist_contract.json', proof)


def install_environment_observer(output):
    """Training uses gym.make; the evaluator uses the direct method hook above."""
    import gymnasium as gym
    original_make = gym.make

    def make(*args, **kwargs):
        identifier = args[0] if args else kwargs.get('id')
        if identifier != 'WheelLeggedJointPD-Flat-v0':
            return original_make(*args, **kwargs)
        install_reward_method()
        result = original_make(*args, **kwargs)
        observe_environment(result, output)
        return result

    gym.make = make
