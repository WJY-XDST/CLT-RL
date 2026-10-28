"""Explicit user-requested leg-only exploration revision of the plain branch.

Reuse the qualified load/save/environment logic in a private module instance.
The live old branch and its pinned files are never modified.
"""
import copy
import importlib.util
from pathlib import Path

import torch

from auto_train_mine import write_json
from joint_pd_continuation import sha
import joint_pd_source200_stationary_runtime as prior

KEY = prior.KEY
LEG_STD_FLOOR = .05
GLOBAL_STD_FLOOR = prior.LEG_STD_FLOOR
WHEEL_STD_FLOOR = prior.WHEEL_STD_FLOOR
HEIGHT_SIGMA = prior.HEIGHT_SIGMA
STANDING_VELOCITY_SCALE = -25.
tires = prior.tires


def profile():
    result = copy.deepcopy(prior.profile())
    result.update(profile='plain_source200_stationary_leg05_drift25_v1',
        minimum_leg_std=LEG_STD_FLOOR, global_minimum_std=GLOBAL_STD_FLOOR,
        minimum_wheel_std=WHEEL_STD_FLOOR, changed_noise_heads=[0, 1, 3, 4],
        native_runtime_sha256=sha(__file__), prior_runtime_sha256=sha(prior.__file__),
        standing_velocity=STANDING_VELOCITY_SCALE,
        authority='explicit_user_joint_height_and_stability_repair_with_increased_leg_exploration')
    return result


def private_runtime(transition_source=None):
    spec = importlib.util.spec_from_file_location('_plain_leg05_runtime_private', prior.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_require = module.require_source
    original_observe = module.observe_environment
    transition_sha = None
    if transition_source is not None:
        # Only the exact selected old-branch descendant can cross this boundary.
        saved, _ = prior.require_source(transition_source)
        if prior.saved_updates(transition_source) <= 201:
            raise ValueError('Do not reload the original model200 for every exploration revision')
        transition_sha = sha(transition_source)

    def require(source, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Original source200 transfer is not an exploration revision')
        if transition_sha is not None and sha(source) == transition_sha:
            saved, cfg = prior.require_source(source)
            if cfg['rewards']['standing_velocity'] != 0.:
                raise ValueError('Require the diagnosed disabled standing-velocity cost at transition')
        else:
            saved, cfg = original_require(source, initial_transfer=False)
            if cfg['rewards']['standing_velocity'] != STANDING_VELOCITY_SCALE:
                raise ValueError('Keep the paired explicit drift-repair reward')
        return saved, cfg

    def observe(env, output):
        if env.unwrapped.cfg.rewards.standing_velocity != STANDING_VELOCITY_SCALE:
            raise ValueError('Actual native environment did not apply the diagnosed standing cost')
        return original_observe(env, output)

    module.profile = profile
    module.require_source = require
    module.observe_environment = observe
    return module


def require_source(source, *, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_runner(output, *, transition_source=None):
    return private_runtime(transition_source).install_runner(output)


def observe_environment(env, output):
    return private_runtime().observe_environment(env, output)


def install_environment_observer(output):
    return private_runtime().install_environment_observer(output)


def install_leg_exploration(output):
    """Add only the authorized leg-floor projection after the exact state load."""
    import wheel_legged_gym_isaaclab.exploration as exploration
    original = exploration.enforce_positive_exploration

    def enforce(policy, optimizer, minimum_std=1.e-4, *, minimum_wheel_std=None):
        if minimum_std != GLOBAL_STD_FLOOR or minimum_wheel_std != WHEEL_STD_FLOOR:
            raise ValueError('Keep the declared unchanged global and wheel exploration guards')
        previous = copy.deepcopy(policy.state_dict())
        previous_optimizer = copy.deepcopy(optimizer.state_dict())
        handle = original(policy, optimizer, minimum_std, minimum_wheel_std=minimum_wheel_std)
        parameter = getattr(policy, 'std', None)
        if not isinstance(parameter, torch.nn.Parameter) or parameter.shape != (6,):
            raise ValueError('Require the original six trainable Gaussian standard deviations')
        indices = torch.tensor([0, 1, 3, 4], device=parameter.device)

        def project(*_args, **_kwargs):
            with torch.no_grad():
                if not torch.isfinite(parameter).all():
                    raise ValueError('Nonfinite exploration cannot be silently repaired')
                parameter[indices] = parameter[indices].clamp_min(LEG_STD_FLOOR)

        project()
        leg_handle = optimizer.register_step_post_hook(project)
        current = policy.state_dict()
        for key, value in previous.items():
            if key != 'std':
                prior.equal_state(value, current[key], key)
        if not torch.equal(previous['std'][[2, 5]], current['std'][[2, 5]]):
            raise ValueError('The leg-only revision changed wheel exploration')
        prior.equal_state(previous_optimizer, optimizer.state_dict(), 'Adam_after_leg_projection')
        Path(output).mkdir(parents=True, exist_ok=True)
        write_json(Path(output) / 'leg_exploration_floor_runtime.json', dict(
            profile=profile(), before_std=previous['std'].cpu().tolist(),
            after_std=current['std'].cpu().tolist(), all17_Adam_bitwise_preserved=True,
            actor_and_critic_bitwise_preserved=True, wheel_std_bitwise_preserved=True,
            projection_after_every_optimizer_step=True, parameter_identity_preserved=True,
            added_loss_or_backward=False, policy_accepted=False))
        print('[INFO]: User-requested leg-only exploration floor .05; wheel floor .03 and all Adam states retained.', flush=True)

        class Handles:
            def remove(self):
                handle.remove()
                leg_handle.remove()

        return Handles()

    exploration.enforce_positive_exploration = enforce
    return original
