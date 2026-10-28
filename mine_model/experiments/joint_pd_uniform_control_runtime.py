"""Versioned ordinary PPO with global reward scalars and one coupled wheel-drive scalar.

Based on the immutable v1 runtime; PPO, exploration, loading and numeric
readers retain their original behavior. Both damping representations are
configured together and checked against the actual PhysX/USD values.
No height, speed, yaw sign or condition selects a different gain.

The JSON recipe contains provenance, reward scalars and noise floors. No command value,
height, speed, yaw sign, or measured failure selects different training logic.
Historical modules supply the previously qualified plain27 runner/observers;
the retired conditional sampler and reward modules are never imported.
"""
import argparse
import copy
from functools import lru_cache
import json
import importlib.util
import math
from pathlib import Path
import sys
import types

import torch

import joint_pd_source2801_velocity010_runtime as retained
import joint_pd_source2201_yaw025_runtime as training_template
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


SCRIPT = Path(__file__).resolve()
KEY, base, criteria_fix = retained.KEY, retained.base, retained.criteria_fix
FLAG = "--uniform_recipe_transition"
RECIPE_PATH = None
NUM_ENVS = 8192


def configure(path):
    global RECIPE_PATH
    selected = Path(path).resolve()
    if RECIPE_PATH is not None and RECIPE_PATH != selected:
        raise ValueError("One immutable recipe per process")
    RECIPE_PATH = selected
    return recipe()


def recipe():
    value = json.loads(RECIPE_PATH.read_text())
    assert value["schema"] == "plain27_uniform_control_v2"
    assert value["command_sampler"] == "original_continuous_with_existing_roles"
    assert value["reward"] == "uniform_existing_reward_coefficients"
    assert value["planned_updates"] == 200 and value["relative_error_target"] == .1
    assert not any(value[key] for key in ("optimizer_reset", "std_reset", "actual_LR_reset", "acceptance_changed"))
    floors = value["exploration"]
    assert 0. < floors["global_minimum_std"] <= min(floors["minimum_leg_std"], floors["minimum_wheel_std"])
    assert set(value["reward_overrides"]) == set(value["reward_changes"])
    assert all(type(v) in (int, float) and math.isfinite(v) for v in value["reward_overrides"].values())
    assert all(sha(path) == digest for path, digest in value["source_pins"].items())
    controls = value["control_overrides"]
    assert set(controls) == {"wheel_damping", "robot.actuators.wheels.damping"}
    assert set(value["control_changes"]) <= set(controls)
    damping = controls["wheel_damping"]
    assert type(damping) in (int, float) and math.isfinite(damping) and damping > 0.
    assert controls["robot.actuators.wheels.damping"] == damping
    return value


@lru_cache(maxsize=1)
def baseline():
    value = recipe()
    path = Path(value["source_checkpoint"]).resolve()
    report = json.loads(Path(value["source_final_verification"]).read_text())
    assert report["verified"] and report["checkpoint_sha256"] == sha(path) == value["source_checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = validate_checkpoint_robot(path)
    assert tensor_finite(saved) and cfg["leg_control_mode"] == "implicit_joint_position"
    controls = value["control_overrides"]
    before_damping = cfg["wheel_damping"]
    assert before_damping == cfg["robot"]["actuators"]["wheels"]["damping"]
    actual_changes = {key: [before_damping, target] for key, target in controls.items() if before_damping != target}
    assert value["control_changes"] == actual_changes
    assert saved["model_state_dict"]["actor.0.weight"].shape[1] == 27
    assert saved["model_state_dict"]["critic.0.weight"].shape[1] == 27
    assert saved["model_state_dict"]["std"].shape == (6,)
    assert not saved.get("infos", {}).get(KEY, {}).get("capacity_only_probe", False)
    actual = saved_updates(path)
    assert actual == report["actual_ppo_updates"] == value["source_actual_updates"]
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(state["step"]) == actual * 20 for state in saved["optimizer_state_dict"]["state"].values())
    return path, saved, cfg


def expected_configuration(cfg, count):
    assert count in (4096, 8192)
    result = copy.deepcopy(cfg)
    for key in ("scene", "terrain"):
        result[key]["num_envs"] = count
    # This is the already effective default of the saved old action schema.
    assert cfg.get("wheel_action_limit", 1.) == 1.
    result.setdefault("wheel_action_limit", 1.)
    for key, value in recipe()["reward_overrides"].items():
        assert key in result["rewards"]
        result["rewards"][key] = value
    controls = recipe()["control_overrides"]
    assert result["wheel_damping"] == result["robot"]["actuators"]["wheels"]["damping"]
    result["wheel_damping"] = controls["wheel_damping"]
    result["robot"]["actuators"]["wheels"]["damping"] = controls["robot.actuators.wheels.damping"]
    return result


def profile_for_count(count):
    value = copy.deepcopy(retained.profile())
    value.update(profile="plain27_uniform_scalars_global_control_continuous_commands_v2",
        native_runtime_sha256=sha(SCRIPT), uniform_recipe_sha256=sha(RECIPE_PATH),
        continuation_source_sha256=recipe()["source_checkpoint_sha256"],
        native_training_environment_count=count, fixed_standing_environment_count=count // 4,
        command_sampler="original_continuous_with_existing_roles", reward="uniform_existing_reward_coefficients",
        uniform_reward_overrides=recipe()["reward_overrides"],
        global_control_overrides=recipe()["control_overrides"],
        global_control_changes=recipe()["control_changes"],
        condition_specific_training_logic=False, changed_noise_heads=[],
        **recipe()["exploration"])
    return value


def profile():
    return profile_for_count(NUM_ENVS)


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, "No original model200 reload"
    path = Path(path).resolve()
    anchor, before, cfg = baseline()
    if path == anchor:
        assert sha(path) == recipe()["source_checkpoint_sha256"]
        return before, cfg
    saved = torch.load(path, map_location="cpu", weights_only=True)
    current = validate_checkpoint_robot(path)
    count = current["scene"]["num_envs"]
    expected = expected_configuration(cfg, count)
    expected["log_dir"] = current["log_dir"]
    assert current == expected and tensor_finite(saved)
    assert saved.get("infos", {}).get(KEY) == profile_for_count(count), "Reject capacity-only or foreign recipes"
    assert saved_updates(path) >= saved_updates(anchor)
    assert saved["model_state_dict"].keys() == before["model_state_dict"].keys()
    assert all(saved["model_state_dict"][key].shape == value.shape for key, value in before["model_state_dict"].items())
    states = saved["optimizer_state_dict"]["state"]
    assert len(states) == 17 and all(float(value["step"]) == saved_updates(path) * 20 for value in states.values())
    return saved, current


def verify_unmodified_execution(raw):
    """Allow only the old read-only terminal logger around the class reward."""
    import source1001_reference_smooth10 as capture
    codes = [value for value in capture.install_failure_capture.__code__.co_consts
             if isinstance(value, types.CodeType) and value.co_name == "reward"]
    assert len(codes) == 1
    reward, wrappers = raw._get_rewards, 0
    while getattr(reward, "__code__", None) is codes[0]:
        closure = dict(zip(reward.__code__.co_freevars, (cell.cell_contents for cell in reward.__closure__)))
        reward = closure["original"]
        wrappers += 1
    assert getattr(reward, "__self__", None) is raw
    assert reward.__func__ is type(raw)._get_rewards
    assert "_resample_commands_for" not in vars(raw)
    return dict(original_class_reward_reached=True, readonly_terminal_capture_wrappers=wrappers,
                command_sampler_instance_replacements=0)


def private_for_count(count, selected_profile=None, *, is_replay=False):
    callback = selected_profile or (lambda: profile_for_count(count))
    spec = importlib.util.spec_from_file_location("_uniform_scalar_plain_runner", base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.profile, module.require_source = callback, require_source

    def observe(env, output):
        from carb.settings import get_settings
        from pxr import UsdGeom, UsdPhysics
        import source1001_reference_smooth10 as capture
        raw = env.unwrapped
        Path(output).mkdir(parents=True, exist_ok=True)
        expected = expected_configuration(baseline()[2], count)
        assert raw.cfg.rewards.to_dict() == expected["rewards"]
        damping = expected["wheel_damping"]
        assert raw.cfg.wheel_damping == raw.cfg.robot.actuators["wheels"].damping == damping
        actual = raw._robot.root_physx_view.get_dof_dampings()[:, raw._wheel_joint_ids]
        assert torch.allclose(actual, torch.full_like(actual, damping), rtol=0., atol=1.e-6)
        usd_damping = []
        for joint in raw.cfg.wheel_joint_names:
            prim = raw.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/" + joint)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            assert drive.GetTypeAttr().Get() == "force"
            usd_damping.append(drive.GetDampingAttr().Get())
            assert abs(usd_damping[-1] - damping) < 1.e-6
            assert abs(drive.GetMaxForceAttr().Get() - expected["wheel_effort_limit"]) < 1.e-5
        write_json(Path(output) / "uniform_global_control_native_contract.json", dict(
            verified=True, declared_global_control_overrides=recipe()["control_overrides"],
            actual_PhysX_wheel_damping_min=float(actual.min()), actual_PhysX_wheel_damping_max=float(actual.max()),
            actual_USD_damping=usd_damping, actual_num_envs=raw.num_envs,
            gain_identical_for_every_command_and_wheel=True, wheel_effort_limit_unchanged=True,
            observer_changes_actions_or_rewards=False, policy_accepted=False))
        assert raw.cfg.leg_control_mode == "implicit_joint_position"
        assert raw.cfg.observation_space == 27 and raw.cfg.action_space == 6
        assert Path(raw.cfg.robot.spawn.usd_path).resolve() == base.tires.ASSET
        assert get_settings().get("/physics/collisionApproximateCylinders") is False
        if not is_replay:
            # YAML/JSON represent tuple ranges as lists. Compare all values
            # after that serialization-only normalization, without changing cfg.
            observed_commands = json.loads(json.dumps(raw.cfg.commands.to_dict()))
            expected_commands = json.loads(json.dumps(expected["commands"]))
            write_json(Path(output) / "uniform_native_command_contract.json", dict(
                actual=observed_commands, expected=expected_commands,
                actual_num_envs=raw.num_envs, requested_num_envs=count,
                values_identical=observed_commands == expected_commands,
                serialization_normalization_only=True, native_configuration_modified=False))
            assert raw.num_envs == count and observed_commands == expected_commands
        for side in ("LW_link", "RW_link"):
            root = "/World/envs/env_0/Robot/" + side
            prim = raw.sim.stage.GetPrimAtPath(root + "/analytic_tire_collision")
            old = raw.sim.stage.GetPrimAtPath(root + "/smooth_tire_collision")
            assert prim.IsA(UsdGeom.Cylinder)
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
            assert UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Get() is False
        masses = raw._robot.root_physx_view.get_masses().sum(-1)
        assert torch.allclose(masses, torch.full_like(masses, 12.34), atol=1.e-4, rtol=0.)
        assert raw._tracking_goal_criteria["relative_error"] == .1
        capture.install_failure_capture(raw, output)
        chain = verify_unmodified_execution(raw)
        assert not any(name.startswith("_historical_yaw_focus") or
                       name.startswith("_historical_yaw_error_focus") for name in vars(raw))
        assert not any(name in sys.modules for name in (
            "historical_yaw_focus_sampler", "historical_yaw_focus_sampler_v2", "historical_yaw_error_focus"))
        write_json(Path(output) / "source200_environment.json", dict(profile=callback(),
            native_environments=raw.num_envs, actual_mass_range_kg=[float(masses.min()), float(masses.max())],
            native_environment_verified=True, observer_changes_actions_or_rewards=False))
        write_json(Path(output) / "uniform_recipe_environment.json", dict(
            verified=True, profile=callback(), actual_num_envs=raw.num_envs,
            original_uniform_reward_structure=True, uniform_reward_overrides=recipe()["reward_overrides"],
            original_command_sampler=True,
            verified_reward_call_chain=chain,
            condition_specific_runtime_modules_loaded=False, conditional_action_feedback=False,
            policy_accepted=False))
        return None

    module.observe_environment = observe
    return module


def install_leg_exploration(output, selected_profile=None):
    """Use the original positivity guard plus a constant floor on leg heads.

    Neither projection uses observations/commands. Both run after ordinary
    optimizer steps and retain parameter identity and all Adam moments.
    """
    import wheel_legged_gym_isaaclab.exploration as exploration
    original = exploration.enforce_positive_exploration
    floors = recipe()["exploration"]
    callback = selected_profile or profile

    def enforce(policy, optimizer, minimum_std=1.e-4, *, minimum_wheel_std=None):
        assert minimum_std == floors["global_minimum_std"]
        assert minimum_wheel_std == floors["minimum_wheel_std"]
        previous, previous_optimizer = copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())
        handle = original(policy, optimizer, minimum_std, minimum_wheel_std=minimum_wheel_std)
        parameter = policy.std
        assert isinstance(parameter, torch.nn.Parameter) and parameter.shape == (6,)
        indices = torch.tensor([0, 1, 3, 4], device=parameter.device)

        def project(*_args, **_kwargs):
            with torch.no_grad():
                assert torch.isfinite(parameter).all()
                parameter[indices] = parameter[indices].clamp_min(floors["minimum_leg_std"])

        project()
        leg_handle = optimizer.register_step_post_hook(project)
        base.equal_state(previous, policy.state_dict(), "initial_saved_NN_and_std_not_reset")
        base.equal_state(previous_optimizer, optimizer.state_dict(), "all17_initial_Adam_not_reset")
        write_json(Path(output) / "uniform_exploration_floor_runtime.json", dict(
            verified=True, profile=callback(), initial_std=parameter.detach().cpu().tolist(),
            all_saved_model_and_optimizer_tensors_bitwise_preserved=True,
            same_floors_for_all_commands_and_observations=True,
            all_six_std_entries_trainable=True, additional_loss_or_backward=False,
            only_floor_of_future_updates_changes=True, policy_accepted=False))
        # Original11 readers use this existing receipt schema. These fields
        # describe the exact initial state, not the future learned std.
        write_json(Path(output) / "leg_exploration_floor_runtime.json", dict(
            profile=callback(), before_std=previous["std"].cpu().tolist(),
            after_std=parameter.detach().cpu().tolist(), all17_Adam_bitwise_preserved=True,
            actor_and_critic_bitwise_preserved=True, wheel_std_bitwise_preserved=True,
            projection_after_every_optimizer_step=True, parameter_identity_preserved=True,
            added_loss_or_backward=False, receipt_schema_compatibility_only=True, policy_accepted=False))

        class Handles:
            def remove(self):
                handle.remove()
                leg_handle.remove()

        return Handles()

    exploration.enforce_positive_exploration = enforce
    return original


def command(source, cfg, label, updates, commands, stop_file):
    saved, current = require_source(source)
    assert cfg == current and commands == current["commands"] and updates == recipe()["planned_updates"]
    argv, final, _ = retained.command(source, cfg, label, updates, commands, Path(stop_file))
    argv[1] = str(SCRIPT)
    argv[3:3] = ["--recipe", str(RECIPE_PATH)]
    for flag, value in {"--num_envs": NUM_ENVS,
                        "--min_exploration_std": recipe()["exploration"]["global_minimum_std"],
                        "--min_wheel_exploration_std": recipe()["exploration"]["minimum_wheel_std"]}.items():
        argv[argv.index(flag) + 1] = str(value)
    assert argv.count("agent.save_interval=5") == 1
    argv[argv.index("agent.save_interval=5")] = "agent.save_interval=1"
    for key, value in recipe()["reward_overrides"].items():
        indices = [i for i, argument in enumerate(argv) if argument.startswith("env.rewards." + key + "=")]
        assert len(indices) == 1
        argv[indices[0]] = "env.rewards." + key + "=" + str(value)
    for key, value in recipe()["control_overrides"].items():
        flag = "env." + key + "="
        assert not any(argument.startswith(flag) for argument in argv)
        argv.append(flag + str(value))
    return argv, final, saved


def native(output, arguments, *, selected_count=None, selected_profile=None, updates=200):
    global NUM_ENVS
    NUM_ENVS = int(arguments[arguments.index("--num_envs") + 1])
    assert NUM_ENVS in (4096, 8192) and (selected_count is None or selected_count == NUM_ENVS)
    assert not any(flag in arguments for flag in ("--reset_optimizer", "--reset_exploration_std", "--save_initial_checkpoint"))
    source = Path(arguments[arguments.index("--checkpoint_path") + 1])
    require_source(source)
    assert int(arguments[arguments.index("--target_iteration") + 1]) - saved_updates(source) == updates
    recipe()
    private_for_count(NUM_ENVS, selected_profile).install_runner(output)
    private_for_count(NUM_ENVS, selected_profile).install_environment_observer(output)
    training_template.previous.install_native_criteria_fix()
    install_leg_exploration(output, selected_profile)
    import train_joint_pd_observed as observed
    import joint_pd_review_interval200 as interval
    interval.UPDATES = updates
    write_json(Path(output).parent / "review_interval_runtime.json", interval.install_observer(observed))
    observed.run(output, base.tires.arguments(arguments))


def replay_command(source, output):
    require_source(source)
    argv = retained.replay_command(baseline()[0], output)
    argv[1] = str(SCRIPT)
    argv[3:3] = ["--recipe", str(RECIPE_PATH)]
    argv[argv.index("--checkpoint") + 1] = str(Path(source).resolve())
    return argv


def replay(arguments):
    assert "--overrides_json" not in arguments
    require_source(Path(arguments[arguments.index("--checkpoint") + 1]))
    tree, path = retained.evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    exec(compile(tree, str(path), "exec"), dict(__name__="__main__", __file__=str(path),
        _install_quality_criteria=criteria_fix.install,
        _observe_source200=private_for_count(NUM_ENVS, is_replay=True).observe_environment))


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    if mode == "native":
        parser.add_argument("--observation-output", type=Path, required=True)
        boundary = sys.argv.index("--")
        args = parser.parse_args(sys.argv[1:boundary])
        configure(args.recipe)
        native(args.observation_output, sys.argv[boundary + 1:])
    elif mode == "replay":
        args, remaining = parser.parse_known_args(sys.argv[1:])
        configure(args.recipe)
        replay(remaining)
    else:
        raise ValueError(mode)
