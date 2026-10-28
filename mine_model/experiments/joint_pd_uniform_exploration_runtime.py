"""Ordinary PPO with a global exploration change and unmodified commands/reward.

The JSON recipe contains provenance and scalar noise floors. No command value,
height, speed, yaw sign, or measured failure selects different training logic.
Historical modules supply the previously qualified plain27 runner/observers;
the retired conditional sampler and reward modules are never imported.
"""
import argparse
import copy
from functools import lru_cache
import json
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
    assert value["schema"] == "plain27_uniform_exploration_v1"
    assert value["command_sampler"] == "original_continuous_with_existing_roles"
    assert value["reward"] == "original_uniform_reward"
    assert value["planned_updates"] == 200 and value["relative_error_target"] == .1
    assert not any(value[key] for key in ("optimizer_reset", "std_reset", "actual_LR_reset", "acceptance_changed"))
    floors = value["exploration"]
    assert 0. < floors["global_minimum_std"] <= min(floors["minimum_leg_std"], floors["minimum_wheel_std"])
    assert all(sha(path) == digest for path, digest in value["source_pins"].items())
    return value


@lru_cache(maxsize=1)
def baseline():
    value = recipe()
    path = Path(value["source_checkpoint"]).resolve()
    report = json.loads(Path(value["source_final_verification"]).read_text())
    assert report["verified"] and report["checkpoint_sha256"] == sha(path) == value["source_checkpoint_sha256"]
    saved, cfg = retained.require_source(path)
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
    return result


def profile_for_count(count):
    value = copy.deepcopy(retained.profile())
    value.update(profile="plain27_original_reward_continuous_commands_uniform_exploration_v1",
        native_runtime_sha256=sha(SCRIPT), uniform_recipe_sha256=sha(RECIPE_PATH),
        continuation_source_sha256=recipe()["source_checkpoint_sha256"],
        native_training_environment_count=count, fixed_standing_environment_count=count // 4,
        command_sampler="original_continuous_with_existing_roles", reward="original_uniform_reward",
        condition_specific_training_logic=False, changed_noise_heads=[2, 5],
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


def private_for_count(count, selected_profile=None):
    callback = selected_profile or (lambda: profile_for_count(count))
    module = retained.private_runtime()
    module.profile, module.require_source = callback, require_source
    original_observe = module.observe_environment

    def observe(env, output):
        result = original_observe(env, output)
        raw = env.unwrapped
        assert raw.cfg.rewards.to_dict() == baseline()[2]["rewards"]
        chain = verify_unmodified_execution(raw)
        assert not any(name.startswith("_historical_yaw_focus") or
                       name.startswith("_historical_yaw_error_focus") for name in vars(raw))
        assert not any(name in sys.modules for name in (
            "historical_yaw_focus_sampler", "historical_yaw_focus_sampler_v2", "historical_yaw_error_focus"))
        write_json(Path(output) / "uniform_recipe_environment.json", dict(
            verified=True, profile=callback(), actual_num_envs=raw.num_envs,
            original_uniform_reward=True, original_command_sampler=True,
            verified_reward_call_chain=chain,
            condition_specific_runtime_modules_loaded=False, conditional_action_feedback=False,
            policy_accepted=False))
        return result

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
        _observe_source200=private_for_count(NUM_ENVS).observe_environment))


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
