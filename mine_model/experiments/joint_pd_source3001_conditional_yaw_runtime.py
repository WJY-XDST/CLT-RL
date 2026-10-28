"""Independent historical3001 continuation, changing only spin practice.

Reuse the original plain27 implicit-position native runner, environment
observers, PPO, exact state restoration and exploration projection. The
selected checkpoint is the explicitly requested historical source; newer
7401/7601 models remain separate preserved comparators.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys
import types

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import joint_pd_source2801_velocity010_runtime as retained
import joint_pd_source2201_yaw025_runtime as native_template
import historical_yaw_focus_sampler as focus


SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / "mine_model/results"
QUALIFICATION = RESULTS / "joint_pd_source3001_conditional_yaw_restart_20261024"
SOURCE_REPORT = RESULTS / "joint_pd_optimization_20261019_025551_001147_source2801_velocity010_200/final_checkpoint_verification_readonly_20261019.json"
ANCHOR_SHA = "f0fa511ebf2ef31b35f4ad311ee67468acc50e6fc0fd216e337ae90c1f7a8a01"
KEY, base, criteria_fix = retained.KEY, retained.base, retained.criteria_fix
FLAG = "--historical_source3001_conditional_yaw"
NUM_ENVS = 8192
DELTA = dict(rewards={}, commands={"spin_practice_distribution": ["original_continuous", focus.specification()]},
             exploration={}, controller={}, physics={})


def bind(function, **values):
    return types.FunctionType(function.__code__, dict(function.__globals__, **values),
                              function.__name__, function.__defaults__, function.__closure__)


@lru_cache(maxsize=1)
def baseline():
    final = json.loads(SOURCE_REPORT.read_text())
    source = Path(final["checkpoint"])
    saved, cfg = retained.require_source(source)
    assert final["verified"] and final["checkpoint_sha256"] == sha(source) == ANCHOR_SHA
    assert final["actual_ppo_updates"] == 3001 and final["all17_Adam_step"] == 60020
    assert cfg["commands"]["training_motion_scale"] == .25 and cfg["scene"]["num_envs"] == 4096
    assert cfg["rewards"]["tracking_ang_vel"] == 1. and cfg["rewards"]["yaw_rate_error_sq"] == -1.
    assert cfg["wheel_damping"] == .5 and cfg["leg_control_mode"] == "implicit_joint_position"
    assert saved["model_state_dict"]["std"].tolist() == final["saved_std"]
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(value["step"]) == 60020 for value in saved["optimizer_state_dict"]["state"].values())
    return source, saved, cfg


def profile_for_count(count):
    assert count in (4096, 8192)
    value = copy.deepcopy(retained.profile())
    value.update(profile="plain_source3001_original_recipe_conditional_spin_v1",
        native_runtime_sha256=sha(SCRIPT), continuation_source_sha256=ANCHOR_SHA,
        authority="explicit_user_historical_restart_preserve_newer_models",
        native_training_environment_count=count, fixed_standing_environment_count=count // 4,
        historical_spin_practice=focus.specification(), sampler_source_sha256=sha(focus.__file__),
        historical_absent_wheel_action_limit_has_same_explicit_default=1.,
        all_saved_NN_17Adam_six_std_and_actual_LR_preserved=True)
    return value


def profile():
    return profile_for_count(NUM_ENVS)


def expected_configuration(cfg, count):
    value = copy.deepcopy(cfg)
    for field in ("scene", "terrain"):
        value[field]["num_envs"] = count
    # Old3001 predates this serialized field. Both control implementations use
    # 1.0 when it is absent; declare that same value in the current schema.
    assert "wheel_action_limit" not in cfg or cfg["wheel_action_limit"] == 1.
    value.setdefault("wheel_action_limit", 1.)
    return value


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, "Retain the explicitly selected3001 state"
    path = Path(path).resolve()
    anchor, before, cfg = baseline()
    if sha(path) == ANCHOR_SHA:
        assert path == anchor.resolve()
        return before, cfg
    saved = torch.load(path, map_location="cpu", weights_only=True)
    current = validate_checkpoint_robot(path)
    count = current["scene"]["num_envs"]
    expected = expected_configuration(cfg, count)
    expected["log_dir"] = current["log_dir"]
    actual = saved_updates(path)
    assert current == expected and actual >= 3001 and tensor_finite(saved)
    assert saved.get("infos", {}).get(KEY) == profile_for_count(count)
    assert saved["model_state_dict"].keys() == before["model_state_dict"].keys()
    assert all(saved["model_state_dict"][key].shape == value.shape for key, value in before["model_state_dict"].items())
    states = saved["optimizer_state_dict"]["state"]
    assert len(states) == 17 and all(float(value["step"]) == actual * 20 for value in states.values())
    return saved, current


def repair_ready():
    record = json.loads((QUALIFICATION / "restart_decision.json").read_text())
    assert record["explicit_user_historical_restart"] and record["current_models_preserved"]
    assert record["source_checkpoint_sha256"] == ANCHOR_SHA and record["source_actual_updates"] == 3001
    assert record["next_configuration_changes"] == DELTA and record["relative_error_target"] == .1
    assert all(sha(path) == digest for path, digest in record["source_pins"].items())
    return record


def private_for_count(count, selected_profile=None):
    callback = selected_profile or (lambda: profile_for_count(count))
    module = retained.private_runtime()
    module.profile, module.require_source = callback, require_source
    original_observe = module.observe_environment

    def observe(env, output):
        result = original_observe(env, output)
        raw, output = env.unwrapped, Path(output)
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        focus.install(raw)
        assert torch.equal(before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        write_json(output / "historical_conditional_spin_runtime.json", dict(
            profile=callback(), actual_num_envs=raw.num_envs,
            source_checkpoint_sha256=ANCHOR_SHA, sampling=focus.specification(),
            no_install_resample_or_RNG_draw=True, rewards_physics_and_controller_unchanged=True,
            four_other_motion_roles_and_fixed_standing_unchanged=True, policy_accepted=False))
        previous_step = raw._pre_physics_step
        calls = 0
        cells = [[0., yaw, height] for height in (.3, .35, .37534645531)
                 for yaw in (-.075, -.0625, -.05, .05, .0625, .075)]
        targets = raw._commands.new_tensor(cells)

        def pre_step(actions):
            nonlocal calls
            previous_step(actions)
            calls += 1
            if calls % 48:
                return
            requested = raw._commands[:, :3].clone()
            requested[:, 0] = raw._target_lin_vel_x
            counts = ((requested[:, None, :] - targets[None, :, :]).abs().amax(-1) < 1.e-6).sum(0).cpu().tolist()
            row = dict(policy_steps=calls, cells=cells, requested_target_counts=counts,
                       focus_selected_requests=int(raw._historical_yaw_focus_selected),
                       snapshot_not_continuous_time_exposure=True)
            with (output / "conditional_spin_exposure.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        raw._pre_physics_step = pre_step
        return result

    module.observe_environment = observe
    return module


def private_runtime(source=None):
    if source is not None:
        require_source(source)
    return private_for_count(NUM_ENVS)


def install_leg_exploration(output, selected_profile=None):
    # Rebind only the runtime metadata; use the exact historical .035/.03/.02 projection.
    original = retained.previous.install_leg_exploration
    return bind(original, profile=selected_profile or profile)(output)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    saved, current = require_source(source)
    assert cfg == current and commands == cfg["commands"] and updates == 200 and not kwargs
    argv, _, _ = retained.command(source, cfg, label, updates, commands, Path(stop_file))
    argv[1] = str(SCRIPT)
    argv[argv.index("--num_envs") + 1] = str(NUM_ENVS)
    argv[argv.index("--observation-output") + 1] = str(Path(stop_file).parent / "native_learning_signal")
    assert argv.count("agent.save_interval=5") == 1
    argv[argv.index("agent.save_interval=5")] = "agent.save_interval=1"
    assert not any(flag in argv for flag in ("--reset_optimizer", "--reset_exploration_std", "--save_initial_checkpoint"))
    return argv, saved_updates(source) + updates - 1, saved


def native(output, arguments):
    global NUM_ENVS
    NUM_ENVS = int(arguments[arguments.index("--num_envs") + 1])
    assert NUM_ENVS in (8192, 4096)
    repair_ready()
    source = Path(arguments[arguments.index("--checkpoint_path") + 1])
    require_source(source)
    assert int(arguments[arguments.index("--num_envs") + 1]) == NUM_ENVS
    return bind(native_template.native, SCRIPT=SCRIPT, FLAG=FLAG, repair_ready=repair_ready,
        require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)(output, arguments)


def replay_command(source, output):
    require_source(source)
    argv = retained.replay_command(source, output)
    argv[argv.index(str(retained.SCRIPT))] = str(SCRIPT)
    return argv


def replay(arguments):
    assert FLAG not in arguments and "--overrides_json" not in arguments
    source = Path(arguments[arguments.index("--checkpoint") + 1])
    require_source(source)
    tree, path = retained.evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    exec(compile(tree, str(path), "exec"), dict(__name__="__main__", __file__=str(path),
        _install_quality_criteria=criteria_fix.install,
        _observe_source200=private_runtime().observe_environment))


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "native":
        boundary = sys.argv.index("--")
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    elif mode == "replay":
        replay(sys.argv[1:])
    else:
        raise ValueError(mode)
