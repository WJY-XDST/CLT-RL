"""Change only the targeted command-grid probabilities from reviewed7401.

Retain the network,17Adam,sixstd,LR,exploration floors and physical control.
The targeted fraction remains30%, standing IDs remain25%, and continuous
sampling and every grid bin remain available. Distribution metadata is saved
in the checkpoint runtime profile, alongside its exact sampler source hash.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

import joint_pd_wheel005_source7201_D100_yaw400_runtime as retained
import joint_pd_AF7001_wheelD075_runtime as template
import joint_pd_source200_leg05_runtime as projection
import plain27_sampler_metadata_adapter as adapter
import weighted_joint_turn_sampler as focus
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / "mine_model/results"
QUALIFICATION = RESULTS / "joint_pd_source7401_D100_C400_focusgrid_qualification_20261024"
DECISION, ADMISSION = QUALIFICATION / "proposal.json", QUALIFICATION / "native_admission.json"
SOURCE_REPORT = RESULTS / "joint_pd_optimization_20261024_065638_819339_c26a87bd_wheel005_source7201_D100_yaw400_200/final_checkpoint_verification_readonly_20261019.json"
ANCHOR_SHA = "7d82cee7bc43b9b13ea22016d844b72f63e3df4ccdfe021e526907011746b688"
FLAG = "--repair_source7401_D100_C400_focusgrid"
LEG_STD_FLOOR, WHEEL_STD_FLOOR, GLOBAL_STD_FLOOR = .01, .005, .005
DAMPING = 1.
NUM_ENVS = json.loads(ADMISSION.read_text())["selected_num_envs"] if ADMISSION.exists() else 8192
assert NUM_ENVS in (8192, 4096)
DELTA = dict(rewards={}, commands={"targeted_turn_joint_grid_distribution":
    ["uniform_v1", {key: list(value) for key, value in focus.WEIGHTS.items()}]},
    exploration={}, physics={}, controller={})
KEY, family, base, bind, criteria_fix, recipe = (
    retained.KEY, retained.family, retained.base, retained.bind, retained.criteria_fix, retained.recipe)
api = adapter.private_module(template.SCRIPT, "_source7401_C400_leg0075_original_plant_runtime")
POPULATION_METADATA = adapter.retarget(api, ("private_for_count",), integers={"*": {7001: 7401}})
_private_count, _native, _observe_drive = api.private_for_count, api.native, api.observe_drive


@lru_cache(maxsize=1)
def baseline():
    final = json.loads(SOURCE_REPORT.read_text())
    source = Path(final["checkpoint"])
    saved, cfg = retained.require_source(source)
    assert final["verified"] and final["checkpoint_sha256"] == sha(source) == ANCHOR_SHA
    assert final["actual_ppo_updates"] == 7401 and final["all17_Adam_step"] == 148020
    assert cfg["wheel_damping"] == cfg["robot"]["actuators"]["wheels"]["damping"] == 1.
    assert cfg["rewards"]["yaw_rate_error_sq"] == -400. and cfg["rewards"]["tracking_sigma_ang"] == .01
    assert cfg["leg_control_mode"] == "implicit_joint_position" and cfg["observation_space"] == 27
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(state["step"]) == 148020 for state in saved["optimizer_state_dict"]["state"].values())
    assert saved["model_state_dict"]["std"].tolist() == final["saved_std"]
    assert saved["model_state_dict"]["actor.0.weight"].shape[1] == 27 and tensor_finite(saved)
    return source, saved, cfg


def expected_configuration(cfg, count):
    assert count in (8192, 4096)
    value = copy.deepcopy(cfg)
    for key in ("scene", "terrain"):
        value[key]["num_envs"] = count
    return value


def profile_for_count(count):
    value = copy.deepcopy(retained.profile_for_count(count))
    value.pop("only_new_factor_vs_completed_same_source_D100", None)
    value.update(profile="plain_source7401_D100_C400_weighted_joint_grid_v1",
                 native_runtime_sha256=sha(SCRIPT), parent_runtime_sha256=sha(retained.SCRIPT),
                 continuation_source_sha256=ANCHOR_SHA, minimum_leg_std=LEG_STD_FLOOR,
                 minimum_wheel_std=WHEEL_STD_FLOOR, global_minimum_std=GLOBAL_STD_FLOOR,
                 changed_noise_heads=[],
                 targeted_turn_joint_grid_weights={key: list(value) for key, value in focus.WEIGHTS.items()},
                 actual_weighted_sampler_source=str(focus.SCRIPT), actual_weighted_sampler_sha256=sha(focus.SCRIPT),
                 command_distribution_metadata_location=f"checkpoint.infos[{KEY!r}]",
                 declared_saved_source_transfer="only_targeted_joint_grid_probabilities",
                 authority="complete735_and_MuJoCo135_undertracking_then_fresh_exact_capacity")
    return value


def profile():
    return profile_for_count(NUM_ENVS)


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, "Never reload originalmodel200 or reset the saved state"
    path = Path(path).resolve()
    source, before, cfg = baseline()
    if sha(path) == ANCHOR_SHA:
        assert path == source.resolve()
        return before, cfg
    assert not transition, "Only the complete reviewed7401 source starts this sampling trial"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = expected_configuration(cfg, NUM_ENVS)
    expected["log_dir"] = current["log_dir"]
    actual, states = saved_updates(path), saved["optimizer_state_dict"]["state"]
    assert tensor_finite(saved) and current == expected and actual >= 7401
    assert saved.get("infos", {}).get(KEY) == profile()
    assert len(states) == 17 and all(float(state["step"]) == actual * 20 for state in states.values())
    assert saved["model_state_dict"].keys() == before["model_state_dict"].keys()
    assert all(saved["model_state_dict"][key].shape == value.shape for key, value in before["model_state_dict"].items())
    return saved, current


def repair_ready():
    value = json.loads(DECISION.read_text())
    assert value["selected_weighted_targeted_grid_only_trial"] and value["all_required_review_scopes_complete"]
    assert value["same_model_cross_engine_low_yaw_undertracking_raw_verified"]
    assert value["source_checkpoint_sha256"] == ANCHOR_SHA
    assert value["source_actual_updates"] == 7401 and value["source_all17_Adam_step"] == 148020
    assert value["planned_actual_updates"] == 200 and value["target_actual_updates"] == 7601
    assert value["next_configuration_changes"] == DELTA and value["relative_error_target"] == .1
    assert not any(value[key] for key in ("optimizer_reset", "std_reset", "actual_LR_reset",
                                        "original_model200_reload", "acceptance_changed"))
    assert all(sha(path) == digest for path, digest in value["source_pins"].items())
    for item in value["required_complete_review_receipts"]:
        assert json.loads(Path(item["path"]).read_text())[item["completion_key"]] is True
    return value


def review_ready(source):
    repair_ready()
    value = json.loads(ADMISSION.read_text())
    assert value["source_checkpoint_sha256"] == sha(source) == ANCHOR_SHA
    assert value["selected_num_envs"] == NUM_ENVS and value["next_configuration_changes"] == DELTA
    assert value["CPU_qualified"] and value["frozen27_qualified"]
    assert value["proposal_decision_sha256"] == sha(DECISION)
    assert all(sha(path) == digest for path, digest in value["source_pins"].items())
    from check_plain27_source7401_D100_C400_focusgrid_environment_capacity import require_capacity_selection
    require_capacity_selection(value["capacity_verified_summary"],
                               block_id=value["capacity_block_id"], source=source, num_envs=NUM_ENVS)
    return value


def observe_drive(env, output, selected_profile):
    return _observe_drive(env, output, selected_profile)


def sync():
    api.NUM_ENVS = NUM_ENVS


def private_for_count(count, profile_api=None):
    sync()
    callback = profile_for_count_callback(count) if profile_api is None else profile_api
    module = _private_count(count, callback)
    inherited_observe = module.observe_environment

    def observe(env, output):
        result = inherited_observe(env, output)
        raw, output = env.unwrapped, Path(output)
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        focus.install(raw)
        assert torch.equal(before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        write_json(output / "weighted_joint_grid_runtime_evidence.json", dict(
            profile=callback(), actual_environments=raw.num_envs,
            sampler_source=str(focus.SCRIPT), sampler_source_sha256=sha(focus.SCRIPT),
            sampler_AST_evidence=focus.AST_EVIDENCE,
            original_fraction_and_fixed_standing_population_preserved=True,
            source_initial_std_Adam_and_LR_not_changed=True,
            install_added_no_RNG_draw_or_resample=True,
            distribution_stored_in_checkpoint_runtime_profile=True,
            probability_accounting=focus.expected_probability_changes(),
            rewards_control_physics_and_acceptance_unchanged=True, policy_accepted=False))
        previous = raw._pre_physics_step
        snapshots = 0
        low, high = raw.cfg.commands.ranges_height
        cells = [[speed, yaw, height] for speed in (-.8, -.6, -.2, .2, .6, .8)
                 for yaw in (-.15, -.075, .075, .15) for height in (low, high)]
        cells += [[0., yaw, high] for yaw in (-.075, .075)]
        targets = raw._commands.new_tensor(cells)
        calls = 0

        def pre_step(actions):
            nonlocal calls, snapshots
            previous(actions)
            calls += 1
            if calls % 48:
                return
            applied = raw._commands[:, :3]
            requested = applied.clone()
            requested[:, 0] = raw._target_lin_vel_x
            counts = torch.stack([
                ((values[:, None, :] - targets[None, :, :]).abs().amax(dim=-1) <= 1.e-6).sum(dim=0)
                for values in (requested, applied)
            ]).cpu().tolist()
            snapshots += 1
            row = dict(policy_steps=calls, original_update_snapshot=snapshots,
                       actual_environments=raw.num_envs, cells_v_yaw_height=cells,
                       requested_target_counts=counts[0], actual_slew_applied_counts=counts[1],
                       whole_population_sampled_at_update_boundaries_only=True,
                       not_exact_continuous_time_exposure=True, policy_accepted=False)
            with (output / "weighted_joint_grid_exposure_snapshots.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        raw._pre_physics_step = pre_step
        return result

    module.observe_environment = observe
    return module


def profile_for_count_callback(count):
    return lambda: profile_for_count(count)


def private_runtime(source=None):
    if source is not None:
        require_source(source)
    return private_for_count(NUM_ENVS)


def install_leg_exploration(output, profile_api=None):
    callback = profile if profile_api is None else profile_api
    def log(*values, **kwargs):
        if values and isinstance(values[0], str) and "User-requested leg-only exploration floor" in values[0]:
            values = ("[INFO]: Original leg floor .01 and wheel/global .005 retained; only targeted grid probabilities change.", *values[1:])
        print(*values, **kwargs)
    return bind(projection.install_leg_exploration, LEG_STD_FLOOR=LEG_STD_FLOOR,
                GLOBAL_STD_FLOOR=GLOBAL_STD_FLOOR, WHEEL_STD_FLOOR=WHEEL_STD_FLOOR,
                profile=callback, print=log)(output)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    sync()
    saved, current = require_source(source)
    assert updates == 200 and cfg == current and commands == current["commands"]
    overrides = kwargs.pop("reward_overrides", {})
    assert not overrides or overrides == {"rewards.yaw_rate_error_sq": -400.}
    assert kwargs.pop("minimum_wheel_std", .005) in (.005, .01, .03)
    assert kwargs.pop("joint_mean_guard_coef", 0.) == 0.
    assert kwargs.pop("joint_height_feedback_coef", 0.) == 0. and not kwargs
    anchor, _, before = baseline()
    argv, _, _ = retained.command(anchor, before, label, 200, before["commands"],
                                 Path(stop_file), minimum_wheel_std=.005)
    argv[1] = str(SCRIPT)
    for flag, value in {"--checkpoint_path": source, "--target_iteration": saved_updates(source) + 200,
                        "--num_envs": NUM_ENVS, "--observation-output": Path(stop_file).parent / "native_learning_signal",
                        "--run_name": label, "--stop_file": Path(stop_file)}.items():
        argv[argv.index(flag) + 1] = str(value)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    assert not any(flag in argv for flag in ("--reset_optimizer", "--reset_exploration_std", "--save_initial_checkpoint"))
    assert retained.FLAG not in argv
    return argv, saved_updates(source) + 199, saved


def native(output, arguments):
    sync()
    return _native(output, arguments)


evaluator_tree = retained.evaluator_tree


def replay_command(source, output):
    require_source(source)
    argv = retained.replay_command(baseline()[0], output)
    argv[argv.index(str(retained.SCRIPT))] = str(SCRIPT)
    argv[argv.index("--checkpoint") + 1] = str(Path(source).resolve())
    return argv


def replay(arguments):
    arguments = list(arguments)
    assert FLAG not in arguments and "--overrides_json" not in arguments
    source = Path(arguments[arguments.index("--checkpoint") + 1])
    require_source(source)
    family._sampler_api.install_sampling_field_restore()
    import gymnasium as gym
    original_make = gym.make
    gym.make = family._sampler_api._wrap_make(original_make)
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(path), "exec"), dict(__name__="__main__", __file__=str(path),
             _install_quality_criteria=criteria_fix.install, _configure_sampling=family._sampler_api._configure,
             _observe_source200=private_runtime().observe_environment))
    finally:
        gym.make = original_make


for name in ("SCRIPT", "RESULTS", "DECISION", "ADMISSION", "SOURCE_REPORT", "ANCHOR_SHA", "FLAG",
             "DAMPING", "NUM_ENVS", "DELTA", "baseline", "profile", "profile_for_count", "require_source",
             "repair_ready", "review_ready", "expected_configuration", "observe_drive", "install_leg_exploration", "recipe"):
    setattr(api, name, globals()[name])
api.private_runtime = private_runtime


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "native":
        boundary = sys.argv.index("--")
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    elif mode == "replay":
        replay(sys.argv[1:])
    else:
        raise ValueError("Only retained-state ordinary PPO and frozen replay")
