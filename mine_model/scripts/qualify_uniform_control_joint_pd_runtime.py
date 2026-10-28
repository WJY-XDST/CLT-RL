# Versioned CPU qualification: identical NN/std/all17 Adam/LR roundtrip checks;
# command differences additionally declare the coupled global wheel-drive scalar.
"""Check exact state recovery and globally configured scalar changes before Kit."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import rsl_rl.runners as runners

import joint_pd_uniform_control_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def qualify(recipe_path, output):
    runtime.configure(recipe_path)
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    source, saved, cfg = runtime.baseline()
    agent = yaml.safe_load((source.parent / "params/agent.yaml").read_text())
    options = dict(agent["policy"])
    assert options.pop("class_name") == "ActorCritic" and agent["algorithm"]["class_name"] == "PPO"
    results = []
    for count in (8192, 4096):
        runtime.NUM_ENVS = count
        root = output / str(count)
        root.mkdir()
        for directory in ("first_load", "exploration", "second_load"):
            (root / directory).mkdir()
        observations = TensorDict({"policy": torch.randn(32, 27)}, [32])

        def bare(cls):
            with contextlib.redirect_stdout(io.StringIO()):
                policy = ActorCritic(observations, agent["obs_groups"], 6, **options)
            runner = object.__new__(cls)
            runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
            runner.current_learning_iteration = 0
            runner.logger_type, runner.disable_logs = "tensorboard", True
            return runner

        upstream, cls = runtime.private_for_count(count).install_runner(root / "first_load")
        try:
            runner = bare(cls)
            runner.load(str(source), load_optimizer=True, map_location="cpu")
            runtime.base.equal_state(saved["model_state_dict"], runner.alg.policy.state_dict(), "NN_and_six_std")
            runtime.base.equal_state(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict(), "all17_Adam")
            means = runner.alg.policy.act_inference(observations).detach().clone()
            import wheel_legged_gym_isaaclab.exploration as exploration
            original = runtime.install_leg_exploration(root / "exploration")
            try:
                floors = runtime.recipe()["exploration"]
                handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
                    floors["global_minimum_std"], minimum_wheel_std=floors["minimum_wheel_std"])
                runtime.base.equal_state(saved["model_state_dict"], runner.alg.policy.state_dict(), "guard_install_NN")
                runtime.base.equal_state(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict(), "guard_install_Adam")
                assert all(parameter.requires_grad for parameter in runner.alg.policy.parameters())
                handle.remove()
            finally:
                exploration.enforce_positive_exploration = original
            offline = root / "roundtrip/offline_metadata_not_trained.pt"
            shutil.copytree(source.parent / "params", offline.parent / "params")
            expected = runtime.expected_configuration(cfg, count)
            (offline.parent / "params/env.yaml").write_text(yaml.safe_dump(expected, sort_keys=False))
            runner.save(str(offline))
            runners.OnPolicyRunner = upstream
            up2, cls2 = runtime.private_for_count(count).install_runner(root / "second_load")
            try:
                restored = bare(cls2)
                restored.load(str(offline), load_optimizer=True, map_location="cpu")
                runtime.base.equal_state(saved["model_state_dict"], restored.alg.policy.state_dict(), "roundtrip_NN")
                runtime.base.equal_state(saved["optimizer_state_dict"], restored.alg.optimizer.state_dict(), "roundtrip_Adam")
                assert restored.alg.learning_rate == saved["optimizer_state_dict"]["param_groups"][0]["lr"]
                assert torch.equal(means, restored.alg.policy.act_inference(observations))
            finally:
                runners.OnPolicyRunner = up2
            argv, final, _ = runtime.command(source, cfg, "uniform_CPU_fixture", 200, cfg["commands"], root / "NOT_A_STOP_REQUEST")
            assert int(argv[argv.index("--target_iteration") + 1]) == runtime.saved_updates(source) + 200
            assert final == runtime.saved_updates(source) + 199
            assert argv[argv.index("--num_envs") + 1] == str(count) and "agent.save_interval=1" in argv
            reference, _, _ = runtime.retained.command(source, cfg, "uniform_CPU_fixture", 200, cfg["commands"], root / "NOT_A_STOP_REQUEST")
            expected_argv = list(reference)
            expected_argv[1] = str(runtime.SCRIPT)
            expected_argv[3:3] = ["--recipe", str(runtime.RECIPE_PATH)]
            for flag, value in {"--num_envs": count, "--min_exploration_std": floors["global_minimum_std"],
                                "--min_wheel_exploration_std": floors["minimum_wheel_std"]}.items():
                expected_argv[expected_argv.index(flag) + 1] = str(value)
            expected_argv[expected_argv.index("agent.save_interval=5")] = "agent.save_interval=1"
            for key, value in runtime.recipe()["reward_overrides"].items():
                indices = [i for i, argument in enumerate(expected_argv) if argument.startswith("env.rewards." + key + "=")]
                assert len(indices) == 1
                expected_argv[indices[0]] = "env.rewards." + key + "=" + str(value)
            for key, value in runtime.recipe()["control_overrides"].items():
                expected_argv.append("env." + key + "=" + str(value))
            assert argv == expected_argv
            results.append(dict(count=count, qualified=True, all_saved_NN_17Adam_sixstd_LR_and_inference_bitwise_preserved=True,
                actual_PPO_updates=0, original_native_arguments_equal_except_declared_global_scalars_parallelism_and_save_interval=True))
        finally:
            runners.OnPolicyRunner = upstream
    retired = [name for name in sys.modules if name in ("historical_yaw_error_focus", "historical_yaw_focus_sampler",
        "historical_yaw_focus_sampler_v2", "joint_pd_source3001_yaw_error_focus_runtime", "joint_pd_source3001_conditional_yaw_runtime")]
    assert not retired
    import source1001_reference_smooth10 as capture
    reward_token = torch.ones(3)

    class Dummy:
        def _get_rewards(self):
            return reward_token

    dummy = Dummy()
    dummy.reset_terminated = torch.zeros(3, dtype=torch.bool)
    dummy.common_step_counter = 0
    capture.install_failure_capture(dummy, output)
    chain = runtime.verify_unmodified_execution(dummy)
    rng = torch.get_rng_state().clone()
    assert dummy._get_rewards() is reward_token and torch.equal(rng, torch.get_rng_state())
    assert chain["readonly_terminal_capture_wrappers"] == 1
    pins = {str(p): sha(p) for p in (SCRIPT, runtime.SCRIPT, Path(recipe_path), source,
        source.parent / "params/env.yaml", source.parent / "params/agent.yaml")}
    report = dict(qualified=True, results=results, original_uniform_reward_structure_and_continuous_sampler_selected=True,
        global_reward_overrides=runtime.recipe()["reward_overrides"],
        global_control_overrides=runtime.recipe()["control_overrides"],
        global_control_changes=runtime.recipe()["control_changes"],
        retired_runtime_modules_loaded=retired, actual_PPO_updates=0, CPU_roundtrips_not_training_sources=True,
        only_original_readonly_wrapper_accepted_and_reward_identity_RNG_preserved=True,
        source_pins=pins, policy_accepted=False)
    write_json(output / "qualification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.recipe, args.output)
    print(json.dumps({key: result[key] for key in ("qualified", "results", "retired_runtime_modules_loaded", "actual_PPO_updates")}))
