"""Archive one verified200 checkpoint, all parameter sidecars, and CPU-load it.

No training or simulation entry point exists. A byte-identical archive is also
checked through the existing runner's real load_optimizer=True path, including
the saved curriculum clock and actual adaptive learning rate.
"""
import argparse
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import rsl_rl.runners as runners

import joint_pd_uniform_control_runtime_v3 as runtime
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
REQUIRED = {"agent.yaml", "env.yaml", "curriculum_clock.yaml", "curriculum_start.yaml",
            "precision_mode.yaml", "control_contract.yaml"}


def preserve(recipe, proof_path, output):
    runtime.configure(recipe)
    proof = json.loads(proof_path.read_text())
    assert proof["verified"] and proof["all200_native_updates_and_original_alignment_verified"]
    source = Path(proof["checkpoint"]).resolve()
    assert sha(source) == proof["checkpoint_sha256"]
    saved, cfg = runtime.require_source(source)
    actual = runtime.saved_updates(source)
    assert actual == proof["actual_ppo_updates"] == runtime.recipe()["source_actual_updates"] + 200
    assert proof["source_checkpoint_sha256"] == runtime.recipe()["source_checkpoint_sha256"]
    assert REQUIRED <= {p.name for p in (source.parent / "params").iterdir() if p.is_file()}
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(v["step"]) == actual * 20 for v in saved["optimizer_state_dict"]["state"].values())
    assert not output.exists(), "Preserve an existing archive instead of overwriting it"
    output.mkdir(parents=True)
    checkpoint = output / source.name
    shutil.copy2(source, checkpoint)
    shutil.copytree(source.parent / "params", output / "params")
    shutil.copy2(recipe, output / "recipe.json")
    shutil.copy2(proof_path, output / "final_checkpoint_verification_readonly.json")
    original_params = sorted(p for p in (source.parent / "params").rglob("*") if p.is_file())
    parameter_files = {}
    for path in original_params:
        relative = path.relative_to(source.parent / "params")
        copied = output / "params" / relative
        assert sha(path) == sha(copied)
        parameter_files[str(relative)] = sha(copied)
    assert sha(source) == sha(checkpoint)
    copied, copied_cfg = runtime.require_source(checkpoint)
    assert copied_cfg == cfg and runtime.saved_updates(checkpoint) == actual
    runtime.base.equal_state(saved, copied, "archive_full_saved_state")
    agent = yaml.safe_load((output / "params/agent.yaml").read_text())
    options = dict(agent["policy"])
    assert options.pop("class_name") == "ActorCritic" and agent["algorithm"]["class_name"] == "PPO"
    observations = TensorDict({"policy": torch.zeros(4, 27)}, batch_size=[4])
    restore = output / "CPU_restore_verification"
    restore.mkdir()
    rng = torch.get_rng_state().clone()
    upstream, cls = runtime.private_for_count(cfg["scene"]["num_envs"]).install_runner(restore)
    try:
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(observations, agent["obs_groups"], 6, **options)
            runner = object.__new__(cls)
            runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
            runner.current_learning_iteration = 0
            runner.logger_type, runner.disable_logs = "tensorboard", True
            runner.load(str(checkpoint), load_optimizer=True, map_location="cpu")
        runtime.base.equal_state(saved["model_state_dict"], runner.alg.policy.state_dict(), "archived_NN_and_six_std")
        runtime.base.equal_state(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict(), "archived_all17_Adam")
        assert runner.alg.learning_rate == saved["optimizer_state_dict"]["param_groups"][0]["lr"]
        assert all(p.requires_grad for p in runner.alg.policy.parameters())
    finally:
        runners.OnPolicyRunner = upstream
    assert torch.equal(rng, torch.get_rng_state())
    assert sha(checkpoint) == sha(source)
    assert all(sha(output / "params" / path) == digest for path, digest in parameter_files.items())
    files = [SCRIPT, runtime.SCRIPT, recipe, proof_path, source, *original_params]
    report = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        original_checkpoint=str(source), archived_checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_updates=actual, all17_Adam_step=actual * 20,
        saved_six_std=saved["model_state_dict"]["std"].tolist(),
        saved_actual_LR=saved["optimizer_state_dict"]["param_groups"][0]["lr"],
        required_parameter_sidecars=sorted(REQUIRED), copied_parameter_files=parameter_files,
        raw_checkpoint_all_parameters_and_optimizer_byte_identical=True,
        archived_clocked_update_count_verified=True,
        actual_CPU_runner_load_optimizer_true_full_state_bitwise_verified=True,
        actual_CPU_runner_adaptive_LR_verified=True, all_six_std_entries_trainable=True,
        CPU_RNG_state_unchanged=True, optimizer_constructed_for_restore_only=True,
        optimizer_step_or_backward_executed=False, simulator_started=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False,
        source_pins={str(path): sha(path) for path in files})
    write_json(output / "preservation_manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = preserve(args.recipe.resolve(), args.proof.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ["verified", "checkpoint_sha256", "actual_updates",
        "all17_Adam_step", "actual_CPU_runner_load_optimizer_true_full_state_bitwise_verified"]}))
