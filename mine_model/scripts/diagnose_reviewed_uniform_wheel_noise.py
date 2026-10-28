"""Matched frozen wheel-noise diagnosis over whole completed replay grids.

This experiment changes a local evaluation sampling vector, never a checkpoint,
optimizer, training sampler, reward, actuator or per-condition controller.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


SCRIPT = Path(__file__).resolve()


def qualify(args):
    import numpy as np
    import joint_pd_uniform_scalar_runtime as runtime
    import probe_uniform_scalar_frozen_sampling as probe
    from auto_train_mine import write_json
    from joint_pd_continuation import sha

    assert not args.output.exists()
    runtime.configure(args.recipe)
    proof_path = args.training / "final_checkpoint_verification_readonly.json"
    proof = json.loads(proof_path.read_text())
    assert proof["verified"] and proof["all200_native_updates_and_original_alignment_verified"]
    checkpoint = Path(proof["checkpoint"])
    saved, _ = runtime.require_source(checkpoint)
    std = saved["model_state_dict"]["std"].cpu().numpy()
    changed = std.copy()
    assert 0. < args.wheel_std < min(float(std[2]), float(std[5]))
    changed[[2, 5]] = np.float32(args.wheel_std)
    model = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_updates=runtime.saved_updates(checkpoint), saved_std=std.tolist(),
        sampling_vectors=dict(saved=std.tolist(), probe=changed.tolist()))
    assert model["actual_updates"] == proof["actual_ppo_updates"]
    scopes = []
    paths = [SCRIPT, runtime.SCRIPT, probe.SCRIPT, probe.hooks.SCRIPT, probe.hooks.template.SCRIPT,
        args.recipe, proof_path, checkpoint, checkpoint.parent / "params/env.yaml", checkpoint.parent / "params/agent.yaml"]
    for reference in args.reference_replays:
        manifest, complete = [json.loads((reference / name).read_text()) for name in ("manifest.json", "completed.json")]
        assert complete["completed"] and manifest["seconds_per_case"] == 20.
        assert manifest["checkpoint_sha256"] == model["checkpoint_sha256"]
        assert not manifest["weights_updated"] and not manifest["sampled_policy_actions"]
        assert len(manifest["seeds"]) == 3
        scopes.append(dict(name=reference.name, reference=str(reference),
            cases=manifest["cases"], seeds=manifest["seeds"], seconds=20.))
        paths.extend(reference / name for name in ("manifest.json", "completed.json", "resolved_env.json", "trace.csv"))
    assert len({item["name"] for item in scopes}) == len(scopes)
    tree, evaluator, hook_proof = probe.evaluator_tree()
    compile(tree, str(evaluator), "exec")
    paths.append(evaluator)
    pins = dict(runtime.recipe()["source_pins"])
    pins.update({str(path.resolve()): sha(path) for path in paths})
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        recipe=str(args.recipe), models=dict(candidate=model), scopes=scopes, source_pins=pins,
        original_four_hook_qualification=hook_proof, training_output=str(args.training),
        after_owned_unit=args.after_owned_unit, checkpoint_and_all17_Adam_not_modified=True,
        one_uniform_sampling_vector_for_every_command=True, policy_accepted=False,
        training_modified=False, optimizer_executed=False)
    args.output.mkdir()
    write_json(args.output / "qualification.json", spec)
    return spec


def physical(directory):
    import numpy as np
    manifest = json.loads((directory / "manifest.json").read_text())
    completed = json.loads((directory / "completed.json").read_text())
    trace = np.genfromtxt(directory / "trace.csv", delimiter=",", names=True)
    assert completed["completed"] and len(trace) == completed["samples"]
    assert all(np.isfinite(trace[name]).all() for name in trace.dtype.names)
    rows = []
    for index, case in enumerate(manifest["cases"]):
        for seed in manifest["seeds"]:
            full = trace[(trace["case_id"] == index) & (trace["seed"] == seed)]
            tail = full[full["time_s"] >= manifest["seconds_per_case"] - 3.]
            error = tail["yaw_rate_rad_s"] - tail["yaw_cmd_rad_s"]
            mse, bias, variance = float(np.mean(error ** 2)), float(np.mean(error)), float(np.var(error))
            assert abs(mse - bias ** 2 - variance) < 1.e-12
            rows.append(dict(case=case, seed=seed,
                survived=not any(np.any(full[name]) for name in ("episode", "terminated", "timeout")),
                yaw_MAE_rad_s=float(np.mean(abs(error))), yaw_MSE=mse,
                yaw_signed_bias_rad_s=bias, yaw_bias_squared=bias ** 2, yaw_temporal_variance=variance))
    assert len(rows) == len(manifest["cases"]) * len(manifest["seeds"])
    return dict(trials=rows, mean_yaw_MAE_rad_s=float(np.mean([row["yaw_MAE_rad_s"] for row in rows])),
        mean_yaw_MSE=float(np.mean([row["yaw_MSE"] for row in rows])),
        mean_yaw_bias_squared=float(np.mean([row["yaw_bias_squared"] for row in rows])),
        mean_yaw_temporal_variance=float(np.mean([row["yaw_temporal_variance"] for row in rows])),
        survivors=sum(row["survived"] for row in rows))


def run(args):
    # Keep the waiting process small and outside the native simulation lock.
    specification = args.output / "qualification.json"
    spec = json.loads(specification.read_text())
    completion = Path(spec["training_output"]) / "independent_review/complete_comparison_agent_decision_required.json"
    while not completion.exists():
        state = subprocess.run(["systemctl", "--user", "show", spec["after_owned_unit"],
            "-p", "MainPID", "-p", "SubState", "-p", "Result"], capture_output=True, text=True, check=True).stdout
        owner = dict(line.split("=", 1) for line in state.splitlines() if "=" in line)
        (args.output / "waiting_owner.json").write_text(json.dumps(owner, indent=2) + "\n")
        if owner.get("MainPID") == "0" and owner.get("SubState") in ("exited", "dead", "failed"):
            raise RuntimeError("Owned review ended without completion; diagnose without restarting PPO")
        time.sleep(5.)
    import numpy as np
    import joint_pd_uniform_scalar_runtime as runtime
    import probe_uniform_scalar_frozen_sampling as probe
    from diagnose_uniform_cost050_policy_drift import assess_sampling
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, sha

    runtime.configure(spec["recipe"])
    assert all(sha(path) == value for path, value in spec["source_pins"].items())
    done = json.loads(completion.read_text())
    assert done["complete"] and done["candidate_actual_updates"] == spec["models"]["candidate"]["actual_updates"]
    completion_sha = sha(completion)
    checkpoint = Path(spec["models"]["candidate"]["checkpoint"])
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    reports = {}
    for scope in spec["scopes"]:
        commands = args.output / (scope["name"] + "_commands.json")
        write_json(commands, scope["cases"])
        diagnoses = {}
        for label in ("saved", "probe"):
            name = scope["name"] + "_" + label
            destination = args.output / name
            argv = runtime.replay_command(checkpoint, destination)
            argv[1] = str(probe.SCRIPT)
            at = argv.index("--recipe")
            del argv[at:at + 2]
            at = argv.index("--seeds") + 1
            argv[at:at + 3] = list(map(str, scope["seeds"]))
            argv += ["--specification", str(specification), "--source-role", "candidate",
                "--sampling-label", label, "--sample-policy-actions", "--command-probe-json", str(commands)]
            write_json(args.output / (name + "_launch.json"), dict(command=argv, diagnostic_only=True,
                checkpoint_sha256=spec["models"]["candidate"]["checkpoint_sha256"]))
            write_json(args.output / "status.json", dict(state=name, diagnostic_active=True,
                training_modified=False, optimizer_executed=False, policy_accepted=False, goal_remains_active=True))
            with (args.output / (name + ".log")).open("x") as stream:
                subprocess.run(argv, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            diagnoses[label] = dict(original_metrics_diagnostic_only=assess_sampling(destination)["summary"],
                physical=physical(destination))
            integrity = json.loads((destination / "sampling_probe_integrity.json").read_text())
            assert integrity["all_model_tensors_bitwise_unchanged"] and integrity["additional_RNG_draws"] == 0
        saved_dir, probe_dir = [args.output / (scope["name"] + "_" + label) for label in ("saved", "probe")]
        with np.load(saved_dir / "sampling_noise.npz") as a, np.load(probe_dir / "sampling_noise.npz") as b:
            assert np.array_equal(a["epsilon"], b["epsilon"])
            assert np.array_equal(a["saved_std"], b["saved_std"])
        with np.load(saved_dir / "policy_inputs.npz") as a, np.load(probe_dir / "policy_inputs.npz") as b:
            assert np.array_equal(a["observations"][0], b["observations"][0])
            assert np.array_equal(a["means"][0], b["means"][0])
        a, b = [json.loads((directory / "resolved_env.json").read_text()) for directory in (saved_dir, probe_dir)]
        b["log_dir"] = a["log_dir"]
        assert a == b
        reports[scope["name"]] = diagnoses
        write_json(args.output / "progress.json", dict(completed_scopes=list(reports), results=reports,
            diagnostic_only=True, training_modified=False, policy_accepted=False))
    assert all(sha(path) == value for path, value in spec["source_pins"].items())
    assert sha(completion) == completion_sha
    write_json(args.output / "complete_frozen_wheel_noise_diagnosis.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), checkpoint_sha256=sha(checkpoint), results=reports,
        same_Gaussian_draws_and_initial_observations_means_each_pair=True, physics_configurations_equal=True,
        checkpoints_and_all17_Adam_unchanged=True, no_condition_specific_training_logic=True,
        optimizer_executed=False, training_modified=False, acceptance_changed=False, policy_accepted=False,
        independent_completed_review=str(completion), independent_completed_review_sha256=completion_sha,
        limitations=["Sampled-policy noise diagnostics do not prove trained policy improvement or deterministic acceptance.",
            "Every command/seed is retained, including any failed sampled trajectories.",
            "A future floor release must preserve saved std/Adam/LR and undergo its own fresh capacity, PPO and independent reviews."]))
    write_json(args.output / "status.json", dict(state="complete_agent_decision_required", diagnostic_active=False,
        training_modified=False, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--training", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--reference-replays", type=Path, nargs="+")
    parser.add_argument("--after-owned-unit")
    parser.add_argument("--wheel-std", type=float)
    args = parser.parse_args()
    if args.qualify_only:
        assert all((args.training, args.recipe, args.reference_replays, args.after_owned_unit, args.wheel_std))
        spec = qualify(args)
        print(json.dumps(dict(qualified=True, scopes=[row["name"] for row in spec["scopes"]],
            candidate_checkpoint_sha256=spec["models"]["candidate"]["checkpoint_sha256"],
            sampling_vectors=spec["models"]["candidate"]["sampling_vectors"], optimizer_executed=False)))
    else:
        run(args)
