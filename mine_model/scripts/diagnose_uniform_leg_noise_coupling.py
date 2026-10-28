"""Complete a matched frozen 2x2 leg/wheel-noise experiment on all17 cases."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import diagnose_uniform_cost050_policy_drift as first
import probe_frozen_joint_noise_vectors as probe
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()


def qualify(reference, output):
    assert not output.exists()
    initial = json.loads((reference / "qualification.json").read_text())
    done = json.loads((reference / "complete_frozen_drift_and_sampling_diagnosis.json").read_text())
    assert initial["qualified"] and done["complete"] and done["each_pair_exact_same_Gaussian_draws"]
    runtime = first.runtime
    runtime.configure(initial["recipe"])
    item = initial["models"]["source"]
    state, cfg = runtime.require_source(Path(item["checkpoint"]))
    saved = state["model_state_dict"]["std"].cpu().numpy()
    vectors, changed = {}, {}
    for label, indices in (("leg_only_low", [0, 1, 3, 4]), ("both_low", [0, 1, 2, 3, 4, 5])):
        vector = saved.copy()
        vector[indices] = np.float32(.01)
        vectors[label], changed[label] = vector.tolist(), indices
    tree, path, proof = probe.original.evaluator_tree()
    compile(tree, str(path), "exec")
    pins = dict(initial["source_pins"])
    pins.update({str(p): sha(p) for p in [SCRIPT, probe.SCRIPT, first.SCRIPT,
        reference / "qualification.json", reference / "complete_frozen_drift_and_sampling_diagnosis.json",
        reference / "source_saved/sampling_noise.npz", reference / "source_probe/sampling_noise.npz",
        reference / "source_saved/policy_inputs.npz", reference / "source_saved/resolved_env.json"]})
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        reference_output=str(reference), reference_specification=str(reference / "qualification.json"),
        checkpoint=item["checkpoint"], checkpoint_sha256=item["checkpoint_sha256"],
        actual_updates=item["actual_updates"], all17_Adam_step=item["actual_updates"] * 20,
        saved_std=saved.tolist(), sampling_vectors=vectors, changed_action_indices=changed,
        cases=initial["cases"], seeds=initial["seeds"], seconds=20., source_pins=pins,
        no_condition_selects_a_sampling_vector=True, original_four_hooks=proof,
        saved_weights_std_Adam_LR_unchanged=True, training_modified=False,
        acceptance_changed=False, policy_accepted=False)
    output.mkdir()
    write_json(output / "qualification.json", spec)
    return spec


def physical_summary(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    data = np.genfromtxt(directory / "trace.csv", delimiter=",", names=True)
    assert all(np.isfinite(data[key]).all() for key in data.dtype.names)
    rows = []
    for cid, case in enumerate(manifest["cases"]):
        for seed in manifest["seeds"]:
            full = data[(data["case_id"] == cid) & (data["seed"] == seed)]
            tail = full[full["time_s"] >= 17.]
            error = tail["yaw_rate_rad_s"] - tail["yaw_cmd_rad_s"]
            squared = float(np.mean(error ** 2))
            bias2 = float(np.mean(error) ** 2)
            variance = float(np.var(error))
            assert abs(squared - bias2 - variance) < 1.e-12
            rows.append(dict(case=case, seed=seed,
                survived=not any(np.any(full[key]) for key in ("episode", "terminated", "timeout")),
                yaw_MAE_rad_s=float(np.mean(abs(error))), yaw_MSE_rad2_s2=squared,
                yaw_signed_bias_rad_s=float(np.mean(error)), yaw_bias_squared=bias2,
                yaw_temporal_variance=variance))
    assert len(rows) == 51
    return dict(trials=rows, mean_yaw_MAE_rad_s=float(np.mean([r["yaw_MAE_rad_s"] for r in rows])),
        mean_yaw_MSE=float(np.mean([r["yaw_MSE_rad2_s2"] for r in rows])),
        mean_yaw_bias_squared=float(np.mean([r["yaw_bias_squared"] for r in rows])),
        mean_yaw_temporal_variance=float(np.mean([r["yaw_temporal_variance"] for r in rows])))


def run(output):
    specification = output / "qualification.json"
    spec = json.loads(specification.read_text())
    assert spec["qualified"] and all(sha(p) == digest for p, digest in spec["source_pins"].items())
    reference = Path(spec["reference_output"])
    runtime = first.runtime
    runtime.configure(json.loads(Path(spec["reference_specification"]).read_text())["recipe"])
    commands = output / "commands.json"
    write_json(commands, spec["cases"])
    summaries = {}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for label in spec["sampling_vectors"]:
        directory = output / label
        argv = runtime.replay_command(Path(spec["checkpoint"]), directory)
        argv[1] = str(probe.SCRIPT)
        at = argv.index("--recipe")
        del argv[at:at + 2]
        at = argv.index("--seeds") + 1
        argv[at:at + 3] = list(map(str, spec["seeds"]))
        argv += ["--noise-specification", str(specification), "--noise-label", label,
                 "--sample-policy-actions", "--command-probe-json", str(commands)]
        write_json(output / f"{label}_launch.json", dict(command=argv,
            saved_checkpoint_sha256=spec["checkpoint_sha256"], diagnosis_only=True))
        write_json(output / "status.json", dict(state=label, diagnostic_active=True,
            training_active=False, goal_remains_active=True, policy_accepted=False))
        with (output / f"{label}.log").open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summaries[label] = first.assess_sampling(directory)["summary"]
        integrity = json.loads((directory / "sampling_probe_integrity.json").read_text())
        assert integrity["all_model_tensors_bitwise_unchanged"] and integrity["additional_RNG_draws"] == 0
        with np.load(reference / "source_saved/sampling_noise.npz") as a, np.load(directory / "sampling_noise.npz") as b:
            assert np.array_equal(a["epsilon"], b["epsilon"])
        with np.load(reference / "source_saved/policy_inputs.npz") as a, np.load(directory / "policy_inputs.npz") as b:
            assert np.array_equal(a["observations"][0], b["observations"][0])
            assert np.array_equal(a["means"][0], b["means"][0])
        old_cfg = json.loads((reference / "source_saved/resolved_env.json").read_text())
        new_cfg = json.loads((directory / "resolved_env.json").read_text())
        new_cfg["log_dir"] = old_cfg["log_dir"]
        assert old_cfg == new_cfg
        write_json(output / "progress.json", dict(completed_scopes=list(summaries),
            summaries=summaries, policy_accepted=False))
    all_paths = dict(saved=reference / "source_saved", wheel_only_low=reference / "source_probe",
                     **{label: output / label for label in spec["sampling_vectors"]})
    physical = {label: physical_summary(path) for label, path in all_paths.items()}
    assert all(sha(p) == digest for p, digest in spec["source_pins"].items())
    write_json(output / "complete_noise_factorial_diagnosis.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), checkpoint_sha256=spec["checkpoint_sha256"],
        new_summaries=summaries, physical=physical, same_original_Gaussian_draws_all_four_arms=True,
        initial_observations_and_means_bitwise_equal=True, physical_configuration_equal=True,
        saved_weights_std_and_Adam_unchanged=True, training_modified=False, acceptance_changed=False,
        cannot_establish_trained_improvement_or_policy_acceptance=True, policy_accepted=False,
        goal_remains_active=True))
    write_json(output / "status.json", dict(state="complete_agent_analysis_required", diagnostic_active=False,
        training_active=False, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if args.qualify_only:
        result = qualify(args.reference.resolve(), args.output.resolve())
        print(json.dumps(dict(qualified=True, cases=len(result["cases"]), sampling_vectors=result["sampling_vectors"])))
    else:
        run(args.output.resolve())
