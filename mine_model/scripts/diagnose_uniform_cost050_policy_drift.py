"""Compare saved mid-block policies and matched frozen wheel-noise vectors.

This controller never trains. Command grids are read-only evaluation data.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import types

import numpy as np

import joint_pd_uniform_scalar_runtime as runtime
import probe_uniform_scalar_frozen_sampling as probe
import diagnose_motion025_full_height as cases_module
import review_historical_source3001_conditional_yaw200 as readers
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()


def qualify(training, output, recipe):
    assert not output.exists()
    runtime.configure(recipe)
    final = json.loads((training / "final_checkpoint_verification_readonly.json").read_text())
    assert final["verified"] and final["actual_ppo_updates"] == 3201
    models = {}
    for role, checkpoint in (("source", runtime.baseline()[0]), ("candidate", Path(final["checkpoint"]))):
        saved, cfg = runtime.require_source(checkpoint)
        std = saved["model_state_dict"]["std"].cpu().numpy()
        changed = std.copy()
        changed[[2, 5]] = np.float32(.01)
        models[role] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            actual_updates=runtime.saved_updates(checkpoint), saved_std=std.tolist(),
            sampling_vectors=dict(saved=std.tolist(), probe=changed.tolist()))
    native = json.loads((training / "policy_update_health_prefix_0200_readonly.json").read_text())
    mid = []
    for update in (50, 100):
        row = next(row for row in native["snapshots"] if row["committed_update"] == update)
        checkpoint = Path(row["checkpoint_after"])
        saved, _ = runtime.require_source(checkpoint)
        assert sha(checkpoint) == row["checkpoint_after_sha256"]
        assert row["actual_updates_before_after"][1] == 3001 + update
        mid.append(dict(prefix=update, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                        actual_updates=3001 + update))
    tree, evaluator, proof = probe.evaluator_tree()
    compile(tree, str(evaluator), "exec")
    pins = dict(runtime.recipe()["source_pins"])
    extra = [SCRIPT, probe.SCRIPT, probe.hooks.SCRIPT, probe.hooks.template.SCRIPT,
             Path(recipe), training / "final_checkpoint_verification_readonly.json",
             Path(evaluator), Path(readers.__file__), Path(cases_module.__file__)]
    for item in [*models.values(), *mid]:
        path = Path(item["checkpoint"])
        extra.extend([path, path.parent / "params/env.yaml", path.parent / "params/agent.yaml"])
    pins.update({str(path): sha(path) for path in extra})
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        recipe=str(recipe), training_output=str(training), models=models, intermediate_checkpoints=mid,
        cases=cases_module.cases(), seeds=[483, 484, 485], seconds=20.,
        source_pins=pins, original_evaluator_hook_qualification=proof,
        training_modified=False, optimizer_updated=False, saved_std_updated=False,
        command_grid_enters_training=False, acceptance_changed=False, policy_accepted=False)
    output.mkdir()
    write_json(output / "qualification.json", spec)
    return spec


def assess_sampling(directory):
    """Adapt the single sampled-policy metadata assertion, preserve numerics."""
    manifest_path = directory / "manifest.json"
    original = readers.read
    metadata_reads = []

    def read(path):
        data = original(path)
        if Path(path) == manifest_path:
            assert data["sampled_policy_actions"] is True
            metadata_reads.append("sampled_policy_actions")
            data = dict(data, sampled_policy_actions=False)
        return data

    function = types.FunctionType(readers.assess.__code__, dict(readers.__dict__, read=read),
                                  readers.assess.__name__)
    result = function(directory)
    assert metadata_reads == ["sampled_policy_actions"]
    result.update(sampling_diagnostic_only=True, policy_accepted=False,
                  original_numeric_metrics_limits_windows_unchanged=True,
                  adapted_metadata_assertions=metadata_reads,
                  cannot_prove_deterministic_acceptance_or_training_improvement=True)
    write_json(directory / "sampled_physical_diagnosis.json", result)
    return result


def run(output):
    specification = output / "qualification.json"
    spec = json.loads(specification.read_text())
    runtime.configure(spec["recipe"])
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    command_plan = output / "commands.json"
    write_json(command_plan, spec["cases"])
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    reports = {}
    work = [(f"prefix{item['prefix']:03d}_deterministic", item, None, None)
            for item in spec["intermediate_checkpoints"]]
    work.extend((f"{role}_{label}", spec["models"][role], role, label)
                for role in ("source", "candidate") for label in ("saved", "probe"))
    for name, item, role, label in work:
        destination = output / name
        if role is None:
            argv = runtime.replay_command(Path(item["checkpoint"]), destination)
        else:
            # Both roles use the unchanged evaluator/physics and their own guards.
            argv = runtime.replay_command(Path(item["checkpoint"]), destination)
            argv[1] = str(probe.SCRIPT)
            at = argv.index("--recipe")
            del argv[at:at + 2]
            argv += ["--specification", str(specification), "--source-role", role,
                     "--sampling-label", label, "--sample-policy-actions"]
        argv[argv.index("--seconds") + 1] = str(spec["seconds"])
        at = argv.index("--seeds") + 1
        argv[at:at + 3] = list(map(str, spec["seeds"]))
        argv += ["--command-probe-json", str(command_plan)]
        write_json(output / f"{name}_launch.json", dict(command=argv, checkpoint_sha256=item["checkpoint_sha256"],
            diagnosis_only=True, weights_and_optimizer_updated=False))
        write_json(output / "status.json", dict(state=name, diagnostic_active=True, training_active=False,
                   policy_accepted=False, goal_remains_active=True))
        with (output / f"{name}.log").open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        reports[name] = readers.assess(destination) if role is None else assess_sampling(destination)
        if role is not None:
            integrity = json.loads((destination / "sampling_probe_integrity.json").read_text())
            assert integrity["all_model_tensors_bitwise_unchanged"] and integrity["additional_RNG_draws"] == 0
        write_json(output / "progress.json", dict(completed_scopes=list(reports),
            summaries={key: value["summary"] for key, value in reports.items()}, policy_accepted=False))
    for role in ("source", "candidate"):
        with np.load(output / f"{role}_saved/sampling_noise.npz") as a, np.load(output / f"{role}_probe/sampling_noise.npz") as b:
            assert np.array_equal(a["epsilon"], b["epsilon"])
        with np.load(output / f"{role}_saved/policy_inputs.npz") as a, np.load(output / f"{role}_probe/policy_inputs.npz") as b:
            assert np.array_equal(a["observations"][0], b["observations"][0])
            assert np.array_equal(a["means"][0], b["means"][0])
        configs = [json.loads((output / f"{role}_{label}/resolved_env.json").read_text()) for label in ("saved", "probe")]
        configs[1]["log_dir"] = configs[0]["log_dir"]
        assert configs[0] == configs[1]
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    result = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        qualification_sha256=sha(specification), summaries={key: value["summary"] for key, value in reports.items()},
        each_pair_exact_same_Gaussian_draws=True, each_pair_bitwise_equal_initial_observations_and_means=True,
        saved_weights_std_and_optimizer_unchanged=True, training_modified=False, acceptance_changed=False,
        policy_accepted=False, goal_remains_active=True)
    write_json(output / "complete_frozen_drift_and_sampling_diagnosis.json", result)
    write_json(output / "status.json", dict(state="complete_agent_analysis_required", diagnostic_active=False,
        training_active=False, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--training", type=Path)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    if args.qualify_only:
        assert args.training and args.recipe
        result = qualify(args.training.resolve(), args.output.resolve(), args.recipe.resolve())
        print(json.dumps(dict(qualified=result["qualified"], cases=len(result["cases"]),
                             roles=list(result["models"]), policy_accepted=False)))
    else:
        run(args.output.resolve())
