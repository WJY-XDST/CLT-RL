"""Frozen sampling diagnosis; preserve every saved model/Adam tensor.

Use the original evaluator's four qualified diagnostic hooks. A different local
sampling vector is evaluation data only and never enters PPO or control logic.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

import joint_pd_uniform_scalar_runtime as runtime
import probe_frozen_joint_pd_sampling as hooks
from joint_pd_continuation import sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()


def bound(specification, role):
    spec = json.loads(Path(specification).read_text())
    assert spec["qualified"] and role in ("source", "candidate")
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    runtime.configure(spec["recipe"])
    item = spec["models"][role]
    checkpoint = Path(item["checkpoint"])
    saved, cfg = runtime.require_source(checkpoint)
    assert sha(checkpoint) == item["checkpoint_sha256"]
    assert tensor_finite(saved) and saved_updates(checkpoint) == item["actual_updates"]
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(value["step"]) == 20 * item["actual_updates"]
               for value in saved["optimizer_state_dict"]["state"].values())
    assert saved["model_state_dict"]["std"].shape == (6,)
    return spec, item, saved, cfg


def evaluator_tree():
    adapter = SimpleNamespace(evaluator_tree=runtime.retained.evaluator_tree)
    tree, path, proof = hooks.evaluator_tree(adapter)
    proof.update(probe_entry_sha256=sha(SCRIPT),
                 original_uniform_runtime_sha256=sha(runtime.SCRIPT))
    return tree, path, proof


def replay(arguments, specification, role, label):
    spec, item, saved, cfg = bound(specification, role)
    assert label in ("saved", "probe")
    assert Path(arguments[arguments.index("--checkpoint") + 1]).resolve() == Path(item["checkpoint"])
    assert all(flag in arguments for flag in ("--sample-policy-actions", "--command-probe-json", "--record_policy_inputs"))
    assert not any(flag in arguments for flag in ("--overrides_json", "--wheel_probe", "--bias-probe-json", "--export_dir"))
    tree, path, evidence = evaluator_tree()
    metadata, noises = {}, []

    def sampling_std(original, checkpoint, actor):
        expected = original.new_tensor(item["saved_std"])
        applied = original.new_tensor(item["sampling_vectors"][label])
        assert torch.equal(original, expected) and torch.equal(actor.std, original)
        assert torch.isfinite(applied).all() and (applied > 0).all()
        changed = torch.nonzero(applied != original, as_tuple=False).flatten().cpu().tolist()
        assert changed == ([] if label == "saved" else [2, 5])
        metadata.update(evidence, specification=str(Path(specification).resolve()),
            specification_sha256=sha(specification), role=role, sampling_label=label,
            checkpoint_sha256=item["checkpoint_sha256"], saved_std=original.cpu().tolist(),
            sampling_std=applied.cpu().tolist(), only_changed_action_indices=changed,
            diagnostic_is_not_training_or_acceptance=True, saved_std_updated=False)
        return applied

    def save(directory, applied, actor, checkpoint):
        current = actor.state_dict()
        assert current.keys() == checkpoint["model_state_dict"].keys()
        assert all(torch.equal(value, checkpoint["model_state_dict"][key]) for key, value in current.items())
        assert sha(item["checkpoint"]) == item["checkpoint_sha256"] and len(noises) == 1000
        np.savez_compressed(Path(directory) / "sampling_noise.npz", epsilon=np.stack(noises),
            sampling_std=applied.cpu().numpy(), saved_std=actor.std.detach().cpu().numpy())
        with (Path(directory) / "sampling_probe_integrity.json").open("x") as stream:
            json.dump(dict(**metadata, all_model_tensors_bitwise_unchanged=True,
                gaussian_draw_count=len(noises), additional_RNG_draws=0), stream, indent=2)
            stream.write("\n")

    observer = (runtime.retained.private_runtime().observe_environment if role == "source"
                else runtime.private_for_count(cfg["scene"]["num_envs"], is_replay=True).observe_environment)
    sys.argv = [str(SCRIPT), *runtime.base.tires.arguments(arguments)]
    exec(compile(tree, str(path), "exec"), dict(__name__="__main__", __file__=str(path),
        _install_quality_criteria=runtime.criteria_fix.install, _observe_source200=observer,
        _sampling_probe_std=sampling_std, _sampling_probe_metadata=lambda: dict(metadata),
        _record_probe_noise=lambda noise: noises.append(noise.detach().cpu().numpy().copy()),
        _save_probe_noise=save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--source-role", choices=("source", "candidate"), required=True)
    parser.add_argument("--sampling-label", choices=("saved", "probe"), required=True)
    args, remaining = parser.parse_known_args()
    assert remaining.pop(0) == "replay"
    replay(remaining, args.specification.resolve(), args.source_role, args.sampling_label)
