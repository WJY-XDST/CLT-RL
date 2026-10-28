"""Generic frozen six-action noise-vector ablation, never a training rule."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

import probe_uniform_scalar_frozen_sampling as original
from joint_pd_continuation import sha

SCRIPT = Path(__file__).resolve()


def replay(arguments, specification, label):
    spec = json.loads(Path(specification).read_text())
    assert spec["qualified"] and label in spec["sampling_vectors"]
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    _, model, _, _ = original.bound(spec["reference_specification"], "source")
    assert model["checkpoint_sha256"] == spec["checkpoint_sha256"]
    assert Path(arguments[arguments.index("--checkpoint") + 1]).resolve() == Path(model["checkpoint"])
    assert all(flag in arguments for flag in ("--sample-policy-actions", "--command-probe-json", "--record_policy_inputs"))
    assert not any(flag in arguments for flag in ("--overrides_json", "--wheel_probe", "--bias-probe-json", "--export_dir"))
    tree, path, proof = original.evaluator_tree()
    metadata, noises = {}, []

    def sampling_std(saved, checkpoint, actor):
        expected = saved.new_tensor(model["saved_std"])
        applied = saved.new_tensor(spec["sampling_vectors"][label])
        assert torch.equal(saved, expected) and torch.equal(actor.std, saved)
        assert torch.isfinite(applied).all() and (applied > 0).all()
        changed = torch.nonzero(applied != saved, as_tuple=False).flatten().cpu().tolist()
        assert changed == spec["changed_action_indices"][label]
        metadata.update(proof, probe_entry_sha256=sha(SCRIPT),
            specification=str(Path(specification).resolve()), specification_sha256=sha(specification),
            checkpoint_sha256=spec["checkpoint_sha256"], sampling_label=label,
            saved_std=saved.cpu().tolist(), sampling_std=applied.cpu().tolist(),
            only_changed_action_indices=changed, saved_std_updated=False,
            diagnostic_is_not_training_or_acceptance=True,
            same_noise_vector_for_every_command=True)
        return applied

    def save(directory, applied, actor, checkpoint):
        state = actor.state_dict()
        assert state.keys() == checkpoint["model_state_dict"].keys()
        assert all(torch.equal(value, checkpoint["model_state_dict"][key]) for key, value in state.items())
        assert sha(model["checkpoint"]) == spec["checkpoint_sha256"] and len(noises) == 1000
        np.savez_compressed(Path(directory) / "sampling_noise.npz", epsilon=np.stack(noises),
            sampling_std=applied.cpu().numpy(), saved_std=actor.std.detach().cpu().numpy())
        with (Path(directory) / "sampling_probe_integrity.json").open("x") as stream:
            json.dump(dict(**metadata, all_model_tensors_bitwise_unchanged=True,
                gaussian_draw_count=len(noises), additional_RNG_draws=0), stream, indent=2)
            stream.write("\n")

    runtime = original.runtime
    sys.argv = [str(SCRIPT), *runtime.base.tires.arguments(arguments)]
    exec(compile(tree, str(path), "exec"), dict(__name__="__main__", __file__=str(path),
        _install_quality_criteria=runtime.criteria_fix.install,
        _observe_source200=runtime.retained.private_runtime().observe_environment,
        _sampling_probe_std=sampling_std, _sampling_probe_metadata=lambda: dict(metadata),
        _record_probe_noise=lambda noise: noises.append(noise.detach().cpu().numpy().copy()),
        _save_probe_noise=save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--noise-specification", type=Path, required=True)
    parser.add_argument("--noise-label", required=True)
    args, rest = parser.parse_known_args()
    assert rest.pop(0) == "replay"
    replay(rest, args.noise_specification.resolve(), args.noise_label)
