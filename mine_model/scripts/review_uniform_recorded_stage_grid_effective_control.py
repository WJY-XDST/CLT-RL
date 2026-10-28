"""Reuse a frozen, declared-drive baseline to isolate subsequent PPO learning.

Saved source weights and Adam still belong to the original checkpoint. Only
qualification of baseline controller metadata admits the already measured
uniform drive intervention. All525 cases and all numerical readers are reused.
"""
import argparse
import copy
from pathlib import Path

import joint_pd_uniform_control_runtime_v3 as runtime
import review_uniform_recorded_stage_grid as original
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def qualify(args):
    runtime.configure(args.recipe)
    source, saved, cfg = runtime.baseline()
    controls, changes = runtime.recipe()["control_overrides"], runtime.recipe()["control_changes"]
    assert set(changes) == {"wheel_damping", "robot.actuators.wheels.damping"}
    assert changes["wheel_damping"] == changes["robot.actuators.wheels.damping"]
    before, after = changes["wheel_damping"]
    assert cfg["wheel_damping"] == cfg["robot"]["actuators"]["wheels"]["damping"] == before
    assert controls["wheel_damping"] == controls["robot.actuators.wheels.damping"] == after
    effective = copy.deepcopy(cfg)
    effective["wheel_damping"] = effective["robot"]["actuators"]["wheels"]["damping"] = after
    proofs = []
    for folder in args.source:
        path = folder / "frozen_uniform_wheel_drive_native_proof.json"
        proof = original.read(path)
        assert proof["verified"] and proof["source_checkpoint_sha256"] == sha(source)
        assert proof["actual_wheel_damping_min"] == proof["actual_wheel_damping_max"] == after
        assert proof["new_PPO_updates"] == 0 and proof["source_weights_std_Adam_unchanged"]
        proofs.append(path)
    old_baseline, old_script = runtime.baseline, original.SCRIPT
    runtime.baseline = lambda: (source, saved, effective)
    original.SCRIPT = SCRIPT
    try:
        original.qualify(args)
    finally:
        runtime.baseline, original.SCRIPT = old_baseline, old_script
    path = args.output / "qualification.json"
    spec = original.read(path)
    spec.update(source_saved_wheel_damping=before, baseline_and_candidate_replay_wheel_damping=after,
                frozen_baseline_intervention_does_not_change_source_checkpoint=True,
                comparison_is_PPO_learning_at_same_controller=True,
                numeric_readers_and_all525_cells_unchanged=True)
    for p in [SCRIPT, old_script, *proofs]:
        spec["source_pins"][str(p.resolve())] = sha(p)
    assert all(sha(p) == digest for p, digest in spec["source_pins"].items())
    write_json(path, spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--pointer", type=Path)
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--after-owned-unit")
    args = parser.parse_args()
    if args.qualify_only:
        assert args.recipe and args.pointer and args.source and args.after_owned_unit
        qualify(args)
    else:
        original.run(args.output.resolve())
