"""Compare every matched frozen replay, including formerly passing trials.

Command values are diagnostic labels only. This CPU-only reader does not load
policies, select training commands, alter rewards or start a simulator.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics

from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
REPORT = "original_and_literal10_conditional_assessment.json"
NUMERIC_METRICS = (
    "speed_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "leg_length_mae_m",
    "leg_angle_mae_rad", "wheel_speed_mae_rad_s", "body_p95_deg",
)
CHECKPOINT_FIELDS = {"checkpoint", "checkpoint_sha256", "checkpoint_iteration"}


def load(directory):
    directory = Path(directory).resolve()
    paths = [directory / name for name in (REPORT, "manifest.json", "completed.json", "resolved_env.json", "trace.csv")]
    assessment, manifest, complete, cfg = [json.loads(path.read_text()) for path in paths[:4]]
    assert assessment["verified"] and complete["completed"]
    assert not assessment["acceptance_changed"] and not assessment["policy_accepted"]
    assert not manifest["weights_updated"] and not manifest["sampled_policy_actions"]
    assert manifest["leg_control_mode"] == "implicit_joint_position"
    assert manifest["policy_dt"] == .02
    for path, digest in assessment["source_pins"].items():
        assert sha(path) == digest, ("assessment input changed", path)
    trials = {(row["case"]["name"], row["seed"]): row for row in assessment["trials"]}
    assert len(trials) == len(manifest["cases"]) * len(manifest["seeds"]) == assessment["summary"]["trials"]
    for row in trials.values():
        relative = row["nonzero_relative_percent"]
        strict = row["original_passed"] and all(value < 10. for value in relative.values() if value is not None)
        assert strict == row["original_AND_literal10_passed"]
        if row["metrics"] is not None:
            for axis, field in (("speed", "speed_mae_m_s"), ("yaw", "yaw_mae_rad_s"), ("height", "height_mae_m")):
                expected = 100. * row["metrics"][field] / abs(row["case"][axis]) if row["case"][axis] else None
                assert expected == relative[axis]
    return assessment, manifest, cfg, trials, {str(path): sha(path) for path in paths}


def transition(before, after):
    return "retained_pass" if before and after else "new_failure" if before else "recovered" if after else "still_failed"


def summarize_pairs(pairs):
    metrics = {}
    for field in NUMERIC_METRICS:
        usable = [row for row in pairs if row["source_metrics"] is not None and row["candidate_metrics"] is not None]
        if usable:
            changes = [row["metric_deltas"][field] for row in usable]
            metrics[field] = dict(
                source_mean=statistics.mean(row["source_metrics"][field] for row in usable),
                candidate_mean=statistics.mean(row["candidate_metrics"][field] for row in usable),
                mean_delta=statistics.mean(changes),
                improved=sum(delta < -1.e-12 for delta in changes),
                worsened=sum(delta > 1.e-12 for delta in changes),
                unchanged=sum(abs(delta) <= 1.e-12 for delta in changes),
            )
    return dict(trials=len(pairs),
        original_transitions=dict(Counter(row["original_transition"] for row in pairs)),
        strict_transitions=dict(Counter(row["strict_transition"] for row in pairs)),
        source_survivors=sum(row["source_survived"] for row in pairs),
        candidate_survivors=sum(row["candidate_survived"] for row in pairs),
        source_failed_metrics=dict(Counter(metric for row in pairs for metric in row["source_failed_metrics"])),
        candidate_failed_metrics=dict(Counter(metric for row in pairs for metric in row["candidate_failed_metrics"])),
        source_strict_failed_axes=dict(Counter(axis for row in pairs for axis in row["source_strict_failed_axes"])),
        candidate_strict_failed_axes=dict(Counter(axis for row in pairs for axis in row["candidate_strict_failed_axes"])),
        numeric_metrics=metrics)


def compare(source, candidate, scope):
    a, am, ac, old, ap = load(source)
    b, bm, bc, new, bp = load(candidate)
    assert old.keys() == new.keys(), "Cases and seeds must match exactly"
    manifest_changes = {key: [am.get(key), bm.get(key)] for key in am.keys() | bm.keys() if am.get(key) != bm.get(key)}
    assert set(manifest_changes) <= CHECKPOINT_FIELDS, ("Evaluation conditions changed", manifest_changes)
    # Reward/exploration settings may differ between trained policies; physical
    # evaluation configuration and observation/control semantics must not.
    cfg_changes = {key: [ac.get(key), bc.get(key)] for key in ac.keys() | bc.keys() if ac.get(key) != bc.get(key)}
    allowed_cfg_changes = {"rewards", "log_dir", "min_action_std", "min_leg_action_std", "min_wheel_action_std"}
    assert set(cfg_changes) <= allowed_cfg_changes, ("Undeclared evaluation configuration changes", list(cfg_changes))
    pairs = []
    for key, before in old.items():
        after = new[key]
        assert before["case"] == after["case"]
        source_metrics, candidate_metrics = before["metrics"], after["metrics"]
        deltas = ({field: candidate_metrics[field] - source_metrics[field] for field in NUMERIC_METRICS}
                  if source_metrics is not None and candidate_metrics is not None else None)
        pairs.append(dict(case=before["case"], seed=key[1],
            source_survived=before["survived"], candidate_survived=after["survived"],
            original_transition=transition(before["original_passed"], after["original_passed"]),
            strict_transition=transition(before["original_AND_literal10_passed"], after["original_AND_literal10_passed"]),
            source_failed_metrics=before["flags"]["failed_metrics"], candidate_failed_metrics=after["flags"]["failed_metrics"],
            source_failed_prerequisites=before["flags"]["failed_prerequisites"],
            candidate_failed_prerequisites=after["flags"]["failed_prerequisites"],
            source_strict_failed_axes=[axis for axis, value in before["nonzero_relative_percent"].items() if value is not None and value >= 10.],
            candidate_strict_failed_axes=[axis for axis, value in after["nonzero_relative_percent"].items() if value is not None and value >= 10.],
            source_metrics=source_metrics, candidate_metrics=candidate_metrics, metric_deltas=deltas,
            source_nonzero_relative_percent=before["nonzero_relative_percent"],
            candidate_nonzero_relative_percent=after["nonzero_relative_percent"],
            source_yaw_signed_bias_rad_s=before["yaw_signed_bias_rad_s"], candidate_yaw_signed_bias_rad_s=after["yaw_signed_bias_rad_s"],
            source_yaw_temporal_std_rad_s=before["yaw_standard_deviation_rad_s"],
            candidate_yaw_temporal_std_rad_s=after["yaw_standard_deviation_rad_s"]))
    grouped = {}
    for axis in ("height", "speed", "yaw"):
        groups = {}
        for row in pairs:
            groups.setdefault(row["case"][axis], []).append(row)
        grouped[axis] = [dict(command_value=value, **summarize_pairs(group)) for value, group in sorted(groups.items())]
    condition_groups = {}
    for row in pairs:
        condition_groups.setdefault(tuple(row["case"][axis] for axis in ("height", "speed", "yaw")), []).append(row)
    conditions = [dict(command=dict(zip(("height", "speed", "yaw"), command)), **summarize_pairs(group))
                  for command, group in sorted(condition_groups.items())]
    return dict(scope=scope, source_directory=str(Path(source).resolve()), candidate_directory=str(Path(candidate).resolve()),
        source_checkpoint_sha256=am["checkpoint_sha256"], candidate_checkpoint_sha256=bm["checkpoint_sha256"],
        source_summary=a["summary"], candidate_summary=b["summary"],
        paired_summary=summarize_pairs(pairs), grouped_diagnosis=grouped, all_conditions=conditions, all_pairs=pairs,
        declared_nonphysical_config_changes=cfg_changes), ap | bp


def main(args):
    assert len(args.source) == len(args.candidate) == len(args.scope)
    assert len(set(args.scope)) == len(args.scope)
    assert not args.output.exists()
    comparisons, pins = [], {str(SCRIPT): sha(SCRIPT)}
    for source, candidate, scope in zip(args.source, args.candidate, args.scope):
        comparison, inputs = compare(source, candidate, scope)
        comparisons.append(comparison)
        pins.update(inputs)
    assert all(sha(path) == value for path, value in pins.items())
    write_json(args.output, dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        comparisons=comparisons, source_pins=pins, every_requested_case_and_seed_retained=True,
        condition_values_used_for_diagnosis_only=True, original_and_literal10_results_separate=True,
        original_passing_cases_explicitly_checked_for_regression=True, acceptance_changed=False,
        optimizer_executed=False, simulation_started=False, training_modified=False, policy_accepted=False,
        limitations=["These matched cases establish results only over their recorded command coverage.",
            "A per-condition association does not prove a cause or justify a condition-specific training branch.",
            "Sustained independent replay and final full velocity coverage remain separate requirements."]))
    print(json.dumps(dict(verified=True, output=str(args.output), scopes=[
        dict(scope=item["scope"], source=item["source_summary"], candidate=item["candidate_summary"],
             original_transitions=item["paired_summary"]["original_transitions"],
             strict_transitions=item["paired_summary"]["strict_transitions"]) for item in comparisons])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--scope", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
