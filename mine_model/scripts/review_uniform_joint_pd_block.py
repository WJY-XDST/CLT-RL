"""Frozen coverage/retention review of a uniform200-update PPO experiment.

The command grid is evaluation data only. It never enters PPO sampling,
reward, observations, action targets, or the controller.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

import joint_pd_uniform_exploration_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases
import review_historical_source3001_conditional_yaw200 as numeric_readers
import diagnose_motion025_full_height as historical_cases
from audit_historical_yaw_sustained_windows import audit as assess_windows


SCRIPT = Path(__file__).resolve()


def run(output, recipe):
    runtime.configure(recipe)
    plan = json.loads((output / "experiment_plan.json").read_text())
    runtime.NUM_ENVS = plan["native_num_envs"]
    proof = json.loads((output / "final_checkpoint_verification_readonly.json").read_text())
    assert proof["verified"] and proof["actual_ppo_updates"] == plan["target_actual_ppo_updates"]
    checkpoint = Path(proof["checkpoint"])
    assert sha(checkpoint) == proof["checkpoint_sha256"]
    _, cfg = runtime.require_source(checkpoint)
    review = output / "independent_review"
    review.mkdir(exist_ok=False)
    low, high = cfg["commands"]["ranges_height"]
    scale = cfg["commands"]["training_motion_scale"]
    v_max = max(abs(v) for v in cfg["commands"]["ranges_lin_vel_x"]) * scale
    w_max = max(abs(v) for v in cfg["commands"]["ranges_ang_vel_yaw"]) * scale
    neighbours = []
    for height in np.linspace(low + .05 * (high - low), high - .05 * (high - low), 4):
        for speed in (-v_max / 4., 0., v_max / 4.):
            for yaw in (-w_max, -w_max * 2. / 3., w_max * 2. / 3., w_max):
                neighbours.append(dict(name=f"heldout_{len(neighbours):03d}", speed=float(speed), yaw=float(yaw), height=float(height)))
    same = historical_cases.cases()
    scopes = (("stage27", same[:9], [83, 84, 85], 20.),
              ("same51", same, [483, 484, 485], 20.),
              ("heldout144", neighbours, [883, 884, 885], 20.),
              ("sustained_all17_x3", same, [983, 984, 985], 120.))
    reports, window_report = {}, None
    # Validate every label and numeric command before launching any Kit job.
    for _, cases, _, _ in scopes:
        expected = [tuple(case[key] for key in ("name", "speed", "yaw", "height")) for case in cases]
        assert command_probe_cases(cases, SimpleNamespace(**cfg["commands"])) == expected
    for name, cases, seeds, seconds in scopes:
        destination = output / "replay" if name == "stage27" else review / name
        commands = review / f"{name}_commands.json"
        write_json(commands, cases)
        argv = runtime.replay_command(checkpoint, destination)
        argv[argv.index("--seconds") + 1] = str(seconds)
        index = argv.index("--seeds") + 1
        argv[index:index + 3] = list(map(str, seeds))
        argv += ["--command-probe-json", str(commands)]
        write_json(review / f"{name}_launch.json", dict(command=argv, checkpoint_sha256=sha(checkpoint),
            conditions_for_evaluation_only=True, action_or_reward_compensation=False))
        with (review / f"{name}.log").open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        reports[name] = numeric_readers.assess(destination)
        manifest = json.loads((destination / "manifest.json").read_text())
        assert manifest["checkpoint_sha256"] == sha(checkpoint)
        assert manifest["cases"] == cases and manifest["seeds"] == seeds
        assert manifest["seconds_per_case"] == seconds and manifest["reset_velocity_uniform_amplitude"] == .05
        assert not any(manifest.get(key) for key in ("weights_updated", "sampled_policy_actions", "policy_input_remapping",
            "action_bias_probes", "wheel_probe", "wheel_reference_limit_probe"))
        if seconds == 120.:
            _, window_report = assess_windows(destination)
            assert window_report["trial_count"] == 51 and window_report["windows_per_trial"] == 101
        write_json(review / "progress.json", dict(completed_scopes=list(reports),
            summaries={key: report["summary"] for key, report in reports.items()},
            goal_remains_active=True, policy_accepted=False))
    source_report = Path(plan["source_review_same51"]) / "original_and_literal10_conditional_assessment.json"
    baseline = json.loads(source_report.read_text())
    source_manifest = json.loads((source_report.parent / "manifest.json").read_text())
    assert source_manifest["checkpoint_sha256"] == plan["source_sha256"]
    before = {(t["case"]["name"], t["seed"]): t for t in baseline["trials"]}
    after = {(t["case"]["name"], t["seed"]): t for t in reports["same51"]["trials"]}
    assert before.keys() == after.keys()
    pairs = [dict(case=before[key]["case"], seed=key[1], before=before[key], after=after[key]) for key in before]
    result = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        source_actual_updates=plan["source_actual_ppo_updates"], candidate_actual_updates=proof["actual_ppo_updates"],
        source_summary=baseline["summary"], candidate_summaries={key: report["summary"] for key, report in reports.items()},
        same51_all_pairs=pairs, sustained_trial_count=window_report["trial_count"],
        sustained_windows_per_trial=window_report["windows_per_trial"],
        sustained_all_original_and_literal10_passed=window_report["original_AND_literal10_all_windows_passed"],
        original_uniform_reward_and_continuous_training_sampler=True,
        condition_grid_used_for_evaluation_only=True, full_plus_minus08_goal_still_required=True,
        all_models_preserved=True, continuation_requires_data_review=True, policy_accepted=False,
        hardware_enabled=False, shutdown=False, goal_remains_active=True,
        source_pins={str(p): sha(p) for p in (SCRIPT, source_report, checkpoint,
            output / "final_checkpoint_verification_readonly.json")})
    write_json(review / "complete_comparison_agent_decision_required.json", result)
    write_json(output / "status.json", dict(state="complete200_reviewed_agent_decision_required", training_active=False,
        actual_ppo_updates=proof["actual_ppo_updates"], checkpoint=str(checkpoint), goal_remains_active=True,
        policy_accepted=False, independent_review_complete=True))
    print(json.dumps({key: result[key] for key in ("complete", "source_summary", "candidate_summaries", "sustained_windows_per_trial")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.recipe.resolve())
