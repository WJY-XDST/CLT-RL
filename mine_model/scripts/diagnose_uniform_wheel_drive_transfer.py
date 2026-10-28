"""Frozen, globally uniform wheel-drive comparison on a complete replay grid.

Only the two representations of the same wheel velocity-drive damping change.
No PPO entry point, conditional controller, command sampler, or reward change
exists here. Every recorded baseline case and seed must be retained.
"""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def evaluator_tree(runtime):
    """Permit precisely the declared diagnostic override, preserving all math."""
    tree, original = runtime.retained.evaluator_tree()
    before = ast.dump(tree)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.unparse(node.test) == "args.command_probe_json"]
    assert len(matches) == 1
    guards = [node for node in matches[0].body if isinstance(node, ast.If)]
    assert len(guards) == 1
    guard = guards[0]
    saved_test = copy.deepcopy(guard.test)

    class DeclaredDriveGuard(ast.NodeTransformer):
        count = 0

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "args" and node.attr == "overrides_json":
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(
                    id="_reject_other_control_overrides", ctx=ast.Load()),
                    args=[ast.Name(id="args", ctx=ast.Load())], keywords=[]), node)
            return node

    change = DeclaredDriveGuard()
    guard.test = change.visit(guard.test)
    assert change.count == 1
    modified_test = guard.test
    guard.test = saved_test
    assert ast.dump(tree) == before
    guard.test = modified_test
    proof = dict(original_evaluator_AST_restored_identical=True, changed_guard_nodes=1,
        permission_extension_only="Exactly the declared, equal top-level and actuator wheel damping",
        original_metric_threshold_window_failure_and_execution_AST_unchanged=True,
        original_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        diagnostic_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest())
    return ast.fix_missing_locations(tree), original, proof


def replay(request_path, arguments):
    import torch
    import joint_pd_uniform_scalar_runtime as runtime
    from auto_train_mine import write_json
    from joint_pd_continuation import sha

    spec = read(request_path)
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    runtime.configure(spec["parent_recipe"])
    source = Path(arguments[arguments.index("--checkpoint") + 1]).resolve()
    saved, cfg = runtime.require_source(source)
    assert str(source) == spec["parent_checkpoint"] and sha(source) == spec["parent_checkpoint_sha256"]
    assert runtime.saved_updates(source) == spec["parent_actual_updates"]
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(value["step"]) == spec["parent_actual_updates"] * 20
               for value in saved["optimizer_state_dict"]["state"].values())
    forbidden = ("--target_iteration", "--sample-policy-actions", "--bias-probe-json", "--wheel-probe",
        "--wheel-reference-limit-probe", "--export_dir", "--heading-hold", "--reset_optimizer")
    assert not any(flag in arguments for flag in forbidden)
    override = read(arguments[arguments.index("--overrides_json") + 1])
    damping = spec["probe_wheel_damping"]
    expected_override = {"wheel_damping": damping, "robot.actuators.wheels.damping": damping}
    assert override == expected_override
    tree, original, ast_proof = evaluator_tree(runtime)
    assert ast_proof == spec["evaluator_AST_qualification"]
    original_observer = runtime.private_for_count(runtime.NUM_ENVS, is_replay=True).observe_environment

    def observe(env, output):
        from pxr import UsdPhysics
        original_observer(env, output)
        raw = env.unwrapped
        before, cuda_before = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        assert raw.cfg.wheel_damping == raw.cfg.robot.actuators["wheels"].damping == damping
        actual = raw._robot.root_physx_view.get_dof_dampings()[:, raw._wheel_joint_ids]
        assert torch.allclose(actual, torch.full_like(actual, damping), rtol=0., atol=1.e-6)
        usd = []
        for joint in raw.cfg.wheel_joint_names:
            prim = raw.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/" + joint)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            assert drive.GetTypeAttr().Get() == "force"
            usd.append(drive.GetDampingAttr().Get())
            assert abs(usd[-1] - damping) < 1.e-6
            assert abs(drive.GetMaxForceAttr().Get() - cfg["wheel_effort_limit"]) < 1.e-5
        for key in ("leg_control_mode", "leg_joint_stiffness", "leg_joint_damping", "leg_effort_limit",
                    "wheel_control_mode", "wheel_effort_limit", "action_scale_vel", "feedforward_force",
                    "feedforward_mode", "feedforward_angle_bias", "feedforward_force_limit",
                    "enable_external_forces_every_iteration", "decimation"):
            assert getattr(raw.cfg, key) == cfg[key], key
        assert raw.cfg.wheel_action_limit == cfg.get("wheel_action_limit", 1.) == 1.
        assert raw.cfg.sim.dt == cfg["sim"]["dt"] == .005
        assert torch.equal(before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        write_json(Path(output) / "frozen_uniform_wheel_drive_native_proof.json", dict(
            verified=True, source_checkpoint_sha256=sha(source), source_actual_updates=runtime.saved_updates(source),
            source_all17_Adam_step=spec["parent_actual_updates"] * 20,
            actual_wheel_damping_min=float(actual.min()), actual_wheel_damping_max=float(actual.max()),
            actual_USD_damping=usd, actual_num_envs=raw.num_envs,
            evaluator_AST_qualification=ast_proof, observer_RNG_unchanged=True,
            global_drive_scalar_for_all_commands=True, no_PPO_updates=True,
            weights_std_and_saved_optimizer_unchanged=True, policy_accepted=False))

    def reject_other_overrides(args):
        return bool(args.overrides_json) and read(args.overrides_json) != expected_override

    sys.argv = [str(SCRIPT), *runtime.base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(original), "exec"), dict(__name__="__main__", __file__=str(original),
            _install_quality_criteria=runtime.criteria_fix.install,
            _reject_other_control_overrides=reject_other_overrides, _observe_source200=observe))
    finally:
        assert all(sha(path) == digest for path, digest in spec["source_pins"].items())


def qualify(args):
    import joint_pd_uniform_scalar_runtime as runtime
    from auto_train_mine import write_json
    from joint_pd_continuation import sha
    from compare_uniform_joint_pd_replay_retention import load

    assert not args.output.exists() and math.isfinite(args.damping) and 0. < args.damping <= 1.5
    original = read(args.baseline / "qualification.json")
    assert read(args.baseline / "complete_paired_height_yaw_factorial315.json")["complete"]
    runtime.configure(original["parent_recipe"])
    source = Path(original["parent_checkpoint"])
    saved, cfg = runtime.require_source(source)
    before = cfg["wheel_damping"]
    assert before == cfg["robot"]["actuators"]["wheels"]["damping"] and args.damping != before
    _, evaluator_path, ast_proof = evaluator_tree(runtime)
    pins = dict(runtime.recipe()["source_pins"])
    for batch in original["batches"]:
        directory = args.baseline / ("parent_" + batch["name"])
        report, manifest, effective_cfg, _, inputs = load(directory)
        assert manifest["checkpoint_sha256"] == sha(source)
        assert manifest["cases"] == batch["cases"] and manifest["seeds"] == original["seeds"]
        assert effective_cfg["wheel_damping"] == effective_cfg["robot"]["actuators"]["wheels"]["damping"] == before
        pins.update(inputs)
    paths = [SCRIPT, runtime.SCRIPT, Path(original["parent_recipe"]), source,
        source.parent / "params/env.yaml", source.parent / "params/agent.yaml", evaluator_path,
        args.baseline / "qualification.json", args.baseline / "complete_paired_height_yaw_factorial315.json",
        SCRIPT.with_name("compare_uniform_joint_pd_replay_retention.py"),
        SCRIPT.with_name("review_historical_source3001_conditional_yaw200.py")]
    pins.update({str(path.resolve()): sha(path) for path in paths})
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        baseline=str(args.baseline.resolve()), parent_recipe=original["parent_recipe"],
        parent_checkpoint=str(source), parent_checkpoint_sha256=sha(source),
        parent_actual_updates=runtime.saved_updates(source), baseline_wheel_damping=before,
        probe_wheel_damping=args.damping, batches=original["batches"], seeds=original["seeds"],
        seconds=original["seconds"], trials=original["trials"],
        evaluator_AST_qualification=ast_proof, source_pins=pins,
        condition_specific_controller=False, command_grid_enters_training=False,
        optimizer_executed=False, acceptance_changed=False, policy_accepted=False)
    assert saved["model_state_dict"]["std"].shape == (6,)
    args.output.mkdir()
    write_json(args.output / "qualification.json", spec)
    print(json.dumps({key: spec[key] for key in ("qualified", "trials", "baseline_wheel_damping", "probe_wheel_damping")}))


def compare_declared_drive(source, candidate, before_damping, after_damping):
    from compare_uniform_joint_pd_replay_retention import load, transition, summarize_pairs, NUMERIC_METRICS
    a, am, ac, old, ap = load(source)
    b, bm, bc, new, bp = load(candidate)
    assert am == bm, "Frozen checkpoint, all commands and evaluation settings must be identical"
    assert old.keys() == new.keys()
    expected = copy.deepcopy(ac)
    assert expected["wheel_damping"] == expected["robot"]["actuators"]["wheels"]["damping"] == before_damping
    expected["wheel_damping"] = expected["robot"]["actuators"]["wheels"]["damping"] = after_damping
    expected["log_dir"] = bc["log_dir"]
    assert bc == expected, "Reject every undeclared configuration change"
    pairs = []
    for key, before in old.items():
        after = new[key]
        assert before["case"] == after["case"]
        # Leg/wheel reference limits are recomputed from each physical trace by
        # the same pinned reader. Command-based and fixed limits remain equal.
        for field in ("speed_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "leg_angle_mae_rad", "body_p95_deg"):
            assert before["limits"][field] == after["limits"][field]
        av, bv = before["metrics"], after["metrics"]
        pairs.append(dict(case=before["case"], seed=key[1],
            source_survived=before["survived"], candidate_survived=after["survived"],
            original_transition=transition(before["original_passed"], after["original_passed"]),
            strict_transition=transition(before["original_AND_literal10_passed"], after["original_AND_literal10_passed"]),
            source_failed_metrics=before["flags"]["failed_metrics"], candidate_failed_metrics=after["flags"]["failed_metrics"],
            source_strict_failed_axes=[axis for axis, value in before["nonzero_relative_percent"].items() if value is not None and value >= 10.],
            candidate_strict_failed_axes=[axis for axis, value in after["nonzero_relative_percent"].items() if value is not None and value >= 10.],
            source_metrics=av, candidate_metrics=bv,
            metric_deltas={field: bv[field] - av[field] for field in NUMERIC_METRICS} if av and bv else None,
            source_nonzero_relative_percent=before["nonzero_relative_percent"],
            candidate_nonzero_relative_percent=after["nonzero_relative_percent"],
            source_yaw_signed_bias_rad_s=before["yaw_signed_bias_rad_s"],
            candidate_yaw_signed_bias_rad_s=after["yaw_signed_bias_rad_s"]))
    return dict(source_summary=a["summary"], probe_summary=b["summary"],
        paired_summary=summarize_pairs(pairs), all_pairs=pairs), ap | bp


def run(args):
    import joint_pd_uniform_scalar_runtime as runtime
    import review_historical_source3001_conditional_yaw200 as readers
    from compare_uniform_joint_pd_replay_retention import summarize_pairs
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, sha

    spec_path = args.output / "qualification.json"
    spec = read(spec_path)
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    runtime.configure(spec["parent_recipe"])
    source = Path(spec["parent_checkpoint"])
    runtime.require_source(source)
    overrides = args.output / "declared_coupled_drive_override.json"
    write_json(overrides, {"wheel_damping": spec["probe_wheel_damping"],
        "robot.actuators.wheels.damping": spec["probe_wheel_damping"]})
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    reports, comparisons, pins = {}, [], dict(spec["source_pins"])
    for batch in spec["batches"]:
        label = batch["name"]
        baseline = Path(spec["baseline"]) / ("parent_" + label)
        destination = args.output / ("probe_" + label)
        commands = args.output / (label + "_commands.json")
        write_json(commands, batch["cases"])
        argv = runtime.replay_command(source, destination)
        assert argv[2:4] == ["replay", "--recipe"]
        argv[1:5] = [str(SCRIPT), "replay", "--request", str(spec_path)]
        argv[argv.index("--seconds") + 1] = str(spec["seconds"])
        at = argv.index("--seeds") + 1
        argv[at:at + 3] = list(map(str, spec["seeds"]))
        argv += ["--command-probe-json", str(commands), "--overrides_json", str(overrides)]
        write_json(args.output / (label + "_launch.json"), dict(command=argv,
            checkpoint_sha256=sha(source), only_change="global synchronized wheel damping", new_PPO_updates=0))
        write_json(args.output / "status.json", dict(state=label, diagnostic_active=True,
            formal_training_active=False, policy_accepted=False, goal_remains_active=True))
        with (args.output / (label + ".log")).open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        reports[label] = readers.assess(destination)["summary"]
        comparison, inputs = compare_declared_drive(baseline, destination,
            spec["baseline_wheel_damping"], spec["probe_wheel_damping"])
        comparisons.append(comparison)
        pins.update(inputs)
        native_proof = destination / "frozen_uniform_wheel_drive_native_proof.json"
        assert read(native_proof)["verified"]
        pins[str(native_proof)] = sha(native_proof)
        write_json(args.output / "progress.json", dict(completed_scopes=list(reports), summaries=reports,
            paired_summary=summarize_pairs([row for c in comparisons for row in c["all_pairs"]]),
            training_modified=False, policy_accepted=False))
    pairs = [row for comparison in comparisons for row in comparison["all_pairs"]]
    assert len(pairs) == spec["trials"] and len({(row["case"]["name"], row["seed"]) for row in pairs}) == len(pairs)
    grouped = {}
    for axis in ("height", "speed", "yaw"):
        values = sorted({row["case"][axis] for row in pairs})
        grouped[axis] = [dict(command_value=value,
            **summarize_pairs([row for row in pairs if row["case"][axis] == value])) for value in values]
    assert all(sha(path) == digest for path, digest in pins.items())
    write_json(args.output / "complete_frozen_uniform_wheel_drive315.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(),
        source_checkpoint_sha256=sha(source), comparisons=comparisons, paired315_summary=summarize_pairs(pairs),
        grouped_diagnosis=grouped, declared_global_damping=[spec["baseline_wheel_damping"], spec["probe_wheel_damping"]],
        source_pins=pins, all_commands_and_seeds_retained=True, original_thresholds_and_relative10_retained=True,
        condition_specific_training_logic=False, optimizer_executed=False, new_PPO_updates=0,
        source_NN_std_and_17Adam_unchanged=True, training_configuration_not_selected_automatically=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False, goal_remains_active=True))
    write_json(args.output / "status.json", dict(state="complete_agent_decision_required", diagnostic_active=False,
        formal_training_active=False, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "run", "replay"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--request", type=Path)
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay_parser = argparse.ArgumentParser(description=__doc__)
        replay_parser.add_argument("mode", choices=("replay",))
        replay_parser.add_argument("--request", type=Path, required=True)
        args, remaining = replay_parser.parse_known_args(sys.argv[1:])
        replay(args.request, remaining)
    else:
        args = parser.parse_args()
        if args.mode == "qualify":
            assert args.baseline and args.output and args.damping
            qualify(args)
        else:
            assert args.output
            run(args)
