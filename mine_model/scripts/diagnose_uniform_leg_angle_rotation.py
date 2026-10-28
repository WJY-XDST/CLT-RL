"""Frozen causal probe: apply the same small motor-angle rotation everywhere.

All commands receive unchanged, minus and plus interventions. This is a
diagnostic experiment, never a policy/controller candidate or acceptance run.
The existing evaluator performs the intervention and computes its telemetry.
"""
import argparse
import ast
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def evaluator_tree(runtime):
    tree, path, saved_restore = runtime.evaluator_tree()
    before = ast.dump(tree)
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "args.command_probe_json"]
    assert len(guards) == 1
    guards = [n for n in guards[0].body if isinstance(n, ast.If)]
    assert len(guards) == 1
    guard = guards[0]
    original = copy.deepcopy(guard.test)

    class PermitFrozenBiasWithCommandData(ast.NodeTransformer):
        count = 0

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "args" and node.attr == "bias_probe_json":
                self.count += 1
                return ast.copy_location(ast.Constant(False), node)
            return node

    change = PermitFrozenBiasWithCommandData()
    guard.test = change.visit(guard.test)
    assert change.count == 1
    modified = guard.test
    guard.test = original
    assert ast.dump(tree) == before
    guard.test = modified
    return ast.fix_missing_locations(tree), path, dict(
        saved_checkpoint_drive_restore=saved_restore,
        restored_original_evaluator_AST_identical=True, changed_permission_attributes=1,
        extension="Existing bounded frozen action-bias diagnosis may use a command-data grid",
        metric_limit_window_survival_controller_and_PPO_AST_unchanged=True,
        acceptance_run=False)


def qualify(args):
    import torch
    import joint_pd_uniform_control_runtime_v3 as runtime
    from joint_pd_continuation import sha
    from auto_train_mine import write_json
    from wheel_legged_gym_isaaclab.action_bias_probe import build_bias_cases
    from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state

    assert not args.output.exists()
    runtime.configure(args.recipe)
    saved, cfg = runtime.require_source(args.checkpoint)
    proof = read(args.final_verification)
    assert proof["verified"] and proof["checkpoint_sha256"] == sha(args.checkpoint)
    assert proof["all200_native_updates_and_original_alignment_verified"]
    assert cfg["wheel_damping"] == cfg["robot"]["actuators"]["wheels"]["damping"] == 1.
    macro = read(args.macro / "qualification.json")
    _, evaluator, permission = evaluator_tree(runtime)
    probes = [dict(name="unchanged", bias=[0.] * 6),
              dict(name="minus", bias=[-.05, -.05, 0., -.05, -.05, 0.]),
              dict(name="plus", bias=[.05, .05, 0., .05, .05, 0.])]
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(v["step"]) == proof["actual_ppo_updates"] * 20
               for v in saved["optimizer_state_dict"]["state"].values())
    nominal = torch.tensor(cfg["leg_joint_position_offset"], dtype=torch.float64).view(1, 4)
    scales = torch.tensor(cfg["leg_joint_position_scale"], dtype=torch.float64)
    l0, t0, _, valid = five_bar_state(nominal[:, [0, 2]], nominal[:, [1, 3]], **cfg["five_bar_geometry"])
    assert valid.all()
    rotations = []
    for probe in probes:
        delta = torch.tensor([probe["bias"][i] for i in (0, 1, 3, 4)], dtype=torch.float64) * scales
        q = nominal + delta
        lengths, angles, _, good = five_bar_state(q[:, [0, 2]], q[:, [1, 3]], **cfg["five_bar_geometry"])
        assert good.all() and torch.allclose(lengths, l0, rtol=0., atol=1.e-12)
        expected = delta[0] * torch.tensor([[1., -1.]], dtype=torch.float64)
        assert torch.allclose(angles - t0, expected, rtol=0., atol=1.e-12)
        rotations.append(dict(probe=probe["name"], normalized_bias=probe["bias"],
            nominal_target_length_change_max_m=float((lengths-l0).abs().max()),
            nominal_target_angle_changes_rad=(angles-t0)[0].tolist()))
    batches = copy.deepcopy(macro["batches"])
    for batch in batches:
        cases = [tuple(c[k] for k in ("name", "speed", "yaw", "height")) for c in batch["cases"]]
        expanded, biases, labels = build_bias_cases(cases, dict(cases=[c[0] for c in cases], probes=probes))
        assert len(expanded) == len(cases) * len(probes)
        assert all(biases[i] == probes[i % len(probes)]["bias"] for i in range(len(biases)))
    paths = [SCRIPT, runtime.SCRIPT, args.recipe, args.checkpoint, args.final_verification,
             args.checkpoint.parent / "params/env.yaml", args.checkpoint.parent / "params/agent.yaml",
             evaluator, args.macro / "qualification.json",
             SCRIPT.with_name("review_historical_source3001_conditional_yaw200.py"),
             SCRIPT.with_name("compare_uniform_joint_pd_replay_retention.py")]
    pins = dict(runtime.recipe()["source_pins"])
    pins.update({str(p.resolve()): sha(p) for p in paths})
    args.output.mkdir()
    for batch in batches:
        commands = args.output / (batch["name"] + "_commands.json")
        biases = args.output / (batch["name"] + "_biases.json")
        write_json(commands, batch["cases"])
        write_json(biases, dict(cases=[c["name"] for c in batch["cases"]], probes=probes))
        pins.update({str(p.resolve()): sha(p) for p in (commands, biases)})
    write_json(args.output / "qualification.json", dict(qualified=True,
        recipe=str(args.recipe.resolve()), checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha(args.checkpoint), actual_updates=proof["actual_ppo_updates"],
        batches=batches, seeds=macro["seeds"], seconds=macro["seconds"],
        probes=probes, nominal_analytic_rotation_check=rotations,
        original_command_seed_trials=macro["trials"], probe_trials=macro["trials"] * len(probes),
        evaluator_permission_qualification=permission,
        prerequisite=str(args.prerequisite.resolve()), after_owned_unit=args.after_owned_unit,
        one_constant_bias_per_probe_shared_by_every_command=True,
        automatic_controller_candidate=False, automatic_training_parameter_selection=False,
        optimizer_executed=False, acceptance_changed=False, policy_accepted=False,
        source_pins=pins))
    print(json.dumps(dict(qualified=True, probe_trials=macro["trials"] * len(probes), rotations=rotations)))


def replay(specification, arguments):
    import torch
    import joint_pd_uniform_control_runtime_v3 as runtime
    from auto_train_mine import write_json
    from joint_pd_continuation import sha

    spec = read(specification)
    assert all(sha(p) == digest for p, digest in spec["source_pins"].items())
    runtime.configure(spec["recipe"])
    source = Path(arguments[arguments.index("--checkpoint") + 1]).resolve()
    assert str(source) == spec["checkpoint"] and sha(source) == spec["checkpoint_sha256"]
    runtime.require_source(source)
    assert not any(flag in arguments for flag in ("--overrides_json", "--export_dir", "--sample-policy-actions",
        "--wheel-reference-limit-probe", "--wheel-probe", "--heading-hold", "--reset_optimizer"))
    commands = Path(arguments[arguments.index("--command-probe-json") + 1])
    biases = Path(arguments[arguments.index("--bias-probe-json") + 1])
    assert str(commands.resolve()) in spec["source_pins"] and str(biases.resolve()) in spec["source_pins"]
    assert read(biases) == dict(cases=[c["name"] for c in read(commands)], probes=spec["probes"])
    tree, path, permission = evaluator_tree(runtime)
    assert permission == spec["evaluator_permission_qualification"]
    original = runtime.private_for_count(runtime.NUM_ENVS, is_replay=True).observe_environment

    def observe(env, output):
        original(env, output)
        raw = env.unwrapped
        actual = raw._robot.root_physx_view.get_dof_dampings()[:, raw._wheel_joint_ids]
        assert torch.allclose(actual, torch.ones_like(actual), rtol=0., atol=1.e-6)
        write_json(Path(output) / "frozen_uniform_leg_rotation_native_proof.json", dict(
            verified=True, checkpoint_sha256=sha(source),
            original_PhysX_drive_and_implicit_joint_position_control_preserved=True,
            action_changes_only_through_existing_bounded_diagnostic_probe=True,
            every_command_receives_the_same_all_four_motor_bias=True,
            evaluator_permission_qualification=permission, optimizer_executed=False,
            source_checkpoint_unchanged=True, policy_acceptance_test=False, policy_accepted=False))

    sys.argv = [str(SCRIPT), *runtime.base.tires.arguments(arguments)]
    exec(compile(tree, str(path), "exec"), dict(__name__="__main__", __file__=str(path),
        _install_quality_criteria=runtime.criteria_fix.install, _observe_source200=observe))


def run(output):
    spec = read(output / "qualification.json")
    while not Path(spec["prerequisite"]).exists():
        owner = subprocess.run(["systemctl", "--user", "show", spec["after_owned_unit"],
            "-p", "MainPID", "-p", "SubState"], capture_output=True, text=True, check=True).stdout
        state = dict(row.split("=", 1) for row in owner.splitlines() if "=" in row)
        if state.get("MainPID") == "0" and state.get("SubState") in ("failed", "dead", "exited"):
            raise RuntimeError("Prerequisite review ended without its receipt; inspect without restarting PPO")
        time.sleep(10.)
    import numpy as np
    import joint_pd_uniform_control_runtime_v3 as runtime
    import review_historical_source3001_conditional_yaw200 as reader
    from compare_uniform_joint_pd_replay_retention import summarize_pairs, transition, NUMERIC_METRICS
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, sha

    runtime.configure(spec["recipe"])
    assert all(sha(p) == digest for p, digest in spec["source_pins"].items())
    checkpoint = Path(spec["checkpoint"])
    runtime.require_source(checkpoint)
    reports, pairs = {}, []
    for batch in spec["batches"]:
        name = batch["name"]
        destination = output / name
        argv = runtime.replay_command(checkpoint, destination)
        argv[1:5] = [str(SCRIPT), "replay", "--specification", str(output / "qualification.json")]
        argv[argv.index("--seconds") + 1] = str(spec["seconds"])
        at = argv.index("--seeds") + 1
        argv[at:at+3] = list(map(str, spec["seeds"]))
        argv += ["--command-probe-json", str(output / (name + "_commands.json")),
                 "--bias-probe-json", str(output / (name + "_biases.json"))]
        write_json(output / (name + "_launch.json"), dict(command=argv, frozen_diagnostic_only=True))
        write_json(output / "status.json", dict(state=name, diagnostic_active=True,
            training_modified=False, policy_accepted=False, goal_remains_active=True))
        with (output / (name + ".log")).open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        assessment = reader.assess(destination)
        manifest = read(destination / "manifest.json")
        assert not manifest["policy_acceptance_test"]
        assert manifest["checkpoint_sha256"] == spec["checkpoint_sha256"]
        labels = manifest["action_bias_probes"]
        assert len(labels) == len(batch["cases"]) * len(spec["probes"])
        z = np.load(destination / "policy_inputs.npz")
        initial = z["observations"][0].reshape(len(batch["cases"]), len(spec["probes"]), len(spec["seeds"]), 27)
        means = z["means"][0].reshape(len(batch["cases"]), len(spec["probes"]), len(spec["seeds"]), 6)
        assert np.array_equal(initial[:, :1].repeat(3, axis=1), initial)
        assert np.array_equal(means[:, :1].repeat(3, axis=1), means)
        indexed = {(r["case"]["name"], r["seed"]): r for r in assessment["trials"]}
        for case in batch["cases"]:
            for seed in spec["seeds"]:
                a = indexed[(case["name"] + "__unchanged", seed)]
                for probe in ("minus", "plus"):
                    b = indexed[(case["name"] + "__" + probe, seed)]
                    av, bv = a["metrics"], b["metrics"]
                    pairs.append(dict(case=case, seed=seed, probe=probe,
                        source_survived=a["survived"], candidate_survived=b["survived"],
                        original_transition=transition(a["original_passed"], b["original_passed"]),
                        strict_transition=transition(a["original_AND_literal10_passed"], b["original_AND_literal10_passed"]),
                        source_failed_metrics=a["flags"]["failed_metrics"], candidate_failed_metrics=b["flags"]["failed_metrics"],
                        source_strict_failed_axes=[k for k,v in a["nonzero_relative_percent"].items() if v is not None and v >= 10.],
                        candidate_strict_failed_axes=[k for k,v in b["nonzero_relative_percent"].items() if v is not None and v >= 10.],
                        source_metrics=av, candidate_metrics=bv,
                        metric_deltas={k:bv[k]-av[k] for k in NUMERIC_METRICS} if av and bv else None,
                        source_yaw_signed_bias_rad_s=a["yaw_signed_bias_rad_s"],
                        candidate_yaw_signed_bias_rad_s=b["yaw_signed_bias_rad_s"]))
        reports[name] = assessment["summary"]
        write_json(output / "progress.json", dict(completed=list(reports), summaries=reports,
            paired_by_probe={p:summarize_pairs([r for r in pairs if r["probe"] == p]) for p in ("minus", "plus")},
            initial_observations_and_raw_actor_means_bitwise_equal_between_probes=True,
            optimizer_executed=False, policy_accepted=False))
    assert len(pairs) == 2 * spec["original_command_seed_trials"]
    assert all(sha(p) == digest for p, digest in spec["source_pins"].items())
    write_json(output / "complete_frozen_uniform_leg_rotation945.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), all_pairs=pairs,
        paired_by_probe={p:summarize_pairs([r for r in pairs if r["probe"] == p]) for p in ("minus", "plus")},
        grouped_by_probe_and_height={p:[dict(height=h, **summarize_pairs([r for r in pairs if r["probe"] == p and r["case"]["height"] == h]))
            for h in sorted({r["case"]["height"] for r in pairs})] for p in ("minus", "plus")},
        source_pins=spec["source_pins"], initial_observations_and_raw_actor_means_bitwise_equal=True,
        fixed_bias_is_not_a_deployment_or_training_change=True, no_automatic_parameter_selection=True,
        acceptance_changed=False, optimizer_executed=False, policy_accepted=False, goal_remains_active=True))
    write_json(output / "status.json", dict(state="complete_agent_diagnosis_required", diagnostic_active=False,
        policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    if mode == "replay":
        parser.add_argument("--specification", type=Path, required=True)
        args, remaining = parser.parse_known_args(sys.argv[1:])
        replay(args.specification, remaining)
    else:
        parser.add_argument("--output", type=Path, required=True)
        if mode == "qualify":
            parser.add_argument("--recipe", type=Path, required=True)
            parser.add_argument("--checkpoint", type=Path, required=True)
            parser.add_argument("--final-verification", type=Path, required=True)
            parser.add_argument("--macro", type=Path, required=True)
            parser.add_argument("--prerequisite", type=Path, required=True)
            parser.add_argument("--after-owned-unit", required=True)
            qualify(parser.parse_args())
        elif mode == "run":
            run(parser.parse_args().output)
        else:
            raise ValueError(mode)
