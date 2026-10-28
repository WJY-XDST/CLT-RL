"""Frozen wheel-drive sensitivity on every case in a completed v3 replay grid.

The sole physical intervention is one common velocity-drive damping for both
wheels and all commands. This diagnostic cannot train or promote a policy.
Numerical scoring and pair comparison come from the unchanged original readers.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import diagnose_uniform_wheel_drive_transfer as original
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def verify_pins(spec):
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())


def qualify(args):
    import joint_pd_uniform_control_runtime_v3 as runtime
    from compare_uniform_joint_pd_replay_retention import load

    assert not args.output.exists()
    assert math.isfinite(args.damping) and 0. < args.damping <= 1.5
    runtime.configure(args.recipe)
    grid = read(args.baseline / "qualification.json")
    complete_path = args.baseline / "complete_paired_current_stage525.json"
    complete = read(complete_path)
    assert complete["complete"] and not complete["policy_accepted"]
    assert Path(grid["recipe"]).resolve() == args.recipe.resolve()
    batches, pins, sources = [], {}, set()
    for item in grid["batches"]:
        baseline = args.baseline / item["name"]
        _, manifest, cfg, trials, inputs = load(baseline)
        source = Path(manifest["checkpoint"]).resolve()
        saved, source_cfg = runtime.require_source(source)
        assert sha(source) == manifest["checkpoint_sha256"] == complete["candidate_checkpoint_sha256"]
        assert len(saved["optimizer_state_dict"]["state"]) == 17
        assert all(float(s["step"]) == runtime.saved_updates(source) * 20
                   for s in saved["optimizer_state_dict"]["state"].values())
        assert manifest["cases"] == item["cases"] and manifest["seeds"] == item["seeds"]
        assert manifest["seconds_per_case"] == item["seconds"]
        before = cfg["wheel_damping"]
        assert before == cfg["robot"]["actuators"]["wheels"]["damping"] == source_cfg["wheel_damping"]
        assert args.damping != before
        assert cfg["rewards"] == source_cfg["rewards"]
        sources.add((str(source), sha(source), runtime.saved_updates(source), before))
        batches.append(dict(name=item["name"], baseline=str(baseline.resolve()),
                            cases=item["cases"], seeds=item["seeds"], seconds=item["seconds"],
                            trials=len(trials)))
        pins.update(inputs)
    assert len(sources) == 1 and sum(x["trials"] for x in batches) == grid["trials"] == 525
    source_name, source_sha, updates, before = sources.pop()
    source = Path(source_name)
    _, evaluator, ast_proof = original.evaluator_tree(runtime)
    paths = [SCRIPT, Path(original.__file__), runtime.SCRIPT, args.recipe, source,
             args.baseline / "qualification.json", complete_path, evaluator,
             SCRIPT.with_name("compare_uniform_joint_pd_replay_retention.py"),
             SCRIPT.with_name("review_historical_source3001_conditional_yaw200.py")]
    paths += sorted((source.parent / "params").glob("*.yaml"))
    pins.update(runtime.recipe()["source_pins"])
    pins.update({str(p.resolve()): sha(p) for p in paths})
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
                recipe=str(args.recipe.resolve()), source_checkpoint=str(source),
                source_checkpoint_sha256=source_sha, actual_updates=updates,
                baseline_wheel_damping=before, probe_wheel_damping=args.damping,
                batches=batches, trials=525, evaluator_AST_qualification=ast_proof,
                source_pins=pins, all_baseline_cases_and_seeds_retained=True,
                condition_specific_control=False, optimizer_executed=False,
                training_configuration_selected=False, acceptance_changed=False,
                policy_accepted=False)
    verify_pins(spec)
    args.output.mkdir()
    write_json(args.output / "qualification.json", spec)
    print(json.dumps({k: spec[k] for k in ("qualified", "trials", "actual_updates",
                                          "baseline_wheel_damping", "probe_wheel_damping")}))


def replay(request, arguments):
    import torch
    import joint_pd_uniform_control_runtime_v3 as runtime

    spec = read(request)
    verify_pins(spec)
    runtime.configure(spec["recipe"])
    source = Path(arguments[arguments.index("--checkpoint") + 1]).resolve()
    saved, cfg = runtime.require_source(source)
    assert str(source) == spec["source_checkpoint"] and sha(source) == spec["source_checkpoint_sha256"]
    assert runtime.saved_updates(source) == spec["actual_updates"]
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    forbidden = ("--target_iteration", "--sample-policy-actions", "--bias-probe-json",
                 "--wheel-probe", "--wheel-reference-limit-probe", "--export_dir",
                 "--heading-hold", "--reset_optimizer")
    assert not any(x in arguments for x in forbidden)
    damping = spec["probe_wheel_damping"]
    expected_override = {"wheel_damping": damping, "robot.actuators.wheels.damping": damping}
    assert read(arguments[arguments.index("--overrides_json") + 1]) == expected_override
    tree, evaluator, ast_proof = original.evaluator_tree(runtime)
    assert ast_proof == spec["evaluator_AST_qualification"]

    def observe(env, output):
        from pxr import UsdGeom, UsdPhysics
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
        assert raw.cfg.rewards.to_dict() == cfg["rewards"]
        assert raw.cfg.observation_space == 27 and raw.cfg.action_space == 6
        assert raw.cfg.leg_control_mode == "implicit_joint_position"
        assert raw.cfg.wheel_action_limit == cfg.get("wheel_action_limit", 1.) == 1.
        assert raw.cfg.sim.dt == cfg["sim"]["dt"] == .005
        assert raw._tracking_goal_criteria["relative_error"] == .1
        assert Path(raw.cfg.robot.spawn.usd_path).resolve() == runtime.base.tires.ASSET
        for side in ("LW_link", "RW_link"):
            root = "/World/envs/env_0/Robot/" + side
            tire = raw.sim.stage.GetPrimAtPath(root + "/analytic_tire_collision")
            assert tire.IsA(UsdGeom.Cylinder)
            assert UsdPhysics.CollisionAPI(tire).GetCollisionEnabledAttr().Get() is True
        masses = raw._robot.root_physx_view.get_masses().sum(-1)
        assert torch.allclose(masses, torch.full_like(masses, 12.34), atol=1.e-4, rtol=0.)
        chain = runtime.verify_unmodified_execution(raw)
        assert not any(name in sys.modules for name in (
            "historical_yaw_focus_sampler", "historical_yaw_focus_sampler_v2", "historical_yaw_error_focus"))
        assert torch.equal(before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        write_json(Path(output) / "frozen_uniform_wheel_drive_native_proof.json", dict(
            verified=True, source_checkpoint_sha256=sha(source), actual_updates=runtime.saved_updates(source),
            all17_saved_Adam_step=runtime.saved_updates(source) * 20,
            actual_wheel_damping_min=float(actual.min()), actual_wheel_damping_max=float(actual.max()),
            actual_USD_damping=usd, actual_num_envs=raw.num_envs,
            evaluator_AST_qualification=ast_proof, original_reward_and_sampler_call_chain=chain,
            observer_RNG_unchanged=True, gain_equal_for_all_commands_and_both_wheels=True,
            original_reward_values_unchanged=True, source_weights_std_Adam_unchanged=True,
            diagnostic_only_not_training_configuration=True, new_PPO_updates=0, policy_accepted=False))

    def reject_other_overrides(args):
        return bool(args.overrides_json) and read(args.overrides_json) != expected_override

    sys.argv = [str(SCRIPT), *runtime.base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(evaluator), "exec"), dict(__name__="__main__", __file__=str(evaluator),
            _install_quality_criteria=runtime.criteria_fix.install,
            _reject_other_control_overrides=reject_other_overrides, _observe_source200=observe))
    finally:
        verify_pins(spec)


def run(args):
    import joint_pd_uniform_control_runtime_v3 as runtime
    import review_historical_source3001_conditional_yaw200 as readers
    from compare_uniform_joint_pd_replay_retention import summarize_pairs

    request = args.output / "qualification.json"
    spec = read(request)
    verify_pins(spec)
    runtime.configure(spec["recipe"])
    source = Path(spec["source_checkpoint"])
    runtime.require_source(source)
    override = args.output / "declared_coupled_drive_override.json"
    write_json(override, {"wheel_damping": spec["probe_wheel_damping"],
                          "robot.actuators.wheels.damping": spec["probe_wheel_damping"]})
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    comparisons, pins = [], dict(spec["source_pins"])
    for batch in spec["batches"]:
        label = batch["name"]
        destination = args.output / ("probe_" + label)
        commands = args.output / (label + "_commands.json")
        assert not destination.exists()
        write_json(commands, batch["cases"])
        argv = runtime.replay_command(source, destination)
        assert argv[2:4] == ["replay", "--recipe"]
        argv[1:5] = [str(SCRIPT), "replay", "--request", str(request)]
        argv[argv.index("--seconds") + 1] = str(batch["seconds"])
        at = argv.index("--seeds") + 1
        argv[at:at + 3] = list(map(str, batch["seeds"]))
        argv += ["--command-probe-json", str(commands), "--overrides_json", str(override)]
        write_json(args.output / (label + "_launch.json"), dict(command=argv, checkpoint_sha256=sha(source),
            only_change="one global synchronized wheel velocity-drive damping", new_PPO_updates=0))
        write_json(args.output / "status.json", dict(state=label, diagnostic_active=True,
            formal_training_active=False, policy_accepted=False, goal_remains_active=True))
        with (args.output / (label + ".log")).open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        readers.assess(destination)
        result, inputs = original.compare_declared_drive(Path(batch["baseline"]), destination,
            spec["baseline_wheel_damping"], spec["probe_wheel_damping"])
        comparisons.append(result)
        pins.update(inputs)
        proof = destination / "frozen_uniform_wheel_drive_native_proof.json"
        assert read(proof)["verified"]
        pins[str(proof)] = sha(proof)
        write_json(args.output / "progress.json", dict(completed_batches=len(comparisons),
            paired_summary=summarize_pairs([r for c in comparisons for r in c["all_pairs"]]),
            training_modified=False, policy_accepted=False))
    pairs = [r for c in comparisons for r in c["all_pairs"]]
    assert len(pairs) == spec["trials"]
    assert len({(r["case"]["name"], r["seed"]) for r in pairs}) == len(pairs)
    grouped = {axis: [dict(command_value=v, **summarize_pairs([r for r in pairs if r["case"][axis] == v]))
                      for v in sorted({r["case"][axis] for r in pairs})]
               for axis in ("height", "speed", "yaw")}
    assert all(sha(path) == digest for path, digest in pins.items())
    write_json(args.output / "complete_frozen_uniform_wheel_drive525.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), comparisons=comparisons,
        paired525_summary=summarize_pairs(pairs), all525_pairs=pairs, grouped_diagnosis=grouped,
        declared_global_damping=[spec["baseline_wheel_damping"], spec["probe_wheel_damping"]],
        source_checkpoint_sha256=sha(source), source_pins=pins, all_cases_seeds_retained=True,
        original_numeric_readers_unchanged=True, new_PPO_updates=0, training_modified=False,
        training_configuration_selected=False, acceptance_changed=False, policy_accepted=False,
        hardware_enabled=False, shutdown=False, goal_remains_active=True))
    write_json(args.output / "status.json", dict(state="complete_agent_decision_required", diagnostic_active=False,
        formal_training_active=False, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "run", "replay"))
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--request", type=Path)
    args, remaining = parser.parse_known_args()
    if args.mode == "replay":
        assert args.request
        replay(args.request, remaining)
    elif args.mode == "qualify":
        assert not remaining and args.recipe and args.baseline and args.output and args.damping
        qualify(args)
    else:
        assert not remaining and args.output
        run(args)
