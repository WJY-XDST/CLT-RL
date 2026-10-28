"""Prepare fresh capacity and queue one reviewed global-scalar PPO block.

Use an immutable pointer per block. A later ordinary continuation supplies
the reviewed descendant's report/replay paths; it does not reload an anchor
implicitly or promote capacity-only weights.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

import joint_pd_uniform_scalar_runtime as runtime
import check_uniform_scalar_environment_capacity as capacity
from joint_pd_continuation import ROOT, sha
from auto_train_mine import write_json


SCRIPT = Path(__file__).resolve()


def systemd(unit, log, command, *, training=False):
    argv = ["systemd-run", "--user", "--unit=" + unit, "--property=Type=exec",
        "--property=RemainAfterExit=yes", "--property=WorkingDirectory=" + str(ROOT),
        "--property=StandardOutput=append:" + str(log), "--property=StandardError=append:" + str(log),
        "--setenv=PYTHONPATH=" + ":".join(str(ROOT / part) for part in
            ("wheel_legged_isaaclab", "mine_model/scripts", "mine_model/experiments")),
        "--setenv=OMP_NUM_THREADS=1", "--setenv=MKL_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=PYTHONDONTWRITEBYTECODE=1", "--setenv=PYTHONUNBUFFERED=1"]
    if training:
        argv.append("--setenv=CLT_UNIFORM_TRAINING_UNIT=" + unit + ".service")
    argv += list(map(str, command))
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    return dict(unit=unit + ".service", command=argv, stderr=result.stderr,
        launched_at=datetime.now().astimezone().isoformat())


def prepare(args):
    recipe = runtime.configure(args.recipe)
    q = args.recipe.resolve().parent
    previous = q / "owned_block200_launch.json"
    if previous.exists():
        unit = json.loads(previous.read_text())["unit"]
        state = subprocess.run(["systemctl", "--user", "show", unit, "-p", "MainPID"],
            capture_output=True, text=True, check=True).stdout
        assert "MainPID=0\n" in state, "An existing owned controller is live; observe it, never duplicate it"
    qualification = json.loads(args.qualification.read_text())
    assert qualification["qualified"]
    source_report = args.source_final_verification or Path(recipe["source_final_verification"])
    report = json.loads(source_report.read_text())
    source = Path(report["checkpoint"]).resolve()
    saved, cfg = runtime.require_source(source)
    actual = runtime.saved_updates(source)
    assert report["verified"] and report["actual_ppo_updates"] == actual
    assert report["checkpoint_sha256"] == sha(source)
    source_mapping = args.source_mapping_replay or Path(recipe["source_mapping_replay"])
    source_same51 = args.source_same51_replay or Path(recipe["source_same51_replay"])
    for directory in (source_mapping, source_same51):
        manifest, completed = [json.loads((directory / name).read_text()) for name in ("manifest.json", "completed.json")]
        assert manifest["checkpoint_sha256"] == sha(source) and completed["completed"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    results = ROOT / "mine_model/results"
    output = results / ("joint_pd_optimization_" + stamp + "_uniform_scalar_200")
    cap = results / ("joint_pd_capacity_uniform_scalar_" + stamp)
    cap.mkdir()
    runtime.NUM_ENVS = 8192
    command, _, _ = runtime.command(source, cfg, "uniform_scalars_n8192_" + output.name + "_u" + str(actual),
        200, cfg["commands"], output / "STOP")
    files = [SCRIPT, args.recipe.resolve(), args.qualification.resolve(), q / "decision.json",
        source_report.resolve(), source, source.parent / "params/env.yaml", source.parent / "params/agent.yaml"]
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and Path(path).resolve().is_relative_to(ROOT):
            if "mine_model/scripts" in str(path) or "mine_model/experiments" in str(path):
                files.append(Path(path).resolve())
    request = dict(created_at=datetime.now().astimezone().isoformat(), block_id=output.name,
        output=str(cap), production_output=str(output), uniform_recipe=str(args.recipe.resolve()),
        production_runtime=str(runtime.SCRIPT), prototype_command=command,
        reward_overrides=recipe["reward_overrides"], source_checkpoint=str(source),
        source_checkpoint_sha256=sha(source), source_actual_updates=actual, source_all17_Adam_step=actual * 20,
        source_num_envs=cfg["scene"]["num_envs"], source_final_verification=str(source_report.resolve()),
        source_mapping_replay=str(source_mapping.resolve()), source_same51_replay=str(source_same51.resolve()),
        source_is_completely_reviewed_production_checkpoint=True, counts=[8192, 4096], probe_updates=5,
        after_units=[], required_complete_reports=[dict(path=str(args.qualification.resolve()), completion_key="qualified"),
            dict(path=str(q / "decision.json"), completion_key="verified")],
        capacity_weights_may_be_promoted=False, optimizer_reset=False, std_reset=False, actual_LR_reset=False,
        source_pins={str(path): sha(path) for path in set(files)}, policy_accepted=False)
    request_path = cap / "request.json"
    write_json(request_path, request)
    for count in (8192, 4096):
        argv = capacity.numeric.probe_command(request, request_path, count,
            results / ("joint_pd_optimization_capacity_" + cap.name + "_n" + str(count)))
        assert int(argv[argv.index("--target_iteration") + 1]) == actual + 5
        for key, value in recipe["reward_overrides"].items():
            assert "env.rewards." + key + "=" + str(value) in argv
    unit_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    capacity_owner = systemd("clt-uniform-scalar-capacity-" + unit_stamp, cap / "controller.log",
        [ROOT / "run_python.sh", capacity.SCRIPT, "--request", request_path])
    pointer = dict(request=str(request_path), capacity=str(cap), planned_production_output=str(output),
        block_id=output.name, capacity_unit=capacity_owner["unit"], uniform_recipe=str(args.recipe.resolve()),
        production_started=False, source_actual_updates=actual, target_actual_updates=actual + 200)
    pointer_path = q / ("prepared_pointer_" + stamp + ".json")
    write_json(pointer_path, pointer)
    write_json(q / "latest_prepared_pointer.json", dict(path=str(pointer_path), **pointer))
    training = systemd("clt-uniform-scalar-block200-" + unit_stamp, q / ("controller_" + stamp + ".log"),
        [ROOT / "run_python.sh", SCRIPT.with_name("train_uniform_scalar_joint_pd_block200.py"), "--pointer", pointer_path],
        training=True)
    write_json(q / "owned_block200_launch.json", dict(training, pointer=str(pointer_path),
        planned_output=str(output), capacity_unit=capacity_owner["unit"],
        production_only_after_both_real5_checks=True, recipe=str(args.recipe.resolve())))
    write_json(q / ("owned_capacity_" + stamp + ".json"), capacity_owner)
    return dict(pointer=str(pointer_path), **pointer, training_unit=training["unit"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--source-final-verification", type=Path)
    parser.add_argument("--source-mapping-replay", type=Path)
    parser.add_argument("--source-same51-replay", type=Path)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False))
