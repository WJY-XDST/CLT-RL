"""Keep all original11 readers at50/100/200; no training callback or update."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import joint_pd_uniform_scalar_runtime as runtime
import monitor_AF7001_yawsigma0025_learning as original
import plain27_sampler_metadata_adapter as adapter
from auto_train_mine import write_json
from joint_pd_continuation import ROOT


SCRIPT = Path(__file__).resolve()


def run(output, recipe):
    runtime.configure(recipe)
    plan = json.loads((output / "experiment_plan.json").read_text())
    runtime.NUM_ENVS = count = plan["native_num_envs"]
    api = adapter.private_module(original.SCRIPT, "_uniform_original11_monitor")
    api.SCRIPT = SCRIPT
    api.runtime = SimpleNamespace(profile=runtime.profile, DELTA=plan["next_configuration_changes"])
    checks = []
    for name, stem, options in original.CHECKS:
        if name == "compare_parallel8192_recorded_joint_targets.py":
            name = "compare_uniform_scalar_recorded_joint_targets.py"
        elif name == "audit_parallel8192_count_bound.py":
            name = "audit_historical_source3001_count_bound.py"
        elif name == "audit_parallel8192_saved_action_limits_v2.py" and count == 4096:
            name = "audit_clipped_joint_pd_policy_kl_saved_limits.py"
        checks.append((name, stem, options))
    assert len(checks) == 11
    api.CHECKS = tuple(checks)
    evidence = adapter.retarget(api, ("tick",), integers={"*": {8192: count, 2048: count // 4}},
        strings={"learning_health_AF7001_yawsigma0025_prefix_": "learning_health_uniform_prefix_"})
    while True:
        record = api.tick(output, "owned_uniform_controller")
        for prefix in record["completed_prefixes"]:
            extra = output / f"sampled_yaw_reference_components_prefix_{prefix:04d}_readonly.json"
            if not extra.exists():
                command = [sys.executable, str(SCRIPT.with_name("audit_historical_source3001_sampled_yaw_components.py")),
                    str(output), "--through", str(prefix)]
                with (output / f"extra_yaw_components_{prefix:04d}.log").open("a") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(output / "uniform_learning_monitor_status.json", dict(record,
            observed_at=datetime.now().astimezone().isoformat(), original11_readers_unchanged=True,
            count_metadata_adapter=evidence, training_modified=False))
        if record["state"] == "all_owned_prefix_diagnostics_completed":
            return
        time.sleep(15.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.recipe.resolve())
    except BaseException as error:
        write_json(args.output / "uniform_learning_inspection_required.json", dict(
            error=repr(error), training_modified=False, policy_accepted=False))
        raise
