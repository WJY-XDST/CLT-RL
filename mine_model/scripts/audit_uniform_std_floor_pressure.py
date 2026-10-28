"""Read saved exploration floors and Adam moments at an original health prefix.

Floors come from the immutable uniform recipe profile. No model parameter,
optimizer, sampling distribution, action, reward or acceptance gate is changed.
Historical moment direction does not predict the next minibatch gradient.
"""
import argparse
import ast
import contextlib
from datetime import datetime
import inspect
import io
import json
from pathlib import Path
import textwrap

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates


SCRIPT = Path(__file__).resolve()


def audit(directory, through):
    directory = directory.resolve()
    assert directory.parent == ROOT / "mine_model/results" and through in (50, 100, 200)
    destination = directory / f"uniform_std_floor_pressure_prefix_{through:04d}_readonly.json"
    assert not destination.exists()
    plan_path = directory / "experiment_plan.json"
    health_path = directory / f"policy_update_health_prefix_{through:04d}_readonly.json"
    plan, health = [json.loads(path.read_text()) for path in (plan_path, health_path)]
    assert plan["ordinary_full_policy_PPO"] and not plan["auxiliary_losses_enabled"]
    assert plan["training_config"]["leg_control_mode"] == "implicit_joint_position"
    assert plan["training_config"]["observation_space"] == 27
    assert health["committed_updates"] == len(health["snapshots"]) == through
    assert health["diagnostic_RNG_states_unchanged"] and not health["optimizer_executed"]
    assert all(sha(path) == digest for path, digest in health["source_pins"].items())
    tree = ast.parse(textwrap.dedent(inspect.getsource(PPO.__init__)))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "self.optimizer" for target in node.targets)]
    expected = ast.parse("optim.Adam(self.policy.parameters(), lr=learning_rate)", mode="eval").body
    assert len(assignments) == 1 and ast.dump(assignments[0].value) == ast.dump(expected)
    source = Path(plan["source_checkpoint"])
    assert sha(source) == plan["source_sha256"]
    agent_path = source.parent / "params/agent.yaml"
    agent = yaml.safe_load(agent_path.read_text())
    options = dict(agent["policy"])
    assert options.pop("class_name") == "ActorCritic" and options["noise_std_type"] == "scalar"
    state = torch.load(source, map_location="cpu", weights_only=True)
    rng = torch.random.get_rng_state().clone()
    observations = TensorDict({"policy": torch.zeros(1, 27)}, batch_size=[1])
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        model = ActorCritic(observations, agent["obs_groups"], 6, **options)
        model.load_state_dict(state["model_state_dict"], strict=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    parameters = list(model.named_parameters())
    names = [name for name, _ in parameters]
    assert len(parameters) == 17 and names.count("std") == 1
    profile = plan["profile"]
    floors = np.full(6, max(profile["global_minimum_std"], profile["minimum_leg_std"]))
    floors[[2, 5]] = max(profile["global_minimum_std"], profile["minimum_wheel_std"])
    assert np.isfinite(floors).all() and np.all(floors > 0.)
    pins = {str(path): sha(path) for path in [SCRIPT, plan_path, health_path, source, agent_path,
        Path(inspect.getsourcefile(PPO)), Path(inspect.getsourcefile(ActorCritic))]}
    rows, arithmetic_error = [], 0.
    for index, snapshot in enumerate(health["snapshots"], 1):
        assert snapshot["committed_update"] == index
        checkpoint = Path(snapshot["checkpoint_after"])
        assert sha(checkpoint) == snapshot["checkpoint_after_sha256"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actual = saved_updates(checkpoint)
        optimizer = saved["optimizer_state_dict"]
        assert actual == plan["source_actual_ppo_updates"] + index
        assert len(optimizer["param_groups"]) == 1 and len(optimizer["state"]) == 17
        group = optimizer["param_groups"][0]
        ids = group["params"]
        assert len(ids) == 17 and not group["weight_decay"]
        assert not group.get("amsgrad", False) and not group.get("maximize", False)
        for pid, (name, parameter) in zip(ids, parameters):
            moment = optimizer["state"][pid]
            assert float(moment["step"]) == actual * 20
            assert moment["exp_avg"].shape == moment["exp_avg_sq"].shape == parameter.shape
            assert saved["model_state_dict"][name].shape == parameter.shape
            assert torch.isfinite(moment["exp_avg"]).all() and torch.isfinite(moment["exp_avg_sq"]).all()
        moment = optimizer["state"][ids[names.index("std")]]
        m, v = [moment[key].double().numpy() for key in ("exp_avg", "exp_avg_sq")]
        sigma = saved["model_state_dict"]["std"].double().numpy()
        beta1, beta2 = group["betas"]
        t = float(moment["step"])
        direction = -group["lr"] * (m / (1. - beta1 ** t)) / (
            np.sqrt(v / (1. - beta2 ** t)) + group["eps"])
        alternative = -(group["lr"] / (1. - beta1 ** t)) * m / (
            np.sqrt(v) / np.sqrt(1. - beta2 ** t) + group["eps"])
        arithmetic_error = max(arithmetic_error, float(abs(direction - alternative).max()))
        assert np.isfinite(direction).all() and np.all(sigma >= floors - 1.e-8)
        binding = abs(sigma - floors) <= 1.e-8
        rows.append(dict(relative_update=index, actual_updates=actual, all17_Adam_step=actual * 20,
            saved_std=sigma.tolist(), floor_binding=binding.tolist(),
            historical_momentum_direction_at_saved_LR=direction.tolist(),
            floor_binding_and_downward_momentum=(binding & (direction < 0.)).tolist(),
            saved_actual_LR=group["lr"]))
        pins[str(checkpoint)] = snapshot["checkpoint_after_sha256"]
    assert arithmetic_error < 1.e-12 and torch.equal(rng, torch.random.get_rng_state())
    groups = {}
    for name, selected in [("entire_prefix", rows), ("first20", rows[:20]), ("last20", rows[-20:])]:
        groups[name] = dict(checkpoints=len(selected),
            floor_binding_fraction_by_head=np.mean([r["floor_binding"] for r in selected], axis=0).tolist(),
            floor_binding_and_downward_momentum_fraction_by_head=np.mean(
                [r["floor_binding_and_downward_momentum"] for r in selected], axis=0).tolist())
    assert all(sha(path) == digest for path, digest in pins.items())
    report = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(), through=through,
        source_actual_updates=plan["source_actual_ppo_updates"], actual_num_envs=plan["native_num_envs"],
        original_parameter_order=names, std_parameter_id_verified_from_installed_PPO=True,
        floors_from_immutable_profile=floors.tolist(), head_order=["LB", "LL", "LW", "RB", "RL", "RW"],
        groups=groups, snapshots=rows, equivalent_Adam_arithmetic_max_error=arithmetic_error,
        source_pins=pins, CPU_RNG_state_unchanged=True, optimizer_constructed=False,
        optimizer_executed=False, simulator_started=False, training_modified=False,
        parameter_change_selected=False, local_optimum_proven=False, policy_accepted=False,
        limitations=["Saved momentum summarizes past gradients, not the next minibatch or an exact blocked optimizer update.",
            "Binding at a lower std bound is not proof of a local optimum or of benefit from lowering the bound.",
            "Policy quality requires independent frozen replay, including all retained passing and failed cases."])
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return destination, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--through", type=int, choices=(50, 100, 200), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    destination, report = audit(args.directory, args.through)
    print(json.dumps(dict(verified=True, report=str(destination), groups=report["groups"],
                         training_modified=False, parameter_change_selected=False)))
