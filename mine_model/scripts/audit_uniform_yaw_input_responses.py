"""Read-only input derivatives on every frozen parent observation.

These are local network sensitivities, never closed-loop gains or a physical
stability prediction. Float64 analysis copies have no parameter gradients.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from audit_recorded_uniform_policy_response import policy
import joint_pd_uniform_scalar_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def run(args):
    assert not args.output.exists()
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    runtime.configure(args.recipe)
    source, before, cfg = runtime.baseline()
    health = json.loads(args.health.read_text())
    last = health["snapshots"][-1]
    checkpoint = Path(last["checkpoint_after"])
    after, _ = runtime.require_source(checkpoint)
    assert sha(checkpoint) == last["checkpoint_after_sha256"]
    models = {}
    for label, saved, path in (("parent", before, source), ("candidate", after, checkpoint)):
        with torch.random.fork_rng(devices=[]):
            model = policy(saved, path)
        assert all(torch.equal(value, saved["model_state_dict"][key]) for key, value in model.state_dict().items())
        models[label] = model.double().requires_grad_(False)
    radius, track, scale = cfg["wheel_radius"], cfg["wheel_track_width"], cfg["action_scale_vel"]
    limit = cfg.get("wheel_action_limit", 1.)
    factor = radius * scale / track
    channels = torch.zeros(27, 4, dtype=torch.float64)
    channels[7, 0] = channels[2, 1] = cfg["obs_scales"]["ang_vel"]
    channels[19, 2] = -.5 * track / radius * cfg["obs_scales"]["dof_vel"]
    channels[20, 2] = -channels[19, 2]
    channels[23, 3], channels[26, 3] = -.5 / factor, .5 / factor
    names = ("command_yaw", "measured_body_yaw", "measured_wheel_rolling_yaw_proxy", "previous_wheel_reference_yaw_proxy")
    delta = 1.e-4
    pins = {str(path): sha(path) for path in (SCRIPT, args.recipe, args.health, source, checkpoint,
        source.parent / "params/env.yaml", source.parent / "params/agent.yaml",
        checkpoint.parent / "params/agent.yaml", Path(policy.__code__.co_filename))}
    groups, max_parity, max_finite_error = [], 0., 0.

    def proxy(model, inputs):
        actions = model.act_inference(TensorDict({"policy": inputs}, [len(inputs)]))
        return actions, factor * (actions[:, 5].clamp(-limit, limit) - actions[:, 2].clamp(-limit, limit))

    for directory in args.source_replays:
        manifest = json.loads((directory / "manifest.json").read_text())
        completed = json.loads((directory / "completed.json").read_text())
        assert completed["completed"] and manifest["checkpoint_sha256"] == sha(source)
        assert not manifest["weights_updated"] and not manifest["sampled_policy_actions"]
        assert manifest["seconds_per_case"] == 20. and manifest["policy_dt"] == .02
        with np.load(directory / "policy_inputs.npz", allow_pickle=False) as archive:
            observations, recorded = archive["observations"][849:].copy(), archive["means"][849:].copy()
        assert observations.shape == (151, len(manifest["cases"]) * len(manifest["seeds"]), 27)
        flat, values = observations.reshape(-1, 27), {}
        for label, model in models.items():
            means, derivatives = [], []
            for start in range(0, len(flat), 1024):
                x = torch.tensor(flat[start:start + 1024], dtype=torch.float64, requires_grad=True)
                mu, z = proxy(model, x)
                gradient, = torch.autograd.grad(z.sum(), x)
                analytical = gradient @ channels
                with torch.no_grad():
                    for column in range(4):
                        _, plus = proxy(model, x.detach() + delta * channels[:, column])
                        _, minus = proxy(model, x.detach() - delta * channels[:, column])
                        finite = (plus - minus) / (2. * delta)
                        max_finite_error = max(max_finite_error, float(abs(finite - analytical[:, column]).max()))
                means.append(mu.detach().numpy())
                derivatives.append(analytical.detach().numpy())
            means = np.concatenate(means).reshape(*observations.shape[:-1], 6)
            values[label] = np.concatenate(derivatives).reshape(*observations.shape[:-1], 4)
            if label == "parent":
                max_parity = max(max_parity, float(abs(means - recorded).max()))
        rows = []
        for case_id, case in enumerate(manifest["cases"]):
            for seed_id, seed in enumerate(manifest["seeds"]):
                index = case_id * len(manifest["seeds"]) + seed_id
                profiles = {label: {name: dict(mean=float(data[:, index, column].mean()),
                    minimum=float(data[:, index, column].min()), maximum=float(data[:, index, column].max()),
                    positive_fraction=float((data[:, index, column] > 0.).mean()))
                    for column, name in enumerate(names)} for label, data in values.items()}
                rows.append(dict(case=case, seed=seed, frames=151, local_input_derivatives=profiles))
        groups.append(dict(directory=str(directory), trials=rows))
        pins.update({str(path): sha(path) for path in (directory / "manifest.json", directory / "completed.json", directory / "policy_inputs.npz")})
    assert max_parity < 1.e-5 and max_finite_error < 1.e-5, (max_parity, max_finite_error)
    assert all(parameter.grad is None for model in models.values() for parameter in model.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(sha(path) == digest for path, digest in pins.items())
    write_json(args.output, dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        parent_actual_updates=runtime.saved_updates(source), candidate_actual_updates=runtime.saved_updates(checkpoint),
        parent_checkpoint_sha256=sha(source), candidate_checkpoint_sha256=sha(checkpoint),
        groups=groups, parent_float64_to_recorded_CUDA_mean_max_error=max_parity,
        input_autograd_to_central_difference_max_error=max_finite_error, physical_input_probe_delta=delta,
        input_channel_directions=channels.tolist(), input_channel_order=names,
        source_pins=pins, CPU_input_autograd_executed=True, model_parameter_gradients_created=False,
        optimizer_constructed=False, optimizer_executed=False, saved_checkpoint_or_Adam_changed=False,
        diagnostic_RNG_draws_added=0, simulator_started=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False,
        limitations=["All other observation channels are held fixed; these derivatives are not closed-loop transfer gains or a stability test.",
            "The wheel quantities are rolling kinematic proxies, not measured body motion or contact forces.",
            "Recorded observations belong to the parent. Candidate closed-loop states may differ.",
            "These float64 analysis copies never feed back into PPO, policy execution or acceptance."]))
    print(json.dumps(dict(verified=True, trials=sum(len(group["trials"]) for group in groups),
        max_forward_error=max_parity, max_finite_difference_error=max_finite_error, optimizer_executed=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--source-replays", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
