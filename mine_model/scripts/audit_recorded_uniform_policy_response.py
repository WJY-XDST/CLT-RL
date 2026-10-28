"""Read-only policy response on every recorded source condition.

Different networks see identical recorded inputs. A local command sensitivity
is a neural-network diagnostic, not a closed-loop gain or acceptance metric.
"""
import argparse
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

import joint_pd_uniform_scalar_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import sha

SCRIPT = Path(__file__).resolve()


def policy(saved, source):
    agent = yaml.safe_load((source.parent / "params/agent.yaml").read_text())
    options = dict(agent["policy"])
    assert options.pop("class_name") == "ActorCritic"
    assert agent["obs_groups"] == {"policy": ["policy"], "critic": ["policy"]}
    dummy = TensorDict({"policy": torch.zeros(1, 27)}, [1])
    with contextlib.redirect_stdout(io.StringIO()):
        result = ActorCritic(dummy, agent["obs_groups"], 6, **options)
    result.load_state_dict(saved["model_state_dict"])
    result.eval()
    return result


def infer(model, values):
    flat = values.reshape(-1, 27)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(flat), 4096):
            batch = torch.from_numpy(flat[start:start + 4096].copy())
            chunks.append(model.act_inference(TensorDict({"policy": batch}, [len(batch)])).numpy())
    return np.concatenate(chunks).reshape(*values.shape[:-1], 6)


def audit(recipe, health_path, replays, output):
    assert not output.exists()
    torch.set_num_threads(1)
    runtime.configure(recipe)
    source, saved, cfg = runtime.baseline()
    health = json.loads(health_path.read_text())
    last = health["snapshots"][-1]
    checkpoint = Path(last["checkpoint_after"])
    candidate, candidate_cfg = runtime.require_source(checkpoint)
    assert sha(checkpoint) == last["checkpoint_after_sha256"]
    assert runtime.saved_updates(checkpoint) == last["actual_updates_before_after"][1]
    source_model, candidate_model = policy(saved, source), policy(candidate, checkpoint)
    rng = torch.random.get_rng_state().clone()
    scale = float(cfg["obs_scales"]["ang_vel"])
    factor = cfg["wheel_radius"] / cfg["wheel_track_width"]
    wheel_scale, limit = cfg["action_scale_vel"], cfg.get("wheel_action_limit", 1.)
    delta = .0025
    groups, pins = [], {str(path): sha(path) for path in
        (SCRIPT, recipe, health_path, source, checkpoint, source.parent / "params/agent.yaml",
         checkpoint.parent / "params/agent.yaml")}
    for directory in replays:
        manifest = json.loads((directory / "manifest.json").read_text())
        complete = json.loads((directory / "completed.json").read_text())
        assert complete["completed"] and manifest["checkpoint_sha256"] == sha(source)
        assert not manifest["sampled_policy_actions"] and not manifest["weights_updated"]
        assert manifest["seconds_per_case"] == 20. and manifest["policy_dt"] == .02
        with np.load(directory / "policy_inputs.npz", allow_pickle=False) as data:
            observations = data["observations"][849:].copy()
            recorded_means = data["means"][849:].copy()
        assert observations.shape == (151, len(manifest["cases"]) * len(manifest["seeds"]), 27)
        assert np.isfinite(observations).all() and np.isfinite(recorded_means).all()
        values = {}
        parity = None
        for label, model in (("source", source_model), ("candidate", candidate_model)):
            mean = infer(model, observations)
            if label == "source":
                parity = float(abs(mean - recorded_means).max())
                assert parity < 1.e-5, parity
            plus, minus = observations.copy(), observations.copy()
            plus[..., 7] += delta * scale
            minus[..., 7] -= delta * scale
            mu_plus, mu_minus = infer(model, plus), infer(model, minus)

            def references(mu):
                wheels = np.clip(mu[..., [2, 5]], -limit, limit) * wheel_scale
                return wheels, factor * (wheels[..., 1] - wheels[..., 0])

            wheels, rolling = references(mean)
            _, proxy_plus = references(mu_plus)
            _, proxy_minus = references(mu_minus)
            values[label] = dict(mean=mean, wheels=wheels, rolling=rolling,
                                  local_gain=(proxy_plus - proxy_minus) / (2. * delta))
        rows = []
        for cid, case in enumerate(manifest["cases"]):
            for sid, seed in enumerate(manifest["seeds"]):
                env = cid * len(manifest["seeds"]) + sid
                profiles = {}
                for label, item in values.items():
                    mu = item["mean"][:, env]
                    proxy = item["rolling"][:, env]
                    profiles[label] = dict(
                        raw_mean_by_action=mu.mean(0).tolist(),
                        leg_mean_outside_unit_envelope_fraction=(abs(mu[:, [0, 1, 3, 4]]) > 1.).mean(0).tolist(),
                        wheel_reference_mean_rad_s=item["wheels"][:, env].mean(0).tolist(),
                        rolling_reference_yaw_proxy_mean_rad_s=float(proxy.mean()),
                        command_minus_reference_proxy_mean_rad_s=float(case["yaw"] - proxy.mean()),
                        local_command_to_rolling_reference_gain_mean=float(item["local_gain"][:, env].mean()),
                        local_command_to_rolling_reference_gain_min=float(item["local_gain"][:, env].min()),
                        local_command_to_rolling_reference_gain_max=float(item["local_gain"][:, env].max()))
                rows.append(dict(case=case, seed=seed, fixed_observation_frames=151, profiles=profiles,
                    normalized_raw_mean_shift_RMS_by_action=np.sqrt(np.mean((
                        values["candidate"]["mean"][:, env] - values["source"]["mean"][:, env]) ** 2, axis=0)).tolist()))
        groups.append(dict(directory=str(directory), trial_count=len(rows), rows=rows,
                           source_mean_recording_parity_max_error=parity))
        pins.update({str(path): sha(path) for path in
            (directory / "manifest.json", directory / "completed.json", directory / "policy_inputs.npz")})
    assert torch.equal(rng, torch.random.get_rng_state())
    for state, model in ((saved, source_model), (candidate, candidate_model)):
        assert all(torch.equal(value, state["model_state_dict"][key]) for key, value in model.state_dict().items())
    assert all(sha(path) == digest for path, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        source_checkpoint_sha256=sha(source), candidate_checkpoint_sha256=sha(checkpoint),
        source_actual_updates=runtime.saved_updates(source), candidate_actual_updates=runtime.saved_updates(checkpoint),
        groups=groups, source_pins=pins, command_perturbation_rad_s=delta,
        identical_recorded_source_states_used_for_both_networks=True,
        all_saved_model_tensors_unchanged=True, diagnostic_RNG_draws_added=0,
        simulator_started=False, backward_or_optimizer_executed=False, training_modified=False,
        case_specific_training_logic_added=False, policy_accepted=False,
        limitations=["Recorded states belong to the source closed loop; candidate closed-loop state distribution can differ.",
            "Wheel rolling yaw is a kinematic proxy, not actual measured body yaw or slip.",
            "Command sensitivity holds every other network input fixed and is not a closed-loop transfer gain.",
            "Symmetric local command probes can slightly exceed a stage boundary and are never executed in physics.",
            "This diagnoses every recorded case, and cannot replace independent frozen physical retention/acceptance."])
    write_json(output, result)
    print(json.dumps(dict(verified=True, trials=sum(g["trial_count"] for g in groups),
                         candidate_actual_updates=result["candidate_actual_updates"], policy_accepted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--source-replays", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.recipe.resolve(), args.health.resolve(), [p.resolve() for p in args.source_replays], args.output.resolve())
