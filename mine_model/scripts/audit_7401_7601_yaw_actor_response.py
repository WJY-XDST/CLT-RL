"""Read-only actor input derivatives at the original steady turning states.

No PPO, optimizer step, weight edit, controller edit or simulation is performed.
Input counterfactuals hold the other 26 observations fixed; they are local
network responses and must not be presented as closed-loop motion predictions.
"""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import audit_7401_7601_yaw_undertracking_chain as chain


OUTPUT = chain.Q / "all648_steady_actor_yaw_command_feedback_derivatives_readonly_20261024.json"


def audit():
    assert not OUTPUT.exists()
    torch.set_num_threads(1)
    torch_rng = torch.random.get_rng_state().clone()
    proof = chain.read(chain.OUTPUT)
    assert proof["verified"] and all(chain.sha(p) == h for p, h in proof["source_pins"].items())
    results, pins = {}, {str(Path(__file__).resolve()): chain.sha(__file__),
                         str(chain.OUTPUT): chain.sha(chain.OUTPUT)}
    maximum_forward_error, maximum_derivative_error = 0., 0.
    k = .06 * 15. / .4199
    for tag, (directory, raw_path) in chain.SOURCES.items():
        raw = chain.read(raw_path)
        checkpoint = Path(raw["checkpoint"])
        assert chain.sha(checkpoint) == raw["checkpoint_sha256"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert len(saved["optimizer_state_dict"]["state"]) == 17
        expected_step = 148020 if tag == "source7401" else 152020
        assert all(float(s["step"]) == expected_step for s in saved["optimizer_state_dict"]["state"].values())
        with torch.random.fork_rng(devices=[]):
            actor = nn.Sequential(nn.Linear(27, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                                  nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 6))
        actor.load_state_dict({key.removeprefix("actor."): value for key, value in saved["model_state_dict"].items()
                               if key.startswith("actor.")}, strict=True)
        actor.eval().requires_grad_(False)
        results[tag] = []
        for height in ("low", "nominal", "high"):
            folder = directory / "combined/candidate" / height
            manifest = chain.read(folder / "manifest.json")
            with np.load(folder / "policy_inputs.npz", allow_pickle=False) as archive:
                observations, recorded = archive["observations"], archive["means"]
            for cid, case in enumerate(manifest["cases"]):
                # All36 cases per height, three seeds and all151 original
                # 17..20s inclusive steady frames are retained in this audit.
                values = observations[849:, cid * 3:cid * 3 + 3].reshape(-1, 27).copy()
                means = recorded[849:, cid * 3:cid * 3 + 3].reshape(-1, 6)
                assert values.shape == (453, 27)
                state = torch.from_numpy(values).requires_grad_()
                output = actor(state)
                forward_error = float(abs(output.detach() - torch.from_numpy(means)).max())
                maximum_forward_error = max(maximum_forward_error, forward_error)
                proxy = k * (output[:, 5] - output[:, 2])
                derivative = torch.autograd.grad(proxy.sum(), state)[0].detach()
                command_slope = derivative[:, 7] * .25
                measured_slope = derivative[:, 2] * .25
                finite = []
                with torch.no_grad():
                    for column, analytic in ((7, command_slope), (2, measured_slope)):
                        plus, minus = state.detach().clone(), state.detach().clone()
                        plus[:, column] += .005 * .25
                        minus[:, column] -= .005 * .25
                        a, b = actor(plus), actor(minus)
                        numerical = k * ((a[:, 5] - a[:, 2]) - (b[:, 5] - b[:, 2])) / .01
                        maximum_derivative_error = max(maximum_derivative_error,
                                                       float(abs(numerical - analytic).max()))
                        finite.append(float(numerical.mean()))
                results[tag].append(dict(case=case, frames_and_seeds=453,
                    native_CPU_actor_mean_max_error=forward_error,
                    command_yaw_physical_derivative_mean=float(command_slope.mean()),
                    measured_body_yaw_physical_derivative_mean=float(measured_slope.mean()),
                    command_yaw_finite_difference_derivative_mean=finite[0],
                    measured_body_yaw_finite_difference_derivative_mean=finite[1],
                    previous_wheel_action_difference_mode_gain_mean=float(((derivative[:, 26] - derivative[:, 23]) / (2. * k)).mean()),
                    command_response_positive_fraction=float((command_slope > 0.).float().mean()),
                    measured_yaw_response_negative_fraction=float((measured_slope < 0.).float().mean())))
        assert all(parameter.grad is None for parameter in actor.parameters())
        pins[str(checkpoint)] = raw["checkpoint_sha256"]
    assert maximum_forward_error <= 5.e-6, maximum_forward_error
    assert maximum_derivative_error <= .005, maximum_derivative_error
    assert torch.equal(torch_rng, torch.random.get_rng_state())
    assert all(chain.sha(p) == h for p, h in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        original_combined_trials_per_checkpoint=324, actor_states_per_checkpoint=48924,
        CPU_actor_to_original_CUDA_six_means_max_error=maximum_forward_error,
        input_derivative_to_central_difference_max_error=maximum_derivative_error,
        finite_difference_physical_yaw_delta_rad_s=.005,
        original27_256_128_64_6_ELU_actor_and_all17_Adam_unchanged=True,
        CPU_Torch_RNG_unchanged=True, results=results, source_pins=pins,
        limitation="Only local responses with all other observations held fixed. Neither causality of historical training nor closed-loop gain or stability follows from these derivatives alone.",
        actor_parameter_gradients_created=False, optimizer_executed=False,
        simulator_started=False, training_modified=False, parameter_change_selected=False,
        acceptance_changed=False, policy_accepted=False)
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    critical = {tag: [r for r in rows if abs(r["case"]["speed"]) == .8 and r["case"]["yaw"] == .075]
                for tag, rows in results.items()}
    print(json.dumps(dict(report=str(OUTPUT), maximum_forward_error=maximum_forward_error,
                         maximum_derivative_error=maximum_derivative_error,
                         critical_positive_yaw=critical), ensure_ascii=False))
    return result


if __name__ == "__main__":
    audit()
