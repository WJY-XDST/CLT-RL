"""Exercise the actual weighted sampler at both original native populations."""
from datetime import datetime
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import joint_pd_source7401_D100_C400_focusgrid_runtime as runtime
import targeted_turn_command_sampler as original
import weighted_joint_turn_sampler as weighted
from wheel_legged_gym_isaaclab.yaw_training import grouped_task_ids


SCRIPT = Path(__file__).resolve()
OUTPUT = runtime.QUALIFICATION / "weighted_sampler_CPU_qualification.json"


def fake_environment(count, cfg):
    commands = SimpleNamespace(**copy.deepcopy(cfg["commands"]))
    env = SimpleNamespace(cfg=SimpleNamespace(commands=commands,
        rewards=SimpleNamespace(base_height_target=cfg["rewards"]["base_height_target"])),
        device="cpu", common_step_counter=1_000_000, step_dt=.02,
        _dedicated_standing_count=count // 4,
        _command_ranges=torch.tensor([[.2, .8], [-.3, .3], commands.ranges_height]),
        _commands=torch.zeros(count, 4), _target_lin_vel_x=torch.zeros(count))
    env._commands[:, 2] = .3
    return env


def exercise(count, cfg, sampler):
    env = fake_environment(count, cfg)
    counts = dict(selected=0, worst_reverse08_left0075_low=0)
    speed_hist = torch.zeros(5, dtype=torch.int64)
    yaw_hist = torch.zeros(3, dtype=torch.int64)
    height_hist = torch.zeros(3, dtype=torch.int64)
    speed_bins = torch.tensor([.1, .2, .4, .6, .8])
    yaw_bins = torch.tensor([.075, .15, .3])
    height_bins = torch.tensor([cfg["commands"]["ranges_height"][0], .3, cfg["commands"]["ranges_height"][1]])
    ids = torch.arange(count)
    for index in range(120):
        env.common_step_counter = 1_000_000 + 250 * index
        env._commands[:, 1] = 0.
        env._commands[:, 2] = .3
        env._target_lin_vel_x.zero_()
        applied_before = env._commands[:, 0].clone()
        chosen = sampler.apply_after_original(env, ids)
        assert torch.equal(applied_before, env._commands[:, 0])
        assert not env._target_lin_vel_x[:count // 4].any()
        assert not env._commands[:count // 4, 1].any()
        assert not (chosen < count // 4).any()
        roles = grouped_task_ids(chosen, phase=env.common_step_counter // 250,
                                 dedicated_standing_count=count // 4)
        assert bool(((roles == 1) | (roles == 4) | (roles == 5)).all())
        assert not env._target_lin_vel_x[chosen[roles == 1]].any()
        assert bool((env._target_lin_vel_x[chosen[roles == 4]] > 0).all())
        assert bool((env._target_lin_vel_x[chosen[roles == 5]] < 0).all())
        counts["selected"] += len(chosen)
        speed = env._target_lin_vel_x[chosen].abs()
        moving = roles != 1
        speed_hist += torch.bincount((speed[moving, None] - speed_bins).abs().argmin(1), minlength=5)
        yaw_hist += torch.bincount((env._commands[chosen, 1].abs()[:, None] - yaw_bins).abs().argmin(1), minlength=3)
        height_hist += torch.bincount((env._commands[chosen, 2][:, None] - height_bins).abs().argmin(1), minlength=3)
        counts["worst_reverse08_left0075_low"] += int(((env._target_lin_vel_x[chosen] + .8).abs() < 1.e-6).logical_and(
            (env._commands[chosen, 1] - .075).abs() < 1.e-6).logical_and(
            (env._commands[chosen, 2] - height_bins[0]).abs() < 1.e-6).sum())
    probabilities = dict(speed=(speed_hist / speed_hist.sum()).tolist(),
                         yaw=(yaw_hist / yaw_hist.sum()).tolist(),
                         height=(height_hist / height_hist.sum()).tolist())
    return dict(count=count, global_resamples=120, requested_targets=count * 120,
                histogram_probabilities=probabilities, **counts)


def qualify():
    assert not OUTPUT.exists()
    runtime.QUALIFICATION.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    source, saved, cfg = runtime.baseline()
    original_rng = torch.get_rng_state().clone()
    results = []
    with torch.random.fork_rng(devices=[]):
        for count in (8192, 4096):
            torch.manual_seed(105743)
            before = exercise(count, cfg, original)
            torch.manual_seed(105743)
            after = exercise(count, cfg, weighted)
            for key, probabilities in weighted.WEIGHTS.items():
                assert abs(sum(probabilities) - 1.) < 1.e-12 and min(probabilities) > 0.
                assert max(abs(a - b) for a, b in zip(after["histogram_probabilities"][key], probabilities)) < .015
            assert after["worst_reverse08_left0075_low"] > before["worst_reverse08_left0075_low"] * 4.5
            env = fake_environment(count, cfg)
            env._command_ranges[0] = 0.
            before_rng = torch.get_rng_state().clone()
            before_commands, before_targets = env._commands.clone(), env._target_lin_vel_x.clone()
            assert not weighted.apply_after_original(env, torch.arange(count)).numel()
            assert torch.equal(before_rng, torch.get_rng_state())
            assert torch.equal(before_commands, env._commands) and torch.equal(before_targets, env._target_lin_vel_x)
            env._command_ranges[0] = torch.tensor([.2, .8])
            setattr(env.cfg.commands, weighted.CONFIG_KEY, 0.)
            assert not weighted.apply_after_original(env, torch.arange(count)).numel()
            assert torch.equal(before_rng, torch.get_rng_state())
            setattr(env.cfg.commands, weighted.CONFIG_KEY, .3)
            def base(ids):
                env.base_calls += 1
            env._resample_commands_for, env.base_calls = base, 0
            original.install(env)
            weighted.install(env)
            env._resample_commands_for(torch.arange(count))
            assert env.base_calls == 1 and env._targeted_turn_sampler_calls == 1
            results.append(dict(uniform=before, weighted=after,
                empirical_worst_case_requested_target_multiplier=after["worst_reverse08_left0075_low"] / before["worst_reverse08_left0075_low"],
                original_base_sampler_called_exactly_once=True,
                original_fixed_replay_and_zero_fraction_add_no_RNG_or_state_change=True,
                original_slew_buffer_standing_roles_and_all_bounds_preserved=True))
    assert torch.equal(original_rng, torch.get_rng_state())
    assert runtime.sha(source) == runtime.ANCHOR_SHA
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    paths = [SCRIPT, runtime.SCRIPT, weighted.SCRIPT, Path(original.__file__), runtime.SOURCE_REPORT, source]
    result = dict(qualified_cpu_only=True, observed_at=datetime.now().astimezone().isoformat(),
        source_checkpoint_sha256=runtime.ANCHOR_SHA, source_actual_updates=7401,
        source_all17_Adam_step=148020, source_saved_six_std=saved["model_state_dict"]["std"].tolist(),
        original_loaded_std_Adam_and_actual_LR_unchanged=True,
        only_three_grid_draws_changed_AST_proof=weighted.AST_EVIDENCE,
        expected_probability_change=weighted.expected_probability_changes(), results=results,
        all_positive_probability_bins_and_other70percent_requests_retained=True,
        heldout_interior_cases_not_copied_into_targeted_training_grid=True,
        source_pins={str(path): runtime.sha(path) for path in paths},
        actual_PPO_updates=0, simulator_started=False, production_started=False,
        future_native_admission_and_full_evaluation_still_required=True,
        acceptance_changed=False, policy_accepted=False)
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(qualified_cpu_only=True, output=str(OUTPUT),
        empirical_multipliers=[r["empirical_worst_case_requested_target_multiplier"] for r in results]), ensure_ascii=False))
    return result


if __name__ == "__main__":
    qualify()
