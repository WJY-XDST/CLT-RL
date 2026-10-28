"""CPU-only compatibility and distribution checks for a provisional sampler.

Passing this file does not admit a native trial, select a source, change a
saved configuration, or establish trained tracking improvement.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

import targeted_turn_command_sampler as candidate

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
ORIGINAL = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/env_commands.py'
_spec = importlib.util.spec_from_file_location('_qualified_original_command_sampler', ORIGINAL)
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)


def run(directory):
    directory = Path(directory).resolve()
    destination = directory / 'provisional_targeted_turn_sampler_CPU_qualification_20261021.json'
    if directory.parent != ROOT / 'mine_model/results' or destination.exists():
        raise ValueError('Use a fresh owned diagnostic output')
    plan_path = directory / 'experiment_plan.json'
    probability_path = directory / 'actual_sampler_joint_probability_crosscheck_readonly_20261021.json'
    plan, baseline = [json.loads(p.read_text()) for p in (plan_path, probability_path)]
    cfg = plan['training_config']['commands']
    if (not plan['ordinary_full_policy_PPO'] or plan['native_num_envs'] != 4096
            or cfg['training_motion_scale'] != 1. or plan['fixed_standing_environments'] != 1024
            or not baseline['closed_form_and_original_sampler_within5_binomial_standard_deviations']):
        raise ValueError('Require the original full-envelope sampler and its independent probability check')
    torch.set_num_threads(1)
    ids = torch.arange(4096)

    def fake(fraction, fixed=False):
        commands = SimpleNamespace(**copy.deepcopy(cfg))
        setattr(commands, candidate.CONFIG_KEY, fraction)
        obj = SimpleNamespace(device='cpu', step_dt=.02, common_step_counter=316848,
            _dedicated_standing_count=1024,
            cfg=SimpleNamespace(commands=commands, rewards=SimpleNamespace(
                base_height_target=plan['training_config']['rewards']['base_height_target'])),
            _command_ranges=torch.tensor([[0., 0.], [0., 0.], [.3, .3]]) if fixed else
                torch.tensor([cfg['ranges_lin_vel_x'], cfg['ranges_ang_vel_yaw'], cfg['ranges_height']]),
            _commands=torch.zeros(4096, 4), _target_lin_vel_x=torch.zeros(4096))
        obj._commands[1024:, 0] = torch.linspace(-.15, .15, 3072)
        obj._commands[:, 3] = 1.234
        obj._resample_commands_for = MethodType(lambda self, selected: original.resample_commands(self, selected), obj)
        return obj

    rng = torch.random.get_rng_state().clone()
    compatibility = []
    partial = torch.tensor([0, 17, 1023, 1024, 1025, 1030, 1371, 2047, 3100, 4095])
    with torch.random.fork_rng(devices=[]):
        for label, fraction, fixed, selected in (
                ('zero_fraction_all', 0., False, ids),
                ('zero_fraction_partial', 0., False, partial),
                ('fixed_replay_nonzero_fraction', .3, True, ids)):
            old, new = fake(fraction, fixed), fake(fraction, fixed)
            torch.manual_seed(9321)
            old._resample_commands_for(selected)
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(9321)
            candidate.install(new)
            bound = new._resample_commands_for
            candidate.install(new)
            if new._resample_commands_for is not bound:
                raise ValueError('Installation must be idempotent')
            new._resample_commands_for(selected)
            if (not torch.equal(old._commands, new._commands)
                    or not torch.equal(old._target_lin_vel_x, new._target_lin_vel_x)
                    or not torch.equal(expected_rng, torch.random.get_rng_state())
                    or new._targeted_turn_sampler_selected):
                raise ValueError('Original commands or RNG changed in a bypass mode')
            compatibility.append(dict(case=label, commands_bitwise_equal=True, requested_speed_bitwise_equal=True,
                                      CPU_RNG_bitwise_equal=True, idempotent_install=True))
        obj = fake(1.)
        torch.manual_seed(8181)
        obj._resample_commands_for(partial)
        before, targets = obj._commands.clone(), obj._target_lin_vel_x.clone()
        chosen = candidate.apply_after_original(obj, partial)
        unchanged = torch.ones(4096, dtype=torch.bool)
        unchanged[chosen] = False
        if (not torch.equal(before[unchanged], obj._commands[unchanged])
                or not torch.equal(targets[unchanged], obj._target_lin_vel_x[unchanged])
                or not torch.equal(before[:, 0], obj._commands[:, 0])
                or not torch.equal(before[:, 3], obj._commands[:, 3])
                or torch.any(chosen < 1024)):
            raise ValueError('Partial resampling changed unselected, standing, applied-speed or heading values')
        roles = candidate.grouped_task_ids(chosen, phase=obj.common_step_counter // 250,
                                          dedicated_standing_count=1024)
        if not chosen.numel() or not torch.all((roles == 1) | (roles == 4) | (roles == 5)):
            raise ValueError('Only declared spin/turn global IDs may receive grid targets')
        speeds = obj._target_lin_vel_x[chosen]
        if (torch.any(speeds[roles == 1]) or torch.any(speeds[roles == 4] <= 0.)
                or torch.any(speeds[roles == 5] >= 0.)):
            raise ValueError('Original role direction or stationary spin was changed')
        obj = fake(.3)
        candidate.install(obj)
        counts = [0, 0, 0]
        batches = 1000
        retained_continuous_turn_samples = 0
        torch.manual_seed(67213)
        for batch in range(batches):
            obj.common_step_counter = 316848 + batch * 250
            applied_before = obj._commands[:, 0].clone()
            obj._resample_commands_for(ids)
            v, y, h = obj._target_lin_vel_x, obj._commands[:, 1], obj._commands[:, 2]
            if (not torch.equal(applied_before, obj._commands[:, 0]) or torch.any(v[:1024])
                    or torch.any(y[:1024]) or v.abs().max() > .8 + 1.e-6
                    or y.abs().max() > .3 + 1.e-6
                    or h.min() < cfg['ranges_height'][0] - 1.e-6
                    or h.max() > cfg['ranges_height'][1] + 1.e-6):
                raise ValueError('Original standing, slew-input or full command bounds changed')
            for index, row in enumerate(baseline['rows']):
                case = row['case']
                counts[index] += int(((abs(v - case['speed']) <= .025)
                    & (abs(y - case['yaw']) <= .0125)
                    & (abs(h - case['height']) <= 1.e-6)).sum())
            roles = candidate.grouped_task_ids(ids, phase=obj.common_step_counter // 250,
                                              dedicated_standing_count=1024)
            turning = (roles == 4) | (roles == 5)
            interior = (abs(h - cfg['ranges_height'][0]) > 1.e-6) & (abs(h - cfg['ranges_height'][1]) > 1.e-6) & (abs(h - .3) > 1.e-6)
            retained_continuous_turn_samples += int((turning & interior).sum())
        if not retained_continuous_turn_samples:
            raise ValueError('Continuous non-grid training support disappeared')
        samples = 4096 * batches
        rows = []
        for row, observed in zip(baseline['rows'], counts):
            old_probability = row['closed_form_joint_probability_per_full_population_draw']
            # Reverse-turn role(.15), one of5 speeds, one of6 signed yaw bins,
            # one of3 heights, with a declared30% replacement probability.
            probability = .7 * old_probability + .3 * .15 / (5 * 6 * 3)
            expected = samples * probability
            z_score = (observed - expected) / math.sqrt(samples * probability * (1 - probability))
            if abs(z_score) >= 5:
                raise ValueError('Joint target frequency disagrees with the independent mixture probability')
            rows.append(dict(case=row['case'], original_probability=old_probability,
                provisional_probability=probability, expected_density_multiplier=probability / old_probability,
                expected_count=expected, observed_count=observed, binomial_z_score=z_score))
    if not torch.equal(rng, torch.random.get_rng_state()):
        raise ValueError('Diagnostic CPU RNG was not restored')
    files = [SCRIPT, Path(candidate.__file__), ORIGINAL,
        Path(candidate.grouped_task_ids.__code__.co_filename), plan_path, probability_path]
    report = dict(observed_at=datetime.now().astimezone().isoformat(), qualified_cpu_only=True,
        provisional_fraction=.3, source_reference_actual_updates=6601, actual_PPO_updates_executed=0,
        compatibility=compatibility, partial_resets_preserve_global_ID_roles_and_unselected_rows=True,
        original_applied_speed_and_heading_buffers_unchanged=True, original_fixed_standing_population=1024,
        original_motion_population=3072, original_full_velocity_range_m_s=[-.8, .8],
        original_full_yaw_range_rad_s=[-.3, .3], original_full_height_range_m=cfg['ranges_height'],
        diagnostic_draws=samples, targeted_selected_count=obj._targeted_turn_sampler_selected,
        retained_continuous_turn_samples=retained_continuous_turn_samples, joint_probabilities=rows,
        native_frozen_transfer_qualified=False, native_training_entry_qualified=False,
        sampler_installed_in_active_native_jobs=False, next_trial_selected=False,
        CPU_diagnostic_RNG_restored=True, native_training_RNG_accessed=False,
        source_pins={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        simulator_started=False, policy_or_optimizer_loaded=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False,
        limitations=['Pure tensor sampler checks only; this is not native transfer, learning or acceptance evidence.',
            'Descriptive neighborhoods are not acceptance thresholds. Unselected turns retain the original continuous sampler.',
            'Any native use requires complete source reviews, explicit trial selection, retained-state load qualification and frozen equivalence checks.'])
    with destination.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, required=True)
    args = parser.parse_args()
    path, result = run(args.training_output)
    print(json.dumps(dict(output=str(path), qualified_cpu_only=True,
        joint_probabilities=result['joint_probabilities'], native_training_entry_qualified=False)))
