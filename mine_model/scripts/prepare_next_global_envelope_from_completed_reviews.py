"""Prescreen the next global ceiling from a fully reviewed immutable candidate.

This prepares/runs only frozen evaluation. It requires full native200 proof,
original four reviews, current paired189, all525 retention and a bitwise CPU
checkpoint/Adam restore. It never selects or starts another PPO block.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
import probe_uniform_macro_motion_envelope as macro
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def wait_retention(folder, unit):
    marker = folder / 'complete_paired_current_stage525.json'
    while not marker.exists():
        assert unit, 'An incomplete retention replay needs its current owner'
        text = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'SubState'],
                              capture_output=True, text=True, check=True).stdout
        state = dict(line.split('=', 1) for line in text.splitlines() if '=' in line)
        assert state['MainPID'] != '0', ('Inspect existing retention without repeating PPO', state)
        time.sleep(15.)
    return marker


def prepare(args, retention_marker):
    runtime.configure(args.recipe)
    final_path = args.training / 'final_checkpoint_verification_readonly.json'
    four_path = args.training / 'independent_review/complete_comparison_agent_decision_required.json'
    wider_path = args.training / 'independent_review/complete_stage_review_with_global_envelope189.json'
    proof, four, wider, grid, archive, paired = [macro.read(path) for path in
        (final_path, four_path, wider_path, retention_marker, args.preservation, args.paired_envelope)]
    assert proof['verified'] and proof['all200_native_updates_and_original_alignment_verified']
    assert four['complete'] and wider['complete'] and grid['complete'] and paired['verified']
    assert archive['actual_CPU_runner_load_optimizer_true_full_state_bitwise_verified']
    checkpoint = Path(proof['checkpoint']).resolve()
    digest = macro.sha(checkpoint)
    assert digest == proof['checkpoint_sha256'] == archive['checkpoint_sha256']
    assert digest == grid['candidate_checkpoint_sha256'] == paired['candidate_checkpoint_sha256']
    assert proof['all17_Adam_step'] == proof['actual_ppo_updates'] * 20
    assert four['candidate_actual_updates'] == grid['candidate_actual_updates'] == proof['actual_ppo_updates']
    assert not archive['optimizer_step_or_backward_executed'] and not grid['acceptance_changed']
    for document in (archive, paired, grid):
        assert all(macro.sha(path) == value for path, value in document['source_pins'].items())
    summary = grid['paired525_summary']
    assert summary['trials'] == summary['source_survivors'] == summary['candidate_survivors'] == 525
    # Preserve the established posture/height/other original physical constraints.
    # A diagnostic prescreen still does not declare yaw/literal10 acceptance.
    assert set(summary['candidate_failed_metrics']) <= {'yaw_mae_rad_s'}
    assert all(not row['candidate_failed_prerequisites'] for row in grid['all525_pairs'])
    _, cfg = runtime.require_source(checkpoint)
    previous = cfg['commands']['training_motion_scale']
    assert type(args.next_scale) is float and math.isfinite(args.next_scale)
    assert 0. < previous < args.next_scale <= 1.
    heights = sorted({row['case']['height'] for row in grid['all525_pairs']})
    assert len(heights) == 7 and [heights[0], heights[-1]] == list(cfg['commands']['ranges_height'])
    speed = max(abs(v) for v in cfg['commands']['ranges_lin_vel_x']) * args.next_scale
    yaw = max(abs(v) for v in cfg['commands']['ranges_ang_vel_yaw']) * args.next_scale
    roles = (('stand', 0., 0.), ('forward', speed, 0.), ('reverse', -speed, 0.),
             ('spin_left', 0., yaw), ('spin_right', 0., -yaw),
             ('forward_left', speed, yaw), ('forward_right', speed, -yaw),
             ('reverse_left', -speed, yaw), ('reverse_right', -speed, -yaw))
    cases = [dict(name=f'envelope_h{i}_{name}', height=h, speed=v, yaw=w)
             for i, h in enumerate(heights) for name, v, w in roles]
    batches = [cases[i:i + 35] for i in range(0, len(cases), 35)]
    assert len(cases) == 63 and len(batches) == 2
    for batch in batches:
        actual = command_probe_cases(batch, SimpleNamespace(**cfg['commands']))
        assert actual == [tuple(row[key] for key in ('name', 'speed', 'yaw', 'height')) for row in batch]
    assert not args.output.exists(), 'Do not overwrite a previous prescreen'
    args.output.mkdir(parents=True)
    commands = args.output / 'commands.json'
    macro.write(commands, cases)
    jobs = []
    for i, batch in enumerate(batches):
        name = f'batch_{i}'
        path = args.output / (name + '_commands.json')
        macro.write(path, batch)
        destination = args.output / name
        argv = runtime.replay_command(checkpoint, destination)
        argv[argv.index('--seconds') + 1] = '20.0'
        start = argv.index('--seeds') + 1
        argv[start:start + 3] = ['883', '884', '885']
        argv += ['--command-probe-json', str(path)]
        jobs.append(dict(name=name, replay=str(destination), command=argv,
                         command_file=str(path), case_count=len(batch)))
    pins = [SCRIPT, Path(macro.__file__), args.recipe, checkpoint, final_path, four_path,
            wider_path, retention_marker, args.preservation, args.paired_envelope,
            checkpoint.parent / 'params/env.yaml', checkpoint.parent / 'params/agent.yaml',
            commands, *[Path(job['command_file']) for job in jobs]]
    value = dict(qualified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint), checkpoint_sha256=digest,
        source_actual_updates=proof['actual_ppo_updates'], recipe=str(args.recipe),
        batches=jobs, previous_training_motion_scale=previous,
        diagnostic_next_motion_scale=args.next_scale, speed_limit_m_s=speed, yaw_limit_rad_s=yaw,
        heights=heights, height_range_m=cfg['commands']['ranges_height'],
        case_count=63, trial_count=189, command_file=str(commands), seconds_per_case=20.,
        prior_physical_minimum_nonzero_yaw_rate=cfg['commands']['yaw_min_abs_rate'] * previous,
        proposed_future_global_raw_minimum_yaw_rate=cfg['commands']['yaw_min_abs_rate'] * previous / args.next_scale,
        only_global_ceiling_is_probed=True, frozen_probe_only=True, cases_enter_training=False,
        native_training_commands_unchanged=True, saved_optimizer_and_policy_not_modified=True,
        original_four_and_current525_and_paired189_required=True,
        saved_Gaussian_replay_is_not_a_deterministic_gate_requirement=True,
        source_pins={str(path): macro.sha(path) for path in pins},
        policy_accepted=False, next_PPO_selected=False)
    macro.write(args.output / 'qualification.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'retention', 'recipe', 'preservation', 'paired-envelope', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--next-scale', type=float, required=True)
    parser.add_argument('--after-owned-unit')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    for name in ('training', 'retention', 'recipe', 'preservation', 'paired_envelope', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    marker = wait_retention(args.retention, args.after_owned_unit)
    result = prepare(args, marker)
    print(json.dumps(dict(qualified=True, source_actual_updates=result['source_actual_updates'],
        speed_limit_m_s=result['speed_limit_m_s'], yaw_limit_rad_s=result['yaw_limit_rad_s'],
        trial_count=189, next_PPO_selected=False)), flush=True)
    if args.run:
        macro.run(args.output)
