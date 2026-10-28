"""Independently reconstruct the two recorded tracking bonuses on every row.

Absolute and command-relative kernels have distinct contracts. A verified
record here establishes reward arithmetic, never policy acceptance or a
causal learning improvement. No simulator, policy or optimizer is created.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from audit_frozen_yaw_reward_sensitivity import quality_from_row
from command_relative_tracking_kernel import command_relative_variance, validate_parameters


SCRIPT = Path(__file__).resolve()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(replays, output):
    assert not output.exists(), 'Preserve completed reward checks'
    pins = {str(p):sha(p) for p in (SCRIPT,Path(quality_from_row.__code__.co_filename),
                                    Path(command_relative_variance.__code__.co_filename))}
    reports = []
    for directory in replays:
        files = [directory / name for name in ('manifest.json','resolved_env.json','completed.json','trace.csv')]
        manifest,cfg,complete = [read(path) for path in files[:3]]
        assert complete['completed'] and not manifest['weights_updated']
        assert not manifest['sampled_policy_actions'] and not manifest['policy_input_remapping']
        rewards = cfg['rewards']
        assert rewards['profile']=='joint_pd_balance_v1'
        evidence = manifest['tracking_goal_feedback_criteria']
        assert sha(evidence['path'])==evidence['sha256']
        criteria = evidence['criteria']
        floor = rewards['tracking_task_quality_gate_floor']
        kernel = manifest.get('application_reward_kernel')
        if kernel is not None:
            assert kernel['version']=='positive_tracking_command_relative_variance_v1'
            assert kernel['same_formula_for_every_environment_height_sign_and_task']
            for axis,sigma in [('speed',rewards['tracking_sigma']),('yaw',rewards['tracking_sigma_ang'])]:
                validate_parameters(sigma,kernel['command_caps'][axis],kernel['common_floor_fraction'])
        expected_per_trial = round(manifest['seconds_per_case']/manifest['policy_dt'])
        expected_keys = {(i,seed) for i in range(len(manifest['cases'])) for seed in manifest['seeds']}
        counts = {key:0 for key in expected_keys}
        maximum = dict(speed_bonus=0.,yaw_bonus=0.)
        old_scalar_disagreement = dict(speed_bonus=0.,yaw_bonus=0.)
        rows = 0
        dt = manifest['policy_dt']
        with files[3].open(newline='') as stream:
            for raw in csv.DictReader(stream):
                row = {key:float(raw[key]) for key in ('case_id','seed','height_m','height_cmd_m',
                    'pitch_rad','roll_rad','speed_cmd_m_s','yaw_cmd_rad_s','speed_m_s','yaw_rate_rad_s',
                    'reward_track_lin_vel','reward_track_ang_vel')}
                assert all(math.isfinite(value) for value in row.values())
                key = int(row['case_id']),int(row['seed'])
                assert key in counts
                counts[key] += 1
                quality = 1. if floor==1. else quality_from_row(row,criteria,floor)
                for axis,command_field,measured_field,sigma_field,coefficient_field,recorded_field,label in (
                    ('speed','speed_cmd_m_s','speed_m_s','tracking_sigma','tracking_lin_vel','reward_track_lin_vel','speed_bonus'),
                    ('yaw','yaw_cmd_rad_s','yaw_rate_rad_s','tracking_sigma_ang','tracking_ang_vel','reward_track_ang_vel','yaw_bonus')):
                    error = row[measured_field]-row[command_field]
                    nominal = rewards[sigma_field]
                    variance = nominal if kernel is None else command_relative_variance(
                        row[command_field],nominal,kernel['command_caps'][axis],kernel['common_floor_fraction'])
                    bound = rewards['clip_single_reward']*dt
                    scale = rewards[coefficient_field]*dt*quality
                    expected = max(-bound,min(bound,scale*math.exp(-error*error/variance)))
                    old = max(-bound,min(bound,scale*math.exp(-error*error/nominal)))
                    maximum[label] = max(maximum[label],abs(expected-row[recorded_field]))
                    old_scalar_disagreement[label] = max(old_scalar_disagreement[label],abs(old-row[recorded_field]))
                rows += 1
        assert all(value==expected_per_trial for value in counts.values())
        assert rows==complete['samples']==expected_per_trial*len(expected_keys)
        assert max(maximum.values())<2.e-6, (directory,maximum)
        if kernel is not None:
            assert max(old_scalar_disagreement.values())>1.e-4, 'The declared revised kernel did not differ from the old scalar arithmetic'
        reports.append(dict(replay=str(directory),checkpoint_sha256=manifest['checkpoint_sha256'],
            samples=rows,all_case_seed_rows_retained=True,reset_count=complete['resets'],
            application_reward_kernel=kernel or {'version':'fixed_absolute_tracking_variance_v1'},
            existing_height_pose_quality_guard_retained=True,
            existing_quality_guard_criteria_sha256=evidence['sha256'],
            positive_kernel_reconstruction_max_error=maximum,
            old_fixed_variance_reconstruction_max_error=old_scalar_disagreement,
            no_trajectory_or_policy_improvement_claim=True))
        pins.update({str(path):sha(path) for path in files})
    assert all(sha(path)==value for path,value in pins.items())
    result = dict(verified=True,observed_at=datetime.now(timezone.utc).isoformat(),replays=reports,
        raw_reward_cross_policy_comparison_requires_identical_application_reward_kernel=True,
        kernel_normalizer_does_not_read_acceptance_but_existing_quality_guard_does=True,
        simulator_started=False,policy_executed=False,optimizer_executed=False,
        training_modified=False,acceptance_changed=False,policy_accepted=False,source_pins=pins)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True,replays=reports,training_modified=False),ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify([path.resolve() for path in args.replay],args.output.resolve())
