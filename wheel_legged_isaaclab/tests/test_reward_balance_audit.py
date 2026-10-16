"""Complete telemetry audits must distinguish disabled and missing rewards."""
import csv
import json

import pytest

from diagnose_reward_balance import audit


def write_trace(directory, *, worst_coefficient=0., include_worst=False,
                include_height=True, corrupt_sum=False):
    criteria=dict(relative_error=.1,speed_absolute_m_s=.025,yaw_absolute_rad_s=.025,
                  wheel_absolute_rad_s=.15,leg_angle_design_range_rad=.2,body_p95_deg=2.5)
    manifest=dict(checkpoint='diagnostic_fixture.pt',checkpoint_sha256='fixture',asset_sha256='fixture',
                  policy_dt=.02,seconds_per_case=.08,cases=[dict(name='stand')],seeds=[83,84,85],
                  tracking_goal_feedback_criteria=dict(criteria=criteria))
    rewards=dict(base_height=2.,base_height_sigma=.0004,base_height_relative_deadband=0.,
                 tracking_goal_violation=worst_coefficient,tracking_goal_penalty_clip=40.,
                 tracking_goal_component_violation=-.75,tracking_goal_component_penalty_clip=200.)
    (directory/'manifest.json').write_text(json.dumps(manifest))
    (directory/'completed.json').write_text(json.dumps(dict(completed=True)))
    (directory/'resolved_env.json').write_text(json.dumps(dict(rewards=rewards)))
    row=dict(case_id=0,episode=0,time_s=0.,speed_m_s=0.,speed_cmd_m_s=0.,
             yaw_rate_rad_s=0.,yaw_cmd_rad_s=0.,height_m=.3,height_cmd_m=.3,
             pitch_rad=0.,roll_rad=0.,reward_tracking_goal_component_violation=0.,
             reward_total=.041 if corrupt_sum else .04)
    for side in ('left','right'):
        row.update({side+'_length_m':.25,side+'_length_ref_m':.25,
                    side+'_angle_rad':0.,side+'_angle_ref_rad':0.,
                    side+'_wheel_rad_s':0.,side+'_wheel_ref_rad_s':0.})
    if include_height:
        row['reward_base_height']=.04
    if include_worst:
        row['reward_tracking_goal_violation']=0.
    with (directory/'trace.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        writer.writeheader()
        for seed in manifest['seeds']:
            for step in range(4):
                writer.writerow(dict(row,time_s=step*.02))


def test_disabled_omitted_reward_is_audited_as_zero(tmp_path):
    write_trace(tmp_path)
    report=audit(tmp_path)
    assert report['complete_finite_rows']==12
    assert max(report['independent_formulas_max_abs_errors'].values())<1.e-12
    assert report['sum_of_recorded_terms_max_abs_error']==0.
    assert report['active_reward_terms']==['base_height']


def test_present_active_reward_remains_auditable(tmp_path):
    write_trace(tmp_path,worst_coefficient=-.5,include_worst=True)
    assert audit(tmp_path)['independent_formulas_max_abs_errors']['tracking_goal_violation']==0.


def test_missing_active_reward_is_rejected_even_when_its_instantaneous_cost_is_zero(tmp_path):
    write_trace(tmp_path,worst_coefficient=-.5)
    with pytest.raises(ValueError,match='Missing active reward telemetry: tracking_goal_violation'):
        audit(tmp_path)


def test_missing_nonzero_height_reward_cannot_pass_audit(tmp_path):
    write_trace(tmp_path,include_height=False)
    with pytest.raises(ValueError,match='reward total differs'):
        audit(tmp_path)


def test_disabled_term_handling_does_not_mask_a_corrupt_total(tmp_path):
    write_trace(tmp_path,corrupt_sum=True)
    with pytest.raises(ValueError,match='reward total differs'):
        audit(tmp_path)


def disable_feedback(directory):
    p=directory/'resolved_env.json';cfg=json.loads(p.read_text())
    cfg['rewards']['tracking_goal_component_violation']=0.
    p.write_text(json.dumps(cfg))
    p=directory/'manifest.json';manifest=json.loads(p.read_text())
    manifest['tracking_goal_feedback_criteria']=None;p.write_text(json.dumps(manifest))


def test_plain_ppo_without_composite_reward_audits_original_gate_ratios_without_enabling_feedback(tmp_path):
    write_trace(tmp_path);disable_feedback(tmp_path)
    report=audit(tmp_path)
    assert report['complete_finite_rows']==12
    assert not report['reward_feedback_criteria_recorded']
    assert report['diagnostic_ratio_criteria_source']=='original_acceptance_file'
    assert report['reward_profile']['tracking_goal_component_violation']==0.
    assert max(report['independent_formulas_max_abs_errors'].values())<1.e-12


def test_active_feedback_still_requires_its_manifest_criteria(tmp_path):
    write_trace(tmp_path)
    p=tmp_path/'manifest.json';manifest=json.loads(p.read_text())
    manifest['tracking_goal_feedback_criteria']=None;p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Missing criteria for active'):
        audit(tmp_path)


@pytest.mark.parametrize('corrupt',[False,True])
def test_actual_quadratic_height_cost_is_audited_and_included_in_local_derivative(tmp_path,corrupt):
    import math
    write_trace(tmp_path);disable_feedback(tmp_path)
    p=tmp_path/'resolved_env.json';cfg=json.loads(p.read_text())
    cfg['rewards'].update(base_height_error_sq=-100.,base_height_penalty_clip=30.)
    p.write_text(json.dumps(cfg))
    error=.05;kernel=2.*math.exp(-error**2/.0004)
    with (tmp_path/'trace.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        row.update(height_m=.35,reward_base_height=.02*kernel,
            reward_base_height_error_sq=-.02*100.*error**2+(.001 if corrupt else 0.))
        row['reward_total']=row['reward_base_height']+row['reward_base_height_error_sq']
    with (tmp_path/'trace.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if corrupt:
        with pytest.raises(ValueError,match='Independent reward formulas disagree'):audit(tmp_path)
    else:
        report=audit(tmp_path)
        assert report['independent_formulas_max_abs_errors']['base_height_error_sq']<1.e-12
        derivative=-2.*error*kernel/.0004-2.*100.*error
        assert report['conditions'][0]['height_state_derivative_mean_per_m_s']==pytest.approx(derivative,abs=.001)
