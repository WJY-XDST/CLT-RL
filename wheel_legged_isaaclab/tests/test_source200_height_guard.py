import copy
import json
from pathlib import Path

import pytest
import continue_joint_pd_source200_height as controller
from test_joint_pd_priority import report,trial
from joint_pd_rollback import admit
from wheel_legged_gym_isaaclab.robot_model import MODEL,MAPPED_FIVE_BAR


@pytest.fixture
def configs():
    source=dict(leg_control_mode='implicit_joint_position',
        leg_joint_position_offset=[-.0707452938,-.0336936327,.0707452938,.0336936327],
        leg_joint_position_scale=[.3]*4,wheel_radius=.06,five_bar_geometry=copy.deepcopy(MAPPED_FIVE_BAR),
        robot=dict(spawn=dict(usd_path='legacy.usd'),
            init_state=dict(pos=[0.,0.,MODEL['nominal_root_height']],joint_pos={'.*':0.})),
        commands=dict(ranges_height=[.28,.32],standing_only_steps=48000,yaw_start_steps=48000,
            reverse_ramp_start_steps=48000,motion_ramp_steps=48000),
        rewards=dict(base_height_error_sq=0.,orientation=-20.,tracking_lin_vel=2.,tracking_ang_vel=1.))
    return source,None


def test_source200_baseline_changes_only_authorized_physics_reset_and_commands(configs):
    source,_=configs
    source['commands'].update(fixed_standing_groups=True,standing_env_fraction=.25,grouped_training=True)
    before=copy.deepcopy(source)
    cfg=controller.starting_config(source,'measured.usda')
    assert source==before
    assert cfg['rewards']==source['rewards']
    assert cfg['leg_joint_position_offset']==source['leg_joint_position_offset']
    assert cfg['leg_joint_position_scale']==source['leg_joint_position_scale']
    assert cfg['robot']['spawn']['usd_path']=='measured.usda'
    assert cfg['commands']['full_height_range_from_start']
    assert cfg['commands']['standing_env_fraction']==.25
    assert all(cfg['commands'][k]==0 for k in ('standing_only_steps','yaw_start_steps','reverse_ramp_start_steps'))


def test_only_height_cost_changes_without_reintroducing_composite_or_auxiliary_losses():
    cfg=dict(rewards=dict(base_height_error_sq=0.,orientation=-20.));used=set()
    values=[]
    for _ in range(4):
        seed,delta,key=controller.height_proposal(cfg,report(),used);used.add(key)
        assert seed==43 and set(delta)=={'base_height_error_sq'}
        values.append(delta['base_height_error_sq'])
    assert values==[-100.,-50.,-250.,-500.]
    cfg['rewards']['base_height_error_sq']=-100.
    assert controller.height_proposal(cfg,report(),set())[1]=={'base_height_error_sq':-100.}


def test_height_gain_cannot_buy_stationary_drift_regression_even_if_starting_baseline_has_failures():
    before=report();trial(before,'height_low')['metrics']['yaw_mae_rad_s']=.035
    after=copy.deepcopy(before)
    for c in after['conditions']:
        for t in c['trials']:t['metrics']['height_mae_m']=0.
    trial(after,'height_low')['metrics']['yaw_mae_rad_s']=.04
    assert not admit(after,before,before)['promote']


def test_owner_selected_start_keeps_endpoint_drift_failures_visible_and_does_not_claim_acceptance(tmp_path,monkeypatch):
    candidate=tmp_path/'candidate';(candidate/'replay').mkdir(parents=True)
    (candidate/'replay/completed.json').touch()
    base=tmp_path/'coordinator';base.mkdir()
    abandoned=tmp_path/'old';abandoned.mkdir()
    cp=tmp_path/'snapshot/model_200.pt'
    (candidate/'status.json').write_text(json.dumps(dict(checkpoint=str(cp),training_active=False)))
    (abandoned/'status.json').write_text(json.dumps(dict(checkpoint=str(tmp_path/'old.pt'),
        checkpoint_sha256=controller.SOURCE_SHA,training_active=False)))
    value=report();value['checkpoint_sha256']=controller.SOURCE_SHA
    trial(value,'height_high')['metrics']['speed_mae_m_s']=.06
    limits=copy.deepcopy(trial(value,'height_high')['limits'])
    monkeypatch.setattr(controller.guard.core,'sha',lambda p:controller.guard.GATE_SHA
        if Path(p).name=='tracking_acceptance.json' else controller.SOURCE_SHA)
    monkeypatch.setattr(controller.guard.core,'saved_updates',lambda p:201)
    monkeypatch.setattr(controller.guard.core,'qualified_wheel_mass_asset',lambda p:('measured.usda',dict(asset_chain_sha256={})))
    monkeypatch.setattr(controller,'assess',lambda p:value)
    assert controller.owner_bootstrap(candidate,abandoned,base)==candidate
    proof=json.loads((base/'source200_baseline_verification.json').read_text())
    assert proof['starting_baseline_only_not_acceptance']
    assert not proof['stationary_pose_passed'] and not proof['policy_accepted']
    assert not proof['full_original_gate_passed']
    assert trial(value,'height_high')['limits']==limits
    plan=json.loads((candidate/'experiment_plan.json').read_text())
    assert plan['owner_selected_checkpoint'] and plan['endpoint_drift_failures_not_hidden']
