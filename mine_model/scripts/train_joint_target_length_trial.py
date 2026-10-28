"""One diagnosed direct-joint decision-cost revision, then 100 native updates.

Historical checkpoints and original acceptance stay immutable. The added
default-zero reward evaluates existing applied motor-target FK, never changes
an action or introduces runtime IK. Failed reverse trials remain visible.
"""
import argparse
import ast
import copy
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, optimizer_step, qualified_wheel_mass_asset, sha, tensor_finite
from diagnose_height_response import actor
from train_joint_pd_posture_repair import GATE_SHA
import train_joint_pd_reward_block as block
from train_joint_pd_standing_seed import nominal_standing_intact, saved_updates
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.reward_signals import joint_target_length_error_squared
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot

KEY = 'joint_target_leg_length_error_sq'
COEFFICIENT = -500.
UPDATES = 100


def revised_config(saved):
    result = copy.deepcopy(saved)
    c, r = result['commands'], result['rewards']
    if (result['leg_control_mode'] != 'implicit_joint_position'
            or result.get('joint_reset_profile') != 'joint_target_nominal_v1'
            or tuple(c['ranges_height']) != SAFE_BASE_HEIGHT_RANGE_M
            or not c['full_height_range_from_start']
            or any(c[k] != 0 for k in ('standing_only_steps','yaw_start_steps','reverse_ramp_start_steps'))
            or c['motion_ramp_steps'] != 48000
            or r['base_height_error_sq'] != -500. or r['orientation'] != -400.
            or r.get(KEY, 0.) != 0. or r['tracking_lin_vel'] <= 0. or r['tracking_ang_vel'] <= 0.):
        raise ValueError('Preserve the diagnosed full-height normal-motion source and all existing rewards')
    if abs(COEFFICIENT)*(.15**2) >= r['base_height_penalty_clip']:
        raise ValueError('The decision cost must not flatten across the full command range')
    r[KEY] = COEFFICIENT
    return result


def normalized_revision_tree(text, filename):
    tree = ast.parse(text)
    if filename == 'wheel_legged_vmc_flat_env.py':
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'WheelLeggedVMCFlatEnv':
                node.body = [n for n in node.body if not
                    (isinstance(n, ast.FunctionDef) and n.name in ('_get_rewards','_validate_configuration'))]
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith('.reward_signals'):
                node.names = [n for n in node.names if n.name not in
                              ('joint_target_length_error_squared','validate_joint_target_length_reward')]
            if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REWARD_NAMES' for t in node.targets):
                node.value.elts = [n for n in node.value.elts if not (isinstance(n,ast.Constant) and n.value==KEY)]
    elif filename == 'wheel_legged_vmc_flat_env_cfg.py':
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'RewardsCfg':
                kept = []
                for item in node.body:
                    if isinstance(item,ast.Assign) and any(isinstance(t,ast.Name) and t.id==KEY for t in item.targets):
                        if ast.literal_eval(item.value) != 0.:
                            raise ValueError('Historical default must remain zero')
                    else:
                        kept.append(item)
                node.body = kept
    elif filename == 'reward_signals.py':
        tree.body = [n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in
                     ('joint_target_length_error_squared','validate_joint_target_length_reward'))]
    elif filename == 'joint_pd_continuation.py':
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name=='validate_continuation_config':
                for item in ast.walk(node):
                    if isinstance(item,ast.Tuple):
                        item.elts = [n for n in item.elts if not (isinstance(n,ast.Tuple) and len(n.elts)==2
                            and isinstance(n.elts[0],ast.Constant) and n.elts[0].value==KEY)]
    else:
        raise ValueError('Unexpected code-revision target')
    return ast.dump(tree, include_attributes=False)


def verify_reward_only_revision(snapshot):
    for name, old in snapshot.items():
        path = Path(name)
        import hashlib
        if hashlib.sha256(old['source_text'].encode()).hexdigest() != old['sha256']:
            raise ValueError('The pre-revision source snapshot is corrupt')
        if normalized_revision_tree(old['source_text'],path.name) != normalized_revision_tree(path.read_text(),path.name):
            raise ValueError('Changes escaped reward/default-zero compatibility boundaries: '+name)


def qualify(previous, output, source, snapshot):
    """CPU mathematical/legacy checks, NOT a rollout or a trained candidate."""
    verify_reward_only_revision(snapshot)
    before_sha = sha(source)
    sys.path.insert(0,str(ROOT/'wheel_legged_isaaclab/tests'))
    try:
        from test_joint_pd_balance_rewards import reward_fixture
    finally:
        sys.path.pop(0)
    old_env = next(v['source_text'] for k,v in snapshot.items() if Path(k).name=='wheel_legged_vmc_flat_env.py')
    generator = torch.Generator().manual_seed(843)
    for _ in range(20):
        left, lv, original = reward_fixture(128,source_text=old_env)
        right, rv, current = reward_fixture(128)
        for attribute in ('root_pos_w','root_ang_vel_b','projected_gravity_b'):
            value = getattr(left._robot.data,attribute)
            value += torch.randn(value.shape,generator=generator)*.02
            getattr(right._robot.data,attribute).copy_(value)
        lv[:] = torch.randn(lv.shape,generator=generator)*.3; rv.copy_(lv)
        left._commands[:,2] = .22534645531+torch.rand((128,),generator=generator)*.15
        right._commands.copy_(left._commands)
        left._raw_actions[:] = torch.randn((128,6),generator=generator)
        right._raw_actions.copy_(left._raw_actions)
        if not torch.equal(original(),current()):
            raise ValueError('Default-zero reward is not bitwise equal to the recorded previous source')
    cfg = validate_checkpoint_robot(source)
    saved = torch.load(source,map_location='cpu',weights_only=True)
    with np.load(previous/'replay/policy_inputs.npz',allow_pickle=False) as data:
        # Only surviving static heights are used for this local derivative check.
        observations = torch.tensor(data['observations'][-150::10, [0,1,2,21,22,23,24,25,26]].copy()).reshape(-1,27)
    with torch.no_grad():
        means = actor(observations,saved['model_state_dict'])
    raw = means.detach().requires_grad_(True)

    def score(value):
        q = value[:,(0,1,3,4)].clamp(-1,1)*value.new_tensor(cfg['leg_joint_position_scale'])+value.new_tensor(cfg['leg_joint_position_offset'])
        length,_,_,valid = five_bar_state(q[:,::2],q[:,1::2],**cfg['five_bar_geometry'])
        if not valid.all():
            raise ValueError('Invalid applied-target FK in local reward test')
        return COEFFICIENT*joint_target_length_error_squared(length,
            observations[:,8]/cfg['obs_scales']['height_measurements'],cfg['rewards']['leg_length_target_height_offset'])

    value = score(raw)
    gradient = torch.autograd.grad(value.sum(),raw)[0]
    if not torch.equal(gradient[:,(2,5)],torch.zeros_like(gradient[:,(2,5)])):
        raise ValueError('The length decision cost must not directly reward wheel actions')
    proposed = raw+.001*gradient/gradient.norm(dim=1,keepdim=True).clamp_min(1.e-9)
    improved_fraction = float((score(proposed)>value).float().mean())
    if improved_fraction < .99 or sha(source)!=before_sha:
        raise ValueError('Local command-conditioned reward direction failed or source changed')
    result = dict(source_checkpoint=str(source),source_sha256=before_sha,
        default_zero_reward_bitwise_equal=True,legacy_randomized_batches=20,batch_size=128,
        raw_action_reward_direction_improved_fraction=improved_fraction,
        direct_wheel_action_gradient_zero=True,physics_and_action_mapping_unchanged=True,
        checkpoint_or_Adam_updated=False,policy_accepted=False,
        limitation='Local FK reward derivatives and default-zero parity only; not a PPO training result, physical response, stability or tracking acceptance.',
        revised_sources={name:sha(Path(name)) for name in snapshot})
    write_json(output/'reward_revision_qualification.json',result)
    return result


def prepare(previous, output, key=KEY, value=COEFFICIENT, updates=UPDATES):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if (previous.parent != ROOT/'mine_model/results' or output.parent != previous.parent
            or previous==output or output.name!='joint_pd_optimization_20261015_joint_target_length_100'
            or key!=KEY or value!=COEFFICIENT or updates!=UPDATES):
        raise ValueError('Use the owned diagnosed 100-update decision-cost revision')
    state = json.loads((previous/'status.json').read_text())
    old_plan = json.loads((previous/'experiment_plan.json').read_text())
    review = json.loads((previous/'learning_review_after_source200_strong_height_100.json').read_text())
    source = Path(state['checkpoint']).resolve()
    if (state['training_active'] or state['state']!='agent_diagnosis_required'
            or state['actual_ppo_updates']!=301 or source.name!='model_300.pt'
            or sha(source)!=state['checkpoint_sha256'] or review['checkpoint_sha256']!=sha(source)):
        raise ValueError('Require the latest retained model_300 review boundary; do not restart an old fork')
    status = output/'status.json'
    if status.exists() and json.loads(status.read_text())['state']!='prepared':
        raise ValueError('Do not repeat a started block; inspect its latest saved boundary')
    saved = torch.load(source,map_location='cpu',weights_only=True)
    if (saved_updates(source)!=301 or optimizer_step(saved)!=6020 or not tensor_finite(saved)
            or len(saved['optimizer_state_dict']['state'])!=17):
        raise ValueError('Retain every weight and all 17 complete Adam states')
    expected = revised_config(validate_checkpoint_robot(source))
    report = json.loads((previous/'replay/assessment.json').read_text())
    if (report['checkpoint_sha256']!=sha(source) or report['total_trials']!=27
            or not nominal_standing_intact(report)):
        raise ValueError('Require full source replay and preserved nominal standing, not only selected successes')
    snapshot_path = output/'pre_reward_revision_source_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text())
    qualification = qualify(previous,output,source,snapshot)
    pins = {}
    changes = {}
    for path,digest in old_plan['source_files'].items():
        current = sha(Path(path))
        if current!=digest:
            if path not in snapshot or snapshot[path]['sha256']!=digest:
                raise ValueError('Unexplained prior source, asset, gate or checkpoint change: '+path)
            changes[path] = dict(before=digest,after=current)
        pins[path] = current
    asset, calibration = qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
    if expected['robot']['spawn']['usd_path']!=asset or sha(ROOT/'mine_model/config/tracking_acceptance.json')!=GATE_SHA:
        raise ValueError('Owner-calibrated physics and the original gate must remain unchanged')
    watched = [source,*sorted((source.parent/'params').glob('*.yaml')),
        previous/'status.json',previous/'experiment_plan.json',previous/'comparison.json',
        previous/'learning_review_after_source200_strong_height_100.json',
        *(previous/'replay'/name for name in ('manifest.json','assessment.json','completed.json','trace.csv')),
        snapshot_path,output/'reward_revision_qualification.json',Path(__file__).resolve(),
        ROOT/'mine_model/scripts/train_joint_pd_reward_block.py',ROOT/'mine_model/scripts/train_joint_pd_posture_repair.py',
        ROOT/'mine_model/scripts/diagnose_reward_balance.py']
    pins.update({str(path):sha(path) for path in watched}); pins.update(calibration['asset_chain_sha256'])
    path = output/'experiment_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['source_files']!=pins or plan['source_checkpoint']!=str(source):
            raise ValueError('The pinned reward revision changed')
    else:
        plan = dict(created_at=datetime.now().astimezone().isoformat(),previous_output=str(previous),
            source_checkpoint=str(source),source_sha256=sha(source),source_actual_ppo_updates=301,
            source_Adam_step=6020,source_Adam_states=17,target_actual_ppo_updates=401,bounded_training_updates=100,
            run_name='joint_target_length_'+output.name+'_u301_'+datetime.now().strftime('%H%M%S_%f'),
            reward_key=KEY,reward_value=COEFFICIENT,only_reward_change={KEY:[0.,COEFFICIENT]},
            source_replay=str(previous/'replay'),protected_reference=old_plan['protected_reference'],
            source_files=pins,qualified_reward_only_code_changes=changes,
            weights_std_Adam_and_adaptive_LR_reset=False,ordinary_full_policy_PPO=True,
            normal_speed_and_yaw_training_retained=True,full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M),
            calibrated_asset=asset,original_acceptance_sha256=GATE_SHA,
            policy_accepted=False,goal_remains_active=True,hardware_enabled=False,shutdown=False)
        write_json(path,plan)
    return plan,expected


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    args.reward_key, args.value, args.updates = KEY, COEFFICIENT, UPDATES
    block.run(args,prepare_function=prepare)
