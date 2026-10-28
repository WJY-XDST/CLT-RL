"""Qualify the complete6201 state and the single linear tracking weight2→3."""
import argparse
import ast
import copy
import json
from pathlib import Path

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT,sha
from review_plain27_single_reward_block200 import rebound
import qualify_plain27_single_reward_transfer as original
import joint_pd_source6201_lin300_runtime as runtime

SCRIPT=Path(__file__).resolve()
OUTPUT=ROOT/'mine_model/results/joint_pd_source6201_lin300_qualification_20261020'


def qualify():
    source,_,_=runtime.anchor()
    result=rebound(original.qualify,SCRIPT=SCRIPT)(source,runtime.SCRIPT,runtime.DECISION,OUTPUT)
    result['source_files'].update({str(p):sha(p) for p in (SCRIPT,original.SCRIPT)})
    result.update(original_single_field_CPU_operations_preserved=True,
        expected_saved_wheel_action_limit=1.25,expected_saved_wheel_reference_cap_rad_s=18.75,
        only_reward_change=runtime.DELTA['rewards'])
    write_json(OUTPUT/'qualification.json',result)
    return result


def linear_formula(rows,cfg,dt):
    criteria=json.loads((ROOT/'mine_model/config/tracking_acceptance.json').read_text())
    h=abs(rows['height_m']-rows['height_cmd_m'])/(criteria['relative_error']*abs(rows['height_cmd_m']))
    a=np.maximum(abs(rows['pitch_rad']),abs(rows['roll_rad']))/np.deg2rad(criteria['body_p95_deg'])
    reward=cfg['rewards']
    floor=reward['tracking_task_quality_gate_floor']
    quality=floor+(1.-floor)/((1.+np.maximum(h-.5,0.)**2)*(1.+np.maximum(a-.5,0.)**2))
    value=reward['tracking_lin_vel']*quality*np.exp(
        -(rows['speed_m_s']-rows['speed_cmd_m_s'])**2/reward['tracking_sigma'])*dt
    return np.minimum(value,reward['clip_single_reward']*dt)


def frozen():
    tree=ast.parse(original.SCRIPT.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='frozen')
    before=copy.deepcopy(function)
    original_expected=copy.deepcopy(next(n.value for n in ast.walk(function)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets)))
    names={
        'reward_track_ang_vel':'reward_track_lin_vel',
        'yaw_sigma010_runtime_evidence.json':'lin300_runtime_evidence.json',
        'independent_angular_formula_max_error':'independent_linear_formula_max_error',
        'height_posture_quality_rewards_exploration_and_execution_unchanged':
            'height_yaw_posture_quality_other_rewards_exploration_and_execution_unchanged',
        'Independent angular formula, total reward delta or actual runtime differs':
            'Independent linear formula, total reward delta or actual runtime differs'}
    class Metadata(ast.NodeTransformer):
        def __init__(self,mapping):
            self.mapping=mapping
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in self.mapping:
                return ast.copy_location(ast.Constant(self.mapping[node.value]),node)
            return node
        def visit_keyword(self,node):
            node=self.generic_visit(node)
            if node.arg in self.mapping:
                node.arg=self.mapping[node.arg]
            return node
    function=Metadata(names).visit(function)
    count=0
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets):
            node.value=ast.parse("linear_formula(new, cb, b['policy_dt'])",mode='eval').body
            count+=1
    restored=copy.deepcopy(function)
    for node in ast.walk(restored):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets):
            node.value=copy.deepcopy(original_expected)
    restored=Metadata({v:k for k,v in names.items()}).visit(restored)
    if count!=1 or ast.dump(restored)!=ast.dump(before):
        raise ValueError('Original physical/observation/action/reward parity or integrity checks changed')
    namespace=dict(original.__dict__,SCRIPT=SCRIPT,linear_formula=linear_formula)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
        str(original.SCRIPT),'exec'),namespace)
    result=namespace['frozen'](runtime.TRAINING,OUTPUT,runtime.SCRIPT)
    baseline=runtime.TRAINING/'replay'
    rows=np.genfromtxt(baseline/'trace.csv',delimiter=',',names=True)
    cfg=json.loads((baseline/'resolved_env.json').read_text())
    old_error=float(abs(linear_formula(rows,cfg,.02)-rows['reward_track_lin_vel']).max())
    if old_error>2.e-7:
        raise ValueError('Independent source linear reward formula does not reproduce')
    result.update(old_linear_formula_max_error=old_error,
        original_parity_reader_AST_restored_identical_except_declared_formula_and_metadata=True,
        only_existing_linear_tracking_coefficient_changed=True,
        orientation60_yaw_height_commands_std_control_physics_unchanged=True)
    result['source_data_sha256'].update({str(p):sha(p) for p in (SCRIPT,original.SCRIPT,runtime.SCRIPT)})
    write_json(OUTPUT/'native_reward_transfer_qualification.json',result)
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('qualify','frozen'))
    args=cli.parse_args()
    result=qualify() if args.mode=='qualify' else frozen()
    print(json.dumps({k:result[k] for k in ('qualified','source_actual_updates','only_reward_change',
        'all_model_and_Adam_tensors_bitwise_preserved','full27_samples','independent_linear_formula_max_error') if k in result}))
