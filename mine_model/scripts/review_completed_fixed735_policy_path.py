"""Retrospective CPU-only reference path for a verified completed PPO50 node.

Every policy is queried on the same735 recorded source observations. These
reference proxies are not candidate trajectories, acceptance or model selection.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def run(training, reference_scope, previous_comparison, output):
    assert not output.exists()
    proof_path = training/'final_checkpoint_verification_readonly.json'
    health_path = training/'original11_health50_complete.json'
    plan_path = training/'experiment_plan.json'
    proof,health,plan = map(read,[proof_path,health_path,plan_path])
    assert proof['verified'] and proof['all50_native_updates_verified'] and health['qualified']
    source_actual = plan['source_actual_ppo_updates']
    assert proof['actual_ppo_updates'] == source_actual+50 and health['actual_updates']==50
    assert plan['ordinary_full_policy_PPO'] and not plan['auxiliary_losses_enabled']
    source = Path(proof['source_checkpoint'])
    final = Path(proof['checkpoint'])
    assert sha(source)==proof['source_checkpoint_sha256'] and sha(final)==proof['checkpoint_sha256']
    scope = read(reference_scope)
    assert scope['trial_count']==735 and scope['unique_numeric_commands']==219
    pins = {str(p):sha(p) for p in [Path(__file__),proof_path,health_path,plan_path,source,
                                    reference_scope,previous_comparison,source.parent/'params/agent.yaml']}
    x,old,cases = [],[],[]
    contract = None
    for batch in scope['batches']:
        folder = Path(batch['replay'])
        manifest_path,cfg_path,archive_path = [folder/name for name in ['manifest.json','resolved_env.json','policy_inputs.npz']]
        manifest,cfg = map(read,[manifest_path,cfg_path])
        assert manifest['checkpoint_sha256']==proof['source_checkpoint_sha256']
        assert not manifest['sampled_policy_actions'] and not manifest['weights_updated']
        now={key:cfg[key] for key in ['wheel_radius','wheel_track_width','action_scale_vel','wheel_action_limit','obs_scales']}
        if contract is None: contract=now
        assert contract==now
        with np.load(archive_path,allow_pickle=False) as archive:
            x.append(archive['observations'][-1].copy())
            old.append(archive['means'][-1].copy())
        cases += [dict(case=case,seed=seed) for case in batch['cases'] for seed in batch['seeds']]
        for path in [manifest_path,cfg_path,archive_path]:pins[str(path)]=sha(path)
    x,recorded=np.concatenate(x),np.concatenate(old)
    assert x.shape==(735,27) and recorded.shape==(735,6) and len(cases)==735
    commands=np.asarray([[row['case'][axis] for axis in ['speed','yaw','height']] for row in cases])
    scales=np.asarray([contract['obs_scales'][key] for key in ['lin_vel','ang_vel','height_measurements']])
    assert np.max(np.abs(x[:,6:9]/scales-commands))<1.e-6
    previous=read(previous_comparison)
    assert previous['verified'] and previous['fixed_observations']==735
    candidate_labels=[name for name,value in previous['results'].items() if value['checkpoint_sha256']==proof['checkpoint_sha256']]
    assert len(candidate_labels)==1
    final_recorded=np.asarray([row['predictions'][candidate_labels[0]]['raw_means'] for row in previous['all_original_trials']])
    assert [(v['case'],v['seed']) for v in cases]==[(v['case'],v['seed']) for v in previous['all_original_trials']]
    options=yaml.safe_load((source.parent/'params/agent.yaml').read_text())
    policy_options=dict(options['policy']);assert policy_options.pop('class_name')=='ActorCritic'
    observations=TensorDict({'policy':torch.from_numpy(x)},[735])
    with contextlib.redirect_stdout(io.StringIO()),torch.random.fork_rng(devices=[]):
        policy=ActorCritic(observations,options['obs_groups'],6,**policy_options)
    policy.eval()
    names=list(policy.named_parameters());assert len(names)==17
    last=next(name.rsplit('.',1)[0] for name,_ in reversed(names) if name.startswith('actor.') and name.endswith('.weight'))
    assert dict(names)[last+'.bias'].shape==(6,)
    models=[(source,source_actual)]+[(final.parent/f'model_{actual-1}.pt',actual) for actual in range(source_actual+1,source_actual+51)]
    predictions,rows=[],[]
    base_bias=None
    source_parity=final_parity=None
    for index,(path,actual) in enumerate(models):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert int(saved['iter'])+1==actual and len(saved['optimizer_state_dict']['state'])==17
        assert {float(item['step']) for item in saved['optimizer_state_dict']['state'].values()}=={actual*20}
        assert saved['model_state_dict']['std'].shape==(6,)
        assert all(bool(torch.isfinite(value).all()) for value in saved['model_state_dict'].values())
        policy.load_state_dict(saved['model_state_dict'],strict=True)
        with torch.inference_mode():mu=policy.act_inference(observations).detach().numpy().copy()
        assert mu.shape==(735,6) and np.isfinite(mu).all()
        if index==0:
            source_parity=float(np.abs(mu-recorded).max());assert source_parity<2.e-6
        if index==50:
            final_parity=float(np.abs(mu-final_recorded).max());assert final_parity<2.e-6
        predictions.append(mu)
        wheels=np.clip(mu[:,[2,5]],-contract['wheel_action_limit'],contract['wheel_action_limit'])*contract['action_scale_vel']
        v=contract['wheel_radius']*wheels.mean(1)
        w=contract['wheel_radius']*(wheels[:,1]-wheels[:,0])/contract['wheel_track_width']
        bias=saved['model_state_dict'][last+'.bias'].double().numpy()
        if base_bias is None:base_bias=bias.copy()
        bias_delta=bias-base_bias
        leg_mu=mu[:,[0,1,3,4]].astype(np.float64)
        soft=plan['training_config']['rewards']['joint_position_action_soft_limit']
        soft_penalty=np.maximum(np.abs(leg_mu)-soft,0.)**2
        entry=dict(relative_update=index,actual_updates=actual,checkpoint=str(path),checkpoint_sha256=sha(path),
            saved_std=saved['model_state_dict']['std'].tolist(),saved_actual_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
            mean_signed_speed_reference_proxy_minus_command_m_s=float((v-commands[:,0]).mean()),
            speed_reference_proxy_MAE_m_s=float(np.abs(v-commands[:,0]).mean()),
            yaw_reference_proxy_MAE_rad_s=float(np.abs(w-commands[:,1]).mean()),
            raw_wheel_mean_clipping_fraction=float((np.abs(mu[:,[2,5]])>contract['wheel_action_limit']).mean()),
            raw_leg_mean_unit_exceedance_fraction=float((np.abs(leg_mu)>1.).mean()),
            raw_leg_mean_unit_overshoot_RMS=float(np.sqrt(np.square(np.maximum(np.abs(leg_mu)-1.,0.)).mean())),
            original_soft_penalty_sum_mean=float(soft_penalty.sum(1).mean()),
            actor_last_bias_delta_all6=bias_delta.tolist(),
            exact_wheel_bias_parameter_component_in_speed_reference_proxy_shift_m_s=float(
                contract['wheel_radius']*contract['action_scale_vel']*bias_delta[[2,5]].mean()))
        if index>0:
            packet_path=training/'native_learning_signal/original_adaptive_KL'/f'update_{index:04d}.json'
            packet=read(packet_path)
            assert packet['verified'] and packet['full_original_PPO_minibatches']==20
            assert all(row['full_minibatch_examples']==98304 for row in packet['rows'])
            entry['original_actual_minibatch_KL_mean']=float(np.mean([r['scheduler_KL_mean'] for r in packet['rows']]))
            entry['original_actual_adaptive_LR_mean']=float(np.mean([r['actual_adaptive_LR_after_scheduler'][0] for r in packet['rows']]))
            pins[str(packet_path)]=sha(packet_path)
        rows.append(entry);pins[str(path)]=sha(path)
    base=rows[0]['mean_signed_speed_reference_proxy_minus_command_m_s']
    for row in rows:
        row['total_speed_reference_proxy_shift_vs_source_m_s']=row['mean_signed_speed_reference_proxy_minus_command_m_s']-base
        assert row['raw_wheel_mean_clipping_fraction']==0.,'Bias decomposition assumes no wheel clipping'
        row['all_actor_weight_changes_component_in_speed_reference_proxy_shift_m_s']=(
            row['total_speed_reference_proxy_shift_vs_source_m_s']-
            row['exact_wheel_bias_parameter_component_in_speed_reference_proxy_shift_m_s'])
    def slope(first,last):
        chosen=rows[first:last+1]
        return float(np.polyfit([r['relative_update'] for r in chosen],
            [r['mean_signed_speed_reference_proxy_minus_command_m_s'] for r in chosen],1)[0])
    output.mkdir()
    archive=output/'all51_policy_outputs_on_same735_inputs.npz'
    np.savez_compressed(archive,observations=x,commands=commands,raw_means=np.stack(predictions),
                        relative_updates=np.arange(51,dtype=np.int64))
    pins[str(archive)]=sha(archive)
    result=dict(verified=True,observed_at=datetime.now(timezone.utc).isoformat(),
        source_checkpoint_sha256=proof['source_checkpoint_sha256'],final_checkpoint_sha256=proof['checkpoint_sha256'],
        source_actual_updates=source_actual,final_actual_updates=proof['actual_ppo_updates'],
        completed_node_updates=50,retrospective_only_after_final50_proof=True,
        fixed_source_observations=735,numeric_commands=219,source_CPU_actor_parity_max_abs_error=source_parity,
        final_CPU_actor_parity_max_abs_error=final_parity,all51_saved_states_and_all17Adam_step_verified=True,
        source_last_actor_layer=last,source_pins=pins,all51_reference_summaries=rows,
        speed_reference_bias_slope_m_s_per_actual_PPO_update=dict(first25=slope(0,25),last25=slope(25,50),last10=slope(40,50)),
        limitations=['Every intermediate policy is queried on fixed source states. These are not intermediate candidate trajectories, actual body tracking errors or acceptance results.',
            'The endpoint is cross-checked against the existing independently computed four-policy735 actor outputs. No policy input is modified and no simulator callback is invoked.',
            'The output-bias versus other-weight split is an exact algebraic accounting of wheel reference means with zero wheel clipping; it is not a causal gradient attribution.',
            'Slopes are descriptive finite-node output changes and do not predict future improvement. No checkpoint is selected or promoted by these proxies. Original735, independent189 and sustained gates remain required.'],
        optimizer_executed=False,backward_executed=False,simulator_started=False,training_modified=False,
        parameter_selected=False,policy_accepted=False)
    assert all(sha(path)==digest for path,digest in pins.items())
    (output/'verified_reference_path.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True,final_checkpoint_sha256=proof['checkpoint_sha256'],
        markers=[{k:r[k] for k in ['relative_update','mean_signed_speed_reference_proxy_minus_command_m_s',
                 'yaw_reference_proxy_MAE_rad_s','original_soft_penalty_sum_mean',
                 'exact_wheel_bias_parameter_component_in_speed_reference_proxy_shift_m_s']} for r in rows if r['relative_update']%10==0],
        slopes=result['speed_reference_bias_slope_m_s_per_actual_PPO_update'],parameter_selected=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--reference-scope',type=Path,required=True)
    parser.add_argument('--previous-fixed-input-comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    run(a.training_output.resolve(),a.reference_scope.resolve(),a.previous_fixed_input_comparison.resolve(),a.output.resolve())
