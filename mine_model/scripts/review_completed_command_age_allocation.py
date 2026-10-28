"""Measure command-hold age and advantage allocation after verified PPO50.

This reads 32 cached streams. It does not import a simulator or training code,
query observations, evaluate policies, change sampling, or execute an optimizer.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def run(training, output, previous_cohorts):
    assert not output.exists()
    proof_path = training/'final_checkpoint_verification_readonly.json'
    health_path = training/'original11_health50_complete.json'
    plan_path = training/'experiment_plan.json'
    proof, health, plan = map(read, [proof_path, health_path, plan_path])
    assert proof['verified'] and proof['all50_native_updates_verified'] and health['qualified']
    assert proof['actual_ppo_updates']-plan['source_actual_ppo_updates'] == health['actual_updates'] == 50
    assert proof['all17_Adam_step'] == proof['actual_ppo_updates']*20
    assert sha(proof['checkpoint']) == proof['checkpoint_sha256']
    cfg = plan['training_config']
    assert plan['native_num_envs'] == 8192 and plan['ordinary_full_policy_PPO']
    assert not plan['auxiliary_losses_enabled']
    dt = cfg['sim']['dt']*cfg['decimation']
    assert abs(dt-.02) < 1.e-12
    parts = {k:[] for k in ['native_commands','native_speed','native_yaw','native_height',
                           'advantages','returns','values','dones','native_timeouts']}
    pins = {str(p):sha(p) for p in [Path(__file__), proof_path, health_path, plan_path, previous_cohorts]}
    ids = None
    files = sorted((training/'native_learning_signal').glob('rollout_*.npz'))
    assert len(files) == 50
    for index,path in enumerate(files,1):
        audit_path = path.parent/f'audit_update_{index:04d}.json'
        audit = read(audit_path)
        assert audit['update'] == index and audit['total_environments'] == 8192
        assert audit['probe_RNG_unchanged'] and not audit['added_loss_or_backward']
        assert len(audit['all_environment_alignment_max_errors']) == 9
        assert all(v == 0 for v in audit['all_environment_alignment_max_errors'].values())
        if ids is None: ids = audit['sampled_environment_ids']
        assert audit['sampled_environment_ids'] == ids and len(ids) == 32
        with np.load(path,allow_pickle=False) as z:
            for name in parts:
                x = z[name].copy()
                assert x.shape[:2] == (48,32) and np.isfinite(x).all()
                if x.ndim == 3 and x.shape[-1] == 1: x = x[...,0]
                parts[name].append(x)
        pins[str(path)], pins[str(audit_path)] = sha(path), sha(audit_path)
    data = {k:np.concatenate(v,axis=0) for k,v in parts.items()}
    commands = data['native_commands'].astype(np.float64)
    done, timeout = data['dones'].astype(bool), data['native_timeouts'].astype(bool)
    assert commands.shape == (2400,32,3) and not np.any(timeout & ~done)
    valid_previous = np.ones(done.shape,dtype=bool)
    valid_previous[0] = False
    valid_previous[1:] = ~done[:-1]
    changed = np.zeros(done.shape,dtype=bool)
    changed[1:] = (np.abs(np.diff(commands,axis=0)) > 1.e-6).any(-1)
    reset_age = ~valid_previous | changed | done
    age = np.zeros(done.shape,dtype=np.int64)
    for t in range(1,2400): age[t] = np.where(reset_age[t],0,age[t-1]+1)
    advantage = data['advantages'].astype(np.float64)
    residual = data['returns'].astype(np.float64)-data['values'].astype(np.float64)
    speed_error = data['native_speed'].astype(np.float64)-commands[...,0]
    yaw_error = data['native_yaw'].astype(np.float64)-commands[...,1]
    height_error = data['native_height'].astype(np.float64)-commands[...,2]
    dedicated = np.broadcast_to(np.asarray(ids)[None,:] < plan['fixed_standing_environments'],done.shape)

    def summarize(mask,population):
        count = int(mask.sum())
        if not count: return dict(samples=0)
        total = int(population.sum())
        absolute_total = float(np.abs(advantage[population]).sum())
        squared_total = float(np.square(advantage[population]).sum())
        return dict(samples=count,sample_fraction=count/total,
            absolute_advantage_mass_share=float(np.abs(advantage[mask]).sum())/absolute_total,
            squared_advantage_mass_share=float(np.square(advantage[mask]).sum())/squared_total,
            normalized_advantage_abs_mean=float(np.abs(advantage[mask]).mean()),
            normalized_advantage_RMS=float(np.sqrt(np.square(advantage[mask]).mean())),
            critic_return_residual_RMS=float(np.sqrt(np.square(residual[mask]).mean())),
            speed_signed_bias_m_s=float(speed_error[mask].mean()),
            speed_MAE_m_s=float(np.abs(speed_error[mask]).mean()),
            yaw_MAE_rad_s=float(np.abs(yaw_error[mask]).mean()),
            height_MAE_m=float(np.abs(height_error[mask]).mean()),
            observed_episode_end_count=int(done[mask].sum()),
            observed_timeout_count=int(timeout[mask].sum()),
            observed_unsafe_end_count=int((done & ~timeout)[mask].sum()))
    all_records = np.ones(done.shape,dtype=bool)
    half_second = age >= 25
    reference = read(previous_cohorts)
    assert reference['verified'] and reference['checkpoint_sha256'] == proof['checkpoint_sha256']
    ref = reference['groups']['command_observed_unchanged_at_least_0p5s']
    half = summarize(half_second,all_records)
    assert half['samples'] == ref['samples']
    assert abs(half['speed_signed_bias_m_s']-ref['speed_signed_bias_m_s']) < 1.e-12
    assert abs(half['critic_return_residual_RMS']-ref['critic_return_residual_RMS']) < 1.e-12
    bounds = [0,25,50,100,150,200,250]
    labels = ['lt0p5s','0p5to1s','1to2s','2to3s','3to4s','4to5s','at_least5s']
    bands = {label:(age>=lo)&((age<bounds[i+1]) if i+1<len(bounds) else True)
             for i,(label,lo) in enumerate(zip(labels,bounds))}
    assert np.all(sum(mask.astype(np.uint8) for mask in bands.values()) == 1)
    populations = {'all_cached_records':all_records,'dedicated_standing':dedicated,
                   'other_continuous_command_roles':~dedicated}
    reports = {}
    for name,population in populations.items():
        table = {key:summarize(mask & population,population) for key,mask in bands.items()}
        assert sum(v['samples'] for v in table.values()) == int(population.sum())
        reports[name] = dict(total=summarize(population,population),age_bands=table,
            observed_unchanged_at_least={str(seconds)+'s':summarize((age>=round(seconds/dt))&population,population)
                for seconds in [.5,1.,2.,3.,4.,5.]})
    result = dict(verified=True,observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=proof['checkpoint_sha256'],actual_updates=proof['actual_ppo_updates'],
        completed_native_updates=50,actual_native_population=8192,cached_environment_ids=ids,
        cached_environments=32,recorded_environment_time_samples=76800,policy_dt=dt,
        existing_global_command_resampling_time_s=cfg['commands']['resampling_time'],
        existing_training_linear_acceleration_limit_m_s2=cfg['commands']['linear_acceleration_limit'],
        original_half_second_cohort_reproduced=True,populations=reports,source_pins=pins,
        limitations=['Command age starts at zero on any observed change in applied vx/yaw/height, current or previous episode end, or the first recorded sample. It does not expose the hidden final ramp target or prove physical steady state.',
            'These are32 cached streams, not all8192. Episode ends and all low-age samples are retained; no training or evaluation samples are discarded.',
            'Absolute/squared advantage mass shares omit the action score and network Jacobian. They are not gradient shares or causal contributions.',
            'Longer holds might change signal allocation but also command coverage and transient practice. These statistics do not demonstrate that a proposed new duration improves tracking.',
            'All age bands and the standing/motion population split are diagnostic only. No corresponding case selector, control branch or sampler is introduced.'],
        optimizer_executed=False,simulator_started=False,policy_evaluated=False,
        training_modified=False,parameter_selected=False,policy_accepted=False)
    assert all(sha(p)==h for p,h in pins.items())
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    motion = reports['other_continuous_command_roles']
    print(json.dumps(dict(verified=True,checkpoint_sha256=proof['checkpoint_sha256'],cached_environments=32,
        motion_samples=motion['total']['samples'],motion_age_at_least2s=motion['observed_unchanged_at_least']['2.0s'],
        motion_age_below0p5s=motion['age_bands']['lt0p5s'],parameter_selected=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-command-cohorts',type=Path,required=True)
    args = parser.parse_args()
    run(args.training_output.resolve(),args.output.resolve(),args.previous_command_cohorts.resolve())
