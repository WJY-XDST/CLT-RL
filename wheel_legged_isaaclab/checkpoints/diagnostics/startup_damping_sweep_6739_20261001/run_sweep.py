import csv,json,math,statistics,subprocess,sys
from pathlib import Path
root=Path('/home/aaa/studyRL/src/CLT-RL')
sys.path.insert(0,str(root/'wheel_legged_isaaclab/scripts/rsl_rl'))
from analyze_response import settling_time
from optimize_training import write_json,source_hashes,sha256
base=root/'IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/evaluation_height_v2/yaw_grouped_recovery_20261001_190548'
out=Path(__file__).resolve().parent
source=json.loads((base/'heading_feedback_gui/command.json').read_text())
checkpoint=source[source.index('--checkpoint_path')+1]
expected=source_hashes()
report={'checkpoint':checkpoint,'checkpoint_sha256':sha256(Path(checkpoint)),'source_hashes':expected,'trials':[]}
def analyze(path):
    with path.open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    if len(rows)!=250 or any(r['step']!=i or abs(r['sim_time_s']-i*.02)>1e-6 for i,r in enumerate(rows)): raise ValueError('Incomplete startup trial')
    if any(not math.isfinite(v) for r in rows for v in r.values()): raise ValueError('Nonfinite trial')
    mean=[math.degrees((r['theta_left']+r['theta_right'])/2) for r in rows]
    steady=statistics.mean(mean[-100:]);centered=[a-steady for a in mean]
    difference=[math.degrees(abs(r['theta_left']-r['theta_right'])) for r in rows]
    return {'resets':sum(r['terminated']+r['time_out'] for r in rows),'common_angle_peak_deg':max(map(abs,mean)),
            'common_settle_half_deg_s':settling_time([abs(v) for v in centered],.5),
            'tail_rms_deg':math.sqrt(statistics.mean(v*v for v in centered[15:100])),
            'variation_first_2s_deg':sum(abs(mean[i]-mean[i-1]) for i in range(1,100)),
            'symmetry_peak_deg':max(difference),
            'body_peak_deg':max(math.degrees(max(abs(r['pitch_est_rad']),abs(r['roll_est_rad']))) for r in rows),
            'speed_peak_m_s':max(math.hypot(r['vel_x_heading'],r['vel_y_heading']) for r in rows),
            'speed_final_p95_m_s':sorted(math.hypot(r['vel_x_heading'],r['vel_y_heading']) for r in rows[-100:])[94]}
for perturb,baseline in [(0.,base/'leg_startup_test/zero_velocity.csv'),(.5,base/'heading_feedback_gui/seed53.csv')]:
    # Baselines use the same policy, seed, fixed command, PI mode and gains.
    # The full GUI trace's first 250 samples are the initial static segment.
    label=f'kd3_reset{perturb:g}'
    baseline_rows=list(csv.DictReader(baseline.open()))[:250]
    trace=out/f'{label}.csv'
    with trace.open('w') as f:
        w=csv.DictWriter(f,fieldnames=baseline_rows[0].keys());w.writeheader();w.writerows(baseline_rows)
    report['trials'].append(dict(kd=3.,reset_velocity=perturb,trace=str(trace),metrics=analyze(trace),reused=True))
    for kd in (4.5,6.):
        if source_hashes()!=expected:raise RuntimeError('Sources changed during sweep')
        label=f'kd{kd:g}_reset{perturb:g}';trace=out/f'{label}.csv'
        command=[str(root/'play_wheel.sh'),'--headless','--device','cuda:0','--num_envs','1','--seed','53','--checkpoint_path',checkpoint,'--fixed_command','0','0','.18','--max_steps','250','--trace_csv',str(trace),'--heading_feedback','env.episode_length_s=1000.0','env.robot.init_state.pos=[0.0,0.0,0.18]',f'env.reset_velocity_initial={perturb}',f'env.reset_velocity_final={perturb}',f'env.kd_theta={kd}']
        write_json(out/f'{label}_command.json',command)
        write_json(out/'state.json',{'state':'testing','case':label,'completed':len(report['trials'])})
        with (out/f'{label}.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError(f'Trial failed: {label}')
        trial=dict(kd=kd,reset_velocity=perturb,trace=str(trace),metrics=analyze(trace),reused=False)
        report['trials'].append(trial);write_json(out/'assessment.json',report);print(json.dumps(trial),flush=True)
write_json(out/'assessment.json',report)
write_json(out/'state.json',{'state':'complete','cases':len(report['trials'])})
