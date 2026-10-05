"""Assess one new-model replay without using failed episodes as steady tracking."""
import argparse
import json
from pathlib import Path
import numpy as np

ACCEPTANCE_PATH=Path(__file__).resolve().parents[1]/'config/tracking_acceptance.json'

def acceptance_criteria():
    criteria={'relative_error':.10,'speed_absolute_m_s':.025,'yaw_absolute_rad_s':.025,
              'wheel_absolute_rad_s':.15,'leg_angle_design_range_rad':.2,
              'body_p95_deg':2.5,'minimum_height_fraction':.9}
    if ACCEPTANCE_PATH.exists():criteria.update(json.loads(ACCEPTANCE_PATH.read_text()))
    for key in ('relative_error','speed_absolute_m_s','yaw_absolute_rad_s',
                'wheel_absolute_rad_s','leg_angle_design_range_rad','body_p95_deg','minimum_height_fraction'):
        if not isinstance(criteria[key],(int,float)) or not np.isfinite(criteria[key]) or criteria[key]<=0:
            raise ValueError('Invalid acceptance criterion: '+key)
    if criteria['relative_error']>=1 or criteria['minimum_height_fraction']>1:
        raise ValueError('Invalid relative acceptance bounds')
    return criteria

def assess(directory,report_name='assessment.json'):
    directory=Path(directory)
    criteria=acceptance_criteria();fraction=criteria['relative_error'];percent=100*fraction
    manifest=json.loads((directory/'manifest.json').read_text())
    completed=json.loads((directory/'completed.json').read_text())
    a=np.genfromtxt(directory/'trace.csv',delimiter=',',names=True)
    expected=round(manifest['seconds_per_case']/manifest['policy_dt'])*len(manifest['cases'])*len(manifest['seeds'])
    if not completed['completed'] or len(a)!=expected or not all(np.isfinite(a[k]).all() for k in a.dtype.names):
        raise ValueError('Incomplete or nonfinite replay')
    conditions=[]
    for cid,case in enumerate(manifest['cases']):
        trials=[]
        for seed in manifest['seeds']:
            d=a[(a['case_id']==cid)&(a['seed']==seed)]
            first=d[d['episode']==0]
            failures=first[(first['terminated']>0)|(first['timeout']>0)]
            survived=len(failures)==0 and len(first)==round(manifest['seconds_per_case']/manifest['policy_dt'])
            tail=first[first['time_s']>=manifest['seconds_per_case']-3.]
            metrics=None
            if survived:
                metrics={'speed_mae_m_s':float(np.mean(abs(tail['speed_m_s']-tail['speed_cmd_m_s']))),
                         'yaw_mae_rad_s':float(np.mean(abs(tail['yaw_rate_rad_s']-tail['yaw_cmd_rad_s']))),
                         'height_mae_m':float(np.mean(abs(tail['height_m']-tail['height_cmd_m']))),
                         'leg_length_mae_m':float(max(np.mean(abs(tail[f'{side}_length_m']-tail[f'{side}_length_ref_m'])) for side in ('left','right'))),
                         'leg_angle_mae_rad':float(max(np.mean(abs(tail[f'{side}_angle_rad']-tail[f'{side}_angle_ref_rad'])) for side in ('left','right'))),
                         'wheel_speed_mae_rad_s':float(max(np.mean(abs(tail[f'{side}_wheel_rad_s']-tail[f'{side}_wheel_ref_rad_s'])) for side in ('left','right'))),
                         'body_p95_deg':float(np.percentile(np.maximum(abs(tail['pitch_rad']),abs(tail['roll_rad'])),95)*180/np.pi),
                         'upright_support':bool(np.min(tail['height_m'])>=criteria['minimum_height_fraction']*case['height']),
                         'command_reached':bool(np.max(abs(tail['speed_cmd_m_s']-case['speed']))<.005
                                                and np.max(abs(tail['yaw_cmd_rad_s']-case['yaw']))<.005)}
            reference_samples=tail if len(tail) else first
            leg_reference=max(.001,min(np.mean(abs(reference_samples[f'{side}_length_ref_m'])) for side in ('left','right')))
            wheel_reference=min(np.mean(abs(reference_samples[f'{side}_wheel_ref_rad_s'])) for side in ('left','right'))
            limits={'speed_mae_m_s':max(criteria['speed_absolute_m_s'],fraction*abs(case['speed'])),
                    'yaw_mae_rad_s':max(criteria['yaw_absolute_rad_s'],fraction*abs(case['yaw'])),
                    'height_mae_m':fraction*case['height'],'leg_length_mae_m':fraction*leg_reference,
                    'leg_angle_mae_rad':fraction*criteria['leg_angle_design_range_rad'],
                    'wheel_speed_mae_rad_s':max(criteria['wheel_absolute_rad_s'],fraction*wheel_reference),
                    'body_p95_deg':criteria['body_p95_deg']}
            passed=survived and metrics['command_reached'] and metrics['upright_support'] and all(metrics[k]<=v for k,v in limits.items())
            diagnostics=None
            if survived:
                diagnostics={'wheel_reference_delta_rms_rad_s':float(max(
                    np.sqrt(np.mean(np.diff(tail[f'{side}_wheel_ref_rad_s'])**2)) for side in ('left','right'))),
                    'yaw_signed_mean_error_rad_s':float(np.mean(tail['yaw_rate_rad_s']-tail['yaw_cmd_rad_s']))}
            trials.append({'seed':seed,'survived':survived,'first_failure_s':float(failures['time_s'][0]) if len(failures) else None,
                           'resets':int(d['terminated'].sum()),'metrics':metrics,'limits':limits,'diagnostics':diagnostics,'passed':bool(passed)})
        conditions.append({**case,'trials':trials,'survivors':sum(t['survived'] for t in trials),'passed':all(t['passed'] for t in trials)})
    report={'checkpoint':manifest['checkpoint'],'checkpoint_sha256':manifest['checkpoint_sha256'],'iteration':manifest['checkpoint_iteration'],
            'survivors':sum(c['survivors'] for c in conditions),'total_trials':len(manifest['cases'])*len(manifest['seeds']),
            'resets':int(a['terminated'].sum()),'conditions':conditions,'passed':all(c['passed'] for c in conditions),
            'acceptance_criteria':criteria,
            'criterion':f"{percent:g} percent sim2sim preparation: all conditions and seeds survive without reset; final 3-second MAE; speed max({criteria['speed_absolute_m_s']:g} m/s,{percent:g} percent target); yaw max({criteria['yaw_absolute_rad_s']:g} rad/s,{percent:g} percent target); height/length {percent:g} percent target; leg angle {fraction*criteria['leg_angle_design_range_rad']:g} rad ({percent:g} percent of {criteria['leg_angle_design_range_rad']:g} rad range); wheel velocity max({criteria['wheel_absolute_rad_s']:g} rad/s,{percent:g} percent reference); posture p95 {criteria['body_p95_deg']:g} deg; steady minimum root height at least {100*criteria['minimum_height_fraction']:g} percent of command. Independent seeds required before export."}
    (directory/report_name).write_text(json.dumps(report,indent=2)+'\n')
    return report

def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    a=np.genfromtxt(directory/'trace.csv',delimiter=',',names=True)
    fig,axes=plt.subplots(4,3,figsize=(13,10),sharex=True,layout='constrained')
    for col,cid in enumerate((0,1,5)):
        d=a[(a['case_id']==cid)&(a['seed']==a['seed'][0])]
        t=d['time_s'].copy()
        errors=[d['speed_m_s']-d['speed_cmd_m_s'],1000*(d['height_m']-d['height_cmd_m']),d['yaw_rate_rad_s']-d['yaw_cmd_rad_s'],np.maximum(abs(d['pitch_rad']),abs(d['roll_rad']))*180/np.pi]
        cut=np.flatnonzero(np.diff(d['episode'])!=0)+1
        for row,y in enumerate(errors):
            x=np.insert(t,cut,np.nan);values=np.insert(y,cut,np.nan)
            axes[row,col].plot(x,values,lw=.9);axes[row,col].axhline(0,color='grey',lw=.5)
            axes[row,col].grid(alpha=.2)
        axes[0,col].set_title(('Stand','Forward 0.3 m/s','Spin +0.3 rad/s')[col])
        axes[-1,col].set_xlabel('Time (s)')
    for ax,label in zip(axes[:,0],('Speed error (m/s)','Height error (mm)','Yaw error (rad/s)','Body angle (deg)')):ax.set_ylabel(label)
    fig.savefig(directory/'trial_errors.png',dpi=150)
    plt.close(fig)
    from matplotlib.backends.backend_pdf import PdfPages
    manifest=json.loads((directory/'manifest.json').read_text())
    report=json.loads((directory/'assessment.json').read_text())
    labels=('Speed error (m/s)','Yaw error (rad/s)','Height error (mm)',
            'Leg length error (mm)','Leg angle error (rad)','Wheel speed error (rad/s)','Body angle (deg)')
    keys=('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','leg_length_mae_m',
          'leg_angle_mae_rad','wheel_speed_mae_rad_s','body_p95_deg')
    with PdfPages(directory/'all_conditions_error_curves.pdf') as pdf:
        for cid,case in enumerate(manifest['cases']):
            fig,axes=plt.subplots(7,3,figsize=(12,14),sharex=True,layout='constrained')
            for col,seed in enumerate(manifest['seeds']):
                d=a[(a['case_id']==cid)&(a['seed']==seed)]
                cut=np.flatnonzero(np.diff(d['episode'])!=0)+1
                x=np.insert(d['time_s'],cut,np.nan)
                errors=[(d['speed_m_s']-d['speed_cmd_m_s'],),(d['yaw_rate_rad_s']-d['yaw_cmd_rad_s'],),
                        (1000*(d['height_m']-d['height_cmd_m']),),
                        tuple(1000*(d[f'{side}_length_m']-d[f'{side}_length_ref_m']) for side in ('left','right')),
                        tuple(d[f'{side}_angle_rad']-d[f'{side}_angle_ref_rad'] for side in ('left','right')),
                        tuple(d[f'{side}_wheel_rad_s']-d[f'{side}_wheel_ref_rad_s'] for side in ('left','right')),
                        (abs(d['pitch_rad'])*180/np.pi,abs(d['roll_rad'])*180/np.pi)]
                trial=report['conditions'][cid]['trials'][col]
                for row,series in enumerate(errors):
                    ax=axes[row,col]
                    for i,y in enumerate(series):
                        ax.plot(x,np.insert(y,cut,np.nan),lw=.7,label=('pitch','roll')[i] if row==6 else ('left','right')[i])
                    limit=trial['limits'][keys[row]]*(1000 if row in (2,3) else 1)
                    ax.axhline(limit,color='#777777',ls='--',lw=.6)
                    if row!=6: ax.axhline(-limit,color='#777777',ls='--',lw=.6)
                    if trial['first_failure_s'] is not None:
                        ax.axvline(trial['first_failure_s'],color='#b43b39',lw=.7)
                    ax.grid(alpha=.2)
                    if col==0:ax.set_ylabel(labels[row])
                    if len(series)>1 and col==2:ax.legend(loc='upper right',fontsize=7)
                axes[0,col].set_title(f'Seed {seed}; resets {trial["resets"]}')
                axes[-1,col].set_xlabel('Time (s)')
            fig.suptitle(case['name']+' | dashed: MAE acceptance limits; red: first failure; curves break at resets',fontsize=10)
            pdf.savefig(fig);plt.close(fig)

def plot_optimization_history(output):
    """Keep tracking tolerance ratios and survival separate across branches."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    keys=('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','leg_length_mae_m',
          'leg_angle_mae_rad','wheel_speed_mae_rad_s','body_p95_deg')
    records=[]
    for path in sorted(output.glob('round_*/replay/assessment.json')):
        report=json.loads(path.read_text())
        row={'round':int(path.parents[1].name.split('_')[1]),'iteration':report['iteration'],
             'checkpoint_sha256':report['checkpoint_sha256'],'survivors':report['survivors'],
             'total_trials':report['total_trials'],'passed':report['passed']}
        for key in keys:
            values=[t['metrics'][key]/t['limits'][key] for c in report['conditions']
                    for t in c['trials'] if t['metrics']]
            row[key]=max(values) if values else float('nan')
        records.append(row)
    if not records:return []
    csv_path=output/'optimization_progress.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained',height_ratios=[3,1])
    rounds=[r['round'] for r in records]
    for key,label in zip(keys,['speed','yaw rate','height','leg length','leg angle','wheel speed','posture']):
        axes[0].plot(rounds,[max(.001,r[key]) if np.isfinite(r[key]) else np.nan for r in records],'.-',label=label,lw=1)
    axes[0].axhline(1,color='black',ls='--',lw=1,label='Acceptance limit')
    axes[0].set_yscale('log');axes[0].set_ylabel('Worst-seed error / acceptance limit')
    axes[0].legend(ncol=4,fontsize=8);axes[0].grid(alpha=.2)
    axes[0].set_title('Full-command replay history; failed trials never count as steady precision')
    axes[1].plot(rounds,[r['survivors']/r['total_trials'] for r in records],'.-',color='#28658e')
    axes[1].set_ylim(-.05,1.05);axes[1].set_ylabel('Survived fraction');axes[1].set_xlabel('Experiment round (warm-start iteration change recorded in CSV)')
    axes[1].grid(alpha=.2)
    fig.savefig(output/'optimization_progress.png',dpi=160);plt.close(fig)
    return [csv_path,output/'optimization_progress.png']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--plot',action='store_true');args=p.parse_args()
    r=assess(args.directory)
    if args.plot:plot(args.directory)
    print(json.dumps({k:r[k] for k in ('iteration','survivors','total_trials','resets','passed')},indent=2))
