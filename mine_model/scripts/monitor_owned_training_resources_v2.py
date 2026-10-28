"""Read current resources for one owned native training unit; never control it."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from check_plain27_environment_capacity import resources


def observe(output, unit):
    plan=json.loads((output/'experiment_plan.json').read_text())
    result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState',
        '-p','SubState','-p','Result','-p','ControlGroup','-p','MemoryCurrent','-p','MemoryPeak',
        '-p','MemorySwapCurrent','-p','MemorySwapPeak'],check=True,capture_output=True,text=True)
    state=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    current=resources()
    cgroup=Path('/sys/fs/cgroup'+state['ControlGroup']) if state.get('ControlGroup') else None
    cgroup_values={}
    if cgroup is not None:
        for name in ('memory.current','memory.peak','memory.swap.current','memory.swap.peak'):
            path=cgroup/name
            if path.exists():
                value=path.read_text().strip()
                cgroup_values[name]=int(value) if value.isdigit() else value
        for name in ('memory.events','memory.stat'):
            path=cgroup/name
            if path.exists():
                values={k:int(v) for k,v in (line.split() for line in path.read_text().splitlines())}
                cgroup_values[name]=values if name=='memory.events' else {k:v for k,v in values.items()
                    if k in ('anon','file','kernel','shmem','inactive_file','active_file','inactive_anon','active_anon')}
    vmstat={k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines())}
    current.update(system_swap_pages_in=vmstat['pswpin'],system_swap_pages_out=vmstat['pswpout'])
    proofs=sorted((output/'native_learning_signal').glob('audit_update_*.json'))
    record=dict(observed_at=datetime.now().astimezone().isoformat(),output=str(output),unit=unit,
        planned_num_envs=plan['native_num_envs'],source_sha256=plan['source_sha256'],
        source_actual_updates=plan['source_actual_ppo_updates'],committed_native_updates=len(proofs),
        current_resources=current,unit_state=state,cgroup_memory=cgroup_values,training_modified=False,policy_accepted=False)
    if proofs:
        last=json.loads(proofs[-1].read_text())
        record.update(last_verified_native_environment_count=last['total_environments'],
            last_alignment_max_errors=last['all_environment_alignment_max_errors'],
            last_native_observer_RNG_unchanged=last['probe_RNG_unchanged'])
        if last['total_environments']!=plan['native_num_envs'] or any(record['last_alignment_max_errors'].values()):
            record['agent_inspection_required']='Native population or alignment mismatch'
    if current['gpu_free_mib']<1024 or current['host_available_bytes']<512*1024**2:
        record['agent_inspection_required']='Current host/GPU reserve is low; inspect actual workload'
    return record


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True);cli.add_argument('--unit',required=True);cli.add_argument('--once',action='store_true');args=cli.parse_args()
    output=args.output.resolve()
    while True:
        record=observe(output,args.unit)
        with (output/'owned_training_resources_v2.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        path=output/'owned_training_resources_latest_v2.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if args.once or record['unit_state'].get('MainPID')=='0':break
        time.sleep(15.)
