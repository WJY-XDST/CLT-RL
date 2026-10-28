"""Exact-source capacity checks for the selected AF7001 leg005 trial.

Retain v5's five real PPO updates, full48 rollout, all17 Adam, original
gradient/alignment/save checks and serial native slot. Extend only source
count metadata and read actual cgroup peak/swap values when systemd omits
them. Capacity weights stay isolated from production selection.
"""
import json
from pathlib import Path

import check_plain27_environment_capacity_v5 as retained
import check_plain27_environment_capacity as original
import joint_pd_AF7001_leg005_capacity_runtime as backend

SCRIPT=Path(__file__).resolve()
_unit_state=original.unit_state


def unit_state(unit):
    state=_unit_state(unit)
    if state.get('ControlGroup'):
        root=Path('/sys/fs/cgroup'+state['ControlGroup'])
        for key,name in [('MemoryCurrent','memory.current'),('MemoryPeak','memory.peak'),
                ('MemorySwapCurrent','memory.swap.current'),('MemorySwapPeak','memory.swap.peak')]:
            path=root/name
            if path.exists():
                state[key]=path.read_text().strip()
        events=root/'memory.events'
        if events.exists():
            state['cgroup_memory_events']={k:int(v) for k,v in
                (line.split() for line in events.read_text().splitlines())}
    return state


def probe_command(request, request_path, count, folder):
    argv=list(request['prototype_command'])
    if (argv[:3] != [str(original.ROOT/'run_python.sh'),request['production_runtime'],'native']
            or argv[argv.index('--checkpoint_path')+1] != request['source_checkpoint']
            or request['source_num_envs'] not in (4096,8192) or count not in (4096,8192)
            or request['probe_updates'] != 5
            or any(x in argv for x in ('--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint'))):
        raise ValueError('Use the exact reviewed source and unmodified ordinary production command')
    args=list(argv[argv.index('--')+1:])
    # The production count flag is replaced by the capacity-only count flag;
    # no controller, reward, precision or exploration option is normalized.
    if backend.production.FLAG in args:
        if request['source_num_envs'] != 4096 or args[args.index('--num_envs')+1] != '8192':
            raise ValueError('Requalify the declared original4096-to8192 prototype')
        args.remove(backend.production.FLAG)
    baseline=list(args)
    values={'--target_iteration':request['source_actual_updates']+5,'--num_envs':count,
        '--run_name':'capacity_only_'+Path(folder).name,'--stop_file':Path(folder)/'CAPACITY_ONLY_STOP'}
    for flag,value in values.items():
        if args.count(flag)!=1:raise ValueError('Require one original native option: '+flag)
        args[args.index(flag)+1]=str(value)
    restored=list(args)
    for flag in values:restored[restored.index(flag)+1]=baseline[baseline.index(flag)+1]
    if restored!=baseline:raise ValueError('Original capacity recipe changed outside counts/output clocks')
    if count!=request['source_num_envs']:args.append(backend.FLAG)
    return [str(original.ROOT/'run_python.sh'),str(backend.SCRIPT),'--capacity-request',str(request_path),
        '--observation-output',str(Path(folder)/'native_learning_signal'),'--capacity-num-envs',str(count),'--',*args]


original.probe_command=probe_command
original.unit_state=unit_state
original.BACKEND=backend.SCRIPT
original.SCRIPT=SCRIPT
require_capacity_selection=original.require_capacity_selection


def main():
    # The inherited entry still validates owned completed reviews, source pins,
    # five finite updates, clean exit and both requested counts before choosing.
    return original.main()


if __name__=='__main__':
    main()
