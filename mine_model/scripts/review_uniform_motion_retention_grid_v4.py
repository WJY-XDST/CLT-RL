"""Retain the full old525 grid while a global command envelope expands.

Only the two declared sampler-configuration fields may differ in fixed-command
replay metadata. Original numerical scoring and every failed case stay intact.
The separate new-envelope189 review is required before this replay starts.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import types

import review_uniform_recorded_stage_grid as original


SCRIPT=Path(__file__).resolve()


def qualify(args):
    import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
    from joint_pd_continuation import sha
    previous=original.SCRIPT
    original.SCRIPT=SCRIPT
    try:original.qualify(args)
    finally:original.SCRIPT=previous
    path=args.output/'qualification.json'
    spec=original.read(path)
    cfg=runtime.baseline()[2]
    spec.update(retention_scope_motion_scale=cfg['commands']['training_motion_scale'],
        candidate_training_motion_scale=runtime.recipe()['command_overrides']['training_motion_scale'],
        old525_is_retention_not_full_new_stage_acceptance=True,
        new_envelope189_marker=str(Path(spec['training'])/'independent_review/complete_stage_review_with_global_envelope189.json'))
    spec['source_pins'][str(SCRIPT)]=sha(SCRIPT)
    spec['source_pins'][str(Path(original.__file__).resolve())]=sha(original.__file__)
    path.write_text(json.dumps(spec,indent=2)+'\n')


def run(output):
    output=output.resolve()
    spec=original.read(output/'qualification.json')
    marker=Path(spec['new_envelope189_marker'])
    while not marker.exists():
        result=subprocess.run(['systemctl','--user','show',spec['after_owned_unit'],
            '-p','MainPID','-p','SubState'],capture_output=True,text=True,check=True)
        state=dict(row.split('=',1) for row in result.stdout.splitlines() if '=' in row)
        assert state['MainPID']!='0',('Inspect owned incomplete review; do not repeat PPO',state)
        time.sleep(15.)
    import numpy as np
    import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
    import compare_uniform_joint_pd_replay_retention as comparator
    from joint_pd_continuation import sha
    completed_new_envelope=original.read(marker)
    assert completed_new_envelope['complete']
    assert completed_new_envelope['new_ceiling_envelope189_summary']['trials']==189
    runtime.configure(spec['recipe'])
    previous_compare,previous_script=comparator.compare,original.SCRIPT

    def compare(source,candidate,scope):
        a,b=[comparator.load(Path(p)) for p in (source,candidate)]
        before,after=a[2],b[2]
        changed={k:[before['commands'].get(k),after['commands'].get(k)]
                 for k in before['commands'].keys()|after['commands'].keys()
                 if before['commands'].get(k)!=after['commands'].get(k)}
        declared=runtime.recipe()['command_changes']
        assert set(changed)<=set(declared)
        assert all(value==declared[key] for key,value in changed.items())
        assert a[1]['command_probe'] and b[1]['command_probe']
        assert a[1]['cases']==b[1]['cases'] and a[1]['seeds']==b[1]['seeds']
        initial=[]
        for folder in (source,candidate):
            with np.load(Path(folder)/'policy_inputs.npz',allow_pickle=False) as z:
                initial.append(z['observations'][0].copy())
        assert np.array_equal(*initial),'Fixed commands and initial observations must remain paired'
        load=comparator.load
        def normalized_load(folder):
            result=list(load(folder))
            if Path(folder).resolve()==Path(candidate).resolve():
                cfg=copy.deepcopy(result[2])
                for key in changed:cfg['commands'][key]=before['commands'][key]
                result[2]=cfg
            return tuple(result)
        qualified=types.FunctionType(previous_compare.__code__,
            dict(previous_compare.__globals__,load=normalized_load),
            previous_compare.__name__,previous_compare.__defaults__,previous_compare.__closure__)
        report,pins=qualified(source,candidate,scope)
        report.update(declared_fixed_command_sampler_metadata_differences=changed,
                      initial27_observations_bitwise_equal=True,
                      original_numeric_comparison_function_code_unchanged=True)
        pins[str(SCRIPT)]=sha(SCRIPT)
        return report,pins

    comparator.compare,original.SCRIPT=compare,SCRIPT
    try:original.run(output)
    finally:comparator.compare,original.SCRIPT=previous_compare,previous_script
    path=output/'complete_paired_current_stage525.json'
    result=original.read(path)
    result.update(old525_is_retention_not_full_new_stage_acceptance=True,
                  retained_source_motion_scale=spec['retention_scope_motion_scale'],
                  candidate_training_motion_scale=spec['candidate_training_motion_scale'],
                  separate_new_envelope189_complete=True)
    result['source_pins'][str(marker)]=sha(marker)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qualify-only',action='store_true')
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--pointer',type=Path)
    parser.add_argument('--source',type=Path,nargs='+')
    parser.add_argument('--after-owned-unit')
    args=parser.parse_args()
    if args.qualify_only:
        assert args.recipe and args.pointer and args.source and args.after_owned_unit
        qualify(args)
    else:run(args.output)
