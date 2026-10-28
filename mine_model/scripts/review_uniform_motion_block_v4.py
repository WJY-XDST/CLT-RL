"""Original four reviews plus a global189 envelope review at the new ceiling."""
import argparse
import json
from pathlib import Path

import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
import review_uniform_control_joint_pd_block_v3 as original
import probe_uniform_macro_motion_envelope as macro


SCRIPT = Path(__file__).resolve()


def run(output, recipe):
    original.run(output, recipe)
    final = macro.read(output/"final_checkpoint_verification_readonly.json")
    checkpoint = Path(final["checkpoint"])
    _, cfg = runtime.require_source(checkpoint)
    grid = macro.read(Path(runtime.recipe()["source_retention_grid"])/"complete_paired_current_stage525.json")
    heights = sorted({row["case"]["height"] for row in grid["all525_pairs"]})
    scale = cfg["commands"]["training_motion_scale"]
    v = max(abs(x) for x in cfg["commands"]["ranges_lin_vel_x"])*scale
    w = max(abs(x) for x in cfg["commands"]["ranges_ang_vel_yaw"])*scale
    roles = (("stand",0.,0.),("forward",v,0.),("reverse",-v,0.),
             ("spin_left",0.,w),("spin_right",0.,-w),
             ("forward_left",v,w),("forward_right",v,-w),
             ("reverse_left",-v,w),("reverse_right",-v,-w))
    cases = [dict(name=f"envelope_h{i}_{name}",height=h,speed=speed,yaw=yaw)
             for i,h in enumerate(heights) for name,speed,yaw in roles]
    assert len(heights)==7 and len(cases)==63
    directory = output/"independent_review/global_envelope189"
    directory.mkdir(exist_ok=False)
    macro.write(directory/"commands.json",cases)
    jobs=[]
    for index,start in enumerate(range(0,len(cases),35)):
        batch=cases[start:start+35]
        commands=directory/f"batch_{index}_commands.json"
        macro.write(commands,batch)
        destination=directory/f"batch_{index}"
        argv=runtime.replay_command(checkpoint,destination)
        argv[argv.index("--seconds")+1]="20.0"
        point=argv.index("--seeds")+1
        argv[point:point+3]=["883","884","885"]
        argv += ["--command-probe-json",str(commands)]
        jobs.append(dict(name=f"batch_{index}",replay=str(destination),command=argv,
                         command_file=str(commands),case_count=len(batch)))
    pins=[SCRIPT,Path(macro.__file__),Path(recipe),checkpoint,
          output/"independent_review/complete_comparison_agent_decision_required.json",
          directory/"commands.json",*[Path(job["command_file"]) for job in jobs]]
    spec=dict(qualified=True,frozen_probe_only=True,checkpoint_sha256=macro.sha(checkpoint),
              batches=jobs,trial_count=189,command_file=str(directory/"commands.json"),
              source_pins={str(p):macro.sha(p) for p in pins},policy_accepted=False)
    macro.write(directory/"qualification.json",spec)
    macro.run(directory)
    report=macro.read(directory/"complete_frozen_macro_envelope.json")
    macro.write(output/"independent_review/complete_stage_review_with_global_envelope189.json",
                dict(complete=True,original_four_reviews_unchanged=True,
                     new_ceiling_envelope189_summary=report["summary"],
                     global_motion_scale=scale,policy_accepted=False,agent_decision_required=True))
    macro.write(output/"status.json",dict(state="complete200_global_stage_reviewed_agent_decision_required",
                training_active=False,policy_accepted=False,goal_remains_active=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--recipe",type=Path,required=True)
    args=parser.parse_args()
    run(args.output.resolve(),args.recipe.resolve())
