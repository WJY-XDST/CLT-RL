"""Workstation-only checks against recorded Isaac inputs and training helpers.

No simulator, UART or optimizer is started. NumPy/explicit PD arithmetic
agreement does not establish implicit-PhysX/real-motor dynamics agreement.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wheelleg_deploy.numpy_policy import NumpyPolicy, dump
from wheelleg_deploy.torque_control import TorqueController, gravity_effort


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify(bundle,replay_root,training_root,output):
    torch.set_num_threads(1)
    policy=NumpyPolicy(bundle)
    cfg,m=policy.cfg,policy.metadata
    checkpoint=Path(m["checkpoint"])
    before=sha(checkpoint)
    if before!=m["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint no longer matches bundle")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    dims=m["dimensions"]
    layers=[]
    for i,(ni,no) in enumerate(zip(dims[:-1],dims[1:])):
        layers.append(torch.nn.Linear(ni,no))
        if i<len(dims)-2:
            layers.append(torch.nn.ELU())
    actor=torch.nn.Sequential(*layers)
    actor.load_state_dict({k[6:]:v for k,v in saved["model_state_dict"].items() if k.startswith("actor.")},strict=True)
    actor.eval()
    source=training_root/"wheel_legged_isaaclab/wheel_legged_gym_isaaclab"
    five=load_module("verify_five_bar",source/"five_bar_vmc.py")
    hardware=load_module("verify_hardware",source/"mcu_hardware.py")
    leg=load_module("verify_leg_control",source/"leg_control.py")
    controller=TorqueController(policy)
    counts,records,observations=0,[],[]
    max_recorded,max_torch,max_target,max_action,max_ff=0.,0.,0.,0.,0.
    batch_directories=sorted(p for p in replay_root.iterdir() if (p/"policy_inputs.npz").is_file())
    trials=0
    for folder in batch_directories:
        with np.load(folder/"policy_inputs.npz",allow_pickle=False) as data:
            all_obs,means=data["observations"],data["means"]
            times=np.unique(np.linspace(0,len(all_obs)-1,7,dtype=int))
            obs=all_obs[times].reshape(-1,27)
            expected=means[times].reshape(-1,6)
            trials+=all_obs.shape[1]
        if np.any(np.abs(obs[:,9:13]/cfg["obs_scales"]["dof_pos"])>=cfg["obs_clip"]["theta"]-1e-5):
            raise ValueError("Recorded joint positions are clipped; cannot reconstruct gravity input")
        with torch.inference_mode():
            original=actor(torch.from_numpy(obs)).numpy()
        numpy=np.array([policy(row.tolist()) for row in obs])
        max_recorded=max(max_recorded,float(np.max(np.abs(numpy-expected))))
        max_torch=max(max_torch,float(np.max(np.abs(numpy-original))))
        q=obs[:,9:13]/cfg["obs_scales"]["dof_pos"]+np.asarray(cfg["leg_joint_position_offset"])
        dq=obs[:,13:17]/cfg["obs_scales"]["dof_vel"]
        wheel=obs[:,19:21]/cfg["obs_scales"]["dof_vel"]
        raw=np.concatenate((q,dq,wheel),axis=1)
        outputs=[controller.step(row.tolist(),state.tolist(),infer=True,previous_action=row[21:27].tolist())
                 for row,state in zip(obs,raw)]
        tor_q=torch.from_numpy(q).double()
        length,angle,jacobian,valid=five.five_bar_state(tor_q[:,::2],tor_q[:,1::2],**cfg["five_bar_geometry"])
        if not bool(valid.all()):
            raise ValueError("Recorded state includes a singular five-bar")
        pitch=torch.asin(torch.from_numpy(obs[:,3]).double().clamp(-1.,1.))[:,None]
        force=hardware.gravity_feedforward(angle,pitch,mode=cfg["feedforward_mode"],
                    force=cfg["feedforward_force"],angle_bias=cfg["feedforward_angle_bias"],
                    force_limit=cfg["feedforward_force_limit"])
        b,l=five.five_bar_torques(jacobian,force,torch.zeros_like(force))
        torch_ff=torch.stack((b[:,0],l[:,0],b[:,1],l[:,1]),dim=-1).numpy()
        numpy_ff=np.array([o["gravity_unclipped_effort_nm"] for o in outputs])
        max_ff=max(max_ff,float(np.max(np.abs(numpy_ff-torch_ff))))
        limits=torch.tensor(cfg["leg_joint_position_limits"],dtype=torch.float64).expand(len(obs),4,2)
        action=torch.from_numpy(numpy[:,[0,1,3,4]]).double()
        target,effective=leg.joint_position_targets(action,cfg["leg_joint_position_offset"],
                    cfg["leg_joint_position_scale"],limits,cfg["leg_joint_target_limit_margin"])
        max_target=max(max_target,float(np.max(np.abs(np.array([o["position_target_rad"] for o in outputs])-target.numpy()))))
        max_action=max(max_action,float(np.max(np.abs(np.array([o["effective_action"] for o in outputs])[:,[0,1,3,4]]-effective.numpy()))))
        observations.append(obs)
        counts+=len(obs)
        records.append(dict(batch=folder.name,trials=all_obs.shape[1],sampled_states=len(obs),
                            policy_inputs_sha256=sha(folder/"policy_inputs.npz")))
    if trials!=735 or len(records)!=12:
        raise ValueError("Expected all 735 original trials across 12 batches")
    if max_recorded>2e-5 or max_torch>2e-5 or max_ff>1e-8 or max_target>1e-12 or max_action>1e-12:
        raise ValueError("Recorded actor/target/feedforward comparison failed")
    # Independent sampled valid motor states also cover +/-0.5 rad pitch.
    generator=np.random.default_rng(20261027)
    random_q=np.asarray(cfg["leg_joint_position_offset"])+generator.uniform(-.3,.3,(2048,4))
    pitch=generator.uniform(-.5,.5,2048)
    q=torch.from_numpy(random_q).double()
    length,angle,jacobian,valid=five.five_bar_state(q[:,::2],q[:,1::2],**cfg["five_bar_geometry"])
    accepted=valid.all(-1).numpy()
    force=hardware.gravity_feedforward(angle,torch.from_numpy(pitch).double()[:,None],
                mode=cfg["feedforward_mode"],force=cfg["feedforward_force"],
                angle_bias=cfg["feedforward_angle_bias"],force_limit=cfg["feedforward_force_limit"])
    b,l=five.five_bar_torques(jacobian,force,torch.zeros_like(force))
    expected=torch.stack((b[:,0],l[:,0],b[:,1],l[:,1]),dim=-1).numpy()
    actual=np.array([gravity_effort(q.tolist(),float(p),cfg)
                     for q,p in zip(random_q[accepted],pitch[accepted])])
    random_error=float(np.max(np.abs(actual-expected[accepted])))
    if random_error>1e-8:
        raise ValueError("Independent gravity comparison failed")
    steps=sorted(float(state["step"]) for state in saved["optimizer_state_dict"]["state"].values())
    after=sha(checkpoint)
    if before!=after:
        raise ValueError("Source checkpoint changed during read-only verification")
    result=dict(verified=True,checkpoint_sha256=before,contract_sha256=m["contract_sha256"],
        source_actual_updates=m["actual_updates"],original_trials=trials,batches=len(records),
        recorded_states=counts,recorded_actor_max_abs_error=max_recorded,torch_cpu_actor_max_abs_error=max_torch,
        target_rad_max_abs_error=max_target,effective_action_max_abs_error=max_action,
        recorded_gravity_nm_max_abs_error=max_ff,random_gravity_states=int(accepted.sum()),
        random_invalid_geometry_states=int((~accepted).sum()),random_gravity_nm_max_abs_error=random_error,
        optimizer_parameter_states=len(steps),optimizer_step_min=min(steps),optimizer_step_max=max(steps),
        checkpoint_or_optimizer_modified=False,ports_opened=False,simulator_started=False,
        explicit_vs_implicit_dynamics_validated=False,real_hardware_validated=False,
        velocity_reconstruction_note="Recorded dq may be clipped; raw unbounded PD feedback is independently unit-tested.",
        records=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    dump(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--replay-root",type=Path,required=True)
    parser.add_argument("--training-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    verify(args.bundle,args.replay_root,args.training_root,args.output)
