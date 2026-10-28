"""Workstation-only closed-loop diagnostic of held NX torques.

Uses the same imported CAD MuJoCo model, frozen Actor and reference mapping.
The reference uses affine simulator servos; torque variants replace all six
drives with direct motors. No UART or hardware is opened, no training occurs.
"""
import argparse
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import math
from collections import deque

import mujoco
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/"wheel_legged_isaaclab/sim2sim"))
import core
from wheelleg_deploy.numpy_policy import NumpyPolicy
from wheelleg_deploy.torque_control import TorqueController


def run_case(xml,cfg,policy,mode,hz,command,seconds,output,warmup_seconds=0.,
             sensor_delay_steps=0,torque_delay_steps=0):
    c=copy.deepcopy(cfg)
    model=mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep=c["sim"]["dt"]/50.
    simulation=core.Simulation(c,model)
    simulation.command=np.array([0.,command[1],command[2]])
    for index in range(round(warmup_seconds/.005)):
        if index%4==0:
            delta=c["commands"]["linear_acceleration_limit"]*.02
            simulation.command[0]+=np.clip(command[0]-simulation.command[0],-delta,delta)
            action=policy(simulation.observation().numpy()[0].tolist())
            simulation.set_action(torch.tensor([action],dtype=torch.float32))
        simulation.apply_control()
        mujoco.mj_step(model,simulation.data,nstep=50)
        mujoco.mj_forward(model,simulation.data)
    initial_motor_force=simulation.data.actuator_force.copy()
    torque_mode=mode!="affine_servo_reference"
    direct_indices=(list(range(6)) if mode=="held_torque" else
                    [2,5] if mode=="wheel_torque_only" else [0,1,3,4])
    if torque_mode:
        model.actuator_gainprm[direct_indices]=0.;model.actuator_gainprm[direct_indices,0]=1.
        model.actuator_biasprm[direct_indices]=0.
        model.actuator_biastype[direct_indices]=mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gaintype[direct_indices]=mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_ctrllimited[direct_indices]=False
    controller=TorqueController(policy)
    if warmup_seconds>0:
        controller.effective_action=simulation.actions[0].numpy().tolist()
        controller.position_target=simulation.joint_ref.tolist()
        controller.wheel_target=(simulation.actions[0,(2,5)].numpy()*c["action_scale_vel"]).tolist()
    raw_columns=np.array([0,1,3,4])
    period=1./hz
    steps=round(period/model.opt.timestep)
    if abs(steps*model.opt.timestep-period)>1e-10 or hz%50:
        raise ValueError("Probe frequency must align with physics and the 50Hz Actor")
    data=simulation.data
    start_time=float(data.time)
    initial=dict(height_m=float(data.qpos[2]),motor_q=data.qpos[simulation.qids].tolist(),
                 motor_dq=data.qvel[simulation.vids].tolist(),
                 held_initial_actuator_force_raw_Nm=initial_motor_force.tolist(),
                 previous_effective_action=simulation.actions[0].numpy().tolist())
    low_duration=0.
    died=None
    records=[]
    sensor_history=deque(maxlen=sensor_delay_steps+1)
    torque_history=deque(maxlen=torque_delay_steps+1)
    elapsed_steps=round(seconds*hz)
    for index in range(elapsed_steps):
        infer=index%(hz//50)==0
        if infer:
            delta=c["commands"]["linear_acceleration_limit"]*.02
            simulation.command[0]+=np.clip(command[0]-simulation.command[0],-delta,delta)
        current_obs=simulation.observation().numpy()[0].tolist()
        current_raw=np.concatenate((data.qpos[simulation.qids[raw_columns]],
                     data.qvel[simulation.vids[raw_columns]],
                     data.qvel[simulation.vids[[2,5]]]*np.array([1.,-1.]))).tolist()
        if not sensor_history:
            sensor_history.extend([(current_obs,current_raw)]*(sensor_delay_steps+1))
        else:
            sensor_history.append((current_obs,current_raw))
        obs,raw=sensor_history[0]
        obs=list(obs)
        # The host owns the current command ramp; sensor age does not delay a
        # locally generated command, matching NxControlSession's replacement.
        obs[6:9]=(simulation.command*np.array([c["obs_scales"]["lin_vel"],
                    c["obs_scales"]["ang_vel"],c["obs_scales"]["height_measurements"]])).tolist()
        try:
            if torque_mode:
                result=controller.step(obs,raw,infer=infer)
                simulation.actions=torch.tensor([result["effective_action"]],dtype=torch.float32)
                simulation.raw=simulation.actions.clone()
                if mode!="held_torque":
                    simulation.apply_control()
                motor=np.array(result["torque_nm"])*np.array([1.,1.,1.,1.,1.,-1.])
                if not torque_history:
                    torque_history.extend([initial_motor_force.copy() for _ in range(torque_delay_steps)])
                torque_history.append(motor)
                data.ctrl[direct_indices]=torque_history[0][direct_indices]
            else:
                if infer:
                    action=policy(obs)
                    simulation.set_action(torch.tensor([action],dtype=torch.float32))
                simulation.apply_control()
            mujoco.mj_step(model,data,nstep=steps)
            mujoco.mj_forward(model,data)
            angular,gravity,velocity,yaw,pitch,roll=simulation.body_state()
            low_duration=low_duration+period if data.qpos[2]<c["min_root_height"] else 0.
            wheel=data.qvel[simulation.vids[[2,5]]]*[1.,-1.]
            row=dict(time_s=float(data.time-start_time),height_m=float(data.qpos[2]),
                 pitch_rad=float(pitch),roll_rad=float(roll),speed_m_s=float(velocity[0]),
                 yaw_rate_rad_s=float(angular[2]),left_wheel_rad_s=float(wheel[0]),right_wheel_rad_s=float(wheel[1]),
                 action_LW=float(simulation.actions[0,2]),action_RW=float(simulation.actions[0,5]),
                 peak_actual_actuator_force_Nm=float(np.max(np.abs(data.actuator_force))),
                 peak_raw_joint_velocity_rad_s=float(np.max(np.abs(data.qvel[simulation.vids]))))
            records.append(row)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                died="nonfinite_state"
            elif low_duration>=c["low_height_terminal_time_s"]:
                died="sustained_low_support_height"
            elif abs(pitch)>c["max_body_pitch"]:
                died="pitch_exceeds_configured_limit"
            elif abs(roll)>c["max_body_roll"]:
                died="roll_exceeds_configured_limit"
            if died:
                break
        except (ValueError,RuntimeError) as error:
            died="controller_or_simulator_rejected_state: "+str(error)
            break
    if records:
        with (output/"trace.csv").open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    result=dict(mode=mode,control_hz=hz,actor_hz=50,physics_dt=model.opt.timestep,
          actual_wheel_damping=c["wheel_damping"],
          total_mass_kg=float(model.body_mass.sum()),command=command,requested_seconds=seconds,
          simulated_seconds=float(data.time-start_time),warmup_seconds=warmup_seconds,initial_state=initial,
          sensor_delay_ms=sensor_delay_steps*period*1000,torque_delay_ms=torque_delay_steps*period*1000,
          direct_motor_columns=direct_indices if torque_mode else [],survived=died is None,failure=died,
          max_abs_wheel_speed_rad_s=max((max(abs(r["left_wheel_rad_s"]),abs(r["right_wheel_rad_s"])) for r in records),default=None),
          max_abs_pitch_deg=max((abs(r["pitch_rad"])*180/np.pi for r in records),default=None),
          last_state=records[-1] if records else None,tracking_acceptance_claimed=False)
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--environment",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=3.)
    p.add_argument("--warmup-seconds",type=float,default=0.)
    p.add_argument("--wheel-damping-experiment",type=float,
                   help="Uniform diagnostic override only; does not modify or export the source bundle")
    p.add_argument("--sensor-delay-steps",type=int,default=0)
    p.add_argument("--torque-delay-steps",type=int,default=0)
    p.add_argument("--reuse-model",type=Path)
    p.add_argument("--hybrid-controls",action="store_true",
                   help="Diagnostic leg-only and wheel-only torque substitutions; production code unchanged")
    p.add_argument("--frequencies",type=int,nargs="+",default=[200,500,1000])
    p.add_argument("--command",type=float,nargs=3,default=[0.,0.,.3])
    args=p.parse_args()
    if not 0<=args.sensor_delay_steps<=100 or not 0<=args.torque_delay_steps<=100:
        raise ValueError("Diagnostic delay steps must be in 0..100")
    if args.output.exists():
        raise ValueError("Preserve existing probe output")
    torch.set_num_threads(1)
    policy=NumpyPolicy(args.bundle)
    cfg=core.load_config(args.environment)
    for key in ("leg_joint_stiffness","leg_joint_damping","wheel_damping","leg_effort_limit",
                "wheel_effort_limit","wheel_action_limit","feedforward_force"):
        if cfg[key]!=policy.cfg[key]:
            raise ValueError("Probe environment and selected bundle differ: "+key)
    if args.wheel_damping_experiment is not None:
        gain=args.wheel_damping_experiment
        if not math.isfinite(gain) or gain<=0:
            raise ValueError("Experimental wheel damping must be positive and finite")
        class ControllerProbePolicy:
            def __init__(self,original):
                self.original=original
                self.cfg=copy.deepcopy(original.cfg)
                self.cfg["wheel_damping"]=gain
                self.schema=original.schema
                self.metadata=original.metadata
            def __call__(self,obs):
                return self.original(obs)
        policy=ControllerProbePolicy(policy)
        cfg["wheel_damping"]=gain
    args.output.mkdir(parents=True)
    source_bytes=Path(__file__).read_bytes()
    (args.output/"executed_probe_source.py").write_bytes(source_bytes)
    if args.reuse_model is not None:
        xml=args.reuse_model.expanduser().resolve()
        audit=json.loads(xml.with_suffix(".json").read_text())
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if digest(xml)!=audit["mjcf_sha256"] or Path(audit["source_usd"])!=Path(cfg["robot"]["spawn"]["usd_path"]).resolve():
            raise ValueError("Cached converted model differs from the selected source")
        for group in ["mesh_sha256","usd_layers"]:
            if any(digest(path)!=expected for path,expected in audit[group].items()):
                raise ValueError("Cached conversion dependency changed")
    else:
        xml=args.output/"scene.xml"
        core.build_model(cfg,xml)
    results=[]
    modes=["held_torque"]+(["wheel_torque_only","leg_torque_only"] if args.hybrid_controls else [])
    for mode,hz in [("affine_servo_reference",200)]+[(mode,f) for mode in modes for f in args.frequencies]:
        folder=args.output/("%s_%dHz"%(mode,hz));folder.mkdir()
        result=run_case(xml,cfg,policy,mode,hz,args.command,args.seconds,folder,args.warmup_seconds,
                        args.sensor_delay_steps,args.torque_delay_steps)
        results.append(result);print(json.dumps(result),flush=True)
    report=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),complete=True,
         checkpoint_sha256=policy.metadata["checkpoint_sha256"],contract_sha256=policy.metadata["contract_sha256"],
         environment_sha256=hashlib.sha256(args.environment.read_bytes()).hexdigest(),results=results,
         probe_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
         wheel_damping_override=args.wheel_damping_experiment,
         source_bundle_modified=False,experimental_controller_not_exported=True,
         diagnostic_only=True,original_acceptance_scope_not_run=True,policy_accepted=False,
         real_NX_timing_tested=False,serial_opened=False,optimizer_executed=False)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
