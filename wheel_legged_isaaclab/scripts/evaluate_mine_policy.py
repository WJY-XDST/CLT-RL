"""Evaluate an unchanged trained actor on flat-ground legacy or CAD robots.

All telemetry is captured before automatic resets. Each condition uses three
independent reset-velocity streams. No training, gain tuning or policy export.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher
ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/2026-10-01_21-06-47_autotune_r0002_20261001_210642/model_7738.pt"
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint",type=Path,default=DEFAULT)
parser.add_argument("--robot",choices=["mine","legacy"],default="mine")
parser.add_argument("--seconds",type=float,default=8.)
parser.add_argument("--settle-seconds",type=float,default=0.,help="Standing interval before each episode's motion ramp")
parser.add_argument("--heading-hold",action="store_true",help="Optional world-heading feedback; default replays the requested constant yaw-rate command")
parser.add_argument("--command-scale",type=float,default=1.,help="Diagnostic curriculum command amplitude; acceptance uses 1.0")
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--seeds", type=int, nargs=3, default=[43,44,45])
parser.add_argument("--overrides_json",type=Path,default=None)
parser.add_argument("--export_dir",type=Path,default=None)
parser.add_argument("--record_policy_inputs",action="store_true",help="Save exact inference inputs and raw means for offline feedback diagnosis")
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app


def main():
    import numpy as np
    import torch
    import yaml
    from rsl_rl.modules import ActorCritic
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.mine_env_cfg import MineWheelLeggedVMCFlatEnvCfg,MANIFEST
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv
    from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
    from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain

    height=.30 if args.robot=="mine" else .18
    cases=[("stand",0.,0.,height),("forward_0p3",.3,0.,height),
           ("forward_0p6",.6,0.,height),("reverse_0p3",-.3,0.,height),
           ("reverse_0p6",-.6,0.,height),("spin_left",0.,.3,height),
           ("spin_right",0.,-.3,height),("height_low",0.,0.,height-.02),
           ("height_high",0.,0.,height+.02)]
    if not 0.<args.command_scale<=1.:
        raise ValueError('Command scale must be in (0, 1].')
    cases=[(name,speed*args.command_scale,yaw*args.command_scale,
            height+(target_height-height)*args.command_scale) for name,speed,yaw,target_height in cases]
    seeds=args.seeds
    cfg=(MineWheelLeggedVMCFlatEnvCfg if args.robot=="mine" else WheelLeggedVMCFlatEnvCfg)()
    if args.overrides_json:
        apply_config_overrides(cfg,json.loads(args.overrides_json.read_text()))
        if cfg.wheel_control_mode == 'implicit_velocity':
            cfg.robot.actuators['wheels'].damping=cfg.wheel_damping
    asset_chain, _ = verify_asset_chain(cfg.robot.spawn.usd_path)
    cfg.scene.num_envs=len(cases)*len(seeds)
    cfg.sim.device=args.device or "cuda:0"
    cfg.seed=43
    cfg.episode_length_s=args.seconds+2
    cfg.reset_velocity_initial=cfg.reset_velocity_final=0.
    cfg.commands.ranges_lin_vel_x=(0.,0.)
    cfg.commands.ranges_height=(height,height)
    cfg.commands.heading_command=False
    cfg.commands.grouped_training=False
    cfg.commands.resampling_time=1000.
    cfg.viewer.eye=(1.,1.,.6)
    cfg.viewer.lookat=(0.,0.,-.08)
    # Commands are per-environment; every seed uses identical initial-velocity
    # draws across all conditions and across the two robot models.
    class EvaluationEnv(WheelLeggedVMCFlatEnv):
        def __init__(self,cfg):
            self.recording=False
            self.reset_counts=np.zeros(cfg.scene.num_envs,dtype=int)
            self.generators=[torch.Generator().manual_seed(seeds[i%3]) for i in range(cfg.scene.num_envs)]
            super().__init__(cfg)
            self.case_commands=torch.tensor([x[1:] for x in cases for _ in seeds],device=self.device)
            self.snapshot=None

        def _reset_idx(self,env_ids):
            if env_ids is None:
                env_ids=self._robot._ALL_INDICES
            super()._reset_idx(env_ids)
            ids=env_ids.tolist() if hasattr(env_ids,"tolist") else list(env_ids)
            vel=torch.stack([(2*torch.rand(6,generator=self.generators[i])-1)*.05 for i in ids]).to(self.device)
            self._robot.write_root_velocity_to_sim(vel,env_ids)
            self._commands[env_ids,2]=self.case_commands[env_ids,2]
            if self.recording:
                self.reset_counts[ids]+=1

        def _update_commands(self):
            pass

        def _get_observations(self):
            # Optional standing settling interval; restart the ramp
            # after failure, preserving the same requested test condition.
            moving=(self.episode_length_buf*self.step_dt>=args.settle_seconds)
            self._target_lin_vel_x[:]=torch.where(moving,self.case_commands[:,0],0.)
            self._commands[:,2]=self.case_commands[:,2]
            yaw_target=self.case_commands[:,1]
            if args.heading_hold:
                q=self._robot.data.root_quat_w
                yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
                straight=yaw_target==0
                yaw_target=torch.where(straight,(-self.cfg.commands.heading_kp*yaw).clamp(-.5,.5),yaw_target)
            self._commands[:,1]=torch.where(moving,yaw_target,0.)
            return super()._get_observations()

        def _get_dones(self):
            self._update_forward_kinematics()
            died,timeout=super()._get_dones()
            if self.recording:
                d=self._robot.data
                g=d.projected_gravity_b
                pitch=torch.asin(g[:,0].clamp(-1,1))
                roll=torch.atan2(-g[:,1],-g[:,2])
                q=d.root_quat_w
                yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
                vel=self._get_heading_frame_horizontal_velocity()
                torques=d.applied_torque[:,self._torque_joint_ids]
                saturation=(torques.abs()>=self._torque_limits*.99).float().mean(-1)
                fields=[self.episode_length_buf*self.step_dt,self._commands[:,0],self._commands[:,1],self._commands[:,2],
                    vel[:,0],vel[:,1],d.root_ang_vel_b[:,2],d.root_pos_w[:,2],pitch,roll,yaw,
                    self._L0[:,0],self._L0[:,1],self._l0_ref_applied[:,0],self._l0_ref_applied[:,1],
                    self._theta0[:,0],self._theta0[:,1],self._actions[:,0]*self.cfg.action_scale_theta,self._actions[:,3]*self.cfg.action_scale_theta,
                    self._get_wheel_vel_forward_positive()[:,0],self._get_wheel_vel_forward_positive()[:,1],
                    self._actions[:,2]*self.cfg.action_scale_vel,self._actions[:,5]*self.cfg.action_scale_vel,
                    torques.abs().max(-1).values,saturation,(self._raw_actions.abs()>1).float().mean(-1),
                    died.float(),timeout.float()]
                fields.extend(self._termination_reasons[key].float() for key in REASONS)
                self.snapshot=torch.stack(fields,-1).cpu().numpy()
                self.snapshot_episodes=self.reset_counts.copy()
            return died,timeout

    REASONS=["base_contact","fallen","pitch","roll","leg_contact","numerical","kinematic_singularity","low_height"]
    columns=["episode_time_s","speed_cmd_m_s","yaw_cmd_rad_s","height_cmd_m","speed_m_s","lateral_speed_m_s","yaw_rate_rad_s","height_m","pitch_rad","roll_rad","yaw_rad",
        "left_length_m","right_length_m","left_length_ref_m","right_length_ref_m","left_angle_rad","right_angle_rad","left_angle_ref_rad","right_angle_ref_rad",
        "left_wheel_rad_s","right_wheel_rad_s","left_wheel_ref_rad_s","right_wheel_ref_rad_s","peak_torque_nm","torque_saturation_fraction","action_clipping_fraction","terminated","timeout"]+["fail_"+r for r in REASONS]
    env=EvaluationEnv(cfg)
    obs,_=env.reset()
    agent=yaml.safe_load((args.checkpoint.parent/"params/agent.yaml").read_text())
    policy_cfg=dict(agent["policy"]);policy_cfg.pop("class_name")
    actor=ActorCritic(obs,agent["obs_groups"],6,**policy_cfg).to(env.device)
    checkpoint=torch.load(args.checkpoint,map_location=env.device,weights_only=True)
    actor.load_state_dict(checkpoint["model_state_dict"],strict=True)
    actor.eval()
    assert obs["policy"].shape==(len(cases)*3,27)
    assert actor.act_inference(obs).shape==(len(cases)*3,6)
    # Cross-check native RSL deterministic inference against the checkpoint actor.
    torch.testing.assert_close(actor.act_inference(obs),actor.actor(obs["policy"]))
    if args.export_dir:
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
        args.export_dir.mkdir(parents=True,exist_ok=True)
        export_policy_as_jit(actor,normalizer=actor.actor_obs_normalizer,path=str(args.export_dir),filename='policy.pt')
        export_policy_as_onnx(actor,path=str(args.export_dir),normalizer=actor.actor_obs_normalizer,filename='policy.onnx')
        exported=torch.jit.load(str(args.export_dir/'policy.pt'),map_location=env.device)
        torch.testing.assert_close(exported(obs['policy']),actor.act_inference(obs))
        contract={"observations":27,"actions":6,"action_order":["LB_angle","left_length","LW_speed","RB_angle","right_length","RW_speed"],"motor_order":["LB_joint","LL_joint","LW_joint","RB_joint","RL_joint","RW_joint"],"base_axes":"X forward, Y left, Z up","angle_reference":"base_link, CAD assembly q=0","policy_dt":env.step_dt,"physics_dt":cfg.sim.dt,"wheel_control_mode":cfg.wheel_control_mode,"wheel_damping":cfg.wheel_damping,"five_bar":cfg.five_bar_geometry,"runtime_config":cfg.to_dict(),"checkpoint":str(args.checkpoint.resolve()),"transfer_validation":"Isaac Lab replay only; new-model MuJoCo replay still required"}
        contract['observation_blocks']=[
            {'slice':[0,3],'name':'body angular velocity','scale':cfg.obs_scales.ang_vel},
            {'slice':[3,6],'name':'projected gravity','scale':1.},
            {'slice':[6,9],'name':'applied forward speed, yaw rate, root height commands','scale':[cfg.obs_scales.lin_vel,cfg.obs_scales.ang_vel,cfg.obs_scales.height_measurements]},
            {'slice':[9,11],'name':'left/right virtual leg angles','scale':cfg.obs_scales.dof_pos},
            {'slice':[11,13],'name':'left/right virtual leg angular velocities','scale':cfg.obs_scales.dof_vel},
            {'slice':[13,15],'name':'left/right virtual leg lengths','scale':cfg.obs_scales.l0},
            {'slice':[15,17],'name':'left/right virtual leg length velocities','scale':cfg.obs_scales.l0_dot},
            {'slice':[17,19],'name':'yaw-aligned horizontal body velocity X/Y','scale':cfg.obs_scales.lin_vel},
            {'slice':[19,21],'name':'forward-positive left/right wheel velocity','scale':cfg.obs_scales.dof_vel},
            {'slice':[21,27],'name':'previous physically applied normalized actions','scale':1.}]
        contract['policy_output']='Unclamped Gaussian mean; clamp and apply VMC reference blending in the simulator.'
        contract['leg_control_mode']=cfg.leg_control_mode
        contract['leg_actuator_contract']=({'reference':'five_bar_inverse(L_ref, theta_ref) in raw CAD motor axes',
            'drive_type':'force','stiffness_nm_per_rad':cfg.leg_joint_stiffness,
            'damping_nm_s_per_rad':cfg.leg_joint_damping,'effort_limit_nm':cfg.leg_effort_limit,
            'explicit_leg_effort':0.,'gravity_feedforward_applied':False}
            if cfg.leg_control_mode=='implicit_joint_reference' else
            {'reference':'J.transpose @ [radial PD + gravity feedforward, angular PD]',
             'effort_limit_nm':cfg.leg_effort_limit})
        contract['export_sha256']={name:hashlib.sha256((args.export_dir/name).read_bytes()).hexdigest() for name in ('policy.pt','policy.onnx')}
        contract['asset_chain_sha256'] = asset_chain
        (args.export_dir/'sim2sim_contract.json').write_text(json.dumps(contract,indent=2,default=str)+'\n')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"resolved_env.json").open("w") as f:
        json.dump(cfg.to_dict(),f,indent=2,default=str)
    manifest={"robot":args.robot,"checkpoint":str(args.checkpoint.resolve()),"checkpoint_iteration":checkpoint["iter"],
        "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "asset_version":MANIFEST["version"] if args.robot=="mine" else "legacy_wl",
        "seconds_per_case":args.seconds,"cases":[dict(name=x[0],speed=x[1],yaw=x[2],height=x[3]) for x in cases],
        "seeds":seeds,"reset_velocity_uniform_amplitude":.05,"physics_dt":cfg.sim.dt,"policy_dt":env.step_dt,
        "settle_seconds":args.settle_seconds,"speed_ramp_m_s2":.8,
        "command_scale":args.command_scale,
        "heading":("straight motion holds initial world yaw; spin uses constant yaw rate" if args.heading_hold
                   else "constant requested yaw rate; no added heading feedback"),
        "telemetry":"post-physics, BEFORE automatic reset; errors are actual minus applied reference",
        "weights_updated":False,"policy_input_remapping":False,"leg_control_mode":cfg.leg_control_mode,
        "torque_telemetry":"Isaac Lab actuator estimate; implicit Drive actual torques are not directly reported",
        "control_config":"model-specific virtual-leg references and selected actuator execution; no retuning during evaluation"}
    if args.robot=='mine':
        manifest['asset_sha256']=hashlib.sha256(Path(cfg.robot.spawn.usd_path).read_bytes()).hexdigest()
        manifest['asset_chain_sha256'] = asset_chain
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    env.recording=True
    policy_inputs=[]
    policy_means=[]
    with (args.output/"trace.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["time_s","case_id","seed","episode"]+columns)
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                actions=actor.act_inference(obs)
                if args.record_policy_inputs:
                    policy_inputs.append(obs['policy'].cpu().numpy().copy())
                    policy_means.append(actions.cpu().numpy().copy())
                obs,_,_,_,_=env.step(actions)
                for eid,row in enumerate(env.snapshot):
                    writer.writerow([(step+1)*env.step_dt,eid//3,seeds[eid%3],int(env.snapshot_episodes[eid]),*row.tolist()])
                if step%100==0:
                    f.flush()
                    print(f"EVAL_PROGRESS robot={args.robot} t={(step+1)*env.step_dt:.2f} resets={env.reset_counts.sum()}",flush=True)
    if args.record_policy_inputs:
        np.savez_compressed(args.output/'policy_inputs.npz',observations=np.stack(policy_inputs),
                            means=np.stack(policy_means),seeds=seeds,policy_dt=env.step_dt)
    final={"completed":True,"samples":round(args.seconds/env.step_dt)*len(cases)*3,"resets":int(env.reset_counts.sum())}
    (args.output/"completed.json").write_text(json.dumps(final,indent=2)+"\n")
    print("POLICY_EVALUATION_COMPLETE "+json.dumps(final),flush=True)
    env.close()


if __name__=="__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
