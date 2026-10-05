"""Exercise the registered project's VMC with deterministic bench references.

No learned policy is used. Fixed elevated base, zero gravity: the test isolates
motor mapping, closed-loop constraints and VMC tracking. --floating instead
checks a short gravity/contact rollout, without claiming balance control.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seconds", type=float, default=12.)
parser.add_argument("--floating", action="store_true")
parser.add_argument("--loop", action="store_true", help="Repeat the visible bench demonstration")
parser.add_argument("--output", type=Path, default=Path("mine_model/results/bench_drive"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    import wheel_legged_gym_isaaclab.tasks
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.mine_env_cfg import MODEL, MANIFEST

    cfg = parse_env_cfg("WheelLeggedVMC-Flat-v0", device=args.device, num_envs=args.num_envs)
    cfg.seed = 37
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.ranges_lin_vel_x = (0.,0.)
    cfg.commands.ranges_height = (.30,.30)
    cfg.height_reference_blend = cfg.height_feedback_gain = 0.
    cfg.episode_length_s = max(20.,args.seconds+1)
    if not args.floating:
        # Preserve the gravity direction used by observations/termination.
        cfg.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        cfg.robot.spawn.articulation_props.fix_root_link = True
        cfg.robot.init_state.pos = (0.,0.,.65)
        cfg.feedforward_force = 0.
        cfg.viewer.eye = (1.0,1.0,.45)
        cfg.viewer.lookat = (0.,0.,-.10)
    env = gym.make("WheelLeggedVMC-Flat-v0",cfg=cfg).unwrapped
    obs,_ = env.reset()
    robot = env._robot
    stage = env.sim.stage
    device = env.device
    # Check relationship remapping in every clone; metrics use live PhysX poses.
    closures = []
    per_env = [0]*args.num_envs
    for prim in stage.Traverse():
        if prim.GetName() not in {x["name"] for x in MODEL["closures"]}:
            continue
        joint = UsdPhysics.Joint(prim)
        path = str(prim.GetPath())
        eid = int(path.split("/env_")[1].split("/")[0])
        if eid >= args.num_envs:
            continue
        per_env[eid] += 1
        frames = []
        for side in (0,1):
            target = getattr(joint,f"GetBody{side}Rel")().GetTargets()[0]
            assert f"/env_{eid}/" in str(target), (path,str(target))
            body_id = robot.body_names.index(target.name)
            pos = torch.tensor(list(getattr(joint,f"GetLocalPos{side}Attr")().Get()),device=device)
            rot = getattr(joint,f"GetLocalRot{side}Attr")().Get()
            q = torch.tensor([rot.GetReal(),*rot.GetImaginary()],device=device)
            axis = quat_apply(q,torch.tensor([0.,0.,1.],device=device))
            frames.append((body_id,pos,axis))
        closures.append((eid,frames))
    assert per_env == [4]*args.num_envs, per_env
    assert obs["policy"].shape == (args.num_envs,27)
    assert len(robot.joint_names) == 14
    mass = robot.root_physx_view.get_generalized_mass_matrices().clone()
    eig = torch.linalg.eigvalsh((mass+mass.transpose(-1,-2))/2)
    assert torch.isfinite(mass).all() and eig.min()>0
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/"tree_mass_matrix.npy",mass.cpu().numpy())
    rows=[]
    peak_gap=peak_axis=peak_fk=peak_effort=0.
    resets=0
    reset_reasons={}
    minimum_root_height=float("inf")
    initial_q=robot.data.joint_pos.clone()
    passive_ids=[i for i,n in enumerate(robot.joint_names) if n not in MODEL["leg_joint_names"]+MODEL["wheel_joint_names"]]
    passive_motion=0.
    capture=None
    status_label=None
    if not args.headless:
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        window=ui.Window("Mine robot: VMC drive test",width=420,height=190)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Fixed-base bench | no balance policy",height=22)
                ui.Label("Length +/-12 mm | angle +/-3.4 deg | wheels +/-3 rad/s",height=22)
                status_label=ui.Label("Starting...",word_wrap=True)
    def measure(reset_mask):
        poses = robot.root_physx_view.get_link_transforms()
        poses=poses.double()
        quat = poses[..., (6,3,4,5)]
        quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True)
        gaps=[]; axes=[]
        for eid,frames in closures:
            values=[]
            for bid,pos,axis in frames:
                values.append((poses[eid,bid,:3]+quat_apply(quat[eid,bid],pos.double()),
                               quat_apply(quat[eid,bid],axis.double())))
            gaps.append(torch.linalg.vector_norm(values[0][0]-values[1][0]))
            dot=(values[0][1]*values[1][1]).sum().abs().clamp(0,1)
            axes.append(torch.acos(dot))
        wheel_ids=[robot.body_names.index(n) for n in ("LW_link","RW_link")]
        # These rigid-body origins coincide with the wheel revolute axes.
        relative=poses[:,wheel_ids,:3]-robot.data.root_pos_w[:,None,:]
        base_q=robot.data.root_quat_w[:,None,:].expand(-1,2,-1).double()
        relative=quat_apply_inverse(base_q,relative)
        actual=relative[..., (0,2)].clone()
        actual[...,1]-=MODEL["hip_z"]
        predicted=torch.stack((-env._L0*env._theta0.sin(),-env._L0*env._theta0.cos()),-1)
        fk=torch.linalg.vector_norm(actual-predicted,dim=-1)
        # env.step resets joint buffers immediately, while the tensor link poses
        # are synchronized on the next physics step. Do not compare those two
        # different instants when an environment has just reset.
        fk=fk[~reset_mask]
        return torch.stack(gaps).max().item()*1000,torch.stack(axes).max().item()*180/math.pi,(fk.max().item()*1000 if fk.numel() else 0.)

    nsteps=round(args.seconds/env.step_dt)
    while True:
        for step in range(nsteps):
            start=time.monotonic()
            t=step*env.step_dt
            # Env 1 follows a different trajectory, detecting incorrect clone coupling.
            phase=torch.arange(args.num_envs,device=device)[:,None]*.7
            legphase=torch.tensor([[0.,.4]],device=device)
            if args.floating:
                lengths=torch.full((args.num_envs,2),MODEL["nominal_leg_length"],device=device)
                angles=torch.zeros_like(lengths)
                speeds=torch.zeros_like(lengths)
            else:
                ramp=min(t/2.,1.)
                lengths=MODEL["nominal_leg_length"]+.012*ramp*torch.sin(2*math.pi*.2*t+phase+legphase)
                angles=.052219456+.06*ramp*torch.sin(2*math.pi*.15*t+phase+legphase)
                speeds=(3.*ramp*torch.sin(2*math.pi*.2*t+phase)).expand(-1,2)
            actions=torch.zeros(args.num_envs,6,device=device)
            actions[:,(0,3)]=angles/cfg.action_scale_theta
            actions[:,(1,4)]=(lengths-cfg.l0_offset)/cfg.action_scale_l0
            actions[:,(2,5)]=speeds/cfg.action_scale_vel
            obs,reward,died,timeout,_=env.step(actions)
            resets+=int((died|timeout).sum().item())
            for name,flag in env._termination_reasons.items():
                reset_reasons[name]=reset_reasons.get(name,0)+int(flag.sum().item())
            env._update_forward_kinematics()
            assert torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.joint_vel).all()
            assert torch.isfinite(obs["policy"]).all() and env._five_bar_valid.all()
            gap,axis,fk=measure(died|timeout)
            minimum_root_height=min(minimum_root_height,robot.data.root_pos_w[:,2].min().item())
            effort=robot.data.applied_torque.abs().max().item()
            peak_gap=max(peak_gap,gap); peak_axis=max(peak_axis,axis); peak_fk=max(peak_fk,fk); peak_effort=max(peak_effort,effort)
            passive_motion=max(passive_motion,(robot.data.joint_pos[:,passive_ids]-initial_q[:,passive_ids]).abs().max().item())
            wheel=env._get_wheel_vel_forward_positive()
            if status_label is not None and step%5==0:
                status_label.text=(f"Time {t:.1f} s | closure gap {gap:.4f} mm\n"
                    f"Length L/R: {env._L0[0,0].item()*1000:.1f} / {env._L0[0,1].item()*1000:.1f} mm\n"
                    f"Angle L/R: {env._theta0[0,0].item()*180/math.pi:.2f} / {env._theta0[0,1].item()*180/math.pi:.2f} deg\n"
                    f"Wheel L/R: {wheel[0,0].item():.2f} / {wheel[0,1].item():.2f} rad/s")
            if not args.headless and step==100:
                capture=capture_viewport_to_file(get_active_viewport(),str(args.output/"viewport.png"))
            for eid in range(args.num_envs):
                for side in range(2):
                    rows.append([t,eid,side,lengths[eid,side].item(),env._L0[eid,side].item(),
                        angles[eid,side].item(),env._theta0[eid,side].item(),speeds[eid,side].item(),wheel[eid,side].item(),gap,fk])
            if not args.headless:
                time.sleep(max(0.,env.step_dt-(time.monotonic()-start)))
        data=np.asarray(rows)
        steady=data[:,0]>=2.
        selected=data[steady] if steady.any() else data
        report={"model_version":MANIFEST["version"],"test":"floating_gravity_smoke" if args.floating else "fixed_base_zero_gravity_vmc",
            "seconds":args.seconds,"num_envs":args.num_envs,"physics_dt":cfg.sim.dt,
            "observations":27,"actions":6,"joint_names":robot.joint_names,"closures_per_env":per_env,
            "peak_closure_gap_mm":peak_gap,"peak_closure_axis_deg":peak_axis,"peak_fk_wheel_error_mm":peak_fk,
            "peak_applied_effort_nm":peak_effort,"passive_motion_rad":passive_motion,"unexpected_resets":resets,"reset_reasons":reset_reasons,
            "minimum_root_height_m":minimum_root_height,
            "balance_validated":False,
            "length_tracking_mae_mm":float(np.abs(selected[:,3]-selected[:,4]).mean()*1000),
            "angle_tracking_mae_deg":float(np.abs(selected[:,5]-selected[:,6]).mean()*180/math.pi),
            "wheel_tracking_mae_rad_s":float(np.abs(selected[:,7]-selected[:,8]).mean()),
            "tree_mass_matrix_shape":list(mass.shape),"tree_mass_matrix_min_eigenvalue":eig.min().item(),
            "note":"PhysX tree mass matrix; external loop constraints are solved separately. This is not a reduced six-actuator mass matrix. No balance controller or policy is used; gravity mode checks mechanical integrity, not standing stability."}
        checks={"closure_gap_below_1mm":peak_gap<1.,"fk_below_1mm":peak_fk<1.,"effort_within_10nm":peak_effort<=10.001}
        if not args.floating:
            checks.update(no_resets=resets==0,passive_motion=passive_motion>.01,
                length_mae_below_3mm=report["length_tracking_mae_mm"]<3.,
                angle_mae_below_1deg=report["angle_tracking_mae_deg"]<1.,
                wheel_mae_below_0_3=report["wheel_tracking_mae_rad_s"]<.3)
        report["checks"]=checks
        report["passed"]=all(checks.values())
        with (args.output/"trace.csv").open("w") as f:
            writer=csv.writer(f);writer.writerow(["time_s","env","side_0_left","length_ref_m","length_m","angle_ref_rad","angle_rad","wheel_ref_rad_s","wheel_rad_s","closure_gap_mm","fk_error_mm"]);writer.writerows(rows)
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("MINE_DRIVE_REPORT "+json.dumps(report),flush=True)
        if not args.loop or args.headless or not app.is_running():
            assert report["passed"], report
            break
        rows=[]
        env.reset()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
