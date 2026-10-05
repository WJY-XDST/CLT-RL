"""Compare gravity support cases on the actual closed-chain articulation."""
import argparse
import csv
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=3.)
parser.add_argument("--gain-sweep", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

def main():
    import torch
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.mine_env_cfg import MineWheelLeggedVMCFlatEnvCfg
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv
    from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_torques

    def priority(j, force, torque, limit):
        radial, angular = j[..., 0, :], j[..., 1, :]
        fmax = limit / radial.abs().amax(-1).clamp_min(1e-8)
        force = force.clamp(-fmax, fmax)
        base = radial * force[..., None]
        denom = torch.where(angular.abs() > 1e-8, angular, torch.ones_like(angular))
        a, b = (-limit-base)/denom, (limit-base)/denom
        low = torch.where(angular.abs() > 1e-8, torch.minimum(a,b), -torch.inf).amax(-1)
        high = torch.where(angular.abs() > 1e-8, torch.maximum(a,b), torch.inf).amin(-1)
        torque = torque.clamp(low, high)
        return five_bar_torques(j,force,torque)

    cfg = MineWheelLeggedVMCFlatEnvCfg()
    gains = [(kt,kl) for kt in (.1,.3,3.) for kl in (10.,30.,90.)]
    cfg.scene.num_envs = len(gains) if args.gain_sweep else 4
    cfg.seed = 43
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.ranges_lin_vel_x = (0.,0.)
    cfg.commands.ranges_height = (.30,.30)
    cfg.episode_length_s = args.seconds+2

    class Env(WheelLeggedVMCFlatEnv):
        def _vmc(self, force, torque):
            if args.gain_sweep:
                return five_bar_torques(self._five_bar_jacobian,force,torque)
            force, torque = force.clone(), torque.clone()
            force[1] = self.cfg.feedforward_force
            torque[1:3] = 0.
            t1,t2 = five_bar_torques(self._five_bar_jacobian,force,torque)
            p1,p2 = priority(self._five_bar_jacobian,force,torque,self.cfg.leg_effort_limit)
            t1[3],t2[3] = p1[3],p2[3]
            return t1,t2

        def _get_dones(self):
            died,timeout = super()._get_dones()
            self.raw_died = died.clone()
            # Record the first collapse without restarting the diagnosis.
            return self._termination_reasons['numerical'], timeout

    env = Env(cfg)
    env.reset()
    names = ([f'kd_theta_{kt}_kd_length_{kl}' for kt,kl in gains] if args.gain_sweep else ['original_zero_action','constant_radial_feedforward','radial_pd_only','radial_priority_pd'])
    args.output.mkdir(parents=True,exist_ok=True)
    meta={'cases':names,'joint_names':env._robot.joint_names,'active_order':[env._robot.joint_names[i] for i in env._torque_joint_ids],'cfg':cfg.to_dict()}
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
    columns=['time_s','case','height_m','pitch_rad','left_length_m','right_length_m','left_theta_rad','right_theta_rad','left_ref_m','right_ref_m','raw_failure']
    columns += ['q_'+n for n in env._robot.joint_names]
    columns += ['tau_'+n for n in meta['active_order']]
    if args.gain_sweep:
        cfg.kd_theta=torch.tensor([g[0] for g in gains],device=env.device)[:,None]
        cfg.kd_l0=torch.tensor([g[1] for g in gains],device=env.device)[:,None]
    first_failure=[None]*len(names)
    min_height=[1.]*len(names)
    actions=torch.zeros(len(names),6,device=env.device)
    with (args.output/'trace.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns)
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                env.step(actions)
                env._update_forward_kinematics()
                d=env._robot.data
                pitch=torch.asin(d.projected_gravity_b[:,0].clamp(-1,1))
                for i in range(len(names)):
                    h=d.root_pos_w[i,2].item();min_height[i]=min(min_height[i],h)
                    failed=env.raw_died[i].item()
                    if failed and first_failure[i] is None:first_failure[i]=(step+1)*env.step_dt
                    row=[(step+1)*env.step_dt,names[i],h,pitch[i].item(),*env._L0[i].tolist(),*env._theta0[i].tolist(),*env._l0_ref_applied[i].tolist(),int(failed)]
                    row+=d.joint_pos[i].tolist()+d.applied_torque[i,env._torque_joint_ids].tolist()
                    writer.writerow(row)
    result={'cases':[{'name':n,'first_failure_s':t,'min_height_m':h} for n,t,h in zip(names,first_failure,min_height)]}
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUPPORT_DIAGNOSIS '+json.dumps(result),flush=True)
    env.close()

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    sys.stdout.flush()
    app.close()
