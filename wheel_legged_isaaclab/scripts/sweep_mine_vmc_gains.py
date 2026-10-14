"""Batched fixed-reference VMC pilot; not a learned-policy acceptance test.

Compare gains using the same CAD, collision, limits and timestep. Each trial
uses zero wheel reference and the same nominal leg reference. Record BEFORE
auto-reset and never promote clean reset samples as sustained support.
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--overrides_json', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seconds', type=float, default=5.)
parser.add_argument('--fixed-base', action='store_true',help='Diagnostic only: separate leg tracking from free-base balance')
parser.add_argument('--angle-ref',type=float,default=0.,help='Common fixed virtual-leg angle in radians; diagnostic only')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main():
    import csv
    import numpy as np
    import torch
    from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv

    if args.seconds <= 0:
        raise ValueError('seconds must be positive')
    if not -0.2 <= args.angle_ref <= 0.2:
        raise ValueError('angle-ref must stay in the designed reference range')
    gains = [(kt, dt, kl, dl)
             for kt, dt in ((50., 3.), (10., 1.), (30., 1.), (60., 2.))
             for kl, dl in ((900., 90.), (900., 15.), (900., 5.), (900., 2.),
                            (1800., 10.), (1800., 20.), (400., 10.), (400., 5.))]
    cfg = WheelLeggedVMCFlatEnvCfg()
    apply_config_overrides(cfg, json.loads(args.overrides_json.read_text()))
    if cfg.leg_control_mode != 'explicit_vmc':
        raise ValueError('The pilot must preserve explicit VMC')
    cfg.scene.num_envs = 3 * len(gains)
    cfg.scene.clone_in_fabric = False
    cfg.sim.device = args.device or 'cuda:0'
    cfg.seed = 43
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.ranges_lin_vel_x = (0., 0.)
    cfg.commands.ranges_ang_vel_yaw = (0., 0.)
    cfg.commands.ranges_height = (.30, .30)
    cfg.commands.grouped_training = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1000.
    cfg.episode_length_s = args.seconds + 2.
    if args.fixed_base:
        cfg.robot.spawn.articulation_props.fix_root_link = True
        cfg.robot.init_state.pos = (0., 0., .8)
        # A suspended base is not a height-control test. Preserve nominal leg
        # references and disable only the irrelevant root-height correction.
        cfg.height_feedback_gain = 0.
    gain_tensor = torch.tensor(gains, device=cfg.sim.device).repeat_interleave(3, dim=0)

    class PilotEnv(WheelLeggedVMCFlatEnv):
        recording = False

        def _vmc(self, unused_force, unused_torque):
            # Replace only virtual-coordinate gain coefficients, not the
            # Jacobian, raw motor signs, gravity, force limits or joint drives.
            kt, dt, kl, dl = (gain_tensor[:, i:i+1] for i in range(4))
            angle_ref = self._actions[:, (0, 3)] * self.cfg.action_scale_theta
            force = kl * (self._l0_ref_applied-self._L0) - dl*self._L0_dot + self.cfg.feedforward_force
            torque = kt * (angle_ref-self._theta0) - dt*self._theta0_dot
            return super()._vmc(force, torque)

        def _get_dones(self):
            self._update_forward_kinematics()
            died, timeout = super()._get_dones()
            if self.recording:
                g = self._robot.data.projected_gravity_b
                torque = self._robot.data.joint_effort_target[:, self._leg_joint_ids]
                self.sample = torch.stack((
                    self._robot.data.root_pos_w[:, 2],
                    torch.asin(g[:, 0].clamp(-1, 1)), torch.atan2(-g[:, 1], -g[:, 2]),
                    self._L0[:, 0], self._L0[:, 1], self._L0_dot[:, 0], self._L0_dot[:, 1],
                    torque.abs().amax(-1), (torque.abs() >= .99*self.cfg.leg_effort_limit).float().mean(-1),
                    died.float(), timeout.float(), self._termination_reasons['low_height'].float(),
                    self._theta0[:, 0], self._theta0[:, 1],
                ), dim=-1).cpu().numpy()
            return died, timeout

    env = PilotEnv(cfg)
    env.reset()
    actions = torch.zeros(cfg.scene.num_envs, 6, device=env.device)
    actions[:,(0,3)] = args.angle_ref/cfg.action_scale_theta
    samples = []
    env.recording = True
    with torch.inference_mode():
        for _ in range(round(args.seconds / env.step_dt)):
            obs, reward, _, _, _ = env.step(actions)
            if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                raise RuntimeError('Nonfinite controller pilot')
            samples.append(env.sample.copy())
    trace = np.stack(samples)
    if not np.isfinite(trace).all():
        raise RuntimeError('Nonfinite pre-reset controller telemetry')
    results = []
    for cid, gain in enumerate(gains):
        trials = []
        for offset in range(3):
            eid = 3*cid+offset
            data = trace[:, eid]
            deaths = np.flatnonzero((data[:, 9] > 0) | (data[:, 10] > 0))
            first = data[:deaths[0]+1] if len(deaths) else data
            trials.append(dict(survived=not len(deaths),
                first_failure_s=float((deaths[0]+1)*env.step_dt) if len(deaths) else None,
                minimum_first_episode_height_m=float(first[:, 0].min()),
                torque_saturation_fraction=float(first[:, 8].mean()),
                final_first_episode_height_m=float(first[-1, 0]),
                final_left_length_m=float(first[-1, 3]),
                final_right_length_m=float(first[-1, 4]),
                final_left_angle_rad=float(first[-1, 12]),
                final_right_angle_rad=float(first[-1, 13]),
                low_height_resets=int(data[:, 11].sum()),
                pitch_peak_deg=float(np.rad2deg(abs(first[:, 1]).max())),
                radial_velocity_peak_m_s=float(abs(first[:, 5:7]).max()),
                length_peak_to_peak_m=float(np.ptp(first[:, 3:5], axis=0).max())))
        results.append(dict(kp_theta=gain[0], kd_theta=gain[1], kp_l0=gain[2], kd_l0=gain[3],
                            survivors=sum(t['survived'] for t in trials), trials=trials))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'trace.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'case_id', 'replica', 'height_m', 'pitch_rad', 'roll_rad',
                         'left_length_m', 'right_length_m', 'left_length_velocity_m_s',
                         'right_length_velocity_m_s', 'peak_leg_target_nm', 'leg_target_saturation',
                         'terminated', 'timeout', 'low_height', 'left_angle_rad', 'right_angle_rad'])
        for step, batch in enumerate(trace):
            for eid, row in enumerate(batch):
                writer.writerow([step, eid//3, eid%3, *row.tolist()])
    report = dict(seconds=args.seconds, fixed_base=args.fixed_base, angle_reference_rad=args.angle_ref,
                  physics_dt=cfg.sim.dt, policy_dt=env.step_dt,
                  controller='explicit virtual-coordinate PD then J transpose; no joint PD',
                  learned_policy_tested=False, replicas='identical nominal reset, not independent acceptance seeds',
                  results=results, runtime_config=cfg.to_dict())
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    print('VMC_GAIN_PILOT '+json.dumps(results), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
