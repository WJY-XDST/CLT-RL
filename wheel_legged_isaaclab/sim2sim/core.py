"""Isaac Lab policy/VMC interfaces without an Isaac Sim dependency."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", ConfigLoader.construct_sequence)


def load_config(path):
    with Path(path).open() as handle:
        cfg = yaml.load(handle, Loader=ConfigLoader)
    if cfg["observation_space"] != 27 or cfg["action_space"] != 6:
        raise ValueError("This replay requires the paired 27-observation, 6-action vehicle interface.")
    cfg["five_bar_geometry"] = ROBOT.validate_robot_contract(cfg)
    # Saved new-vehicle configs may contain unused inherited fields. They
    # must never re-enter an actuator or kinematic calculation.
    for key in ("l1", "l2", "offset", "vmc_legacy_angular_mapping"):
        cfg.pop(key, None)
    for key in ("leg_control_mode", "wheel_control_mode"):
        if key not in cfg:
            raise ValueError(f"Checkpoint must specify {key}")
    LEG_CONTROL.require_leg_control_mode(cfg['leg_control_mode'])
    if cfg['wheel_control_mode'] != 'implicit_velocity':
        raise ValueError('MuJoCo replay currently requires implicit wheel velocity control')
    return cfg


def load_robot_model():
    spec = importlib.util.spec_from_file_location("sim2sim_robot", PROJECT / "wheel_legged_gym_isaaclab/robot_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROBOT = load_robot_model()


def load_five_bar():
    spec = importlib.util.spec_from_file_location("sim2sim_five_bar", PROJECT / "wheel_legged_gym_isaaclab/five_bar_vmc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIVE_BAR = load_five_bar()

_leg_spec = importlib.util.spec_from_file_location(
    'sim2sim_leg_control', PROJECT / 'wheel_legged_gym_isaaclab/leg_control.py')
LEG_CONTROL = importlib.util.module_from_spec(_leg_spec)
_leg_spec.loader.exec_module(LEG_CONTROL)


def load_actor(checkpoint, agent_path):
    """Reconstruct only the deterministic actor, rejecting unsupported state."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    with Path(agent_path).open() as handle:
        policy = yaml.load(handle, Loader=ConfigLoader)["policy"]
    if any("normalizer" in key for key in state):
        raise ValueError("Normalized checkpoint: export a TorchScript actor including its normalizer first.")
    activations = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh,
                   "selu": torch.nn.SELU, "leakyrelu": torch.nn.LeakyReLU,
                   "sigmoid": torch.nn.Sigmoid}
    activation = activations[policy["activation"].lower()]
    dimensions = [27, *policy["actor_hidden_dims"], 6]
    layers = []
    for index, (inputs, outputs) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(torch.nn.Linear(inputs, outputs))
        if index < len(dimensions) - 2:
            layers.append(activation())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items()
                           if key.startswith("actor.")}, strict=True)
    return actor.eval()


def build_model(cfg, output_path):
    """Convert only the imported closed-chain USD, retaining its asset audit."""
    if cfg["leg_model"] != "mine_five_bar":
        raise ValueError("Only the imported closed-five-bar vehicle is supported")
    from mine_model import build_mine_model
    return build_mine_model(cfg, output_path)


class Simulation:
    def __init__(self, cfg, model):
        self.cfg, self.model = cfg, model
        if cfg["leg_model"] != "mine_five_bar":
            raise ValueError("Only the imported closed-five-bar vehicle is supported")
        LEG_CONTROL.require_leg_control_mode(cfg['leg_control_mode'])
        if cfg['wheel_control_mode'] != 'implicit_velocity':
            raise ValueError('MuJoCo replay requires implicit wheel velocity control')
        legs, wheels = cfg["leg_joint_names"], cfg["wheel_joint_names"]
        self.joints = (legs[0], legs[1], wheels[0], legs[2], legs[3], wheels[1])
        self.data = mujoco.MjData(model)
        self.base_id = model.body("base_link").id
        self.qids = np.array([model.joint(name).qposadr[0] for name in self.joints])
        self.vids = np.array([model.joint(name).dofadr[0] for name in self.joints])
        # Reuse CAD/contact geometry, but never inherit an old IK actuator's
        # gains/caps from its MJCF. All six drives follow the selected config.
        for index, name in enumerate(self.joints):
            if model.actuator_trnid[index, 0] != model.joint(name).id:
                raise ValueError('MJCF actuator order differs from LB/LL/LW/RB/RL/RW')
            wheel = index in (2, 5)
            position = not wheel and cfg['leg_control_mode'] == LEG_CONTROL.DIRECT_JOINT_POSITION_MODE
            kp = cfg['leg_joint_stiffness'] if position else 0.
            kd = cfg['wheel_damping'] if wheel else cfg['leg_joint_damping'] if position else 0.
            model.actuator_gainprm[index, 0] = kp if position else kd if wheel else 1.
            model.actuator_biasprm[index, :3] = [0., -kp, -kd]
            limit = cfg['wheel_effort_limit'] if wheel else cfg['leg_effort_limit']
            model.actuator_forcelimited[index] = True
            model.actuator_forcerange[index] = [-limit, limit]
        leg_jids = [model.joint(self.joints[i]).id for i in (0, 1, 3, 4)]
        self.joint_limits = torch.tensor(model.jnt_range[leg_jids][None], dtype=torch.float32)
        initial = cfg["robot"]["init_state"]
        self.data.qpos[:3] = initial["pos"]
        self.data.qpos[3:7] = initial["rot"]
        self.data.qpos[self.qids] = [initial["joint_pos"].get(name, initial["joint_pos"].get(".*", 0.)) for name in self.joints]
        self.actions = torch.zeros(1, 6)
        self.raw = self.actions.clone()
        self.command = np.array([0., 0., cfg["rewards"]["base_height_target"]])
        self.length_ref = np.zeros(2)
        self.torques = np.zeros(6)
        mujoco.mj_forward(model, self.data)

    def coordinates(self):
        q = torch.as_tensor(self.data.qpos[self.qids], dtype=torch.float32)
        v = torch.as_tensor(self.data.qvel[self.vids], dtype=torch.float32)
        b, l = q[[0, 3]][None], q[[1, 4]][None]
        length, angle, jacobian, valid = FIVE_BAR.five_bar_state(b, l, **self.cfg["five_bar_geometry"])
        virtual_velocity = (jacobian @ torch.stack((v[[0, 3]], v[[1, 4]]), -1)[None, ..., None]).squeeze(-1)
        self.jacobian, self.kinematics_valid = jacobian, valid
        return b, l, length, angle, virtual_velocity[..., 0], virtual_velocity[..., 1]

    def body_state(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        velocity = np.zeros(6)
        # BODY velocity is at its COM, matching Isaac Lab root_lin_vel_w.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base_id, velocity, 0)
        angular_body = rotation.T @ velocity[:3]
        gravity_body = rotation.T @ np.array([0., 0., -1.])
        forward = rotation[:2, 0]
        forward = forward / max(np.linalg.norm(forward), 1e-6)
        heading_velocity = np.array([velocity[3:5] @ forward,
                                     velocity[3:5] @ np.array([-forward[1], forward[0]])])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        return angular_body, gravity_body, heading_velocity, yaw, pitch, roll

    def observation(self):
        c, s = self.cfg["obs_clip"], self.cfg["obs_scales"]
        _, _, length, angle, length_dot, angle_dot = self.coordinates()
        angular, gravity, velocity, *_ = self.body_state()
        wheel = self.data.qvel[self.vids[[2, 5]]] * np.array([1, -1])

        def clip(value, key, scale=1):
            array = np.asarray(value).reshape(-1)
            return np.clip(np.nan_to_num(array, nan=0, posinf=c[key], neginf=-c[key]),
                           -c[key], c[key]) * scale

        if self.cfg['leg_control_mode'] == LEG_CONTROL.DIRECT_JOINT_POSITION_MODE:
            position = self.data.qpos[self.qids[[0, 1, 3, 4]]] - np.asarray(self.cfg['leg_joint_position_offset'])
            joint_velocity = self.data.qvel[self.vids[[0, 1, 3, 4]]]
            leg_obs = (clip(position, 'theta', s['dof_pos']), clip(joint_velocity, 'theta_dot', s['dof_vel']))
        else:
            leg_obs = (clip(angle, 'theta', s['dof_pos']), clip(angle_dot, 'theta_dot', s['dof_vel']),
                       clip(length, 'leg_length', s['l0']), clip(length_dot, 'leg_length_dot', s['l0_dot']))
        obs = np.concatenate((clip(angular, "ang_vel", s["ang_vel"]),
                              clip(gravity, "projected_gravity"),
                              self.command * [s["lin_vel"], s["ang_vel"], s["height_measurements"]],
                              *leg_obs,
                              clip(velocity, "lin_vel", s["lin_vel"]),
                              clip(wheel, "wheel_vel", s["dof_vel"]),
                              clip(self.actions, "action"))).astype(np.float32)
        if obs.shape != (27,) or not np.isfinite(obs).all():
            raise ValueError("Invalid policy observation.")
        return torch.from_numpy(obs)[None]

    def set_action(self, action):
        self.raw = torch.nan_to_num(action.detach(), nan=0, posinf=1, neginf=-1)
        self.actions = self.raw.clamp(-1, 1).clone()

    def apply_control(self):
        c = self.cfg
        t1, t2, length, angle, length_dot, angle_dot = self.coordinates()
        pitch = torch.tensor([[self.body_state()[4]]], dtype=torch.float32)
        gravity = ROBOT._hardware.gravity_feedforward(
            angle, pitch, mode=c.get('feedforward_mode', 'constant'),
            force=c['feedforward_force'], angle_bias=c.get('feedforward_angle_bias', 0.),
            force_limit=c.get('feedforward_force_limit', 150.),
        )
        if c['leg_control_mode'] == LEG_CONTROL.DIRECT_JOINT_POSITION_MODE:
            target, applied = LEG_CONTROL.joint_position_targets(
                self.raw[:, LEG_CONTROL.LEG_ACTION_COLUMNS], c['leg_joint_position_offset'],
                c['leg_joint_position_scale'], self.joint_limits, c['leg_joint_target_limit_margin'],
            )
            self.actions[:, LEG_CONTROL.LEG_ACTION_COLUMNS] = applied
            self.joint_ref = target[0].numpy()
            ff_b, ff_l = FIVE_BAR.five_bar_torques(self.jacobian, gravity, torch.zeros_like(angle))
            ff = torch.stack((ff_b[:, 0], ff_l[:, 0], ff_b[:, 1], ff_l[:, 1]), -1)[0]
            ff = ff.clamp(-c['leg_effort_limit'], c['leg_effort_limit']).numpy()
            if not c['leg_joint_gravity_feedforward']:
                ff[:] = 0.
            # Constant actuator bias adds gravity BEFORE total drive clipping.
            # Adding qfrc_applied after a saturated drive would exceed the cap.
            self.model.actuator_biasprm[[0, 1, 3, 4], 0] = ff
            self.data.ctrl[[0, 1, 3, 4]] = self.joint_ref
            self.data.ctrl[[2, 5]] = self.actions[0, (2, 5)].numpy() * c['action_scale_vel'] * [1, -1]
            self.length_ref = FIVE_BAR.five_bar_state(target[:, ::2], target[:, 1::2], **c['five_bar_geometry'])[0][0].numpy()
            return
        angle_ref = (self.actions[:, (0, 3)] * c["action_scale_theta"]).clamp(c["theta0_ref_min"], c["theta0_ref_max"])
        length_ref = self.raw[:, (1, 4)].clamp(-1, 1) * c["action_scale_l0"] + c["l0_offset"]
        nominal = self.command[2] + c["leg_length_height_offset"]
        length_ref = (1 - c["height_reference_blend"]) * length_ref + c["height_reference_blend"] * nominal
        correction = np.clip((self.command[2] - self.data.qpos[2]) * c["height_feedback_gain"],
                             -c["height_feedback_max_adjustment"], c["height_feedback_max_adjustment"])
        length_ref = (length_ref + correction).clamp(c["l0_ref_min"], c["l0_ref_max"])
        if self.command[0] >= c["forward_support_speed_threshold"]:
            length_ref = torch.maximum(length_ref, torch.tensor(min(c["forward_support_min_leg_length"],
                                                                    self.command[2] + c["forward_support_height_margin"])))
        self.actions[:, (1, 4)] = (length_ref - c["l0_offset"]) / c["action_scale_l0"]
        force = c['kp_l0'] * (length_ref - length) - c['kd_l0'] * length_dot + gravity
        swing = c['kp_theta'] * (angle_ref - angle) - c['kd_theta'] * angle_dot
        b, l = FIVE_BAR.five_bar_torques(self.jacobian, force, swing)
        effort = torch.stack((b[:, 0], l[:, 0], b[:, 1], l[:, 1]), -1)[0]
        effort = effort.clamp(-c['leg_effort_limit'], c['leg_effort_limit']).numpy()
        self.joint_ref = np.zeros(4)  # No position drive in explicit VMC.
        wheel_ref = self.actions[0, (2, 5)].numpy() * c["action_scale_vel"] * [1, -1]
        self.data.ctrl[:] = [effort[0], effort[1], wheel_ref[0], effort[2], effort[3], wheel_ref[1]]
        self.length_ref = length_ref.numpy()[0].copy()

    def step(self):
        for _ in range(self.cfg["decimation"]):
            self.apply_control()
            # MuJoCo's saturated affine actuator has zero force derivative
            # once clipped. Resolve the tiny wheel inertia with internal steps,
            # holding the same 200 Hz reference and 50 Hz actor contract.
            for _ in range(self.cfg.get('_mujoco_physics_substeps', 1)):
                mujoco.mj_step(self.model, self.data)
            # Refresh poses/velocities after integration, before observation/control.
            mujoco.mj_forward(self.model, self.data)
        self.torques = self.data.actuator_force.copy()
