"""Isaac Lab policy/VMC interfaces without an Isaac Sim dependency."""

import importlib.util
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJECT / "checkpoints/optimized/yaw_heading_followup_20261001_200205_round0001_improved"
JOINTS = ("lf0_Joint", "lf1_Joint", "l_wheel_Joint", "rf0_Joint", "rf1_Joint", "r_wheel_Joint")


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", ConfigLoader.construct_sequence)


def load_config(path):
    with Path(path).open() as handle:
        cfg = yaml.load(handle, Loader=ConfigLoader)
    if cfg["observation_space"] != 27 or cfg["action_space"] != 6:
        raise ValueError("This deployment requires the 27-observation, 6-action VMC interface.")
    # Archived serial-leg configs predate the explicit controller mode fields.
    cfg.setdefault("leg_model", "serial")
    cfg.setdefault("leg_control_mode", "explicit_vmc")
    cfg.setdefault("wheel_control_mode", "explicit_effort")
    return cfg


def load_vmc():
    spec = importlib.util.spec_from_file_location("sim2sim_vmc", PROJECT / "wheel_legged_gym_isaaclab/vmc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VMC = load_vmc()


def load_five_bar():
    spec = importlib.util.spec_from_file_location("sim2sim_five_bar", PROJECT / "wheel_legged_gym_isaaclab/five_bar_vmc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIVE_BAR = load_five_bar()


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
    """Use MuJoCo's URDF importer to preserve link inertia and collision geometry."""
    if cfg["leg_model"] == "mine_five_bar":
        from mine_model import build_mine_model
        return build_mine_model(cfg, output_path)
    urdf_path = PROJECT / "assets/robots/wl/urdf/wl.urdf"
    urdf = ET.parse(urdf_path).getroot()
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(extension, "compiler", meshdir=str(urdf_path.parent.parent / "meshes"),
                  discardvisual="false", fusestatic="false")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    with tempfile.TemporaryDirectory() as directory:
        converted = Path(directory) / "robot.xml"
        mujoco.mj_saveLastXML(str(converted), imported)
        root = ET.parse(converted).getroot()
    root.set("model", "wheel_legged_sim2sim")
    root.find("compiler").set("meshdir", str(urdf_path.parent.parent / "meshes"))
    assets = root.find("asset")
    ET.SubElement(assets, "texture", name="ground_checker", type="2d", builtin="checker",
                  rgb1="0.22 0.28 0.30", rgb2="0.46 0.52 0.54", width="512", height="512")
    ET.SubElement(assets, "material", name="ground_material", texture="ground_checker",
                  texrepeat="2 2", texuniform="true", reflectance="0")
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.55 0.70 0.85", rgb2="0.90 0.93 0.95", width="512", height="3072")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(timestep=str(cfg["sim"]["dt"]), gravity="0 0 -9.81",
                         integrator="implicitfast", cone="elliptic", iterations="100")
    world = root.find("worldbody")
    base = world.find("body[@name='base_link']")
    if base is None:
        raise ValueError("URDF conversion lost base_link; cannot deploy a fixed-base model.")
    ET.SubElement(base, "freejoint", name="root")
    # Floor/robot masks disable self collisions, matching the Isaac Lab asset.
    for geom in root.findall(".//geom"):
        if geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0":
            continue
        geom.attrib.update(contype="2", conaffinity="1", group="3", friction="0.5 0.005 0.0001",
                           condim="3", solref="0.01 1")
    for joint in root.findall(".//joint"):
        joint.attrib.update(damping="0", frictionloss="0", armature="0")
    ET.SubElement(world, "geom", name="floor", type="plane", size="200 200 0.1",
                  contype="1", conaffinity="2", friction="0.5 0.005 0.0001",
                  material="ground_material", condim="3", solref="0.01 1")
    ET.SubElement(world, "light", pos="0 -2 4", dir="0 0 -1", directional="true")
    actuator = ET.SubElement(root, "actuator")
    for name, limit in zip(JOINTS, (30, 30, 5, 30, 30, 5)):
        ET.SubElement(actuator, "motor", name=name + "_motor", joint=name,
                      ctrllimited="true", ctrlrange=f"{-limit} {limit}")
    ET.SubElement(root, "statistic", center="0 0 0.25", extent="1.2")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(output_path))


class Simulation:
    def __init__(self, cfg, model):
        self.cfg, self.model = cfg, model
        self.joints = (("LB_joint", "LL_joint", "LW_joint", "RB_joint", "RL_joint", "RW_joint")
                       if cfg["leg_model"] == "mine_five_bar" else JOINTS)
        self.data = mujoco.MjData(model)
        self.base_id = model.body("base_link").id
        self.qids = np.array([model.joint(name).qposadr[0] for name in self.joints])
        self.vids = np.array([model.joint(name).dofadr[0] for name in self.joints])
        initial = cfg["robot"]["init_state"]
        self.data.qpos[:3] = initial["pos"]
        self.data.qpos[3:7] = initial["rot"]
        self.data.qpos[self.qids] = [initial["joint_pos"].get(name, initial["joint_pos"].get(".*", 0.)) for name in self.joints]
        self.actions = torch.zeros(1, 6)
        self.raw = self.actions.clone()
        self.command = np.array([0., 0., .18])
        self.length_ref = np.zeros(2)
        self.torques = np.zeros(6)
        mujoco.mj_forward(model, self.data)

    def coordinates(self):
        q = torch.as_tensor(self.data.qpos[self.qids], dtype=torch.float32)
        v = torch.as_tensor(self.data.qvel[self.vids], dtype=torch.float32)
        if self.cfg["leg_model"] == "mine_five_bar":
            b, l = q[[0, 3]][None], q[[1, 4]][None]
            length, angle, jacobian, valid = FIVE_BAR.five_bar_state(b, l, **self.cfg["five_bar_geometry"])
            virtual_velocity = (jacobian @ torch.stack((v[[0, 3]], v[[1, 4]]), -1)[None, ..., None]).squeeze(-1)
            self.jacobian, self.kinematics_valid = jacobian, valid
            return b, l, length, angle, virtual_velocity[..., 0], virtual_velocity[..., 1]
        t1 = torch.stack((q[0], -q[3]))[None]
        t2 = torch.stack((q[1], -q[4]))[None] + torch.pi / 2
        v1 = torch.stack((v[0], -v[3]))[None]
        v2 = torch.stack((v[1], -v[4]))[None]
        kwargs = {key: self.cfg[key] for key in ("l1", "l2", "offset")}
        length, angle = VMC.leg_coordinates(t1, t2, **kwargs)
        length_next, angle_next = VMC.leg_coordinates(t1 + .001 * v1, t2 + .001 * v2, **kwargs)
        return t1, t2, length, angle, (length_next - length) / .001, (angle_next - angle) / .001

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

        obs = np.concatenate((clip(angular, "ang_vel", s["ang_vel"]),
                              clip(gravity, "projected_gravity"),
                              self.command * [s["lin_vel"], s["ang_vel"], s["height_measurements"]],
                              clip(angle, "theta", s["dof_pos"]),
                              clip(angle_dot, "theta_dot", s["dof_vel"]),
                              clip(length, "leg_length", s["l0"]),
                              clip(length_dot, "leg_length_dot", s["l0_dot"]),
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
        if c["leg_model"] == "mine_five_bar":
            if c["leg_control_mode"] != "implicit_joint_reference" or c["wheel_control_mode"] != "implicit_velocity":
                raise ValueError("New-model deployment requires the trained implicit leg/wheel drive contract.")
            b, l, _ = FIVE_BAR.five_bar_inverse(length_ref, angle_ref, **c["five_bar_geometry"])
            self.joint_ref = torch.stack((b[0, 0], l[0, 0], b[0, 1], l[0, 1])).numpy()
            wheel_ref = self.actions[0, (2, 5)].numpy() * c["action_scale_vel"] * [1, -1]
            self.data.ctrl[:] = [self.joint_ref[0], self.joint_ref[1], wheel_ref[0],
                                 self.joint_ref[2], self.joint_ref[3], wheel_ref[1]]
            self.length_ref = length_ref.numpy()[0].copy()
            return
        force = c["kp_l0"] * (length_ref - length) - c["kd_l0"] * length_dot + c["feedforward_force"]
        torque = c["kp_theta"] * (angle_ref - angle) - c["kd_theta"] * angle_dot
        hip, knee = VMC.virtual_leg_torques(t1, t2, length, angle, force, torque,
                                          l1=c["l1"], l2=c["l2"],
                                          legacy_angular_mapping=c["vmc_legacy_angular_mapping"])
        wheel = torch.tensor(self.data.qvel[self.vids[[2, 5]]] * [1, -1], dtype=torch.float32)[None]
        wheel_torque = c["wheel_damping"] * (self.actions[:, (2, 5)] * c["action_scale_vel"] - wheel)
        self.torques = np.clip(torch.stack((hip[0, 0], knee[0, 0], wheel_torque[0, 0],
                                            -hip[0, 1], -knee[0, 1], -wheel_torque[0, 1])).numpy(),
                               -np.array([30, 30, 5, 30, 30, 5]), np.array([30, 30, 5, 30, 30, 5]))
        self.length_ref = length_ref.numpy()[0].copy()
        self.data.ctrl[:] = self.torques

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
        if self.cfg["leg_model"] == "mine_five_bar":
            self.torques = self.data.actuator_force.copy()
