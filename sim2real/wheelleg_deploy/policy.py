"""Export a self-contained, deterministic actor; no Isaac Sim/RSL-RL import."""

import importlib.util
import hashlib
import json
from pathlib import Path

from .schema import make_schema, validate_observation, finite_vector, ACTION_DIM

CONFIG_KEYS = (
    "leg_model", "leg_control_mode", "wheel_control_mode", "observation_space", "action_space",
    "sim", "decimation", "obs_scales", "obs_clip", "commands", "five_bar_geometry",
    "leg_joint_names", "wheel_joint_names", "wheel_radius", "wheel_track_width",
    "leg_effort_limit", "wheel_effort_limit", "leg_joint_stiffness", "leg_joint_damping",
    "wheel_damping", "action_scale_vel", "leg_joint_position_offset", "leg_joint_position_scale",
    "leg_joint_target_limit_margin", "leg_joint_gravity_feedforward", "hardware_profile",
    "feedforward_mode", "feedforward_force", "feedforward_angle_bias", "feedforward_force_limit",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path):
    import yaml
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor("tag:yaml.org,2002:python/tuple", Loader.construct_sequence)
    return yaml.load(Path(path).read_text(), Loader=Loader)


def prepare_bundle(checkpoint, output):
    import torch
    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Bundle destination already exists; use a new directory (no silent overwrite)")
    paired = checkpoint.parent / "params"
    cfg, agent = load_yaml(paired / "env.yaml"), load_yaml(paired / "agent.yaml")
    if cfg["observation_space"] != 27 or cfg["action_space"] != 6:
        raise ValueError("Only the current 27-observation flat policy is supported; not the 39-D jump policy")
    if cfg["leg_control_mode"] != "implicit_joint_position" or cfg["wheel_control_mode"] != "implicit_velocity":
        raise ValueError("Only direct-joint position and implicit wheel velocity policies can be exported; IK is retired")
    source = Path(__file__).resolve().parents[2] / "wheel_legged_isaaclab/wheel_legged_gym_isaaclab/robot_model.py"
    spec = importlib.util.spec_from_file_location("sim2real_robot_contract", source)
    robot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(robot)
    cfg["five_bar_geometry"] = robot.validate_robot_contract(cfg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state, spec_policy = saved["model_state_dict"], agent["policy"]
    if spec_policy["class_name"] != "ActorCritic" or agent.get("empirical_normalization"):
        raise ValueError("Unsupported recurrent/normalized actor; export the complete inference module instead")
    if spec_policy.get("actor_obs_normalization") or any("normalizer" in key for key in state):
        raise ValueError("Actor normalization cannot be silently discarded")
    activations = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh,
                   "selu": torch.nn.SELU, "leakyrelu": torch.nn.LeakyReLU, "sigmoid": torch.nn.Sigmoid}
    dimensions, layers = [27, *spec_policy["actor_hidden_dims"], 6], []
    for i, (n_in, n_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(torch.nn.Linear(n_in, n_out))
        if i < len(dimensions)-2:
            layers.append(activations[spec_policy["activation"].lower()]())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict({key[6:]: value for key, value in state.items() if key.startswith("actor.")}, strict=True)
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Actor weights contain non-finite values")
    actor.eval()
    cfg = {key: cfg[key] for key in CONFIG_KEYS}
    schema = make_schema(cfg)
    dt = float(cfg["sim"]["dt"]) * int(cfg["decimation"])
    if not 0 < dt < .1:
        raise ValueError("Invalid trained policy period")
    metadata = dict(format="clt_mine_joint_position_bundle_v2", checkpoint=str(checkpoint),
                    checkpoint_sha256=sha256(checkpoint), iteration=int(saved["iter"]),
                    env_sha256=sha256(paired / "env.yaml"), agent_sha256=sha256(paired / "agent.yaml"),
                    schema_sha256=schema["sha256"], policy_period_s=dt,
                    inner_loop_period_s=cfg["sim"]["dt"],
                    hardware_approved=False, policy_precision_accepted=False,
                    execution_contract="direct_joint_position_pd_gravity_and_wheel_velocity",
                    warning="Offline preparation only; hardware PD dynamics, MCU protocol and watchdog are NOT validated")
    output.mkdir(parents=True)
    torch.jit.script(actor).save(str(output / "policy.pt"))
    metadata["policy_sha256"] = sha256(output / "policy.pt")
    for name, data in (("runtime_config.json", cfg), ("obs_schema.json", schema)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    metadata["runtime_config_sha256"] = sha256(output / "runtime_config.json")
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    sample = [0.] * 27
    sample[5] = -1.
    sample[8] = sum(cfg["commands"]["ranges_height"]) / 2 * cfg["obs_scales"]["height_measurements"]
    (output / "synthetic_obs.json").write_text(json.dumps({"synthetic_not_hardware": True, "obs": sample}, indent=2) + "\n")
    return metadata


class BundlePolicy:
    def __init__(self, folder):
        import torch
        folder = Path(folder).expanduser().resolve()
        self.metadata = json.loads((folder / "manifest.json").read_text())
        self.cfg = json.loads((folder / "runtime_config.json").read_text())
        self.schema = json.loads((folder / "obs_schema.json").read_text())
        if self.metadata["format"] != "clt_mine_joint_position_bundle_v2":
            raise ValueError("Unsupported or retired IK bundle format; export a fresh direct-joint bundle")
        if sha256(folder / "policy.pt") != self.metadata["policy_sha256"]:
            raise ValueError("Policy artifact changed since export")
        if sha256(folder / "runtime_config.json") != self.metadata["runtime_config_sha256"]:
            raise ValueError("Controller configuration changed since export")
        if make_schema(self.cfg) != self.schema or self.schema["sha256"] != self.metadata["schema_sha256"]:
            raise ValueError("Observation schema/config mismatch")
        self.actor = torch.jit.load(str(folder / "policy.pt"), map_location="cpu").eval()

    def __call__(self, observation):
        import torch
        values = validate_observation(observation, self.schema, self.cfg)
        with torch.inference_mode():
            result = self.actor(torch.tensor([values], dtype=torch.float32))
        if not isinstance(result, torch.Tensor) or result.shape != (1, ACTION_DIM):
            raise ValueError("Policy must produce one 6-D action, not a distribution or critic output")
        return finite_vector(result[0].tolist(), ACTION_DIM, "actor output")
