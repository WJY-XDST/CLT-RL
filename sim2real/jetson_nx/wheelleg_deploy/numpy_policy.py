"""Portable FP32 Actor bundle. NX runtime needs NumPy, not Torch/Isaac/CUDA.

Export happens on the training workstation from an explicitly selected trusted
checkpoint. A verified common controller change is recorded separately from
unchanged policy weights. No checkpoint or optimizer is rewritten.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from .policy import CONFIG_KEYS, load_yaml, sha256
from .schema import make_schema, validate_observation, finite_vector
from .targets import clip_action

FORMAT = "clt_nx_torque_numpy_v1"
CONTROL_CONTRACT = dict(
    revision="direct_position_explicit_pd_gravity_v1",
    execution="NX computes final output-shaft Nm; STM32 executes without another PD or gravity term",
    torque_order=["LB","LL","LW","RB","RL","RW"],
    leg_axes="raw CAD motor axes", wheel_axes="both forward positive",
    wire_version=1, observation_floats=27, raw_actuator_floats=10,
    gravity="trained analytic feedforward, limited before sum; total torque limited after sum",
    gas_spring_feedforward=False,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def contract_digest(weights_sha, config_sha, schema_sha):
    return hashlib.sha256(canonical(dict(format=FORMAT, control=CONTROL_CONTRACT, weights=weights_sha,
                                         configuration=config_sha, observation=schema_sha))).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def prepare_numpy_bundle(checkpoint, output, *, verified_replay=None):
    import numpy as np
    import torch
    from pxr import Usd, UsdPhysics
    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Bundle destination already exists")
    paired = checkpoint.parent/"params"
    cfg, agent = load_yaml(paired/"env.yaml"), load_yaml(paired/"agent.yaml")
    cfg = {key: cfg[key] for key in CONFIG_KEYS}
    if cfg["leg_control_mode"] != "implicit_joint_position" or cfg["wheel_control_mode"] != "implicit_velocity":
        raise ValueError("Expected direct joint positions and wheel velocities")
    if cfg["observation_space"] != 27 or cfg["action_space"] != 6:
        raise ValueError("Unsupported observation/action dimensions")
    policy_spec = agent["policy"]
    if (policy_spec["class_name"] != "ActorCritic" or policy_spec["activation"].lower() != "elu"
            or agent.get("empirical_normalization") or policy_spec.get("actor_obs_normalization")):
        raise ValueError("This exporter supports the original non-normalized ELU ActorCritic only")
    source = Path(__file__).resolve().parents[3]/"wheel_legged_isaaclab/wheel_legged_gym_isaaclab/robot_model.py"
    spec = importlib.util.spec_from_file_location("nx_export_robot", source)
    robot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(robot)
    cfg["five_bar_geometry"] = robot.validate_robot_contract(cfg)
    checkpoint_sha = sha256(checkpoint)
    changes = []
    if verified_replay is not None:
        replay = Path(verified_replay).resolve()
        manifest = json.loads((replay/"manifest.json").read_text())
        receipt_path = replay/"native_common_wheel_gain_verified.json"
        receipt = json.loads(receipt_path.read_text())
        resolved = json.loads((replay/"resolved_env.json").read_text())
        if not receipt["verified"] or manifest["checkpoint_sha256"] != checkpoint_sha or receipt["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Frozen native replay does not match the selected checkpoint")
        if receipt["weights_updated"] or receipt["condition_specific_feedback"]:
            raise ValueError("Expected frozen policy and a uniform controller setting")
        actual = receipt["actual_PhysX_wheel_damping_range"]
        if len(actual) != 2 or actual[0] != actual[1] or actual[0] != resolved["wheel_damping"]:
            raise ValueError("Actual wheel gain does not match replay configuration")
        for key in CONFIG_KEYS:
            if key not in ("commands", "sim", "wheel_damping", "five_bar_geometry") and cfg[key] != resolved[key]:
                raise ValueError("Unexpected frozen replay control change: "+key)
        if cfg["sim"]["dt"] != resolved["sim"]["dt"]:
            raise ValueError("Physics timestep changed")
        changes.append(dict(field="wheel_damping", source=cfg["wheel_damping"], selected=actual[0],
                            authority=str(receipt_path), authority_sha256=sha256(receipt_path)))
        cfg["wheel_damping"] = actual[0]
    # Read the composed USD without starting Isaac. These are simulator limits,
    # not newly certified physical stops. Actual hardware calibration is separate.
    full_source = load_yaml(paired/"env.yaml")
    asset = Path(full_source["robot"]["spawn"]["usd_path"])
    stage = Usd.Stage.Open(str(asset))
    limits = {}
    for prim in stage.TraverseAll():
        if prim.GetName() in cfg["leg_joint_names"] and prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            limits[prim.GetName()] = [math.radians(joint.GetLowerLimitAttr().Get()),
                                      math.radians(joint.GetUpperLimitAttr().Get())]
    cfg["leg_joint_position_limits"] = [limits[name] for name in cfg["leg_joint_names"]]
    cfg["enable_external_forces_every_iteration"] = full_source["enable_external_forces_every_iteration"]
    cfg["upper_gas_spring_feedforward"] = False
    cfg["nominal_vehicle_mass_kg"] = 2*cfg["feedforward_force"]/9.81
    schema = make_schema(cfg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved["model_state_dict"]
    if any("normalizer" in k for k in state):
        raise ValueError("Normalization must not be silently omitted")
    dims = [27, *policy_spec["actor_hidden_dims"], 6]
    arrays, layers = {}, []
    for i, (ni, no) in enumerate(zip(dims[:-1], dims[1:])):
        weight, bias = state["actor.%d.weight" % (2*i)], state["actor.%d.bias" % (2*i)]
        if tuple(weight.shape) != (no, ni) or tuple(bias.shape) != (no,):
            raise ValueError("Unexpected actor tensor shape")
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Non-finite policy weights")
        arrays["w%d" % i], arrays["b%d" % i] = weight.numpy().astype(np.float32), bias.numpy().astype(np.float32)
        layers.append(torch.nn.Linear(ni, no))
        if i < len(dims)-2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict({k[6:]: v for k,v in state.items() if k.startswith("actor.")}, strict=True)
    actor.eval()
    output.mkdir(parents=True)
    np.savez(output/"actor_fp32.npz", **arrays)
    dump(output/"runtime_config.json", cfg)
    dump(output/"obs_schema.json", schema)
    config_sha, weights_sha = sha256(output/"runtime_config.json"), sha256(output/"actor_fp32.npz")
    metadata = dict(format=FORMAT, control_contract=CONTROL_CONTRACT,
                    checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                    iteration=int(saved["iter"]), actual_updates=int(saved["iter"])+1,
                    dimensions=dims, activation="elu", policy_sha256=weights_sha,
                    runtime_config_sha256=config_sha, schema_sha256=schema["sha256"],
                    contract_sha256=contract_digest(weights_sha,config_sha,schema["sha256"]),
                    env_sha256=sha256(paired/"env.yaml"), agent_sha256=sha256(paired/"agent.yaml"),
                    asset_sha256=sha256(asset), selected_controller_changes=changes,
                    policy_period_s=cfg["sim"]["dt"]*cfg["decimation"],
                    trained_physics_period_s=cfg["sim"]["dt"], host_control_default_hz=200,
                    execution_topology="NX computes one PD plus gravity; STM32 executes final Nm requests",
                    policy_precision_accepted=False, hardware_approved=False,
                    dynamics_gap="Implicit training PD versus explicit NX PD plus UART delay requires validation")
    dump(output/"manifest.json", metadata)
    # Synthetic numerical parity is export validation, not closed-loop acceptance.
    generator = np.random.default_rng(20261027)
    obs = generator.normal(0,.25,(256,27)).astype(np.float32)
    obs[:,3:6] = np.asarray([0,0,-1],np.float32)
    obs[:,6:9] = np.asarray([0,0,sum(cfg["commands"]["ranges_height"])/2*cfg["obs_scales"]["height_measurements"]],np.float32)
    with torch.inference_mode():
        expected = actor(torch.from_numpy(obs)).numpy()
    policy = NumpyPolicy(output)
    actual = np.asarray([policy(row.tolist()) for row in obs],np.float32)
    error = float(np.max(np.abs(actual-expected)))
    if error > 2.e-5:
        raise ValueError("NumPy/torch actor mismatch: %g" % error)
    np.savez(output/"synthetic_golden.npz", obs=obs, action=expected)
    dump(output/"export_verification.json", dict(checkpoint_sha256=checkpoint_sha,
         contract_sha256=metadata["contract_sha256"], synthetic_cases=len(obs), max_abs_error=error,
         checkpoints_or_Adam_modified=False, hardware_opened=False, motion_validation=False))
    return metadata


class NumpyPolicy:
    def __init__(self, folder):
        import numpy as np
        folder = Path(folder).expanduser().resolve()
        self.metadata = json.loads((folder/"manifest.json").read_text())
        self.cfg = json.loads((folder/"runtime_config.json").read_text())
        self.schema = json.loads((folder/"obs_schema.json").read_text())
        m = self.metadata
        if m["format"] != FORMAT or m["activation"] != "elu" or m.get("control_contract") != CONTROL_CONTRACT:
            raise ValueError("Unsupported NumPy policy format")
        if sha256(folder/"actor_fp32.npz") != m["policy_sha256"] or sha256(folder/"runtime_config.json") != m["runtime_config_sha256"]:
            raise ValueError("Policy/configuration hash mismatch")
        if make_schema(self.cfg) != self.schema or self.schema["sha256"] != m["schema_sha256"]:
            raise ValueError("Observation schema mismatch")
        if contract_digest(m["policy_sha256"],m["runtime_config_sha256"],m["schema_sha256"]) != m["contract_sha256"]:
            raise ValueError("Contract digest mismatch")
        dims = m["dimensions"]
        if dims[0] != 27 or dims[-1] != 6 or any(not isinstance(v,int) or not 0<v<=4096 for v in dims):
            raise ValueError("Invalid Actor dimensions")
        self.layers = []
        with np.load(folder/"actor_fp32.npz", allow_pickle=False) as saved:
            if set(saved.files) != {name+str(i) for i in range(len(dims)-1) for name in ('w','b')}:
                raise ValueError("Unexpected weight entries")
            for i, (ni,no) in enumerate(zip(dims[:-1],dims[1:])):
                w,b = saved['w%d'%i], saved['b%d'%i]
                if w.dtype != np.float32 or b.dtype != np.float32 or w.shape != (no,ni) or b.shape != (no,):
                    raise ValueError("Invalid FP32 weight tensor")
                if not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ValueError("Non-finite Actor weights")
                self.layers.append((w,b))

    def __call__(self, observation):
        import numpy as np
        x = np.asarray(validate_observation(observation,self.schema,self.cfg),np.float32)
        for i,(w,b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers)-1:
                negative = x < 0
                x[negative] = np.expm1(x[negative])
        return finite_vector(x.tolist(),6,"actor output")

    def bounded_action(self, observation):
        return clip_action(self(observation),self.cfg)
