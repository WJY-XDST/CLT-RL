"""The current flat policy's wire-level observation contract, not a sensor driver."""

import hashlib
import json
import math

OBS_DIM, ACTION_DIM = 27, 6
ACTION_NAMES = ["LB_position", "LL_position", "left_wheel", "RB_position", "RL_position", "right_wheel"]

# Start, names, physical units, training clip key, training scale key.
BLOCKS = [
    (0, ["gyro_x", "gyro_y", "gyro_z"], "rad/s, body frame", "ang_vel", "ang_vel"),
    (3, ["gravity_x", "gravity_y", "gravity_z"], "unit gravity, body frame", "projected_gravity", None),
    (9, ["q_LB_offset", "q_LL_offset", "q_RB_offset", "q_RL_offset"], "rad, raw CAD motor angle minus nominal offset", "theta", "dof_pos"),
    (13, ["dq_LB", "dq_LL", "dq_RB", "dq_RL"], "rad/s, raw CAD motor axes", "theta_dot", "dof_vel"),
    (17, ["vx", "vy"], "m/s, horizontal yaw-aligned frame", "lin_vel", "lin_vel"),
    (19, ["wheel_left", "wheel_right"], "rad/s, BOTH forward-positive", "wheel_vel", "dof_vel"),
    (21, ["previous_" + name for name in ACTION_NAMES], "effective applied normalized reference", "action", None),
]


def finite_vector(values, size, name):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} requires exactly {size} scalar values")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} requires finite numbers; NaN/Inf/bools are rejected")
    return [float(v) for v in values]


def make_schema(cfg):
    if cfg.get('leg_control_mode') != 'implicit_joint_position':
        raise ValueError('Only the direct-joint position schema is supported; retired IK bundles must not be reused')
    fields = []
    for start, names, unit, clip_key, scale_key in BLOCKS:
        scale = float(cfg["obs_scales"][scale_key]) if scale_key else 1.0
        bound = float(cfg["obs_clip"][clip_key])
        for index, name in enumerate(names, start):
            fields.append(dict(index=index, name=name, physical_unit=unit,
                               scale=scale, physical_clip=bound))
    for index, name, unit, key in [(6, "command_vx", "m/s", "lin_vel"),
                                   (7, "command_yaw_rate", "rad/s", "ang_vel"),
                                   (8, "command_height", "m", "height_measurements")]:
        fields.append(dict(index=index, name=name, physical_unit=unit,
                           scale=float(cfg["obs_scales"][key]), physical_clip=None))
    schema = dict(version="mine_joint_position_obs27_v2", observation_dimension=OBS_DIM,
                  action_dimension=ACTION_DIM, action_order=ACTION_NAMES,
                  fields=sorted(fields, key=lambda f: f["index"]),
                  preprocessing="MCU clips physical blocks THEN multiplies scale, exactly once; no training noise",
                  previous_action="Re-encode the actual clipped motor reference as (q_target-offset)/scale; wheel actions are clipped normalized references; never use an unclipped actor mean")
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    schema["sha256"] = hashlib.sha256(encoded).hexdigest()
    return schema


def validate_observation(values, schema, cfg):
    values = finite_vector(values, OBS_DIM, "obs")
    for field in schema["fields"]:
        clip = field["physical_clip"]
        if clip is not None and abs(values[field["index"]]) > clip * field["scale"] + 1e-5:
            raise ValueError(f"obs[{field['index']}] exceeds the agreed scaled bound")
    g_norm = sum(v*v for v in values[3:6]) ** .5
    if not .9 <= g_norm <= 1.1:
        raise ValueError("Projected gravity must be a unit vector, not acceleration in m/s^2")
    commands = [values[i] / schema["fields"][i]["scale"] for i in (6, 7, 8)]
    ranges = [cfg["commands"]["ranges_lin_vel_x"], cfg["commands"]["ranges_reverse_lin_vel_x"]]
    x_min, x_max = min(0., *(r[0] for r in ranges)), max(0., *(r[1] for r in ranges))
    yaw_min, yaw_max = cfg["commands"]["ranges_ang_vel_yaw"]
    height_min, height_max = cfg["commands"]["ranges_height"]
    for value, lower, upper in zip(commands, (x_min, yaw_min, height_min), (x_max, yaw_max, height_max)):
        if not lower - 1e-6 <= value <= upper + 1e-6:
            raise ValueError("Command is outside the selected checkpoint's training range")
    return values
