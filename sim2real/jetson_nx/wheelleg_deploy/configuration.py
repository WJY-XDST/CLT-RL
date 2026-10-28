"""Portable run configuration and the MCU settings for this exact Actor."""
import json
from pathlib import Path


DEFAULTS = dict(baudrate=2000000,control_hz=200,state_timeout_s=.03,
                duration_s=0.,log_path=None,command_override=None)
KEYS = {"bundle","port","baudrate","control_hz","state_timeout_s","duration_s",
        "log_path","command_override"}


def run_options(args):
    result = dict(DEFAULTS)
    if args.config is not None:
        path = args.config.expanduser().resolve()
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved,dict) or set(saved)-KEYS:
            raise ValueError("Unknown run configuration fields")
        result.update(saved)
        for key in ("bundle","log_path"):
            value=result.get(key)
            if value is not None:
                if not isinstance(value,str) or not value:
                    raise ValueError(key+" must be a nonempty path")
                local=Path(value).expanduser()
                result[key]=local if local.is_absolute() else path.parent/local
    for key in ("bundle","port","baudrate","control_hz","state_timeout_s","duration_s","log_path","command_override"):
        value=getattr(args,dict(duration_s="duration",log_path="log",
                               command_override="motion_command").get(key,key),None)
        if value is not None:
            result[key]=value
    if not result.get("bundle") or not isinstance(result.get("port"),str) or not result["port"]:
        raise ValueError("Explicit --bundle and --port required, directly or in --config")
    # Execution can only be requested explicitly on the command line. A copied
    # config cannot silently turn a shadow session into a torque execution run.
    result["execute"]=args.execute
    return result


def mcu_settings(policy):
    cfg=policy.cfg
    ranges=[cfg["commands"]["ranges_lin_vel_x"],cfg["commands"]["ranges_reverse_lin_vel_x"]]
    return dict(contract_sha256=policy.metadata["contract_sha256"],
                torque_limit_nm=[cfg["leg_effort_limit"],cfg["leg_effort_limit"],cfg["wheel_effort_limit"],
                                 cfg["leg_effort_limit"],cfg["leg_effort_limit"],cfg["wheel_effort_limit"]],
                action_limit=[1.,1.,cfg["wheel_action_limit"],1.,1.,cfg["wheel_action_limit"]],
                command_min=[min(0.,*(r[0] for r in ranges)),cfg["commands"]["ranges_ang_vel_yaw"][0],cfg["commands"]["ranges_height"][0]],
                command_max=[max(0.,*(r[1] for r in ranges)),cfg["commands"]["ranges_ang_vel_yaw"][1],cfg["commands"]["ranges_height"][1]],
                observation_action_limit=cfg["obs_clip"]["action"],
                command_timeout_us=30000,max_source_age_us=30000)


def mcu_initializer(policy):
    cfg=mcu_settings(policy)
    def floats(values):
        result=[]
        for value in values:
            text=format(float(value),".9g")
            if "." not in text and "e" not in text.lower():
                text+=".0"
            result.append(text+"f")
        return "{"+", ".join(result)+"}"
    lines=["/* Generated from the selected bundle. Physical axes/zeros still require alignment. */",
           "static const RLHost_Config rl_host_config = {",
           "    {"+", ".join("0x%02x"%b for b in bytes.fromhex(cfg["contract_sha256"]))+"},"]
    for key in ("torque_limit_nm","action_limit","command_min","command_max"):
        lines.append("    "+floats(cfg[key])+", /* "+key+" */")
    lines += ["    "+floats([cfg["observation_action_limit"]])[1:-1]+",",
              "    %du, %du"%(cfg["command_timeout_us"],cfg["max_source_age_us"]),"};"]
    return "\n".join(lines)
