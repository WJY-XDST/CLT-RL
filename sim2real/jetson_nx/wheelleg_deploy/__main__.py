"""NX Actor/torque/UART project plus retained offline reference tools."""

import argparse
import json
import sys
import time
from pathlib import Path

from .policy import BundlePolicy, prepare_bundle
from .protocol import OfflineJsonCodec
from .session import PolicySession
from .transport import MemoryTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze an explicit new-vehicle checkpoint and its exact contract")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    nx_prepare=commands.add_parser('prepare-nx',help='Workstation export: portable FP32 NumPy Actor and controller contract')
    nx_prepare.add_argument('--checkpoint',type=Path,required=True)
    nx_prepare.add_argument('--output',type=Path,required=True)
    nx_prepare.add_argument('--verified-replay',type=Path)
    nx_check=commands.add_parser('check-nx',help='Validate portable bundle, infer and calculate a synthetic torque vector')
    nx_check.add_argument('--bundle',type=Path,required=True)
    mcu=commands.add_parser('mcu-config',help='Print the C initializer matching the selected portable bundle')
    mcu.add_argument('--bundle',type=Path,required=True)
    bench=commands.add_parser('benchmark-nx',help='Run compute/scheduler timing on this machine; no UART or motor output')
    bench.add_argument('--bundle',type=Path,required=True)
    bench.add_argument('--duration',type=float,default=30.)
    bench.add_argument('--control-hz',type=int,default=200)
    bench.add_argument('--output',type=Path)
    nx_run=commands.add_parser('run-nx',help='Receive STM32 obs/raw sensors and send final motor torques; default shadow mode')
    nx_run.add_argument('--config',type=Path)
    nx_run.add_argument('--bundle',type=Path)
    nx_run.add_argument('--port')
    nx_run.add_argument('--baudrate',type=int)
    nx_run.add_argument('--control-hz',type=int)
    nx_run.add_argument('--state-timeout',dest='state_timeout_s',type=float)
    nx_run.add_argument('--execute',action='store_true',help='Request execution; compatible MCU must independently accept it')
    nx_run.add_argument('--log',type=Path)
    nx_run.add_argument('--duration',type=float,help='0 runs until interruption')
    nx_run.add_argument('--command',dest='motion_command',type=float,nargs=3,metavar=('VX','YAW','HEIGHT'))
    check = commands.add_parser("check", help="Check bundle hashes and run one synthetic inference (no UART)")
    check.add_argument("--bundle", type=Path, required=True)
    dry = commands.add_parser("dry-run", help="Offline byte-loopback obs -> policy -> reference requests")
    dry.add_argument("--bundle", type=Path, required=True)
    dry.add_argument("--obs", type=Path, help="JSON {obs:[27 scaled values]}; defaults to clearly labelled synthetic sample")
    dry.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.command=='benchmark-nx':
            from .benchmark import benchmark
            print(json.dumps(benchmark(args.bundle,duration_s=args.duration,
                  control_hz=args.control_hz,output=args.output),ensure_ascii=False,indent=2))
            return 0
        if args.command=='prepare-nx':
            from .numpy_policy import prepare_numpy_bundle
            print(json.dumps(prepare_numpy_bundle(args.checkpoint,args.output,verified_replay=args.verified_replay),ensure_ascii=False,indent=2))
            return 0
        if args.command in ('check-nx','run-nx','mcu-config'):
            from .numpy_policy import NumpyPolicy
            from .nx_runtime import run_serial,warmup
            if args.command=='run-nx':
                from .configuration import run_options
                print(json.dumps(run_serial(**run_options(args)),indent=2))
                return 0
            policy=NumpyPolicy(args.bundle)
            if args.command=='mcu-config':
                from .configuration import mcu_initializer
                print(mcu_initializer(policy))
                return 0
            warmup(policy)
            from .torque_control import TorqueController
            obs=[0.]*27
            obs[5]=-1.
            obs[8]=sum(policy.cfg['commands']['ranges_height'])/2*policy.cfg['obs_scales']['height_measurements']
            raw=policy.cfg['leg_joint_position_offset']+[0.]*6
            result=TorqueController(policy).step(obs,raw,infer=True)
            print(json.dumps(dict(status='synthetic_actor_and_torque_ok',**result),indent=2))
            return 0
        if args.command == "prepare":
            print(json.dumps(prepare_bundle(args.checkpoint, args.output), ensure_ascii=False, indent=2))
            return 0
        import torch
        torch.set_num_threads(1)
        policy = BundlePolicy(args.bundle)
        obs_file = getattr(args, "obs", None) or args.bundle / "synthetic_obs.json"
        values = json.loads(obs_file.read_text())["obs"]
        # Warm up CPU kernels BEFORE creating a timed session.
        policy(values)
        if args.command == "check":
            print(json.dumps(dict(status="offline_bundle_ok", iteration=policy.metadata["iteration"],
                                  raw_action=policy(values), hardware_approved=False), indent=2))
            return 0
        if not 1 <= args.steps <= 10000:
            raise ValueError("steps must be 1..10000")
        session, rx, tx, wire = PolicySession(policy), OfflineJsonCodec(), OfflineJsonCodec(), MemoryTransport()
        for index in range(args.steps):
            message = dict(kind="observation", seq=index, device_time_us=index*round(policy.metadata["policy_period_s"]*1e6),
                           schema_sha256=policy.schema["sha256"], obs=values)
            frame = rx.encode(message)
            incoming = rx.feed(frame[:17]) + rx.feed(frame[17:])
            response = session.handle(incoming[0])
            wire.write(tx.encode(response))
            print(json.dumps(tx.feed(wire.sent[-1])[0], ensure_ascii=False), flush=True)
            if response["kind"] != "reference_request":
                return 2
            # A repeated synthetic observation is a pipeline test, NOT closed-
            # loop motion or a test of the MCU's effective previous action.
            time.sleep(policy.metadata["policy_period_s"])
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
