"""Prepare/check/dry-run tools only. No real serial drive command is exposed."""

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
    check = commands.add_parser("check", help="Check bundle hashes and run one synthetic inference (no UART)")
    check.add_argument("--bundle", type=Path, required=True)
    dry = commands.add_parser("dry-run", help="Offline byte-loopback obs -> policy -> reference requests")
    dry.add_argument("--bundle", type=Path, required=True)
    dry.add_argument("--obs", type=Path, help="JSON {obs:[27 scaled values]}; defaults to clearly labelled synthetic sample")
    dry.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    try:
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


if __name__ == "__main__":
    raise SystemExit(main())
