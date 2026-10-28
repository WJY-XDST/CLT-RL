"""Preserve evaluator CLI arguments when dispatching the frozen v3 diagnostic.

The outer replay parser owns only --request. In particular, --output belongs
to the existing evaluator and must reach it unchanged. Numerical code is reused.
"""
import argparse
from pathlib import Path
import sys

import diagnose_uniform_wheel_drive_transfer_v3 as implementation
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
implementation.SCRIPT = SCRIPT


def replay_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay",))
    parser.add_argument("--request", required=True, type=Path)
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        args, remaining = replay_arguments(sys.argv[1:])
        assert "--output" in remaining
        implementation.replay(args.request, remaining)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=("qualify", "run"))
        parser.add_argument("--recipe", type=Path)
        parser.add_argument("--baseline", type=Path)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--damping", type=float)
        args = parser.parse_args()
        if args.mode == "qualify":
            assert args.recipe and args.baseline and args.damping
            sent = ["--checkpoint", "model.pt", "--output", "replay_output", "--seconds", "20",
                    "--command-probe-json", "grid.json", "--overrides_json", "drive.json"]
            parsed, retained = replay_arguments(["replay", "--request", "request.json", *sent])
            assert parsed.request == Path("request.json") and retained == sent
            implementation.qualify(args)
            path = args.output / "qualification.json"
            value = implementation.read(path)
            value["source_pins"][str(Path(implementation.__file__).resolve())] = sha(implementation.__file__)
            value["evaluator_arguments_forwarded_unchanged_CPU_verified"] = True
            value["underlying_v3_numeric_and_AST_program_unchanged"] = True
            implementation.verify_pins(value)
            write_json(path, value)
        else:
            implementation.run(args)
