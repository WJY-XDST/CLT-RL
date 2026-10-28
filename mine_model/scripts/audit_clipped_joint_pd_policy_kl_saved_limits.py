"""Existing mixed clipped-distribution KL at the saved per-head limits."""
import argparse
import json
from pathlib import Path

from audit_saved_action_limits import audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--updates', type=int, nargs='+', required=True)
    args = parser.parse_args()
    path, _ = audit('kl', args.output, args.through, args.updates)
    print(json.dumps(dict(report=str(path), read_only=True)))
