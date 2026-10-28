"""Existing exploration audit with the saved wheel bounds."""
import argparse
import json
from pathlib import Path

from audit_saved_action_limits import audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    path, _ = audit('exploration', args.output, args.through)
    print(json.dumps(dict(report=str(path), read_only=True)))
