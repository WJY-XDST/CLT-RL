"""Extend the old frozen common-gain diagnostic to a single scalar <=6.

Only its old local probe bound changes from2 to6. The original replay,
PhysX-value verification, mean actor, limits, reward and assessment code remain
identical. This file has no PPO, optimizer, deployment or promotion entry point.
"""
import ast
import copy
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'mine_model/scripts/replay_joint_pd_wheel_gain.py'
EXPECTED_SOURCE_SHA = 'f832e6eddd0d84f56767be408892a882c8417b751b42b972254bb553c9e964bb'


def qualified_tree():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA
    tree = ast.parse(SOURCE.read_text())
    original = ast.dump(tree, include_attributes=False)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
               and ast.unparse(node.test) == 'math.isfinite(gain) and 0.5 <= gain <= 2.0']
    assert len(matches) == 1, 'The original diagnostic guard changed'
    comparison = matches[0].test.values[1]
    assert isinstance(comparison.comparators[-1], ast.Constant)
    comparison.comparators[-1].value = 6.0
    changed = copy.deepcopy(tree)
    comparison.comparators[-1].value = 2.0
    assert ast.dump(tree, include_attributes=False) == original
    return ast.fix_missing_locations(changed)


if __name__ == '__main__':
    # The original file location is retained only to resolve its existing
    # application imports. The new adapter is separately pinned by the request.
    exec(compile(qualified_tree(), str(SOURCE), 'exec'),
         dict(__name__='__main__', __file__=str(SOURCE)))
