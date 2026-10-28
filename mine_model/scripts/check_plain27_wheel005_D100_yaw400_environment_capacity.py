"""Original five-update capacity gates with the declared reward expectation.

Keep both counts, all17Adam, alignment and resource tests. Extend only the
saved-config expectation to the explicitly selected existing yaw coefficient.
"""
import ast
import copy
from pathlib import Path

import check_plain27_wheel005_D100_environment_capacity as retained
import joint_pd_wheel005_source7201_D100_yaw400_capacity_runtime as backend


SCRIPT = Path(__file__).resolve()
numeric = retained.original.original
retained.retained.backend = backend
numeric.probe_command = retained.retained.probe_command
numeric.BACKEND, numeric.SCRIPT = backend.SCRIPT, SCRIPT
tree = copy.deepcopy(retained.original.modified)
before = ast.dump(tree)
function = tree.body[0]
matches = [i for i, node in enumerate(function.body) if isinstance(node, ast.If)
           and ast.dump(node.test) == ast.dump(ast.parse('after != expected', mode='eval').body)]
assert len(matches) == 1
index = matches[0]
added = ast.parse("expected['rewards']['yaw_rate_error_sq'] = request['proposed_reward_overrides']['rewards.yaw_rate_error_sq']").body
function.body[index:index] = added
labels = [node for node in ast.walk(function) if isinstance(node, ast.keyword)
          and node.arg == 'only_declared_wheel_drive_count_and_output_configuration_changed']
assert len(labels) == 1
labels[0].arg = 'only_declared_yaw_reward_wheel_drive_count_and_output_configuration_changed'
modified = copy.deepcopy(tree)
del function.body[index:index + len(added)]
labels[0].arg = 'only_declared_wheel_drive_count_and_output_configuration_changed'
assert ast.dump(tree) == before, 'Keep every original numerical and resource gate'
exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), 'exec'), numeric.__dict__)
require_capacity_selection, main = numeric.require_capacity_selection, numeric.main


if __name__ == '__main__':
    main()
