"""Keep original five-update resource/Adam/alignment checks for one drive D.

The saved-config expectation includes exactly the two synchronized fields
implementing this one controller factor. No numerical or resource gate moves.
"""
import ast
import copy
from pathlib import Path

import check_plain27_leg005_environment_capacity_v7 as retained
import check_plain27_environment_capacity as original
import joint_pd_AF7001_wheelD075_capacity_runtime as backend


SCRIPT = Path(__file__).resolve()
retained.backend = backend
original.probe_command = retained.probe_command
original.BACKEND, original.SCRIPT = backend.SCRIPT, SCRIPT
tree = copy.deepcopy(retained.retained.modified)
before = ast.dump(tree)
function = tree.body[0]
indices = [index for index, node in enumerate(function.body) if isinstance(node, ast.If)
           and ast.dump(node.test) == ast.dump(ast.parse('after != expected', mode='eval').body)]
assert len(indices) == 1
index = indices[0]
added = ast.parse("expected['wheel_damping'] = request['proposed_controller_overrides']['wheel_damping']\n"
                  "expected['robot']['actuators']['wheels']['damping'] = request['proposed_controller_overrides']['robot.actuators.wheels.damping']").body
function.body[index:index] = added
labels = [node for node in ast.walk(function) if isinstance(node, ast.keyword)
          and node.arg == 'only_environment_count_and_output_configuration_changed']
assert len(labels) == 1
labels[0].arg = 'only_declared_wheel_drive_count_and_output_configuration_changed'
modified = copy.deepcopy(tree)
del function.body[index:index + len(added)]
labels[0].arg = 'only_environment_count_and_output_configuration_changed'
assert ast.dump(tree) == before, 'Original numerical/resource verifier must remain intact'
exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), 'exec'), original.__dict__)
require_capacity_selection, main = original.require_capacity_selection, original.main


if __name__ == '__main__':
    main()
