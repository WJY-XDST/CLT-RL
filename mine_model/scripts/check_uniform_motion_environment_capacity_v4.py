"""Keep all original capacity gates; admit only declared global command fields."""
import ast
import copy
from pathlib import Path

import joint_pd_uniform_motion_capacity_runtime_v4 as backend
import check_uniform_control_environment_capacity_v3 as original


SCRIPT = Path(__file__).resolve()
numeric = original.numeric
numeric.BACKEND, numeric.SCRIPT = backend.SCRIPT, SCRIPT
original.retained.backend = backend
tree = copy.deepcopy(original.modified)
before = ast.dump(tree)
body = tree.body[0].body
indices = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "expected" for target in node.targets)]
assert len(indices) == 1
index = indices[0]+1
insert = ast.parse("""
commands = request['command_overrides']
assert set(commands) == {'training_motion_scale', 'yaw_min_abs_rate'}
expected['commands'].update(commands)
""").body
body[index:index] = insert
qualified_tree = copy.deepcopy(tree)
del body[index:index+len(insert)]
assert ast.dump(tree) == before
exec(compile(ast.fix_missing_locations(qualified_tree), str(SCRIPT), "exec"), numeric.__dict__)
qualified_verify = numeric.verify_finished_probe


def verify_finished_probe(folder, request, count):
    result = qualified_verify(folder, request, count)
    result.update(declared_global_command_overrides=request["command_overrides"],
                  original_capacity_numeric_gates_and_full_Adam_checks_unchanged=True,
                  original_verifier_AST_restored_after_metadata_insertion=True)
    return result


numeric.verify_finished_probe = verify_finished_probe
require_capacity_selection = numeric.require_capacity_selection
main = numeric.main

if __name__ == "__main__":
    main()
