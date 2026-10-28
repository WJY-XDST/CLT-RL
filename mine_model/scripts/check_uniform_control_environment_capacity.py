# Versioned global-control workflow: unchanged numerical readers and ordinary
# PPO; controller differences must be explicitly declared in the immutable recipe.
"""Reuse the original capacity scorer with a declared global controller scalar.

The scorer, PPO updates and tracking gates remain unchanged. Both wheel-drive
damping settings and environment counts are explicit configuration inputs.
Source clocks come from each request, rather than a historical update number.
"""
import ast
import copy
from pathlib import Path

import check_plain27_environment_capacity_v6 as retained
import check_plain27_environment_capacity as numeric
import check_plain27_environment_capacity_v5 as verifier
import capacity_passive_resource_profile as passive
import joint_pd_uniform_control_capacity_runtime as backend


SCRIPT = Path(__file__).resolve()
retained.backend = backend
numeric.BACKEND, numeric.SCRIPT = backend.SCRIPT, SCRIPT
tree = copy.deepcopy(verifier.modified)
before = ast.dump(tree)
body = tree.body[0].body
indices = [index for index, node in enumerate(body) if isinstance(node, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "expected" for target in node.targets)]
assert len(indices) == 1
index = indices[0] + 1
insert = ast.parse("""
assert before.get('wheel_action_limit', 1.0) == after['wheel_action_limit'] == 1.0
expected.setdefault('wheel_action_limit', 1.0)
for key, value in request['reward_overrides'].items():
    assert key in expected['rewards']
    expected['rewards'][key] = value
controls = request['control_overrides']
assert set(controls) == {'wheel_damping', 'robot.actuators.wheels.damping'}
assert controls['wheel_damping'] == controls['robot.actuators.wheels.damping'] > 0.
expected['wheel_damping'] = controls['wheel_damping']
expected['robot']['actuators']['wheels']['damping'] = controls['robot.actuators.wheels.damping']
""").body
body[index:index] = insert
modified = copy.deepcopy(tree)
del body[index:index + len(insert)]
assert ast.dump(tree) == before
exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), "exec"), numeric.__dict__)
_verify = numeric.verify_finished_probe


def verify_finished_probe(folder, request, count):
    result = _verify(folder, request, count)
    result.update(only_environment_count_and_output_configuration_changed=False,
        declared_global_reward_overrides=request["reward_overrides"],
        declared_global_control_overrides=request["control_overrides"],
        only_declared_global_scalars_count_and_output_changed=True)
    return result


numeric.verify_finished_probe = verify_finished_probe
passive.install(numeric)
require_capacity_selection, main = numeric.require_capacity_selection, numeric.main


if __name__ == "__main__":
    main()
