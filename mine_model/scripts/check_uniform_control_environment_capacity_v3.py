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
import joint_pd_uniform_control_capacity_runtime_v3 as backend
import json
import time


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


def _wait_for_probe_cgroup_release(folder, timeout_s=180.):
    run_path = Path(folder) / 'run.json'
    deadline = time.monotonic() + timeout_s
    control_group = None
    while time.monotonic() < deadline:
        if run_path.exists():
            try:
                record = json.loads(run_path.read_text())
                state = record.get('final_unit_state', {})
                control_group = state.get('ControlGroup') or record.get('unit_state', {}).get('ControlGroup')
            except (OSError, ValueError):
                pass
        if not control_group or not (Path('/sys/fs/cgroup') / control_group.lstrip('/')).exists():
            return dict(verified=True, control_group=control_group)
        time.sleep(.5)
    raise RuntimeError('Capacity probe cgroup did not drain before next count')


_original_run_probe = numeric.run_probe


def run_probe(request, request_path, output, count, policy):
    result = _original_run_probe(request, request_path, output, count, policy)
    result['cgroup_release'] = _wait_for_probe_cgroup_release(result['folder'])
    Path(result['folder'], 'cgroup_release_after_probe.json').write_text(
        json.dumps(result['cgroup_release'], ensure_ascii=False, indent=2) + '\n')
    return result


numeric.run_probe = run_probe


if __name__ == "__main__":
    main()
