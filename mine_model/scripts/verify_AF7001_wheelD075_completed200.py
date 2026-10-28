"""Original complete200 verifier with the explicitly declared drive/count delta."""
import ast
import copy
from pathlib import Path

import plain27_sampler_metadata_adapter as adapter
import verify_completed_plain27_joint_pd_trial as original
import joint_pd_AF7001_wheelD075_runtime as runtime


SCRIPT = Path(__file__).resolve()


def forbidden_controller_change(plan):
    return not (plan['action_map_or_controller_changed'] is True
        and plan['action_map_changed'] is False
        and plan['next_configuration_changes'] == runtime.DELTA
        and plan['only_controller_change'] == runtime.DELTA['controller'])


def declared_configuration_differences(plan, before, after):
    count = plan['native_num_envs']
    assert count in (8192, 4096) and plan['fixed_standing_environments'] == count // 4
    assert before['wheel_damping'] == before['robot']['actuators']['wheels']['damping'] == .5
    assert after['wheel_damping'] == after['robot']['actuators']['wheels']['damping'] == .75
    assert plan['next_configuration_changes'] == runtime.DELTA
    assert before['rewards'] == after['rewards'] and before['commands'] == after['commands']
    result = {'/wheel_damping', '/robot/actuators/wheels/damping'}
    for key in ('scene', 'terrain'):
        assert before[key]['num_envs'] == 4096 and after[key]['num_envs'] == count
        if count != 4096:
            result.add('/' + key + '/num_envs')
    return result


def verify(directory, destination=None):
    directory = Path(directory).resolve()
    plan = original.json.loads((directory / 'experiment_plan.json').read_text())
    count = plan['native_num_envs']
    assert count == runtime.NUM_ENVS
    module = adapter.private_module(original.SCRIPT, '_wheelD075_original_completed200_verifier')
    module.SCRIPT = SCRIPT
    parsed = ast.parse(original.SCRIPT.read_text())
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'verify')
    tree = ast.Module(body=[function], type_ignores=[])
    before, changes = ast.dump(tree), []
    guard = ast.dump(ast.parse("plan['action_map_or_controller_changed']", mode='eval').body)
    counts = 0
    for node in ast.walk(function):
        if isinstance(node, ast.BoolOp):
            for index, child in enumerate(node.values):
                if ast.dump(child) == guard:
                    changes.append((node, index, child))
                    node.values[index] = ast.parse('forbidden_controller_change(plan)', mode='eval').body
                    counts += 1
    assert counts == 1
    indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'delta' for target in node.targets)
        and ast.unparse(node.value) == 'differences(cfg, current_cfg)']
    assert len(indices) == 1
    index = indices[0]
    function.body.insert(index, ast.parse('expected.update(declared_configuration_differences(plan, cfg, current_cfg))').body[0])
    labels = [node for node in ast.walk(function) if isinstance(node, ast.keyword)
              and node.arg == 'all200_all4096_alignment_max_errors']
    assert len(labels) == 1
    labels[0].arg = f'all200_all{count}_alignment_max_errors'
    modified = copy.deepcopy(tree)
    function.body.pop(index)
    labels[0].arg = 'all200_all4096_alignment_max_errors'
    for node, item, child in reversed(changes):
        node.values[item] = child
    assert ast.dump(tree) == before
    module.forbidden_controller_change = forbidden_controller_change
    module.declared_configuration_differences = declared_configuration_differences
    exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), 'exec'), module.__dict__)
    return module.verify(directory, destination)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    print({k: result[k] for k in ('verified', 'actual_ppo_updates', 'all17_Adam_step', 'saved_configuration_differences')})
