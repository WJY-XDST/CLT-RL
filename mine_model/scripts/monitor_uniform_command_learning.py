"""Reuse all11 learning readers with the actual configured standing count."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path

import joint_pd_uniform_command_runtime as runtime
import monitor_uniform_control_joint_pd_learning_v3 as original
import plain27_sampler_metadata_adapter as adapter
from audit_uniform_motion_milestone import audit as milestone_audit

SCRIPT = Path(__file__).resolve()


def implementation():
    tree = ast.parse(inspect.getsource(original.run))
    initial = ast.dump(tree)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and n.value == 'compare_uniform_control_recorded_joint_targets_v3.py']
    assert len(nodes) == 1
    nodes[0].value = 'compare_uniform_command_recorded_joint_targets.py'
    changed = copy.deepcopy(tree)
    nodes[0].value = 'compare_uniform_control_recorded_joint_targets_v3.py'
    assert ast.dump(tree) == initial
    loops = [n for n in ast.walk(changed) if isinstance(n, ast.While)]
    assert len(loops) == 1
    body = loops[0].body
    points = [i for i, n in enumerate(body) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'record' for t in n.targets)]
    assert len(points) == 1
    insert = ast.parse("""
for milestone in (50, 100, 150, 200):
    completed = record['committed_native_updates']
    ready = completed > milestone or (milestone == 200 and record['state'] == 'all_owned_prefix_diagnostics_completed')
    path = output / f'agent_review_milestone_{milestone:04d}_readonly.json'
    if ready and not path.exists():
        milestone_audit(output, milestone)
""").body
    body[points[0] + 1:points[0] + 1] = insert
    scope = dict(original.__dict__, SCRIPT=SCRIPT, milestone_audit=milestone_audit)
    exec(compile(ast.fix_missing_locations(changed), str(SCRIPT), 'exec'), scope)
    return scope['run']


def run(output, recipe):
    runtime.configure(recipe)
    plan = json.loads((output / 'experiment_plan.json').read_text())
    standing = runtime.standing_count(plan['native_num_envs'])
    assert standing == plan['fixed_standing_environments']
    assert standing == int(plan['native_num_envs'] * plan['training_config']['commands']['standing_env_fraction'])
    old = adapter.retarget

    def retarget(module, names, **kwargs):
        if names == ('tick',):
            kwargs = copy.deepcopy(kwargs)
            assert kwargs['integers']['*'][2048] == plan['native_num_envs'] // 4
            kwargs['integers']['*'][2048] = standing
        return old(module, names, **kwargs)

    adapter.retarget = retarget
    try:
        return implementation()(output, recipe)
    finally:
        adapter.retarget = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.recipe.resolve())
    except BaseException as error:
        original.write_json(args.output / 'uniform_learning_inspection_required.json',
                            dict(error=repr(error), training_modified=False, policy_accepted=False))
        raise
