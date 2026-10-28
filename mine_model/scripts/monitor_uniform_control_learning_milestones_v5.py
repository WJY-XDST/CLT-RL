"""Keep the original11 readers and add read-only50/100/150/200 snapshots.

This is the scalar-only workflow. It does not import the global-command
transition runtime, change a numeric reader, or run an optimizer update.
"""
import argparse
import ast
import copy
import inspect
from pathlib import Path

import monitor_uniform_control_joint_pd_learning_v3 as original
from audit_uniform_motion_milestone import audit as milestone_audit


SCRIPT = Path(__file__).resolve()
tree = ast.parse(inspect.getsource(original.run))
before = ast.dump(tree)
loops = [node for node in ast.walk(tree) if isinstance(node, ast.While)]
assert len(loops) == 1
body = loops[0].body
points = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == 'record' for t in node.targets)]
assert len(points) == 1
insert = ast.parse("""
for milestone in (50, 100, 150, 200):
    completed = record['committed_native_updates']
    ready = completed > milestone or (milestone == 200 and record['state'] == 'all_owned_prefix_diagnostics_completed')
    path = output / f'agent_review_milestone_{milestone:04d}_readonly.json'
    if ready and not path.exists():
        milestone_audit(output, milestone)
""").body
at = points[0] + 1
body[at:at] = insert
restored = copy.deepcopy(tree)
restored_loop = [node for node in ast.walk(restored) if isinstance(node, ast.While)][0]
del restored_loop.body[at:at + len(insert)]
assert ast.dump(restored) == before
scope = dict(original.__dict__, SCRIPT=SCRIPT, milestone_audit=milestone_audit)
exec(compile(ast.fix_missing_locations(tree), str(SCRIPT), 'exec'), scope)
run = scope['run']


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
