"""Reuse all11 numerical readers; route only the global-command metadata audit."""
import argparse
import ast
import copy
import inspect
from pathlib import Path
import time

import joint_pd_uniform_motion_envelope_runtime_v4
import monitor_uniform_control_joint_pd_learning_v3 as original
from audit_uniform_motion_milestone import audit as milestone_audit


SCRIPT = Path(__file__).resolve()
tree = ast.parse(inspect.getsource(original.run))
before = ast.dump(tree)
nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
         and node.value == "compare_uniform_control_recorded_joint_targets_v3.py"]
assert len(nodes) == 1
nodes[0].value = "compare_uniform_motion_recorded_joint_targets_v4.py"
qualified = copy.deepcopy(tree)
nodes[0].value = "compare_uniform_control_recorded_joint_targets_v3.py"
assert ast.dump(tree) == before
# The original11 at50/100/200 are retained. Add the user's150 milestone
# without changing any existing reader's supported prefixes or formulas.
loop_nodes=[node for node in ast.walk(qualified) if isinstance(node,ast.While)]
assert len(loop_nodes)==1
loop=loop_nodes[0].body
points=[i for i,node in enumerate(loop) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='record' for t in node.targets)]
assert len(points)==1
insert=ast.parse("""
for milestone in (50,100,150,200):
    completed=record['committed_native_updates']
    ready=completed>milestone or (milestone==200 and record['state']=='all_owned_prefix_diagnostics_completed')
    path=output/f'agent_review_milestone_{milestone:04d}_readonly.json'
    if ready and not path.exists():
        milestone_audit(output,milestone)
""").body
loop[points[0]+1:points[0]+1]=insert
scope = dict(original.__dict__, SCRIPT=SCRIPT,milestone_audit=milestone_audit)
exec(compile(ast.fix_missing_locations(qualified), str(SCRIPT), "exec"), scope)
run = scope["run"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.recipe.resolve())
    except BaseException as error:
        original.write_json(args.output/"uniform_learning_inspection_required.json",
                            dict(error=repr(error), training_modified=False, policy_accepted=False))
        raise
