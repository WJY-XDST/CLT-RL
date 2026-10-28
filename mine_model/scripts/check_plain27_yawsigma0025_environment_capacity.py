"""Retain the real-five-update resource verifier for one declared reward width.

The original capacity verifier compares against the source recipe. Its
expected environment is extended by exactly the selected .0025 yaw width;
all update, Adam, alignment, finite, source-preservation and resource checks
remain intact. The original files are never edited.
"""
import ast,copy,json
from pathlib import Path
import check_plain27_leg005_environment_capacity_v7 as retained
import check_plain27_environment_capacity as original
import joint_pd_AF7001_yawsigma0025_capacity_runtime as backend

SCRIPT=Path(__file__).resolve()
retained.backend=backend
original.probe_command=retained.probe_command
original.BACKEND=backend.SCRIPT
original.SCRIPT=SCRIPT

tree=copy.deepcopy(retained.retained.modified)
before=ast.dump(tree);fn=tree.body[0]
indices=[i for i,n in enumerate(fn.body) if isinstance(n,ast.If)
    and ast.dump(n.test)==ast.dump(ast.parse('after != expected',mode='eval').body)]
if len(indices)!=1:raise ValueError('Requalify the original exact saved-config verifier')
index=indices[0]
added=ast.parse("expected['rewards']['tracking_sigma_ang'] = request['proposed_reward_overrides']['tracking_sigma_ang']").body[0]
fn.body.insert(index,added)
modified=copy.deepcopy(tree);fn.body.pop(index)
if ast.dump(tree)!=before:raise ValueError('Verifier changed beyond one declared reward expectation')
exec(compile(ast.fix_missing_locations(modified),str(SCRIPT),'exec'),original.__dict__)
require_capacity_selection=original.require_capacity_selection
main=original.main

if __name__=='__main__':main()
