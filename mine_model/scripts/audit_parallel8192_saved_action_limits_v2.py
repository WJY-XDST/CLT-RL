"""Original clipped-distribution KL with8192 admission and saved action bounds.

Only the declared population comparison, its label, and the previously
qualified wheel-bound normalization are adapted. Numerical formulas,
checkpoint continuity, RNG, and independent quadrature remain unchanged.
"""
import argparse
import ast
import json
from pathlib import Path

from audit_saved_action_limits import transform
from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import audit_clipped_joint_pd_policy_kl as original

SCRIPT = Path(__file__).resolve()


def audit(output, through, selected):
    import numpy as np

    output = Path(output).resolve()
    plan_path = output / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    limit = checkpoint_wheel_action_limit(plan['training_config'])
    bounds = np.array([1., 1., limit, 1., 1., limit])
    tree = transform(original.audit, [
        ("plan['native_num_envs'] != 4096", "plan['native_num_envs'] != 8192"),
        ("'Require the qualified4096/plain27 direct-joint execution contract'",
         "'Require the qualified8192/plain27 direct-joint execution contract'"),
        ("torch.zeros((4096,27),device='cuda')", "torch.zeros((8192,27),device='cuda')"),
        ("'Native4096-row TF32 source distribution not reproduced exactly'",
         "'Native8192-row TF32 source distribution not reproduced exactly'"),
        ('clipped_kl(m0, s0, m1, s1)',
         'clipped_kl(m0 / HEAD_LIMITS, s0 / HEAD_LIMITS, m1 / HEAD_LIMITS, s1 / HEAD_LIMITS)'),
        ('[x.ravel() for x in (m0,s0,m1,s1)]',
         '[(x / HEAD_LIMITS).ravel() for x in (m0,s0,m1,s1)]'),
        ('np.clip(m1, -1., 1.)', 'np.clip(m1, -HEAD_LIMITS, HEAD_LIMITS)'),
        ('np.clip(m0, -1., 1.)', 'np.clip(m0, -HEAD_LIMITS, HEAD_LIMITS)'),
    ])
    scope = dict(original.__dict__, HEAD_LIMITS=bounds, __file__=str(SCRIPT))
    exec(compile(ast.fix_missing_locations(tree), str(original.__file__), 'exec'), scope)
    path, report = scope['audit'](output, through, selected)
    report.update(saved_action_limits_by_head=bounds.tolist(), wheel_action_limit=limit,
        native_num_envs=8192, original_formula_and_all_other_AST_restored_identical=True,
        scalar_unit_clamp_not_assumed_for_wheel_heads=True)
    report['source_pins'].update({str(p):sha(p) for p in
        (SCRIPT, Path(original.__file__), plan_path)})
    if any(sha(p) != h for p, h in report['source_pins'].items()):
        raise ValueError('Preserve every8192 saved-policy and diagnostic input')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return path, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--updates', type=int, nargs='+', required=True)
    args = parser.parse_args()
    path, _ = audit(args.output, args.through, args.updates)
    print(json.dumps(dict(report=str(path), read_only=True, native_num_envs=8192)))
