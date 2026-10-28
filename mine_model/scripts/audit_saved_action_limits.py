"""Use saved per-head execution limits in existing read-only distribution audits.

KL is invariant when each compared distribution is divided by the same
positive bound. Thus the original unit-bound atom/interior formula and its
independent quadrature apply unchanged to normalized mean/std values.
"""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import audit_effective_joint_pd_exploration as exploration
import audit_clipped_joint_pd_policy_kl as kl

SCRIPT = Path(__file__).resolve()


def transform(function, declarations):
    original = ast.parse(inspect.getsource(function))
    specs = [(ast.dump(ast.parse(before, mode='eval').body),
        ast.parse(before, mode='eval').body, ast.parse(after, mode='eval').body)
        for before, after in declarations]
    counts = [0] * len(specs)

    class Replace(ast.NodeTransformer):
        def visit(self, node):
            dump = ast.dump(node)
            for index, (before, _, after) in enumerate(specs):
                if dump == before:
                    counts[index] += 1
                    return ast.copy_location(copy.deepcopy(after), node)
            return super().visit(node)

    changed = Replace().visit(copy.deepcopy(original))
    if counts != [1] * len(specs):
        raise ValueError('Requalify only the declared saved-action-bound expressions: ' + repr(counts))
    reverse = {ast.dump(after): before for _, before, after in specs}

    class Restore(ast.NodeTransformer):
        def visit(self, node):
            old = reverse.get(ast.dump(node))
            return copy.deepcopy(old) if old is not None else super().visit(node)

    if ast.dump(Restore().visit(copy.deepcopy(changed))) != ast.dump(original):
        raise ValueError('Existing raw Gaussian, window, clock or evidence checks changed')
    return changed


def implementation(kind, bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (6,) or not np.array_equal(bounds[[0, 1, 3, 4]], np.ones(4)):
        raise ValueError('Keep the four original direct-joint action bounds')
    if kind == 'exploration':
        module = exploration
        declarations = [
            ('np.clip(raw, -1., 1.)', 'np.clip(raw, -HEAD_LIMITS, HEAD_LIMITS)'),
            ('np.clip(mean, -1., 1.)', 'np.clip(mean, -HEAD_LIMITS, HEAD_LIMITS)'),
            ('abs(sampled) > 1.', 'abs(sampled) > HEAD_LIMITS'),
            ('abs(mu) > 1.', 'abs(mu) > HEAD_LIMITS'),
        ]
    elif kind == 'kl':
        module = kl
        declarations = [
            ('clipped_kl(m0, s0, m1, s1)',
                'clipped_kl(m0 / HEAD_LIMITS, s0 / HEAD_LIMITS, m1 / HEAD_LIMITS, s1 / HEAD_LIMITS)'),
            ('[x.ravel() for x in (m0,s0,m1,s1)]',
                '[(x / HEAD_LIMITS).ravel() for x in (m0,s0,m1,s1)]'),
            ('np.clip(m1, -1., 1.)', 'np.clip(m1, -HEAD_LIMITS, HEAD_LIMITS)'),
            ('np.clip(m0, -1., 1.)', 'np.clip(m0, -HEAD_LIMITS, HEAD_LIMITS)'),
        ]
    else:
        raise ValueError('Only exploration and clipped-distribution KL use this bound adapter')
    tree = transform(module.audit, declarations)
    scope = dict(module.__dict__, HEAD_LIMITS=bounds, __file__=str(SCRIPT))
    exec(compile(ast.fix_missing_locations(tree), str(module.__file__), 'exec'), scope)
    return scope['audit'], Path(module.__file__)


def audit(kind, output, through, selected=None):
    output = Path(output).resolve()
    plan_path = output / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    limit = checkpoint_wheel_action_limit(plan['training_config'])
    bounds = [1., 1., limit, 1., 1., limit]
    function, original = implementation(kind, bounds)
    result = function(output, through, selected) if kind == 'kl' else function(output, through)
    path, report = result
    report.update(saved_action_limits_by_head=bounds, wheel_action_limit=limit,
        original_formula_and_all_other_AST_restored_identical=True,
        scalar_unit_clamp_not_assumed_for_wheel_heads=True)
    pins = report['source_pins']
    pins.update({str(p): sha(p) for p in (SCRIPT, original, plan_path)})
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Saved action envelope or read-only audit evidence changed')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return path, report


def qualify_math():
    limit = np.array([1., 1., 1.25, 1., 1., 1.25])
    m0 = np.array([-1.4, -.2, -1.4, .7, 1.3, 1.18])
    s0 = np.array([.05, .1, .05, .04, .07, .04])
    m1, s1 = m0 + .03, s0 * 1.05
    value = kl.clipped_kl(m0 / limit, s0 / limit, m1 / limit, s1 / limit)
    numerical = np.array([kl.numeric_kl(a / c, b / c, d / c, e / c)
        for a, b, d, e, c in zip(m0, s0, m1, s1, limit)])
    error = float(abs(value - numerical).max())
    raw = np.log(s1 / s0) + (s0**2 + (m0 - m1)**2) / (2. * s1**2) - .5
    if error > 1.e-10 or np.any(value > raw + 1.e-10):
        raise ValueError('Per-head KL normalization or numerical integration differs')
    implementation('exploration', limit)
    implementation('kl', limit)
    return dict(qualified=True, per_head_numeric_quadrature_max_error=error,
        saved_action_limits_by_head=limit.tolist(), both_original_audits_AST_restored_identical=True,
        optimizer_executed=False, simulator_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('exploration', 'kl', 'qualify'))
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--through', type=int)
    parser.add_argument('--updates', type=int, nargs='+')
    args = parser.parse_args()
    if args.kind == 'qualify':
        print(json.dumps(qualify_math()))
    else:
        path, _ = audit(args.kind, args.output, args.through, args.updates)
        print(json.dumps(dict(report=str(path), read_only=True)))
