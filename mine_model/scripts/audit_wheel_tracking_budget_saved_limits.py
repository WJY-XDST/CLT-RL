"""Use saved wheel bounds in the original read-only rolling-error budget.

Only the raw-mean-to-reference reconstruction and wheel clipping counter use
the saved bound. Original physical errors, signed components, steady windows,
leg bounds, source integrity checks and acceptance flags remain unchanged.
"""
import argparse
import ast
import copy
import json
from pathlib import Path

from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import audit_wheel_tracking_budget as original

SCRIPT = Path(__file__).resolve()


def audit(replay, output):
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve earlier immutable diagnostics')
    cfg = json.loads((replay / 'resolved_env.json').read_text())
    limit = checkpoint_wheel_action_limit(cfg)
    tree = ast.parse(Path(original.__file__).read_text())
    baseline = copy.deepcopy(tree)
    clip = ast.dump(ast.parse('np.clip(raw[:, [2, 5]], -1., 1.)', mode='eval').body)
    counter = ast.dump(ast.parse('np.abs(raw[:, [2, 5]]) > 1.', mode='eval').body)
    counts = dict(reference_bound=0, wheel_counter=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.dump(node) == clip:
            node.args[1:] = [ast.UnaryOp(op=ast.USub(), operand=ast.Name(id='SAVED_WHEEL_BOUND', ctx=ast.Load())),
                            ast.Name(id='SAVED_WHEEL_BOUND', ctx=ast.Load())]
            counts['reference_bound'] += 1
        if isinstance(node, ast.Compare) and ast.dump(node) == counter:
            node.comparators = [ast.Name(id='SAVED_WHEEL_BOUND', ctx=ast.Load())]
            counts['wheel_counter'] += 1
    if counts != dict(reference_bound=1, wheel_counter=1):
        raise ValueError('Requalify exactly the saved wheel reference map and counter')
    restored = copy.deepcopy(tree)
    for node in ast.walk(restored):
        if (isinstance(node, ast.Call) and len(node.args) == 3
                and isinstance(node.args[2], ast.Name) and node.args[2].id == 'SAVED_WHEEL_BOUND'):
            node.args[1:] = [ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1.)), ast.Constant(value=1.)]
        if (isinstance(node, ast.Compare) and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == 'SAVED_WHEEL_BOUND'):
            node.comparators = [ast.Constant(value=1.)]
    if ast.dump(restored) != ast.dump(baseline):
        raise ValueError('Original physical formulas, masks or acceptance checks changed')
    namespace = dict(original.__dict__, __name__='_saved_wheel_tracking_budget', SAVED_WHEEL_BOUND=limit)
    exec(compile(ast.fix_missing_locations(tree), str(original.__file__), 'exec'), namespace)
    result = namespace['audit'](replay, output)
    result.update(saved_wheel_action_limit=limit, saved_reference_cap_rad_s=limit * cfg['action_scale_vel'],
        original_numeric_reader_AST_restored_identical=True,
        only_reference_reconstruction_and_wheel_counter_bounds_changed=True,
        original_steady_windows_metrics_and_flags_unchanged=True,
        optimizer_executed=False, simulator_started=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False)
    result['source_pins'].update({str(p): sha(p) for p in (SCRIPT, Path(original.__file__))})
    # The original reader exclusively creates the destination, after all native
    # data checks. This write only adds the declared bound and adapter proof.
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.replay, args.output)
    print(json.dumps({k: result[k] for k in ('complete_trials', 'saved_wheel_action_limit',
        'target_from_raw_mean_max_error_rad_s', 'signed_decomposition_closure_max_error',
        'original_numeric_reader_AST_restored_identical', 'policy_accepted')}))
