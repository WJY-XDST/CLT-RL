"""Original full27 replay plus a read-only current-physics/recipe observation."""
import ast
import copy
from pathlib import Path
import sys

from joint_pd_continuation import ROOT

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_source200_stationary_runtime as runtime


def evaluator_tree():
    path = ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'
    tree = ast.parse(path.read_text())
    original = copy.deepcopy(tree)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    targets = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == 'EvaluationEnv']
    if len(targets) != 1:
        raise ValueError('Requalify the original evaluator constructor')
    main.body.insert(targets[0] + 1, ast.parse('_observe_source200(env, args.output)').body[0])
    restored = copy.deepcopy(tree)
    restored_main = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    restored_main.body = [n for n in restored_main.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == '_observe_source200')]
    if ast.dump(restored) != ast.dump(original):
        raise ValueError('Replay changed outside its one read-only constructor observer')
    return ast.fix_missing_locations(tree), path


if __name__ == '__main__':
    values = list(sys.argv[1:])
    flag = '--initial_source200_transfer'
    initial = flag in values
    if initial:
        values.remove(flag)
    source = Path(values[values.index('--checkpoint') + 1]).resolve()
    runtime.require_source(source, initial_transfer=initial)
    sys.argv = [sys.argv[0], *runtime.tires.arguments(values)]
    tree, original = evaluator_tree()
    exec(compile(tree, str(original), 'exec'), dict(
        __name__='__main__', __file__=str(original), _observe_source200=runtime.observe_environment))
