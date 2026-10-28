"""Original full evaluation with explicit paired analytic-tire/twist reward metadata."""
import ast
import copy
from functools import partial
from pathlib import Path
import sys

import rsl_rl.modules
import yaml

from joint_pd_continuation import ROOT

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_analytic_tire_runtime as tires
import joint_pd_stationary_twist_runtime as twist
from joint_pd_height_features_candidate import JointPdHeightFeatureActorCritic
from joint_pd_height_feature_runtime import require_paired_feature_model
from wheel_legged_gym_isaaclab.robot_model import MODEL


def evaluator_tree():
    original = ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'
    tree = ast.parse(original.read_text())
    before = copy.deepcopy(tree)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    imports = [i for i, n in enumerate(main.body) if isinstance(n, ast.ImportFrom)
        and n.module.endswith('.wheel_legged_vmc_flat_env')]
    constructors = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == 'EvaluationEnv']
    if len(imports) != 1 or len(constructors) != 1:
        raise ValueError('Requalify the original evaluator constructor/import sites')
    main.body.insert(constructors[0] + 1, ast.parse('_observe_twist_environment(env, args.output)').body[0])
    main.body.insert(imports[0] + 1, ast.parse('_install_twist_reward_method()').body[0])
    restored = copy.deepcopy(tree)
    restored_main = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    restored_main.body = [n for n in restored_main.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id in ('_observe_twist_environment', '_install_twist_reward_method'))]
    if ast.dump(restored) != ast.dump(before):
        raise ValueError('Evaluator changed outside the two explicit reward hooks')
    ast.fix_missing_locations(tree)
    return tree, original


if __name__ == '__main__':
    values = list(sys.argv[1:])
    flag = '--diagnostic_stationary_reward_transfer'
    transfer = flag in values
    if transfer:
        values.remove(flag)
    source = Path(values[values.index('--checkpoint') + 1]).resolve()
    saved, cfg = twist.require_source(source, initial_transfer=transfer)
    output = Path(values[values.index('--output') + 1]).resolve()
    model = require_paired_feature_model(saved, cfg, MODEL['hip_z'])
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    if agent['policy']['class_name'] != 'JointPdHeightFeatureActorCritic':
        raise ValueError('Keep the actual paired feature policy')
    rsl_rl.modules.ActorCritic = partial(JointPdHeightFeatureActorCritic, height_feature_model=model)
    sys.argv = [sys.argv[0], *tires.arguments(values)]
    tree, original = evaluator_tree()
    namespace = dict(__name__='__main__', __file__=str(original),
        _install_twist_reward_method=twist.install_reward_method,
        _observe_twist_environment=twist.observe_environment)
    exec(compile(tree, str(original), 'exec'), namespace)
