"""Load quality-reward criteria independently of optional violation penalties.

The historical environment is pinned by prior checkpoints. Apply this small,
versioned fix in the new runtime, leaving its old source and evidence intact.
Only the criteria-loading predicate and its matching validation predicate
change; reward formulas, physics, action maps and termination stay original.
"""
import ast
import copy
import inspect
from pathlib import Path
import textwrap
import types

from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
ENVIRONMENT = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
FLOOR = 'self.cfg.rewards.tracking_task_quality_gate_floor < 1.'


def transformed(method, name):
    tree = ast.parse(textwrap.dedent(method))
    original = ast.dump(tree)
    references = []
    if name == '__init__':
        expression = ast.parse('self.cfg.rewards.tracking_goal_violation != 0. or self.cfg.rewards.tracking_goal_component_violation != 0.', mode='eval').body
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(expression):
                references.append((node, 'test', node.test))
    elif name == '_validate_configuration':
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'validate_task_quality_gate'):
                if len(node.args) != 2 or node.keywords:
                    raise ValueError('Requalify the original quality validation signature')
                expected = ast.parse('self.cfg.rewards.tracking_goal_component_violation or self.cfg.rewards.tracking_goal_violation', mode='eval').body
                if ast.dump(node.args[1]) != ast.dump(expected):
                    raise ValueError('The original validation dependency changed')
                references.append((node, 'args', list(node.args)))
    else:
        raise ValueError('Only the two configuration dependency predicates may change')
    if len(references) != 1:
        raise ValueError('Requalify one exact original predicate: ' + name)
    node, field, prior = references[0]
    if field == 'test':
        node.test = ast.BoolOp(op=ast.Or(), values=[copy.deepcopy(prior), ast.parse(FLOOR, mode='eval').body])
    else:
        node.args[1] = ast.BoolOp(op=ast.Or(), values=[copy.deepcopy(prior[1]), ast.parse(FLOOR, mode='eval').body])
    changed = copy.deepcopy(tree)
    setattr(node, field, prior)
    if ast.dump(tree) != original:
        raise ValueError('Environment changed beyond its declared criteria dependency')
    return ast.fix_missing_locations(changed), dict(method=name,
        only_change='Criteria required when either optional penalty or the quality reward is enabled',
        restored_AST_identical=True)


def install(cls):
    existing = cls.__dict__.get('_quality_criteria_runtime_proof')
    if existing is not None:
        if existing['implementation_sha256'] != sha(SCRIPT):
            raise ValueError('Another quality-criteria implementation is already installed')
        return existing
    changes, pending = [], {}
    for name in ('__init__', '_validate_configuration'):
        function = cls.__dict__[name]
        tree, evidence = transformed(inspect.getsource(function), name)
        # Compile as a method, preserving the original __class__ closure used
        # by zero-argument super() when reattaching the original class cell.
        stub = ast.ClassDef(name='_CompiledQualityCriteriaMethod', bases=[], keywords=[],
                            body=tree.body, decorator_list=[])
        wrapper = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
        scope = dict(function.__globals__)
        exec(compile(wrapper, str(ENVIRONMENT), 'exec'), scope)
        compiled = scope['_CompiledQualityCriteriaMethod'].__dict__[name]
        if compiled.__code__.co_freevars != function.__code__.co_freevars:
            raise ValueError('Preserve original method closure and super() semantics')
        revised = types.FunctionType(compiled.__code__, function.__globals__, function.__name__,
                                     function.__defaults__, function.__closure__)
        revised.__kwdefaults__ = function.__kwdefaults__
        revised.__annotations__ = function.__annotations__
        pending[name] = revised
        changes.append(evidence)
    for name, function in pending.items():
        setattr(cls, name, function)
    proof = dict(implementation=str(SCRIPT), implementation_sha256=sha(SCRIPT),
        original_environment=str(ENVIRONMENT), original_environment_sha256=sha(ENVIRONMENT),
        methods=changes, original_class_and_super_closure_preserved=True,
        action_map_or_physics_or_termination_or_reward_formula_changed=False,
        optional_violation_penalties_enabled=False)
    cls._quality_criteria_runtime_proof = proof
    return proof


def qualify_predicates():
    """Exercise real source predicates without importing Isaac/starting Kit."""
    from types import SimpleNamespace
    from wheel_legged_gym_isaaclab.reward_signals import validate_task_quality_gate
    source = ast.parse(ENVIRONMENT.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'WheelLeggedVMCFlatEnv')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init, a = transformed(ast.unparse(methods['__init__']), '__init__')
    validation, b = transformed(ast.unparse(methods['_validate_configuration']), '_validate_configuration')
    new_predicate = next(n.test for n in ast.walk(init) if isinstance(n, ast.If)
        and isinstance(n.test, ast.BoolOp) and any(ast.dump(v) == ast.dump(ast.parse(FLOOR, mode='eval').body)
                                                 for v in n.test.values))
    validation_predicate = next(n.args[1] for n in ast.walk(validation) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'validate_task_quality_gate')
    rows = []
    for floor, penalty in ((1., 0.), (.5, 0.), (.05, 0.), (1., -.1), (.5, -.1)):
        cfg = SimpleNamespace(rewards=SimpleNamespace(tracking_goal_violation=penalty,
            tracking_goal_component_violation=0., tracking_task_quality_gate_floor=floor))
        scope = dict(self=SimpleNamespace(cfg=cfg))
        needed = bool(eval(compile(ast.fix_missing_locations(ast.Expression(new_predicate)), str(SCRIPT), 'eval'), scope))
        available = bool(eval(compile(ast.fix_missing_locations(ast.Expression(validation_predicate)), str(SCRIPT), 'eval'), scope))
        if needed != (floor < 1. or penalty != 0.) or available != needed:
            raise ValueError('Real loading and validation predicates disagree')
        validate_task_quality_gate(floor, available)
        rows.append(dict(floor=floor, violation_penalty=penalty, criteria_loaded=needed, valid=True))
    for invalid in (0., 1.1, float('nan')):
        try:
            validate_task_quality_gate(invalid, True)
        except ValueError:
            pass
        else:
            raise ValueError('Invalid quality floors were admitted')
    return dict(qualified=True, original_environment_sha256=sha(ENVIRONMENT),
        implementation_sha256=sha(SCRIPT), methods=[a, b], configurations=rows,
        enabled_reward_works_with_both_penalties_zero=True,
        disabled_reward_preserves_original_no_criteria_path=True,
        invalid_floors_still_rejected=True, simulator_started=False)
