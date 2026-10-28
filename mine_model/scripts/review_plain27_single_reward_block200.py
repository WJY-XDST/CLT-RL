"""Retain all original full51/static readers for one declared reward-field trial."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import types

from auto_train_mine import write_json
from joint_pd_continuation import sha
from qualify_source801_drift50_transfer import differences
import review_plain27_exploration_block200 as template

SCRIPT = Path(__file__).resolve()


def rebound(function, **values):
    scope = dict(function.__globals__, **values)
    result = types.FunctionType(function.__code__, scope, function.__name__,
        function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def build(launch_path, static_qualification):
    tree = ast.parse(inspect.getsource(template.build))
    before = ast.dump(tree)
    old = ast.parse("('commands', 'rewards', 'controller', 'physics')", mode='eval').body
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Tuple) and ast.dump(n) == ast.dump(old)]
    if len(matches) != 1:
        raise ValueError('Recheck the original launch delta guard')
    node = matches[0].elts[1]
    node.value = 'exploration'
    changed = copy.deepcopy(tree)
    node.value = 'rewards'
    if ast.dump(tree) != before:
        raise ValueError('Original reviewer build changed outside the declared delta guard')
    scope = dict(template.__dict__, SCRIPT=SCRIPT)
    exec(compile(ast.fix_missing_locations(changed), template.__file__, 'exec'), scope)
    bound = scope['build'](launch_path, static_qualification)
    native = bound.native
    if (bound.launch['next_configuration_changes'] != native.DELTA
            or native.DELTA['rewards'] != {native.REWARD_KEY: [native.OLD_VALUE, native.REWARD_VALUE]}
            or any(native.DELTA[k] for k in ('commands', 'exploration', 'controller', 'physics'))):
        raise ValueError('Require exactly the one reviewed existing reward field')

    def normalize_native_reward(prior, current, runtime):
        # Called after the original log-directory normalization. Every other
        # native field must remain equal, including all physical/control fields.
        if (prior['rewards'][runtime.REWARD_KEY] != runtime.OLD_VALUE
                or current['rewards'][runtime.REWARD_KEY] != runtime.REWARD_VALUE
                or differences(prior, current) != ['/rewards/' + runtime.REWARD_KEY]):
            raise ValueError('Native configurations differ beyond the single declared reward field')
        prior['rewards'][runtime.REWARD_KEY] = runtime.REWARD_VALUE

    tree = ast.parse(inspect.getsource(template.shared.review51))
    before = ast.dump(tree)
    test = ast.dump(ast.parse('native_before != native_after', mode='eval').body)
    inserted, edits = False, []
    for n in list(ast.walk(tree)):
        if isinstance(n, ast.Constant) and n.value == 'paired_source4201_candidate4401_full_height.json':
            edits.append((n, 'value', n.value))
            n.value = f'paired_source{bound.source_clock}_candidate{bound.target_clock}_full_height.json'
        for field, value in list(ast.iter_fields(n)):
            if isinstance(value, list):
                for index, statement in enumerate(value):
                    if isinstance(statement, ast.If) and any(ast.dump(x) == test for x in ast.walk(statement.test)):
                        if inserted:
                            raise ValueError('Recheck the original native configuration comparator')
                        updated = list(value)
                        updated.insert(index, ast.parse('normalize_native_reward(native_before, native_after, runtime)').body[0])
                        edits.append((n, field, value))
                        setattr(n, field, updated)
                        inserted = True
    if not inserted or len(edits) != 2:
        raise ValueError('Requalify the pair label and native configuration normalization insertion')
    changed = copy.deepcopy(tree)
    for n, field, value in reversed(edits):
        setattr(n, field, value)
    if ast.dump(tree) != before:
        raise ValueError('Original native replay, numeric reader or threshold operations changed')
    bound.worker.normalize_native_reward = normalize_native_reward
    exec(compile(ast.fix_missing_locations(changed), template.shared.__file__, 'exec'), bound.worker.__dict__)
    bound.replay_metadata_evidence.append(dict(
        declared_single_reward_native_configuration_normalization=native.DELTA['rewards'],
        original_metric_formulas_and_native_commands_preserved=True, restored_AST_identical=True))
    return bound


def qualify(launch_path, static_qualification, checkpoint, qualification):
    result = rebound(template.qualify, build=build, SCRIPT=SCRIPT)(
        launch_path, static_qualification, checkpoint, qualification)
    bound = build(launch_path, static_qualification)
    result.update(only_native_configuration_difference=bound.native.DELTA['rewards'],
        native_environment_configuration_unchanged=False,
        native_configuration_change_is_only_declared_reward=True,
        exploration_only_difference={}, single_existing_reward_field_trial=True,
        reviewer_adapter_evidence=bound.replay_metadata_evidence)
    result['source_pins'][str(template.SCRIPT)] = sha(template.SCRIPT)
    write_json(Path(qualification), result)
    return result


def run(launch_path, static_qualification, qualification, outputs, static_output):
    tree = ast.parse(inspect.getsource(template.run))
    before = ast.dump(tree)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
        and n.value == 'native_environment_configuration_unchanged']
    if len(matches) != 1:
        raise ValueError('Recheck the original configuration declaration guard')
    matches[0].value = 'native_configuration_change_is_only_declared_reward'
    changed = copy.deepcopy(tree)
    matches[0].value = 'native_environment_configuration_unchanged'
    if ast.dump(tree) != before:
        raise ValueError('Original verification or numerical run operations changed')
    scope = dict(template.__dict__, SCRIPT=SCRIPT, build=build)
    exec(compile(ast.fix_missing_locations(changed), template.__file__, 'exec'), scope)
    return scope['run'](launch_path, static_qualification, qualification, outputs, static_output)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-launch', type=Path, required=True)
    cli.add_argument('--static-qualification', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    cli.add_argument('--outputs', type=Path, nargs=4)
    cli.add_argument('--static-output', type=Path)
    args = cli.parse_args()
    result = (qualify(args.training_launch, args.static_qualification, args.qualify_checkpoint, args.qualification)
        if args.qualify_checkpoint else run(args.training_launch, args.static_qualification,
            args.qualification, args.outputs, args.static_output))
    print(json.dumps({k: result[k] for k in ('qualified', 'static_retention_verified', 'policy_accepted') if k in result}))
