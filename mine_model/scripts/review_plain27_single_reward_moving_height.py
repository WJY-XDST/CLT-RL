"""Original moving-height plan and metrics after a single reward-field block."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import sha
from qualify_source801_drift50_transfer import differences
import review_plain27_moving_height_after_block200 as template
import review_plain27_single_reward_block200 as owner

SCRIPT = Path(__file__).resolve()


def build(review_launch, qualification, output):
    return owner.rebound(template.build, SCRIPT=SCRIPT, owner=owner)(review_launch, qualification, output)


def qualify(review_launch, qualification, output, checkpoint):
    tree = ast.parse(inspect.getsource(template.qualify))
    before = ast.dump(tree)
    original = ast.dump(ast.parse("{'/log_dir'}", mode='eval').body)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Set) and ast.dump(n) == original]
    if len(matches) != 1:
        raise ValueError('Recheck the original offline metadata configuration guard')
    old = matches[0].elts
    matches[0].elts = ast.parse("{'/log_dir', '/rewards/' + bound.native.REWARD_KEY}", mode='eval').body.elts
    changed = copy.deepcopy(tree)
    matches[0].elts = old
    if ast.dump(tree) != before:
        raise ValueError('Original phase plan, metric, flags or command checks changed')
    scope = dict(template.__dict__, SCRIPT=SCRIPT, owner=owner, build=build)
    exec(compile(ast.fix_missing_locations(changed), template.__file__, 'exec'), scope)
    # Preserve the original CPU reader operations, then check exact old/new
    # values rather than accepting an arbitrary field value at this boundary.
    context = build(review_launch, qualification, output)
    native = context.bound.native
    _, prior = native.require_source(context.bound.launch['source_checkpoint'], transition=True)
    _, current = native.require_source(checkpoint)
    if (prior['rewards'][native.REWARD_KEY] != native.OLD_VALUE
            or current['rewards'][native.REWARD_KEY] != native.REWARD_VALUE
            or set(differences(prior, current)) - {'/log_dir'} != {'/rewards/' + native.REWARD_KEY}):
        raise ValueError('Require exactly the declared offline reward-field transition')
    result = scope['qualify'](review_launch, qualification, output, checkpoint)
    result.update(only_declared_reward_configuration_change=native.DELTA['rewards'],
        original_offline_metadata_guard_AST_restored_identically=True)
    result['source_pins'][str(template.SCRIPT)] = sha(template.SCRIPT)
    write_json(Path(qualification), result)
    return result


def run(review_launch, qualification, output):
    return owner.rebound(template.run, SCRIPT=SCRIPT, owner=owner, build=build)(review_launch, qualification, output)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--review-launch', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    args = cli.parse_args()
    result = (qualify(args.review_launch, args.qualification, args.output, args.qualify_checkpoint)
        if args.qualify_checkpoint else run(args.review_launch, args.qualification, args.output))
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates', 'supplemental_steps_ready', 'policy_accepted') if k in result}))
