"""Frozen Gaussian diagnostic: change only local wheel sampling std.

Never change actor.std, optimizer, commands, physics or acceptance. Reuse the
qualified original evaluator and record its actual Gaussian draws for pairing.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

import joint_pd_source2201_yaw025_runtime as runtime
from joint_pd_continuation import sha

SCRIPT = Path(__file__).resolve()
SOURCE_SHA = 'f0a15a94bd7772f34ed0f0acc202c9af5f3355642be946102ce7a5174d49faba'


def evaluator_tree():
    tree, path = runtime.evaluator_tree()
    original = ast.dump(tree)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    edits = []

    def insert(body, index, code):
        node = ast.parse(code).body[0]
        body.insert(index, node)
        edits.append((body, node))

    sample_blocks = [n for n in main.body if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'args.sample_policy_actions']
    if len(sample_blocks) != 1:
        raise ValueError('Requalify saved std loading')
    insert(sample_blocks[0].body, len(sample_blocks[0].body),
        'policy_std = _sampling_probe_std(policy_std, checkpoint, actor)')
    manifests = [i for i, n in enumerate(main.body) if isinstance(n, ast.Expr)
        and 'manifest.json' in ast.unparse(n) and '.write_text(' in ast.unparse(n)]
    if len(manifests) != 1:
        raise ValueError('Requalify original manifest location')
    insert(main.body, manifests[0], 'manifest.update(sampling_std_probe=_sampling_probe_metadata())')
    blocks = [n for n in ast.walk(main) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'args.sample_policy_actions'
        and any(isinstance(c, ast.Assign) and ast.unparse(c.targets[0]) == 'noise' for c in n.body)]
    if len(blocks) != 1 or ast.unparse(blocks[0].body[0].targets[0]) != 'noise':
        raise ValueError('Requalify dedicated original Gaussian draw')
    insert(blocks[0].body, 1, '_record_probe_noise(noise)')
    finals = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == 'final']
    if len(finals) != 1:
        raise ValueError('Requalify completion artifact location')
    insert(main.body, finals[0], '_save_probe_noise(args.output, policy_std, actor, checkpoint)')
    changed = copy.deepcopy(tree)
    for body, node in reversed(edits):
        body.remove(node)
    if ast.dump(tree) != original:
        raise ValueError('Evaluator changed beyond four declared diagnostic hooks')
    proof = dict(original_evaluator_sha256=sha(path), probe_entry_sha256=sha(SCRIPT),
        hook_count=4, original_AST_restored_identically=True,
        noise_draw_and_action_addition_AST_unchanged=True,
        source_actor_std_and_optimizer_unchanged=True,
        relative_error_target=.1, acceptance_changed=False, policy_accepted=False)
    return ast.fix_missing_locations(changed), path, proof


def replay(arguments, wheel_std):
    import numpy as np
    import torch
    source = Path(arguments[arguments.index('--checkpoint') + 1]).resolve()
    runtime.require_source(source)
    if sha(source) != SOURCE_SHA or wheel_std not in (.02, .03):
        raise ValueError('Only the completed2601 checkpoint and paired wheel std .03/.02')
    if (any(flag in arguments for flag in ('--export_dir', '--overrides_json', '--command-scale',
            '--wheel_probe', '--heading-hold', '--curriculum-progress', '--bias-probe-json'))
            or '--sample-policy-actions' not in arguments
            or '--command-probe-json' not in arguments or '--record_policy_inputs' not in arguments):
        raise ValueError('Frozen recorded command diagnosis only')
    tree, path, proof = evaluator_tree()
    metadata, noises = {}, []

    def sampling_std(saved_std, checkpoint, actor):
        expected = saved_std.new_tensor([.05, .05, .03, .05, .05, .03])
        if not torch.equal(saved_std, expected) or not torch.equal(actor.std, saved_std):
            raise ValueError('Unexpected saved exploration state')
        applied = saved_std.detach().clone()
        applied[[2, 5]] = wheel_std
        metadata.update(proof, checkpoint_sha256=SOURCE_SHA,
            saved_std=saved_std.cpu().tolist(), sampling_std=applied.cpu().tolist(),
            only_changed_action_indices=[2, 5] if wheel_std == .02 else [],
            diagnostic_is_not_training_or_acceptance=True)
        return applied

    def save(directory, applied, actor, checkpoint):
        state = actor.state_dict()
        if state.keys() != checkpoint['model_state_dict'].keys() or not all(
                torch.equal(value, checkpoint['model_state_dict'][key]) for key, value in state.items()):
            raise ValueError('Frozen model state changed during sampling diagnosis')
        if sha(source) != SOURCE_SHA or len(noises) != 1000:
            raise ValueError('Source changed or incomplete20s draws')
        np.savez_compressed(Path(directory) / 'sampling_noise.npz', epsilon=np.stack(noises),
            sampling_std=applied.cpu().numpy(), saved_std=actor.std.detach().cpu().numpy())
        (Path(directory) / 'sampling_probe_integrity.json').write_text(json.dumps(dict(
            **metadata, all_model_tensors_bitwise_unchanged=True,
            gaussian_draw_count=len(noises), additional_RNG_draws=0), indent=2) + '\n')

    sys.argv = [str(SCRIPT), *runtime.base.tires.arguments(arguments)]
    exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
        _install_quality_criteria=runtime.criteria_fix.install,
        _observe_source200=runtime.private_runtime().observe_environment,
        _sampling_probe_std=sampling_std, _sampling_probe_metadata=lambda: dict(metadata),
        _record_probe_noise=lambda noise: noises.append(noise.detach().cpu().numpy().copy()),
        _save_probe_noise=save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--wheel-sampling-std', type=float, choices=(.02, .03))
    parser.add_argument('--qualify', action='store_true')
    args, rest = parser.parse_known_args()
    if args.qualify:
        _, _, result = evaluator_tree()
        print(json.dumps(result))
    else:
        if not rest or rest.pop(0) != 'replay' or args.wheel_sampling_std is None:
            parser.error('Use replay and --wheel-sampling-std')
        replay(rest, args.wheel_sampling_std)
