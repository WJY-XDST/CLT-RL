"""Frozen-policy Gaussian head ablation; never imported by PPO training.

Keep the existing evaluator, its RNG draws, physics and mean network. A single
constant mask suppresses chosen exploration components for every command. The
requested actions and original draws are recorded for independent verification.
"""
import argparse
import ast
import copy
import hashlib
from pathlib import Path


def masked_evaluator_tree(original_factory, heads):
    selected = tuple(heads)
    if not selected or len(set(selected)) != len(selected) or any(i not in range(6) for i in selected):
        raise ValueError('Select distinct action heads in [0, 5]')
    mask = [float(i in selected) for i in range(6)]

    def build():
        tree, path, proof = original_factory()
        original = ast.dump(tree)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        blocks = [n for n in ast.walk(main) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'args.sample_policy_actions']
        setup = [n for n in blocks if any(isinstance(s, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'policy_noise_generator' for t in s.targets)
                 for s in n.body)]
        sample = [n for n in blocks if any(isinstance(s, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'noise' for t in s.targets)
                  for s in n.body)]
        assert len(setup) == len(sample) == 1
        init = ast.parse(
            'policy_std = policy_std * policy_std.new_tensor(' + repr(mask) + ')\n'
            '_diagnostic_draws = []\n_diagnostic_actions = []').body
        setup[0].body.extend(init)
        capture = ast.parse(
            '_diagnostic_draws.append(noise.detach().cpu().numpy().copy())\n'
            '_diagnostic_actions.append(actions.detach().cpu().numpy().copy())').body
        sample[0].body.extend(capture)
        manifest = [n.value for n in main.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'manifest' for t in n.targets)]
        assert len(manifest) == 1 and isinstance(manifest[0], ast.Dict)
        manifest[0].keys.extend([ast.Constant('diagnostic_noise_heads'),
                                  ast.Constant('diagnostic_effective_noise_std')])
        manifest[0].values.extend([ast.parse(repr(list(selected)), mode='eval').body,
            ast.parse('policy_std.detach().cpu().tolist()', mode='eval').body])
        ends = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'final' for t in n.targets)]
        assert len(ends) == 1
        save = ast.parse(
            "np.savez_compressed(args.output / 'diagnostic_actions.npz', "
            'requested_actions=np.stack(_diagnostic_actions), '
            'standardized_draws=np.stack(_diagnostic_draws))').body
        main.body[ends[0]:ends[0]] = save
        changed = copy.deepcopy(tree)
        del main.body[ends[0]:ends[0] + len(save)]
        del manifest[0].keys[-2:]
        del manifest[0].values[-2:]
        del sample[0].body[-len(capture):]
        del setup[0].body[-len(init):]
        assert ast.dump(tree) == original
        proof.update(
            frozen_Gaussian_head_mask=mask, diagnostic_noise_heads=list(selected),
            original_AST_restored_identical_after_removing_noise_mask_and_recording=True,
            original_AST_before_head_mask_sha256=hashlib.sha256(original.encode()).hexdigest(),
            original_mean_network_physics_rewards_and_RNG_draws_unchanged=True,
            added_RNG_draws=0, additional_loss_or_backward=False,
            case_specific_training_or_control=False, not_an_acceptance_test=True)
        return ast.fix_missing_locations(changed), path, proof

    return build


def main():
    import sys
    import joint_pd_calibrated_precision_noise_runtime as runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--noise-heads', type=int, nargs='+', required=True)
    at = sys.argv.index('--')
    args = parser.parse_args(sys.argv[1:at])
    remaining = sys.argv[at + 1:]
    assert '--sample-policy-actions' in remaining and '--command-probe-json' in remaining
    runtime.configure(args.recipe)
    runtime.evaluator_tree = masked_evaluator_tree(runtime.evaluator_tree, args.noise_heads)
    runtime.replay(remaining)


if __name__ == '__main__':
    main()
