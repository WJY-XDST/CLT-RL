"""Source-bound frozen sampling probe using the original four evaluator hooks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import joint_pd_source2601_sampling_std_probe as template
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()


def bound(specification):
    specification = Path(specification).resolve()
    spec = json.loads(specification.read_text())
    path = Path(spec['runtime'])
    if (not spec['qualified'] or path.parent != ROOT / 'mine_model/experiments'
            or sha(path) != spec['runtime_sha256']
            or any(sha(path) != digest for path, digest in spec['source_pins'].items())):
        raise ValueError('Keep the source-bound frozen probe specification and implementation')
    loader = importlib.util.spec_from_file_location('_frozen_sampling_bound_native_runtime', path)
    runtime = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runtime)
    return spec, runtime


def evaluator_tree(runtime):
    function = template.evaluator_tree
    retained = types.FunctionType(function.__code__, dict(template.__dict__, runtime=runtime, SCRIPT=SCRIPT),
        function.__name__)
    tree, path, proof = retained()
    proof['hook_template_sha256'] = sha(template.SCRIPT)
    return tree, path, proof


def replay(arguments, specification, label):
    import numpy as np
    import torch
    spec, runtime = bound(specification)
    source = Path(arguments[arguments.index('--checkpoint') + 1]).resolve()
    state, cfg = runtime.require_source(source)
    if (str(source) != spec['checkpoint'] or sha(source) != spec['checkpoint_sha256']
            or saved_updates(source) != spec['actual_updates'] or not tensor_finite(state)
            or len(state['optimizer_state_dict']['state']) != 17
            or any(float(value['step']) != spec['all17_Adam_step']
                   for value in state['optimizer_state_dict']['state'].values())
            or cfg['leg_control_mode'] != 'implicit_joint_position'
            or label not in spec['sampling_vectors']
            or '--sample-policy-actions' not in arguments
            or '--command-probe-json' not in arguments or '--record_policy_inputs' not in arguments
            or any(flag in arguments for flag in ('--export_dir', '--overrides_json', '--command-scale',
                '--wheel_probe', '--heading-hold', '--curriculum-progress', '--bias-probe-json'))):
        raise ValueError('Use only the exact frozen source and declared local sampling vector')
    tree, path, proof = evaluator_tree(runtime)
    metadata, noises = {}, []

    def sampling_std(saved, checkpoint, actor):
        expected = saved.new_tensor(spec['saved_std'])
        applied = saved.new_tensor(spec['sampling_vectors'][label])
        changed = torch.nonzero(applied != saved, as_tuple=False).flatten().cpu().tolist()
        if (not torch.equal(saved, expected) or not torch.equal(actor.std, saved)
                or not torch.isfinite(applied).all() or not (applied > 0.).all()
                or changed != ([] if label == 'saved' else spec['changed_action_indices'])):
            raise ValueError('Saved state or local diagnostic sampling entries differ')
        metadata.update(proof, specification=str(Path(specification).resolve()),
            specification_sha256=sha(specification), sampling_label=label,
            checkpoint_sha256=sha(source), saved_std=saved.cpu().tolist(),
            sampling_std=applied.cpu().tolist(), only_changed_action_indices=changed,
            diagnostic_is_not_training_or_acceptance=True)
        return applied

    def save(directory, applied, actor, checkpoint):
        current = actor.state_dict()
        if current.keys() != checkpoint['model_state_dict'].keys() or not all(
                torch.equal(value, checkpoint['model_state_dict'][key]) for key, value in current.items()):
            raise ValueError('Frozen actor/critic/std changed')
        if sha(source) != spec['checkpoint_sha256'] or len(noises) != 1000:
            raise ValueError('Changed source or incomplete original20s Gaussian sequence')
        np.savez_compressed(Path(directory) / 'sampling_noise.npz', epsilon=np.stack(noises),
            sampling_std=applied.cpu().numpy(), saved_std=actor.std.detach().cpu().numpy())
        with (Path(directory) / 'sampling_probe_integrity.json').open('x') as stream:
            json.dump(dict(**metadata, all_model_tensors_bitwise_unchanged=True,
                gaussian_draw_count=len(noises), additional_RNG_draws=0), stream, indent=2)
            stream.write('\n')

    sys.argv = [str(SCRIPT), *runtime.base.tires.arguments(arguments)]
    exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
        _install_quality_criteria=runtime.criteria_fix.install,
        _observe_source200=runtime.private_runtime().observe_environment,
        _sampling_probe_std=sampling_std, _sampling_probe_metadata=lambda: dict(metadata),
        _record_probe_noise=lambda noise: noises.append(noise.detach().cpu().numpy().copy()),
        _save_probe_noise=save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--sampling-label', choices=('saved', 'probe'), required=True)
    args, rest = parser.parse_known_args()
    if not rest or rest.pop(0) != 'replay':
        parser.error('Use replay with a source-bound specification')
    replay(rest, args.specification, args.sampling_label)
