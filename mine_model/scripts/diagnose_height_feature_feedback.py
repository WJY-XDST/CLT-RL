"""Feature-aware replacement for the legacy frozen-policy feedback reader.

Reuse every original metric/counterfactual definition. Only its nested actor
forward is replaced in a private function; active source files stay intact.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_policy_feedback as original
from diagnose_height_feature_response import actor as feature_actor
from joint_pd_continuation import sha


def joint_pd_feedback(observations, means, manifest, cfg, state):
    tree = ast.parse(inspect.getsource(original.joint_pd_feedback))
    outer = tree.body[0]
    nested = [node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == 'actor']
    if len(nested) != 1 or [arg.arg for arg in nested[0].args.args] != ['value']:
        raise ValueError('Legacy feedback reader changed; requalify the scoped actor replacement')
    nested[0].body = ast.parse('return feature_actor(value, state)').body
    ast.fix_missing_locations(tree)
    namespace = dict(vars(original), feature_actor=feature_actor)
    exec(compile(tree, str(Path(__file__).resolve()) + ':private_original_metrics', 'exec'), namespace)
    return namespace['joint_pd_feedback'](observations, means, manifest, cfg, state)


def diagnose_joint_pd(directory, output):
    import sys
    from joint_pd_continuation import ROOT
    from wheel_legged_gym_isaaclab.robot_model import MODEL, validate_checkpoint_robot
    sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
    from joint_pd_height_feature_runtime import require_paired_feature_model
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = json.loads((directory / 'resolved_env.json').read_text())
    checkpoint = Path(manifest['checkpoint'])
    if sha(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('The independently replayed checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = dict(saved['model_state_dict'])
    width = state['actor.0.weight'].shape[1]
    if width == 29:
        paired = validate_checkpoint_robot(checkpoint)
        state['_encoder_imu_model'] = require_paired_feature_model(saved, paired, MODEL['hip_z'])
    elif width != 27:
        raise ValueError('Undeclared policy first-layer width')
    with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as arrays:
        report = joint_pd_feedback(arrays['observations'], arrays['means'], manifest, cfg, state)
    report.update(feature_aware_feedback_reader=True, internal_firstlayer_columns=width,
        original_feedback_metric_definitions_unchanged=True,
        diagnostic_sources_sha256={str(Path(p).resolve()): sha(p)
            for p in (__file__, original.__file__)}, policy_accepted=False)
    target = Path(output)
    if target.exists():
        raise ValueError('Preserve earlier feedback evidence')
    target.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = diagnose_joint_pd(args.directory, args.output)
    print(json.dumps(dict(iteration=value['iteration'], actor_parity=value['actor_inference_parity_max'],
                         internal_columns=value['internal_firstlayer_columns'], policy_accepted=False)))
