"""Conditional source4001 sampler expansion; no promotion without complete evidence."""
from pathlib import Path
import sys
import types

import joint_pd_source3801_motion050_runtime as previous
import joint_pd_source3601_motion0375_runtime as template
import plain27_sampler_metadata_adapter as adapter
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '832b28199d07f35e68258af8c978dd4c5a309fc7030bb6e14b4288e634a8e571'
MOTION_SCALE = .625
FLAG = '--begin_verified_motion0625'
DECISION = ROOT / 'mine_model/results/joint_pd_source4001_motion0625_decision_20261019.json'
DELTA = dict(rewards={}, commands={'training_motion_scale': [.5, MOTION_SCALE]},
    exploration={}, controller={}, physics={})
_api = adapter.private_module(template.__file__, '_source4001_motion0625_original_runtime')
_api.SCRIPT, _api.ANCHOR_SHA, _api.MOTION_SCALE = SCRIPT, ANCHOR_SHA, MOTION_SCALE
_api.FLAG, _api.DECISION, _api.DELTA, _api.previous = FLAG, DECISION, DELTA, previous
ADAPTER_EVIDENCE = adapter.retarget(_api, ('profile', 'repair_ready', 'private_runtime'),
    integers={'*': {3601: 4001, 72020: 80020}},
    strings={'source3601': 'source4001', 'motion0375': 'motion0625'},
    keyword_names={'parent_leg020_runtime_sha256': 'parent_motion050_runtime_sha256'},
    comparisons={'private_runtime': [("cfg['commands']['training_motion_scale']", .25, .5)]})
_original_profile = _api.profile


def profile():
    result = _original_profile()
    result['sampler_metadata_adapter_sha256'] = sha(adapter.SCRIPT)
    return result


def install_leg_exploration(output):
    # Exported functions can belong to a private module. Retain their actual
    # defining globals, including their original imports and nested wrappers.
    function = previous.install_leg_exploration
    retained = types.FunctionType(function.__code__, dict(function.__globals__, profile=profile),
        function.__name__, function.__defaults__, function.__closure__)
    return retained(output)


_api.profile, _api.install_leg_exploration = profile, install_leg_exploration
for _name in ('KEY', 'base', 'criteria_fix', 'recipe', 'repair_ready', 'private_runtime', 'require_source',
        'native', 'evaluator_tree', 'replay', 'command', 'replay_command'):
    globals()[_name] = getattr(_api, _name)


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output', sys.argv[2]]:
            raise ValueError('Keep the native observation output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only ordinary PPO and frozen replay')
