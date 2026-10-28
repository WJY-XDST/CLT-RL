"""One source3801 sampler expansion, retaining the original plain27 PPO state."""
from pathlib import Path
import sys

import joint_pd_source3601_motion0375_runtime as previous
import plain27_sampler_metadata_adapter as adapter
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'bf04c87ef6f4c1e1f0e185171529f030afb229d452848b9476e100a2d9df1ab5'
MOTION_SCALE = .5
FLAG = '--begin_verified_motion050'
DECISION = ROOT / 'mine_model/results/joint_pd_source3801_motion050_decision_20261019.json'
DELTA = dict(rewards={}, commands={'training_motion_scale': [.375, MOTION_SCALE]},
    exploration={}, controller={}, physics={})
_api = adapter.private_module(previous.__file__, '_source3801_motion050_original_runtime')
_api.SCRIPT, _api.ANCHOR_SHA, _api.MOTION_SCALE = SCRIPT, ANCHOR_SHA, MOTION_SCALE
_api.FLAG, _api.DECISION, _api.DELTA, _api.previous = FLAG, DECISION, DELTA, previous
ADAPTER_EVIDENCE = adapter.retarget(_api, ('profile', 'repair_ready', 'private_runtime'),
    integers={'*': {3601: 3801, 72020: 76020}},
    strings={'source3601': 'source3801', 'motion0375': 'motion050'},
    keyword_names={'parent_leg020_runtime_sha256': 'parent_motion0375_runtime_sha256'},
    comparisons={'private_runtime': [("cfg['commands']['training_motion_scale']", .25, .375)]})
_original_profile = _api.profile


def profile():
    result = _original_profile()
    result['sampler_metadata_adapter_sha256'] = sha(adapter.SCRIPT)
    return result


_api.profile = profile
for _name in ('KEY', 'base', 'criteria_fix', 'recipe', 'repair_ready', 'private_runtime', 'require_source',
        'install_leg_exploration', 'native', 'evaluator_tree', 'replay', 'command', 'replay_command'):
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
