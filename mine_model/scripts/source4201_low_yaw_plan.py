"""One reviewed low-yaw coverage repair within the retained plain27 runtime."""
import json
from pathlib import Path
from types import SimpleNamespace

import joint_pd_source4001_motion0625_runtime as runtime
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
DECISION = RESULTS / 'joint_pd_source4201_low_yaw_decision_20261019.json'
QUALIFICATION = RESULTS / 'joint_pd_source4201_low_yaw_qualification_20261019'
ANCHOR_SHA = 'bcc903c3187fc3760277aaec4d4fa8a68dd37503f3ec358da9126a1317829cba'
DELTA = dict(rewards={}, commands={'yaw_min_abs_rate': [.15, 0.]},
    exploration={}, controller={}, physics={})


def ready():
    decision = json.loads(DECISION.read_text())
    static = json.loads(Path(decision['source_static_retention_report']).read_text())
    gap = json.loads(Path(decision['command_coverage_gap_report']).read_text())
    pair = json.loads(Path(decision['frozen_sampling_pair_report']).read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 4201 or decision['source_all17_Adam_step'] != 84020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not decision['source_is_unaccepted_repair_branch']
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or not gap['sampled_command_hole_verified'] or gap['source_checkpoint_sha256'] != ANCHOR_SHA
            or pair['checkpoint_sha256'] != ANCHOR_SHA or not pair['exact_same_Gaussian_draws']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or decision['faster_curriculum_authorized']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require exact4201 own-static, physical, learning, coverage and matched-noise review')
    return decision


def require_source(path, transition=False):
    # This is a configuration continuation in the SAME execution profile. The
    # old family transfer flag and its historical4001 anchor must not be reused.
    if transition and sha(path) != ANCHOR_SHA:
        raise ValueError('Only the reviewed4201 state starts this coverage block')
    return runtime.require_source(path)


def api_runtime():
    return SimpleNamespace(**dict(runtime.__dict__, DECISION=DECISION, DELTA=DELTA,
        ANCHOR_SHA=ANCHOR_SHA, repair_ready=ready, require_source=require_source))
