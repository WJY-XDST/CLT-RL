"""Preserved signed fullspeed height comparison for6601 and6801."""
from datetime import datetime
import copy
import json
from pathlib import Path

from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import review_lin300_fullspeed_height_transitions as original
import review_source6601_targeted_turn030_block200 as post
import joint_pd_source6601_targeted_turn030_runtime as runtime
import qualify_source6601_targeted_turn030_transfer as qualifier

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = qualifier.OUTPUT / 'fullspeed_height/qualification.json'
SOURCE_QUALIFICATION = RESULTS / 'joint_pd_source6401_yaw200_qualification_20261021/fullspeed_height/qualification.json'
OUTPUT = RESULTS / 'joint_pd_targeted_turn030_fullspeed_height_transitions_source6601_candidate6801_20261021'
AFTER_UNIT = 'clt-joint-pd-source6801-targeted-turn030-original-and-retained-reviews-after200-20261021.service'
_api = adapter.private_module(original.SCRIPT, '_targeted_turn030_signed_fullspeed_height_steps')
_api.SCRIPT, _api.runtime, _api.QUALIFICATION, _api.OUTPUT = SCRIPT, runtime, QUALIFICATION, OUTPUT
_api.ORIGINAL_COMPLETE = post.OUTPUT / 'complete_with_retained_interior_reviews.json'
_api.ORIGINAL_QUALIFICATION = post.QUALIFICATION
ADAPTER = adapter.retarget(_api, ('run',),
    integers={'*': {6201: 6601, 6401: 6801, 128020: 136020}},
    exact_strings=post.metadata_strings(original.run, {6201: 6601, 6401: 6801}))


def read(path):
    return json.loads(Path(path).read_text())


def qualify():
    if QUALIFICATION.exists() or OUTPUT.exists():
        raise ValueError('Preserve signed-transition qualifications/data')
    decision = runtime.repair_ready()
    old = read(SOURCE_QUALIFICATION)
    original.core.intact(old['source_pins'])
    launch = read(qualifier.OUTPUT / 'training_launch.json')
    source, _, _ = runtime.anchor()
    if (not old['qualified'] or old['command_plan'] != original.plan()
            or old['seeds'] != original.SEEDS or launch['target_actual_updates'] != 6801):
        raise ValueError('Keep all nine signed trajectories, full height span and paired seeds')
    proof = copy.deepcopy(old)
    proof.update(observed_at=datetime.now().astimezone().isoformat(), after_unit=AFTER_UNIT,
        training=launch['output'], output=str(OUTPUT),
        original_complete_required=str(_api.ORIGINAL_COMPLETE), source_checkpoint=str(source),
        source_checkpoint_sha256=sha(source), source_actual_updates=6601,
        candidate_actual_updates=6801, candidate_all17_Adam_step=136020,
        source_completed_signed_transition_review=decision['source_fullspeed_height_transition_review'],
        only_source_target_ownership_metadata_adapted=True, metadata_adapter=ADAPTER,
        original_qualification=str(SOURCE_QUALIFICATION), policy_accepted=False)
    for p in (SCRIPT, original.SCRIPT, post.SCRIPT, post.QUALIFICATION, SOURCE_QUALIFICATION,
            runtime.SCRIPT, qualifier.OUTPUT / 'training_launch.json',
            Path(decision['source_fullspeed_height_transition_review'])):
        proof['source_pins'][str(p)] = sha(p)
    original.core.intact(proof['source_pins'])
    QUALIFICATION.parent.mkdir()
    original.core.save(QUALIFICATION, proof)
    return proof


def run():
    proof = read(QUALIFICATION)
    _api.TRAINING = Path(proof['training'])
    return _api.run()


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    result = qualify() if args.qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'completed', 'policy_accepted') if k in result}))
