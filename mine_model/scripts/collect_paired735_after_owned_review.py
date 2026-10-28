"""Collect the qualified paired735 audit after one named owned review unit.

Waiting and report collection cannot launch a simulator, choose a model or
extend PPO. A terminal review error is left for agent inspection.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
from audit_plain27_after_queued_reviews import wait_unit
import audit_paired_retained735_tracking as collector

SCRIPT = Path(__file__).resolve()


def run(request_path):
    request_path = Path(request_path).resolve()
    request = collector.read(request_path)
    status = Path(request['wait_status'])
    qualification = collector.read(request['collector_qualification'])
    if (request['waiter_script_sha256'] != collector.sha(SCRIPT)
            or request['collector_script_sha256'] != collector.sha(collector.SCRIPT)
            or not qualification['qualified']
            or qualification['source_pins'][str(collector.SCRIPT)] != collector.sha(collector.SCRIPT)
            or any(collector.sha(path) != digest for path, digest in request['request_source_pins'].items())
            or not request['after_unit'] or Path(request['output']).exists()
            or not status.resolve().is_relative_to(collector.ROOT / 'mine_model/results')):
        raise ValueError('Require the unchanged real735-qualified collector and a fresh owned output')
    unit = wait_unit(request['after_unit'], status)
    complete = collector.read(request['complete_all_scopes'])
    if (not complete['completed'] or complete['checkpoint_sha256'] != request['candidate_checkpoint_sha256']
            or not complete['original735_static810_original120_complete']
            or not complete['signed_fullspeed_height_complete'] or not complete['small_yaw120_complete']):
        raise ValueError('Finish every exact selected-candidate scope before paired source comparison')
    result = collector.audit(request_path)
    if result['candidate_checkpoint_sha256'] != request['candidate_checkpoint_sha256']:
        raise ValueError('Paired collector read another branch at the same update ordinal')
    write_json(status, dict(observed_at=datetime.now().astimezone().isoformat(),
        state='paired735_complete_agent_source_selection_required', after_unit=request['after_unit'],
        original_review_terminal_state=unit, report=request['output'],
        summaries=result['summaries'], literal_rescues=len(result['literal_rescues']),
        literal_regressions=len(result['literal_regressions']),
        simulator_started=False, optimizer_executed=False, training_modified=False,
        can_select_source=False, can_extend_training=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.request)
    except BaseException as error:
        request = collector.read(args.request)
        write_json(Path(request['wait_status']).with_name('paired735_collector_error.json'),
            dict(observed_at=datetime.now().astimezone().isoformat(), error=repr(error),
                PPO_restarted=False, simulator_started=False, optimizer_executed=False,
                training_modified=False, policy_accepted=False))
        raise
    print(json.dumps(dict(verified=report['verified'], summaries=report['summaries'], policy_accepted=False)))
