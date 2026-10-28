"""Reuse original50/100/200 learning readers with the declared drive profile."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import plain27_sampler_metadata_adapter as adapter
import monitor_AF7001_yawsigma0025_learning as original
import joint_pd_AF7001_wheelD075_runtime as runtime


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(original.SCRIPT, '_wheelD075_original_learning_readers')
api.SCRIPT, api.runtime = SCRIPT, runtime
EVIDENCE = adapter.retarget(api, ('tick',), exact_strings={
    'learning_health_AF7001_yawsigma0025_prefix_': 'learning_health_AF7001_wheelD075_prefix_'})
assert api.tick.__globals__ is api.__dict__ and api.runtime is runtime


def owner(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID',
        '-p', 'Result', '-p', 'SubState'], text=True, capture_output=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    status = output / 'learning_monitor_wheelD075_status.json'
    while True:
        live = owner(args.unit)
        try:
            record = api.tick(output, args.unit)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            assert live.get('MainPID') != '0', (live, repr(error))
            record = dict(observed_at=datetime.now().astimezone().isoformat(),
                state='waiting_for_owned_committed_data', detail=str(error), training_modified=False)
        except BaseException as error:
            original.write(status, dict(observed_at=datetime.now().astimezone().isoformat(),
                state='agent_inspection_required', error=repr(error), unit_state=live,
                training_modified=False, PPO_restarted=False, policy_accepted=False))
            raise
        record.update(unit_state=live, metadata_adapter=EVIDENCE, controller_factor=runtime.DELTA['controller'])
        original.write(status, record)
        if record.get('state') == 'all_owned_prefix_diagnostics_completed':
            break
        if live.get('MainPID') == '0':
            assert live.get('Result') == 'success' and live.get('SubState') == 'exited', live
        time.sleep(15.)
