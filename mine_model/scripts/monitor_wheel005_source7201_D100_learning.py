"""The same11 original read-only learning readers at50/100/200 updates."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import monitor_AF7001_yawsigma0025_learning as original
import monitor_AF7001_wheelD075_learning as owner_api
import joint_pd_wheel005_source7201_D100_runtime as runtime
import plain27_sampler_metadata_adapter as adapter


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(original.SCRIPT, '_source7201_D100_original_learning_readers')
api.SCRIPT, api.runtime = SCRIPT, runtime
api.CHECKS = tuple((
    'compare_wheel005_source7201_D100_recorded_joint_targets.py'
    if name == 'compare_parallel8192_recorded_joint_targets.py' else name, stem, options)
    for name, stem, options in original.CHECKS)
assert len(api.CHECKS) == 11
assert sum(a != b for a, b in zip(api.CHECKS, original.CHECKS)) == 1
METADATA = adapter.retarget(api, ('tick',), exact_strings={
    'learning_health_AF7001_yawsigma0025_prefix_': 'learning_health_wheel005_D100_prefix_'})
assert api.tick.__globals__ is api.__dict__


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--unit', required=True)
    args = cli.parse_args()
    output = args.output.resolve()
    status = output / 'learning_monitor_wheel005_D100_status.json'
    while True:
        live = owner_api.owner(args.unit)
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
        record.update(unit_state=live, metadata_adapter=METADATA,
            controller_factor=runtime.DELTA['controller'], exploration_factor={},
            only_one_declared_contract_reader_rebound=True, native_PPO_restarted=False)
        original.write(status, record)
        if record.get('state') == 'all_owned_prefix_diagnostics_completed':
            break
        if live.get('MainPID') == '0':
            assert live.get('Result') == 'success' and live.get('SubState') == 'exited', live
        time.sleep(15.)
