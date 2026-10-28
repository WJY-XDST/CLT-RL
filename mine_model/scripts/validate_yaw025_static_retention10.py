"""Use the unchanged static-retention validators for the yaw025 descendant."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import source1201_orientation40
import joint_pd_source2201_yaw025_runtime as runtime
import validate_quality50_static_retention10 as template
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_yaw025_static_retention_qualification_20261018.json'
LAUNCH = RESULTS / 'joint_pd_source2201_yaw025_qualification_20261018/training_launch.json'
FULL51_LAUNCH = RESULTS / 'joint_pd_yaw025_paired51_launch_20261018.json'


def adapted():
    spec = importlib.util.spec_from_file_location('_yaw025_static_retention', template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runtime, module.SCRIPT, module.QUALIFICATION = runtime, SCRIPT, QUALIFICATION
    original_dependencies = module.dependencies

    def dependencies():
        result = original_dependencies()
        result[str(Path(template.__file__))] = sha(template.__file__)
        return result

    module.dependencies = dependencies
    return module


def wait_review(training_output, full51):
    training = json.loads(LAUNCH.read_text())
    review = json.loads(FULL51_LAUNCH.read_text())
    if Path(training['output']) != training_output or Path(review['output']) != full51:
        raise ValueError('Wait only on the exact owned native200 and paired51')
    while True:
        try:
            state = json.loads((full51 / 'status.json').read_text())
            ready = (state['state'] == 'agent_review_required' and not state['diagnostic_active']
                and all((full51 / 'candidate' / name).is_file() for name in (
                    'motion025_full_height_diagnosis.json', 'completed.json', 'trace.csv')))
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
        if ready:
            return
        raw = subprocess.check_output(['systemctl', '--user', 'show', review['unit'],
            '--property=MainPID,ActiveState,SubState,ExecMainStatus,LoadState'], text=True)
        owner = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        row = dict(observed_at=datetime.now().astimezone().isoformat(),
            state='waiting_complete200_full27_full51', exact_review_unit=review['unit'],
            owner=owner, training_output=str(training_output), full51=str(full51),
            checkpoint_selected=False, training_restarted=False, policy_accepted=False)
        write_json(RESULTS / 'joint_pd_yaw025_retention_waiter_status_20261018.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if (owner['ActiveState'] not in ('active', 'activating', 'deactivating', 'reloading')
                or (owner['SubState'] == 'exited' and owner['MainPID'] == '0')):
            raise ValueError('Exact review owner ended without complete evidence; inspect without repeating PPO')
        time.sleep(30.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-checkpoint', type=Path)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--full51', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    module = adapted()
    if args.qualify_checkpoint:
        if args.wait:
            parser.error('CPU qualification never waits or starts native validation')
        result = module.qualify(args.qualify_checkpoint, args.training_output, args.output)
    else:
        if args.full51 is None:
            parser.error('--full51 is required')
        if args.wait:
            wait_review(args.training_output.resolve(), args.full51.resolve())
        result = module.run(args.training_output, args.full51, args.output)
    print(json.dumps(dict(qualified=result.get('qualified'), state=result.get('state'),
        static_retention_verified=result.get('static_retention_verified'),
        policy_accepted=False), ensure_ascii=False))
