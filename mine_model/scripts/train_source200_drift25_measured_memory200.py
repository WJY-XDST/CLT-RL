"""Unchanged4096 drift-repair PPO with a measured native-memory reservation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from auto_train_mine import write_json
from joint_pd_continuation import sha
import train_source200_leg05_height200 as controller

ROOT = controller.ROOT
MEASURED = ROOT / 'mine_model/results/joint_pd_optimization_20261017_112929_286819_source200_stationary_broad200_4096/resource_explanation_for_user_20261017.json'
COMPLETED = MEASURED.parent / 'native_learning_signal/completed_external.json'
ORIGINAL_PREPARE = controller.prepare


def memory_requirement():
    measured = json.loads(MEASURED.read_text())
    complete = json.loads(COMPLETED.read_text())
    if (measured['current_num_envs'] != 4096 or complete['actual_observed_updates'] != 200
            or complete['optimizer_minibatches'] != 4000 or complete['PPO_updates_repeated']):
        raise ValueError('Require real complete4096 training with its measured native highwater')
    peak = measured['process_memory']['39090']['VmHWM'].split()
    if len(peak) != 2 or peak[1] != 'kB':
        raise ValueError('Requalify the native highwater units')
    peak_bytes = int(peak[0]) * 1024
    if not 11 * 1024**3 <= peak_bytes <= 12 * 1024**3:
        raise ValueError('Reassess the observed4096 memory profile')
    return peak_bytes, peak_bytes + 4 * 1024**3


def prepare(output, previous):
    plan, expected = ORIGINAL_PREPARE(output, previous)
    peak, requirement = memory_requirement()
    plan.update(resource_reservation_method='Measured native4096 RSS highwater plus4GiB reserve',
        measured_native_RSS_highwater_bytes=peak, combined_memory_required_bytes=requirement,
        physical_available_memory_required_bytes=6 * 1024**3,
        measured_highwater_was_sampled_midblock=True,
        completed_full200_under_same4096_configuration=True,
        native_configuration_and_transition_batch_unchanged=True)
    plan['source_files'].update({str(p):sha(p) for p in (Path(__file__).resolve(), MEASURED, COMPLETED)})
    write_json(Path(output) / 'experiment_plan.json', plan)
    return plan, expected


def wait_memory(directory):
    peak, requirement = memory_requirement()
    while True:
        if (directory / 'STOP').exists():
            raise RuntimeError('Explicit stop requested during resource wait')
        values = {line.split(':')[0]:int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        ready = values['MemAvailable'] >= 6 * 1024**3 and values['MemAvailable'] + values['SwapFree'] >= requirement
        write_json(directory / 'native_memory_preflight_measured4096.json', dict(
            updated_at=datetime.now().astimezone().isoformat(), ready=ready, num_envs=4096,
            available_bytes=values['MemAvailable'], swap_free_bytes=values['SwapFree'],
            required_combined_bytes=requirement, measured_native_RSS_highwater_bytes=peak,
            reserve_bytes=4 * 1024**3, native_config_changed=False))
        if ready:
            return
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        controller.prepare = prepare
        controller.helper.wait_memory = wait_memory
        controller.run(args.output, args.previous, args.unit)
