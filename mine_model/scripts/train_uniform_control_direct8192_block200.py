"""Run ordinary PPO at the user's explicit8192 selection, without capacity probes.

The retained controller's internal ``capacity`` pointer locates an explicit
allocation receipt here. Its schema records zero probes; it is never presented
as a passed capacity test. PPO, full-state recovery, numerical readers and the
declared common-drive contract validator remain the retained implementations.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import train_uniform_control_joint_pd_block200_v3 as original
import train_uniform_control_joint_pd_block200_v4 as declared
import train_uniform_control_joint_pd_block200_v5 as routing
import train_AF7001_wheelD075_wheel005_block200 as resources
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def require_explicit_allocation(path, *, block_id, source, num_envs):
    value = original.read(path)
    assert value['schema'] == 'plain27_explicit_environment_allocation_v1'
    assert value['selection'] == 'explicit_user_direct8192_without_capacity_tests'
    assert value['block_id'] == block_id
    assert value['selected_num_envs'] == num_envs == 8192
    assert value['capacity_tests_performed'] is False and value['probe_updates'] == 0
    assert value['user_instructions'] == ['不用测', '直接8192']
    assert Path(value['source_checkpoint']).resolve() == Path(source).resolve()
    assert value['source_checkpoint_sha256'] == sha(source)
    assert value['automatic_4096_fallback'] is False
    return value


def record_direct_start(output, plan):
    assert plan['native_num_envs'] == 8192
    write_json(Path(output) / 'explicit8192_resource_policy.json', dict(
        observed_at=datetime.now(timezone.utc).isoformat(), selected_num_envs=8192,
        user_selected_direct_start=True, capacity_tests_performed=False,
        probe_updates=0, resource_preflight_wait_skipped=True,
        automatic_4096_fallback=False, training_parameters_changed_here=False))


def run(pointer_path):
    pointer_path = Path(pointer_path).resolve()
    pointer = original.read(pointer_path)
    request = original.read(pointer['request'])
    output = Path(pointer['planned_production_output']).resolve()
    allocation = Path(pointer['allocation']).resolve()
    assert allocation == (Path(pointer['capacity']) / 'complete.json').resolve()
    require_explicit_allocation(allocation, block_id=pointer['block_id'],
                                source=Path(request['source_checkpoint']), num_envs=8192)
    saved = (original.SCRIPT, original.subprocess, original.write_json,
             original.require_capacity_selection, resources.wait_resources)

    def write(path, value):
        if Path(path).resolve() == output / 'experiment_plan.json':
            value = routing.align_plan_metadata(value)
            value.pop('capacity_admission', None)
            value.update(environment_allocation=str(allocation),
                         environment_allocation_mode='explicit_user_direct8192',
                         capacity_tests_performed=False, actual_capacity_probe_updates=0,
                         resource_preflight_wait_skipped_by_user=True,
                         automatic_4096_fallback=False,
                         direct_allocation_controller=str(SCRIPT))
        return saved[2](path, value)

    original.SCRIPT = SCRIPT
    original.subprocess = routing.ForwardSubprocess(saved[1])
    original.write_json = write
    original.require_capacity_selection = require_explicit_allocation
    resources.wait_resources = record_direct_start
    try:
        # Keep the prospectively qualified common-drive metadata validator as
        # well as the v5 monitor/replay routing; no native PPO code is replaced.
        return declared.run(pointer_path)
    finally:
        (original.SCRIPT, original.subprocess, original.write_json,
         original.require_capacity_selection, resources.wait_resources) = saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.pointer)
    except BaseException as error:
        pointer = original.read(args.pointer)
        output = Path(pointer['planned_production_output'])
        output.mkdir(exist_ok=True)
        write_json(output / 'controller_inspection_required.json', dict(
            error=repr(error), checkpoint_files_preserved=True,
            training_not_automatically_restarted=True,
            direct8192_selected_by_user=True, capacity_tests_performed=False,
            automatic_4096_fallback=False, goal_remains_active=True,
            policy_accepted=False))
        raise
