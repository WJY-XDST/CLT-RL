"""Keep global curriculum metadata aligned before the unchanged PPO observer starts.

The v6 subprocess forwarding fix is retained. Only experiment-plan metadata is
corrected; the typed training configuration, native PPO and numerical readers
are never changed by this controller wrapper.
"""
import argparse
import copy
from pathlib import Path

import train_uniform_motion_block200_v6 as implementation
import joint_pd_uniform_motion_envelope_runtime_v4 as runtime


SCRIPT = Path(__file__).resolve()


def align_plan_metadata(value, changes):
    typed = copy.deepcopy(value['training_config'])
    commands = value['training_config']['commands']
    assert set(changes) == {'training_motion_scale', 'yaw_min_abs_rate'}
    assert all(commands[key] == transition[1] for key, transition in changes.items())
    scale = commands['training_motion_scale']
    value['training_motion_scale'] = scale
    value['only_command_change'] = copy.deepcopy(changes)
    value['typed_saved_commands_unchanged'] = False
    value['applied_training_speed_range_m_s'] = [
        commands['ranges_reverse_lin_vel_x'][0] * scale,
        commands['ranges_lin_vel_x'][1] * scale,
    ]
    value['applied_training_yaw_range_rad_s'] = [rate * scale for rate in commands['ranges_ang_vel_yaw']]
    value['command_metadata_aligned_before_readonly_monitor'] = True
    assert value['training_config'] == typed
    return value


def run(pointer):
    original = implementation.implementation.original
    previous_write, previous_script = original.write_json, implementation.SCRIPT

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = align_plan_metadata(value, runtime.recipe()['command_changes'])
        return previous_write(path, value)

    original.write_json, implementation.SCRIPT = write, SCRIPT
    try:
        return implementation.run(pointer)
    finally:
        original.write_json, implementation.SCRIPT = previous_write, previous_script


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.pointer.resolve())
    except BaseException as error:
        original = implementation.implementation.original
        request = original.read(args.pointer)
        output = Path(request['planned_production_output'])
        output.mkdir(exist_ok=True)
        original.write_json(output / 'controller_inspection_required.json', dict(
            error=repr(error), checkpoint_files_preserved=True,
            training_not_automatically_restarted=True, goal_remains_active=True, policy_accepted=False))
        raise
