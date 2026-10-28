"""One ordinary PPO review interval with a declared global command change.

Use the user's direct8192 allocation without capacity probes. The original
controller and numerical readers remain in use; counts and command-difference
metadata follow the actual typed configuration instead of a fixed quarter.
"""
import argparse
import copy
from pathlib import Path

import joint_pd_uniform_command_runtime as runtime
import train_uniform_control_joint_pd_block200_v3 as original
import train_uniform_control_direct8192_block200 as allocation
import train_uniform_control_joint_pd_block200_v5 as routing
import train_AF7001_wheelD075_wheel005_block200 as resources
from auto_train_mine import write_json

SCRIPT = Path(__file__).resolve()


def align_plan(value, recipe):
    value = copy.deepcopy(value)
    commands = value['training_config']['commands']
    changes = recipe['command_changes']
    assert not value['only_controller_change'] and not value['only_reward_change']
    assert all(commands[k] == v[1] for k, v in changes.items())
    standing = int(value['native_num_envs'] * commands['standing_env_fraction'])
    scale = commands['training_motion_scale']
    value['next_configuration_changes']['commands'] = copy.deepcopy(changes)
    value.update(fixed_standing_environments=standing,
        motion_training_environments=value['native_num_envs'] - standing,
        standing_env_fraction=commands['standing_env_fraction'],
        only_command_change=copy.deepcopy(changes), typed_saved_commands_unchanged=False,
        training_motion_scale=scale,
        applied_speed_range_m_s=[min(commands['ranges_reverse_lin_vel_x']) * scale,
                                max(commands['ranges_lin_vel_x']) * scale],
        applied_yaw_range_rad_s=[v * scale for v in commands['ranges_ang_vel_yaw']],
        command_sampler_algorithm_unchanged=True, condition_specific_sampling=False,
        environment_allocation_mode='explicit_user_direct8192', capacity_tests_performed=False,
        actual_capacity_probe_updates=0, automatic_4096_fallback=False,
        scheduled_readonly_milestones=[50, 100, 150, 200],
        original_native_PPO_and_all11_and_all4_numeric_readers_unchanged=True,
        global_command_controller=str(SCRIPT))
    assert value['profile']['fixed_standing_environment_count'] == standing
    value.pop('capacity_admission', None)
    for key in list(value):
        if key in ('unchanged_recipe_continuation',) or 'commands_controller_physics_and_reset_configurations_unchanged' in key:
            value.pop(key)
    return value


def run(pointer):
    pointer = Path(pointer).resolve()
    spec = original.read(pointer)
    output = Path(spec['planned_production_output']).resolve()
    runtime.configure(spec['uniform_recipe'])
    assert not runtime.recipe()['control_changes']
    selected = Path(spec['allocation']).resolve()
    assert selected == (Path(spec['capacity']) / 'complete.json').resolve()
    allocation.require_explicit_allocation(selected, block_id=spec['block_id'],
        source=Path(original.read(spec['request'])['source_checkpoint']), num_envs=8192)
    previous = (original.SCRIPT, original.subprocess, original.write_json,
                original.require_capacity_selection, resources.wait_resources)
    routes = {'monitor_uniform_control_joint_pd_learning_v3.py': SCRIPT.with_name('monitor_uniform_command_learning.py'),
              'review_uniform_control_joint_pd_block_v3.py': SCRIPT.with_name('review_uniform_command_block.py')}

    class Processes(routing.ForwardSubprocess):
        def routed(self, command):
            if isinstance(command, (list, tuple)) and len(command) > 1 and Path(command[1]).name in routes:
                command = list(command)
                command[1] = str(routes[Path(command[1]).name])
            return command

    def write(path, value):
        if Path(path).resolve() == output / 'experiment_plan.json':
            value = align_plan(value, runtime.recipe())
            value['environment_allocation'] = str(selected)
        elif Path(path).name == 'final_checkpoint_verification_readonly.json':
            value = dict(value, declared_global_command_scalar_changes=runtime.recipe()['command_changes'],
                         physical_controller_reward_and_class_sampler_unchanged=True)
        return previous[2](path, value)

    original.SCRIPT, original.subprocess, original.write_json = SCRIPT, Processes(previous[1]), write
    original.require_capacity_selection = allocation.require_explicit_allocation
    resources.wait_resources = allocation.record_direct_start
    try:
        return original.run(pointer)
    finally:
        (original.SCRIPT, original.subprocess, original.write_json,
         original.require_capacity_selection, resources.wait_resources) = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.pointer)
    except BaseException as error:
        spec = original.read(args.pointer)
        output = Path(spec['planned_production_output'])
        output.mkdir(exist_ok=True)
        write_json(output / 'controller_inspection_required.json', dict(error=repr(error),
            checkpoint_files_preserved=True, training_not_automatically_restarted=True,
            goal_remains_active=True, policy_accepted=False))
        raise
