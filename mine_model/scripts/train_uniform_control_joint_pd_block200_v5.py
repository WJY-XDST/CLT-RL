"""Reuse scalar-only ordinary PPO; route read-only milestone and stage reviews.

The source checkpoint, optimizer, exploration, native arguments and original
numeric readers are untouched. The facade delegates every subprocess constant
and attribute, including PIPE/STDOUT, and routes only the two reader scripts.
"""
import argparse
from pathlib import Path

import train_uniform_control_joint_pd_block200_v3 as original
import monitor_uniform_control_learning_milestones_v5 as monitor
import review_uniform_control_global_stage_v5 as review
from auto_train_mine import write_json


SCRIPT = Path(__file__).resolve()
ROUTES = {'monitor_uniform_control_joint_pd_learning_v3.py': monitor.SCRIPT,
          'review_uniform_control_joint_pd_block_v3.py': review.SCRIPT}


class ForwardSubprocess:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, key):
        return getattr(self.delegate, key)

    def routed(self, command):
        if isinstance(command, (list, tuple)) and len(command) > 1:
            target = ROUTES.get(Path(command[1]).name)
            if target is not None:
                command = list(command)
                command[1] = str(target)
        return command

    def Popen(self, command, *args, **kwargs):
        return self.delegate.Popen(self.routed(command), *args, **kwargs)

    def run(self, command, *args, **kwargs):
        return self.delegate.run(self.routed(command), *args, **kwargs)


def align_plan_metadata(value):
    value = dict(value)
    for key in ('height3_orientation40_velocity025_unchanged',
                'height3_orientation40_quality50_yaw025_unchanged',
                'unchanged_recipe_continuation',
                'existing_reward_command_controller_physics_and_reset_configurations_unchanged',
                'existing_exploration_commands_controller_physics_and_reset_configurations_unchanged'):
        value.pop(key, None)
    commands = value['training_config']['commands']
    scale = commands['training_motion_scale']
    assert not value['next_configuration_changes']['commands']
    value.update(training_motion_scale=scale,
                 applied_speed_range_m_s=[min(commands['ranges_reverse_lin_vel_x']) * scale,
                                          max(commands['ranges_lin_vel_x']) * scale],
                 applied_yaw_range_rad_s=[x * scale for x in commands['ranges_ang_vel_yaw']],
                 saved_command_configuration_unchanged=True,
                 declared_global_reward_scalars_only=True,
                 scheduled_readonly_milestones=[50, 100, 150, 200],
                 all_original11_and_original4_numeric_readers_unchanged=True,
                 legacy_profile_banners_not_authoritative_for_actual_commands=True)
    return value


def run(pointer):
    pointer = Path(pointer).resolve()
    output = Path(original.read(pointer)['planned_production_output'])
    old_script, old_process, old_write = original.SCRIPT, original.subprocess, original.write_json

    def write(path, value):
        if Path(path) == output / 'experiment_plan.json':
            value = align_plan_metadata(value)
        return old_write(path, value)

    original.SCRIPT, original.subprocess, original.write_json = SCRIPT, ForwardSubprocess(old_process), write
    try:
        return original.run(pointer)
    finally:
        original.SCRIPT, original.subprocess, original.write_json = old_script, old_process, old_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.pointer)
    except BaseException as error:
        pointer = original.read(args.pointer)
        destination = Path(pointer['planned_production_output'])
        destination.mkdir(exist_ok=True)
        write_json(destination / 'controller_inspection_required.json', dict(
            error=repr(error), checkpoint_files_preserved=True,
            training_not_automatically_restarted=True, goal_remains_active=True, policy_accepted=False))
        raise
