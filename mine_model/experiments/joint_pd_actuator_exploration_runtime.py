"""A declared once-only actuator-vector std adjustment after verified full-state restoration.

The pinned parent supplies ordinary PPO, full-state continuation, global
controller/floor settings and all original replay readers. This adapter changes
only fixed actuator-role std minima before sampling and after optimizer steps.
It never selects parameters from a command, state or failed evaluation case.
"""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import joint_pd_uniform_source_refinement_runtime as base
from resume_vector_std_pulse import RestoredVectorStdPulse, install_future_vector_floor


SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = base.SCRIPT
ORIGINAL_CONFIGURE = base.configure
ORIGINAL_PROFILE = base.profile_for_count
ORIGINAL_IMMEDIATE = base.immediate_changes
ORIGINAL_COMMAND = base.command
ORIGINAL_REPLAY_COMMAND = base.replay_command
ORIGINAL_INSTALL = base.install_leg_exploration
ORIGINAL_QUALIFY = base.qualify
_path = None
_spec = None
_armed_checkpoint = None


def __getattr__(name):
    return getattr(base, name)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def configure(path):
    global _path, _spec
    path = Path(path).resolve()
    if _path is not None:
        assert path == _path, 'One immutable explicitly declared recipe per process'
        return base.recipe()
    specification = read(path)
    assert specification['schema'] == 'actuator_vector_postload_exploration_recipe_v1'
    parent_path = Path(specification['parent_recipe']).resolve()
    assert base.sha(parent_path) == specification['parent_recipe_sha256']
    ORIGINAL_CONFIGURE(parent_path)
    source, saved, _ = base.baseline()
    pulse = specification['postload_std_adjustment']
    # Constructing the helper validates the exact real source and its Adam state.
    RestoredVectorStdPulse(checkpoint=source, checkpoint_sha256=pulse['source_checkpoint_sha256'],
                     actual_updates=pulse['source_actual_updates'],
                     minimum_std_by_head=pulse['minimum_std_by_head'])
    assert pulse['source_actual_updates'] == base.saved_updates(source)
    assert pulse['minimum_std_by_head'] == [pulse['leg_minimum_std'], pulse['leg_minimum_std'],
        base.recipe()['exploration']['minimum_wheel_std'], pulse['leg_minimum_std'],
        pulse['leg_minimum_std'], base.recipe()['exploration']['minimum_wheel_std']]
    assert pulse['leg_minimum_std'] > base.recipe()['exploration']['minimum_leg_std']
    assert pulse['source_std'][2] >= pulse['minimum_std_by_head'][2]
    assert pulse['source_std'][5] >= pulse['minimum_std_by_head'][5]
    assert pulse['source_std'] == saved['model_state_dict']['std'].tolist()
    assert pulse['phase'] == 'after_full_restore_and_future_floor_before_first_stochastic_rollout'
    assert pulse['once_per_source_continuation'] and pulse['all_six_entries_trainable']
    assert not any(pulse[key] for key in
                   ('optimizer_reset', 'actual_LR_reset', 'Actor_Critic_mean_value_reset',
                    'case_specific_training_logic', 'reapply_to_descendant_checkpoints'))
    assert pulse['std_tensor_intentionally_changed']
    assert set(specification) == {'schema', 'parent_recipe', 'parent_recipe_sha256',
                                  'postload_std_adjustment'}
    _path, _spec = path, specification
    namespaces = list({id(value): value for value in [base.__dict__, *base.NAMESPACES]}.values())
    for namespace in namespaces:
        namespace.update(SCRIPT=SCRIPT, profile_for_count=profile_for_count)
    ORIGINAL_INSTALL.__globals__['install_leg_exploration'] = install_leg_exploration
    base.configure = configure
    base.profile_for_count = profile_for_count
    base.immediate_changes = immediate_changes
    base.command = command
    base.parent.configure = configure
    base.parent.command = command
    for namespace in namespaces:
        namespace['replay_command'] = replay_command
    return base.recipe()


def profile_for_count(count):
    result = ORIGINAL_PROFILE(count)
    result.update(
        profile='plain27_actuator_vector_exploration_v1',
        native_runtime_sha256=base.sha(SCRIPT),
        uniform_recipe_sha256=base.sha(_path),
        parent_recipe_sha256=_spec['parent_recipe_sha256'],
        explicit_postload_std_adjustment=copy.deepcopy(_spec['postload_std_adjustment']),
        saved_initial_std_reset=True,
        Actor_Critic_mean_value_weights_reset=False,
        all17_Adam_reset=False, actual_saved_LR_reset=False,
        full_source_state_verified_before_declared_std_change=True,
        initial_std_intentionally_changed_after_restoration=True,
        saved_leg_PD_rewards_action_maps_and_exploration_retained=False,
        saved_leg_PD_action_maps_and_exploration_retained=False,
        same_postload_std_minimum_for_every_head_and_command=False,
        same_global_future_floor_for_every_command_and_head=False,
        same_common_future_floor_for_every_command_and_head=False,
        same_future_floor_for_every_command_within_actuator_role=True,
        saved_initial_distribution_retained=False,
        minimum_leg_std=_spec['postload_std_adjustment']['leg_minimum_std'],
        base_guard_floors=copy.deepcopy(base.recipe()['exploration']),
        effective_future_minimum_std_by_head=copy.deepcopy(_spec['postload_std_adjustment']['minimum_std_by_head']),
        initial_wheel_std_intentionally_changed=False,
        actually_changed_initial_noise_heads=[0, 1, 3, 4],
        pulse_not_reapplied_to_descendant_checkpoint=True,
        pulse_helper_sha256=base.sha(Path(__file__).with_name('resume_vector_std_pulse.py')))
    result.pop('source_NN_sixstd_17Adam_actualLR_reset', None)
    result['source_state_tensors_intentionally_changed'] = ['std']
    return result


def immediate_changes(cfg, source_floors=None):
    result = ORIGINAL_IMMEDIATE(cfg, source_floors)
    pulse = _spec['postload_std_adjustment']
    result['exploration']['postload_vector_minimum_std'] = [
        copy.deepcopy(pulse['source_std']), copy.deepcopy(pulse['minimum_std_by_head'])]
    before_leg = (base._source_recipe['exploration'] if source_floors is None else source_floors)['minimum_leg_std']
    if before_leg == pulse['leg_minimum_std']:
        result['exploration'].pop('minimum_leg_std', None)
    else:
        result['exploration']['minimum_leg_std'] = [before_leg, pulse['leg_minimum_std']]
    return result


def command(source, cfg, label, updates, commands, stop_file):
    argv, final, saved = ORIGINAL_COMMAND(source, cfg, label, updates, commands, stop_file)
    at = argv.index('--recipe') + 1
    assert argv[at] == str(base._path)
    argv[at] = str(_path)
    assert argv[1] == str(SCRIPT)
    assert '--reset_exploration_std' not in argv and '--reset_optimizer' not in argv
    return argv, final, saved


def replay_command(checkpoint, output):
    argv = ORIGINAL_REPLAY_COMMAND(checkpoint, output)
    at = argv.index('--recipe') + 1
    assert Path(argv[at]).resolve() == base._path
    argv[at] = str(_path)
    assert argv[1] == str(SCRIPT)
    return argv


@contextmanager
def armed_checkpoint(checkpoint):
    global _armed_checkpoint
    assert _armed_checkpoint is None
    _armed_checkpoint = Path(checkpoint).resolve()
    try:
        yield
    finally:
        _armed_checkpoint = None


def install_leg_exploration(output, selected_profile=None):
    """Install the existing future floor first; adjust restored std second."""
    import wheel_legged_gym_isaaclab.exploration as exploration
    previous = ORIGINAL_INSTALL(output, selected_profile)
    existing_guard = exploration.enforce_positive_exploration
    if _armed_checkpoint is None:
        # Used only by the separately labelled CPU full-restore qualification.
        return previous
    checkpoint = _armed_checkpoint
    loaded_sha = base.sha(checkpoint)
    source, _, _ = base.baseline()
    pulse_spec = _spec['postload_std_adjustment']
    pulse = RestoredVectorStdPulse(checkpoint=source,
        checkpoint_sha256=pulse_spec['source_checkpoint_sha256'],
        actual_updates=pulse_spec['source_actual_updates'],
        minimum_std_by_head=pulse_spec['minimum_std_by_head'])

    def enforce(policy, optimizer, minimum_std=1.e-4, *, minimum_wheel_std=None):
        handle = existing_guard(policy, optimizer, minimum_std,
                                minimum_wheel_std=minimum_wheel_std)
        try:
            receipt = pulse.apply_once(policy, optimizer,
                loaded_checkpoint_sha256=loaded_sha,
                actual_learning_rate=optimizer.param_groups[0]['lr'])
            vector_handle, _ = install_future_vector_floor(
                policy, optimizer, pulse_spec['minimum_std_by_head'])
            receipt.update(loaded_checkpoint=str(checkpoint), profile=profile_for_count(8192),
                original_future_floor_installed_before_adjustment=True,
                base_guard_floors=copy.deepcopy(base.recipe()['exploration']),
                effective_future_minimum_std_by_head=copy.deepcopy(pulse_spec['minimum_std_by_head']),
                future_vector_floor_installed_after_declared_adjustment=True,
                future_projection_after_every_original_optimizer_step=True,
                original_PPO_loss_and_update_unchanged=True,
                restored_std_evidence_phase_separate_from_first_rollout=True)
            write(Path(output) / 'explicit_postload_exploration_runtime.json', receipt)
            for name in ('uniform_exploration_floor_runtime.json',
                         'leg_exploration_floor_runtime.json'):
                path = Path(output) / name
                evidence = read(path)
                evidence.update(
                    verification_phase='full_restoration_and_future_floor_before_declared_std_adjustment',
                    explicit_postload_std_adjustment_receipt=str(
                        Path(output) / 'explicit_postload_exploration_runtime.json'),
                    first_rollout_std_unchanged_from_saved=not receipt['applied'],
                    actual_std_after_declared_adjustment=policy.std.detach().cpu().tolist())
                write(path, evidence)
            class Handles:
                def remove(self):
                    handle.remove()
                    vector_handle.remove()
            return Handles()
        except BaseException:
            handle.remove()
            if 'vector_handle' in locals():
                vector_handle.remove()
            raise

    exploration.enforce_positive_exploration = enforce
    return previous


def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    assert int(arguments[arguments.index('--num_envs') + 1]) == 8192
    saved, _ = base.require_source(source)
    assert not saved.get('CPU_FIXTURE_ONLY_NOT_NATIVE_CHECKPOINT', False)
    with armed_checkpoint(source):
        return base.native(output, arguments)


def annotate_training_evidence(path, value):
    """Keep legacy restore receipts scoped to the phase they actually verify."""
    result = copy.deepcopy(value)
    name = Path(path).name
    if name == 'experiment_plan.json':
        result.update(profile=profile_for_count(8192), learned_std_reset=True,
            saved_initial_std_reset=True,
            explicit_postload_std_adjustment=copy.deepcopy(_spec['postload_std_adjustment']),
            source_mean_value_weights_and_all17_Adam_actualLR_preserved=True,
            source_std_preserved_at_full_load_then_intentionally_adjusted=True,
            only_additional_change_vs_completed_control='fixed leg std minimum with declared once-only initialization; wheel noise retained',
            actual_future_projection_floors=dict(base.recipe()['exploration'], minimum_leg_std=_spec['postload_std_adjustment']['leg_minimum_std']),
            same_common_future_floor_for_every_command_and_head=False,
            same_future_floor_for_every_command_within_actuator_role=True,
            initial_wheel_std_intentionally_changed=False,
            metadata_delta_scope='full saved source restored; declared std adjustment precedes first rollout')
        for key in ('saved_leg_PD_action_maps_exploration_and_reward_structure_retained',):
            result[key] = False
        result['original_observation_action_maps_leg_PD_rewards_and_exploration_retained'] = False
    elif name == 'final_checkpoint_verification_readonly.json':
        result.pop('source_initial_NN_std_Adam_and_LR_preserved', None)
        result.update(full_saved_NN_std_Adam_LR_preserved_at_load_before_declared_pulse=True,
            source_mean_value_weights_all17_Adam_and_actualLR_preserved_before_training=True,
            source_std_intentionally_adjusted_after_full_restore=True,
            explicit_std_runtime_receipt=str(Path(path).parent / 'native_learning_signal' /
                                            'explicit_postload_exploration_runtime.json'))
    return result


def train(request_path):
    import train_saved_joint_pd_review50 as workflow
    request = read(request_path)
    configure(request['recipe'])
    assert request['source_initial_std_reset'] is True
    assert request['postload_std_adjustment'] == _spec['postload_std_adjustment']
    import subprocess
    previous = workflow.write_json
    previous_subprocess = subprocess.run

    def routed(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and len(argv) > 2:
            argv = list(argv)
            if Path(argv[1]).resolve() == SCRIPT and argv[2] == 'compare':
                at = argv.index('--recipe') + 1
                assert Path(argv[at]).resolve() in (base._path, _path)
                argv[at] = str(_path)
            if Path(argv[1]).name in ('audit_joint_pd_policy_updates.py',
                                       'audit_parallel8192_saved_action_limits_v2.py'):
                from declared_vector_distribution_reader import qualify
                directory = Path(argv[2]).resolve()
                qualify(directory)
                argv = [argv[0], str(SCRIPT.with_name('declared_vector_distribution_reader.py')),
                    '--reader-script', argv[1], '--directory', str(directory), *argv[3:]]
        return previous_subprocess(argv, *args, **kwargs)

    def write_evidence(path, value):
        return previous(path, annotate_training_evidence(path, value))

    workflow.write_json = write_evidence
    subprocess.run = routed
    try:
        return base.train(request_path)
    finally:
        workflow.write_json = previous
        subprocess.run = previous_subprocess


def qualify_restore_phase(path, output):
    configure(path)
    result = ORIGINAL_QUALIFY(path, output)
    result.update(qualification_phase='full_source_restore_before_declared_std_adjustment',
                  first_rollout_exploration_not_qualified_by_this_phase=True,
                  explicit_adjustment_integration_qualification_still_required=True)
    write(output / 'qualification.json', result)
    return result


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        at = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:at]); configure(args.recipe)
        native(args.observation_output, sys.argv[at + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:]); configure(args.recipe)
        base.replay(remaining)
    elif mode == 'qualify-restore':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        qualify_restore_phase(args.recipe.resolve(), args.output.resolve())
    elif mode == 'compare':
        parser.add_argument('directory', type=Path); parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:]); configure(args.recipe)
        base.compare(args.directory.resolve(), args.through)
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:]); configure(args.recipe)
        train(args.request.resolve())
    else:
        raise ValueError(mode)
