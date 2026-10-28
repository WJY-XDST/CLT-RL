"""Retained7201 ordinary PPO for the isolated .75 -> 1.0 wheel-drive trial.

Rewards, commands, six saved std, leg PD and feedforward remain unchanged.
This entry cannot train before the complete frozen318 and exact-source
capacity/CPU/frozen27 admission. Original runtime files stay immutable.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

import joint_pd_AF7001_wheelD075_runtime as previous
import joint_pd_AF7001_wheelD075_wheel005_runtime as source_runtime
import plain27_sampler_metadata_adapter as adapter
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_wheel005_source7201_D100_qualification_20261024'
DECISION = QUALIFICATION / 'proposal.json'
ADMISSION = QUALIFICATION / 'native_admission.json'
SOURCE_REPORT = RESULTS / 'joint_pd_optimization_20261024_004349_671271_7bf5f989_AF7001_D075_wheel005_200/final_checkpoint_verification_readonly_20261019.json'
ANCHOR_SHA = '3964b3a21906dad8160f08d8ca4e1d2310a6bcc2d179277729a63e8a92756785'
FLAG = '--repair_wheel005_source7201_driveD100'
DAMPING = 1.
NUM_ENVS = json.loads(ADMISSION.read_text())['selected_num_envs'] if ADMISSION.exists() else 8192
assert NUM_ENVS in (8192, 4096)
DELTA = dict(rewards={}, commands={}, exploration={}, physics={}, controller={
    'wheel_damping': [.75, 1.], 'robot.actuators.wheels.damping': [.75, 1.]})
KEY, family, base, bind = previous.KEY, previous.family, previous.base, previous.bind
recipe = previous.recipe
api = adapter.private_module(previous.SCRIPT, '_wheel005_source7201_D100_original_runtime')
POPULATION_METADATA = adapter.retarget(api, ('private_for_count',),
    integers={'*': {7001: 7201}})
_private_count, _native = api.private_for_count, api.native
expected_configuration = api.expected_configuration


@lru_cache(maxsize=1)
def baseline():
    final = json.loads(SOURCE_REPORT.read_text())
    source = Path(final['checkpoint'])
    saved, cfg = source_runtime.require_source(source)
    assert final['verified'] and final['checkpoint_sha256'] == sha(source) == ANCHOR_SHA
    assert final['actual_ppo_updates'] == 7201 and final['all17_Adam_step'] == 144020
    assert cfg['wheel_damping'] == cfg['robot']['actuators']['wheels']['damping'] == .75
    assert cfg['rewards']['tracking_sigma_ang'] == .01
    assert cfg['rewards']['yaw_rate_error_sq'] == -200.
    assert saved['model_state_dict']['actor.0.weight'].shape[1] == 27
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert all(float(s['step']) == 144020 for s in saved['optimizer_state_dict']['state'].values())
    return source, saved, cfg


def profile_for_count(count):
    assert count in (8192, 4096)
    value = copy.deepcopy(source_runtime.profile_for_count(count))
    value.pop('only_new_factor_vs_completed_same_source_D075_trial', None)
    value.update(profile='plain_wheel005_source7201_driveD100_v1',
        wheel_velocity_drive_damping=DAMPING,
        native_runtime_sha256=sha(SCRIPT),
        parent_runtime_sha256=sha(source_runtime.SCRIPT),
        continuation_source_sha256=ANCHOR_SHA,
        only_new_factor_vs_reviewed_source7201='wheel_velocity_drive_D_.75_to_1.0',
        minimum_leg_std=.01, minimum_wheel_std=.005, global_minimum_std=.005,
        authority='complete_source_reviews_frozen318_then_fresh_exact_recipe_capacity')
    return value


def profile():
    return profile_for_count(NUM_ENVS)


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer, 'Never reload originalmodel200 or reset saved state'
    path = Path(path).resolve()
    source, before, cfg = baseline()
    if sha(path) == ANCHOR_SHA:
        assert path == source.resolve()
        return before, cfg
    assert not transition, 'Only exact reviewed7201 crosses this drive boundary'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = expected_configuration(cfg, NUM_ENVS)
    expected['log_dir'] = current['log_dir']
    actual, states = saved_updates(path), saved['optimizer_state_dict']['state']
    assert tensor_finite(saved) and current == expected and actual >= 7201
    assert saved.get('infos', {}).get(KEY) == profile()
    assert len(states) == 17 and all(float(s['step']) == actual * 20 for s in states.values())
    assert saved['model_state_dict'].keys() == before['model_state_dict'].keys()
    assert all(saved['model_state_dict'][key].shape == value.shape
               for key, value in before['model_state_dict'].items())
    return saved, current


def repair_ready():
    proof = json.loads(DECISION.read_text())
    assert proof['selected_single_wheel_drive_trial']
    assert proof['all_required_review_scopes_complete'] and proof['frozen318_complete']
    assert proof['source_checkpoint_sha256'] == ANCHOR_SHA
    assert proof['source_actual_updates'] == 7201 and proof['source_all17_Adam_step'] == 144020
    assert proof['planned_actual_updates'] == 200 and proof['next_configuration_changes'] == DELTA
    assert proof['relative_error_target'] == .1
    assert not any(proof[key] for key in ('optimizer_reset', 'std_reset', 'actual_LR_reset',
        'original_model200_reload', 'acceptance_changed'))
    assert all(sha(path) == digest for path, digest in proof['source_pins'].items())
    for entry in proof['required_complete_review_receipts']:
        assert json.loads(Path(entry['path']).read_text())[entry['completion_key']] is True
    return proof


def review_ready(source):
    repair_ready()
    proof = json.loads(ADMISSION.read_text())
    assert proof['source_checkpoint_sha256'] == sha(source) == ANCHOR_SHA
    assert proof['selected_num_envs'] == NUM_ENVS and proof['next_configuration_changes'] == DELTA
    assert proof['CPU_qualified'] and proof['frozen27_qualified']
    assert proof['proposal_decision_sha256'] == sha(DECISION)
    assert all(sha(path) == digest for path, digest in proof['source_pins'].items())
    from check_plain27_wheel005_D100_environment_capacity import require_capacity_selection
    require_capacity_selection(proof['capacity_verified_summary'],
        block_id=proof['capacity_block_id'], source=source, num_envs=NUM_ENVS)
    return proof


def sync():
    api.NUM_ENVS = NUM_ENVS


def private_for_count(count, profile_api=None):
    sync()
    callback = (lambda: profile_for_count(count)) if profile_api is None else profile_api
    return _private_count(count, callback)


def private_runtime(source=None):
    if source is not None:
        require_source(source)
    return private_for_count(NUM_ENVS)


def install_leg_exploration(output, profile_api=None):
    callback = profile if profile_api is None else profile_api
    return source_runtime.install_leg_exploration(output, profile_api=callback)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    sync()
    saved, current = require_source(source)
    assert updates == 200 and cfg == current and commands == current['commands']
    assert not kwargs.pop('reward_overrides', {})
    assert kwargs.pop('minimum_wheel_std', .005) in (.005, .01, .03)
    assert kwargs.pop('joint_mean_guard_coef', 0.) == 0.
    assert kwargs.pop('joint_height_feedback_coef', 0.) == 0. and not kwargs
    anchor, _, before = baseline()
    argv, _, _ = source_runtime.command(anchor, before, label, 200,
        before['commands'], Path(stop_file), minimum_wheel_std=.005)
    argv[1] = str(SCRIPT)
    for flag, value in {'--checkpoint_path': source,
        '--target_iteration': saved_updates(source) + 200, '--num_envs': NUM_ENVS,
        '--observation-output': Path(stop_file).parent / 'native_learning_signal',
        '--run_name': label, '--stop_file': Path(stop_file)}.items():
        argv[argv.index(flag) + 1] = str(value)
    for name in ('env.wheel_damping', 'env.robot.actuators.wheels.damping'):
        assert argv.count(name + '=0.75') == 1
        argv[argv.index(name + '=0.75')] = name + '=1.0'
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    assert not any(flag in argv for flag in
        ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'))
    return argv, saved_updates(source) + 199, saved


def native(output, arguments):
    sync()
    return _native(output, arguments)


evaluator_tree = source_runtime.evaluator_tree


def replay_command(source, output):
    require_source(source)
    argv = source_runtime.replay_command(baseline()[0], output)
    argv[argv.index(str(source_runtime.SCRIPT))] = str(SCRIPT)
    argv[argv.index('--checkpoint') + 1] = str(Path(source).resolve())
    return argv


def frozen_transfer_command(source, output):
    assert sha(source) == ANCHOR_SHA
    return replay_command(source, output) + [FLAG, '--overrides_json',
        str(QUALIFICATION / 'wheel_damping100_only.json')]


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if sha(source) == ANCHOR_SHA and not transition:
        return source_runtime.replay(arguments)
    if transition:
        assert json.loads(Path(arguments[arguments.index('--overrides_json') + 1]).read_text()) == {'wheel_damping': 1.}
    else:
        assert '--overrides_json' not in arguments
    family._sampler_api.install_sampling_field_restore()
    import gymnasium as gym
    original_make = gym.make
    gym.make = family._sampler_api._wrap_make(original_make)
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
            _install_quality_criteria=family.criteria_fix.install,
            _configure_sampling=family._sampler_api._configure,
            _observe_source200=private_runtime().observe_environment))
    finally:
        gym.make = original_make


for name in ('SCRIPT', 'RESULTS', 'DECISION', 'ADMISSION', 'SOURCE_REPORT', 'ANCHOR_SHA',
             'FLAG', 'DAMPING', 'NUM_ENVS', 'DELTA', 'baseline', 'profile', 'profile_for_count',
             'require_source', 'repair_ready', 'review_ready', 'private_for_count',
             'private_runtime', 'install_leg_exploration'):
    setattr(api, name, globals()[name])


def __getattr__(name):
    return getattr(api, name)


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'native':
        boundary = sys.argv.index('--')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    elif mode == 'replay':
        replay(sys.argv[1:])
    else:
        raise ValueError('Only retained-state ordinary PPO and frozen replay')
