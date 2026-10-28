"""Verify one completed200-update plain27 PD trial without running an optimizer."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import torch
import yaml

from joint_pd_continuation import ROOT, sha, tensor_finite
from qualify_source801_drift50_transfer import differences
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()


def verify(directory, destination=None):
    directory = Path(directory).resolve()
    destination = (Path(destination).resolve() if destination is not None else
        directory / 'final_checkpoint_verification_readonly_20261019.json')
    if destination.exists():
        raise ValueError('Preserve the existing immutable checkpoint verification')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    status = json.loads((directory / 'status.json').read_text())
    complete_path = directory / 'native_learning_signal/completed_external.json'
    complete = json.loads(complete_path.read_text())
    start, target = plan['source_actual_ppo_updates'], plan['target_actual_ppo_updates']
    count, minibatches = plan['bounded_training_updates'], plan['optimizer_minibatches_per_actual_update']
    if (count != 200 or target - start != count or minibatches != 20
            or not plan['ordinary_full_policy_PPO'] or plan['architecture_migrated']
            or plan['auxiliary_losses_enabled'] or plan['optimizer_reset']
            or plan['learned_std_reset'] or plan['action_map_or_controller_changed']
            or status['state'] != 'agent_review_required' or status['training_active']
            or complete['actual_observed_updates'] != count
            or complete['actual_ppo_updates'] != target or complete['PPO_updates_repeated']
            or complete['optimizer_minibatches'] != count * minibatches
            or not complete['verified_outside_Kit']):
        raise ValueError('Require the exact completed200/full27 ordinary retained-state trial')
    paths = [Path(path) for path, digest in plan['source_files'].items()
        if digest == plan['profile']['native_runtime_sha256']
        and Path(path).parent == ROOT / 'mine_model/experiments']
    if len(paths) != 1 or sha(paths[0]) != plan['profile']['native_runtime_sha256']:
        raise ValueError('Require the single native runtime pinned by this trial')
    spec = importlib.util.spec_from_file_location('_completed_trial_verification_runtime', paths[0])
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    source, checkpoint = Path(plan['source_checkpoint']), Path(status['checkpoint'])
    try:
        before, cfg = runtime.require_source(source)
    except ValueError:
        before, cfg = runtime.require_source(source, transition=True)
    after, current_cfg = runtime.require_source(checkpoint)
    if (sha(source) != plan['source_sha256'] or sha(checkpoint) != status['checkpoint_sha256']
            or sha(checkpoint) != complete['checkpoint_sha256']
            or Path(complete['checkpoint']) != checkpoint or not tensor_finite(before)
            or not tensor_finite(after)):
        raise ValueError('Checkpoint identity or finite retained state differs')
    for path, saved, actual in ((source, before, start), (checkpoint, after, target)):
        states = saved['optimizer_state_dict']['state']
        if (saved_updates(path) != actual or len(states) != 17
                or any(float(value['step']) != actual * minibatches for value in states.values())):
            raise ValueError('Require every original Adam state at the exact real-update clock')
    changed = [not torch.equal(before['model_state_dict']['actor.6.weight'][index],
                              after['model_state_dict']['actor.6.weight'][index]) for index in range(6)]
    if not all(changed):
        raise ValueError('Investigate a nonupdated actor output before selecting this model')
    expected = {'/log_dir'}
    expected.update('/rewards/' + key for key in plan['only_reward_change'])
    expected.update('/commands/' + key for key in plan['only_command_change'])
    delta = differences(cfg, current_cfg)
    if (set(delta) != expected or current_cfg['leg_control_mode'] != 'implicit_joint_position'
            or current_cfg['observation_space'] != 27):
        raise ValueError('Saved configuration differs beyond the declared isolated trial')
    agent = yaml.safe_load((checkpoint.parent / 'params/agent.yaml').read_text())
    if (agent['policy']['class_name'] != 'ActorCritic'
            or agent['policy'].get('actor_obs_normalization')
            or agent['algorithm']['class_name'] != 'PPO'
            or agent['algorithm'].get('joint_height_feedback_coef', 0.)
            or agent['algorithm'].get('joint_mean_guard_coef', 0.)):
        raise ValueError('Require the unchanged plain27 ActorCritic and ordinary PPO losses')
    load_path = directory / 'native_learning_signal/source200_runtime_load.json'
    guard_path = directory / 'native_learning_signal/leg_exploration_floor_runtime.json'
    load, guard = [json.loads(path.read_text()) for path in (load_path, guard_path)]
    initial_std = before['model_state_dict']['std'].tolist()
    if (not load['all_model_and_optimizer_tensors_bitwise_preserved']
            or load['checkpoint_sha256'] != sha(source) or load['source_actual_updates'] != start
            or load['all17_Adam_step'] != start * minibatches
            or load['initial_std'] != initial_std
            or load['optimizer_LR'] != before['optimizer_state_dict']['param_groups'][0]['lr']
            or guard['before_std'] != initial_std or guard['after_std'] != initial_std
            or not guard['all17_Adam_bitwise_preserved']
            or not guard['actor_and_critic_bitwise_preserved']
            or not guard['parameter_identity_preserved'] or guard['added_loss_or_backward']):
        raise ValueError('Initial saved model, all17 Adam, std or actual LR was reset')
    proofs, gradient, rates, max_errors = [], [], [], {}
    for index in range(1, count + 1):
        path = directory / f'native_learning_signal/audit_update_{index:04d}.json'
        record = json.loads(path.read_text())
        errors = record['all_environment_alignment_max_errors']
        if (record['update'] != index or record['total_environments'] != plan['native_num_envs']
                or len(errors) != 9 or any(errors.values()) or not record['probe_RNG_unchanged']
                or record['added_loss_or_backward'] or len(record['optimizer_steps']) != minibatches
                or sha(path) != complete['records_sha256'][str(path)]):
            raise ValueError('Resolve native alignment, RNG or optimizer accounting first')
        for key, value in errors.items():
            max_errors[key] = max(max_errors.get(key, 0.), value)
        for step in record['optimizer_steps']:
            gradient.append(step['actor_gradient_norm_after_global_clip'])
            rates.append(step['adaptive_learning_rate'])
        proofs.append(path)
    if min(gradient) <= 0. or min(rates) <= 0.:
        raise ValueError('Actor learning or actual adaptive LR stopped')
    if any(sha(path) != digest for path, digest in plan['source_files'].items()):
        raise ValueError('Pinned source or code changed during the trial')
    files = [SCRIPT, plan_path, complete_path, source, checkpoint, load_path, guard_path,
             checkpoint.parent / 'params/agent.yaml', checkpoint.parent / 'params/env.yaml', *proofs]
    result = dict(observed_at=datetime.now().astimezone().isoformat(), verified=True,
        source_checkpoint=str(source), source_checkpoint_sha256=sha(source), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), source_actual_updates=start, actual_ppo_updates=target,
        exact_actual_update_delta=count, all17_source_Adam_step=start * minibatches,
        all17_Adam_step=target * minibatches, all17_Adam_step_delta=count * minibatches,
        original_optimizer_minibatches=count * minibatches, repeated_last_iteration=False,
        all_model_and_Adam_tensors_finite=True,
        initial_source_model_Adam_std_and_actual_LR_bitwise_preserved=True, initial_std_not_reset=True,
        all_six_actor_output_heads_changed=changed, saved_std=after['model_state_dict']['std'].tolist(),
        saved_adaptive_LR=[group['lr'] for group in after['optimizer_state_dict']['param_groups']],
        saved_configuration_differences=delta, only_reward_change=plan['only_reward_change'],
        only_command_change=plan['only_command_change'], only_exploration_change=plan['only_exploration_change'],
        all200_all4096_alignment_max_errors=max_errors, all200_original_RNG_unchanged=True,
        actor_gradient_range=[min(gradient), max(gradient)], adaptive_LR_range=[min(rates), max(rates)],
        source_pins_unchanged=True, source_pin_count=len(plan['source_files']),
        source_files={str(path): sha(path) for path in files}, optimizer_executed=False,
        training_modified=False, policy_accepted=False,
        limitation='Checkpoint, optimizer and native data verification only; original full27/full51 and independent retention remain separate required evidence.')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    result = verify(args.directory, args.destination)
    print(json.dumps({key: result[key] for key in ('verified', 'actual_ppo_updates',
        'all17_Adam_step', 'saved_std', 'saved_adaptive_LR', 'source_pin_count')}))
