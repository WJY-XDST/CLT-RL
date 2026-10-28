"""A matched ordinary-PPO entropy comparison; no std/optimizer reset."""
import copy
import json
from pathlib import Path

from joint_pd_priority import STATIC_CASES, trials


def entropy_zero_command(argv):
    prefix = 'agent.algorithm.entropy_coef='
    fields = [i for i, value in enumerate(argv) if value.startswith(prefix)]
    if len(fields) != 1 or float(argv[fields[0]].split('=', 1)[1]) != .001:
        raise ValueError('Require exactly the matched control entropy coefficient')
    if any(flag in argv for flag in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Never reset a matched entropy source')
    result = list(argv)
    result[fields[0]] = prefix+'0.0'
    return result


def canonical_command(argv):
    skipped = ('--observation-output', '--run_name', '--stop_file')
    result = []
    index = 0
    while index < len(argv):
        if argv[index] in skipped:
            index += 2
        else:
            result.append(argv[index])
            index += 1
    return result


def qualify_control(directory, source, expected, load_config, sha):
    directory = Path(directory)
    plan = json.loads((directory/'experiment_plan.json').read_text())
    status = json.loads((directory/'status.json').read_text())
    proof = json.loads((directory/'resume_verification.json').read_text())
    report = json.loads((directory/'replay/assessment.json').read_text())
    native = json.loads((directory/'native_learning_signal/completed_external.json').read_text())
    target = plan['target_actual_ppo_updates']
    if (plan['bounded_training_updates'] != 100 or target-plan['source_actual_ppo_updates'] != 100
            or plan['entropy_coefficient'] != .001 or plan['training_motion_scale'] != 0.
            or plan['only_reward_change'] != {'joint_target_leg_length_error_sq': [0., -500.]}
            or sha(source) != plan['source_sha256'] or status['training_active']
            or status['actual_ppo_updates'] != target or proof['actual_ppo_updates'] != target
            or proof['optimizer_step'] != target*20 or proof['optimizer_states'] != 17
            or not proof['all_tensors_finite'] or not all(proof['actor_rows_updated'])
            or not proof['shared_actor_updated'] or not proof['critic_updated']
            or sha(Path(status['checkpoint'])) != proof['checkpoint_sha256']
            or report['checkpoint_sha256'] != proof['checkpoint_sha256']
            or native['checkpoint_sha256'] != proof['checkpoint_sha256']
            or native['actual_observed_updates'] != 100 or native['optimizer_minibatches'] != 2000
            or native['PPO_updates_repeated'] != 0 or not native['all_probe_RNG_unchanged']
            or not native['original_control_and_PPO_loss_unchanged']
            or any(native['all_environment_alignment_max_errors'].values())):
        raise ValueError('Require a completed, checkpoint-matched control and the identical full source')
    for (name, _), (_, row) in trials(report).items():
        if name in STATIC_CASES and (not row['survived'] or row['resets'] or any(
                row['metrics'][key] >= row['limits'][key]-1.e-12
                for key in ('body_p95_deg', 'speed_mae_m_s', 'yaw_mae_rad_s'))):
            raise ValueError('Matched control must preserve stationary posture/drift')
    control = load_config(status['checkpoint'])
    before, after = copy.deepcopy(control), copy.deepcopy(expected)
    for value in (before, after):
        value.pop('log_dir', None)
    if before != after:
        raise ValueError('Keep the entire matched control environment/reward dictionary')
    return dict(directory=str(directory), source_sha256=plan['source_sha256'],
        matched_control_checkpoint_sha256=proof['checkpoint_sha256'],
        only_agent_difference={'entropy_coef': [.001, 0.]}, all_other_environment_config_equal=True,
        optimizer_reset=False, learned_std_reset=False, policy_accepted=False)
