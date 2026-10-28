"""Qualify retained state, criteria-loading fix and frozen quality rewards."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import rsl_rl.runners as runners

import source1201_orientation40
import joint_pd_source2001_quality50_runtime as repair
from auto_train_mine import write_json
from joint_pd_continuation import sha, optimizer_step, validate_continuation_config
from qualify_source801_drift50_transfer import differences

SCRIPT = Path(__file__).resolve()


def cpu(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(exist_ok=False)
    repair.repair_ready()
    predicate_proof = repair.criteria_fix.qualify_predicates()
    saved, cfg = repair.require_source(source, transition=True)
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    if (agent['algorithm']['class_name'] != 'PPO'
            or agent['algorithm'].get('joint_height_feedback_coef', 0.)
            or agent['algorithm'].get('joint_mean_guard_coef', 0.)):
        raise ValueError('Keep ordinary full-policy PPO with no auxiliary loss')
    options = copy.deepcopy(agent['policy'])
    if options.pop('class_name') != 'ActorCritic':
        raise ValueError('Keep the original plain27 actor and critic')
    observation = TensorDict({'policy': torch.linspace(-.2, .2, 32 * 27).reshape(32, 27)}, batch_size=[32])
    observation['policy'][:, 5] = -1.
    observation['policy'][:, 8] = 1.5

    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(observation, agent['obs_groups'], 6, **options)
        runner = object.__new__(cls)
        runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
        runner.current_learning_iteration = 0
        runner.logger_type, runner.disable_logs = 'tensorboard', True
        return runner

    first = output / 'first_load'
    first.mkdir()
    upstream, cls = repair.private_runtime(source).install_runner(first)
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        repair.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'model')
        repair.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'all17_Adam')
        with torch.no_grad():
            before = runner.alg.policy.act_inference(observation).clone()
        expected = copy.deepcopy(cfg)
        expected['rewards'][repair.REWARD_KEY] = repair.QUALITY_FLOOR
        delta = {'rewards.' + repair.REWARD_KEY: repair.QUALITY_FLOOR}
        validate_continuation_config(cfg, expected, cfg['commands'], reward_overrides=delta)
        if differences(cfg, expected) != ['/rewards/' + repair.REWARD_KEY]:
            raise ValueError('Only one existing reward configuration field may change')
        roundtrip = output / 'roundtrip'
        roundtrip.mkdir()
        shutil.copytree(source.parent / 'params', roundtrip / 'params')
        (roundtrip / 'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))
        offline = roundtrip / 'offline_source2001_quality50_untrained.pt'
        runner.save(str(offline))
        runners.OnPolicyRunner = upstream
        second_dir = output / 'second_load'
        second_dir.mkdir()
        upstream2, cls2 = repair.private_runtime().install_runner(second_dir)
        try:
            restored = bare(cls2)
            restored.load(str(offline), load_optimizer=True, map_location='cpu')
            repair.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_model')
            repair.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_all17_Adam')
            with torch.no_grad():
                if not torch.equal(before, restored.alg.policy.act_inference(observation)):
                    raise ValueError('Untrained metadata roundtrip changed deterministic inference')
        finally:
            runners.OnPolicyRunner = upstream2
        try:
            repair.require_source(source)
        except ValueError:
            pass
        else:
            raise ValueError('Undeclared old-profile transfer was admitted')
        argv, final, _ = repair.command(source, cfg, 'offline_source2001_quality50_probe', 200,
            cfg['commands'], output / 'not_a_native_stop_request', reward_overrides=delta)
        if (final != 2200 or argv[argv.index('--target_iteration') + 1] != '2201'
                or argv[argv.index('--num_envs') + 1] != '4096'
                or argv.count(repair.FLAG) != 1 or argv.count('agent.policy.class_name=ActorCritic') != 1):
            raise ValueError('Preserve the exact200/full4096/plain27 CLI contract')
        repair.evaluator_tree()
        result = dict(qualified=True, profile=repair.profile(), source_checkpoint=str(source),
            source_sha256=sha(source), source_actual_updates=repair.saved_updates(source),
            source_all17_Adam_step=optimizer_step(saved), all17_Adam_bitwise_preserved=True,
            actor_and_critic_bitwise_preserved=True, deterministic_inference_bitwise_equal=True,
            saved_std_bitwise_preserved=True, actual_saved_adaptive_LR_preserved=True,
            criteria_dependency_fix=predicate_proof, both_violation_penalties_still_zero=True,
            one_configuration_field_changed=True, only_reward_change={repair.REWARD_KEY: [1., repair.QUALITY_FLOOR]},
            implementation_sha256=sha(repair.SCRIPT), qualifier_sha256=sha(SCRIPT),
            criteria_fix_sha256=sha(repair.criteria_fix.SCRIPT),
            actual_PPO_updates_executed=0, simulator_started=False,
            offline_checkpoint_not_trained=True, policy_accepted=False)
        write_json(output / 'qualification.json', result)
        return result
    finally:
        runners.OnPolicyRunner = upstream


def frozen(previous, output):
    before, after = Path(previous) / 'replay', Path(output) / 'native_reward_transfer'
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = read(before, 'manifest.json'), read(after, 'manifest.json')
    ca, cb = read(before, 'resolved_env.json'), read(after, 'resolved_env.json')
    da, db = read(before, 'completed.json'), read(after, 'completed.json')
    if (a['checkpoint_sha256'] != repair.ANCHOR_SHA or b['checkpoint_sha256'] != repair.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not da['completed'] or not db['completed']
            or da['samples'] != 27000 or db['samples'] != 27000
            or da['resets'] != 27 or db['resets'] != da['resets']):
        raise ValueError('Require the same full27 source trajectories, including all failed/reset trials')
    if (differences(ca, cb) != ['/rewards/' + repair.REWARD_KEY]
            or ca['rewards'][repair.REWARD_KEY] != 1.
            or cb['rewards'][repair.REWARD_KEY] != repair.QUALITY_FLOOR
            or cb['rewards']['tracking_goal_violation'] != 0.
            or cb['rewards']['tracking_goal_component_violation'] != 0.):
        raise ValueError('Only the existing smooth motion reward multiplier changes')
    old, new = [np.genfromtxt(root / 'trace.csv', delimiter=',', names=True) for root in (before, after)]
    changed = {'reward_total', 'reward_track_lin_vel', 'reward_track_ang_vel'}
    if (old.dtype.names != new.dtype.names or len(new) != 27000
            or any(not np.array_equal(old[key], new[key]) for key in old.dtype.names if key not in changed)):
        raise ValueError('Frozen physical outputs, actions or another reward changed')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[key], y[key]) for key in x.files):
            raise ValueError('Frozen raw observations or deterministic action means changed')
    criteria = json.loads((repair.ROOT / 'mine_model/config/tracking_acceptance.json').read_text())
    # Independent NumPy reconstruction uses recorded post-physics height,
    # pitch and roll, not pre-physics policy observations.
    h = np.abs(new['height_m'] - new['height_cmd_m']) / (criteria['relative_error'] * np.abs(new['height_cmd_m']))
    angle = np.maximum(np.abs(new['pitch_rad']), np.abs(new['roll_rad'])) * (180. / np.pi) / criteria['body_p95_deg']
    q = repair.QUALITY_FLOOR + (1. - repair.QUALITY_FLOOR) / (
        (1. + np.maximum(h - .5, 0.) ** 2) * (1. + np.maximum(angle - .5, 0.) ** 2))
    lin = 2. * np.exp(-(new['speed_cmd_m_s'] - new['speed_m_s']) ** 2
        / cb['rewards']['tracking_sigma']) * b['policy_dt'] * q
    yaw = np.exp(-(new['yaw_cmd_rad_s'] - new['yaw_rate_rad_s']) ** 2
        / cb['rewards']['tracking_sigma_ang']) * b['policy_dt'] * q
    formula = float(max(np.max(np.abs(lin - new['reward_track_lin_vel'])),
                        np.max(np.abs(yaw - new['reward_track_ang_vel']))))
    total_errors = np.abs(new['reward_total'] - old['reward_total']
        - (new['reward_track_lin_vel'] - old['reward_track_lin_vel'])
        - (new['reward_track_ang_vel'] - old['reward_track_ang_vel']))
    # The source includes27 unsafe terminations with a -20 reward. Account
    # for FP32 total-sum roundoff at that magnitude; no tracking gate changes.
    roundoff = 8. * np.finfo(np.float32).eps * np.maximum.reduce((
        np.ones(len(new)), np.abs(old['reward_total']), np.abs(new['reward_total'])))
    total = float(total_errors.max())
    if formula > 2.e-7 or np.any(total_errors > roundoff) or q.min() < repair.QUALITY_FLOOR or q.max() > 1.:
        raise ValueError('Independent quality formula, bounds or summed reward delta differs')
    patch = read(after, 'quality_criteria_runtime_patch.json')
    if (patch['implementation_sha256'] != sha(repair.criteria_fix.SCRIPT)
            or not patch['both_violation_penalties_zero']
            or patch['loaded_criteria_evidence']['sha256'] != sha(repair.ROOT / 'mine_model/config/tracking_acceptance.json')):
        raise ValueError('Require the actual independently loaded original criteria')
    result = dict(qualified=True, source_sha256=repair.ANCHOR_SHA,
        only_reward_change={repair.REWARD_KEY: [1., repair.QUALITY_FLOOR]},
        full27_samples=27000, identical_source_resets_including_failures=27,
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        independent_quality_formula_max_error=formula, total_reward_delta_max_error=total,
        total_reward_delta_FP32_roundoff_bound_range=[float(roundoff.min()), float(roundoff.max())],
        actual_quality_range=[float(q.min()), float(q.max())],
        both_violation_penalties_zero=True, criteria_loaded_by_quality_reward_alone=True,
        source_data_sha256={str(root / name): sha(root / name) for root in (before, after)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')},
        implementation_sha256=sha(repair.SCRIPT), qualifier_sha256=sha(SCRIPT),
        criteria_fix_sha256=sha(repair.criteria_fix.SCRIPT),
        PPO_updates_performed=0, policy_accepted=False,
        limitation='Frozen reward and physical equivalence only. Source stability/height is not accepted.')
    write_json(Path(output) / 'native_reward_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-previous', type=Path)
    args = parser.parse_args()
    result = frozen(args.frozen_previous, args.output) if args.frozen_previous else cpu(args.source, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_data_sha256'}, ensure_ascii=False))
