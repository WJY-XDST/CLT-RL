"""Read cached action clipping only after a verified complete ordinary-PPO50 node.

No environment, inference, optimizer or policy callback is imported or executed.
The 32 cached environment streams are diagnostic samples, not acceptance tests.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


LEG_COLUMNS = (0, 1, 3, 4)
HEAD_NAMES = ('LB', 'LL', 'RB', 'RL')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def unit_interval_probability(mu, sigma):
    """P(-1 < X < 1) for the recorded unbounded scalar Normal policy."""
    assert mu.shape == sigma.shape and np.all(sigma > 0)
    low = (-1. - mu) / (math.sqrt(2.) * sigma)
    high = (1. - mu) / (math.sqrt(2.) * sigma)
    low_survival = np.fromiter((.5 * math.erfc(float(x)) for x in low.flat),
                              dtype=np.float64, count=low.size).reshape(low.shape)
    high_survival = np.fromiter((.5 * math.erfc(float(x)) for x in high.flat),
                               dtype=np.float64, count=high.size).reshape(high.shape)
    return np.clip(low_survival - high_survival, 0., 1.)


def run(training, output, joint_map_reference=None):
    assert not output.exists(), output
    proof_path = training / 'final_checkpoint_verification_readonly.json'
    health_path = training / 'original11_health50_complete.json'
    plan_path = training / 'experiment_plan.json'
    proof, health, plan = map(read, (proof_path, health_path, plan_path))
    assert proof['verified'] and proof['all50_native_updates_verified']
    assert health['qualified'] and health['actual_updates'] == 50
    assert proof['actual_ppo_updates'] - plan['source_actual_ppo_updates'] == 50
    assert proof['all17_Adam_step'] == proof['actual_ppo_updates'] * 20
    assert sha(proof['checkpoint']) == proof['checkpoint_sha256']
    assert proof['source_initial_NN_std_Adam_and_LR_preserved']
    cfg = plan['training_config']
    assert plan['ordinary_full_policy_PPO'] and not plan['auxiliary_losses_enabled']
    assert cfg['leg_control_mode'] == 'implicit_joint_position'
    offset = np.asarray(cfg['leg_joint_position_offset'], dtype=np.float64)
    scale = np.asarray(cfg['leg_joint_position_scale'], dtype=np.float64)
    assert offset.shape == scale.shape == (4,) and np.all(scale > 0)
    dt = cfg['sim']['dt'] * cfg['decimation']
    assert abs(dt - .02) < 1.e-12
    signals = training / 'native_learning_signal'
    files = sorted(signals.glob('rollout_*.npz'))
    assert len(files) == 50
    pins = {str(path): sha(path) for path in [Path(__file__), proof_path, health_path, plan_path]}
    root = Path(__file__).resolve().parents[2]
    map_reference = read(joint_map_reference) if joint_map_reference else None
    if map_reference:
        assert map_reference['verified']
        assert np.array_equal(np.asarray(map_reference['saved_joint_offsets_rad']), offset)
        assert np.array_equal(np.asarray(map_reference['saved_joint_action_scales_rad']), scale)
        pins[str(joint_map_reference)] = sha(joint_map_reference)
    for relative in ['wheel_legged_isaaclab/wheel_legged_gym_isaaclab/leg_control.py',
                     'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py']:
        source_path = root / relative
        expected_pin = plan['source_files'].get(str(source_path))
        if expected_pin is None:
            assert source_path.name == 'leg_control.py' and map_reference is not None
            expected_pin = map_reference['source_pins'][str(source_path)]
        assert expected_pin == sha(source_path)
        pins[str(source_path)] = sha(source_path)
    parts = {key: [] for key in ('mu', 'sigma', 'actions', 'native_raw_actions',
                                 'native_leg_targets', 'dones')}
    ids, native_action_error, target_map_error = None, 0., 0.
    for number, path in enumerate(files, 1):
        audit_path = signals / f'audit_update_{number:04d}.json'
        audit = read(audit_path)
        assert audit['update'] == number and audit['total_environments'] == 8192
        assert audit['probe_RNG_unchanged'] and not audit['added_loss_or_backward']
        assert len(audit['all_environment_alignment_max_errors']) == 9
        assert all(value == 0 for value in audit['all_environment_alignment_max_errors'].values())
        if ids is None:
            ids = audit['sampled_environment_ids']
        assert ids == audit['sampled_environment_ids'] and len(ids) == 32
        with np.load(path, allow_pickle=False) as archive:
            for key in parts:
                value = archive[key].copy()
                assert value.shape[:2] == (48, 32) and np.isfinite(value).all()
                parts[key].append(value)
            raw = archive['native_raw_actions'][..., LEG_COLUMNS].astype(np.float64)
            targets = archive['native_leg_targets'].astype(np.float64)
            expected = offset + scale * np.clip(raw, -1., 1.)
            target_map_error = max(target_map_error, float(np.abs(targets - expected).max()))
            native_action_error = max(native_action_error, float(np.abs(
                archive['actions'] - archive['native_raw_actions']).max()))
        pins[str(path)], pins[str(audit_path)] = sha(path), sha(audit_path)
    assert native_action_error == 0. and target_map_error < 1.e-6
    data = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    mu = data['mu'][..., LEG_COLUMNS].astype(np.float64)
    sigma = data['sigma'][..., LEG_COLUMNS].astype(np.float64)
    raw = data['native_raw_actions'][..., LEG_COLUMNS].astype(np.float64)
    assert mu.shape == sigma.shape == raw.shape == (2400, 32, 4)
    probability = unit_interval_probability(mu, sigma)
    # For scalar clipping, E[d clip(X,-1,1)/d mu] = P(-1<X<1).
    # This says nothing about motor torque, closed-chain dynamics or contact.
    normalized_applied = (data['native_leg_targets'].astype(np.float64) - offset) / scale
    effective_noise = normalized_applied - np.clip(mu, -1., 1.)
    raw_noise = raw - mu
    soft = cfg['rewards']['joint_position_action_soft_limit']
    coefficient = cfg['rewards']['joint_position_action_saturation']
    bound = cfg['rewards']['clip_single_reward']
    raw_penalty = np.square(np.maximum(np.abs(np.clip(raw, -10., 10.)) - soft, 0.)).sum(-1)
    weighted_rate = raw_penalty * coefficient
    weighted_step = np.clip(weighted_rate, -bound, bound) * dt

    def summary(start, end):
        subset = slice(start, end)
        result = dict(environment_time_samples=(end-start)*32,
            leg_head_samples=(end-start)*32*4,
            raw_leg_mean_outside_unit_fraction=float((np.abs(mu[subset]) > 1.).mean()),
            sampled_raw_leg_action_outside_unit_fraction=float((np.abs(raw[subset]) > 1.).mean()),
            sampled_environment_time_any_leg_action_outside_unit_fraction=float(
                (np.abs(raw[subset]) > 1.).any(-1).mean()),
            Gaussian_expected_clipped_fraction=float((1. - probability[subset]).mean()),
            Gaussian_expected_unit_clip_mean_sensitivity=float(probability[subset].mean()),
            leg_head_fraction_with_expected_clip_mean_sensitivity_below_1percent=float(
                (probability[subset] < .01).mean()),
            reconstructed_saturation_reward_per_control_step_mean=float(weighted_step[subset].mean()),
            reconstructed_saturation_reward_per_control_step_min=float(weighted_step[subset].min()),
            reconstructed_saturation_reward_clipped_fraction=float((weighted_rate[subset] <= -bound).mean()),
            heads={})
        for h, name in enumerate(HEAD_NAMES):
            x = mu[subset, :, h]
            s = sigma[subset, :, h]
            noise = raw_noise[subset, :, h]
            effective = effective_noise[subset, :, h]
            raw_rms = float(np.sqrt(np.mean(noise**2)))
            effective_rms = float(np.sqrt(np.mean(effective**2)))
            result['heads'][name] = dict(raw_mean_abs_max=float(np.abs(x).max()),
                raw_mean_outside_unit_fraction=float((np.abs(x) > 1.).mean()),
                sampled_raw_action_outside_unit_fraction=float((np.abs(raw[subset, :, h]) > 1.).mean()),
                Gaussian_expected_clipped_fraction=float((1.-probability[subset, :, h]).mean()),
                Gaussian_expected_unit_clip_mean_sensitivity=float(probability[subset, :, h].mean()),
                saved_policy_std_mean=float(s.mean()), raw_noise_RMS=raw_rms,
                effective_action_noise_RMS=effective_rms,
                effective_to_raw_noise_RMS_ratio=effective_rms/raw_rms if raw_rms else None,
                effective_joint_target_noise_RMS_rad=effective_rms*float(scale[h]))
        return result
    groups = {'all50': summary(0, 2400), 'first10': summary(0, 480),
              'last10': summary(1920, 2400)}
    result = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        training_output=str(training), checkpoint=proof['checkpoint'],
        checkpoint_sha256=proof['checkpoint_sha256'], actual_updates=proof['actual_ppo_updates'],
        completed_native_updates=50, actual_native_population=8192, cached_environments=32,
        cached_environment_ids=ids, recorded_environment_time_samples=76800,
        recorded_leg_head_samples=307200, head_order=list(HEAD_NAMES),
        source_model_std_17Adam_actualLR_preserved=True,
        native_action_storage_max_error=native_action_error,
        recorded_actual_joint_targets_vs_unit_action_map_max_error_rad=target_map_error,
        global_existing_saturation_coefficient=coefficient, existing_soft_limit=soft,
        groups=groups, source_pins=pins,
        limitations=['The streams are32 cached environments, not all8192. Counts and saturation rates must not be reported as full-population or frozen-policy acceptance rates.',
            'Gaussian sensitivity describes the scalar normalized clip operation only. It is not a measured joint-torque sensitivity, physical stop margin or closed-chain stability bound.',
            'Joint-target noise uses sampled action minus the clipped policy mean; its RMS includes the truncation bias. It is not an unbiased clipped-action standard deviation.',
            'The per-term saturation reward is reconstructed from cached raw actions and the pinned formula/configuration; this cache does not separately store the delivered component reward.',
            'First10 versus last10 also contain changing states and commands. Reduced clipping alone does not prove better tracking or a causal explanation. Original735, independent189 and sustained checks remain necessary.'],
        simulator_started=False, policy_evaluated=False, optimizer_executed=False,
        added_loss_or_backward=False, parameter_selected=False, policy_accepted=False)
    assert all(sha(path) == digest for path, digest in pins.items())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True, completed_native_updates=50,
        actual_native_population=8192, cached_environments=32, checkpoint_sha256=proof['checkpoint_sha256'],
        target_map_max_error_rad=target_map_error,
        raw_mean_outside_unit_first10=groups['first10']['raw_leg_mean_outside_unit_fraction'],
        raw_mean_outside_unit_last10=groups['last10']['raw_leg_mean_outside_unit_fraction'],
        expected_clip_fraction_first10=groups['first10']['Gaussian_expected_clipped_fraction'],
        expected_clip_fraction_last10=groups['last10']['Gaussian_expected_clipped_fraction'],
        parameter_selected=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--joint-map-source-reference', type=Path)
    args = parser.parse_args()
    run(args.training_output.resolve(), args.output.resolve(),
        args.joint_map_source_reference.resolve() if args.joint_map_source_reference else None)
