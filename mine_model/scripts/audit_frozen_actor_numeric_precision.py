"""Measure fixed-actor precision/shape effects without physics or PPO updates.

Use all recorded states from selected native updates plus three retained
steady frames of every original735 trial. CUDA batch sizes are neural-network
rows, not simulation environments. Backend flags affect this process only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_height_response import actor


SCRIPT = Path(__file__).resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stats(value):
    value = np.asarray(value, dtype=np.float64)
    assert value.size and np.isfinite(value).all()
    return dict(mean=float(value.mean()), p95=float(np.quantile(value, .95)),
                maximum=float(value.max()))


def difference(before, after, cfg):
    delta = after.astype(np.float64) - before.astype(np.float64)
    bound = cfg['wheel_action_limit']
    wheels = ((after[..., [2, 5]].clip(-bound, bound)
               - before[..., [2, 5]].clip(-bound, bound)) * cfg['action_scale_vel'])
    speed = cfg['wheel_radius'] * wheels.mean(-1)
    yaw = cfg['wheel_radius'] / cfg['wheel_track_width'] * (wheels[..., 1] - wheels[..., 0])
    return dict(raw_mean_absolute_difference=stats(abs(delta)),
        raw_max_absolute_difference_by_head=abs(delta).reshape(-1, 6).max(0).tolist(),
        wheel_speed_reference_difference_rad_s=stats(abs(wheels)),
        rolling_speed_proxy_difference_m_s=stats(abs(speed)),
        rolling_yaw_proxy_difference_rad_s=stats(abs(yaw)))


def ratio_error(old_mean, recomputed, sigma, epsilon):
    sigma = sigma.astype(np.float64)
    offset = (recomputed.astype(np.float64) - old_mean.astype(np.float64)) / sigma
    log_ratio = .5 * (epsilon ** 2 - (epsilon - offset) ** 2).sum(-1)
    ratio = np.exp(log_ratio)
    return dict(zero_parameter_update_KL=stats(.5 * (offset ** 2).sum(-1)),
                absolute_ratio_minus_one=stats(abs(ratio - 1.)),
                ratio_outside_original_0p8_1p2_fraction=float(((ratio < .8) | (ratio > 1.2)).mean()))


def audit(plan_path, output):
    plan = read(plan_path)
    assert not output.exists()
    assert torch.cuda.is_available(), 'GPU parity requires the same CUDA device; do not infer it on CPU'
    free, total = torch.cuda.mem_get_info()
    assert free >= 2 * 1024 ** 3, 'Defer this small read-only audit until at least2GiB CUDA memory is free'
    pins = {str(p): sha(p) for p in (SCRIPT, plan_path)}
    scope = Path(plan['fixed_scope'])
    scope_plan = read(scope / 'qualification.json')
    scope_complete = read(scope / 'complete_frozen_command_matrix.json')
    assert scope_complete['complete'] and scope_plan['trial_count'] == 735
    pins.update({str(p): sha(p) for p in (scope / 'qualification.json',
                                       scope / 'complete_frozen_command_matrix.json')})
    fixed_inputs, recorded_source_means, fixed_cases = [], [], []
    for job in scope_plan['batches']:
        directory = Path(job['replay'])
        path = directory / 'policy_inputs.npz'
        with np.load(path, allow_pickle=False) as values:
            indices = [850, 925, 999]
            assert values['observations'].shape[0] == 1000
            fixed_inputs.append(values['observations'][indices].reshape(-1, 27).copy())
            recorded_source_means.append(values['means'][indices].reshape(-1, 6).copy())
        fixed_cases.extend((c['name'], seed) for c in job['cases'] for seed in job['seeds'])
        pins[str(path)] = sha(path)
    assert len(fixed_cases) == len(set(fixed_cases)) == 735
    fixed_inputs = np.concatenate(fixed_inputs)
    recorded_source_means = np.concatenate(recorded_source_means)
    assert fixed_inputs.shape == (735 * 3, 27)
    before_flag = torch.backends.cuda.matmul.allow_tf32
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    rows = []
    try:
        with torch.inference_mode():
            for view in plan['native_views']:
                owner = Path(view['output'])
                spec = read(owner / 'experiment_plan.json')
                index = view['update']
                record = read(owner / f'native_learning_signal/audit_update_{index:04d}.json')
                ids = record['sampled_environment_ids']
                assert spec['native_num_envs'] == 8192 and len(ids) == 32
                runs = list((SCRIPT.parents[2] / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat').glob('*_' + spec['run_name']))
                assert len(runs) == 1
                checkpoint = (Path(spec['source_checkpoint']) if index == 1
                    else runs[0] / f"model_{spec['source_actual_ppo_updates'] + index - 2}.pt")
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                states = saved['optimizer_state_dict']['state']
                expected = spec['source_actual_ppo_updates'] + index - 1
                assert len(states) == 17 and {int(s['step']) for s in states.values()} == {expected * 20}
                weights = saved['model_state_dict']
                # The verified production plan already contains the normalized
                # configuration. Native YAML includes Python tuple tags; this
                # numerical reader needs only ordinary JSON scalar settings.
                cfg = spec['training_config']
                data_path = owner / f'native_learning_signal/rollout_{index:04d}.npz'
                with np.load(data_path, allow_pickle=False) as arrays:
                    data = {key: arrays[key].copy() for key in ('observations', 'mu', 'sigma', 'actions')}
                assert data['observations'].shape == (48, 32, 27)
                assert (data['sigma'] > 0).all()
                assert np.array_equal(data['sigma'],
                    np.broadcast_to(weights['std'].numpy(), data['sigma'].shape))
                gpu_weights = {key: value.to('cuda:0') for key, value in weights.items()}

                def infer(inputs, batch_size, enabled, sample_ids=None):
                    torch.backends.cuda.matmul.allow_tf32 = enabled
                    indices = (torch.arange(len(inputs), device='cuda:0') if sample_ids is None
                               else torch.tensor(sample_ids, device='cuda:0'))
                    matrix = torch.zeros((batch_size, 27), device='cuda:0')
                    matrix[indices] = torch.from_numpy(inputs).to('cuda:0')
                    return actor(matrix, gpu_weights)[indices].cpu().numpy()

                by_mode = {}
                for enabled in (True, False):
                    shapes = {}
                    for size in (8192, 98304):
                        shapes[size] = np.stack([infer(frame, size, enabled, ids)
                                                 for frame in data['observations']])
                    by_mode[enabled] = shapes
                parity = float(abs(by_mode[True][8192] - data['mu']).max())
                assert parity <= 1.e-6, ('Native TF32+batch8192 reproduction failed', view['name'], parity)
                epsilon = (data['actions'].astype(np.float64) - data['mu']) / data['sigma']
                numerical = {}
                for enabled in (True, False):
                    numerical[str(enabled)] = ratio_error(by_mode[enabled][8192],
                        by_mode[enabled][98304], data['sigma'], epsilon)
                fixed = {}
                for enabled in (True, False):
                    fixed[enabled] = {size: infer(fixed_inputs, size, enabled) for size in (8192, 98304)}
                cpu_fixed = actor(torch.from_numpy(fixed_inputs), weights).numpy()
                row = dict(name=view['name'], source_actual_updates=expected,
                    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    saved_std=weights['std'].tolist(), native_TF32_8192_parity_max_error=parity,
                    native_states=48 * 32, original735_fixed_states=len(fixed_inputs),
                    zero_weight_update_density_error_by_allow_tf32=numerical,
                    native_TF32_vs_FP32=difference(by_mode[True][8192], by_mode[False][8192], cfg),
                    original735_fixed_TF32_vs_FP32=difference(fixed[True][8192], fixed[False][8192], cfg),
                    original735_fixed_CPU_vs_FP32=difference(cpu_fixed, fixed[False][8192], cfg),
                    original735_fixed_TF32_shape8192_vs98304=difference(fixed[True][8192], fixed[True][98304], cfg),
                    original735_fixed_FP32_shape8192_vs98304=difference(fixed[False][8192], fixed[False][98304], cfg))
                if sha(checkpoint) == scope_complete['checkpoint_sha256']:
                    row['original735_CPU_source_recorded_mean_max_error'] = float(abs(cpu_fixed - recorded_source_means).max())
                    assert row['original735_CPU_source_recorded_mean_max_error'] < 1.e-4
                rows.append(row)
                for path in (checkpoint, data_path, runs[0] / 'params/env.yaml', owner / 'experiment_plan.json'):
                    pins[str(path)] = sha(path)
                del gpu_weights
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before_flag
    assert all(sha(path) == digest for path, digest in pins.items())
    report = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        GPU=torch.cuda.get_device_properties(0).name, torch_version=torch.__version__,
        initial_CUDA_free_GiB=free / 1024 ** 3, total_CUDA_GiB=total / 1024 ** 3,
        maximum_neural_network_rows=98304, simulation_environments_created=0,
        original735_trials_retained=735, fixed_frames_per_trial=3, results=rows,
        all_rng_unchanged=True, checkpoint_and_optimizer_files_unchanged=True,
        backend_flag_restored_in_this_process=True, other_processes_modified=False,
        simulator_started=False, optimizer_executed=False, backward_executed=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False,
        limitations=['All comparisons use frozen weights and cached observations. Rolling proxies are not measured body-motion errors.',
                     'Numerical differences alone cannot prove the cause of tracking error or qualify a policy.'],
        source_pins=pins)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, GPU=report['GPU'], views=len(rows),
        results=[dict(name=r['name'], native_TF32_parity=r['native_TF32_8192_parity_max_error'],
            zero_update_KL_TF32=r['zero_weight_update_density_error_by_allow_tf32']['True']['zero_parameter_update_KL'],
            zero_update_KL_FP32=r['zero_weight_update_density_error_by_allow_tf32']['False']['zero_parameter_update_KL'],
            fixed_yaw_proxy_difference=r['original735_fixed_TF32_vs_FP32']['rolling_yaw_proxy_difference_rad_s']) for r in rows]), ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.plan.resolve(), args.output.resolve())
