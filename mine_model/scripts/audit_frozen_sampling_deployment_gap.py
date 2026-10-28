"""Compare frozen deterministic and matched-noise full51 physical records.

No simulation, fitting, optimizer update or acceptance decision is performed.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(value):
    return dict(signed_mean=float(value.mean()), MAE=float(abs(value).mean()), std=float(value.std()))


def audit(deterministic, saved, probe, destination):
    directories = dict(deterministic=Path(deterministic).resolve(), saved_sampling=Path(saved).resolve(),
        probe_sampling=Path(probe).resolve())
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve the existing immutable diagnostic')
    captures, pins, initial, epsilons = {}, {str(Path(__file__).resolve()): sha(__file__)}, {}, {}
    for label, directory in directories.items():
        names = ['manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz']
        if label != 'deterministic':
            names += ['sampling_noise.npz', 'sampling_probe_integrity.json', 'sampled_tracking_diagnosis.json']
        pins.update({str(directory / name):sha(directory / name) for name in names})
        manifest, complete, cfg = [json.loads((directory / name).read_text()) for name in names[:3]]
        if (len(manifest['cases']) != 17 or manifest['seeds'] != [483, 484, 485]
                or manifest['seconds_per_case'] != 20. or not complete['completed']
                or complete['samples'] != 51000 or manifest['weights_updated']
                or manifest['policy_input_remapping'] or manifest['command_scale'] != 1.
                or manifest['sampled_policy_actions'] != (label != 'deterministic')
                or manifest['action_bias_probes'] or manifest['wheel_probe']
                or manifest['wheel_reference_limit_probe'] is not None):
            raise ValueError('Use complete unmodified full51 recordings with all failures visible')
        data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
        with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as arrays:
            means = arrays['means'].copy()
            initial[label] = (arrays['observations'][0].copy(), means[0].copy())
        if len(data) != 51000 or means.shape != (1000, 51, 6) or not np.isfinite(means).all():
            raise ValueError('Incomplete original traces or policy means')
        raw = means.copy()
        if label != 'deterministic':
            integrity = json.loads((directory / 'sampling_probe_integrity.json').read_text())
            with np.load(directory / 'sampling_noise.npz', allow_pickle=False) as arrays:
                epsilon, std = arrays['epsilon'].copy(), arrays['sampling_std'].copy()
            if (epsilon.shape != (1000, 3, 6) or std.shape != (6,)
                    or not integrity['all_model_tensors_bitwise_unchanged']
                    or integrity['additional_RNG_draws'] or integrity['gaussian_draw_count'] != 1000
                    or manifest['policy_noise_seed'] != 43):
                raise ValueError('Saved state or matched Gaussian draws changed')
            epsilons[label] = epsilon
            raw += np.tile(epsilon, (1, 17, 1)) * std
        expected_wheel = np.clip(raw[:, :, (2, 5)], -1., 1.) * cfg['action_scale_vel']
        wheel_parity, rows = 0., []
        for cid, case in enumerate(manifest['cases']):
            for sid, seed in enumerate(manifest['seeds']):
                index = 3 * cid + sid
                part = data[(data['case_id'] == cid) & (data['seed'] == seed)]
                if len(part) != 1000 or not np.allclose(part['time_s'], np.arange(1,1001)*.02, rtol=0.,atol=1.e-8):
                    raise ValueError('Complete original physical timeline required')
                ref = np.stack([part['left_wheel_ref_rad_s'], part['right_wheel_ref_rad_s']], -1)
                wheel_parity = max(wheel_parity, float(abs(expected_wheel[:, index] - ref).max()))
                survived = bool(not np.any(part['episode']) and not np.any(part['terminated']) and not np.any(part['timeout']))
                for window, start in [('original_final3s',17.), ('diagnostic_final10s',10.)]:
                    tail = part[part['time_s'] >= start]
                    channels, closure = None, None
                    if survived:
                        reference = np.stack([tail['left_wheel_ref_rad_s'], tail['right_wheel_ref_rad_s']], -1)
                        actual = np.stack([tail['left_wheel_rad_s'], tail['right_wheel_rad_s']], -1)
                        channels, closure = {}, 0.
                        radius, track = cfg['wheel_radius'], cfg['wheel_track_width']
                        for kind, body, command, ref_proxy, measured_proxy in (
                            ('yaw',tail['yaw_rate_rad_s'],tail['yaw_cmd_rad_s'],
                             radius*(reference[:,1]-reference[:,0])/track,radius*(actual[:,1]-actual[:,0])/track),
                            ('speed',tail['speed_m_s'],tail['speed_cmd_m_s'],
                             radius*reference.mean(-1),radius*actual.mean(-1))):
                            error = body-command
                            pieces = dict(reference_proxy_minus_command=ref_proxy-command,
                                measured_wheel_proxy_minus_reference_proxy=measured_proxy-ref_proxy,
                                body_minus_measured_wheel_proxy=body-measured_proxy)
                            closure = max(closure,float(abs(sum(pieces.values())-error).max()))
                            channels[kind] = dict(body_minus_command=stats(error),
                                signed_components={key:stats(value) for key,value in pieces.items()})
                        if closure > 1.e-12:
                            raise ValueError('Physical signed decomposition does not close')
                    failures = part[(part['episode'] > 0) | (part['terminated'] > 0)]
                    rows.append(dict(case=case,seed=seed,window=window,survived=survived,
                        first_failure_s=float(failures['time_s'][0]) if len(failures) else None,
                        channels=channels,signed_closure_error=closure,
                        clipping_reduced_noise_RMS_by_head=np.sqrt(np.mean((
                            np.clip(raw[:,index],-1.,1.)-np.clip(means[:,index],-1.,1.))**2,axis=0)).tolist()))
        if wheel_parity > 1.e-5:
            raise ValueError('Recorded noise and means do not reconstruct the physical wheel targets')
        captures[label] = dict(manifest=manifest,config=cfg,trials=rows,resets=complete['resets'],
            wheel_reference_reconstruction_max_error_rad_s=wheel_parity)
    first = captures['deterministic']
    for value in captures.values():
        m, cfg = value['manifest'], dict(value['config'])
        cfg['log_dir'] = first['config']['log_dir']
        if (cfg != first['config'] or m['cases'] != first['manifest']['cases']
                or m['checkpoint_sha256'] != first['manifest']['checkpoint_sha256']
                or m['tracking_goal_feedback_criteria'] != first['manifest']['tracking_goal_feedback_criteria']):
            raise ValueError('Frozen model, physical configuration, commands or criteria differ')
    if (not np.array_equal(epsilons['saved_sampling'],epsilons['probe_sampling'])
            or not all(all(np.array_equal(x,y) for x,y in zip(initial['deterministic'],value))
                       for value in initial.values())
            or any(sha(path) != digest for path,digest in pins.items())):
        raise ValueError('Noise pairing, original initial inference or immutable inputs differ')
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=first['manifest']['checkpoint_sha256'], captures=captures, source_pins=pins,
        all17_cases_and_all3_seeds_retained=True, original_acceptance_unchanged=True,
        bitwise_identical_initial_observations_and_means=True, exact_same_Gaussian_draws=True,
        optimizer_executed=False, simulator_started=False, training_modified=False, policy_accepted=False,
        limitations=['One common action-noise seed43 across cases and three reset seeds; time samples are correlated.',
            'The final10s window is diagnostic only. Original final3s acceptance and every failure remain unchanged.',
            'Proxy components add signed errors exactly; they do not isolate friction, slip or a wheel servo bug.',
            'A frozen sampling change is neither learned improvement nor proof of a local optimum.'])
    with destination.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deterministic',type=Path,required=True)
    parser.add_argument('--saved',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.deterministic,args.saved,args.probe,args.output)
    print(json.dumps(dict(report=str(args.output),reconstruction_errors={key:value[
        'wheel_reference_reconstruction_max_error_rad_s'] for key,value in result['captures'].items()},training_modified=False)))
