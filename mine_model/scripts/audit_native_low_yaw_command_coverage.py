"""Read captured command support by height and motion group; optionally follow40/100/200.

No training hook, simulation, model load, RNG draw, optimizer step or gate change.
Time-exposure counts include repeated commands and are not independent samples.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
RESULTS = ROOT / 'mine_model/results'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'low_yaw_command_coverage_prefix_{through:04d}_readonly.json'
    if destination.exists():
        raise ValueError('Preserve the existing immutable command-coverage report')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = plan['training_config']['commands']
    scale = plan['training_motion_scale']
    if (not plan['ordinary_full_policy_PPO'] or plan['native_num_envs'] != 4096
            or plan['fixed_standing_environments'] != 1024 or scale != .625):
        raise ValueError('Qualify this full4096 current motion envelope before interpreting coverage')
    pins = {str(SCRIPT): sha(SCRIPT), str(plan_path): sha(plan_path)}
    ids, chunks, maximum_errors = None, [], {}
    for index in range(1, through + 1):
        data_path = directory / f'native_learning_signal/rollout_{index:04d}.npz'
        proof_path = data_path.with_name(f'audit_update_{index:04d}.json')
        proof = json.loads(proof_path.read_text())
        ids = proof['sampled_environment_ids'] if ids is None else ids
        errors = proof['all_environment_alignment_max_errors']
        if (proof['update'] != index or proof['sampled_environment_ids'] != ids
                or proof['total_environments'] != 4096 or len(errors) != 9
                or any(errors.values()) or not proof['probe_RNG_unchanged'] or proof['added_loss_or_backward']):
            raise ValueError('Require contiguous aligned original command records without RNG changes')
        with np.load(data_path, allow_pickle=False) as data:
            command = data['native_commands'].copy()
        if command.shape != (48, len(ids), 3) or not np.isfinite(command).all():
            raise ValueError('Invalid original command shape or finite values')
        chunks.append(command)
        maximum_errors.update(errors)
        pins.update({str(p): sha(p) for p in (data_path, proof_path)})
    command = np.concatenate(chunks)
    vx, yaw, height = [command[..., i] for i in range(3)]
    low, high = cfg['ranges_height']
    static = np.broadcast_to(np.asarray(ids)[None] < 1024, vx.shape)
    if np.any(command[..., :2][static]):
        raise ValueError('Dedicated standing environments received moving commands')
    ymax = scale * max(abs(v) for v in cfg['ranges_ang_vel_yaw'])
    ymin = scale * cfg['yaw_min_abs_rate']
    nonzero = abs(yaw) > 1.e-8
    if np.any(abs(yaw) > ymax + 1.e-6) or np.any(nonzero & (abs(yaw) < ymin - 1.e-6)):
        raise ValueError('Observed original commands violate the declared sampler bounds')
    populations = dict(all=np.ones(vx.shape, dtype=bool), dedicated_standing=static,
        motion_population=~static, forward=(~static & (vx > 1.e-8)),
        reverse=(~static & (vx < -1.e-8)), spin=(~static & (abs(vx) < 1.e-8) & nonzero))
    heights = dict(all=np.ones(vx.shape, dtype=bool),
        low=np.isclose(height, low, atol=1.e-6, rtol=0.),
        high=np.isclose(height, high, atol=1.e-6, rtol=0.))
    heights['interior'] = ~(heights['low'] | heights['high'])
    groups = {}
    for population, population_mask in populations.items():
        groups[population] = {}
        for label, height_mask in heights.items():
            mask = population_mask & height_mask
            values = yaw[mask & nonzero]
            groups[population][label] = dict(time_exposures=int(mask.sum()),
                nonzero_yaw_time_exposures=len(values),
                abs_nonzero_yaw_min=None if not len(values) else float(abs(values).min()),
                abs_nonzero_yaw_max=None if not len(values) else float(abs(values).max()),
                low_yaw_positive_exposures=int(np.sum((values > 0.) & (values < .075))),
                low_yaw_negative_exposures=int(np.sum((values < 0.) & (values > -.075))),
                abs_yaw_near_retained_0p075_within_0p01=int(np.sum(abs(abs(values) - .075) <= .01)),
                time_exposures_below_old_0p09375_minimum=int(np.sum(abs(values) < .09375)))
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Input evidence changed while reading it')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=through,
        native_environments=4096, sampled_environment_ids=ids,
        source_actual_updates=plan['source_actual_ppo_updates'],
        yaw_minimum_config=cfg['yaw_min_abs_rate'], declared_abs_nonzero_yaw_support_closure=[ymin, ymax],
        groups=groups, all4096_alignment_max_errors=maximum_errors,
        dedicated_standing_commands_exact_zero=True, source_pins=pins,
        RNG_draws_added=0, simulator_started=False, optimizer_loaded=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False, learned_improvement_proven=False,
        limitations=['Only32 captured environments; every4096-row command alignment was separately checked.',
            'Counts are time exposures, including held commands and ramped linear speeds; they are not independent draws.',
            'These are applied commands, not closed-loop tracking or fixed-policy acceptance measurements.',
            'Coverage does not prove escape from a local optimum or establish that it caused a tracking deficit.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def follow():
    while True:
        try:
            pointer = json.loads((RESULTS / 'joint_pd_active_joint_tracking.json').read_text())
            output = Path(pointer['directory']).resolve()
            plan = json.loads((output / 'experiment_plan.json').read_text())
            if output.parent != RESULTS:
                raise ValueError('Unexpected active training output')
            completed = []
            if plan['training_motion_scale'] == .625:
                count = len(list((output / 'native_learning_signal').glob('audit_update_*.json')))
                finished = (output / 'native_learning_signal/completed_external.json').exists()
                for prefix in (40, 100, 200):
                    path = output / f'low_yaw_command_coverage_prefix_{prefix:04d}_readonly.json'
                    if path.exists():
                        if json.loads(path.read_text())['committed_updates'] != prefix:
                            raise ValueError('Existing report has another committed clock')
                        completed.append(prefix)
                    elif count > prefix or finished and count == prefix:
                        audit(output, prefix)
                        completed.append(prefix)
            status = dict(observed_at=datetime.now().astimezone().isoformat(), directory=str(output),
                unit=pointer['unit'], completed_prefixes=completed, script_sha256=sha(SCRIPT),
                state='observing', simulator_started=False, training_modified=False, policy_accepted=False)
            (RESULTS / 'joint_pd_low_yaw_coverage_monitor_status_20261019.json').write_text(json.dumps(status, indent=2))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(15.)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path, nargs='?')
    cli.add_argument('--through', type=int)
    cli.add_argument('--follow', action='store_true')
    args = cli.parse_args()
    if args.follow:
        follow()
    else:
        result = audit(args.directory, args.through)
        print(json.dumps(dict(committed_updates=result['committed_updates'],
            support=result['declared_abs_nonzero_yaw_support_closure'],
            high_spin=result['groups']['spin']['high'], training_modified=False)))
