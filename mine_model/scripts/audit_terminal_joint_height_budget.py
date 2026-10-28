"""Decompose captured pre-reset height deficits with the original joint map/FK.

All calculations consume already captured unsafe states. FK lengths are
telemetry, not a new VMC/IK target or controller. No simulation or policy runs.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from joint_pd_continuation import ROOT, sha
from wheel_legged_gym_isaaclab import five_bar_vmc as geometry
from wheel_legged_gym_isaaclab import leg_control as mapping

SCRIPT = Path(__file__).resolve()


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'terminal_joint_height_budget_prefix_{through:04d}_readonly.json'
    if directory.parent != ROOT / 'mine_model/results' or destination.exists():
        raise ValueError('Preserve one owned immutable native diagnostic')
    plan_path = directory / 'experiment_plan.json'
    age_path = directory / f'terminal_event_ages_prefix_{through:04d}_readonly.json'
    geo_path = directory / f'joint_target_geometric_exploration_prefix_{through:04d}_readonly.json'
    plan, ages, geo = [json.loads(path.read_text()) for path in (plan_path, age_path, geo_path)]
    cfg = plan['training_config']
    if (ages['committed_prefix'] != through or geo['committed_updates'] != through
            or cfg['leg_control_mode'] != mapping.DIRECT_JOINT_POSITION_MODE
            or cfg['observation_space'] != 27 or plan['native_num_envs'] != 4096
            or not ages['raw_and_RSL_unsafe_counts_identical']
            or not ages['all_prefix_native4096_alignment_errors_zero']
            or not ages['native_RNG_states_unchanged']
            or ages['training_modified'] or geo['training_modified']):
        raise ValueError('Require the original qualified4096/plain27 pre-reset records')
    events = ages['events']
    if len(events) != ages['all4096_unsafe_events'] or not events:
        raise ValueError('Require complete nonempty unsafe-event accounting')
    fields = ('joint_position', 'joint_target', 'raw_action', 'leg_length', 'command', 'height')
    data = {field: np.asarray([e['state_before_reset'][field] for e in events], dtype=np.float32)
        for field in fields}
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError('Nonfinite terminal state needs separate diagnosis')
    count = len(events)
    if (data['joint_position'].shape != (count, 4) or data['joint_target'].shape != (count, 4)
            or data['raw_action'].shape != (count, 6) or data['leg_length'].shape != (count, 2)
            or data['command'].shape != (count, 3)):
        raise ValueError('Requalify the original motor/action/command order')
    rng = torch.random.get_rng_state().clone()
    raw = torch.from_numpy(data['raw_action'][:, mapping.LEG_ACTION_COLUMNS])
    boundaries = raw.new_tensor(geo['USD_composed_joint_limits_rad']).expand(count, -1, -1)
    projected, _ = mapping.joint_position_targets(raw, cfg['leg_joint_position_offset'],
        cfg['leg_joint_position_scale'], boundaries, cfg['leg_joint_target_limit_margin'])
    targets = torch.from_numpy(data['joint_target'])
    actual = torch.from_numpy(data['joint_position'])
    mapping_error = float((projected - targets).abs().max())
    target_length, _, _, target_valid = geometry.five_bar_state(
        targets[:, ::2], targets[:, 1::2], **cfg['five_bar_geometry'])
    measured_length, _, _, measured_valid = geometry.five_bar_state(
        actual[:, ::2], actual[:, 1::2], **cfg['five_bar_geometry'])
    measured_error = float((measured_length - torch.from_numpy(data['leg_length'])).abs().max())
    if (mapping_error > 1.e-7 or measured_error > 2.e-7
            or not target_valid.all() or not measured_valid.all()):
        raise ValueError('Original motor map/FK does not reproduce captured unsafe states')
    reference = target_length.numpy().astype(np.float64).mean(-1)
    measured = data['leg_length'].astype(np.float64).mean(-1)
    commanded = data['command'][:, 2].astype(np.float64)
    height = data['height'].astype(np.float64)
    offset = cfg['rewards']['leg_length_target_height_offset']
    components = np.stack((commanded + offset - reference,
        reference - measured, measured - offset - height), axis=-1)
    deficit = commanded - height
    closure = float(abs(components.sum(-1) - deficit).max())
    tracking = abs(data['joint_target'].astype(np.float64) - data['joint_position']).max(-1)
    if closure > 1.e-12:
        raise ValueError('Signed height-budget identity does not close')
    names = ('commanded_upright_length_proxy_minus_policy_joint_target_length',
        'target_length_minus_measured_length', 'measured_length_minus_upright_height_proxy')
    populations = dict(all_events=np.ones(count, bool),
        positive_height_deficit=deficit > 0.,
        height_deficit_above10pct=deficit > .1 * commanded,
        initial_physical_age_upper_bound_le2s=np.asarray([
            e['physical_age_upper_bound_s'] <= 2. for e in events]),
        other_physical_age_upper_bound=np.asarray([
            e['physical_age_upper_bound_s'] > 2. for e in events]))
    groups = {}
    for label, mask in populations.items():
        if not mask.any():
            continue
        values = components[mask]
        groups[label] = dict(events=int(mask.sum()),
            height_deficit_signed_mean_m=float(deficit[mask].mean()),
            height_deficit_abs_median_m=float(np.median(abs(deficit[mask]))),
            component_signed_mean_m=values.mean(0).tolist(),
            component_absolute_median_m=np.median(abs(values), axis=0).tolist(),
            largest_absolute_component_counts=np.bincount(
                np.argmax(abs(values), axis=-1), minlength=3).tolist(),
            maximum_active_joint_tracking_error_rad_median=float(np.median(tracking[mask])),
            maximum_active_joint_tracking_error_rad_P95=float(np.quantile(tracking[mask], .95)))
    rows = [dict(update=e['update'], environment_id=e['environment_id'], reasons=e['reasons'],
        command=e['command'], physical_age_upper_bound_s=e['physical_age_upper_bound_s'],
        command_age_s=e['command_age_s'], actual_height_m=float(height[i]),
        height_deficit_signed_m=float(deficit[i]),
        mean_policy_joint_target_length_m=float(reference[i]), mean_measured_length_m=float(measured[i]),
        signed_components_m=components[i].tolist(), maximum_joint_tracking_error_rad=float(tracking[i]))
        for i, e in enumerate(events)]
    paths = (SCRIPT, plan_path, age_path, geo_path, Path(mapping.__file__), Path(geometry.__file__))
    pins = {**ages['source_pins'], **geo['source_pins'], **{str(path): sha(path) for path in paths}}
    if not torch.equal(rng, torch.random.get_rng_state()) or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Immutable inputs or RNG changed during read-only calculations')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_prefix=through,
        events=count, component_order=list(names), offset_m=offset, groups=groups, per_event=rows,
        original_motor_target_map_max_error_rad=mapping_error,
        original_measured_FK_length_max_error_m=measured_error, signed_identity_max_error_m=closure,
        original_mapper_and_FK_reused_without_changes=True, source_pins=pins,
        simulator_started=False, policy_loaded=False, optimizer_executed=False,
        RNG_draws_added=0, RNG_state_unchanged=True, training_modified=False,
        acceptance_changed=False, policy_accepted=False,
        limitations=[
            'Unsafe evolving-policy Gaussian training states only; no survival probability, closed-loop improvement or independent acceptance follows.',
            'The commanded upright length is a height-reference proxy, not an extra leg-length command. Direct actions still specify joint-angle targets.',
            'The last component includes pose, leg-angle and reference-origin effects. It is not evidence of a geometry or height-sensor bug.',
            'A short policy target at an already unsafe state may be a response to a preceding disturbance. This terminal snapshot does not establish the initiating cause.',
            'Counts of the largest absolute component are not causal contribution percentages. The10pct subgroup is descriptive and does not replace any original acceptance window or terminal gate.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    path, result = audit(args.directory, args.through)
    print(json.dumps(dict(report=str(path), groups=result['groups'], training_modified=False)))
