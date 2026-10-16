"""Single existing raw-wheel reward ablation after a reviewed saved handoff.

The execution limit, applied actions, inference and every command group stay
unchanged. A raw output outside the limit receives distinguishable rewards,
instead of relying solely on rewards from identical clipped references.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
import continue_joint_pd_priority as priority

ROOT = priority.ROOT
KEY = 'wheel_action_saturation'
COEFFICIENT = -2.
HANDOFF_REASON = 'agent_selected_wheel_saturation_reward_handoff'


def revised_config(saved, coefficient=COEFFICIENT):
    import copy
    if (coefficient not in (0., COEFFICIENT)
            or saved['leg_control_mode'] != 'implicit_joint_position'
            or saved['rewards']['orientation'] != -400.
            or saved['rewards']['base_height_error_sq'] != -500.
            or saved['rewards'][KEY] not in (0., COEFFICIENT)
            or not saved['commands']['fixed_standing_groups']
            or saved['commands']['standing_env_fraction'] != .25):
        raise ValueError('Require the diagnosed direct-PD full-policy source')
    expected = copy.deepcopy(saved)
    expected['rewards'][KEY] = coefficient
    return expected


def prepare(previous, output, coefficient, updates):
    if coefficient not in (0., COEFFICIENT) or (Path(previous)/'STOP').exists():
        raise ValueError('Use only the qualified raw-wheel cost and respect owner stops')
    interval = priority.core.INTERVAL
    try:
        priority.core.INTERVAL = updates
        plan, saved = priority.core.prepare(previous, output, .001)
    finally:
        priority.core.INTERVAL = interval
    expected = revised_config(saved, coefficient)
    old = saved['rewards'][KEY]
    plan.update(reward_key=KEY, reward_value=coefficient,
        only_reward_change={KEY: [old, coefficient]} if old != coefficient else {},
        only_training_change='Enable existing raw-wheel cost only after confirming deep clipping; zero inside valid range',
        importance=['standing_posture_and_survival', 'height_tracking', 'motion_tracking'],
        no_training_phases=True, command_dictionary_unchanged=True, dedicated_standing_fraction=.25,
        target_FK_height_cost_retained=saved['rewards']['joint_target_leg_length_error_sq'],
        effective_leg_entropy_coefficient=.001, effective_wheel_entropy_coefficient=.001,
        physical_reference_limit_changed=False, additional_reward_formula=False,
        hypothesis='Raw wheel means far beyond the clamp and small std reduce applied-reference diversity. '
                   'A distinguishable raw-output cost may help; it does not prove the unique fall cause or success.')
    for path in (Path(__file__).resolve(), ROOT/'wheel_legged_isaaclab/tests/test_wheel_saturation_continuation.py',
                 priority.split.ENTRY, priority.split.POLICY):
        plan['source_files'][str(path)] = priority.core.sha(path)
    profile = Path(previous)/'native_entropy_profile.json'
    if profile.exists(): plan['source_files'][str(profile)] = priority.core.sha(profile)
    for name in ('manifest.json', 'trace.csv', 'policy_inputs.npz'):
        path = Path(previous)/'replay'/name
        if path.exists(): plan['source_files'][str(path)] = priority.core.sha(path)
    write_json(Path(output)/'experiment_plan.json', plan)
    return plan, expected


def diagnose_source(previous):
    """Frozen recorded means and learned std, not an intervention or rollout."""
    import numpy as np
    import torch
    previous = Path(previous)
    state = json.loads((previous/'status.json').read_text())
    manifest = json.loads((previous/'replay/manifest.json').read_text())
    if manifest['checkpoint_sha256'] != state['checkpoint_sha256']:
        raise ValueError('Raw-output diagnosis must cover the exact latest checkpoint')
    saved = torch.load(state['checkpoint'], map_location='cpu', weights_only=True)
    std = saved['model_state_dict']['std'].cpu().numpy()[[2,5]]
    with np.load(previous/'replay/policy_inputs.npz', allow_pickle=False) as data:
        raw = data['means'][:,:,[2,5]].copy()
    trace = np.genfromtxt(previous/'replay/trace.csv', delimiter=',', names=True)
    n = len(manifest['cases'])*len(manifest['seeds'])
    if (raw.ndim != 3 or raw.shape[1:] != (n,2) or len(trace) != raw.shape[0]*n
            or not np.isfinite(raw).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError('Require complete finite raw-output and exploration telemetry')
    episodes = trace['episode'].reshape(raw.shape[0],n)
    trials = []
    for cid, case in enumerate(manifest['cases']):
        if case['name'] not in ('reverse_0p3','reverse_0p6'): continue
        for si, seed in enumerate(manifest['seeds']):
            eid = cid*len(manifest['seeds'])+si
            first = episodes[:,eid] == 0
            deep = (np.abs(raw[:,eid])-1.) > 3.*std[None,:]
            trials.append(dict(case=case['name'], seed=seed,
                first_episode_deeply_clipped_wheel_coordinates=int(deep[first].sum()),
                max_first_episode_raw_wheel_mean_abs=float(np.abs(raw[first,eid]).max())))
    if len(trials) != 6:
        raise ValueError('Require both original reverse cases and three seeds')
    return dict(checkpoint=state['checkpoint'], checkpoint_sha256=state['checkpoint_sha256'],
        wheel_std=std.tolist(), trials=trials,
        requires_raw_cost=any(t['first_episode_deeply_clipped_wheel_coordinates'] >= 10 for t in trials),
        diagnostic_only_not_acceptance=True,
        interpretation='Repeated raw means more than 3 learned standard deviations outside the execution range. '
            'The ten-coordinate trigger selects a reward experiment, never changes acceptance or the execution limit.')


def reviewed_handoff(previous, coordinator):
    previous, coordinator = Path(previous).resolve(), Path(coordinator).resolve()
    marker = json.loads((coordinator/'STOP').read_text())
    cursor = json.loads((coordinator/'coordinator.json').read_text())
    state = json.loads((previous/'status.json').read_text())
    if (marker.get('reason') != HANDOFF_REASON or marker.get('not_an_owner_pause') is not True
            or cursor.get('state') != 'explicit_stop_respected'
            or Path(cursor['previous_output']).resolve() != previous
            or state.get('training_active') or state['state'] != 'agent_review_required'
            or not (previous/'replay/completed.json').exists()):
        raise ValueError('Require a completed canonical review and the exact owned handoff; do not resume an owner pause')
    report = json.loads((previous/'replay/assessment.json').read_text())
    if report['checkpoint_sha256'] != priority.core.sha(Path(state['checkpoint'])):
        raise ValueError('Handoff review does not match the exact latest checkpoint')
    priority.priority_summary(report)
    return state['checkpoint']


def run(previous, base, unit, predecessor):
    source = reviewed_handoff(previous, predecessor)
    base = Path(base).resolve()
    base.mkdir(exist_ok=True)
    diagnosis = diagnose_source(previous)
    write_json(base/'source_raw_wheel_diagnosis.json', diagnosis)
    saved_cfg = priority.core.validate_checkpoint_robot(Path(source))
    coefficient = (COEFFICIENT if diagnosis['requires_raw_cost'] or saved_cfg['rewards'][KEY] == COEFFICIENT else 0.)
    write_json(base/'owned_handoff.json', dict(previous_output=str(previous), checkpoint=source,
        predecessor=str(predecessor), reason=HANDOFF_REASON, created_at=datetime.now().astimezone().isoformat(),
        weights_reset=False, optimizer_reset=False, policy_accepted=False))
    original = priority.prepare, priority.REWARD_KEY, priority.next_shaping_cost
    try:
        priority.prepare = prepare
        priority.REWARD_KEY = KEY
        priority.next_shaping_cost = lambda current, comparisons: coefficient
        priority.run(previous, base, unit)
    finally:
        priority.prepare, priority.REWARD_KEY, priority.next_shaping_cost = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    run(args.previous, args.coordinator, args.unit, args.predecessor)
