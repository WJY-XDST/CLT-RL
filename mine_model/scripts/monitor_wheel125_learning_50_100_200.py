"""Read-only50/100/200 review, using the saved wheel execution envelope.

This monitor cannot restart or extend training, reset a state, or accept a
policy. It preserves the existing health, critic, geometry and event readers.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import monitor_joint_pd_learning_health as health_template
import monitor_joint_pd_critic_and_command_coverage as critic_template
import monitor_joint_pd_geometric_exploration as geometry_template
import monitor_joint_pd_optimizer_exploration_context_v2 as extra_template
import monitor_joint_pd_event_history as events
import audit_native_terminal_event_ages as ages

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
STATUS = RESULTS / 'joint_pd_wheel125_learning_monitor_50_100_200_status_20261020.json'


def reader(module, name):
    current = adapter.private_module(module.__file__, name)
    proof = adapter.retarget(current, ('tick',), integers={'*': {40: 50}})
    return current, proof


health, health_proof = reader(health_template, '_wheel125_health_50_100_200')
health.CHECKS = (
    ('audit_effective_joint_pd_exploration_saved_limits.py', 'effective_exploration', []),
    ('audit_native_motion_training_profile.py', 'native_motion_training', []),
    ('audit_joint_pd_policy_updates.py', 'policy_update_health', ['--interval', '1', '--device', 'cuda:0']),
)
critic, critic_proof = reader(critic_template, '_wheel125_critic_50_100_200')
geometry, geometry_proof = reader(geometry_template, '_wheel125_geometry_50_100_200')
extra = adapter.private_module(extra_template.__file__, '_wheel125_extra_50_100_200')
extra.PREFIXES = (50, 100, 200)
extra.AUDITORS = tuple('audit_clipped_joint_pd_policy_kl_saved_limits.py'
    if name == 'audit_clipped_joint_pd_policy_kl.py' else name for name in extra.AUDITORS)
extra.STATUS = RESULTS / 'joint_pd_wheel125_optimizer_exploration_context_status_20261020.json'
PROOFS = dict(health=health_proof, critic=critic_proof, geometry=geometry_proof)


def tick():
    pointer = json.loads((RESULTS / 'joint_pd_active_joint_tracking.json').read_text())
    output = Path(pointer['directory']).resolve()
    plan = json.loads((output / 'experiment_plan.json').read_text())
    if plan['training_config'].get('wheel_action_limit', 1.) != 1.25:
        return dict(state='waiting_for_owned_wheel125_trial', training_modified=False)
    if (output.parent != RESULTS or plan['bounded_training_updates'] != 200
            or plan['early_learning_inspection_actual_updates'] != [50, 100]
            or plan['only_controller_change'] != {'wheel_action_limit': [1., 1.25]}):
        raise ValueError('Require the explicitly reviewed200 block with50/100 early inspections')
    health.tick()
    critic.tick()
    geometry.tick()
    extra.tick()
    events.save(output)
    count = len(list((output / 'native_learning_signal').glob('audit_update_*.json')))
    finished = (output / 'native_learning_signal/completed_external.json').is_file()
    done = []
    for prefix in (50, 100, 200):
        if not (count > prefix or finished and count == prefix):
            continue
        age = output / f'terminal_event_ages_prefix_{prefix:04d}_readonly.json'
        if not age.exists():
            ages.audit(output, prefix)
        required = ('learning_health_combined', 'policy_update_health', 'critic_and_command_coverage',
            'joint_target_geometric_exploration', 'gradient_allocation', 'critic_fit_by_resampling_phase',
            'matched_clipping_atoms', 'clipped_action_policy_KL', 'terminal_event_ages')
        if all((output / f'{stem}_prefix_{prefix:04d}_readonly.json').is_file() for stem in required):
            done.append(prefix)
    return dict(observed_at=datetime.now().astimezone().isoformat(), state='observing',
        directory=str(output), unit=pointer['unit'], completed_prefixes=done,
        committed_native_updates=count, saved_wheel_action_limit=1.25,
        early_inspection_updates=[50, 100], full_review_updates=200,
        metadata_only_interval_adapters=PROOFS, monitor_sha256=sha(SCRIPT),
        actual_environment_action_contract_checked_separately=True,
        training_modified=False, optimizer_executed=False, simulator_started=False,
        can_extend_training=False, can_accept_policy=False, policy_accepted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            result = tick()
        except (FileNotFoundError, json.JSONDecodeError):
            result = dict(observed_at=datetime.now().astimezone().isoformat(), state='waiting_for_committed_data',
                training_modified=False, policy_accepted=False)
        except BaseException as error:
            result = dict(observed_at=datetime.now().astimezone().isoformat(), state='agent_inspection_required',
                error=repr(error), training_modified=False, can_extend_training=False, policy_accepted=False)
        temp = STATUS.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temp.replace(STATUS)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.once:
            break
        time.sleep(15.)
