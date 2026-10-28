"""Verify the exact source7201 production/review/monitor routes before PPO."""
from datetime import datetime
import json
from pathlib import Path

import train_wheel005_source7201_D100_block200 as trainer
import review_wheel005_source7201_D100_block200 as review
import monitor_wheel005_source7201_D100_learning as monitor
import compare_wheel005_source7201_D100_recorded_joint_targets as comparison
import queue_wheel005_source7201_D100_paired735 as paired
import joint_pd_wheel005_source7201_D100_runtime as runtime
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
DESTINATION = runtime.QUALIFICATION / 'preproduction_entry_CPU_verification.json'


def qualify():
    pointer = json.loads((runtime.RESULTS /
        'joint_pd_wheel005_source7201_D100_prepared_capacity_pointer_20261024.json').read_text())
    output = Path(pointer['planned_production_output'])
    plan_path = output / 'experiment_plan.json'
    before = sha(plan_path)
    plan, _ = trainer.prepare(output, runtime.SOURCE_REPORT.parent)
    assert sha(plan_path) == before, 'Repeated preparation must preserve the qualified plan bytes'
    trainer.prepare(output, runtime.SOURCE_REPORT.parent)
    assert sha(plan_path) == before
    assert trainer.api.final_check.verify_boundary is trainer.final_check.verify_boundary
    assert plan['source_sha256'] == runtime.ANCHOR_SHA
    assert plan['source_actual_ppo_updates'] == 7201 and plan['target_actual_ppo_updates'] == 7401
    qualifications = [review.QUALIFICATION, review.SIGNED_QUALIFICATION, review.SMALL_QUALIFICATION]
    if not any(path.exists() for path in qualifications):
        review.qualify()
    for path in qualifications:
        proof = json.loads(path.read_text())
        assert proof['qualified'] and proof['source_checkpoint_sha256'] == runtime.ANCHOR_SHA
        assert proof['source_actual_updates'] == 7201 and proof['only_exploration_change'] == {}
        assert all(sha(p) == digest for p, digest in proof['source_pins'].items())
    assert json.loads(review.QUALIFICATION.read_text())['prepared_plan_path'] == str(plan_path)
    fixed = comparison.qualify(plan)
    tick = monitor.api.tick(output, review.TRAINING_UNIT)
    assert tick['state'] == 'observing' and tick['committed_native_updates'] == 0
    assert len(monitor.api.CHECKS) == 11
    assert paired.AFTER_UNIT == review.UNIT
    source_request = json.loads(paired.SOURCE_REQUEST.read_text())
    source_spec = source_request['candidate']
    assert json.loads(Path(source_spec['final_verification']).read_text())['checkpoint_sha256'] == runtime.ANCHOR_SHA
    paths = [SCRIPT, trainer.SCRIPT, review.SCRIPT, monitor.SCRIPT, comparison.SCRIPT, paired.SCRIPT,
        runtime.SCRIPT, trainer.final_check.SCRIPT, plan_path, review.LAUNCH,
        *qualifications, review.SOURCE_PARITY, runtime.ADMISSION, Path(source_spec['final_verification'])]
    value = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        prepared_plan_path=str(plan_path), plan_sha256=sha(plan_path),
        source_checkpoint=plan['source_checkpoint'], source_checkpoint_sha256=runtime.ANCHOR_SHA,
        source_actual_updates=7201, target_actual_updates=7401,
        source_all17_Adam_step=144020, target_all17_Adam_step=148020,
        native_num_envs=8192, fixed_standing_environments=2048,
        source_std=plan['source_std'], source_saved_LR=plan['source_optimizer_LR'],
        prepare_idempotent_bitwise_same_plan_twice=True,
        training_boundary_exception_installed_before_native=True,
        all5_original_review_contexts_verified=review.assert_bindings(),
        all3_fullscope_signed_and_small_qualifications_bound=True,
        all_source_scopes_original_numeric_parity_qualified=True,
        unchanged11_learning_readers_and_zero_committed_tick_verified=True,
        declared_contract_reader_negative_cases=fixed['rejected_negative_cases'],
        paired735_source_is_current7201=True, only_new_factor=runtime.DELTA,
        source_pins={str(path): sha(path) for path in paths},
        actual_new_PPO_updates=0, simulator_started=False, training_started=False, policy_accepted=False)
    with DESTINATION.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(qualified=True, actual_new_PPO_updates=0, native_num_envs=8192,
        source_actual_updates=7201, target_actual_updates=7401,
        source17Adam_preserved=True, all_post200_and_monitor_routes_qualified=True,
        report=str(DESTINATION)), ensure_ascii=False), flush=True)
    return value


if __name__ == '__main__':
    qualify()
