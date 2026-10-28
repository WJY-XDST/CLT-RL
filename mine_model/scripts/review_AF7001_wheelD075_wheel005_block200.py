"""Bind the five unchanged review contexts to the new matched wheel-floor run.

Use the proven fresh-original-function builder. Rebind only runtime, owned
routes and source pins. Never reinspect already compiled adapter functions,
revise a case/window or change an original numeric acceptance reader.
"""
import argparse
import copy
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import plain27_sampler_metadata_adapter as adapter
import review_AF7001_wheelD075_block200_v2 as original
import joint_pd_AF7001_wheelD075_wheel005_runtime as native_runtime
import verify_AF7001_wheelD075_wheel005_completed200 as final_check
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
PREPARATION = native_runtime.QUALIFICATION
LAUNCH = PREPARATION / 'owned_block200_launch.json'
QUALIFICATION = PREPARATION / 'post200_complete_qualification.json'
SIGNED_QUALIFICATION = PREPARATION / 'post200_signed_qualification.json'
SMALL_QUALIFICATION = PREPARATION / 'post200_small_yaw120_qualification.json'
RESULTS = native_runtime.RESULTS
OUTPUT = RESULTS / 'joint_pd_AF7001_D075_wheel005_review7201_20261024'
STATIC = RESULTS / 'joint_pd_static_retention_AF7001_D075_wheel005_source7201_20261024'
EXTRA = RESULTS / 'joint_pd_AF7001_D075_wheel005_signed_height_20261024'
SMALL = RESULTS / 'joint_pd_AF7001_D075_wheel005_small_yaw120_20261024'
UNIT = 'clt-joint-pd-af7001-D075-wheel005-full-reviews-20261024.service'
TRAINING_UNIT = 'clt-joint-pd-af7001-D075-wheel005-200-20261024.service'
TEMPLATE = original.template


def ready():
    native_runtime.repair_ready()
    prior = TEMPLATE.old_context.ready()
    native_runtime.review_ready(Path(prior['source_checkpoint']))
    return dict(prior, next_configuration_changes=copy.deepcopy(native_runtime.DELTA))


runtime = SimpleNamespace(**dict(original.runtime.__dict__,
    SCRIPT=native_runtime.SCRIPT, FLAG=native_runtime.FLAG,
    profile=native_runtime.profile, require_source=native_runtime.require_source,
    replay_command=native_runtime.replay_command,
    repair_ready=ready, DELTA=native_runtime.DELTA))

# Construct each fresh original function only once through the qualified v2
# builder, then change its globals. No inspect.getsource on its compiled run.
builder = adapter.private_module(original.SCRIPT, '_D075_wheel005_fresh_review_builder')
api = builder.api
for name in ('SCRIPT', 'PREPARATION', 'LAUNCH', 'QUALIFICATION', 'SIGNED_QUALIFICATION',
             'SMALL_QUALIFICATION', 'OUTPUT', 'STATIC', 'EXTRA', 'SMALL', 'UNIT',
             'TRAINING_UNIT', 'runtime', 'native_runtime', 'final_check'):
    setattr(builder, name, globals()[name])
    setattr(api, name, globals()[name])
api.ready = ready
for module in (api.base, api.known, api.retained, api.signed, api.small):
    module.SCRIPT, module.runtime = SCRIPT, runtime
for module in (api.base, api.known, api.retained):
    module.QUALIFICATION, module.OUTPUT, module.STATIC = QUALIFICATION, OUTPUT, STATIC
api.base.final_check = api.retained.final_check = final_check
api.signed.QUALIFICATION, api.signed.OUTPUT, api.signed.AFTER_UNIT = SIGNED_QUALIFICATION, EXTRA, TRAINING_UNIT
api.small.QUALIFICATION, api.small.OUTPUT = SMALL_QUALIFICATION, SMALL

# This is the sole global (outside the owned output) report filename. Give the
# new experiment its own name while retaining all numeric code constants.
old_name = 'joint_pd_verified_static_source7201_AF7001_wheelD075_20261022.json'
new_name = 'joint_pd_verified_static_source7201_AF7001_wheelD075_wheel005_20261024.json'
old_run = api.base.run
assert old_run.__code__.co_consts.count(old_name) == 1
new_code = old_run.__code__.replace(co_consts=tuple(
    new_name if value == old_name else value for value in old_run.__code__.co_consts))
assert new_code.co_code == old_run.__code__.co_code
api.base.run = FunctionType(new_code, api.base.__dict__, old_run.__name__,
                           old_run.__defaults__, old_run.__closure__)


def assert_bindings():
    result = builder.assert_bindings()
    assert api.qualify.__globals__ is api.__dict__ and api.run.__globals__ is api.__dict__
    assert api.native_runtime is native_runtime and api.ready is ready
    assert api.LAUNCH == LAUNCH and api.PREPARATION == PREPARATION
    assert api.signed.AFTER_UNIT == TRAINING_UNIT
    assert api.base.final_check is api.retained.final_check is final_check
    assert runtime.profile() == native_runtime.profile()
    assert result['unique_static_report_filename'] == new_name
    return dict(result, matched_wheel005_profile_and_admission_bound=True,
        all_five_original_case_seed_window_and_numeric_readers_unchanged=True)


def annotated_save(path, value):
    path = Path(path)
    if path in (QUALIFICATION, SIGNED_QUALIFICATION, SMALL_QUALIFICATION):
        required = (SCRIPT, original.SCRIPT, original.previous.SCRIPT, TEMPLATE.SCRIPT,
            native_runtime.SCRIPT, final_check.SCRIPT,
            SCRIPT.with_name('train_AF7001_wheelD075_wheel005_block200.py'),
            SCRIPT.with_name('recover_AF7001_wheelD075_completed200_boundary.py'))
        value = dict(value, source_pins={**value['source_pins'],
            **{str(p): sha(p) for p in required}}, fresh_module_binding_preflight=assert_bindings(),
            only_exploration_change=copy.deepcopy(native_runtime.DELTA['exploration']),
            matched_trial_source_actual_updates=7001, future_six_std_floors=[.01,.01,.005,.01,.01,.005])
    if path == QUALIFICATION:
        value = dict(value, all_original_scopes_and_numeric_readers_retained=True,
            native_num_envs=native_runtime.NUM_ENVS,
            declared_expected_configuration_differences=original.previous.expected_differences(),
            only_controller_change=native_runtime.DELTA['controller'],
            cached_small54_original_numeric_report_reproduced_exactly=True,
            full_frozen27_controller_expectation_verified=True)
    elif path == SMALL_QUALIFICATION:
        value = dict(value, candidate_expected_reward_difference={},
            candidate_expected_controller_difference=native_runtime.DELTA['controller'])
    TEMPLATE.core.save(path, value)


api.core = SimpleNamespace(**dict(TEMPLATE.core.__dict__, save=annotated_save))
assert_bindings()


def qualify():
    assert_bindings()
    # The outer builder also reproduces the complete frozen configuration and
    # cached small54 reports through the original readers before admission.
    return builder.qualify()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    result = qualify() if args.qualify_only else api.run()
    print(json.dumps({key: result[key] for key in ('qualified','completed','policy_accepted') if key in result}))
