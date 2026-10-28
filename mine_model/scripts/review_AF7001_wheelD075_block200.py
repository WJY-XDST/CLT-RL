"""Retain the entire original review suite for the single wheel-drive trial.

Reuse the established readers with output/count/controller metadata changes.
Small-yaw expected configuration explicitly includes both synchronized drive
fields; its scorer, complete54 rows and98 windows remain the original code.
"""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import plain27_sampler_metadata_adapter as adapter
import review_AF7001_yawsigma0025_block200 as template
import joint_pd_AF7001_wheelD075_runtime as native_runtime
import verify_AF7001_wheelD075_completed200 as final_check
from qualify_source801_drift50_transfer import differences
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
RESULTS = native_runtime.RESULTS
PREPARATION = RESULTS / 'joint_pd_AF7001_wheelD075_qualification_20261023'
LAUNCH = PREPARATION / 'owned_block200_launch.json'
QUALIFICATION = PREPARATION / 'post200_complete_qualification.json'
SIGNED_QUALIFICATION = PREPARATION / 'post200_signed_qualification.json'
SMALL_QUALIFICATION = PREPARATION / 'post200_small_yaw120_qualification.json'
OUTPUT = RESULTS / 'joint_pd_AF7001_wheelD075_review7201_20261023'
STATIC = RESULTS / 'joint_pd_static_retention_AF7001_wheelD075_source7201_20261023'
EXTRA = RESULTS / 'joint_pd_AF7001_wheelD075_fullspeed_height_source7001_candidate7201_20261023'
SMALL = RESULTS / 'joint_pd_AF7001_wheelD075_small_yaw120_20261023'
UNIT = 'clt-joint-pd-af7001-wheelD075-full-reviews-20261023.service'
TRAINING_UNIT = 'clt-joint-pd-af7001-wheelD075-200-20261023.service'


def read(path):
    return json.loads(Path(path).read_text())


def expected_differences():
    before = native_runtime.baseline()[2]
    after = native_runtime.expected_configuration(before, native_runtime.NUM_ENVS)
    after['log_dir'] = 'THE_NEXT_OWNED_OUTPUT_DIRECTORY'
    return differences(before, after)


def ready():
    native_runtime.repair_ready()
    old = template.old_context.ready()
    native_runtime.review_ready(Path(old['source_checkpoint']))
    return dict(old, next_configuration_changes=native_runtime.DELTA)


runtime = SimpleNamespace(**dict(template.old_context.api_runtime().__dict__,
    SCRIPT=native_runtime.SCRIPT, FLAG=native_runtime.FLAG, profile=native_runtime.profile,
    require_source=native_runtime.require_source, replay_command=native_runtime.replay_command,
    repair_ready=ready, DELTA=native_runtime.DELTA))


def labels(function, clocks):
    return {key: value.replace('AF7001_yawsigma0025', 'AF7001_wheelD075')
            for key, value in template.labels(function, clocks).items()}


api = adapter.private_module(template.SCRIPT, '_wheelD075_complete_original_review_adapter')
for name in ('SCRIPT', 'PREPARATION', 'LAUNCH', 'QUALIFICATION', 'SIGNED_QUALIFICATION',
             'SMALL_QUALIFICATION', 'OUTPUT', 'STATIC', 'EXTRA', 'SMALL', 'UNIT', 'runtime',
             'native_runtime', 'final_check'):
    setattr(api, name, globals()[name])
api.ready, api.labels = ready, labels
api.ADAPTERS = copy.deepcopy(template.ADAPTERS[:3])
for module in (api.base, api.known, api.retained):
    module.SCRIPT, module.QUALIFICATION, module.OUTPUT, module.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
    module.runtime = runtime
api.base.final_check = api.retained.final_check = final_check
api.ADAPTERS += adapter.retarget(api.base, ('run',),
    integers={'*': {6001: 7201, 120020: 144020}},
    exact_strings=labels(template.core.run, {6001: 7201}))


def install_retained():
    tree = ast.parse(inspect.getsource(template.retained_template.run))
    before, changes = ast.dump(tree), []
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
              and ast.unparse(node.left) == "final['saved_configuration_differences']"]
    assert len(guards) == 1
    assert ast.literal_eval(guards[0].comparators[0]) == ['/log_dir', '/rewards/tracking_lin_vel']
    changes.append((guards[0], 'comparators', copy.deepcopy(guards[0].comparators)))
    guards[0].comparators = [ast.parse('EXPECTED_CFG_DIFFERENCES', mode='eval').body]
    replacements = labels(template.retained_template.run, {6201: 7001, 6401: 7201})
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (6201, 6401, 128020):
            changes.append((node, 'value', node.value))
            node.value = {6201: 7001, 6401: 7201, 128020: 144020}[node.value]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replacements:
            changes.append((node, 'value', node.value))
            node.value = replacements[node.value]
    modified = copy.deepcopy(tree)
    for node, field, value in reversed(changes):
        setattr(node, field, value)
    assert ast.dump(tree) == before
    api.retained.EXPECTED_CFG_DIFFERENCES = expected_differences()
    exec(compile(ast.fix_missing_locations(modified), str(template.retained_template.SCRIPT), 'exec'), api.retained.__dict__)
    return dict(original_run_AST_restored_identical=True,
                declared_saved_configuration_differences=expected_differences())


api.ADAPTERS.append(install_retained())
api.signed.SCRIPT, api.signed.QUALIFICATION, api.signed.OUTPUT = SCRIPT, SIGNED_QUALIFICATION, EXTRA
api.signed.AFTER_UNIT, api.signed.runtime = TRAINING_UNIT, runtime
api.small.SCRIPT, api.small.QUALIFICATION, api.small.OUTPUT, api.small.runtime = SCRIPT, SMALL_QUALIFICATION, SMALL, runtime


def install_small_expected_controller():
    tree = ast.parse(inspect.getsource(template.small_template.run))
    before = ast.dump(tree)
    function = tree.body[0]
    indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
               and ast.unparse(node.value) == 'template.assess(directory, candidate, expected, proof)']
    assert len(indices) == 1
    index = indices[0]
    added = ast.parse("expected['wheel_damping'] = .75\n"
                      "expected['robot']['actuators']['wheels']['damping'] = .75").body
    function.body[index:index] = added
    modified = copy.deepcopy(tree)
    del function.body[index:index + len(added)]
    assert ast.dump(tree) == before
    exec(compile(ast.fix_missing_locations(modified), str(template.small_template.SCRIPT), 'exec'), api.small.__dict__)
    api.ADAPTERS.extend(adapter.retarget(api.small, ('run',), exact_strings=template.METADATA))
    # retarget reads the original source function; add the same declared
    # expectation after its filename-only adapter has been compiled.
    tree = copy.deepcopy(modified)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in template.METADATA:
            node.value = template.METADATA[node.value]
    exec(compile(ast.fix_missing_locations(tree), str(template.small_template.SCRIPT), 'exec'), api.small.__dict__)
    return dict(original_small_run_AST_restored_identical=True,
                only_declared_candidate_expected_controller_fields_added=True,
                scorer_and_98_windows_unchanged=True)


api.ADAPTERS.append(install_small_expected_controller())
original_save = template.core.save


def annotated_save(path, value):
    path = Path(path)
    if path in (QUALIFICATION, SIGNED_QUALIFICATION, SMALL_QUALIFICATION):
        required = (SCRIPT, template.SCRIPT, native_runtime.SCRIPT, final_check.SCRIPT,
                    SCRIPT.with_name('train_AF7001_wheelD075_block200.py'))
        value = dict(value, source_pins={**value['source_pins'],
            **{str(file): sha(file) for file in required}})
    if path == QUALIFICATION:
        value = dict(value, all_original_scopes_and_numeric_readers_retained=True,
            native_num_envs=native_runtime.NUM_ENVS,
            declared_expected_configuration_differences=expected_differences(),
            only_controller_change=native_runtime.DELTA['controller'],
            inherited_reward_key_is_unchanged='yaw_rate_error_sq=-200',
            full_frozen27_controller_expectation_verified=True,
            cached_small54_original_numeric_report_reproduced_exactly=True)
    elif path == SMALL_QUALIFICATION:
        value = dict(value, candidate_expected_reward_difference={},
            candidate_expected_controller_difference=native_runtime.DELTA['controller'])
    original_save(path, value)


api.core = SimpleNamespace(**dict(template.core.__dict__, save=annotated_save))
api.ADAPTERS += adapter.retarget(api, ('run',), exact_strings={
    'complete_all_scopes_AF7001_yawsigma0025.json': 'complete_all_scopes_AF7001_wheelD075.json'})


def qualify():
    assert native_runtime.NUM_ENVS == 8192
    native = read(PREPARATION / 'frozen27/resolved_env.json')
    original = read(native_runtime.SOURCE_REPORT.parent / 'replay/resolved_env.json')
    expected = native_runtime.expected_configuration(original, 27)
    expected['log_dir'] = native['log_dir']
    assert expected == native
    small_proof = read(template.small_template.QUALIFICATION)
    reference = read(small_proof['source_reference_resolved_env'])
    report, _ = template.small_template.template.assess(template.small_template.SOURCE_DIRECTORY,
        Path(small_proof['source_checkpoint']), reference, small_proof)
    assert report == read(template.small_template.CACHE / 'candidate7001/all54_raw_and_windows_verified.json')
    return api.qualify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    args = parser.parse_args()
    value = qualify() if args.qualify_only else api.run()
    print(json.dumps({key: value[key] for key in ('qualified', 'completed', 'policy_accepted') if key in value}))
