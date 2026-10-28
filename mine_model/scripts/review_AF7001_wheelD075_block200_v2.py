"""Fresh original-function review modules; never re-adapt compiled functions.

The v1 preflight detected inspect.getsource on a previously compiled function
re-executing module-level legacy bindings. No native review or production PPO
had started. Keep that evidence; construct each original module once here.
"""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

import review_AF7001_wheelD075_block200 as previous
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
template, adapter = previous.template, previous.adapter
native_runtime, final_check, runtime = previous.native_runtime, previous.final_check, previous.runtime
PREPARATION = previous.PREPARATION
QUALIFICATION = PREPARATION / 'post200_complete_qualification_v2.json'
SIGNED_QUALIFICATION = PREPARATION / 'post200_signed_qualification_v2.json'
SMALL_QUALIFICATION = PREPARATION / 'post200_small_yaw120_qualification_v2.json'
OUTPUT, STATIC, EXTRA, SMALL = previous.OUTPUT, previous.STATIC, previous.EXTRA, previous.SMALL
UNIT, TRAINING_UNIT = previous.UNIT, previous.TRAINING_UNIT
api = ModuleType('_wheelD075_fresh_complete_original_review')
api.__dict__.update(previous.api.__dict__)
for name in ('SCRIPT', 'QUALIFICATION', 'SIGNED_QUALIFICATION', 'SMALL_QUALIFICATION',
             'runtime', 'native_runtime', 'final_check'):
    setattr(api, name, globals()[name])
api.ADAPTERS = copy.deepcopy(template.ADAPTERS[:3])
api.core_api = SimpleNamespace(**dict(template.core.__dict__, errors=api.errors, static_raw=api.static_raw))


api.base = adapter.private_module(template.core.SCRIPT, '_wheelD075_fresh_original_full_scope')
api.base.SCRIPT, api.base.QUALIFICATION, api.base.OUTPUT, api.base.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
api.base.runtime, api.base.final_check = runtime, final_check
api.base.errors, api.base.static_raw = api.errors, api.static_raw
api.ADAPTERS += adapter.retarget(api.base, ('run',), integers={'*': {6001: 7201, 120020: 144020}},
    exact_strings=previous.labels(template.core.run, {6001: 7201}))
api.known = adapter.private_module(template.known_template.SCRIPT, '_wheelD075_fresh_original_known54')
api.known.SCRIPT, api.known.QUALIFICATION, api.known.OUTPUT, api.known.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
api.known.runtime, api.known._api, api.known.original = runtime, api.base, api.core_api
api.ADAPTERS += adapter.retarget(api.known, ('run',), exact_strings=template.METADATA)
api.retained = adapter.private_module(template.retained_template.SCRIPT, '_wheelD075_fresh_original_retained24')
api.retained.SCRIPT, api.retained.QUALIFICATION, api.retained.OUTPUT, api.retained.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
api.retained.runtime, api.retained.final_check = runtime, final_check
api.retained._base, api.retained._api = api.base, api.known
api.retained.core, api.retained.long_raw = api.core_api, api.long_raw
install = FunctionType(previous.install_retained.__code__, dict(previous.__dict__, api=api),
                       previous.install_retained.__name__, previous.install_retained.__defaults__,
                       previous.install_retained.__closure__)
api.ADAPTERS.append(install())

api.signed = adapter.private_module(template.signed_template.SCRIPT, '_wheelD075_fresh_signed_height')
api.signed.SCRIPT, api.signed.QUALIFICATION, api.signed.OUTPUT = SCRIPT, SIGNED_QUALIFICATION, EXTRA
api.signed.AFTER_UNIT, api.signed.runtime = TRAINING_UNIT, runtime
api.signed.SOURCE = template.old_signed.previous.OUTPUT / 'candidate7001'
api.ADAPTERS += adapter.retarget(api.signed, ('run',),
    integers={'*': {6801: 7001, 7001: 7201, 140020: 144020}},
    exact_strings=template.signed_labels,
    keyword_names={'source6801': 'source7001', 'candidate7001': 'candidate7201'})
api.small = adapter.private_module(template.small_template.SCRIPT, '_wheelD075_fresh_small_yaw120')
api.small.SCRIPT, api.small.QUALIFICATION, api.small.OUTPUT, api.small.runtime = SCRIPT, SMALL_QUALIFICATION, SMALL, runtime


def install_small():
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
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in template.METADATA:
            edits.append((node, node.value))
            node.value = template.METADATA[node.value]
    modified = copy.deepcopy(tree)
    del function.body[index:index + len(added)]
    for node, value in edits:
        node.value = value
    assert ast.dump(tree) == before
    exec(compile(ast.fix_missing_locations(modified), str(template.small_template.SCRIPT), 'exec'), api.small.__dict__)
    return dict(original_small_run_AST_restored_identical=True,
        only_declared_candidate_expected_controller_fields_added=True,
        original_numeric_scorer_and_98_windows_unchanged=True)


api.ADAPTERS.append(install_small())


def assert_bindings():
    modules = (api.base, api.known, api.retained, api.signed, api.small)
    for module in modules:
        assert module.runtime is runtime
        assert module.run.__globals__ is module.__dict__
        assert module.SCRIPT == SCRIPT
    for module in modules[:3]:
        assert module.OUTPUT == OUTPUT and module.QUALIFICATION == QUALIFICATION
    assert api.signed.OUTPUT == EXTRA and api.signed.QUALIFICATION == SIGNED_QUALIFICATION
    assert api.small.OUTPUT == SMALL and api.small.QUALIFICATION == SMALL_QUALIFICATION
    assert api.known._api is api.base and api.retained._api is api.known and api.retained._base is api.base
    assert api.base.native.__globals__ is api.base.__dict__
    assert api.retained.EXPECTED_CFG_DIFFERENCES == previous.expected_differences()
    static = [value for value in api.base.run.__code__.co_consts
              if isinstance(value, str) and 'joint_pd_verified_static' in value]
    assert len(static) == 1 and 'wheelD075' in static[0] and 'yawsigma0025' not in static[0]
    return dict(all_five_review_contexts_bound_to_selected_runtime=True,
        all_five_run_globals_bound_to_their_own_fresh_modules=True,
        all_output_and_qualification_routes_bound=True,
        source_and_target_native_replay_paths_explicit=True,
        unique_static_report_filename=static[0], compiled_functions_not_reinspected=True)


def annotated_save(path, value):
    path = Path(path)
    if path in (QUALIFICATION, SIGNED_QUALIFICATION, SMALL_QUALIFICATION):
        required = (SCRIPT, previous.SCRIPT, template.SCRIPT, native_runtime.SCRIPT, final_check.SCRIPT,
            SCRIPT.with_name('train_AF7001_wheelD075_block200.py'),
            SCRIPT.with_name('train_AF7001_wheelD075_block200_v2.py'))
        value = dict(value, source_pins={**value['source_pins'], **{str(p): sha(p) for p in required}},
            fresh_module_binding_preflight=assert_bindings())
    if path == QUALIFICATION:
        value = dict(value, all_original_scopes_and_numeric_readers_retained=True,
            native_num_envs=native_runtime.NUM_ENVS,
            declared_expected_configuration_differences=previous.expected_differences(),
            only_controller_change=native_runtime.DELTA['controller'],
            cached_small54_original_numeric_report_reproduced_exactly=True,
            full_frozen27_controller_expectation_verified=True,
            supersedes_preflight_invalid_v1=str(previous.QUALIFICATION))
    elif path == SMALL_QUALIFICATION:
        value = dict(value, candidate_expected_reward_difference={},
                     candidate_expected_controller_difference=native_runtime.DELTA['controller'])
    template.core.save(path, value)


api.core = SimpleNamespace(**dict(template.core.__dict__, save=annotated_save))
for name in ('qualify', 'run'):
    function = getattr(previous.api, name)
    setattr(api, name, FunctionType(function.__code__, api.__dict__, function.__name__,
                                   function.__defaults__, function.__closure__))
assert_bindings()


def qualify():
    assert_bindings()
    assert native_runtime.NUM_ENVS == 8192
    actual = previous.read(PREPARATION / 'frozen27/resolved_env.json')
    original = previous.read(native_runtime.SOURCE_REPORT.parent / 'replay/resolved_env.json')
    expected = native_runtime.expected_configuration(original, 27)
    expected['log_dir'] = actual['log_dir']
    assert expected == actual
    proof = previous.read(template.small_template.QUALIFICATION)
    reference = previous.read(proof['source_reference_resolved_env'])
    report, _ = template.small_template.template.assess(template.small_template.SOURCE_DIRECTORY,
        Path(proof['source_checkpoint']), reference, proof)
    assert report == previous.read(template.small_template.CACHE / 'candidate7001/all54_raw_and_windows_verified.json')
    return api.qualify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    args = parser.parse_args()
    result = qualify() if args.qualify_only else api.run()
    print(json.dumps({key: result[key] for key in ('qualified', 'completed', 'policy_accepted') if key in result}))
