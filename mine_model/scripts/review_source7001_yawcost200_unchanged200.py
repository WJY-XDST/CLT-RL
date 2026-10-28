"""All original735/static/dynamic/120s reviews for unchanged7001 to7201."""
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

from joint_pd_continuation import sha
import plain27_sampler_metadata_adapter as adapter
import source7001_yawcost200_unchanged_context as context
import review_wheel125_training_block as core
import review_source6001_orientation060_block200 as known_template
import review_source6201_lin300_block200 as retained_template
import verify_completed_plain27_joint_pd_trial as final_check

SCRIPT = Path(__file__).resolve()
RESULTS = context.RESULTS
QUALIFICATION = context.QUALIFICATION/'post200_original_review_qualification.json'
OUTPUT, STATIC = context.OUTPUT, context.STATIC
runtime = context.api_runtime()
qualifier = SimpleNamespace(OUTPUT=context.QUALIFICATION)


def strings(function, clocks):
    result = {}
    for node in ast.walk(ast.parse(inspect.getsource(function))):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        changed = re.sub(r'(source|candidate)(6001|6201|6401)',
            lambda m: m.group(1)+str(clocks.get(int(m.group(2)), int(m.group(2)))), value)
        changed = changed.replace('_20261020', '_20261022')
        if value == 'joint_pd_verified_static_source6001_wheel125_20261020.json':
            changed = 'joint_pd_verified_static_source7201_yawcost200_unchanged_20261022.json'
        if changed != value:
            result[value] = changed
    return result


_base = adapter.private_module(core.SCRIPT, '_same200_original_all_scope_review')
_base.SCRIPT, _base.QUALIFICATION, _base.OUTPUT, _base.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_base.runtime, _base.final_check = runtime, final_check
BASE_ADAPTER = adapter.retarget(_base, ('run',),
    integers={'*': {6001: 7201, 120020: 144020}},
    exact_strings=strings(core.run, {6001: 7201}))
_known = adapter.private_module(known_template.SCRIPT, '_same200_original_known54_review')
_known.SCRIPT, _known.QUALIFICATION, _known.OUTPUT, _known.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_known.runtime, _known.qualifier, _known._api = runtime, qualifier, _base
_root = adapter.private_module(retained_template.SCRIPT, '_same200_original_retained24_review')
_root.SCRIPT, _root.QUALIFICATION, _root.OUTPUT, _root.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_root.runtime, _root.qualifier, _root.final_check = runtime, qualifier, final_check
_root._base, _root._api = _base, _known


def install_root():
    tree = ast.parse(inspect.getsource(retained_template.run))
    before, changes = ast.dump(tree), []
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
        and ast.unparse(n.left) == "final['saved_configuration_differences']"]
    if len(guards) != 1 or ast.literal_eval(guards[0].comparators[0]) != ['/log_dir', '/rewards/tracking_lin_vel']:
        raise ValueError('Requalify the original exact configuration-diff guard')
    changes.append((guards[0], 'comparators', copy.deepcopy(guards[0].comparators)))
    guards[0].comparators = [ast.parse("['/log_dir']", mode='eval').body]
    labels = strings(retained_template.run, {6201: 7001, 6401: 7201})
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (6201, 6401, 128020):
            changes.append((node, 'value', node.value))
            node.value = {6201: 7001, 6401: 7201, 128020: 144020}[node.value]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in labels:
            changes.append((node, 'value', node.value))
            node.value = labels[node.value]
    modified = copy.deepcopy(tree)
    for node, field, value in reversed(changes):
        setattr(node, field, value)
    if ast.dump(tree) != before:
        raise ValueError('Original metrics, windows, gates or failed None semantics changed')
    exec(compile(ast.fix_missing_locations(modified), str(retained_template.SCRIPT), 'exec'), _root.__dict__)
    return dict(original_run_AST_restored_identical=True,
        declared_metadata_and_unchanged_diff_guard_changes=len(changes),
        exact_saved_configuration_diff_required=['/log_dir'])


ROOT_ADAPTER = install_root()


def qualify():
    decision = context.ready()
    launch_path = context.QUALIFICATION/'training_launch.json'
    launch = context.read(launch_path)
    entry_path = context.QUALIFICATION/'full_original_block_entry_preflight.json'
    entry = context.read(entry_path)
    plan_path = Path(entry['prepared_plan_path'])
    plan = context.read(plan_path)
    cpu_path = context.QUALIFICATION/'qualification.json'
    cpu = context.read(cpu_path)
    old_path = RESULTS/'joint_pd_source6601_targeted_turn030_qualification_v3_20261021/post200_original_review_qualification.json'
    old = context.read(old_path)
    if (QUALIFICATION.exists() or OUTPUT.exists() or STATIC.exists()
            or not cpu['qualified'] or not entry['qualified']
            or launch['source_checkpoint_sha256'] != context.ANCHOR_SHA
            or launch['source_actual_updates'] != 7001 or launch['target_actual_updates'] != 7201
            or launch['target_all17_Adam_step'] != 144020 or launch['planned_actual_updates'] != 200
            or plan['next_configuration_changes'] != context.DELTA or plan['profile'] != runtime.profile()
            or plan['early_learning_inspection_actual_updates'] != [50, 100]
            or cpu['actual_PPO_updates_executed'] or cpu['optimizer_steps_executed']):
        raise ValueError('Require exact unchanged200, own complete7001 reviews and retained CPU load')
    source, _, cfg = context.anchor()
    if cfg != plan['training_config']:
        raise ValueError('Preflight must retain the complete saved source configuration')
    fixture = context.SOURCE_REVIEW/'full_height_1.0/candidate'
    reader, captured, evidence = core.combined.reader(old['full_height_cases']['1.0'], old['full_seeds'])
    rebuilt = reader(fixture, source)
    prior = context.read(fixture/'motion025_full_height_diagnosis.json')
    if len(captured) != 1 or rebuilt['trials'] != prior['trials'] or rebuilt['total_trials'] != 51:
        raise ValueError('Original real51 physical metrics and strict flags must reproduce exactly')
    proof = dict(old)
    proof.update(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(launch_path), training=launch['output'], after_unit=launch['unit'],
        output=str(OUTPUT), static_output=str(STATIC), target_actual_updates=7201,
        target_all17_Adam_step=144020, prepared_plan_path=str(plan_path),
        actual_plan_must_match_preflight_source_profile_and_configuration=True,
        known_interior_source_directory=str(context.SOURCE_REVIEW/'known_interior/candidate'),
        known_interior_source_raw=str(context.SOURCE_REVIEW/'known_interior/all54_raw_verified.json'),
        retained_left_source_directory=str(context.SOURCE_REVIEW/'retained_left_interior/candidate'),
        retained_left_source_raw=str(context.SOURCE_REVIEW/'retained_left_interior/all24_raw_verified.json'),
        complete_review_expected=str(OUTPUT/'complete_with_retained_interior_reviews.json'),
        source7001_all_own735_static_dynamic_120_and_signedheight_complete=True,
        unchanged_recipe_uses_source_own_native_fixtures=True, new_native_transfer_required=False,
        original_real51_metrics_flags_reproduced=True, reader_adapter=evidence,
        source_target_metadata_adapters=BASE_ADAPTER+[ROOT_ADAPTER],
        acceptance_changed=False, policy_accepted=False)
    pins = {**old['source_pins'], **plan['source_files'], **decision['source_pins'], **cpu['source_files']}
    for p in (SCRIPT, context.SCRIPT, core.SCRIPT, known_template.SCRIPT, retained_template.SCRIPT,
            adapter.SCRIPT, final_check.SCRIPT, runtime.SCRIPT, old_path, launch_path,
            entry_path, cpu_path, plan_path, fixture/'manifest.json', fixture/'trace.csv'):
        pins[str(p)] = sha(p)
    core.intact(pins)
    proof['source_pins'] = pins
    core.save(QUALIFICATION, proof)
    return proof


def run():
    proof = context.read(QUALIFICATION)
    staged = context.read(proof['prepared_plan_path'])
    # The actual directory is created by the normal training entry. Review
    # qualification is completed before launch using its immutable CPU plan.
    from audit_plain27_after_queued_reviews import wait_unit
    OUTPUT.parent.mkdir(exist_ok=True)
    wait_unit(proof['after_unit'], context.QUALIFICATION/'waiting_for_owned_training.json')
    actual = context.read(Path(proof['training'])/'experiment_plan.json')
    for key in ('source_sha256', 'source_actual_ppo_updates', 'target_actual_ppo_updates',
                'bounded_training_updates', 'training_config', 'profile', 'next_configuration_changes'):
        if actual[key] != staged[key]:
            raise ValueError('Actual trained source/recipe differs from the qualified unchanged entry: '+key)
    return _root.run()


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    result = qualify() if args.qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates',
        'all_original_and_retained_reviews_complete', 'policy_accepted') if k in result}))
