"""Verify all five capacity updates outside Kit's process shutdown.

Keep v1/v2 and their CPU proofs. Kit may exit without returning to runpy;
the production trainer already verifies persisted records in its parent.
Apply the same original two-record validator to all five capacity records,
including an overlapping final pair, while counting each Adam update once.
No native process, optimizer update or physical step is repeated here.
"""
import ast
import copy
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

import check_plain27_environment_capacity as original
import finish_joint_pd_native_audit as audit
import plain27_sampler_metadata_adapter as adapter

SCRIPT = Path(__file__).resolve()


def capacity_external_finalization(folder, request, count):
    folder = Path(folder)
    directory = folder / 'native_learning_signal'
    native = original.read(directory / 'plan.json')
    if (native['source_sha256'] != request['source_checkpoint_sha256']
            or native['source_actual_ppo_updates'] != request['source_actual_updates']
            or native['target_actual_ppo_updates'] != request['source_actual_updates'] + 5
            or native['requested_updates'] != 5
            or any(original.sha(p) != digest for p, digest in native['source_files_sha256'].items())):
        raise ValueError('Preserve the exact five-update source, target and native observation code')
    paths = sorted(directory.glob('audit_update_*.json'))
    records = [original.read(path) for path in paths]
    if len(records) != 5 or [r['update'] for r in records] != [1, 2, 3, 4, 5]:
        raise ValueError('Require all five completed capacity updates, including the odd final update')
    reader = adapter.private_module(audit.__file__, '_capacity_original_native_pair_validator')
    evidence = adapter.retarget(reader, ('summarize_records',), integers={'*': {8192: count}})
    pairs = [(0, 1), (2, 3), (3, 4)]
    for left, right in pairs:
        reader.summarize_records([dict(records[left], update=1), dict(records[right], update=2)])
    for record in records:
        if record['total_environments'] != count:
            raise ValueError('Actual native environment count differs')
        if any(not math.isfinite(float(value)) for value in record['original_loss_metrics'].values()):
            raise ValueError('Nonfinite original PPO loss')
        for step in record['optimizer_steps']:
            if (step['adaptive_learning_rate'] <= 0.
                    or step['actor_gradient_norm_after_global_clip'] < 0.
                    or step['critic_gradient_norm_after_global_clip'] < 0.):
                raise ValueError('Invalid recorded original optimizer update')
        with np.load(directory / ('rollout_' + str(record['update']).zfill(4) + '.npz'), allow_pickle=False) as arrays:
            if (arrays['observations'].shape != (48, 32, 27)
                    or arrays['actions'].shape != (48, 32, 6)
                    or any(not np.isfinite(arrays[key]).all() for key in arrays.files)
                    or not np.array_equal(arrays['actions'], arrays['native_raw_actions'])):
                raise ValueError('Incomplete, nonfinite or mismatched full48-step native rollout')
    maximum = {key: max(r['all_environment_alignment_max_errors'][key] for r in records)
               for key in audit.ALIGNMENT_KEYS}
    result = dict(completed=True, actual_observed_updates=5,
        optimizer_minibatches=sum(len(r['optimizer_steps']) for r in records),
        all_environment_alignment_max_errors=maximum, all_probe_RNG_unchanged=True,
        actual_environment_count=count, original_native_pair_validator_used=True,
        original_pair_numeric_AST_restored_identical=evidence,
        pair_update_indices_1based=[[a + 1, b + 1] for a, b in pairs],
        overlapping_fourth_record_is_not_counted_twice=True,
        all_five_records_counted_once=True, verified_outside_Kit=True,
        in_process_completion_returned=(directory / 'completed.json').exists(),
        PPO_updates_repeated=0, original_records_and_checkpoints_unchanged=True,
        records_sha256={str(p): original.sha(p) for p in paths},
        source_files_sha256={str(SCRIPT): original.sha(SCRIPT), str(audit.__file__): original.sha(audit.__file__)},
        observed_at=datetime.now().astimezone().isoformat(), policy_accepted=False)
    original.write(directory / 'completed_external_capacity.json', result)
    return result


# Rebuild from the original source rather than inspecting a previously
# compiled adapter, whose line numbers no longer refer to that source.
module_tree = ast.parse(Path(original.__file__).read_text())
function = next(n for n in module_tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'verify_finished_probe')
tree = ast.Module(body=[function], type_ignores=[])
before = ast.dump(tree)
completion = [n for n in function.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'complete' for t in n.targets)]
if len(completion) != 1 or ast.unparse(completion[0].value) != "read(folder / 'native_learning_signal/completed.json')":
    raise ValueError('Requalify the original in-process completion read')
old_value = copy.deepcopy(completion[0].value)
completion[0].value = ast.parse('capacity_external_finalization(folder, request, count)', mode='eval').body
indices = [i for i, n in enumerate(function.body) if isinstance(n, ast.If)
    and ast.dump(n.test) == ast.dump(ast.parse('agent_before != agent_after', mode='eval').body)]
if len(indices) != 1:
    raise ValueError('Requalify the original full-agent comparison')
index = indices[0]
added = ast.parse("""
agent_before['run_name'] = agent_after['run_name']
agent_before['algorithm']['learning_rate'] = load['optimizer_LR']
""").body
function.body[index:index] = added
modified = copy.deepcopy(tree)
del function.body[index:index + len(added)]
completion[0].value = old_value
if ast.dump(tree) != before:
    raise ValueError('Original saved-checkpoint verification changed beyond the declared completion and metadata adapters')
original.capacity_external_finalization = capacity_external_finalization
exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), 'exec'), original.__dict__)
original.SCRIPT = SCRIPT
require_capacity_selection = original.require_capacity_selection
main = original.main


if __name__ == '__main__':
    main()
