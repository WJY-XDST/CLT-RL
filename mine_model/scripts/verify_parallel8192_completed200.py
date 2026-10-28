"""Original completed200 verifier with explicit8192 population metadata.

Keep every original tensor, Adam, gradient, LR, RNG and native-alignment
check. Add only the declared scene/terrain count differences to the saved
configuration comparison and label the verified population truthfully.
"""
import argparse
import ast
import copy
from pathlib import Path

import plain27_sampler_metadata_adapter as adapter
import verify_completed_plain27_joint_pd_trial as original

SCRIPT = Path(__file__).resolve()
_api = adapter.private_module(original.SCRIPT, '_parallel8192_original_completed200_verifier')
_api.SCRIPT = SCRIPT


def declared_population_differences(plan, before, after):
    if (plan['native_num_envs'] != 8192 or plan['fixed_standing_environments'] != 2048
            or before['scene']['num_envs'] not in (4096, 8192)
            or before['terrain']['num_envs'] != before['scene']['num_envs']
            or after['scene']['num_envs'] != 8192 or after['terrain']['num_envs'] != 8192
            or any(plan['next_configuration_changes'][k] for k in
                   ('rewards', 'commands', 'exploration', 'controller', 'physics'))):
        raise ValueError('This entry admits only unchanged ordinary PPO with8192 environments')
    return {'/' + key + '/num_envs' for key in ('scene', 'terrain')
            if before[key]['num_envs'] != after[key]['num_envs']}


module = ast.parse(Path(original.SCRIPT).read_text())
function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
tree = ast.Module(body=[function], type_ignores=[])
before = ast.dump(tree)
indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'delta' for t in node.targets)
    and ast.unparse(node.value) == 'differences(cfg, current_cfg)']
if len(indices) != 1:
    raise ValueError('Requalify the original exact saved-configuration comparison')
index = indices[0]
added = ast.parse('expected.update(declared_population_differences(plan, cfg, current_cfg))').body[0]
function.body.insert(index, added)
keywords = [n for n in ast.walk(tree) if isinstance(n, ast.keyword)
            and n.arg == 'all200_all4096_alignment_max_errors']
if len(keywords) != 1:
    raise ValueError('Requalify the original verified population result key')
keywords[0].arg = 'all200_all8192_alignment_max_errors'
modified = copy.deepcopy(tree)
function.body.pop(index)
keywords[0].arg = 'all200_all4096_alignment_max_errors'
if ast.dump(tree) != before:
    raise ValueError('Original completed200 numerical tests changed outside declared count metadata')
_api.declared_population_differences = declared_population_differences
exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), 'exec'), _api.__dict__)


def verify(directory, destination=None):
    directory = Path(directory).resolve()
    plan = original.json.loads((directory / 'experiment_plan.json').read_text())
    if plan['native_num_envs'] != 8192 or plan['fixed_standing_environments'] != 2048:
        raise ValueError('Never reinterpret4096 records as8192 evidence')
    destination = Path(destination) if destination is not None else (
        directory / 'final_checkpoint_verification_readonly_parallel8192.json')
    return _api.verify(directory, destination)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--destination', type=Path)
    args = cli.parse_args()
    proof = verify(args.directory, args.destination)
    print({k:proof[k] for k in ('verified', 'actual_ppo_updates', 'all17_Adam_step',
                              'saved_configuration_differences')})
