"""Original fixed-source comparison with declared count metadata and saved wheels."""
import argparse
import ast
import json
from pathlib import Path

from joint_pd_continuation import sha
from audit_saved_action_limits import transform
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import compare_plain27_recorded_joint_targets as original

SCRIPT = Path(__file__).resolve()


def expected_population_difference(configs):
    before, after = configs
    if (before['scene']['num_envs'] not in (4096,8192)
            or after['scene']['num_envs'] != 8192 or after['terrain']['num_envs'] != 8192
            or before['scene']['num_envs'] != before['terrain']['num_envs']):
        raise ValueError('Admit only the declared8192 source/candidate population boundary')
    return {'/log_dir'} | {'/' + key + '/num_envs' for key in ('scene','terrain')
                           if before[key]['num_envs'] != after[key]['num_envs']}


def compare(directory, through):
    tree = transform(original.compare, [
        ("{'/log_dir'}", 'expected_population_difference(cfgs)'),
        ('np.clip(r[..., (2, 5)], -1., 1.)',
         'np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))')])
    scope = dict(original.__dict__, SCRIPT=SCRIPT,
        expected_population_difference=expected_population_difference,
        saved_limit=checkpoint_wheel_action_limit)
    exec(compile(ast.fix_missing_locations(tree), str(original.SCRIPT), 'exec'), scope)
    path, report = scope['compare'](Path(directory), through)
    receipt = Path(directory) / ('parallel8192_fixed_source_comparison_prefix_' + str(through).zfill(4) + '_qualification.json')
    if receipt.exists():
        raise ValueError('Preserve the original fixed-input qualification')
    receipt.write_text(json.dumps(dict(qualified=True, original_numeric_mapping_and_FK_unchanged=True,
        original_all17_Adam_RNG_and_input_identity_checks_preserved=True,
        only_declared_count_metadata_and_saved_wheel_counter_bound_adapted=True,
        actual_native_population=8192, source_report=str(path), source_report_sha256=sha(path),
        original_reader_sha256=sha(original.SCRIPT), wrapper_sha256=sha(SCRIPT),
        simulator_started=False, optimizer_executed=False, policy_accepted=False),
        ensure_ascii=False, indent=2)+'\n')
    return path, report


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory',type=Path)
    cli.add_argument('--through',type=int,required=True)
    args=cli.parse_args()
    path, report=compare(args.directory,args.through)
    print(dict(report=str(path),fixed_input_only=True,training_modified=False))
