"""Reuse original numerical readers with8192 count assertions and labels."""
import argparse
import importlib
from pathlib import Path

from joint_pd_continuation import sha
import plain27_sampler_metadata_adapter as adapter

SCRIPT = Path(__file__).resolve()
READERS = {
    'geometry': 'audit_joint_target_geometric_exploration',
    'gradient': 'audit_plain27_gradient_allocation',
    'phase': 'audit_critic_fit_by_resampling_phase',
    'events': 'audit_native_terminal_event_ages',
    'atoms': 'audit_matched_plain27_clipping_atoms',
}


def audit(kind, directory, through):
    original = importlib.import_module(READERS[kind])
    api = adapter.private_module(original.__file__, '_parallel8192_' + kind + '_original_reader')
    api.SCRIPT = SCRIPT
    evidence = adapter.retarget(api, ('audit',), integers={'*': {4096: 8192}},
        strings={'4096': '8192'}, keyword_names={
            'all4096_ended_episodes': 'all8192_ended_episodes',
            'all4096_unsafe_events': 'all8192_unsafe_events',
            'all4096_numerical_events': 'all8192_numerical_events',
            'all_prefix_native4096_alignment_errors_zero': 'all_prefix_native8192_alignment_errors_zero'})
    result = api.audit(Path(directory), through)
    # Original helpers either return(path, report) or just the report.
    if isinstance(result, tuple):
        destination, report = result
    else:
        report = result
        stems = {'phase': 'critic_fit_by_resampling_phase', 'atoms': 'matched_clipping_atoms'}
        destination = Path(directory) / (stems[kind] + '_prefix_' + str(through).zfill(4) + '_readonly.json')
    # A separate receipt preserves each original numerical report verbatim.
    receipt = Path(directory) / ('parallel8192_' + kind + '_reader_prefix_' + str(through).zfill(4) + '_qualification.json')
    if receipt.exists():
        raise ValueError('Preserve the completed reader qualification')
    import json
    receipt.write_text(json.dumps(dict(qualified=True, original_reader=str(original.__file__),
        original_reader_sha256=sha(original.__file__), wrapper_sha256=sha(SCRIPT),
        only_population_assertions_and_labels_changed=True, original_numeric_AST_restored_identical=evidence,
        actual_native_count=8192, original_report=str(destination), original_report_sha256=sha(destination),
        simulator_started=False, optimizer_executed=False, training_modified=False,
        policy_accepted=False), ensure_ascii=False, indent=2)+'\n')
    return destination, report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reader', choices=READERS, required=True)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    destination, report = audit(args.reader, args.directory, args.through)
    print(dict(reader=args.reader, report=str(destination), actual_native_count=8192,
               training_modified=False, policy_accepted=False))
