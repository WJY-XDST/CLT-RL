"""Validate the canonical source-record contract of the original target reader.

This is a preflight for post-training diagnostic input files, never a selector
for training conditions or a substitute for the full735 acceptance matrix.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def validate(record, checkpoint):
    record, checkpoint = Path(record).resolve(), Path(checkpoint).resolve()
    manifest = json.loads((record / 'manifest.json').read_text())
    complete = json.loads((record / 'completed.json').read_text())
    differences = {}
    checks = {
        'completed': (complete['completed'], True),
        'native_samples': (complete['samples'], 27000),
        'resets': (complete['resets'], 0),
        'cases': (len(manifest['cases']), 9),
        'seeds': (manifest['seeds'], [83, 84, 85]),
        'seconds_per_case': (manifest['seconds_per_case'], 20.),
        'policy_dt': (manifest['policy_dt'], .02),
        'weights_updated': (manifest['weights_updated'], False),
        'sampled_policy_actions': (manifest['sampled_policy_actions'], False),
        'policy_input_remapping': (manifest['policy_input_remapping'], False),
        'command_scale': (manifest['command_scale'], 1.),
        'checkpoint': (Path(manifest['checkpoint']).resolve(), checkpoint),
        'checkpoint_sha256': (manifest['checkpoint_sha256'], sha(checkpoint)),
    }
    for name, (actual, required) in checks.items():
        if actual != required:
            differences[name] = dict(actual=str(actual) if isinstance(actual, Path) else actual,
                                     required=str(required) if isinstance(required, Path) else required)
    if differences:
        raise ValueError(json.dumps(dict(source_record_contract_mismatch=differences)))
    with np.load(record / 'policy_inputs.npz', allow_pickle=False) as archive:
        observations, means = archive['observations'], archive['means']
        if (observations.shape != (1000, 27, 27) or means.shape != (1000, 27, 6)
                or not np.isfinite(observations).all() or not np.isfinite(means).all()):
            raise ValueError('Original canonical27 input/output timeline is incomplete or nonfinite')
    files = [record / name for name in ['manifest.json', 'completed.json', 'policy_inputs.npz', 'trace.csv']]
    return dict(verified=True, comparison_source_record=str(record), checkpoint=str(checkpoint),
                checkpoint_sha256=manifest['checkpoint_sha256'], canonical_trials=27,
                original_reader_shape_seed_time_and_zero_reset_contract_verified=True,
                original_numeric_reader_not_modified=True,
                source_pins={str(p): sha(p) for p in files + [Path(__file__), checkpoint]},
                simulator_started=False, optimizer_executed=False, policy_accepted=False)
