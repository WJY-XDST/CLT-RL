"""Select a pinned experiment runtime for the unchanged original735 reader.

Only the runtime import is selected in this isolated process. The existing
scope builder, metrics, cases, seeds and replay controller remain unchanged.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runtime-module', required=True)
    own, arguments = parser.parse_known_args()
    assert own.runtime_module.startswith('joint_pd_') and own.runtime_module.endswith('_runtime')
    assert own.runtime_module.replace('_', '').isalnum()
    spec = importlib.util.find_spec(own.runtime_module)
    assert spec is not None and Path(spec.origin).resolve().parent == ROOT / 'mine_model/experiments'
    selected = importlib.import_module(own.runtime_module)
    key = 'joint_pd_calibrated_common_noise_runtime'
    previous = sys.modules.get(key)
    sys.modules[key] = selected
    helper = ROOT / 'mine_model/scripts/review_saved_joint_pd_same_scope.py'
    previous_argv = sys.argv
    try:
        sys.argv = [str(helper), *arguments]
        runpy.run_path(str(helper), run_name='__main__')
        if arguments[0] == 'qualify':
            output = Path(arguments[arguments.index('--output') + 1]).resolve()
            path = output / 'qualification.json'
            record = json.loads(path.read_text())
            assert record['qualified'] and record['trial_count'] == 735
            own_path = Path(__file__).resolve()
            record['source_pins'][str(own_path)] = hashlib.sha256(own_path.read_bytes()).hexdigest()
            record['selected_runtime_module'] = own.runtime_module
            record['runtime_import_selection_only_original_reader_unchanged'] = True
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            temporary.replace(path)
    finally:
        sys.argv = previous_argv
        if previous is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = previous


if __name__ == '__main__':
    main()
