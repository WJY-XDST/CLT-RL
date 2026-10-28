"""Reuse the owned-service reviewer with a selected pinned replay runtime.

Only qualify/run command prefixes are routed through the runtime selector.
The service checks, cases, seed contracts, metrics and completion checks are
the unchanged original implementations. This has no training entry point.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


class RuntimeScopedSubprocess:
    def __init__(self, delegate, runtime_name):
        self.delegate = delegate
        self.runtime_name = runtime_name

    def routed(self, command):
        helper = ROOT / 'mine_model/scripts/review_saved_joint_pd_same_scope.py'
        if (isinstance(command, (list, tuple)) and len(command) > 2
                and command[0] == str(ROOT / 'run_python.sh') and command[1] == str(helper)):
            assert command[2] in ('qualify', 'run')
            return [command[0], str(ROOT / 'mine_model/scripts/review_saved_joint_pd_runtime_scope.py'),
                    '--runtime-module', self.runtime_name, *command[2:]]
        return command

    def run(self, command, *args, **kwargs):
        return self.delegate.run(self.routed(command), *args, **kwargs)


def run(request_path):
    request = json.loads(request_path.read_text())
    name = request['runtime_module']
    assert name.startswith('joint_pd_') and name.endswith('_runtime')
    assert name.replace('_', '').isalnum()
    source = ROOT / 'mine_model/scripts/review_joint_pd_scope_after_owned_service.py'
    spec = importlib.util.spec_from_file_location('_selected_runtime_original_owned_reviewer', source)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    api.subprocess = RuntimeScopedSubprocess(subprocess, name)
    return api.run(request_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.request.resolve())
    except BaseException as error:
        import review_joint_pd_scope_after_owned_service as original
        request = original.read(args.request)
        original.write(Path(request['status_path']), dict(
            state='inspection_required_no_automatic_restart', error=repr(error),
            upstream_unit=request['upstream_unit'], checkpoints_preserved=True,
            optimizer_executed=False, policy_accepted=False))
        raise
