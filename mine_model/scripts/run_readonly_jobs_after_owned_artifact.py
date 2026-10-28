"""Wait without Torch/Kit, then run pinned read-only review commands.

This workflow helper never starts, stops, or restarts PPO. The native owner
and a completed artifact govern the wait; a missing artifact after a terminal
owner requires inspection. Review qualification remains the reviewer's job.
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess

import watch_uniform_diagnostics_and_archive as lightweight


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def run(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = lightweight.read(spec_path)
    assert spec['orchestration_verified'] and spec['read_only_jobs_only']
    assert not spec['can_start_or_restart_PPO']
    digest = lightweight.sha(spec_path)
    status = spec_path.with_name(spec_path.stem + '_status.json')

    def unchanged():
        assert lightweight.sha(spec_path) == digest
        assert all(lightweight.sha(p) == v for p, v in spec['source_pins'].items())

    def check(report):
        value = lightweight.read(report['path'])
        for key, expected in report['required_fields'].items():
            assert value[key] == expected, (report['path'], key, value[key], expected)

    unchanged()
    for gate in spec['gate_reports']:
        lightweight.wait_for(Path(gate['path']), spec['after_owned_unit'], status,
                             'waiting_for_completed_owned_review_artifact')
        unchanged()
        check(gate)
    completed = []
    for job in spec['jobs']:
        unchanged()
        command = job['command']
        assert Path(command[0]).resolve() == ROOT / 'run_python.sh'
        entry = Path(command[1]).resolve()
        assert entry.parent == ROOT / 'mine_model/scripts'
        assert spec['source_pins'][str(entry)] == lightweight.sha(entry)
        assert job['training_modified'] is False
        with Path(job['log']).open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for report in job['required_reports']:
            check(report)
        completed.append(job['name'])
        lightweight.write(status, dict(phase='readonly_job_completed', completed_jobs=completed,
            observed_at=datetime.now().astimezone().isoformat(), training_modified=False,
            optimizer_executed_here=False, PPO_restarted=False, policy_accepted=False))
    unchanged()
    lightweight.write(spec_path.with_name(spec_path.stem + '_complete.json'), dict(
        complete=True, completed_jobs=completed, source_spec_sha256=digest,
        training_modified=False, optimizer_executed_here=False,
        PPO_restarted=False, policy_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.spec)
    except BaseException as error:
        lightweight.write(args.spec.with_name(args.spec.stem + '_inspection_required.json'), dict(
            error=repr(error), training_modified=False, PPO_restarted=False, policy_accepted=False))
        raise
