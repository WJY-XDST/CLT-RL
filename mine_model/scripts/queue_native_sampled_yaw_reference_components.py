"""Wait for an owned completed200 learning audit, then read its saved data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--monitor-unit', required=True)
    cli.add_argument('--reader-sha256', required=True)
    args = cli.parse_args()
    directory = args.output.resolve()
    reader = Path(__file__).with_name('audit_native_sampled_yaw_reference_components.py')
    assert hashlib.sha256(reader.read_bytes()).hexdigest() == args.reader_sha256
    result = directory / 'sampled_yaw_reference_components_prefix_0200_readonly.json'
    while True:
        status = subprocess.check_output(['systemctl', '--user', 'show', args.monitor_unit,
            '--property=MainPID,SubState,Result'], text=True)
        owner = dict(line.split('=', 1) for line in status.splitlines() if '=' in line)
        if result.exists():
            record = json.loads(result.read_text())
            assert record['verified'] and record['committed_prefix'] == 200
            assert record['source_pins'][str(reader.resolve())] == args.reader_sha256
            break
        readiness = [directory / (stem + '_prefix_0200_readonly.json') for stem in
                     ('learning_health_wheel005_D100', 'yaw_objective_reader_binding')]
        if all(path.exists() for path in readiness):
            assert all(json.loads(path.read_text()).get(key) for path, key in
                       zip(readiness, ('qualified', 'original_reader_result_verified')))
            assert hashlib.sha256(reader.read_bytes()).hexdigest() == args.reader_sha256
            os.execv(sys.executable, [sys.executable, str(reader), str(directory), '--through', '200'])
        assert owner.get('MainPID') != '0', ('Inspect the owned learning monitor', owner)
        time.sleep(15.)
