"""Sample only an owned capacity cgroup; never alter training or admission."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import threading
import time


SCRIPT = Path(__file__).resolve()


def install(numeric):
    original = numeric.run_probe

    def run_probe(request, request_path, output, count, policy):
        folder = numeric.ROOT / 'mine_model/results' / (
            'joint_pd_optimization_capacity_' + output.name + '_n' + str(count))
        stop, rows = threading.Event(), []

        def collect():
            while not stop.is_set():
                try:
                    live = json.loads((folder / 'run.json').read_text())
                    control_group = live.get('unit_state', {}).get('ControlGroup')
                    if control_group:
                        path = Path('/sys/fs/cgroup') / control_group.lstrip('/')
                        row = dict(observed_at=datetime.now().astimezone().isoformat(),
                                   monotonic_seconds=time.monotonic(), control_group=control_group,
                                   original_committed_update_files=len(list((folder / 'native_learning_signal').glob('audit_update_*.json'))))
                        for name in ('memory.stat', 'cpu.stat'):
                            row[name] = {key: int(value) for key, value in
                                         (line.split() for line in (path / name).read_text().splitlines())}
                        for name in ('memory.current', 'memory.peak', 'memory.swap.current'):
                            row[name] = int((path / name).read_text())
                        row['memory.pressure'] = (path / 'memory.pressure').read_text()
                        query = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory',
                                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
                        row['GPU_percent_percent_watts_SM_MHz_memory_MHz'] = query.stdout.strip()
                        row['GPU_query_returncode'] = query.returncode
                        rows.append(row)
                        with (folder / 'passive_memory_CPU_GPU_profile.jsonl').open('a') as stream:
                            stream.write(json.dumps(row, allow_nan=False) + '\n')
                except (OSError, ValueError, KeyError):
                    pass  # A cgroup can disappear normally between two samples.
                stop.wait(2.)

        thread = threading.Thread(target=collect, name='owned-capacity-memory-profile', daemon=True)
        thread.start()
        try:
            result = original(request, request_path, output, count, policy)
        finally:
            stop.set()
            thread.join(timeout=10.)
        summary = dict(sample_count=len(rows), sampling_interval_seconds=2.,
                       performance_measurement_includes_passive_observer_overhead=True,
                       original_admission_result_unchanged=True,
                       no_PPO_callback_RNG_action_or_reward_modified=True,
                       no_cpu_policy_cache_or_user_app_changes=True)
        if rows:
            summary.update(peak_memory_current_bytes=max(r['memory.current'] for r in rows),
                           peak_anonymous_bytes=max(r['memory.stat'].get('anon', 0) for r in rows),
                           peak_file_bytes=max(r['memory.stat'].get('file', 0) for r in rows),
                           peak_kernel_bytes=max(r['memory.stat'].get('kernel', 0) for r in rows),
                           peak_shmem_bytes=max(r['memory.stat'].get('shmem', 0) for r in rows),
                           peak_file_mapped_bytes=max(r['memory.stat'].get('file_mapped', 0) for r in rows),
                           caution='Individual component maxima need not occur simultaneously and must not be added blindly.')
        (folder / 'passive_profile_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return result

    numeric.run_probe = run_probe
