"""Compare actual new-model training throughput under a bounded memory budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--overrides', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--counts', nargs='+', type=int, default=[1024, 2048, 4096])
    p.add_argument('--iterations', type=int, default=5)
    args = p.parse_args()
    if min(args.counts) <= 0 or args.iterations < 3:
        raise ValueError('Positive counts and at least three updates required')
    import torch
    checkpoint = args.checkpoint.resolve()
    iteration = int(torch.load(checkpoint, map_location='cpu', weights_only=True)['iter'])
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(args.overrides.read_text())
    import sys
    sys.path.insert(0, str(ROOT/'wheel_legged_isaaclab'))
    from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain
    asset_chain, _ = verify_asset_chain(overrides['robot.spawn.usd_path'])
    (output/'overrides.json').write_text(json.dumps(overrides, indent=2)+'\n')
    results = []
    for count in args.counts:
        folder = output/str(count); folder.mkdir(exist_ok=True)
        unit = f'mine-scale-{count}-{int(time.time())}'
        run_name = f'mine_scale_benchmark_{count}_{output.name}'
        train = [str(ROOT/'run_python.sh'), 'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
                 '--task', 'WheelLeggedVMC-Flat-v0', '--headless', '--num_envs', str(count),
                 '--seed', '43', '--run_name', run_name, '--checkpoint_path', str(checkpoint),
                 '--target_iteration', str(iteration+args.iterations), '--wheel_head_only_finetune',
                 'agent.algorithm.entropy_coef=0.001']
        train += ['env.'+k+'='+json.dumps(v) for k,v in overrides.items()]
        command = ['systemd-run', '--user', '--wait', '--pipe', '--quiet', '--collect',
                   '--unit', unit, '--property=MemoryMax=22G', '--property=MemorySwapMax=1G',
                   '--property=RuntimeMaxSec=600', '--working-directory='+str(ROOT),
                   '--setenv=PYTHONUNBUFFERED=1', '--setenv=PYTHONPATH='+str(ROOT/'wheel_legged_isaaclab'),
                   *train]
        record = {'num_envs': count, 'source_checkpoint': str(checkpoint),
                  'asset_chain_sha256': asset_chain,
                  'source_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'command': command, 'unit': unit, 'status': 'running',
                  'memory_peak_bytes': 0, 'gpu_memory_peak_mib': 0}
        (folder/'run.json').write_text(json.dumps(record, indent=2)+'\n')
        started = time.monotonic()
        last_update, last_progress = 0, started
        resource_aborted = False
        print(f'SCALE_START {count}', flush=True)
        with (folder/'training.log').open('w') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while child.poll() is None:
                current = subprocess.run(['systemctl', '--user', 'show', unit,
                    '-p', 'MemoryCurrent', '-p', 'MemoryPeak'], capture_output=True, text=True)
                values = re.findall(r'Memory(?:Current|Peak)=(\d+)', current.stdout)
                if values: record['memory_peak_bytes'] = max(record['memory_peak_bytes'], *map(int,values))
                updates = len(re.findall(r'Learning iteration\s+\d+/', (folder/'training.log').read_text(errors='replace')))
                if updates > last_update:
                    last_update, last_progress = updates, time.monotonic()
                available = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text()).group(1))*1024
                if (not resource_aborted and updates > 0 and time.monotonic()-last_progress > 120
                        and available < 1.5*2**30 and record['memory_peak_bytes'] > 21*2**30):
                    record['failure_reason'] = 'memory pressure; no new PPO update for over 120 seconds with under 1.5 GiB system memory available'
                    record['manual_resource_abort'] = True
                    subprocess.run(['systemctl', '--user', 'kill', '--signal=KILL', unit], capture_output=True)
                    resource_aborted = True
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True)
                if gpu.returncode == 0:
                    record['gpu_memory_peak_mib'] = max(record['gpu_memory_peak_mib'], int(gpu.stdout.splitlines()[0]))
                (folder/'run.json').write_text(json.dumps(record, indent=2)+'\n')
                time.sleep(1)
        record['exit_code'] = child.returncode
        record['wall_seconds'] = time.monotonic()-started
        journal = subprocess.run(['journalctl', '--user', '-u', unit, '--no-pager', '-n', '12', '-o', 'cat'],
                                 capture_output=True, text=True).stdout
        (folder/'systemd_journal.log').write_text(journal)
        if 'oom-kill' in journal:
            record['failure_reason'] = 'cgroup out of memory'
        log = (folder/'training.log').read_text(errors='replace')
        computations = re.findall(r'Computation:\s+([\d.]+) steps/s \(collection:\s*([\d.]+)s, learning\s*([\d.]+)s\)', log)
        steady = computations[1:] or computations
        completed = len(re.findall(r'Learning iteration\s+\d+/', log))
        record['completed_iterations'] = completed
        record['finite_training'] = not bool(re.search(r'(?i)(?:loss|tracking/|reward)[^\n]*\b(?:nan|inf)\b|CUDA error|OutOfMemoryError|PhysX error|\[Error\] \[omni\.physx(?:\.tensors)?\.plugin\]', log))
        if steady:
            record['mean_steps_per_second'] = sum(float(v[0]) for v in steady)/len(steady)
            record['mean_collection_seconds'] = sum(float(v[1]) for v in steady)/len(steady)
            record['mean_learning_seconds'] = sum(float(v[2]) for v in steady)/len(steady)
        runs = list((ROOT/'IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat').glob('*_'+run_name))
        saved = [f for r in runs for f in r.glob('model_*.pt') if re.fullmatch(r'model_\d+', f.stem)]
        if saved: record['checkpoint'] = str(max(saved, key=lambda f:int(f.stem.split('_')[1])))
        record['passed'] = bool(child.returncode == 0 and completed >= args.iterations and record['finite_training'] and saved)
        record['status'] = 'completed' if record['passed'] else 'failed'
        (folder/'run.json').write_text(json.dumps(record, indent=2)+'\n')
        results.append(record)
        summary = {'results': results, 'complete': len(results)==len(args.counts),
                   'criterion': f'same source checkpoint, asset chain and settings; {args.iterations} PPO updates per scale; saved checkpoint, no physics/numerical errors and clean exit required; exclude first update from throughput'}
        successful = [r for r in results if r['passed'] and r.get('mean_steps_per_second')]
        if successful:
            best = max(successful, key=lambda r:r['mean_steps_per_second'])
            summary['recommended_num_envs'] = best['num_envs']
            summary['maximum_passed_num_envs'] = max(r['num_envs'] for r in successful)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print('SCALE_RESULT '+json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
