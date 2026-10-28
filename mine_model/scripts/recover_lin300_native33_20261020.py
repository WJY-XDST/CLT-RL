"""Resume only the 33 unsaved updates of the shutdown-interrupted lin300 block.

The original 167 checkpoints/rollouts remain immutable. This changes only the
process-local observer counter and output paths, not ordinary PPO or control.
"""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
RESULTS = ROOT / 'mine_model/results'
OLD = RESULTS / 'joint_pd_optimization_20261020_195952_083563_source6201_lin300_200'
OUTPUT = RESULTS / 'joint_pd_optimization_20261020_recovery_source6368_lin300_remaining33'
QUALIFICATION = RESULTS / 'joint_pd_lin300_recovery_qualification_20261020'
UNIT = 'clt-joint-pd-lin300-recovery6368-6401-20261020.service'
RETAINED, REMAINING, START, RESUME, TARGET = 167, 33, 6201, 6368, 6401


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(text)
        os.replace(temporary, path)


def intact(pins):
    changed = [p for p, h in pins.items() if digest(p) != h]
    if changed:
        raise ValueError('Retained recovery inputs changed: ' + repr(changed[:8]))


def api():
    import joint_pd_source6201_lin300_runtime as runtime
    from train_joint_pd_standing_seed import saved_updates
    return runtime, saved_updates


def records(directory, count, start):
    import numpy as np
    runtime, saved_updates = api()
    original = read(directory / 'experiment_plan.json')
    runs = list((ROOT / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat').glob(
        '*_' + original['run_name']))
    if len(runs) != 1:
        raise ValueError('Require the unique native run for this exact segment')
    run = runs[0]
    if len(list(run.glob('model_*.pt'))) != count:
        raise ValueError('Inspect an unexpected native checkpoint count')
    native = directory / 'native_learning_signal'
    if (len(list(native.glob('audit_update_*.json'))) != count
            or len(list(native.glob('rollout_*.npz'))) != count):
        raise ValueError('Inspect any longer/shorter native segment before recovery')
    rows, pins = [], {}
    for index in range(1, count + 1):
        actual = start + index
        checkpoint = run / f'model_{actual - 1}.pt'
        saved, cfg = runtime.require_source(checkpoint)
        states = saved['optimizer_state_dict']['state']
        if (saved_updates(checkpoint) != actual or len(states) != 17
                or {float(s['step']) for s in states.values()} != {float(actual * 20)}):
            raise ValueError('An exact finite17 Adam checkpoint clock was lost')
        audit = native / f'audit_update_{index:04d}.json'
        archive = native / f'rollout_{index:04d}.npz'
        row = read(audit)
        errors = row['all_environment_alignment_max_errors']
        if (row['update'] != index or row['total_environments'] != 4096
                or len(row['optimizer_steps']) != 20 or len(errors) != 9 or any(errors.values())
                or not row['probe_RNG_unchanged'] or row['added_loss_or_backward']):
            raise ValueError('A retained full4096 native alignment failed')
        with np.load(archive, allow_pickle=False) as arrays:
            if (arrays['observations'].shape != (48, 32, 27)
                    or arrays['actions'].shape != (48, 32, 6)
                    or any(not np.isfinite(arrays[key]).all() for key in arrays.files)
                    or not np.array_equal(arrays['actions'], arrays['native_raw_actions'])):
                raise ValueError('A retained archive is incomplete/nonfinite or changes native actions')
        paths = (checkpoint, audit, archive)
        pins.update({str(p): digest(p) for p in paths})
        rows.append(dict(actual_update=actual, all17_Adam_step=actual * 20,
            segment_local_update=index, checkpoint=str(checkpoint), audit=str(audit), archive=str(archive)))
    return rows, pins, run


def qualify():
    import contextlib
    import io
    from types import SimpleNamespace
    import torch
    import yaml
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    import rsl_rl.runners as runners
    import joint_pd_review_interval200 as interval
    import train_joint_pd_observed as observed
    runtime, saved_updates = api()
    if OUTPUT.exists() or QUALIFICATION.exists():
        raise ValueError('Preserve an existing recovery qualification/segment')
    original = read(OLD / 'experiment_plan.json')
    intact(original['source_files'])
    rows, pins, run = records(OLD, RETAINED, START)
    source = Path(rows[-1]['checkpoint'])
    saved, cfg = runtime.require_source(source)
    if saved_updates(source) != RESUME or saved['iter'] != RESUME - 1:
        raise ValueError('Resume only the latest completed native update6368')
    QUALIFICATION.mkdir()
    agent = yaml.safe_load((run / 'params/agent.yaml').read_text())
    options = copy.deepcopy(agent['policy'])
    if options.pop('class_name') != 'ActorCritic':
        raise ValueError('Keep plain27 ActorCritic')
    obs = TensorDict({'policy': torch.linspace(-.2, .2, 32 * 27).reshape(32, 27)}, batch_size=[32])
    obs['policy'][:, 5] = -1.
    obs['policy'][:, 8] = 1.5

    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(obs, agent['obs_groups'], 6, **options)
        runner = object.__new__(cls)
        runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
        runner.current_learning_iteration = 0
        runner.logger_type, runner.disable_logs = 'tensorboard', True
        return runner

    load_dir = QUALIFICATION / 'real_runner_CPU_load'
    load_dir.mkdir()
    upstream, cls = runtime.private_runtime().install_runner(load_dir)
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'recovery_model')
        runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'recovery_Adam')
        if runner.current_learning_iteration != saved['iter']:
            raise ValueError('Preserve saved iteration indexing; original train.py normalizes actual clock')
        reference = bare(cls)
        reference.alg.policy.load_state_dict(saved['model_state_dict'])
        with torch.no_grad():
            a, b = runner.alg.policy.act_inference(obs), reference.alg.policy.act_inference(obs)
            if not torch.isfinite(a).all() or not torch.equal(a, b):
                raise ValueError('Loaded deterministic CPU inference differs')
    finally:
        runners.OnPolicyRunner = upstream
    argv = read(OLD / 'launch.json')['command'][:]
    argv[1] = str(SCRIPT)
    argv[argv.index('--observation-output') + 1] = str(OUTPUT / 'native_learning_signal')
    argv[argv.index('--checkpoint_path') + 1] = str(source)
    argv[argv.index('--run_name') + 1] = 'plain_lin300_recovered6368_' + OUTPUT.name
    argv[argv.index('--stop_file') + 1] = str(OUTPUT / 'STOP')
    argv.remove(runtime.FLAG)
    old_interval = interval.UPDATES
    try:
        interval.UPDATES = REMAINING
        adapter = interval.install_observer(observed)
        arguments = argv[argv.index('--') + 1:]
        if observed.continuation_source(arguments) != (source, RESUME, TARGET):
            raise ValueError('Exact remaining33 source/target admission failed')
        wrong = list(arguments)
        wrong[wrong.index('--target_iteration') + 1] = str(TARGET - 1)
        try:
            observed.continuation_source(wrong)
        except ValueError:
            pass
        else:
            raise ValueError('A wrong remaining counter was admitted')
    finally:
        interval.UPDATES = old_interval
    pins.update(original['source_files'])
    pins.update({str(p): digest(p) for p in (SCRIPT, OLD / 'experiment_plan.json',
        OLD / 'launch.json', OLD / 'native_learning_signal/plan.json',
        *(run / 'params').glob('*.yaml'))})
    proof = dict(qualified=True, created_at=datetime.now().astimezone().isoformat(),
        resume_checkpoint=str(source), resume_checkpoint_sha256=digest(source),
        source_actual_updates=RESUME, source_saved_iter_index=RESUME - 1,
        source_all17_Adam_step=RESUME * 20, retained_original_updates=RETAINED,
        remaining_updates=REMAINING, target_actual_updates=TARGET, target_all17_Adam_step=TARGET * 20,
        source_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(),
        all167_checkpoint_clocks_finite_and_continuous=True, all167_native_alignment_errors_zero=True,
        all_model_all17_Adam_actual_LR_and_std_bitwise_preserved_on_real_runner_load=True,
        deterministic_CPU_inference_bitwise_preserved=True, wrong_counter_rejected=True,
        observer_interval_adapter=adapter, checkpoint_rows=rows, source_pins=pins,
        PPO_updates_performed=0, simulator_started=False, simulator_and_RNG_state_restored=False,
        interrupted_by_OS_shutdown=True, new_simulator_episode_initialization_required=True,
        policy_accepted=False)
    save(QUALIFICATION / 'qualification.json', proof, exclusive=True)
    OUTPUT.mkdir()
    pins = dict(pins, **{str(QUALIFICATION / 'qualification.json'): digest(QUALIFICATION / 'qualification.json')})
    plan = dict(proof)
    plan.update(source_pins=pins, recovery_entry_sha256=digest(SCRIPT), native_command=argv,
        unit=UNIT, original_output=str(OLD), original_source_actual_updates=START,
        planned_actual_updates=REMAINING, original_block_total_updates=200,
        run_name=argv[argv.index('--run_name') + 1], source_checkpoint=str(source),
        source_sha256=digest(source), source_actual_ppo_updates=RESUME,
        source_files=pins, training_config=cfg, policy_accepted=False)
    save(OUTPUT / 'recovery_plan.json', plan, exclusive=True)
    save(OUTPUT / 'experiment_plan.json', plan, exclusive=True)
    save(OUTPUT / 'launch.json', dict(command=argv, source_sha256=digest(source)), exclusive=True)
    print(json.dumps({k: v for k, v in proof.items() if k not in ('source_pins', 'checkpoint_rows')}, ensure_ascii=False), flush=True)
    return proof


def native(output, arguments):
    import joint_pd_review_interval200 as interval
    runtime, saved_updates = api()
    plan = read(OUTPUT / 'recovery_plan.json')
    intact(plan['source_pins'])
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    saved, cfg = runtime.require_source(source)
    if (Path(output) != OUTPUT / 'native_learning_signal' or saved_updates(source) != RESUME
            or digest(source) != plan['resume_checkpoint_sha256'] or runtime.FLAG in arguments
            or int(arguments[arguments.index('--target_iteration') + 1]) != TARGET
            or arguments != plan['native_command'][plan['native_command'].index('--') + 1:]
            or {float(s['step']) for s in saved['optimizer_state_dict']['state'].values()} != {float(RESUME * 20)}):
        raise ValueError('Only the qualified retained-state6368 to6401 remainder may run')
    interval.UPDATES = REMAINING
    runtime.native(Path(output), arguments)


def controller():
    plan = read(OUTPUT / 'recovery_plan.json')
    intact(plan['source_pins'])
    if (digest(SCRIPT) != plan['recovery_entry_sha256'] or (OUTPUT / 'training.log').exists()):
        raise ValueError('Preserve the qualified entry and any existing native attempt')
    import train_source200_drift25_measured_memory200 as memory
    memory.wait_memory(OUTPUT)
    status = dict(state='training_initialization', training_active=True,
        source_actual_updates=RESUME, live_ppo_updates=RESUME, retained_original_block_updates=RETAINED,
        segment_updates=REMAINING, target_actual_ppo_updates=TARGET, checkpoint=plan['resume_checkpoint'],
        policy_accepted=False, goal_remains_active=True)
    save(OUTPUT / 'status.json', status)
    log_path = OUTPUT / 'training.log'
    with log_path.open('x') as stream:
        env = dict(os.environ, PYTHONPATH=':'.join(str(ROOT / p) for p in
            ('wheel_legged_isaaclab', 'mine_model/scripts', 'mine_model/experiments')),
            PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        child = subprocess.Popen(plan['native_command'], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status['child_pid'] = child.pid
        save(OUTPUT / 'status.json', status)
        while child.poll() is None:
            contents = log_path.read_text(errors='replace')
            if 'Assertion Failed!' in contents or 'Recursion not allowed' in contents:
                status.update(state='native_assertion_requires_diagnosis', training_active=False)
                save(OUTPUT / 'status.json', status)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise RuntimeError('Native assertion preserved for diagnosis, not ignored')
            iterations = re.findall(r'Learning iteration\s+(\d+)/(\d+)', contents)
            if iterations:
                actual = int(iterations[-1][0]) + 1
                if not RESUME < actual <= TARGET:
                    raise ValueError('Unexpected native learner clock')
                status.update(state='training', live_ppo_updates=actual,
                    retained_original_block_updates=actual - START,
                    updated_at=datetime.now().astimezone().isoformat())
                save(OUTPUT / 'status.json', status)
            time.sleep(2.)
        code = child.returncode
    status.update(training_active=False, child_exit_code=code,
        state='segment_native_exited' if code == 0 else 'segment_exit_requires_diagnosis',
        updated_at=datetime.now().astimezone().isoformat())
    save(OUTPUT / 'status.json', status)
    if code:
        raise RuntimeError('Native recovery exit code ' + str(code))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'qualify':
        qualify()
    elif mode == 'controller':
        controller()
    elif mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[2:boundary] != ['--observation-output', sys.argv[3]]:
            raise ValueError('Keep the exact native output contract')
        native(Path(sys.argv[3]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Explicit qualification, controller or native mode required')
