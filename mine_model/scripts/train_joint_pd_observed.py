"""Run the original 100-update PPO continuation with read-only instrumentation.

Keep actions, rewards, gradients, checkpoints, optimizer and RNG semantics.
Collect native transition alignment and fixed-input update diagnostics only.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from rsl_rl.runners import OnPolicyRunner

from auto_train_mine import write_json
from audit_joint_pd_native_training import run_original_training
from diagnose_plain_joint_pd_height import geometry, require_plain_ppo
from joint_pd_continuation import ROOT, sha
from joint_pd_training_observer import TrainingObserver
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


def continuation_source(arguments):
    if any(flag in arguments for flag in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Retain the current policy and complete optimizer')
    source = Path(arguments[arguments.index('--checkpoint_path')+1]).resolve()
    target = int(arguments[arguments.index('--target_iteration')+1])
    actual = saved_updates(source)
    if target-actual != 100:
        raise ValueError('Observe exactly 100 actual PPO updates between full reviews')
    cfg = validate_checkpoint_robot(source)
    if cfg['leg_control_mode'] != 'implicit_joint_position':
        raise ValueError('Observe direct joint-position PD only')
    return source, actual, target


def run(output, arguments):
    import math
    import yaml
    output = Path(output).resolve()
    if (output.name != 'native_learning_signal' or output.parent.parent != ROOT/'mine_model/results'
            or not output.parent.name.startswith('joint_pd_optimization_')):
        raise ValueError('Use a fresh observation directory under the owned native trial')
    source, actual, target = continuation_source(arguments)
    require_plain_ppo(yaml.safe_load((source.parent/'params/agent.yaml').read_text()))
    source_sha = sha(source)
    output.mkdir(exist_ok=False)
    record = dict(source_checkpoint=str(source), source_sha256=source_sha,
        source_actual_ppo_updates=actual, target_actual_ppo_updates=target,
        requested_updates=100, original_training_arguments=arguments,
        read_only_instrumentation=True, added_loss_or_backward=False,
        actions_or_rewards_changed=False, optimizer_reset=False, policy_accepted=False,
        source_files_sha256={str(Path(p).resolve()): sha(p)
                             for p in (__file__, sys.modules[TrainingObserver.__module__].__file__)})
    write_json(output/'plan.json', record)
    observers = []
    original_init = OnPolicyRunner.__init__

    def init(runner, env, *args, **kwargs):
        original_init(runner, env, *args, **kwargs)
        raw = env.unwrapped
        cfg = raw.cfg.to_dict()
        limits = raw._robot.data.joint_pos_limits[0, raw._leg_joint_ids].cpu().tolist()
        observers.append(TrainingObserver(runner.alg, geometry(cfg, limits),
                                         env=raw, cfg=cfg, output=output))

    try:
        OnPolicyRunner.__init__ = init
        run_original_training(arguments)
        if len(observers) != 1 or len(observers[0].updates) != 100:
            raise ValueError('The original runner did not complete the declared 100 observed updates')
        updates = observers[0].updates
        errors = {key: max(row['all_environment_alignment_max_errors'][key] for row in updates)
                  for key in updates[0]['all_environment_alignment_max_errors']}
        if (not errors or any(not math.isfinite(v) or v > 5.e-6 for v in errors.values())
                or sha(source) != source_sha or any(not row['probe_RNG_unchanged'] for row in updates)
                or any(len(row['optimizer_steps']) != 20 for row in updates)):
            raise ValueError('Native storage, RNG or actual optimizer-clock alignment needs diagnosis')
        write_json(output/'completed.json', dict(record, completed=True, actual_observed_updates=100,
            all_environment_alignment_max_errors=errors, optimizer_minibatches=2000,
            all_probe_RNG_unchanged=True, native_control_or_loss_changed=False,
            updated_at=datetime.now().astimezone().isoformat(),
            limitation='Learning-signal diagnosis only; independent full tracking replay is still required'))
    except BaseException as error:
        write_json(output/'diagnosis_required.json', dict(record, reason=repr(error),
            observed_updates=sum(len(value.updates) for value in observers),
            all_native_checkpoints_preserved=True, updated_at=datetime.now().astimezone().isoformat()))
        raise
    finally:
        OnPolicyRunner.__init__ = original_init
        for observer in observers:
            observer.close()


if __name__ == '__main__':
    if '--' not in sys.argv:
        raise ValueError('Separate observer options from unchanged original training arguments with --')
    boundary = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-output', type=Path, required=True)
    args = parser.parse_args(sys.argv[1:boundary])
    run(args.observation_output, sys.argv[boundary+1:])
