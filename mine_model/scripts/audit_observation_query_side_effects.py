"""Reproduce observation-query mutations using the actual task methods on CPU.

This does not start Isaac, advance physics, train PPO or modify task code.
"""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch

ROOT = Path('/home/aaa/studyRL/src/CLT-RL')
ENV = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
RUNNER = Path('/home/aaa/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py')
WRAPPER = ROOT / 'IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py'
BASE = ROOT / 'IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py'
OBSERVER = ROOT / 'mine_model/scripts/joint_pd_training_observer.py'
OUT = ROOT / 'mine_model/results/joint_pd_body_plus1p3kg_20261026/observation_query_audit_20261026'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(path, name):
    tree = ast.parse(path.read_text())
    found = [n for c in tree.body if isinstance(c, ast.ClassDef)
             for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise AssertionError((path, name, len(found)))
    return found[0]


def observation_calls(tree):
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ('get_observations', '_get_observations')]


namespace = {'torch': torch}
for name in ('_get_observations', '_update_commands'):
    node = copy.deepcopy(method(ENV, name))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(ENV), 'exec'), namespace)


def fixture(*, grouped=True):
    env = SimpleNamespace(
        cfg=SimpleNamespace(commands=SimpleNamespace(
            linear_acceleration_limit=.8, heading_command=False,
            grouped_training=grouped, ranges_lin_vel_x=(-.8, .8), resampling_time=5.),
            obs_clip=SimpleNamespace(ang_vel=100., projected_gravity=1., lin_vel=100.,
                                     wheel_vel=100., action=100.),
            obs_scales=SimpleNamespace(ang_vel=1., lin_vel=1., dof_vel=1.)),
        step_dt=.02, num_envs=2, device='cpu', common_step_counter=17,
        _last_grouped_command_phase=0, episode_length_buf=torch.tensor([0, 17]),
        _commands=torch.zeros(2, 4), _target_lin_vel_x=torch.tensor([.8, -.8]),
        _actions=torch.full((2, 6), .2), _last_actions=torch.zeros(2, 6, 2),
        _commands_scale=torch.ones(3), _tracking_diagnostics={}, extras={'log': {}},
        resamples=[],
    )
    env._last_actions[:, :, 0] = .1
    env._last_actions[:, :, 1] = 0.
    env._robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1)))
    env._sanitize_observation = lambda value, limit: value.clamp(-limit, limit)
    env._get_heading_frame_horizontal_velocity = lambda: torch.zeros(2, 2)
    env._get_wheel_vel_forward_positive = lambda: torch.zeros(2, 2)
    env._get_leg_observation = lambda: torch.zeros(2, 8)
    env._resample_commands_for = lambda ids: env.resamples.append(ids.tolist())
    for name in ('_get_observations', '_update_commands'):
        setattr(env, name, MethodType(namespace[name], env))
    return env


def snapshot(env):
    return {'clock': env.common_step_counter, 'commands': env._commands[:, 0].tolist(),
            'history': env._last_actions[0, 0].tolist(),
            'resample_calls': len(env.resamples),
            'next_action_smooth_if_action_is_0p3': float((
                .3 - 2 * env._last_actions[0, 0, 0]
                + env._last_actions[0, 0, 1]).square())}


def main():
    env = fixture()
    rows = [snapshot(env)]
    for _ in range(2):
        obs = env._get_observations()['policy']
        assert obs.shape == (2, 27)
        rows.append(snapshot(env))
    assert rows[0]['clock'] == rows[1]['clock'] == rows[2]['clock'] == 17
    assert torch.allclose(torch.tensor(rows[1]['commands']), torch.tensor([.016, -.016]))
    assert torch.allclose(torch.tensor(rows[2]['commands']), torch.tensor([.032, -.032]))
    assert rows[1]['resample_calls'] == rows[2]['resample_calls'] == 0
    assert torch.allclose(torch.tensor(rows[1]['history']), torch.tensor([.2, .1]))
    assert torch.allclose(torch.tensor(rows[2]['history']), torch.tensor([.2, .2]))
    assert rows[1]['next_action_smooth_if_action_is_0p3'] < 1.e-12
    assert abs(rows[2]['next_action_smooth_if_action_is_0p3'] - .01) < 1.e-7

    # Grouped phase guard prevents two resamples, but not the ramp/history writes.
    boundary = fixture()
    boundary.common_step_counter = 250
    boundary._get_observations()
    boundary._get_observations()
    assert len(boundary.resamples) == 1

    # Non-grouped modulo sampling has no per-step read guard.
    ordinary = fixture(grouped=False)
    ordinary._get_observations()
    ordinary._get_observations()
    assert ordinary.resamples == [[0], [0]]

    init_calls = observation_calls(method(RUNNER, '__init__'))
    learn_calls = observation_calls(method(RUNNER, 'learn'))
    assert len(init_calls) == len(learn_calls) == 1
    learn = method(RUNNER, 'learn')
    loops = [n for n in ast.walk(learn) if isinstance(n, (ast.For, ast.While))]
    assert not any(observation_calls(n) for n in loops)
    assert not observation_calls(ast.parse(OBSERVER.read_text()))
    assert len(observation_calls(method(WRAPPER, 'get_observations'))) == 1
    assert not observation_calls(method(WRAPPER, 'step'))
    assert len(observation_calls(method(BASE, 'step'))) == 1

    result = dict(
        verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        actual_task_methods_extracted_without_rewriting=True,
        CPU_fixture_only=True, physics_started=False, optimizer_executed=False,
        task_or_library_code_modified=False, hardware_enabled=False,
        repeated_grouped_observation_reads=rows,
        grouped_boundary_resamples_for_two_reads=len(boundary.resamples),
        non_grouped_modulo_boundary_resamples_for_two_reads=ordinary.resamples,
        normal_training_lifecycle=dict(
            wrapper_constructor_calls_reset=True,
            runner_constructor_observation_query_lines=init_calls,
            learn_start_observation_query_lines=learn_calls,
            learn_rollout_loop_extra_observation_queries=0,
            direct_env_step_observation_queries=1,
            active_training_observer_extra_observation_queries=0),
        interpretation=(
            'The current callback is stateful. Repeating it at one clock advances the '
            'linear ramp and overwrites second action history. Current grouped sampling '
            'is phase guarded. Native RSL rollout uses env.step returned observations; '
            'its constructor/learn startup queries must not be counted as physics steps. '
            'This fixture does not establish that this behavior caused steady yaw error.'),
        source_pins={str(p): sha(p) for p in (ENV, RUNNER, WRAPPER, BASE, OBSERVER, Path(__file__))})
    OUT.mkdir(exist_ok=False)
    (OUT / 'observation_query_audit.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('verified', 'repeated_grouped_observation_reads',
        'grouped_boundary_resamples_for_two_reads', 'non_grouped_modulo_boundary_resamples_for_two_reads',
        'normal_training_lifecycle')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
