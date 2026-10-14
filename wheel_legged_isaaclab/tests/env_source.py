"""Inspect production methods without starting Isaac Sim or replacing equations."""
import ast
from pathlib import Path


ENV_DIRECTORY = Path(__file__).resolve().parents[1] / (
    'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat'
)
ENV_FILES = ('wheel_legged_vmc_flat_env.py', 'env_commands.py')


def environment_tree():
    """Return every real environment component, preserving method decorators."""
    return ast.Module(
        body=[node for name in ENV_FILES
              for node in ast.parse((ENV_DIRECTORY / name).read_text()).body],
        type_ignores=[],
    )


def environment_source():
    return '\n'.join((ENV_DIRECTORY / name).read_text() for name in ENV_FILES)
