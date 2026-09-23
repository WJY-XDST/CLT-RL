"""Wheel-Legged-Gym (Isaac Lab version).

A migration of the wheel-legged robot reinforcement learning environments from
the legacy Isaac Gym (Preview 4) based project to Isaac Lab (Isaac Sim 5.1).
"""

import os

# Root directory of the project
WHEEL_LEGGED_GYM_ISAACLAB_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")
)

__version__ = "0.1.0"
