"""Tasks package for wheel-legged robot environments.

Import direct tasks eagerly so their ``gym.register`` calls execute when the
external extension is imported by Isaac Lab training scripts.
"""

from . import direct  # noqa: F401
