"""A continuous command normalization for an existing tracking kernel.

This module has no environment installation or training entry point. Callers
choose the global command cap and one positive floor fraction explicitly.
There is no height, velocity-sign, task-group or evaluation-case branch.
"""
import math


def validate_parameters(nominal_variance, command_cap, floor_fraction):
    values = (nominal_variance, command_cap, floor_fraction)
    if any(not math.isfinite(float(value)) or float(value) <= 0. for value in values):
        raise ValueError('Kernel variance, command cap and floor fraction must be finite and positive')
    if floor_fraction > 1.:
        raise ValueError('Use a floor fraction within (0,1]')


def command_relative_variance(command, nominal_variance, command_cap, floor_fraction):
    """Return a positive, sign-symmetric variance with its original cap value.

    `command` may be a scalar or a tensor; validation belongs before rollout.
    At |command|=command_cap the old variance is retained. At zero the floor
    prevents division by zero and gives a finite absolute tracking tolerance.
    The function does not read an acceptance threshold or any replay labels.
    """
    floor2 = (command_cap * floor_fraction) ** 2
    return nominal_variance * (command * command + floor2) / (command_cap ** 2 + floor2)
