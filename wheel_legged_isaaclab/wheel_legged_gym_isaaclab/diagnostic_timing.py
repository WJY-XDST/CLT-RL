"""Timing invariants for simulator convergence experiments (not training)."""


def control_stride(physics_hz: int, control_hz: int, policy_hz: int = 50) -> int:
    """Return the number of physics steps per held VMC command.

    Integer ratios make the comparison independent of floating-point clocks.
    Changing physics_hz must not silently change either controller period.
    """
    for name, value in (("physics_hz", physics_hz), ("control_hz", control_hz),
                        ("policy_hz", policy_hz)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if physics_hz % control_hz or control_hz % policy_hz:
        raise ValueError("Physics / VMC / policy frequencies must have integer ratios")
    return physics_hz // control_hz


def is_control_update(substep: int, stride: int) -> bool:
    """The first update is at t=0; commands are held on intervening steps."""
    if substep < 0 or stride <= 0:
        raise ValueError("Invalid diagnostic substep or control stride")
    return substep % stride == 0
