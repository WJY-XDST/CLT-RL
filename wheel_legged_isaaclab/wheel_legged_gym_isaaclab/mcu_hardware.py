"""Output-shaft parameters from the supplied series_balance_rtos firmware.

Authority: the outer MDK-ARM/project_h7.uvprojx build, not its older nested copy.
The MG8016-i6 torque/current calibration was confirmed by the owner from a
customer-support curve and successful real-vehicle use on 2026-10-10. Preserve
that calibration rather than substituting a generic product-sheet constant.
These are firmware command limits, NOT verified continuous motor ratings.
Raw encoder / CAN conversions belong at the hardware boundary. Isaac joint
coordinates already describe output shafts; applying a gear ratio again here
would multiply torque incorrectly.
"""
import math


HARDWARE_PROFILE = "series_balance_rtos_outer_20261010"
MCU_SOURCE = "sim2real/series_balance_rtos (2)/series_balance_rtos/series_balance_rtos"
LEG_MOTOR_MODEL = "翎控 MG8016-i6"
LEG_TORQUE_CALIBRATION_AUTHORITY = "owner_confirmed_support_curve_and_real_vehicle_20261010"

# cubotMiddleware/Devices/LK_motor.c: torque / 1.078 * 2048 / 33,
# then clamp to +/-2048. 1.078 is the deployed output-shaft conversion, not a
# bare-motor constant to multiply by a gearbox again. 33 A is the firmware's
# command full scale; it is NOT an independently verified continuous rating.
LEG_TORQUE_PER_AMP = 1.078
LEG_CURRENT_LIMIT_A = 33.0
LEG_EFFORT_LIMIT = LEG_TORQUE_PER_AMP * LEG_CURRENT_LIMIT_A

# control/torque.h + torque.c: clamp the final 3508 CAN command to +/-16000.
# torque_sen=0.3 changes the upstream LQR request gain, NOT the final cap.
WHEEL_TORQUE_PER_CAN_UNIT = 0.00030059
WHEEL_CAN_LIMIT = 16000
WHEEL_EFFORT_LIMIT = WHEEL_TORQUE_PER_CAN_UNIT * WHEEL_CAN_LIMIT
WHEEL_UPSTREAM_TORQUE_GAIN = 0.3
WHEEL_RADIUS = 0.06
WHEEL_RPM_TO_OUTPUT_RAD_S = 0.006643  # executed control/wheel.c conversion
WHEEL_HEADER_REDUCTION_RATE = 0.052631  # unused 1:19 declaration, conflicts above
WHEEL_CONTROL_TRACK_WIDTH = 0.45  # observer calibration, not a CAD measurement
LEG_ENCODER_ZERO_COUNTS = (51904, 37894, 11792, 14533)
LEG_ENCODER_COUNTS_PER_TURN = 65536
VMC_CONTROL_HZ = 500

# control/torque.c M1 and F0_control, control/WBR.c pitch_com, control/vmc.c.
# Owner-confirmed definition: M1=6.17 kg is HALF the nominal WHOLE vehicle
# mass, including its wheels. Each leg's base feedforward is m_vehicle*g/2.
# This parameter does not come from CAD supported_mass_kg or exclude wheels.
# Keep the nominal firmware load separate from the CAD rigid-body mass sum.
GRAVITY = 9.81
NOMINAL_VEHICLE_MASS_KG = 12.34
GRAVITY_COMPENSATION_MASS_PER_LEG = NOMINAL_VEHICLE_MASS_KG / 2.0
GRAVITY_FORCE = GRAVITY * GRAVITY_COMPENSATION_MASS_PER_LEG
GRAVITY_FORCE_LIMIT = 150.0
GRAVITY_ANGLE_BIAS = math.radians(2.0) + 1.57 - math.pi / 2

# The active LK torque-control path does not define safe motor angle or rated
# output-speed bounds. Encoder wrap, 350 deg/s recovery commands, and the
# unused DM MIT protocol ranges must NOT be promoted to MG8016-i6 limits.
LEG_POSITION_LIMIT_RAD = None
LEG_SPEED_LIMIT_RAD_S = None
WHEEL_SPEED_LIMIT_RAD_S = None
MCU_LIMIT_EVIDENCE = {
    'leg_effort_nm': {'value': LEG_EFFORT_LIMIT, 'source': 'cubotMiddleware/Devices/LK_motor.c:torque_to_out',
                      'meaning': 'output-shaft command clamp, not verified continuous torque rating'},
    'wheel_effort_nm': {'value': WHEEL_EFFORT_LIMIT, 'source': 'control/torque.c:torque_motor3508; control/torque.h',
                        'meaning': 'final +/-16000 CAN command times deployed conversion'},
    'gravity_radial_force_n': {'value': GRAVITY_FORCE_LIMIT, 'source': 'control/torque.c:F0_control',
                              'meaning': 'gravity feedforward clamp, not a motor torque limit'},
    'leg_position_rad': {'value': LEG_POSITION_LIMIT_RAD, 'source': None,
                         'meaning': 'no active-joint position hard limit established by supplied LK torque path'},
    'leg_speed_rad_s': {'value': LEG_SPEED_LIMIT_RAD_S, 'source': None,
                       'meaning': 'no output-shaft speed hard limit established by supplied LK torque path'},
    'wheel_speed_rad_s': {'value': WHEEL_SPEED_LIMIT_RAD_S, 'source': None,
                         'meaning': 'no output-shaft speed hard limit established by supplied 3508 torque path'},
}


def wheel_rpm_to_output_speed(rpm):
    """Firmware forward-positive output rad/s for raw [left, right] RPM."""
    return rpm * rpm.new_tensor([-1.0, 1.0]) * WHEEL_RPM_TO_OUTPUT_RAD_S


def gravity_feedforward(leg_angle, body_pitch, *, mode, force, angle_bias=0.0,
                        force_limit=GRAVITY_FORCE_LIMIT):
    """Outward-positive radial force [N], with no gas-spring contribution.

    The configured base force is half the nominal whole-vehicle weight.
    MCU angle compensation converts that vertical support into radial force;
    it does not change the mass definition or impose equal instantaneous loads.

    leg_angle is backward-positive relative to base_link; body_pitch uses the
    simulation Y-axis convention. Their sum corresponds to MCU theta0 after
    encoder/frame conversion. MCU F0 is inward-positive, hence F_sim = -F0.
    The near-zero cosine guard keeps the MCU +/-150 N cap finite at singularity.
    """
    import torch
    if not math.isfinite(force) or force < 0:
        raise ValueError("Gravity force must be nonnegative and finite")
    if mode == "constant":
        return torch.full_like(leg_angle, force)
    if mode != "mcu_pitch_compensated":
        raise ValueError(f"Unknown gravity feedforward mode: {mode}")
    if not math.isfinite(force_limit) or force_limit <= 0 or not math.isfinite(angle_bias):
        raise ValueError("Gravity compensation needs a positive cap and finite angle bias")
    angle = leg_angle + body_pitch + angle_bias
    # Reproduce the wrap and sign check of get_l_theta / F0_control.
    angle = torch.where(angle > 1.1, angle - 2 * math.pi, angle)
    cosine = angle.cos()
    denominator = torch.where(cosine < 0, -cosine.abs().clamp_min(1e-6),
                              cosine.abs().clamp_min(1e-6))
    result = (force / denominator).clamp(-force_limit, force_limit)
    return torch.where(angle > math.pi / 2, -result, result)


def synchronize_actuator_limits(cfg):
    """Make software clipping, USD drives and Isaac actuator limits agree."""
    for group, limit in (("legs", cfg.leg_effort_limit), ("wheels", cfg.wheel_effort_limit)):
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"{group} effort limit must be positive and finite")
        actuator = cfg.robot.actuators[group]
        actuator.effort_limit_sim = limit
        # ImplicitActuator raises for contradictory legacy/current fields.
        actuator.effort_limit = None
