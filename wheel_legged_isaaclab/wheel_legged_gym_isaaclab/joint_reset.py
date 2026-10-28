"""Consistent closed-chain reset poses; never a runtime leg controller."""
import copy
import math

from wheel_legged_gym_isaaclab.robot_model import MODEL

CAD_ZERO_RESET = 'cad_zero_v0'
NOMINAL_JOINT_RESET = 'joint_target_nominal_v1'
DEFAULT_JOINT_OFFSETS = (-.0707452938, -.0336936327, .0707452938, .0336936327)


def _points(q_b, q_l, sign, geometry):
    upper, lower = geometry['upper'], geometry['lower']
    theta_b, theta_l = (geometry['offsets'][0]+sign*q_b,
                        geometry['offsets'][1]+sign*q_l)
    c = (upper[0]*math.cos(theta_b), upper[0]*math.sin(theta_b))
    a = (upper[1]*math.cos(theta_l), upper[1]*math.sin(theta_l))
    delta = (a[0]-c[0], a[1]-c[1])
    d = math.hypot(*delta)
    if d < 1.e-7:
        raise ValueError('Singular reset linkage')
    along = (lower[0]**2-lower[1]**2+d*d)/(2*d)
    height2 = lower[0]**2-along*along
    if height2 <= 1.e-10:
        raise ValueError('Invalid reset linkage branch')
    normal = (-delta[1]/d, delta[0]/d)
    p = tuple(c[k]+along*delta[k]/d+geometry['branch']*math.sqrt(height2)*normal[k]
              for k in (0, 1))
    return a, c, p


def closed_chain_reset_pose(offset, geometry, wheel_radius=MODEL['wheel_radius'], hip_z=MODEL['hip_z']):
    """Active targets plus passive four-bar/parallelogram coordinates, in rad.

    L1 follows A->P, L2 follows C->P. In the authored parallelogram L3
    remains parallel to B, while S remains parallel to L2. The articulation
    includes L3_S closures, not the external S joints. Both wheels start at 0.
    """
    if len(offset) != 4 or not all(math.isfinite(v) for v in offset):
        raise ValueError('Require four finite reset motor offsets')
    if geometry['branch'] != 1 or geometry['scale'] <= 0:
        raise ValueError('Require the imported closed-chain assembly branch')
    result, heights, legs = {}, [], []
    for prefix, sign, qb, ql in (('L', 1., *offset[:2]), ('R', -1., *offset[2:])):
        a0, c0, p0 = _points(0., 0., sign, geometry)
        a, c, p = _points(qb, ql, sign, geometry)
        def direction_change(origin, old_origin):
            value = math.atan2(p[1]-origin[1], p[0]-origin[0])-math.atan2(p0[1]-old_origin[1], p0[0]-old_origin[0])
            return math.atan2(math.sin(value), math.cos(value))/sign
        l1, l2 = direction_change(a, a0), direction_change(c, c0)
        result.update({prefix+'B_joint': qb, prefix+'L_joint': ql,
            prefix+'L1_joint': l1-ql, prefix+'L2_joint': l2-l1,
            prefix+'L3_joint': qb-l2, prefix+'L3_'+prefix+'S_closure': l2-qb,
            prefix+'W_joint': 0.})
        wheel = tuple(geometry['scale']*v for v in p)
        heights.append(wheel_radius-hip_z-wheel[0])
        legs.append(dict(length_m=math.hypot(*wheel), angle_rad=math.atan2(-wheel[1], -wheel[0])))
    if abs(heights[0]-heights[1]) > 1.e-6:
        raise ValueError('A level reset needs matching left/right wheel support heights')
    return dict(joint_positions=result, root_height_m=sum(heights)/2., legs=legs)


def reset_profile(config):
    mode = config.get('leg_control_mode', 'explicit_vmc')
    profile = config.get('joint_reset_profile', CAD_ZERO_RESET)
    if profile not in (CAD_ZERO_RESET, NOMINAL_JOINT_RESET):
        raise ValueError('Unknown joint reset profile')
    return profile if mode == 'implicit_joint_position' else CAD_ZERO_RESET


def configure_reset_pose(cfg):
    """Resolve the actual spawn/reset template before creating the articulation.

    Historical checkpoint replay explicitly selects cad_zero_v0. A diagnostic
    fixture's deliberately elevated root (e.g. .9 m hanging) stays elevated.
    """
    profile = reset_profile(dict(leg_control_mode=cfg.leg_control_mode,
                                 joint_reset_profile=cfg.joint_reset_profile))
    pose = closed_chain_reset_pose(cfg.leg_joint_position_offset, cfg.five_bar_geometry,
                                   cfg.wheel_radius, MODEL['hip_z'])
    cfg.robot.init_state.joint_pos = (dict(pose['joint_positions']) if profile == NOMINAL_JOINT_RESET else {'.*': 0.})
    x, y, z = cfg.robot.init_state.pos
    if any(math.isclose(z, value, abs_tol=1.e-8, rel_tol=0.) for value in
           (MODEL['nominal_root_height'], pose['root_height_m'])):
        cfg.robot.init_state.pos = (x, y, pose['root_height_m'] if profile == NOMINAL_JOINT_RESET else MODEL['nominal_root_height'])
    return pose


def require_matching_reset_profile(saved, current, allow_change=False):
    before, after = reset_profile(saved), reset_profile(current)
    if before != after and not (allow_change and before == CAD_ZERO_RESET and after == NOMINAL_JOINT_RESET):
        raise ValueError('Reset pose changed; qualify it and explicitly use --allow_reset_pose_change')
    return dict(source_profile=before, target_profile=after, changed=before != after)


def nominal_reset_config(saved):
    """Declare only the authorized initial-pose migration, preserving action meaning."""
    if saved.get('leg_control_mode') != 'implicit_joint_position':
        raise ValueError('Nominal motor reset migration requires direct-joint mode')
    result = copy.deepcopy(saved)
    result['joint_reset_profile'] = NOMINAL_JOINT_RESET
    pose = closed_chain_reset_pose(result['leg_joint_position_offset'], result['five_bar_geometry'],
                                   result['wheel_radius'], MODEL['hip_z'])
    initial = result['robot']['init_state']
    initial['joint_pos'] = dict(pose['joint_positions'])
    initial['pos'] = [*initial['pos'][:2], pose['root_height_m']]
    return result
