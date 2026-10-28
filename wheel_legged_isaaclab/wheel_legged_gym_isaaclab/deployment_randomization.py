"""Command-independent payload, material, CoM and implicit-PD uncertainty.

The existing sensor/delay hooks are reused. This opt-in extension does not
change nominal training, the policy interface, command sampling or scoring.
Material coefficients below are actor material values, not a claim about a
measured tire/ground pair. Effective friction also depends on the ground and
the PhysX combination rule, which are recorded separately.
"""
from dataclasses import asdict, dataclass
import math

import torch

from .robustness import RobustnessSession, RobustnessSpec


def _interval(name, value, *, positive=False, bounds=None):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or value[0] > value[1]):
        raise ValueError('Invalid interval: ' + name)
    if positive and value[0] <= 0:
        raise ValueError('Positive interval required: ' + name)
    if bounds and (value[0] < bounds[0] or value[1] > bounds[1]):
        raise ValueError('Interval outside physical bounds: ' + name)


@dataclass(frozen=True)
class DeploymentRandomizationSpec(RobustnessSpec):
    payload_body_name: str = 'base_link'
    body_mass_offset_kg: tuple = (0., 0.)
    body_inertia_scale: tuple = (1., 1.)
    body_com_x_m: tuple = (0., 0.)
    body_com_y_m: tuple = (0., 0.)
    body_com_z_m: tuple = (0., 0.)
    material_static_friction: tuple | None = None
    dynamic_friction_fraction: tuple = (1., 1.)
    restitution: tuple | None = None
    leg_stiffness_scale: tuple = (1., 1.)
    leg_damping_scale: tuple = (1., 1.)
    wheel_damping_scale: tuple = (1., 1.)

    def validate(self):
        super().validate()
        if not isinstance(self.payload_body_name, str) or not self.payload_body_name:
            raise ValueError('A payload body name is required')
        for name in ('body_mass_offset_kg', 'body_com_x_m', 'body_com_y_m', 'body_com_z_m'):
            _interval(name, getattr(self, name))
        for name in ('body_inertia_scale', 'leg_stiffness_scale',
                     'leg_damping_scale', 'wheel_damping_scale'):
            _interval(name, getattr(self, name), positive=True)
        _interval('dynamic_friction_fraction', self.dynamic_friction_fraction,
                  positive=True, bounds=(0., 1.))
        if self.material_static_friction is not None:
            _interval('material_static_friction', self.material_static_friction,
                      bounds=(0., math.inf))
            if tuple(self.friction_scale) != (1., 1.):
                raise ValueError('Choose absolute material friction or legacy scaling, not both')
        elif tuple(self.dynamic_friction_fraction) != (1., 1.):
            raise ValueError('Dynamic friction fraction requires absolute static friction')
        if self.restitution is not None:
            _interval('restitution', self.restitution, bounds=(0., 1.))


class DeploymentRandomizationSession(RobustnessSession):
    def _apply_properties(self):
        env, spec = self.env, self.spec
        robot, view = env._robot, env._robot.root_physx_view
        names = list(robot.body_names)
        if names.count(spec.payload_body_name) != 1:
            raise ValueError('Payload body must identify exactly one rigid body')
        body = names.index(spec.payload_body_name)
        wheels = [names.index(name) for name in ('LW_link', 'RW_link')]
        if body in wheels:
            raise ValueError('The measured wheels cannot be the payload body')
        ids = torch.arange(env.num_envs, device='cpu')
        before_mass, before_inertia = view.get_masses().clone(), view.get_inertias().clone()
        masses, inertias = before_mass.clone(), before_inertia.clone()
        scale = self._uniform(spec.mass_scale, (env.num_envs, 1), 'cpu')
        moving = [i for i in range(len(names)) if i not in wheels]
        masses[:, moving] *= scale
        payload = self._uniform(spec.body_mass_offset_kg, (env.num_envs,), 'cpu')
        masses[:, body] += payload
        if not torch.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError('Payload randomization would create a nonpositive or nonfinite mass')
        ratio = masses / before_mass
        shape = self._uniform(spec.inertia_scale, (env.num_envs, 1), 'cpu')
        body_shape = self._uniform(spec.body_inertia_scale, (env.num_envs,), 'cpu')
        inertias *= (ratio * shape)[..., None]
        inertias[:, body] *= body_shape[:, None]
        matrices = inertias.reshape(*inertias.shape[:-1], 3, 3)
        if (not torch.isfinite(inertias).all()
                or not torch.allclose(matrices, matrices.transpose(-1, -2), atol=1.e-7, rtol=1.e-5)
                or (torch.linalg.eigvalsh(matrices) <= 0).any()):
            raise ValueError('Randomized inertia must remain symmetric and positive definite')

        materials = view.get_material_properties().clone()
        before_materials = materials.clone()
        friction_scale = self._uniform(spec.friction_scale, (env.num_envs, 1), 'cpu')
        materials[..., :2] *= friction_scale[..., None]
        if spec.material_static_friction is not None:
            mu_static = self._uniform(spec.material_static_friction, (env.num_envs, 1), 'cpu')
            fraction = self._uniform(spec.dynamic_friction_fraction, (env.num_envs, 1), 'cpu')
            materials[..., 0] = mu_static
            materials[..., 1] = mu_static * fraction
        if spec.restitution is not None:
            materials[..., 2] = self._uniform(spec.restitution, (env.num_envs, 1), 'cpu')
        if (materials[..., :2] < 0).any() or (materials[..., 1] > materials[..., 0]).any():
            raise ValueError('Material friction must satisfy 0 <= dynamic <= static')

        com_changed = any(tuple(getattr(spec, 'body_com_' + axis + '_m')) != (0., 0.)
                          for axis in ('x', 'y', 'z'))
        coms, com_shift = None, None
        if com_changed:
            coms = view.get_coms().clone()
            com_shift = torch.stack([self._uniform(getattr(spec, 'body_com_' + axis + '_m'),
                                                 (env.num_envs,), 'cpu')
                                     for axis in ('x', 'y', 'z')], -1)
            coms[:, body, :3] += com_shift
        nominal_force = float(env.cfg.feedforward_force)
        changed_mass = tuple(spec.mass_scale) != (1., 1.) or tuple(spec.body_mass_offset_kg) != (0., 0.)
        changed_inertia = changed_mass or tuple(spec.inertia_scale) != (1., 1.) or tuple(spec.body_inertia_scale) != (1., 1.)
        if changed_mass:
            view.set_masses(masses, ids)
        if changed_inertia:
            view.set_inertias(inertias, ids)
        if (tuple(spec.friction_scale) != (1., 1.) or spec.material_static_friction is not None
                or spec.restitution is not None):
            view.set_material_properties(materials, ids)
        if com_changed:
            view.set_coms(coms, ids)
        actual_mass, actual_inertia = view.get_masses(), view.get_inertias()
        actual_materials = view.get_material_properties()
        if not (torch.allclose(actual_mass, masses, atol=1.e-6, rtol=1.e-6)
                and torch.allclose(actual_inertia, inertias, atol=1.e-7, rtol=1.e-6)
                and torch.allclose(actual_materials, materials, atol=1.e-6, rtol=1.e-6)):
            raise RuntimeError('Physical randomization readback does not match its request')
        if com_changed and not torch.allclose(view.get_coms(), coms, atol=1.e-7, rtol=1.e-6):
            raise RuntimeError('Randomized CoM readback does not match its request')
        assert torch.equal(actual_mass[:, wheels], before_mass[:, wheels])
        assert float(env.cfg.feedforward_force) == nominal_force
        gains = self._apply_gains()
        ground = env.cfg.terrain.physics_material
        self.evidence = dict(enabled=True, configuration=asdict(spec),
            nominal_mass_kg=before_mass.sum(-1).tolist(), actual_mass_kg=actual_mass.sum(-1).tolist(),
            payload_body_name=spec.payload_body_name, body_mass_offset_kg=payload.tolist(),
            payload_body_mass_kg=actual_mass[:, body].tolist(),
            actual_payload_body_com=view.get_coms()[:, body].tolist(),
            actual_payload_body_inertia=actual_inertia[:, body].tolist(),
            body_inertia_shape_scale=body_shape.tolist(),
            mass_and_inertia_readback_verified=True, inertia_positive_definite=True,
            measured_wheel_masses_preserved=True,
            nominal_material_coefficients=before_materials[0].tolist(),
            actual_material_coefficients=actual_materials.tolist(),
            ground_material_configuration=dict(static_friction=float(ground.static_friction),
                dynamic_friction=float(ground.dynamic_friction),
                friction_combine_mode=ground.friction_combine_mode),
            material_coefficients_are_not_effective_pair_coefficients=True,
            body_com_shift_m=com_shift.tolist() if com_shift is not None else None,
            body_com_written=bool(com_changed), body_com_readback_verified=True,
            implicit_PD_readback=gains,
            sensor_delay_ms=(self.sensor_ages * env.cfg.sim.dt * 1000.).tolist(),
            action_delay_ms=(self.action_ages * env.cfg.sim.dt * 1000.).tolist(),
            physical_properties_readback_verified=True,
            nominal_feedforward_per_leg_N=nominal_force,
            gravity_feedforward_retains_nominal_mass=True,
            parameters_chosen_from_commands=False, optimizer_executed=False,
            limitations=['Engineering stress ranges require later hardware measurement.',
                         'Payload is colocated with nominal body CoM unless an independent CoM offset is specified.',
                         'CoM and positive inertia shape perturbations model uncertainty, not an exact new CAD assembly.',
                         'Noise affects actor sensors; implicit servos and local gravity feedforward use simulator state.',
                         'Delay resolution is one physics step; no inference-time or substep accuracy claim is made.',
                         'Implicit actuator estimated-torque buffers are not measurements of final PD plus feedforward effort.'])

    def _apply_gains(self):
        env, spec = self.env, self.spec
        robot, view = env._robot, env._robot.root_physx_view
        changes = (('leg_stiffness_scale', 'get_dof_stiffnesses',
                    'write_joint_stiffness_to_sim', '_leg_joint_ids'),
                   ('leg_damping_scale', 'get_dof_dampings',
                    'write_joint_damping_to_sim', '_leg_joint_ids'),
                   ('wheel_damping_scale', 'get_dof_dampings',
                    'write_joint_damping_to_sim', '_wheel_joint_ids'))
        result = {}
        for field, getter, writer, selection in changes:
            if tuple(getattr(spec, field)) == (1., 1.):
                continue
            joints = [int(value) for value in getattr(env, selection)]
            before = getattr(view, getter)()[:, joints].clone()
            factor = self._uniform(getattr(spec, field), (env.num_envs, 1), 'cpu')
            target = before * factor
            getattr(robot, writer)(target.to(env.device), joint_ids=joints)
            actual = getattr(view, getter)()[:, joints]
            if not torch.allclose(actual, target, atol=1.e-5, rtol=1.e-6):
                raise RuntimeError('Implicit PD gain readback failed: ' + field)
            result[field] = dict(factor=factor[:, 0].tolist(), actual=actual.tolist(),
                                 nominal=before.tolist(), readback_verified=True)
        return result
