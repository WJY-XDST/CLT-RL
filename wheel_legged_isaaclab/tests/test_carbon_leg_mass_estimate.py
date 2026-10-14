"""Check CAD mass integration against analytical solids and weighed assemblies."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from estimate_carbon_leg_mass import combine_parts, estimate, mesh_properties


@pytest.mark.parametrize('scale', [(0.001, 0.001, 0.001), (0.002, 0.001, 0.0005)])
def test_millimetre_box_rotation_and_origin_match_analytic_inertia(tmp_path, scale):
    dimensions_mm = np.array([80., 30., 4.])
    center_mm = np.array([17., -23., 9.])
    mesh = trimesh.creation.box(extents=dimensions_mm)
    mesh.apply_translation(center_mm)
    path = tmp_path/'plate.stl'
    mesh.export(path)
    origin = np.array([.2, -.1, .03])
    angles = np.array([.17, -.42, .63])
    result = mesh_properties(path, np.array(scale), origin, angles)
    dimensions = dimensions_mm * scale
    volume = np.prod(dimensions)
    rotation = Rotation.from_euler('xyz', angles).as_matrix()
    diagonal = volume / 12. * np.array([
        dimensions[1]**2 + dimensions[2]**2,
        dimensions[0]**2 + dimensions[2]**2,
        dimensions[0]**2 + dimensions[1]**2,
    ])
    assert result['volume_m3'] == pytest.approx(volume, rel=1.e-8)
    np.testing.assert_allclose(result['com_link_frame'], rotation @ (center_mm * scale) + origin,
                               rtol=0., atol=1.e-10)
    np.testing.assert_allclose(result['inertia_unit_density_link_frame'],
                               rotation @ np.diag(diagonal) @ rotation.T, rtol=1.e-7, atol=1.e-16)


def test_inverted_solid_does_not_silently_become_positive_mass(tmp_path):
    mesh = trimesh.creation.box(extents=[.1, .04, .003])
    mesh.invert()
    path = tmp_path/'inverted.stl'
    mesh.export(path)
    with pytest.raises(ValueError, match='positive consistently oriented'):
        mesh_properties(path, np.ones(3), np.zeros(3), np.zeros(3))


def test_separated_equal_cubes_obey_parallel_axis_theorem_and_density():
    edge = .02
    volume = edge**3
    local_unit_inertia = np.eye(3) * volume * edge**2 / 6.
    offset = .05
    parts = [dict(volume_m3=volume, estimated_material='carbon_plate',
                  com_link_frame=[x, 0., 0.],
                  inertia_unit_density_link_frame=local_unit_inertia.tolist())
             for x in (-offset, offset)]
    density = 1600.
    result = combine_parts(parts, density, 7000.)
    mass = 2. * volume * density
    expected = np.diag([mass * edge**2 / 6.,
                        mass * (edge**2 / 6. + offset**2),
                        mass * (edge**2 / 6. + offset**2)])
    assert result['mass_kg'] == pytest.approx(mass)
    np.testing.assert_allclose(result['com_local'], np.zeros(3), atol=1.e-14)
    np.testing.assert_allclose(result['tensor_link_frame'], expected, rtol=1.e-12, atol=1.e-16)
    principal_axes = Rotation.from_quat(result['principal_axes_xyzw']).as_matrix()
    np.testing.assert_allclose(
        principal_axes @ np.diag(result['principal_inertia_kg_m2']) @ principal_axes.T,
        expected, rtol=1.e-12, atol=1.e-16)


def test_different_materials_shift_center_of_mass():
    volume = .001
    parts = [dict(volume_m3=volume, estimated_material=material,
                  com_link_frame=[x, 0., 0.],
                  inertia_unit_density_link_frame=(np.eye(3) * 1.e-6).tolist())
             for material, x in [('carbon_plate', -.1), ('fitting', .1)]]
    result = combine_parts(parts, 1600., 6400.)
    assert result['mass_kg'] == pytest.approx(8.)
    np.testing.assert_allclose(result['com_local'], [.06, 0., 0.], atol=1.e-14)


@pytest.mark.parametrize('carbon_density', [1500., 1600., 1800.])
def test_actual_cad_density_uncertainty_preserves_weighed_leg_mass(carbon_density):
    root = Path(__file__).resolve().parents[2]
    report = estimate(root/'mine_model/exports/export_20261003_181515/mine_robot.urdf',
                      1.935 - .665, carbon_density)
    assert not report['carbon_density_measured']
    assert not report['individual_link_masses_measured']
    for side in ('L', 'R'):
        budget = report['legs'][side]
        assert budget['total_nonwheel_mass_kg'] + .665 == pytest.approx(1.935, abs=1.e-12)
        assert budget['carbon_plate_mass_kg'] + budget['fitting_mass_kg'] == pytest.approx(1.27)
        assert 2000. < budget['effective_fitting_density_kg_m3'] < 9000.
    for suffix in ('B', 'S', 'L', 'L1', 'L2', 'L3'):
        left = report['body_mass_properties']['L' + suffix + '_link']
        right = report['body_mass_properties']['R' + suffix + '_link']
        assert left['mass_kg'] == pytest.approx(right['mass_kg'], rel=2.e-6)
        for body in (left, right):
            values = np.linalg.eigvalsh(body['tensor_link_frame'])
            assert values[0] > 0.
            assert values[-1] <= values[0] + values[1] + 1.e-10
