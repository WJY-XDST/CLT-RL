"""Estimate leg mass/COM/inertia from original CAD meshes and assembly weighing.

Structural plates use an estimated CFRP density. Pulley, shaft and bearing
geometry is grouped as fittings; its effective density is fitted to the owner's
non-wheel assembly weight. Individual parts remain estimates, not measurements.
URDF visual geometry is used, never simplified collision hulls or boxes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

LEG_LINKS = {s: [s + n + '_link' for n in ('B', 'S', 'L', 'L1', 'L2', 'L3')] for s in ('L', 'R')}
CARBON_PLATES = {
    'part_411___1_15.stl', 'part_410___1_15.stl', 'part_365___1_15.stl',
    'part_368___1_15.stl', 'part_369___1_15.stl', 'part_370___1_15.stl',
    'part_154___1_15.stl', 'part_156___1_15.stl', 'part_157___1_15.stl',
    'part_163___1_15.stl', 'part_160___1_15.stl', 'part_159___1_15.stl',
}
DENSITY_REFERENCE = 'https://www.hexcel.com/wp-content/uploads/2026/01/HexPly_913_us_DataSheet.pdf'


def mesh_properties(path, scale, xyz, rpy):
    mesh = trimesh.load_mesh(path, process=True)
    mesh.apply_scale(scale)
    if not mesh.is_winding_consistent or not np.isfinite(mesh.vertices).all() or mesh.volume <= 0:
        raise ValueError('CAD mesh lacks a positive consistently oriented signed volume: ' + str(path))
    # Some bearing exports contain coincident internal faces/nonmanifold joins.
    # Their oriented volume integral is accepted only after translation tests;
    # do not replace their hollow geometry with a convex hull.
    shifted = mesh.copy()
    translation = np.array([.31, -.27, .18])
    shifted.apply_translation(translation)
    np.testing.assert_allclose(shifted.volume, mesh.volume, rtol=1.e-8, atol=1.e-14)
    np.testing.assert_allclose(shifted.center_mass - translation, mesh.center_mass, atol=1.e-8)
    np.testing.assert_allclose(shifted.moment_inertia, mesh.moment_inertia, rtol=1.e-6, atol=1.e-12)
    rotation = Rotation.from_euler('xyz', rpy).as_matrix()
    center = rotation @ mesh.center_mass + xyz
    inertia = rotation @ mesh.moment_inertia @ rotation.T
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                volume_m3=float(mesh.volume), com_link_frame=center.tolist(),
                inertia_unit_density_link_frame=inertia.tolist(),
                watertight=bool(mesh.is_watertight), winding_consistent=True,
                signed_integral_translation_verified=True,
                estimated_material='carbon_plate' if path.name in CARBON_PLATES else 'fitting')


def combine_parts(parts, carbon_density, fitting_density):
    masses = np.array([p['volume_m3'] * (carbon_density if p['estimated_material'] == 'carbon_plate'
                                        else fitting_density) for p in parts])
    centers = np.array([p['com_link_frame'] for p in parts])
    mass = float(masses.sum())
    center = (masses[:, None] * centers).sum(axis=0) / mass
    inertia = np.zeros((3, 3))
    for p, m, c in zip(parts, masses, centers):
        density = carbon_density if p['estimated_material'] == 'carbon_plate' else fitting_density
        offset = c - center
        inertia += density * np.asarray(p['inertia_unit_density_link_frame'])
        inertia += m * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    inertia = (inertia + inertia.T) / 2.
    moments, axes = np.linalg.eigh(inertia)
    if moments.min() <= 0 or moments[-1] > moments[:2].sum() + 1.e-10:
        raise ValueError('Estimated inertia is not physically admissible.')
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1.
    return dict(mass_kg=mass, com_local=center.tolist(), tensor_link_frame=inertia.tolist(),
                principal_inertia_kg_m2=moments.tolist(),
                principal_axes_xyzw=Rotation.from_matrix(axes).as_quat().tolist())


def estimate(urdf_path, nonwheel_mass, carbon_density):
    if not np.isfinite(nonwheel_mass) or nonwheel_mass <= 0 or not 1200 <= carbon_density <= 1900:
        raise ValueError('Need a positive measured assembly mass and explicit estimated CFRP density.')
    tree = ET.parse(urdf_path)
    result = dict(urdf=str(urdf_path.resolve()),
                  urdf_sha256=hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                  nonwheel_leg_mass_kg=nonwheel_mass, carbon_density_kg_m3=carbon_density,
                  carbon_density_measured=False, density_reference=DENSITY_REFERENCE,
                  plate_material_authority='owner_confirmed_carbon_fiber_plates',
                  individual_link_masses_measured=False, legs={}, body_mass_properties={})
    for side, names in LEG_LINKS.items():
        groups = {}
        for name in names:
            link = tree.find("link[@name='" + name + "']")
            groups[name] = []
            for visual in link.findall('visual'):
                geometry = visual.find('geometry/mesh')
                origin = visual.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                scale = np.fromstring(geometry.get('scale', '1 1 1'), sep=' ')
                groups[name].append(mesh_properties(urdf_path.parent / geometry.get('filename'), scale, xyz, rpy))
        parts = [p for group in groups.values() for p in group]
        carbon_volume = sum(p['volume_m3'] for p in parts if p['estimated_material'] == 'carbon_plate')
        fitting_volume = sum(p['volume_m3'] for p in parts if p['estimated_material'] == 'fitting')
        fitting_density = (nonwheel_mass - carbon_density * carbon_volume) / fitting_volume
        if not 2000 <= fitting_density <= 9000:
            raise ValueError('Weighed mass and CAD volumes imply implausible effective fitting density.')
        for name, group in groups.items():
            result['body_mass_properties'][name] = combine_parts(group, carbon_density, fitting_density)
        total = sum(result['body_mass_properties'][n]['mass_kg'] for n in names)
        assert abs(total - nonwheel_mass) < 1.e-10
        result['legs'][side] = dict(carbon_plate_volume_m3=carbon_volume, fitting_volume_m3=fitting_volume,
            carbon_plate_mass_kg=carbon_volume * carbon_density,
            fitting_mass_kg=nonwheel_mass - carbon_volume * carbon_density,
            effective_fitting_density_kg_m3=fitting_density, total_nonwheel_mass_kg=total,
            mesh_groups=groups)
    result['method'] = ('mass = estimated material density * signed CAD solid volume; '
                        'fit accessory effective density to measured nonwheel leg total; '
                        'COM and inertia integrated in URDF link frames with parallel-axis theorem')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--leg-with-wheel-mass-kg', type=float, required=True)
    parser.add_argument('--wheel-mass-kg', type=float, required=True)
    parser.add_argument('--carbon-density-kg-m3', type=float, default=1600.)
    args = parser.parse_args()
    report = estimate(args.urdf, args.leg_with_wheel_mass_kg - args.wheel_mass_kg,
                      args.carbon_density_kg_m3)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(link_masses_kg={n: p['mass_kg'] for n, p in report['body_mass_properties'].items()},
                         material_budget={s: {k: v for k, v in data.items() if k != 'mesh_groups'}
                                          for s, data in report['legs'].items()}), indent=2))


if __name__ == '__main__':
    main()
