"""Add declared chassis payloads through a new USD mass-property layer.

When supplied, payload centres are offsets from the original base COM (+X
forward) and point-payload parallel-axis terms are included. When only a
symmetric installation is known, retain the nominal COM and explicitly use
the same-shape inertia estimate; do not invent battery coordinates.
The source asset, all limbs, collisions and joint frames remain immutable.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_assembly_masses import body_properties
from match_wheel_inertia import digest, inspect_asset
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def parallel_axis(mass, displacement):
    vector = np.asarray(displacement, dtype=float)
    return mass * (np.dot(vector, vector) * np.eye(3) - np.outer(vector, vector))


def combine_payloads(original, payloads):
    centre = np.asarray(original['com_local'], dtype=float)
    mass = float(original['mass_kg'])
    parts = [(mass, centre)]
    for payload in payloads:
        added = payload['mass_kg']
        offset = np.asarray(payload['offset_from_original_body_com_m'], dtype=float)
        if not math.isfinite(added) or added <= 0. or offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Payload requires positive finite mass and a finite 3D COM offset')
        parts.append((added, centre + offset))
    total = sum(value for value, _ in parts)
    combined_centre = sum(value * point for value, point in parts) / total
    tensor = np.asarray(original['tensor_link_frame'], dtype=float)
    for value, point in parts:
        tensor = tensor + parallel_axis(value, point - combined_centre)
    diagonal, axes = np.linalg.eigh(tensor)
    assert np.all(diagonal > 0.) and np.max(diagonal) < .5 * np.sum(diagonal)
    if np.linalg.det(axes) < 0.:
        axes[:, 0] *= -1.
    quaternion = Rotation.from_matrix(axes).as_quat()
    return dict(mass_kg=total, com_local=combined_centre.tolist(),
                tensor_link_frame=tensor.tolist(), principal_inertia_kg_m2=diagonal.tolist(),
                principal_axes_xyzw=quaternion.tolist())


def same_shape_estimate(original, payloads):
    added = sum(part['mass_kg'] for part in payloads)
    assert all(math.isfinite(part['mass_kg']) and part['mass_kg'] > 0. for part in payloads)
    total = original['mass_kg'] + added
    ratio = total / original['mass_kg']
    return dict(mass_kg=total, com_local=original['com_local'],
                tensor_link_frame=(np.asarray(original['tensor_link_frame']) * ratio).tolist(),
                principal_inertia_kg_m2=(np.asarray(original['principal_inertia_kg_m2']) * ratio).tolist(),
                principal_axes_xyzw=original['principal_axes_xyzw'])


def build(source, target, payload_path, report_path):
    if target.exists() or report_path.exists():
        raise FileExistsError('Preserve each authored asset and mass-property report')
    spec = json.loads(payload_path.read_text())
    assert spec['payloads'] and spec['coordinate_frame'] == 'original_base_link_COM_X_forward'
    assert spec['intrinsic_payload_inertias_measured'] is False
    before = inspect_asset(source)
    properties = body_properties(source)
    method = spec['inertia_model']
    if method == 'point_payloads_at_declared_coordinates':
        expected = combine_payloads(properties['base_link'], spec['payloads'])
    elif method == 'same_shape_about_retained_com':
        assert spec['front_and_existing_rear_batteries_symmetric']
        assert not spec['absolute_payload_centres_measured']
        expected = same_shape_estimate(properties['base_link'], spec['payloads'])
    else:
        raise ValueError('Declare a supported mass-property estimation method')
    layer = Sdf.Layer.CreateNew(str(target))
    layer.defaultPrim = 'mine_robot'
    layer.subLayerPaths = [os.path.relpath(source, target.parent)]
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    api = UsdPhysics.MassAPI(stage.OverridePrim('/mine_robot/base_link'))
    api.CreateMassAttr(expected['mass_kg'])
    api.CreateCenterOfMassAttr(Gf.Vec3f(*expected['com_local']))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*expected['principal_inertia_kg_m2']))
    q = expected['principal_axes_xyzw']
    api.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    layer.Save()
    after = inspect_asset(target)
    current = body_properties(target)
    assert properties.keys() == current.keys()
    for name in properties:
        if name != 'base_link':
            assert properties[name] == current[name], name
    np.testing.assert_allclose(current['base_link']['com_local'], expected['com_local'], rtol=0., atol=1.e-8)
    np.testing.assert_allclose(current['base_link']['tensor_link_frame'], expected['tensor_link_frame'], rtol=1.e-6, atol=1.e-8)
    added_mass = sum(part['mass_kg'] for part in spec['payloads'])
    assert abs(after['total_mass_kg'] - before['total_mass_kg'] - added_mass) < 2.e-6
    assert after['wheels'] == before['wheels'] and digest(source) == before['sha256']
    report = dict(verified=True, source=str(source), source_sha256=digest(source),
                  output=str(target), output_sha256=digest(target), payload_spec=spec,
                  payload_spec_sha256=digest(payload_path), added_body_mass_kg=added_mass,
                  before=before, after=after, old_base_properties=properties['base_link'],
                  calculated_base_properties=expected, body_mass_properties=current,
                  source_asset_unchanged=True, all_nonbase_mass_COM_inertia_unchanged=True,
                  wheels_and_leg_assemblies_unchanged=True,
                  base_inertia_method=method,
                  inertia_is_estimated_not_measured=True,
                  absolute_payload_coordinates_invented=False,
                  payload_intrinsic_inertia_omitted=(method == 'point_payloads_at_declared_coordinates'),
                  native_PhysX_loading_still_required=True, policy_accepted=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'payloads', 'report'):
        cli.add_argument('--' + name, type=Path, required=True)
    args = cli.parse_args()
    result = build(args.source.resolve(), args.output.resolve(), args.payloads.resolve(), args.report.resolve())
    print(json.dumps(dict(output=result['output'], total_mass_kg=result['after']['total_mass_kg'],
                          base_mass_kg=result['calculated_base_properties']['mass_kg'],
                          base_COM_m=result['calculated_base_properties']['com_local'],
                          inertia_estimated=True, native_loading_verified=False)))
