"""Calibrate measured vehicle/leg totals while retaining versioned CAD sources.

The owner weighed each leg WITH its wheel, WITHOUT the body-mounted hip motors.
Individual moving-link mass/COM/inertia are estimated from CAD solid volumes,
material assumptions and measured assembly totals. Missing body mass
is initially lumped at the existing base COM; added rotational inertia is
unknown, so existing base inertia is retained. These assumptions are recorded.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# This module bootstraps the supplied SDK's USD libraries before importing pxr.
from match_wheel_inertia import digest, inspect_asset
from pxr import Gf, Sdf, Usd, UsdPhysics

LEG_LINKS = {side: [side + name + '_link' for name in ('B', 'S', 'L', 'L1', 'L2', 'L3')]
             for side in ('L', 'R')}


def body_properties(path):
    stage = Usd.Stage.Open(str(path))
    result = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.MassAPI(prim)
        axes = api.GetPrincipalAxesAttr().Get()
        quaternion = [*axes.GetImaginary(), axes.GetReal()]
        diagonal = np.asarray(api.GetDiagonalInertiaAttr().Get())
        rotation = Rotation.from_quat(quaternion).as_matrix()
        result[prim.GetName()] = dict(mass_kg=api.GetMassAttr().Get(),
            com_local=list(api.GetCenterOfMassAttr().Get()),
            principal_axes_xyzw=quaternion, principal_inertia_kg_m2=diagonal.tolist(),
            tensor_link_frame=(rotation @ np.diag(diagonal) @ rotation.T).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--vehicle-mass-kg', type=float, required=True)
    parser.add_argument('--leg-with-wheel-mass-kg', type=float, required=True)
    parser.add_argument('--wheel-mass-kg', type=float, required=True)
    parser.add_argument('--leg-mass-estimate', type=Path, required=True)
    args = parser.parse_args()
    if (not all(math.isfinite(x) and x > 0 for x in
                (args.vehicle_mass_kg, args.leg_with_wheel_mass_kg, args.wheel_mass_kg))
            or args.leg_with_wheel_mass_kg <= args.wheel_mass_kg
            or args.vehicle_mass_kg <= 2 * args.leg_with_wheel_mass_kg):
        raise ValueError('Measured vehicle/leg/wheel masses must have a positive residual body mass.')
    source, target = args.source.resolve(), args.output.resolve()
    if target.exists():
        raise FileExistsError('Keep qualified physics versions immutable: ' + str(target))
    before = inspect_asset(source)
    before['body_mass_properties'] = body_properties(source)
    nonwheel_leg_mass = args.leg_with_wheel_mass_kg - args.wheel_mass_kg
    estimate = json.loads(args.leg_mass_estimate.read_text())
    if (not math.isclose(estimate['nonwheel_leg_mass_kg'], nonwheel_leg_mass, abs_tol=1.e-10)
            or estimate['individual_link_masses_measured']
            or set(estimate['body_mass_properties']) != {*LEG_LINKS['L'], *LEG_LINKS['R']}):
        raise ValueError('Volume estimate does not match the measured leg mass budget.')
    body_mass = args.vehicle_mass_kg - 2 * args.leg_with_wheel_mass_kg
    expected_bodies = {'base_link', 'LW_link', 'RW_link', *LEG_LINKS['L'], *LEG_LINKS['R']}
    if set(before['body_masses_kg']) != expected_bodies:
        raise ValueError('Mass budget requires the current fifteen-body articulation.')
    for wheel in ('LW_link', 'RW_link'):
        if not math.isclose(before['body_masses_kg'][wheel], args.wheel_mass_kg, abs_tol=1.e-6):
            raise ValueError('Use the owner-calibrated identical-wheel source asset.')
    layer = Sdf.Layer.CreateNew(str(target))
    layer.defaultPrim = 'mine_robot'
    layer.subLayerPaths = [os.path.relpath(source, target.parent)]
    stage = Usd.Stage.Open(layer)
    UsdPhysics.MassAPI(stage.OverridePrim('/mine_robot/base_link')).CreateMassAttr(body_mass)
    for side, names in LEG_LINKS.items():
        for name in names:
            estimated = estimate['body_mass_properties'][name]
            api = UsdPhysics.MassAPI(stage.OverridePrim('/mine_robot/' + name))
            api.CreateMassAttr(estimated['mass_kg'])
            api.CreateCenterOfMassAttr(Gf.Vec3f(*estimated['com_local']))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*estimated['principal_inertia_kg_m2']))
            q = estimated['principal_axes_xyzw']
            api.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    layer.Save()
    after = inspect_asset(target)
    after['body_mass_properties'] = body_properties(target)
    assert math.isclose(after['total_mass_kg'], args.vehicle_mass_kg, abs_tol=2.e-6)
    for side, names in LEG_LINKS.items():
        assert math.isclose(sum(after['body_masses_kg'][n] for n in names)
                            + after['body_masses_kg'][side + 'W_link'],
                            args.leg_with_wheel_mass_kg, abs_tol=1.e-6)
    for name, original in before['body_mass_properties'].items():
        current = after['body_mass_properties'][name]
        if name in ('base_link', 'LW_link', 'RW_link'):
            assert current['com_local'] == original['com_local']
            assert current['principal_axes_xyzw'] == original['principal_axes_xyzw']
            assert current['principal_inertia_kg_m2'] == original['principal_inertia_kg_m2']
        else:
            props = estimate['body_mass_properties'][name]
            np.testing.assert_allclose(current['com_local'], props['com_local'], atol=1.e-8)
            np.testing.assert_allclose(current['tensor_link_frame'], props['tensor_link_frame'],
                                       rtol=2.e-5, atol=1.e-9)
    assert after['wheels'] == before['wheels']
    assert digest(source) == before['sha256']
    report = dict(before=before, after=after, source_unchanged=True,
        authority='owner_measured_vehicle_and_leg_assembly_masses',
        measured_vehicle_mass_kg=args.vehicle_mass_kg,
        measured_leg_with_wheel_mass_kg=args.leg_with_wheel_mass_kg,
        measured_wheel_mass_kg=args.wheel_mass_kg,
        leg_mass_excludes_body_mounted_hip_motors=True,
        nonwheel_leg_mass_kg=nonwheel_leg_mass, inferred_fixed_body_mass_kg=body_mass,
        leg_mass_estimate_path=str(args.leg_mass_estimate.resolve()),
        leg_mass_estimate_sha256=digest(args.leg_mass_estimate),
        individual_link_masses_measured=False,
        mass_distribution_method=estimate['method'],
        base_added_mass_kg=body_mass - before['body_masses_kg']['base_link'],
        base_inertia_method='added missing body mass approximated at existing CAD COM; original inertia retained',
        leg_inertia_method='CAD volume integrated inertia and COM with measured assembly budget; unmeasured',
        fixed_body_com_and_inertia_unchanged=True, wheel_properties_unchanged=True,
        qualification='USD composition and measured mass budgets only; native loading still required')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('measured_vehicle_mass_kg',
        'measured_leg_with_wheel_mass_kg', 'nonwheel_leg_mass_kg', 'inferred_fixed_body_mass_kg',
        'mass_distribution_method', 'base_added_mass_kg', 'qualification')}, indent=2))


if __name__ == '__main__':
    main()
