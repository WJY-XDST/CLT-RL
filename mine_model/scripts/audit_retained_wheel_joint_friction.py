"""Inspect the saved USD wheel joints without starting Kit or a physics scene.

Run with the installed omni.usd.libs and omni.usd.schema.physx on PYTHONPATH
and their bin directories on LD_LIBRARY_PATH. The result describes composed
USD values and saved configuration; it does not measure a running PhysX view.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attribute_report(prim, name):
    attr = prim.GetAttribute(name)
    return dict(valid=bool(attr),
                authored=bool(attr and attr.HasAuthoredValueOpinion()),
                value=attr.Get() if attr else None)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    output = args.output.resolve()
    destination = output / 'wheel_joint_friction_authored_USD_readonly_20261023.json'
    if destination.exists():
        raise FileExistsError('Keep the original read-only receipt: ' + str(destination))
    plan_file = output / 'experiment_plan.json'
    plan = json.loads(plan_file.read_text())
    cfg = plan['training_config']
    asset = Path(cfg['robot']['spawn']['usd_path']).resolve()
    source = Path(plan['source_checkpoint']).resolve()
    inputs = [plan_file, asset, source, Path(__file__).resolve()]
    pins_before = {str(path): sha(path) for path in inputs}

    # Register the installed metadata before importing schemas, as Kit normally
    # performs this setup. Registration loads no application/physics context.
    from pxr import Plug
    cache = ROOT / 'IsaacLab/_isaac_sim/extscache'
    usd = cache / 'omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311'
    physx = cache / 'omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353'
    plugins = [usd / 'bin/usd/plugInfo.json',
               physx / 'plugins/PhysxSchema/resources/plugInfo.json',
               physx / 'plugins/PhysxSchemaAddition/resources/plugInfo.json',
               physx / 'plugins/OmniUsdPhysicsDeformableSchema/resources/plugInfo.json']
    registry = []
    for path in plugins:
        registered = Plug.Registry().RegisterPlugins(str(path))
        registry.append(dict(path=str(path), sha256=sha(path),
                             newly_registered=[p.name for p in registered]))
    from pxr import PhysxSchema, Usd, UsdPhysics
    stage = Usd.Stage.Open(str(asset))
    if not stage:
        raise RuntimeError('The exact saved asset could not be composed')
    layers = [Path(layer.realPath).resolve()
              for layer in stage.GetUsedLayers() if layer.realPath]
    layer_pins_before = {str(path): sha(path) for path in layers}
    rows = []
    for prim in stage.Traverse():
        if prim.GetName() not in ('LW_joint', 'RW_joint'):
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError('Expected the two retained revolute wheel joints')
        schema = PhysxSchema.PhysxJointAPI(prim)
        friction = schema.GetJointFrictionAttr()
        rows.append(dict(path=str(prim.GetPath()), name=prim.GetName(),
            has_PhysxJointAPI=prim.HasAPI(PhysxSchema.PhysxJointAPI),
            schema_joint_friction=dict(valid=bool(friction),
                authored=bool(friction and friction.HasAuthoredValueOpinion()),
                value=friction.Get() if friction else None),
            attributes={name: attribute_report(prim, name) for name in (
                'physxJoint:jointFriction', 'physxJoint:dynamicFriction',
                'physxJoint:viscousFriction', 'physxJoint:maxJointVelocity',
                'physxJoint:armature', 'drive:angular:physics:damping',
                'drive:angular:physics:stiffness', 'drive:angular:physics:maxForce')},
            authored_friction_and_drive_attributes={
                a.GetName(): a.Get() for a in prim.GetAuthoredAttributes()
                if 'friction' in a.GetName().lower() or 'drive:' in a.GetName()}))
    rows.sort(key=lambda row: row['name'])
    if [row['name'] for row in rows] != ['LW_joint', 'RW_joint']:
        raise ValueError('Retain both and only the two wheel joints')
    same = rows[0]['schema_joint_friction'] == rows[1]['schema_joint_friction']
    if {str(path): sha(path) for path in inputs} != pins_before:
        raise ValueError('An inspected input changed during the read')
    if {str(path): sha(path) for path in layers} != layer_pins_before:
        raise ValueError('A used USD layer changed during the read')
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        exact_saved_USD=str(asset), source_checkpoint=str(source),
        wheel_joints=rows, both_composed_joint_friction_equal=same,
        saved_runtime_wheel_control_mode=cfg['wheel_control_mode'],
        saved_runtime_wheel_damping=cfg['wheel_damping'],
        saved_runtime_wheel_actuator=cfg['robot']['actuators']['wheels'],
        plugin_registration=registry, source_pins=pins_before,
        used_USD_layer_pins=layer_pins_before, all_inspected_inputs_unchanged=True,
        simulator_started=False, running_PhysX_view_queried=False,
        physics_stepped=False, optimizer_executed=False, training_modified=False,
        policy_accepted=False,
        limitations=[
            'These are composed USD/schema defaults, not a measurement of live actuator friction.',
            'The environment subsequently creates a velocity drive using saved wheel_damping; '
            'an absent authored USD drive does not mean the running damping is zero.',
            'Joint friction, wheel-ground contact friction and finite drive tracking error '
            'are different quantities. This receipt alone does not establish a yaw-error cause.',
        ])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, receipt=str(destination),
                         same_friction=same, wheels=rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
