"""Enable only the user-confirmed leg-stop contact pairs on the base-box asset.

CAD vertices are unchanged. Concave stop/shank CAD uses convex decomposition;
body and shape pair filters prevent unrelated internal contacts.
"""
import hashlib
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next((ROOT / 'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent))
from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protected-tree', action='store_true')
    parser.add_argument('--collision-groups', action='store_true')
    parser.add_argument('--output-name')
    parser.add_argument('--precise-meshes', action='store_true')
    parser.add_argument('--source', type=Path, help='Collision asset to use; solver-tree selection remains independent')
    args = parser.parse_args()
    folder = ROOT / 'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515'
    source = folder / ('mine_closed_chain_training_protected.usd' if args.protected_tree else 'mine_closed_chain_base_box.usd')
    if args.source:
        source = args.source.resolve()
        if source.parent != folder or not source.is_file():
            raise ValueError('Source must be an existing asset in the export folder')
    output = folder / ('mine_closed_chain_training_protected_contacts.usd' if args.protected_tree else 'mine_closed_chain_base_box_stops.usd')
    if args.collision_groups:
        output = folder / 'mine_closed_chain_training_protected_contacts_groups.usd'
    if args.output_name:
        if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.usd'):
            raise ValueError('Expected a USD filename')
        output = folder / args.output_name
    if output.exists():
        raise FileExistsError(output)
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [str(source)]
    original = Usd.Stage.Open(str(source))
    stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    root = stage.GetPrimAtPath('/mine_robot/base_link')
    root.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool, custom=False).Set(True)
    pair_specs = [('LB_link', 'part_413_4_28', 'LS_link', 'part_410___1_15', 'LS_joint'),
                  ('RB_link', 'part_192_4_28', 'RS_link', 'part_156___1_15', 'RS_joint')]
    allowed_bodies = {frozenset((a, b)) for a, _, b, _, _ in pair_specs}
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for index, body in enumerate(bodies):
        rel = UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel()
        for other in bodies[index + 1:]:
            if frozenset((body.GetName(), other.GetName())) not in allowed_bodies:
                rel.AddTarget(other.GetPath())
    records = []
    for upper, stop_part, lower, shank_part, joint_name in pair_specs:
        for body in (upper, lower):
            stage.GetPrimAtPath('/mine_robot/' + body + '/collisions').SetInstanceable(False)
        def shape_path(body, part):
            return '/mine_robot/' + body + '/collisions/' + part + '/node_STL_BINARY_'
        stop = stage.GetPrimAtPath(shape_path(upper, stop_part))
        shank = stage.GetPrimAtPath(shape_path(lower, shank_part))
        if not stop or not shank:
            raise ValueError('Missing confirmed stop/contact surface')
        upper_shapes = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/mine_robot/' + upper)) if p.HasAPI(UsdPhysics.CollisionAPI)]
        lower_shapes = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/mine_robot/' + lower)) if p.HasAPI(UsdPhysics.CollisionAPI)]
        for a in upper_shapes:
            rel = UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel()
            for b in lower_shapes:
                if a != stop or b != shank:
                    rel.AddTarget(b.GetPath())
        for prim in (stop, shank):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
            prim.AddAppliedSchema('PhysxConvexDecompositionCollisionAPI')
            prim.CreateAttribute('physxConvexDecompositionCollision:hullVertexLimit', Sdf.ValueTypeNames.Int, custom=False).Set(64)
            prim.CreateAttribute('physxConvexDecompositionCollision:maxConvexHulls', Sdf.ValueTypeNames.Int, custom=False).Set(64)
            prim.CreateAttribute('physxConvexDecompositionCollision:errorPercentage', Sdf.ValueTypeNames.Float, custom=False).Set(1.0)
            if args.precise_meshes:
                # The collider must be the Mesh itself. Keeping CollisionAPI
                # only on its CAD Xform wrapper makes PhysX ignore mesh-specific
                # decomposition/contact settings on these imported parts.
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
                UsdPhysics.CollisionAPI.Apply(prim.GetChild('mesh')).CreateCollisionEnabledAttr(True)
                for target in (prim, prim.GetChild('mesh')):
                    UsdPhysics.MeshCollisionAPI.Apply(target).CreateApproximationAttr('convexDecomposition')
                    target.AddAppliedSchema('PhysxConvexDecompositionCollisionAPI')
                    target.CreateAttribute('physxConvexDecompositionCollision:hullVertexLimit', Sdf.ValueTypeNames.Int, custom=False).Set(64)
                    target.CreateAttribute('physxConvexDecompositionCollision:maxConvexHulls', Sdf.ValueTypeNames.Int, custom=False).Set(64)
                    target.CreateAttribute('physxConvexDecompositionCollision:errorPercentage', Sdf.ValueTypeNames.Float, custom=False).Set(.5)
                    target.AddAppliedSchema('PhysxCollisionAPI')
                    target.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float, custom=False).Set(.001)
                    target.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float, custom=False).Set(0.)
        joint = UsdPhysics.Joint(stage.GetPrimAtPath('/mine_robot/joints/' + joint_name))
        if args.protected_tree:
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateCollisionEnabledAttr(True)
            closure_name = 'LL3_LS_closure' if upper == 'LB_link' else 'RL3_RS_closure'
            UsdPhysics.Joint(stage.GetPrimAtPath('/mine_robot/loop_joints/' + closure_name)).CreateExcludeFromArticulationAttr(False)
        else:
            joint.CreateCollisionEnabledAttr(True)
        records.append({'stop': str(stop.GetChild('mesh').GetPath() if args.precise_meshes else stop.GetPath()),
                        'contact_surface': str(shank.GetChild('mesh').GetPath() if args.precise_meshes else shank.GetPath()),
                        'joint': str(joint.GetPath()), 'approximation': 'convexDecomposition',
                        'mesh_points_unchanged': True})
    if args.collision_groups:
        group_names = ('ordinary', 'stops', 'shanks')
        groups = {}
        for name in group_names:
            group = UsdPhysics.CollisionGroup.Define(stage, '/mine_robot/collision_filters/' + name)
            group.CreateMergeGroupNameAttr('mine_' + name)
            groups[name] = group
        for name in group_names:
            for other in group_names:
                if name == 'ordinary' or other == 'ordinary' or name == other:
                    groups[name].CreateFilteredGroupsRel().AddTarget(groups[other].GetPath())
        stop_paths = {r['stop'] for r in records}
        shank_paths = {r['contact_surface'] for r in records}
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                prim.GetRelationship('physics:filteredPairs').ClearTargets(True)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() != False:
                path = str(prim.GetPath())
                name = 'stops' if path in stop_paths else 'shanks' if path in shank_paths else 'ordinary'
                groups[name].GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(prim.GetPath())
    checks = []
    for prim in original.Traverse(Usd.TraverseInstanceProxies()):
        current = stage.GetPrimAtPath(prim.GetPath())
        if prim.HasAPI(UsdPhysics.MassAPI) or prim.IsA(UsdPhysics.Joint):
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if name.startswith(('physics:', 'physxJoint:')) and name not in ('physics:collisionEnabled', 'physics:excludeFromArticulation'):
                    if current.GetAttribute(name).Get() != attr.Get():
                        raise ValueError('Dynamics/joint frame changed')
            for rel in prim.GetRelationships():
                if rel.GetName().startswith('physics:') and current.GetRelationship(rel.GetName()).GetTargets() != rel.GetTargets():
                    raise ValueError('Joint relation changed')
            checks.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            for name in ('points', 'faceVertexCounts', 'faceVertexIndices'):
                if current.GetAttribute(name).Get() != prim.GetAttribute(name).Get():
                    raise ValueError('CAD mesh changed: ' + str(prim.GetPath()))
    stage.GetRootLayer().subLayerPaths = [source.name]
    stage.GetRootLayer().Export(str(output))
    manifest = {'source': str(source), 'overlay': str(output),
                'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, output)},
                'confirmed_stop_pairs': records, 'unchanged_dynamics_and_joint_frames': checks,
                'status': 'diagnostic; physical stop sweep and policy replay required',
                'solver_tree_changed': args.protected_tree,
                'precise_mesh_approximation': args.precise_meshes,
                'collision_filter': 'three collision groups; ordinary robot shapes excluded from internal contact' if args.collision_groups else 'body and shape FilteredPairsAPI',
                'runtime_override_required': {'robot.spawn.articulation_props.enabled_self_collisions': True}}
    if args.protected_tree:
        manifest['runtime_override_required']['robot.actuators.passive.joint_names_expr'] = ['[LR]L[123]_joint', '[LR]L3_[LR]S_closure']
        manifest['tree_description'] = 'LS/RS knee hinges become external loop constraints; LL3_LS/RL3_RS existing closures enter the articulation tree. All 18 hinge frames/bodies/axes retained; active motor and wheel names unchanged.'
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
