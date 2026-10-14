"""Read-only joint checks for the Isaac Sim hands-on closed-chain tutorial.

In Isaac Sim's Script Editor:
    import runpy
    guide = runpy.run_path("/absolute/path/to/tutorial_closed_chain_check.py")
    guide["inspect_current_stage"](require_closed=False)

This helper never authors attributes, changes control modes, or saves a stage.
Geometry checks use authored USD transforms. Run them while the timeline is
STOPPED, not to infer live PhysX constraint errors from potentially stale USD.
"""
from pathlib import Path
import argparse
import json
import math

MODEL_DIR = Path(__file__).resolve().parents[1]
REFERENCE = MODEL_DIR / "exports/export_20261003_181515/mine_closed_chain.usd"
CLOSURES = MODEL_DIR / "config/closures.json"
METADATA = MODEL_DIR / "exports/export_20261003_181515/model.json"
ACTIVE = {"LB_joint", "LL_joint", "RB_joint", "RL_joint", "LW_joint", "RW_joint"}


def _bodies(stage, robot):
    from pxr import Usd, UsdPhysics
    result = {}
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if prim.GetName() in result:
                raise ValueError(f"Duplicate rigid body name: {prim.GetName()}")
            result[prim.GetName()] = prim
    return result


def _robot(stage, path):
    if path:
        robot = stage.GetPrimAtPath(path)
        if not robot:
            raise ValueError(f"Robot prim does not exist: {path}")
        return robot
    from pxr import UsdPhysics
    base_links = [p for p in stage.Traverse()
                  if p.GetName() == "base_link" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(base_links) != 1:
        raise ValueError("Specify robot_path; the stage must contain exactly one base_link")
    return base_links[0].GetParent()


def _frame(joint, side):
    from pxr import Gf
    value = Gf.Matrix4d(1.)
    value.SetRotate(Gf.Rotation(Gf.Quatd(getattr(joint, f"GetLocalRot{side}Attr")().Get())))
    value.SetTranslateOnly(Gf.Vec3d(getattr(joint, f"GetLocalPos{side}Attr")().Get()))
    return value


def _eulers(rotation):
    from pxr import Gf
    # Match Isaac Sim 5.1's GfQuatEulerAttributeModel, not a different RPY order.
    return [round(float(v), 6) for v in rotation.Decompose(
        Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))]


def check_stage(stage, robot_path=None, require_closed=True):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    robot = _robot(stage, robot_path)
    bodies = _bodies(stage, robot)
    reference_stage = Usd.Stage.Open(str(REFERENCE))
    reference_bodies = _bodies(reference_stage, reference_stage.GetDefaultPrim())
    specs = json.loads(CLOSURES.read_text())["joints"]
    metadata = json.loads(METADATA.read_text())
    expected_tree = {p.GetName() for p in reference_stage.Traverse()
                     if p.IsA(UsdPhysics.RevoluteJoint)
                     and not UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()}
    expected_closures = {s["name"] for s in specs}
    errors, warnings = [], []
    if set(bodies) != set(reference_bodies):
        raise ValueError(f"Wrong body set: {set(bodies) ^ set(reference_bodies)}")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-9:
        errors.append("Stage Meters Per Unit must be 1.0")
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        errors.append("Stage Up Axis must be Z for the normalized URDF")
    cache, reference_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    base_world = cache.GetLocalToWorldTransform(bodies["base_link"])
    reference_base = reference_cache.GetLocalToWorldTransform(reference_bodies["base_link"])
    # Only the initial CAD assembly is a valid input for the printed frame plan.
    initial_pose = True
    for name, prim in bodies.items():
        actual = cache.GetLocalToWorldTransform(prim) * base_world.GetInverse()
        expected = reference_cache.GetLocalToWorldTransform(reference_bodies[name]) * reference_base.GetInverse()
        delta = actual * expected.GetInverse()
        if delta.ExtractTranslation().GetLength() > 1e-4 or abs(delta.ExtractRotation().GetAngle()) > .1:
            initial_pose = False
    if not initial_pose:
        warnings.append("Not the initial assembly pose; reset before using FRAME_PLAN values")

    plan = []
    if initial_pose:
        for spec in specs:
            cad = [(a - b) * .001 for a, b in zip(
                spec["anchor_cad_mm"], metadata["base_origin_cad_mm"])]
            anchor_base = Gf.Vec3d(cad[2], cad[0], cad[1])
            axis_cad = spec["axis_cad"]
            axis_base = Gf.Vec3d(axis_cad[2], axis_cad[0], axis_cad[1])
            target = Gf.Matrix4d(1.)
            target.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), axis_base))
            target.SetTranslateOnly(anchor_base)
            target *= base_world
            record = dict(name=spec["name"], axis="Z", exclude=True)
            for side in (0, 1):
                body = bodies[spec[f"body{side}"]]
                local = target * cache.GetLocalToWorldTransform(body).GetInverse()
                record[f"body{side}"] = str(body.GetPath())
                record[f"local_pos{side}_m"] = [round(float(v), 9) for v in local.ExtractTranslation()]
                record[f"local_rot{side}_xyz_deg"] = _eulers(local.ExtractRotation())
            plan.append(record)

    all_joints = [p for p in Usd.PrimRange(robot) if p.IsA(UsdPhysics.Joint)]
    joints = {}
    for prim in all_joints:
        name = prim.GetName()
        if name in joints:
            errors.append(f"Duplicate joint name: {name}")
        joints[name] = UsdPhysics.Joint(prim)
    missing = expected_tree - set(joints)
    if missing:
        errors.append(f"Missing original joints: {sorted(missing)}")
    missing_closures = expected_closures - set(joints)
    if require_closed and missing_closures:
        errors.append(f"Missing closures: {sorted(missing_closures)}")
    graph = {name: set() for name in bodies}
    tree_edges, external, hinge_records = [], [], []
    for name, joint in joints.items():
        prim = joint.GetPrim()
        # An imported static-base world fixture is not one of the 18 hinges.
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            warnings.append(f"Non-revolute fixture or extra joint: {prim.GetPath()}")
            continue
        targets = [getattr(joint, f"GetBody{side}Rel")().GetTargets() for side in (0, 1)]
        if any(len(v) != 1 for v in targets):
            errors.append(f"{name}: both bodies must have exactly one target")
            continue
        paths = [v[0] for v in targets]
        pair = [stage.GetPrimAtPath(v).GetName() for v in paths]
        if any(n not in bodies or bodies[n].GetPath() != path for n, path in zip(pair, paths)):
            errors.append(f"{name}: Body 0/1 must reference rigid-body Xforms, not meshes")
            continue
        if joint.GetJointEnabledAttr().Get() is False:
            errors.append(f"{name}: joint is disabled")
        if name in {"LB_joint", "LL_joint", "RB_joint", "RL_joint"} and pair[0] != "base_link":
            errors.append(f"{name}: the active hip must use base_link as Body 0")
        spec = next((s for s in specs if s["name"] == name), None)
        if spec and pair != [spec["body0"], spec["body1"]]:
            errors.append(f"{name}: wrong closure body order {pair}")
        if name not in ACTIVE and prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            drive = UsdPhysics.DriveAPI(prim, "angular")
            gains = (drive.GetStiffnessAttr().Get() or 0., drive.GetDampingAttr().Get() or 0.)
            maximum = drive.GetMaxForceAttr().Get()
            if any(abs(v) > 1e-12 for v in gains) and maximum != 0.:
                errors.append(f"{name}: passive joint has a nonzero Angular Drive")
            else:
                warnings.append(f"{name}: zero-force/gain DriveAPI is inert; remove it in a clean passive asset")
        if name not in expected_tree | expected_closures:
            errors.append(f"Unexpected revolute joint: {name}")
        excluded = bool(joint.GetExcludeFromArticulationAttr().Get())
        if excluded:
            external.append(name)
        else:
            tree_edges.append(name)
            graph[pair[0]].add(pair[1])
            graph[pair[1]].add(pair[0])
        world = [_frame(joint, side) * cache.GetLocalToWorldTransform(bodies[pair[side]])
                 for side in (0, 1)]
        gap = (world[0].ExtractTranslation() - world[1].ExtractTranslation()).GetLength()
        axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[
            UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()]
        directions = [w.TransformDir(axis).GetNormalized() for w in world]
        angle = math.degrees(math.acos(max(-1., min(1., Gf.Dot(*directions)))))
        if gap > 1e-4:
            errors.append(f"{name}: authored anchor mismatch {gap*1000:.6f} mm")
        if angle > .1:
            errors.append(f"{name}: authored hinge axes differ by {angle:.6f} degrees")
        hinge_records.append(dict(name=name, bodies=pair, excluded=excluded,
                                  gap_mm=float(gap*1000), axis_error_deg=angle))
    visited, todo = set(), ["base_link"]
    while todo:
        node = todo.pop()
        if node not in visited:
            visited.add(node)
            todo.extend(graph[node] - visited)
    if len(tree_edges) != len(bodies) - 1 or len(visited) != len(bodies):
        errors.append("Articulation edges must form a connected, acyclic 14-edge tree over all 15 bodies")
    if require_closed and (len(hinge_records) != 18 or len(external) != 4):
        errors.append("Closed mechanism must have 18 enabled hinges and 4 external hinges")
    roots = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        belongs = prim.GetPath().HasPrefix(robot.GetPath())
        if prim.IsA(UsdPhysics.Joint):
            targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
            belongs = belongs or any(p.HasPrefix(robot.GetPath()) for p in targets)
        if belongs:
            roots.append(str(prim.GetPath()))
    if len(roots) != 1:
        errors.append(f"Exactly one Articulation Root is required for this mechanism, found {roots}")
    return dict(passed=not errors, robot=str(robot.GetPath()), rigid_bodies=len(bodies),
                hinges=len(hinge_records), tree_hinges=len(tree_edges), external_hinges=external,
                articulation_roots=roots, initial_assembly_pose=initial_pose,
                frame_plan=plan, hinges_detail=hinge_records, errors=errors, warnings=warnings,
                scope="Authored USD geometry only; not a live PhysX or VMC support test")


def inspect_current_stage(robot_path=None, require_closed=True):
    import omni.usd
    import omni.timeline
    if not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError("Click STOP before checking authored USD; PAUSE is not enough")
    result = check_stage(omni.usd.get_context().get_stage(), robot_path, require_closed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("PASS" if result["passed"] else "FAIL", "- authored geometry and articulation tree only")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, type=Path)
    parser.add_argument("--robot-path")
    parser.add_argument("--open-chain", action="store_true")
    args = parser.parse_args()
    from pxr import Usd
    report = check_stage(Usd.Stage.Open(str(args.usd.resolve())), args.robot_path,
                         require_closed=not args.open_chain)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
