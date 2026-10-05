"""Import the CAD URDF and optionally add explicitly specified loop joints.

Run with ../run_python.sh run_isaac.py --open-chain --headless --steps 240.
This is a fixed-base, zero-gravity mechanism inspection, not a balance controller.
"""
import argparse
import json
import math
import re
import shutil
import traceback
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
JOINT_RENAMES = {"link_002_joint": "RB_joint"}


def load_model():
    model = json.loads((ROOT / "source/user_model.json").read_text())
    for joint in model["joints"]:
        for key in ("name", "joint_id"):
            joint[key] = JOINT_RENAMES.get(joint[key], joint[key])
        if joint["name"] in ("RB_joint", "RL_joint", "LB_joint", "LL_joint") and joint["parent"] != "base_link":
            raise ValueError(f"{joint['name']} must use base_link as its parent and angle reference")
    return model


def physics_body_transforms(articulation):
    """Read live PhysX poses; headless steps need not synchronize USD transforms."""
    import numpy as np
    from pxr import Gf
    poses = articulation._physics_view.get_link_transforms()[0]
    if not np.isfinite(poses).all():
        raise RuntimeError("Non-finite PhysX rigid body pose")
    transforms = {}
    for name, pose in zip(articulation.body_names, poses):
        x,y,z,qx,qy,qz,qw = map(float, pose)
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Rotation(Gf.Quatd(qw,qx,qy,qz)))
        matrix.SetTranslateOnly(Gf.Vec3d(x,y,z))
        transforms[name] = matrix
    return transforms


def prepare_urdf():
    """Work around importer mesh basenames that are invalid USD identifiers."""
    target = ROOT / "prepared"
    (target / "meshes").mkdir(parents=True, exist_ok=True)
    tree = ET.parse(ROOT / "source/robot.urdf")
    for joint in tree.findall("joint"):
        joint.set("name", JOINT_RENAMES.get(joint.get("name"), joint.get("name")))
    copied = set()
    for mesh in tree.findall(".//mesh"):
        source = ROOT / "source" / mesh.get("filename")
        name = re.sub(r"[^A-Za-z0-9_]", "_", source.stem) + source.suffix
        if source not in copied:
            shutil.copyfile(source, target / "meshes" / name)
            copied.add(source)
        mesh.set("filename", "meshes/" + name)
    path = target / "robot.urdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    (target / "user_model.json").write_text(json.dumps(load_model(), indent=2, ensure_ascii=False)+"\n")
    return path


def ensure_source():
    if (ROOT / "source/robot.urdf").exists():
        return
    target = (ROOT / "source").resolve()
    with zipfile.ZipFile(ROOT / "inputs/local_murktupf_658wsk_urdf_stl.zip") as archive:
        for name in archive.namelist():
            if not (target / name).resolve().is_relative_to(target):
                raise ValueError(f"Unsafe archive path: {name}")
        archive.extractall(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="0: keep GUI running")
    parser.add_argument("--sweep", action="store_true", help="Drive each active joint through a +/-10 degree cycle")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--open-chain", action="store_true")
    choice.add_argument("--closures", type=Path, help="JSON with confirmed CAD loop anchors")
    args = parser.parse_args()
    if args.headless and args.steps <= 0:
        parser.error("--headless requires --steps > 0")
    ensure_source()
    model = load_model()
    urdf_path = prepare_urdf()
    links = {link["name"] for link in model["links"]}
    closures = json.loads(args.closures.read_text())["joints"] if args.closures else []
    if args.closures and not closures:
        parser.error("Closure configuration must contain confirmed joints")
    names = {joint["name"] for joint in model["joints"]}
    for joint in closures:
        if joint["name"] in names:
            parser.error(f"Duplicate joint name: {joint['name']}")
        names.add(joint["name"])
        if joint["body0"] not in links or joint["body1"] not in links or joint["body0"] == joint["body1"]:
            parser.error(f"Invalid closure bodies: {joint}")
        for key in ("anchor_cad_mm", "axis_cad"):
            if len(joint[key]) != 3 or not all(math.isfinite(x) for x in joint[key]):
                parser.error(f"Invalid {key}: {joint}")
        if sum(x*x for x in joint["axis_cad"]) < 1e-12:
            parser.error("Joint axis cannot be zero")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})
    try:
        import carb
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux

        enable_extension("isaacsim.asset.importer.urdf")
        app.update()
        from isaacsim.asset.importer.urdf import _urdf
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils.viewports import set_camera_view

        out = ROOT / "build"
        out.mkdir(exist_ok=True)
        mode = "closed_chain" if closures else "open_chain"
        # Import into a file so mesh references resolve relative to a real layer.
        ok, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        if not ok:
            raise RuntimeError("URDF import config failed")
        cfg.merge_fixed_joints = False
        cfg.fix_base = True
        cfg.import_inertia_tensor = True
        cfg.self_collision = False
        cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_NONE
        cfg.convex_decomp = False
        raw_path = out / "imported.usd"
        ok, imported_path = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=str(urdf_path),
            import_config=cfg, dest_path=str(raw_path))
        if not ok or not imported_path:
            raise RuntimeError("URDF import failed")
        omni.usd.get_context().open_stage(str(raw_path))
        for _ in range(5):
            app.update()
        stage = omni.usd.get_context().get_stage()
        robot = stage.GetDefaultPrim()
        if not robot:
            robot = stage.GetPrimAtPath(imported_path)
        if not robot:
            raise RuntimeError(f"Imported robot prim is missing: {imported_path}")
        stage.SetDefaultPrim(robot)
        bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        if set(bodies) != links:
            raise RuntimeError(f"Rigid body mismatch: {set(bodies) ^ links}")
        joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
        xml = ET.parse(urdf_path).getroot()
        limits = {j.get("name"): float(j.find("limit").get("effort")) for j in xml.findall("joint")}
        active, passive = [], []
        for spec in model["joints"]:
            prim = joints[spec["name"]]
            if spec["name"] in ("RB_joint", "RL_joint", "LB_joint", "LL_joint"):
                joint = UsdPhysics.RevoluteJoint(prim)
                if joint.GetBody0Rel().GetTargets() != [bodies["base_link"].GetPath()]:
                    raise RuntimeError(f"{spec['name']} is not connected to base_link")
                joint.CreateBody0Rel().SetTargets([bodies["base_link"].GetPath()])
                prim.SetCustomDataByKey("angleReference", "base_link")
                prim.SetCustomDataByKey("angleZero", "exported assembly pose")
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            if spec["actuation"] == "passive":
                passive.append(spec["name"])
                continue
            active.append(spec["name"])
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(100.0)
            drive.CreateDampingAttr(2.0)
            drive.CreateMaxForceAttr(limits[spec["name"]])
            drive.CreateTargetPositionAttr(0.0)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateEnabledSelfCollisionsAttr(False)
                api.CreateSolverPositionIterationCountAttr(64)
                api.CreateSolverVelocityIterationCountAttr(8)

        cache = UsdGeom.XformCache()
        closure_records = []
        for spec in closures:
            # CAD export coordinates are millimetres; URDF root has base_origin removed.
            anchor = Gf.Vec3d(*[(v-b)*0.001 for v,b in zip(spec["anchor_cad_mm"], model["base_origin_xyz"])])
            axis = Gf.Vec3d(*spec["axis_cad"]).GetNormalized()
            frame = Gf.Matrix4d(1.0)
            frame.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), axis))
            frame.SetTranslateOnly(anchor)
            path = robot.GetPath().AppendPath("loop_joints/" + spec["name"])
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            joint.CreateAxisAttr("Z")
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateCollisionEnabledAttr(False)
            local_anchors = []
            for i in (0, 1):
                body = bodies[spec[f"body{i}"]]
                local = frame * cache.GetLocalToWorldTransform(body).GetInverse()
                pos = Gf.Vec3f(local.ExtractTranslation())
                rot = Gf.Quatf(local.ExtractRotationQuat())
                getattr(joint, f"CreateBody{i}Rel")().SetTargets([body.GetPath()])
                getattr(joint, f"CreateLocalPos{i}Attr")(pos)
                getattr(joint, f"CreateLocalRot{i}Attr")(rot)
                local_anchors.append(list(pos))
            closure_records.append({**spec, "local_anchors_m": local_anchors})

        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityMagnitudeAttr(0.0)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(240)
        UsdLux.DomeLight.Define(stage, "/inspectionLight").CreateIntensityAttr(1000)
        set_camera_view(eye=[0.85, 0.65, 0.6], target=[0.04, -0.06, -0.08])
        stage_path = out / f"{mode}.usda"
        stage.GetRootLayer().Export(str(stage_path))
        # Explicit USD updates allow joint-anchor validation after simulation steps.
        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
        sim = SimulationContext(physics_dt=1/240, rendering_dt=1/60, stage_units_in_meters=1.0)
        sim.initialize_physics()
        sim.play()
        from isaacsim.core.prims import Articulation
        import numpy as np
        articulation = Articulation(str(robot.GetPath()))
        articulation.initialize()
        dof_names = articulation.dof_names
        joint_ranges = {name: [float("inf"), float("-inf")] for name in dof_names}
        per_closure_gap = {record["name"]: 0.0 for record in closure_records}
        per_closure_axis_error = {record["name"]: 0.0 for record in closure_records}
        five_bars = {}
        mapping_errors = {}
        if {r["name"] for r in closure_records} == {"RL3_RS_closure", "RL2_RB_closure", "LL3_LS_closure", "LL2_LB_closure"}:
            from five_bar import FiveBar
            five_bars = {side: FiveBar(side) for side in ("right", "left")}
            mapping_errors = {side: 0.0 for side in five_bars}
        count = 0
        max_gap = 0.0
        while app.is_running() and (args.steps == 0 or count < args.steps):
            if args.sweep:
                # One independent actuator at a time, smoothly returning to zero.
                phase_length = 960
                actuator = active[(count // phase_length) % len(active)]
                phase = (count % phase_length) / phase_length
                target = 10.0 * math.sin(2*math.pi*phase)**3
                for name in active:
                    UsdPhysics.DriveAPI(joints[name], "angular").GetTargetPositionAttr().Set(target if name == actuator else 0.0)
            sim.step(render=not args.headless)
            positions = articulation.get_joint_positions()[0]
            if not np.isfinite(positions).all():
                raise RuntimeError("Non-finite articulation joint state")
            for name, angle in zip(dof_names, np.rad2deg(positions)):
                joint_ranges[name][0] = min(joint_ranges[name][0], float(angle))
                joint_ranges[name][1] = max(joint_ranges[name][1], float(angle))
            body_transforms = physics_body_transforms(articulation)
            joint_positions = dict(zip(dof_names, positions))
            for side, five_bar in five_bars.items():
                prefix = "R" if side == "right" else "L"
                motor_angles = [joint_positions[name] for name in five_bar.motor_names]
                predicted = five_bar.points(motor_angles)["W"]
                actual = np.array(body_transforms[prefix+"W_link"].ExtractTranslation())[1:] + np.array(model["base_origin_xyz"])[1:]*0.001
                mapping_errors[side] = max(mapping_errors[side], float(np.linalg.norm(predicted-actual))*1000)
            for record in closure_records:
                points = [body_transforms[record[f"body{i}"]].Transform(
                    Gf.Vec3d(*record["local_anchors_m"][i])) for i in (0, 1)]
                gap = (points[0]-points[1]).GetLength()
                max_gap = max(max_gap, gap)
                per_closure_gap[record["name"]] = max(per_closure_gap[record["name"]], gap*1000)
                joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(robot.GetPath().AppendPath("loop_joints/"+record["name"])))
                world_axes = []
                for side in (0, 1):
                    rotation = Gf.Rotation(Gf.Quatd(getattr(joint, f"GetLocalRot{side}Attr")().Get()))
                    local_axis = rotation.TransformDir(Gf.Vec3d(0,0,1))
                    world_axes.append(body_transforms[record[f"body{side}"]].TransformDir(local_axis).GetNormalized())
                axis_error = math.degrees(math.acos(max(-1.0,min(1.0,Gf.Dot(*world_axes)))))
                per_closure_axis_error[record["name"]] = max(per_closure_axis_error[record["name"]], axis_error)
            count += 1
        report = {"mode": mode, "fixed_base": True, "gravity_m_s2": 0,
                  "steps": count, "physics_dt": 1/240, "rigid_bodies": len(bodies),
                  "active_joints": active, "passive_joints": passive,
                  "closures": closure_records,
                  "max_closure_gap_mm": max_gap*1000 if closures else None,
                  "per_closure_max_gap_mm": per_closure_gap,
                  "per_closure_max_axis_error_deg": per_closure_axis_error,
                  "pose_source": "PhysX articulation link transforms",
                  "five_bar_max_wheel_error_mm": mapping_errors,
                  "sweep": args.sweep, "joint_ranges_deg": joint_ranges,
                  "stage": str(stage_path),
                  "scope": "Zero-gravity mechanism inspection only; no locomotion or RL validation."}
        suffix = "_sweep" if args.sweep else ""
        (out / f"{mode}{suffix}_report.json").write_text(json.dumps(report, indent=2)+"\n")
        print("MINE_MODEL_RESULT " + json.dumps(report), flush=True)
        sim.stop()
        if max_gap > 0.001:
            raise RuntimeError(f"Loop closure error exceeds 1 mm: {max_gap*1000:.3f} mm")
        if any(error > 0.2 for error in per_closure_axis_error.values()):
            raise RuntimeError(f"Loop axis error exceeds 0.2 degrees: {per_closure_axis_error}")
        if any(error > 1.0 for error in mapping_errors.values()):
            raise RuntimeError(f"Five-bar wheel mapping error exceeds 1 mm: {mapping_errors}")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
