"""Export an immutable, standard-frame closed-chain robot and install a project copy."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from run_isaac import ROOT, ensure_source, prepare_urdf, load_model
from five_bar import FiveBar

ensure_source()
prepared = prepare_urdf()
stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
output = ROOT / "exports" / f"export_{stamp}"
output.mkdir(parents=True, exist_ok=False)
shutil.copytree(prepared.parent / "meshes", output / "meshes")
tree = ET.parse(prepared)
tree.getroot().set("name", "mine_robot")
# CAD +Z -> standard +X (forward), CAD +X -> +Y (left), CAD +Y -> +Z (up).
rotation = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
def rotate_origin(element):
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    origin.set("xyz", " ".join(f"{v:.12g}" for v in rotation @ xyz))
    composed = rotation @ Rotation.from_euler("xyz", rpy).as_matrix()
    origin.set("rpy", " ".join(f"{v:.12g}" for v in Rotation.from_matrix(composed).as_euler("xyz")))

base = tree.getroot().find("link[@name='base_link']")
for element in base:
    if element.tag in ("visual", "collision", "inertial"):
        rotate_origin(element)
for joint in tree.findall("joint"):
    if joint.find("parent").get("link") == "base_link":
        rotate_origin(joint)
    if joint.get("name") in ("RW_joint", "LW_joint"):
        joint.set("type", "continuous")
        joint.find("limit").attrib.pop("lower", None)
        joint.find("limit").attrib.pop("upper", None)
        joint.find("limit").set("velocity", "40")
urdf_path = output / "mine_robot.urdf"
tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
model = load_model()
closures = json.loads((ROOT / "config/closures.json").read_text())["joints"]
hip_cad = np.array(next(j["origin"]["xyz"] for j in model["joints"] if j["name"] == "RL_joint"))
hip_z = float((rotation @ ((hip_cad-np.array(model["base_origin_xyz"]))*0.001))[2])
leg = FiveBar("right")
metadata = {
    "name": "mine_closed_chain", "version": stamp,
    "coordinate_frame": "x_forward_y_left_z_up", "cad_to_base_rotation": rotation.tolist(),
    "base_origin_cad_mm": model["base_origin_xyz"],
    "hip_z": hip_z, "wheel_radius": 0.06, "wheel_track_width": 0.4199,
    "mass_kg": sum(link["mass"] for link in model["links"]),
    "supported_mass_kg": sum(link["mass"] for link in model["links"] if link["name"] not in ("LW_link","RW_link")),
    "leg_effort_limit": 10.0, "wheel_effort_limit": 10.0,
    "leg_joint_names": ["LB_joint", "LL_joint", "RB_joint", "RL_joint"],
    "wheel_joint_names": ["LW_joint", "RW_joint"],
    "five_bar": {"upper": leg.upper.tolist(), "lower": leg.lower.tolist(),
                 "offsets": leg.offsets.tolist(), "scale": leg.scale, "branch": leg.branch},
    "nominal_leg_length": float(np.linalg.norm(leg.forward([0,0]))),
    "nominal_root_height": float(-leg.forward([0,0])[0]-hip_z+0.06),
    "input_angle_reference": "base_link, q=0 at CAD assembly pose",
    "closures": closures,
    "original_urdf_zip_sha256": hashlib.sha256((ROOT/'inputs/local_murktupf_658wsk_urdf_stl.zip').read_bytes()).hexdigest(),
    "notes": ["URDF is the spanning tree; the four loop constraints are in USD.",
              "Wheel joints are continuous in this export; original source limits are preserved in source/.",
              "Masses and inertias, including the original left/right wheel mass asymmetry, are preserved."]}

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    enable_extension("isaacsim.asset.importer.urdf")
    app.update()
    from isaacsim.asset.importer.urdf import _urdf
    ok,cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.fix_base = False
    cfg.merge_fixed_joints = False
    cfg.import_inertia_tensor = True
    cfg.self_collision = False
    cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_NONE
    cfg.convex_decomp = False
    raw = output / "imported.usd"
    ok,path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(urdf_path),
                                      import_config=cfg, dest_path=str(raw))
    if not ok or not path:
        raise RuntimeError("Standard-frame URDF import failed")
    stage = Usd.Stage.Open(str(raw))
    robot = stage.GetPrimAtPath(path)
    stage.SetDefaultPrim(robot)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    bodies = {p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI.Apply(prim)
            api.CreateEnabledSelfCollisionsAttr(False)
            api.CreateSolverPositionIterationCountAttr(64)
            api.CreateSolverVelocityIterationCountAttr(8)
    cache = UsdGeom.XformCache()
    for spec in closures:
        anchor = rotation @ ((np.array(spec["anchor_cad_mm"])-model["base_origin_xyz"])*.001)
        axis = rotation @ spec["axis_cad"]
        frame = Gf.Matrix4d(1.)
        frame.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), Gf.Vec3d(*axis)))
        frame.SetTranslateOnly(Gf.Vec3d(*anchor))
        joint = UsdPhysics.RevoluteJoint.Define(stage, robot.GetPath().AppendPath("loop_joints/"+spec["name"]))
        joint.CreateAxisAttr("Z")
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        for side in (0,1):
            body = bodies[spec[f"body{side}"]]
            local = frame * cache.GetLocalToWorldTransform(body).GetInverse()
            getattr(joint,f"CreateBody{side}Rel")().SetTargets([body.GetPath()])
            getattr(joint,f"CreateLocalPos{side}Attr")(Gf.Vec3f(local.ExtractTranslation()))
            getattr(joint,f"CreateLocalRot{side}Attr")(Gf.Quatf(local.ExtractRotationQuat()))
    for name in metadata["leg_joint_names"]:
        prim=stage.GetPrimAtPath(robot.GetPath().AppendPath("joints/"+name))
        if UsdPhysics.Joint(prim).GetBody0Rel().GetTargets() != [bodies["base_link"].GetPath()]:
            raise RuntimeError(f"Wrong angle reference: {name}")
        prim.SetCustomDataByKey("angleReference", "base_link")
    usd_path=output/"mine_closed_chain.usd"
    stage.Flatten().Export(str(usd_path))
    metadata["usd_sha256"]=hashlib.sha256(usd_path.read_bytes()).hexdigest()
    (output/"model.json").write_text(json.dumps(metadata,indent=2)+"\n")
    shutil.copyfile(ROOT/"config/closures.json", output/"closures_cad.json")
    # Install a versioned project asset, retaining the upstream robot unchanged.
    asset_root=ROOT.parent/"wheel_legged_isaaclab/assets/robots/mine"
    installed=asset_root/output.name
    installed.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(usd_path,installed/usd_path.name)
    shutil.copyfile(output/"model.json",installed/"model.json")
    manifest={"version":stamp,"usd":str(Path(output.name)/usd_path.name),
              "metadata":str(Path(output.name)/"model.json"),"source_export":str(output)}
    (asset_root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("EXPORTED_MINE_MODEL "+json.dumps(manifest),flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
