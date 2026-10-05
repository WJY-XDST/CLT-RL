"""Compare the installed USD's physical data with the versioned URDF, offline."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
usd_lib = next((ROOT / "IsaacLab/_isaac_sim/extscache").glob("omni.usd.libs-*/pxr")).parent
sys.path.insert(0, str(usd_lib))
from pxr import Gf, Usd, UsdGeom, UsdPhysics

assets = ROOT / "wheel_legged_isaaclab/assets/robots/mine"
manifest = json.loads((assets / "manifest.json").read_text())
model = json.loads((assets / manifest["metadata"]).read_text())
urdf = ET.parse(Path(manifest["source_export"]) / "mine_robot.urdf")
stage = Usd.Stage.Open(str(assets / manifest["usd"]))
bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
report = {"body_count": len(bodies), "links": {}, "closures": {}, "motor_parent": {}, "wheel_axes_base": {}}
cache = UsdGeom.XformCache()
base_inv = cache.GetLocalToWorldTransform(bodies["base_link"]).GetInverse()
total_mass = 0.
checks = []
for link in urdf.findall("link"):
    name = link.get("name")
    inertial = link.find("inertial")
    mass = float(inertial.find("mass").get("value"))
    origin = inertial.find("origin")
    com = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rot = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    i = inertial.find("inertia")
    a = {k: float(v) for k, v in i.attrib.items()}
    tensor = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])
    tensor = rot @ tensor @ rot.T
    api = UsdPhysics.MassAPI(bodies[name])
    usd_mass = api.GetMassAttr().Get()
    usd_com = np.array(api.GetCenterOfMassAttr().Get())
    diag = np.array(api.GetDiagonalInertiaAttr().Get())
    q = api.GetPrincipalAxesAttr().Get()
    usd_rot = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
    usd_tensor = usd_rot @ np.diag(diag) @ usd_rot.T
    passed = bool(np.isclose(mass, usd_mass, rtol=1e-6) and np.allclose(com, usd_com, atol=1e-8, rtol=1e-6) and np.allclose(tensor, usd_tensor, atol=1e-8, rtol=1e-5) and (diag > 0).all())
    report["links"][name] = {"mass_kg": usd_mass, "mass_error_kg": abs(mass-usd_mass), "com_max_error_m": float(np.max(np.abs(com-usd_com))), "inertia_max_error_kg_m2": float(np.max(np.abs(tensor-usd_tensor))), "passed": passed}
    total_mass += usd_mass
    checks.append(passed)
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    j = UsdPhysics.RevoluteJoint(prim)
    name = prim.GetName()
    if name in model["leg_joint_names"]:
        parent = str(j.GetBody0Rel().GetTargets()[0])
        report["motor_parent"][name] = parent
        checks.append(parent == str(bodies["base_link"].GetPath()))
    if name in model["wheel_joint_names"]:
        parent = stage.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
        q = j.GetLocalRot0Attr().Get()
        axis = Gf.Vec3d(*{"X": (1,0,0), "Y": (0,1,0), "Z": (0,0,1)}[j.GetAxisAttr().Get()])
        axis = Gf.Rotation(Gf.Quatd(q)).TransformDir(axis)
        axis = (cache.GetLocalToWorldTransform(parent) * base_inv).TransformDir(axis)
        report["wheel_axes_base"][name] = list(axis)
        checks.append(bool(np.allclose(axis, [0, 1 if name == "LW_joint" else -1, 0], atol=1e-6)))
    if name.endswith("_closure"):
        points = []
        for side in (0, 1):
            body = stage.GetPrimAtPath(getattr(j, f"GetBody{side}Rel")().GetTargets()[0])
            p = Gf.Vec3d(getattr(j, f"GetLocalPos{side}Attr")().Get())
            points.append(np.array(cache.GetLocalToWorldTransform(body).Transform(p)))
        gap = float(np.linalg.norm(points[0]-points[1]))
        excluded = j.GetExcludeFromArticulationAttr().Get()
        report["closures"][name] = {"zero_gap_m": gap, "exclude_from_articulation": excluded}
        checks.append(gap < 1e-6 and bool(excluded))
report["total_mass_kg"] = total_mass
checks.extend([len(bodies) == 15, len(report["closures"]) == 4, len(report["motor_parent"]) == 4, len(report["wheel_axes_base"]) == 2, abs(total_mass-model["mass_kg"]) < 1e-5])
report["passed"] = all(checks)
output = ROOT / "mine_model/results/parameter_audit"
output.mkdir(parents=True, exist_ok=True)
(output / "physical_parameters.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
