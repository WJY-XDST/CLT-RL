"""Paused Isaac Sim joint inspector for the exported CAD mechanism."""
import argparse
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", type=int, default=0, help="0 keeps the window open")
parser.add_argument("--self-test", action="store_true", help="Check angle controls and FK before opening")
parser.add_argument("--open-chain", action="store_true", help="Use the original unconstrained FK inspector")
args = parser.parse_args()
closed_chain = (ROOT / "build/closed_chain.usda").exists() and not args.open_chain
if args.self_test and closed_chain:
    parser.error("Kinematic --self-test requires --open-chain; use run_isaac.py --sweep for closed-chain physics")
scene_path = ROOT / "build" / ("closed_chain.usda" if closed_chain else "open_chain.usda")
if not scene_path.exists():
    raise SystemExit("First run: ../run_python.sh run_isaac.py --open-chain --headless --steps 240")

from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "width": 1600, "height": 1000})
try:
    import carb
    import omni.ui as ui
    import omni.usd
    import omni.timeline
    from omni.physx.bindings._physx import SETTING_DISPLAY_JOINTS
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    from pxr import Gf, UsdGeom, UsdPhysics, Sdf

    from run_isaac import load_model, physics_body_transforms
    model = load_model()
    if closed_chain:
        for closure in json.loads((ROOT / "config/closures.json").read_text())["joints"]:
            model["joints"].append({"name": closure["name"], "parent": closure["body0"],
                "child": closure["body1"], "origin": {"xyz": closure["anchor_cad_mm"]},
                "axis": closure["axis_cad"], "actuation": "passive", "closure": True,
                "limit": {"lower": -math.pi, "upper": math.pi}})
    omni.usd.get_context().open_stage(str(scene_path))
    for _ in range(10):
        app.update()
    stage = omni.usd.get_context().get_stage()
    # All inspection edits live in a session layer, leaving the physical asset intact.
    stage.SetEditTarget(stage.GetSessionLayer())
    omni.timeline.get_timeline_interface().stop()
    carb.settings.get_settings().set_bool(SETTING_DISPLAY_JOINTS, True)
    viewport = get_active_viewport()
    camera = UsdGeom.Camera.Define(stage, "/InspectionCamera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))
    camera.CreateFocalLengthAttr(32.0)
    viewport.set_active_camera(str(camera.GetPath()))

    def camera_view(eye, target):
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,1,0)).GetInverse()
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(matrix)

    def overview():
        camera_view([-1.2, 0.55, 1.15], [0.035, -0.065, -0.045])

    overview()
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    markers = {}
    positions = {}
    specs = {j["name"]: j for j in model["joints"]}
    base_origin = model["base_origin_xyz"]
    for spec in model["joints"]:
        name = spec["name"]
        joint = UsdPhysics.RevoluteJoint(joints[name])
        if name in ("RB_joint", "RL_joint", "LB_joint", "LL_joint"):
            expected = stage.GetDefaultPrim().GetPath().AppendChild("base_link")
            if joint.GetBody0Rel().GetTargets() != [expected]:
                raise RuntimeError(f"{name} must be relative to base_link")
        body = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        local_frame = Gf.Matrix4d(1)
        local_frame.SetRotate(Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())))
        local_frame.SetTranslateOnly(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        world_frame = local_frame * UsdGeom.XformCache().GetLocalToWorldTransform(body)
        positions[name] = world_frame.ExtractTranslation()
        marker = UsdGeom.Xform.Define(stage, "/InspectionJoints/" + name)
        marker.AddTransformOp().Set(world_frame)
        color = (0.05, 0.85, 1.0) if spec["actuation"] != "passive" else (1.0, 0.48, 0.05)
        sphere = UsdGeom.Sphere.Define(stage, marker.GetPath().AppendChild("pivot"))
        sphere.CreateRadiusAttr(0.005)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cylinder = UsdGeom.Cylinder.Define(stage, marker.GetPath().AppendChild("axis"))
        cylinder.CreateAxisAttr(joint.GetAxisAttr().Get())
        cylinder.CreateRadiusAttr(0.0018)
        cylinder.CreateHeightAttr(0.09)
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        markers[name] = marker

    current = [model["joints"][0]["name"]]
    details = None
    only_selected = [False]
    angles = {name: 0.0 for name in specs}
    angle_model = ui.SimpleFloatModel(0.0)
    angle_controls = None
    measured_label = None
    syncing_angle = [False]
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    zero_world = {name: UsdGeom.XformCache().GetLocalToWorldTransform(p) for name,p in bodies.items()}
    axes = {"X": Gf.Vec3d(1,0,0), "Y": Gf.Vec3d(0,1,0), "Z": Gf.Vec3d(0,0,1)}
    live_transforms = {}

    def joint_frame(joint, side):
        frame = Gf.Matrix4d(1)
        frame.SetRotate(Gf.Rotation(Gf.Quatd(getattr(joint, f"GetLocalRot{side}Attr")().Get())))
        frame.SetTranslateOnly(Gf.Vec3d(getattr(joint, f"GetLocalPos{side}Attr")().Get()))
        return frame

    def pose_chain():
        # USD uses row-vector transforms: C = inverse(F1) * R(q) * F0 * P.
        # Recompute the whole tree, so descendants retain their relative angles.
        omni.timeline.get_timeline_interface().stop()
        world = {"base_link": zero_world["base_link"]}
        pending = list(model["joints"])
        while pending:
            ready = [spec for spec in pending if spec["parent"] in world]
            if not ready:
                raise RuntimeError("Joint inspector requires an acyclic parent-child tree")
            for spec in ready:
                name = spec["name"]
                joint = UsdPhysics.RevoluteJoint(joints[name])
                pivot_world = joint_frame(joint, 0) * world[spec["parent"]]
                rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(axes[joint.GetAxisAttr().Get()], angles[name]))
                child_world = joint_frame(joint, 1).GetInverse() * rotation * pivot_world
                world[spec["child"]] = child_world
                body = bodies[spec["child"]]
                parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(body.GetParent())
                UsdGeom.Xformable(body).MakeMatrixXform().Set(child_world * parent_world.GetInverse())
                markers[name].MakeMatrixXform().Set(pivot_world)
                positions[name] = pivot_world.ExtractTranslation()
                pending.remove(spec)

    def angle_changed(value):
        if syncing_angle[0]:
            return
        spec = specs[current[0]]
        lower = math.degrees(spec["limit"]["lower"])
        upper = math.degrees(spec["limit"]["upper"])
        requested = value.get_value_as_float()
        if not math.isfinite(requested):
            requested = angles[current[0]]
        angle = max(lower, min(upper, requested))
        angles[current[0]] = angle
        if abs(angle - requested) > 1e-6:
            syncing_angle[0] = True
            angle_model.set_value(angle)
            syncing_angle[0] = False
        if closed_chain:
            if spec["actuation"] != "passive":
                UsdPhysics.DriveAPI(joints[current[0]], "angular").GetTargetPositionAttr().Set(angle)
                if not simulation.is_playing():
                    simulation.play()
        else:
            pose_chain()

    def zero_all():
        angles.update({name: 0.0 for name in angles})
        syncing_angle[0] = True
        angle_model.set_value(0.0)
        syncing_angle[0] = False
        if closed_chain:
            for name, spec in specs.items():
                if spec["actuation"] != "passive":
                    UsdPhysics.DriveAPI(joints[name], "angular").GetTargetPositionAttr().Set(0.0)
            if not simulation.is_playing():
                simulation.play()
        else:
            pose_chain()

    angle_model.add_value_changed_fn(angle_changed)

    def update_markers():
        cache = UsdGeom.XformCache()
        for name, marker in markers.items():
            joint = UsdPhysics.RevoluteJoint(joints[name])
            body = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            transform = joint_frame(joint, 0) * (live_transforms[body.GetName()] if body.GetName() in live_transforms
                                                else cache.GetLocalToWorldTransform(body))
            marker.MakeMatrixXform().Set(transform)
            positions[name] = transform.ExtractTranslation()
            imageable = UsdGeom.Imageable(marker)
            if only_selected[0] and name != current[0]:
                imageable.MakeInvisible()
            else:
                imageable.MakeVisible()

    def select_joint(name):
        current[0] = name
        syncing_angle[0] = True
        angle_model.set_min(math.degrees(specs[name]["limit"]["lower"]))
        angle_model.set_max(math.degrees(specs[name]["limit"]["upper"]))
        angle_model.set_value(angles[name])
        syncing_angle[0] = False
        omni.usd.get_context().get_selection().set_selected_prim_paths([str(joints[name].GetPath())], True)
        spec = specs[name]
        if angle_controls is not None:
            angle_controls.enabled = not closed_chain or spec["actuation"] != "passive"
        pos = spec["origin"]["xyz"]
        limit_text = ("Free revolute closure" if spec.get("closure") else
                      f"Limits: {spec['limit']['lower']:.3f} .. {spec['limit']['upper']:.3f} rad")
        details.text = (f"{name}   [{spec['actuation']}]\n"
                        f"{spec['parent']} -> {spec['child']}\n"
                        f"CAD pivot (mm):\n  {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}\n"
                        f"CAD axis: {spec['axis']}\n"
                        f"{limit_text}\n"
                        f"Angle reference: {spec['parent']}\n"
                        "Zero: exported assembly pose.")
        update_markers()

    def focus():
        pos = positions[current[0]]
        camera_view(list(pos + Gf.Vec3d(-0.32, 0.10, 0.30)), list(pos))

    def select_body(which):
        joint = UsdPhysics.RevoluteJoint(joints[current[0]])
        path = getattr(joint, f"GetBody{which}Rel")().GetTargets()[0]
        omni.usd.get_context().get_selection().set_selected_prim_paths([str(path)], True)

    def hide_base(value):
        base = stage.GetDefaultPrim().GetPath().AppendPath("base_link/visuals")
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(base))
        if value.get_value_as_bool():
            imageable.MakeInvisible()
        else:
            imageable.MakeVisible()

    simulation = None
    if closed_chain:
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import Articulation
        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
        simulation = SimulationContext(physics_dt=1/240, rendering_dt=1/60, stage_units_in_meters=1.0)
        simulation.initialize_physics()
        simulation.play()
        articulation = Articulation(str(stage.GetDefaultPrim().GetPath()))
        articulation.initialize()
        from five_bar import FiveBar
        import numpy as np
        five_bars = {side: FiveBar(side) for side in ("right", "left")}
        five_bar_curves = {}
        for side, color in [("right", (0.95,0.12,0.7)), ("left", (0.15,0.9,0.3))]:
            curve = UsdGeom.BasisCurves.Define(stage, "/EquivalentFiveBar/" + side)
            curve.CreateTypeAttr("linear")
            curve.CreateWrapAttr("nonperiodic")
            curve.CreateCurveVertexCountsAttr([5])
            curve.CreateWidthsAttr([0.003])
            curve.SetWidthsInterpolation("constant")
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            five_bar_curves[side] = curve

    def toggle_five_bar(value):
        prim = UsdGeom.Imageable(stage.GetPrimAtPath("/EquivalentFiveBar"))
        if value.get_value_as_bool():
            prim.MakeVisible()
        else:
            prim.MakeInvisible()

    window = ui.Window("Mine Model - Joint Inspector", width=450, height=860)
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label("CLOSED CHAIN | 18 joints | 6 actuators" if closed_chain else
                     "OPEN CHAIN | paused kinematic preview", height=22)
            ui.Label("Cyan: active   Orange: passive", height=20)
            with ui.HStack(height=28):
                ui.Button("Overview", clicked_fn=overview)
                ui.Button("Right side", clicked_fn=lambda: camera_view([-1.2,-0.08,-0.05],[0.03,-0.08,-0.05]))
                ui.Button("Left side", clicked_fn=lambda: camera_view([1.2,-0.08,-0.05],[0.03,-0.08,-0.05]))
            with ui.HStack(height=22):
                ui.Label("Hide base visuals", width=165)
                check_base = ui.CheckBox(width=25)
                check_base.model.add_value_changed_fn(hide_base)
            with ui.HStack(height=22):
                ui.Label("Only selected axis", width=165)
                check_only = ui.CheckBox(width=25)
                def toggle_only(value):
                    only_selected[0] = value.get_value_as_bool()
                    update_markers()
                check_only.model.add_value_changed_fn(toggle_only)
            if closed_chain:
                with ui.HStack(height=22):
                    ui.Label("Equivalent five-bar", width=165)
                    check_mapping = ui.CheckBox(width=25)
                    check_mapping.model.set_value(True)
                    check_mapping.model.add_value_changed_fn(toggle_five_bar)
            with ui.ScrollingFrame(height=280):
                with ui.VStack(spacing=3):
                    for spec in model["joints"]:
                        name = spec["name"]
                        tag = "P" if spec["actuation"] == "passive" else "A"
                        ui.Button(f"[{tag}] {name}: {spec['parent']} -> {spec['child']}",
                                  height=26, clicked_fn=lambda n=name: select_joint(n))
            with ui.HStack(height=28):
                ui.Button("Focus joint", clicked_fn=focus)
                ui.Button("Select parent", clicked_fn=lambda: select_body(0))
                ui.Button("Select child", clicked_fn=lambda: select_body(1))
            ui.Label("Actuator target (degrees)" if closed_chain else "Joint angle (degrees)", height=22)
            with ui.VStack(height=64, spacing=4) as angle_controls:
                with ui.HStack(height=30):
                    ui.FloatSlider(model=angle_model, min=-90.0, max=90.0, step=0.1)
                    ui.FloatField(model=angle_model, width=85, step=0.1)
                with ui.HStack(height=28):
                    ui.Button("-5 deg", clicked_fn=lambda: angle_model.set_value(angles[current[0]]-5))
                    ui.Button("+5 deg", clicked_fn=lambda: angle_model.set_value(angles[current[0]]+5))
                    ui.Button("Zero joint", clicked_fn=lambda: angle_model.set_value(0.0))
            with ui.HStack(height=28):
                ui.Button("Zero all", clicked_fn=zero_all)
            measured_label = ui.Label("", height=32, word_wrap=True)
            mapping_label = ui.Label("", height=38 if closed_chain else 0, word_wrap=True)
            details = ui.Label("", word_wrap=True, height=170)
            ui.Label("Alt + left drag: orbit | wheel: zoom\n" +
                     ("Passive joints follow the loop constraints." if closed_chain else "Open-chain inspection only."),
                     word_wrap=True, height=40)
    select_joint(current[0])
    if args.self_test:
        max_gap = 0.0
        max_angle_error = 0.0
        for name in specs:
            zero_all()
            select_joint(name)
            angle_model.set_value(15.0)
            cache = UsdGeom.XformCache()
            for other in specs:
                joint = UsdPhysics.RevoluteJoint(joints[other])
                frames = [joint_frame(joint, side) * cache.GetLocalToWorldTransform(
                    stage.GetPrimAtPath(getattr(joint, f"GetBody{side}Rel")().GetTargets()[0])) for side in (0,1)]
                max_gap = max(max_gap, (frames[0].ExtractTranslation()-frames[1].ExtractTranslation()).GetLength())
                relative = frames[1] * frames[0].GetInverse()
                q = relative.ExtractRotationQuat()
                measured = math.degrees(2*math.atan2(Gf.Dot(q.GetImaginary(), axes[joint.GetAxisAttr().Get()]), q.GetReal()))
                max_angle_error = max(max_angle_error, abs(measured-angles[other]))
        zero_all()
        cache = UsdGeom.XformCache()
        reset_error = max(abs(cache.GetLocalToWorldTransform(bodies[name])[i][j]-matrix[i][j])
                          for name,matrix in zero_world.items() for i in range(4) for j in range(4))
        assert max_gap < 1e-6, max_gap
        assert max_angle_error < 1e-4, max_angle_error
        assert reset_error < 1e-5, reset_error
        select_joint(model["joints"][0]["name"])
        result = {"joints_tested": len(specs), "test_angle_deg": 15,
                  "max_pivot_gap_m": max_gap, "max_angle_error_deg": max_angle_error,
                  "max_reset_matrix_error": reset_error}
        (ROOT / "build/angle_control_test.json").write_text(json.dumps(result, indent=2)+"\n")
        print("ANGLE_CONTROL_TEST " + json.dumps(result), flush=True)
    print(f"JOINT_INSPECTOR_READY: {len(specs)} joints, closed_chain={closed_chain}, angle slider enabled", flush=True)
    count = 0
    capture = None
    while app.is_running() and (args.frames == 0 or count < args.frames):
        if closed_chain:
            for _ in range(3):
                simulation.step(render=False)
            simulation.step(render=True)
            live_transforms = physics_body_transforms(articulation)
            actual_q = dict(zip(articulation.dof_names, articulation.get_joint_positions()[0]))
            mapping_status = []
            for side, five_bar in five_bars.items():
                prefix = "R" if side == "right" else "L"
                try:
                    points = five_bar.points([actual_q[n] for n in five_bar.motor_names])
                    scaled = {key: points["O"]+five_bar.scale*(points[key]-points["O"]) for key in ("O","A","P","C")}
                    # Offset along the hinge axis only, so both mechanism sides remain visible.
                    x = live_transforms[prefix+"W_link"].ExtractTranslation()[0] + (-0.025 if side == "right" else 0.025)
                    vertices = [Gf.Vec3f(float(x), float(scaled[key][0]-base_origin[1]*0.001),
                                        float(scaled[key][1]-base_origin[2]*0.001)) for key in ("O","C","P","A","O")]
                    five_bar_curves[side].GetPointsAttr().Set(vertices)
                    actual_w = np.array(live_transforms[prefix+"W_link"].ExtractTranslation())[1:] + np.array(base_origin[1:])*0.001
                    error = np.linalg.norm(actual_w-points["W"])*1000
                    mapping_status.append(f"{prefix}: {error:.3f} mm")
                except ValueError:
                    mapping_status.append(f"{prefix}: singular configuration")
            mapping_label.text = "Five-bar 210 / 250 mm | wheel error\n"+"   ".join(mapping_status)
            name = current[0]
            joint = UsdPhysics.RevoluteJoint(joints[name])
            cache = UsdGeom.XformCache()
            frames = [joint_frame(joint, side) * live_transforms[stage.GetPrimAtPath(
                getattr(joint, f"GetBody{side}Rel")().GetTargets()[0]).GetName()] for side in (0,1)]
            q = (frames[1] * frames[0].GetInverse()).ExtractRotationQuat()
            measured = math.degrees(2*math.atan2(Gf.Dot(q.GetImaginary(), axes[joint.GetAxisAttr().Get()]), q.GetReal()))
            gap = (frames[0].ExtractTranslation()-frames[1].ExtractTranslation()).GetLength()*1000
            measured_label.text = f"Actual: {measured:+.2f} deg | pivot gap: {gap:.3f} mm"
            if specs[name]["actuation"] == "passive":
                syncing_angle[0] = True
                angle_model.set_value(measured)
                syncing_angle[0] = False
            update_markers()
        else:
            app.update()
        count += 1
        if count == 90:
            capture = capture_viewport_to_file(viewport, str(ROOT / "build/joint_inspection.png"))
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
