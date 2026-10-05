"""Equivalent coaxial five-bar kinematics of the CAD linkage.

Planar coordinates are CAD (Y, Z), metres. Motor inputs are URDF radians,
ordered (RB, RL) on the right and (LB, LL) on the left. P is the L1/L2 pin;
the parallelogram gives wheel W = O + scale * (P - O).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class FiveBar:
    def __init__(self, side="right"):
        if side not in ("right", "left"):
            raise ValueError("side must be right or left")
        model = json.loads((ROOT / "source/user_model.json").read_text())
        joints = {j["name"]: j for j in model["joints"]}
        closures = json.loads((ROOT / "config/closures.json").read_text())["joints"]
        prefix = "R" if side == "right" else "L"
        self.side = side
        self.reference_frame = "base_link"
        self.motor_names = (prefix+"B_joint", prefix+"L_joint")
        self.sign = -1 if side == "right" else 1
        self.origin = np.array(joints[prefix+"L_joint"]["origin"]["xyz"][1:])*0.001
        a0 = np.array(joints[prefix+"L1_joint"]["origin"]["xyz"][1:])*0.001
        p0 = np.array(joints[prefix+"L2_joint"]["origin"]["xyz"][1:])*0.001
        c0 = np.array(next(c for c in closures if c["name"] == prefix+"L2_"+prefix+"B_closure")["anchor_cad_mm"][1:])*0.001
        k0 = np.array(joints[prefix+"S_joint"]["origin"]["xyz"][1:])*0.001
        self.wheel_zero = np.array(joints[prefix+"W_joint"]["origin"]["xyz"][1:])*0.001
        self.upper = np.array([np.linalg.norm(c0-self.origin), np.linalg.norm(a0-self.origin)])
        self.lower = np.array([np.linalg.norm(p0-c0), np.linalg.norm(p0-a0)])
        self.offsets = np.arctan2(np.array([c0,a0])[:,1]-self.origin[1],
                                  np.array([c0,a0])[:,0]-self.origin[0])
        self.scale = np.linalg.norm(k0-self.origin)/self.upper[0]
        direction = a0-c0
        normal = np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        self.branch = 1 if np.dot(p0-(a0+c0)/2,normal)>0 else -1

    def base_angles(self, motor_angles):
        """Arm directions in base_link YZ: +Y is zero, toward +Z positive.

        These include the CAD assembly offsets. URDF q remains zero at the
        exported pose; both B and L q are relative to base_link, never L to B.
        """
        return self.offsets+self.sign*np.asarray(motor_angles,dtype=float)

    def motor_angles_from_base(self, base_angles):
        """Convert unwrapped base_link arm directions into URDF motor radians."""
        return (np.asarray(base_angles,dtype=float)-self.offsets)/self.sign

    def points(self, motor_angles):
        q = np.asarray(motor_angles, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("Expected two finite motor angles in radians")
        theta = self.offsets+self.sign*q
        c,a = self.origin+self.upper[:,None]*np.column_stack((np.cos(theta),np.sin(theta)))
        delta = a-c
        distance = np.linalg.norm(delta)
        if distance < 1e-9:
            raise ValueError("Five-bar singularity: intermediate pivots coincide")
        along = (self.lower[0]**2-self.lower[1]**2+distance**2)/(2*distance)
        height2 = self.lower[0]**2-along**2
        if height2 < -1e-12:
            raise ValueError("Motor angles have no closed five-bar solution")
        normal = np.array([-delta[1],delta[0]])/distance
        p = c+along*delta/distance+self.branch*np.sqrt(max(0,height2))*normal
        w = self.origin+self.scale*(p-self.origin)
        return {"O": self.origin.copy(), "A": a, "P": p, "C": c, "W": w}

    def forward(self, motor_angles):
        """Wheel position relative to hip, CAD (Y, Z), metres."""
        points = self.points(motor_angles)
        return points["W"]-points["O"]

    def jacobian(self, motor_angles):
        """d(wheel Y,Z) / d(motor B,L), metres/radian."""
        q = np.asarray(motor_angles, dtype=float)
        points = self.points(q)
        theta = self.offsets+self.sign*q
        derivatives = self.sign*self.upper[:,None]*np.column_stack((-np.sin(theta),np.cos(theta)))
        constraints = np.array([points["P"]-points["C"],points["P"]-points["A"]])
        if np.linalg.cond(constraints)>1e8:
            raise ValueError("Five-bar Jacobian is singular")
        rhs = np.diag(np.sum(constraints*derivatives,axis=1))
        return self.scale*np.linalg.solve(constraints,rhs)

    def virtual_state(self, motor_angles):
        """(leg length m, angle rad from CAD -Y toward +Z)."""
        y,z = self.forward(motor_angles)
        return np.array([np.hypot(y,z),np.arctan2(z,-y)])

    def virtual_jacobian(self, motor_angles):
        y,z = self.forward(motor_angles)
        length = np.hypot(y,z)
        if length < 1e-9:
            raise ValueError("Virtual leg length is zero")
        return np.array([[y/length,z/length],[z/length**2,-y/length**2]]) @ self.jacobian(motor_angles)

    def motor_torques(self, motor_angles, radial_force, angular_torque):
        """Virtual work: [tau_B,tau_L] = J_(length,angle)^T [F,T]."""
        return self.virtual_jacobian(motor_angles).T @ np.array([radial_force,angular_torque])
