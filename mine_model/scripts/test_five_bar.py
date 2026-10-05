"""Independent CAD endpoint and finite-difference checks for the five-bar map."""
import unittest
import numpy as np
from five_bar import FiveBar


class FiveBarTests(unittest.TestCase):
    def test_exported_wheel_and_effective_lengths(self):
        for side in ("right", "left"):
            model = FiveBar(side)
            np.testing.assert_allclose(model.points([0,0])["W"], model.wheel_zero, atol=1e-7)
            np.testing.assert_allclose(model.upper*model.scale, [0.210,0.210], atol=1e-7)
            np.testing.assert_allclose(model.lower*model.scale, [0.250,0.250], atol=1e-7)

    def test_cartesian_and_virtual_jacobians(self):
        for side in ("right", "left"):
            model = FiveBar(side)
            for q in np.deg2rad([[0,0],[10,0],[-10,0],[0,10],[0,-10],[7,-8]]):
                for forward, jacobian in ((model.forward, model.jacobian),
                                          (model.virtual_state, model.virtual_jacobian)):
                    delta = np.eye(2)*1e-6
                    numerical = np.column_stack([(forward(q+d)-forward(q-d))/2e-6 for d in delta])
                    np.testing.assert_allclose(jacobian(q), numerical, atol=1e-7)

    def test_mirrored_motor_axis_sign(self):
        right, left = FiveBar("right"), FiveBar("left")
        q = np.deg2rad([8,-6])
        np.testing.assert_allclose(right.forward(q), left.forward(-q), atol=1e-7)
        np.testing.assert_allclose(right.jacobian(q), -left.jacobian(-q), atol=1e-7)

    def test_base_link_reference_is_independent_of_thigh(self):
        for side in ("right", "left"):
            model = FiveBar(side)
            q = np.deg2rad([12,-7])
            theta = model.base_angles(q)
            np.testing.assert_allclose(model.motor_angles_from_base(theta),q,atol=1e-12)
            changed = model.base_angles(q+np.deg2rad([5,0]))
            self.assertAlmostEqual(theta[1], changed[1])


if __name__ == "__main__":
    unittest.main()
