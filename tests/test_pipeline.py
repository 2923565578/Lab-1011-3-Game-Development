"""
tests/test_pipeline.py

Numerical unit tests for the graphics math and the renderer. These are the
"small automated numerical tests" required by Section 16.1 of the handbook:
they check transforms, projections, cell mapping, jump physics, collision,
z-buffer occlusion and the bilinear filtering extension before the whole game
is debugged visually.

Run with:
    python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine3d as e3
import game3d as g3


class TestTransforms(unittest.TestCase):
    def test_rotate_then_translate(self):
        """Handbook Section 5.2 worked example: (1,0,0) -> (2,0,2)."""
        m = e3.translation(2, 0, 3) @ e3.rotation_y(math.pi / 2)
        p = e3.transform_point(m, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(p, [2.0, 0.0, 2.0], atol=1e-9)

    def test_scale_then_rotate_is_not_commutative(self):
        s = e3.scaling(2, 1, 1)
        r = e3.rotation_y(math.pi / 4)
        a = (r @ s) @ np.array([1.0, 0.0, 0.0, 1.0])
        b = (s @ r) @ np.array([1.0, 0.0, 0.0, 1.0])
        self.assertFalse(np.allclose(a, b))

    def test_direction_ignores_translation(self):
        m = e3.translation(10, 20, 30) @ e3.rotation_y(0.3)
        d = m @ np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(d[3]), 0.0)


class TestCoordinates(unittest.TestCase):
    def test_cell_center_matches_handbook(self):
        """cell_center_3d(2, 1) -> (2.5, 0.0, 1.5) in one-unit-per-cell space."""
        center = np.array([2 + 0.5, 0.0, 1 + 0.5])
        np.testing.assert_allclose(center, [2.5, 0.0, 1.5])

    def test_start_cell_in_world_pixels(self):
        """Original start cell (0,7) maps to (0.5, 7.5) world units = (25, 375)."""
        g = g3.Game(headless=True)
        p = g.maze.center(g.maze.start)
        np.testing.assert_allclose(p, [25.0, 0.0, 375.0], atol=1e-6)
        np.testing.assert_allclose(
            [p[0] / g3.CELL, p[2] / g3.CELL], [0.5, 7.5], atol=1e-9)

    def test_wall_thickness_scaled(self):
        self.assertAlmostEqual(g3.WALL_T / g3.CELL, 0.16, places=6)


class TestProjection(unittest.TestCase):
    def _ndc(self, projection, point):
        cam = e3.Camera(fov=60.0, near=0.1, far=1000.0, projection=projection,
                        ortho_height=5.0)
        cam.look((0, 0, -8), (0, 0, 0))
        clip = cam.projection_matrix(1.6) @ cam.view() @ np.array([*point, 1.0])
        return clip[:3] / clip[3]

    def test_perspective_shrinks_with_distance(self):
        near = self._ndc("perspective", (1.0, 0.0, -7.0))
        far = self._ndc("perspective", (1.0, 0.0, -30.0))
        self.assertGreater(abs(near[0]), abs(far[0]) * 2.5)

    def test_orthographic_keeps_size(self):
        near = self._ndc("orthographic", (1.0, 0.0, -7.0))
        far = self._ndc("orthographic", (1.0, 0.0, -30.0))
        self.assertAlmostEqual(float(near[0]), float(far[0]), places=5)

    def test_perspective_formula_worked_example(self):
        """f=500, camera-space (1,2,5) -> pixel (420,40) at 640x480."""
        f, w, h = 500.0, 640, 480
        sx = w / 2 + f * (1.0 / 5.0)
        sy = h / 2 - f * (2.0 / 5.0)
        self.assertAlmostEqual(sx, 420.0)
        self.assertAlmostEqual(sy, 40.0)


class TestJumpPhysics(unittest.TestCase):
    def test_peak_and_flight_time(self):
        v0, g = g3.JUMP_SPEED, g3.GRAVITY
        self.assertAlmostEqual(g3.JUMP_PEAK, v0 * v0 / (2 * g), places=6)
        self.assertAlmostEqual(g3.JUMP_FLIGHT, 2 * v0 / g, places=6)

    def test_euler_integration_reaches_expected_peak(self):
        dt, h, v = 1.0 / 120.0, 0.0, g3.JUMP_SPEED
        peak = 0.0
        for _ in range(600):
            v -= g3.GRAVITY * dt
            h += v * dt
            peak = max(peak, h)
            if h <= 0:
                break
        self.assertAlmostEqual(peak, g3.JUMP_PEAK, delta=g3.JUMP_PEAK * 0.05)

    def test_jump_is_not_below_water_clearance(self):
        self.assertGreater(g3.JUMP_PEAK, g3.WATER_SAFE_CLEARANCE)


class TestCollision(unittest.TestCase):
    def setUp(self):
        self.g = g3.Game(headless=True)

    def test_wall_blocks_player(self):
        self.assertTrue(self.g.circle_hits_wall(g3.WALL_T * 0.5, g3.CELL * 0.5, 11))

    def test_corridor_center_is_free(self):
        self.assertFalse(self.g.circle_hits_wall(g3.CELL * 0.5, g3.CELL * 0.5, 11))


class TestRenderer(unittest.TestCase):
    def test_z_buffer_hides_far_surface(self):
        r = e3.Renderer(120, 90, fog_color=(0, 0, 0))
        cam = e3.Camera(fov=60.0, near=0.1, far=200.0)
        cam.look((0, 0, -12), (0, 0, 0))
        r.begin(cam)
        r.fog_enabled = False
        far = e3.translation(0, 0, 4) @ e3.scaling(6, 6, 0.1)
        near = e3.translation(0, 0, -4) @ e3.scaling(3, 3, 0.1)
        r.draw_mesh(e3.box((1, 1, 1)), far, color=(1.0, 0.0, 0.0), lighting=False, cull=False)
        r.draw_mesh(e3.box((1, 1, 1)), near, color=(0.0, 0.3, 1.0), lighting=False, cull=False)
        px = r.color[45, 60]
        self.assertGreater(float(px[2]), float(px[0]))

    def test_near_plane_clipping_does_not_explode(self):
        r = e3.Renderer(160, 120)
        cam = e3.Camera(fov=70.0, near=1.0)
        cam.look((0, 6, -4), (0, 0, 6))
        r.begin(cam)
        r.fog_enabled = False
        bg = r.color.copy()
        model = e3.translation(0, 0, 0) @ e3.scaling(100, 1, 100)
        r.draw_mesh(e3.plane(1.0, 1.0), model, color=(0.6, 0.7, 0.3),
                    lighting=False, cull=False)
        self.assertFalse(np.isnan(r.color).any())
        covered = (np.abs(r.color - bg).sum(axis=2) > 0.05).sum()
        self.assertGreater(covered, 0)
        self.assertLess(covered, r.w * r.h * 0.90)

    def test_bilinear_filter_creates_more_grey_levels(self):
        tex = np.zeros((8, 8, 3), dtype=np.float32)
        yy, xx = np.mgrid[0:8, 0:8]
        tex[((yy // 2) + (xx // 2)) % 2 == 0] = 0.05
        tex[((yy // 2) + (xx // 2)) % 2 == 1] = 1.0
        quad = e3.plane(10, 10, uv=(1, 1))
        results = {}
        for mode in ("nearest", "bilinear"):
            r = e3.Renderer(200, 200, fog_color=(0, 0, 0))
            cam = e3.Camera(fov=50.0, near=0.5)
            cam.look((0, 30, -20), (0, 0, 0))
            r.begin(cam)
            r.fog_enabled = False
            r.filter_mode = mode
            r.draw_mesh(quad, None, color=(1, 1, 1), texture=tex, cull=False,
                        lighting=False)
            strip = r.color[80:120, 70:130, 0]
            results[mode] = len(np.unique(np.round(strip * 255).astype(np.uint8)))
        self.assertGreater(results["bilinear"], results["nearest"] * 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
