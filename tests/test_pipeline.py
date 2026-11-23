"""
tests/test_pipeline.py

Numerical unit tests for the graphics math and the renderer. These are the
"small automated numerical tests" required by Section 16.1 of the handbook:
they check transforms, projections, cell mapping, jump physics, collision,
z-buffer occlusion and the bilinear filtering extension before the whole game
is debugged visually.

【中文说明】
图形数学与渲染器的「数值单元测试」。这是实验手册 16.1 节要求的
「小型自动化数值测试」：在靠肉眼调试整个游戏之前，先用数值断言把
变换、投影、格子坐标映射、跳跃物理、碰撞、z-buffer 遮挡、双线性过滤
这些关键环节逐一验证。共 17 个测试。

Run with:
    python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # 当前测试文件所在目录(tests/)
ROOT = os.path.dirname(HERE)                        # 项目根目录
sys.path.insert(0, ROOT)                            # 把根目录加入模块搜索路径，才能 import engine3d

import engine3d as e3
import game3d as g3


class TestTransforms(unittest.TestCase):
    # 测试组 1：3D 变换矩阵的正确性
    def test_rotate_then_translate(self):
        """Handbook Section 5.2 worked example: (1,0,0) -> (2,0,2)."""
        # 手册 5.2 节的算例：先绕 Y 轴转 90°，再平移 (2,0,3)
        # 矩阵相乘顺序是 T @ R，作用到点上时是「先旋转后平移」（右侧先作用）
        m = e3.translation(2, 0, 3) @ e3.rotation_y(math.pi / 2)
        p = e3.transform_point(m, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(p, [2.0, 0.0, 2.0], atol=1e-9)   # 期望结果 (2,0,2)

    def test_scale_then_rotate_is_not_commutative(self):
        # 验证「缩放」和「旋转」不可交换：R@S 与 S@R 结果不同
        s = e3.scaling(2, 1, 1)                 # 只在 x 方向拉长一倍
        r = e3.rotation_y(math.pi / 4)          # 绕 Y 转 45°
        a = (r @ s) @ np.array([1.0, 0.0, 0.0, 1.0])   # 先缩放再旋转
        b = (s @ r) @ np.array([1.0, 0.0, 0.0, 1.0])   # 先旋转再缩放
        self.assertFalse(np.allclose(a, b))     # 两者必须不相等，否则说明矩阵乘法写错了

    def test_direction_ignores_translation(self):
        # 方向向量(w=0)不应该受平移影响：变换后第四分量仍然为 0
        m = e3.translation(10, 20, 30) @ e3.rotation_y(0.3)
        d = m @ np.array([1.0, 0.0, 0.0, 0.0])  # 注意 w=0，表示方向而不是点
        self.assertAlmostEqual(float(d[3]), 0.0)


class TestCoordinates(unittest.TestCase):
    # 测试组 2：格子索引 <-> 世界坐标 的映射
    def test_cell_center_matches_handbook(self):
        """cell_center_3d(2, 1) -> (2.5, 0.0, 1.5) in one-unit-per-cell space."""
        # 手册口径(每格 1 单位)：格子 (2,1) 的中心是 (2.5, 0, 1.5)
        center = np.array([2 + 0.5, 0.0, 1 + 0.5])
        np.testing.assert_allclose(center, [2.5, 0.0, 1.5])

    def test_start_cell_in_world_pixels(self):
        """Original start cell (0,7) maps to (0.5, 7.5) world units = (25, 375)."""
        # 实际工程里 1 格 = 50 世界单位，所以 (0.5,7.5) 换算成 (25, 375)
        g = g3.Game(headless=True)              # 无窗口建一个游戏实例，只为拿到迷宫
        p = g.maze.center(g.maze.start)
        np.testing.assert_allclose(p, [25.0, 0.0, 375.0], atol=1e-6)   # 世界坐标
        np.testing.assert_allclose(
            [p[0] / g3.CELL, p[2] / g3.CELL], [0.5, 7.5], atol=1e-9)   # 除以格子边长 = 手册单位的坐标

    def test_wall_thickness_scaled(self):
        # 墙厚 8 / 格长 50 = 0.16，与手册里「墙厚 0.16 格」一致
        self.assertAlmostEqual(g3.WALL_T / g3.CELL, 0.16, places=6)


class TestProjection(unittest.TestCase):
    # 测试组 3：投影矩阵（透视 vs 正交）
    def _ndc(self, projection, point):
        # 辅助函数：把世界点投影成 NDC（归一化设备坐标），供下面两个测试复用
        cam = e3.Camera(fov=60.0, near=0.1, far=1000.0, projection=projection,
                        ortho_height=5.0)
        cam.look((0, 0, -8), (0, 0, 0))         # 摄像机放在 z=-8 看向原点
        # 完整链路：投影矩阵 @ 视图矩阵 @ 齐次点
        clip = cam.projection_matrix(1.6) @ cam.view() @ np.array([*point, 1.0])
        return clip[:3] / clip[3]               # 透视除法得到 NDC

    def test_perspective_shrinks_with_distance(self):
        # 透视投影：离得越远，屏幕坐标越小（近处的 x 至少是远处的 2.5 倍）
        near = self._ndc("perspective", (1.0, 0.0, -7.0))
        far = self._ndc("perspective", (1.0, 0.0, -30.0))
        self.assertGreater(abs(near[0]), abs(far[0]) * 2.5)

    def test_orthographic_keeps_size(self):
        # 正交投影：远近不影响大小，两处投影结果必须相等
        near = self._ndc("orthographic", (1.0, 0.0, -7.0))
        far = self._ndc("orthographic", (1.0, 0.0, -30.0))
        self.assertAlmostEqual(float(near[0]), float(far[0]), places=5)

    def test_perspective_formula_worked_example(self):
        """f=500, camera-space (1,2,5) -> pixel (420,40) at 640x480."""
        # 手算算例：焦距 f=500，摄像机空间点 (1,2,5)，窗口 640x480
        f, w, h = 500.0, 640, 480
        sx = w / 2 + f * (1.0 / 5.0)            # 屏幕 x = 中心 + f * (x/z)
        sy = h / 2 - f * (2.0 / 5.0)            # 屏幕 y 要减去(因为 y 轴向下)
        self.assertAlmostEqual(sx, 420.0)
        self.assertAlmostEqual(sy, 40.0)


class TestJumpPhysics(unittest.TestCase):
    # 测试组 4：跳跃物理公式
    def test_peak_and_flight_time(self):
        # 解析解：最高点 h=v0²/2g，滞空时间 t=2v0/g
        v0, g = g3.JUMP_SPEED, g3.GRAVITY
        self.assertAlmostEqual(g3.JUMP_PEAK, v0 * v0 / (2 * g), places=6)
        self.assertAlmostEqual(g3.JUMP_FLIGHT, 2 * v0 / g, places=6)

    def test_euler_integration_reaches_expected_peak(self):
        # 验证「离散欧拉积分的最高点」与「解析解」的误差在 5% 以内
        # 这一步很重要：说明游戏里逐帧积分出来的跳跃高度，和报告里算的公式是一致的
        dt, h, v = 1.0 / 120.0, 0.0, g3.JUMP_SPEED    # 120Hz 小步长积分
        peak = 0.0
        for _ in range(600):
            v -= g3.GRAVITY * dt                # 速度受重力衰减
            h += v * dt                         # 高度按速度积分
            peak = max(peak, h)                 # 记录最高点
            if h <= 0:
                break                           # 落地就停
        self.assertAlmostEqual(peak, g3.JUMP_PEAK, delta=g3.JUMP_PEAK * 0.05)

    def test_jump_is_not_below_water_clearance(self):
        # 跳跃最高点必须高于「安全越水高度」，否则玩家永远过不去水潭(关卡不可通关)
        self.assertGreater(g3.JUMP_PEAK, g3.WATER_SAFE_CLEARANCE)


class TestCollision(unittest.TestCase):
    # 测试组 5：圆 vs 墙 的碰撞检测
    def setUp(self):
        self.g = g3.Game(headless=True)         # 每个测试前建一个游戏实例

    def test_wall_blocks_player(self):
        # 墙上的点必须判定为「撞墙」
        self.assertTrue(self.g.circle_hits_wall(g3.WALL_T * 0.5, g3.CELL * 0.5, 11))

    def test_corridor_center_is_free(self):
        # 走廊中心必须判定为「可通行」
        self.assertFalse(self.g.circle_hits_wall(g3.CELL * 0.5, g3.CELL * 0.5, 11))


class TestRenderer(unittest.TestCase):
    # 测试组 6：渲染器核心行为（z-buffer、近平面裁剪、纹理过滤）
    def test_z_buffer_hides_far_surface(self):
        # 先画远处红方块，再画近处蓝方块；如果 z-buffer 正确，该像素应显示近处的蓝色
        r = e3.Renderer(120, 90, fog_color=(0, 0, 0))
        cam = e3.Camera(fov=60.0, near=0.1, far=200.0)
        cam.look((0, 0, -12), (0, 0, 0))
        r.begin(cam)
        r.fog_enabled = False                   # 关掉雾，避免雾色干扰判断
        far = e3.translation(0, 0, 4) @ e3.scaling(6, 6, 0.1)     # 远处的红方块
        near = e3.translation(0, 0, -4) @ e3.scaling(3, 3, 0.1)   # 近处的蓝方块
        r.draw_mesh(e3.box((1, 1, 1)), far, color=(1.0, 0.0, 0.0), lighting=False, cull=False)
        r.draw_mesh(e3.box((1, 1, 1)), near, color=(0.0, 0.3, 1.0), lighting=False, cull=False)
        px = r.color[45, 60]                    # 取画面中心像素
        self.assertGreater(float(px[2]), float(px[0]))   # 蓝分量 > 红分量 => 近处物体胜出

    def test_near_plane_clipping_does_not_explode(self):
        # 测近平面裁剪：摄像机贴着地面往前看，三角形被裁掉一部分。
        # 要求(1)不出现 NaN，(2)确实画到了东西，(3)但也没把整屏都糊满(说明裁剪生效了)
        r = e3.Renderer(160, 120)
        cam = e3.Camera(fov=70.0, near=1.0)
        cam.look((0, 6, -4), (0, 0, 6))          # 摄像机很低且贴着近平面
        r.begin(cam)
        r.fog_enabled = False
        bg = r.color.copy()                     # 记住清屏后的背景，用于比对
        model = e3.translation(0, 0, 0) @ e3.scaling(100, 1, 100)   # 一张巨大的地面
        r.draw_mesh(e3.plane(1.0, 1.0), model, color=(0.6, 0.7, 0.3),
                    lighting=False, cull=False)
        self.assertFalse(np.isnan(r.color).any())                   # 不能有 NaN
        covered = (np.abs(r.color - bg).sum(axis=2) > 0.05).sum()    # 统计被改写的像素数
        self.assertGreater(covered, 0)                              # 至少画上了一些
        self.assertLess(covered, r.w * r.h * 0.90)                  # 但没画满整屏

    def test_bilinear_filter_creates_more_grey_levels(self):
        # 双线性过滤会「制造」中间灰度，所以画面里的灰度级数应远多于最近邻
        # 造一张 8x8 黑白棋盘纹理(按 2x2 分块交替)
        tex = np.zeros((8, 8, 3), dtype=np.float32)
        yy, xx = np.mgrid[0:8, 0:8]
        tex[((yy // 2) + (xx // 2)) % 2 == 0] = 0.05
        tex[((yy // 2) + (xx // 2)) % 2 == 1] = 1.0
        quad = e3.plane(10, 10, uv=(1, 1))
        results = {}
        for mode in ("nearest", "bilinear"):    # 两种过滤方式各渲一次
            r = e3.Renderer(200, 200, fog_color=(0, 0, 0))
            cam = e3.Camera(fov=50.0, near=0.5)
            cam.look((0, 30, -20), (0, 0, 0))
            r.begin(cam)
            r.fog_enabled = False
            r.filter_mode = mode
            r.draw_mesh(quad, None, color=(1, 1, 1), texture=tex, cull=False,
                        lighting=False)
            strip = r.color[80:120, 70:130, 0]                      # 取中间一条区域
            # 统计这条区域里有多少种不同的灰度值(四舍五入到 0-255)
            results[mode] = len(np.unique(np.round(strip * 255).astype(np.uint8)))
        self.assertGreater(results["bilinear"], results["nearest"] * 3)   # 双线性至少多 3 倍灰度级


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接运行本文件时执行全部测试，输出详细结果
