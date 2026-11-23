"""
tools/capture_screenshots.py

Renders every evidence screenshot required by the handbook (Section 16.3).
Headless: no window is opened.

【中文说明】
证据截图生成脚本。手册 16.3 节要求提交一系列截图，这个脚本把 13 张全部一次性生成。
关键点是「无头(headless)」运行：设置 SDL_VIDEODRIVER=dummy，不打开真实窗口，
直接渲染到内存里的 Surface，所以可以在服务器/后台自动跑，不需要显示器。

    screenshots/00_wireframe_cube_maze.png   first wireframe cube + small maze
                                            第一个能跑的 3D 场景：线框立方体 + 小迷宫
    screenshots/01_perspective_full.png     perspective view of the whole maze
                                            透视投影下的整张迷宫
    screenshots/02_orthographic_full.png    same camera, orthographic projection
                                            同一机位的正交投影(对比用)
    screenshots/03_wireframe.png            triangle wireframe of the maze
                                            迷宫的三角形线框
    screenshots/04_depth_buffer.png         z-buffer visualisation
                                            深度缓冲可视化
    screenshots/05_occlusion.png            controlled z-buffer occlusion test
                                            受控的 z-buffer 遮挡实验
    screenshots/06_gameplay.png             worm + apple + bird + water in one frame
                                            一帧里同时有虫子/苹果/鸟/水
    screenshots/07_extension1_fog_on.png    Extension 1: fog ON
                                            扩展 1：雾开
    screenshots/08_extension1_fog_off.png   Extension 1: fog OFF, same frame
                                            扩展 1：同一帧，雾关
    screenshots/09_extension2_nearest.png   Extension 2: nearest sampling
                                            扩展 2：最近邻采样
    screenshots/10_extension2_bilinear.png  Extension 2: bilinear sampling
                                            扩展 2：双线性采样
    screenshots/11_resize_16x9.png          aspect test, 16:9
                                            宽高比测试 16:9
    screenshots/12_resize_4x3.png           aspect test, 4:3
                                            宽高比测试 4:3
"""

import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")     # 无头模式：不创建真实窗口
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont      # Pillow 只用来给截图加说明文字条

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine3d as e3
import game3d as g3

OUT = os.path.join(ROOT, "screenshots")     # 输出目录
os.makedirs(OUT, exist_ok=True)


def save(surface, name):
    # 保存一张截图并打印尺寸，方便确认没有存成空图
    pygame.image.save(surface, os.path.join(OUT, name))
    print("saved", name, surface.get_size())


def label(name, text):
    """Add a caption strip under a screenshot (for before/after evidence)."""
    # 在截图下面拼一条说明条：这样 07/08、09/10 这种「前后对比图」一眼能看出哪张是哪张
    path = os.path.join(OUT, name)
    img = Image.open(path).convert("RGB")       # 读回刚存的图
    bar = 52                                    # 说明条高度(像素)
    canvas = Image.new("RGB", (img.width, img.height + bar), (16, 18, 24))   # 新建更高的画布
    canvas.paste(img, (0, 0))                   # 原图贴在上方
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)   # Windows 粗体
    except Exception:
        font = ImageFont.load_default()         # 没有这个字体就用默认字体
    draw.text((18, img.height + 12), text, fill=(235, 240, 235), font=font)   # 写文字
    canvas.save(path)                           # 覆盖保存


def set_view(g, eye, target, projection="perspective", ortho_height=None):
    # 一次性设置摄像机：投影方式 + 位置/注视点(+ 可选的正交半高)
    g.projection = projection
    g.camera.projection = projection
    if ortho_height is not None:
        g.camera.ortho_height = ortho_height
    g.camera.look(eye, target)


def mini_wireframe_cube():
    """Handbook Appendix E: a cube, a floor and a small handmade maze."""
    # 手册附录 E 的场景：线框立方体 + 地面 + 一个手工搭的 3x3 小迷宫
    # 用途是证明「第一步能跑起来」——所有阶段都至少工作了一次
    r = e3.Renderer(1000, 650, fog_color=(0.10, 0.13, 0.18),
                    sky_top=(0.03, 0.04, 0.07), sky_bottom=(0.07, 0.09, 0.12))   # 深色背景，线框更清楚
    cam = e3.Camera(fov=55.0, near=0.5)
    cam.look((9, 9, -15), (7, 2.0, 8))
    r.begin(cam)
    r.fog_enabled = False                    # 关雾，避免线框被雾色冲淡
    r.wireframe = True                       # 开线框模式
    r.draw_mesh(e3.plane(60, 60), e3.translation(10, 0, 10),
                color=(0.45, 0.55, 0.40), cull=False)     # 地面
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(0.0, 2.6, 3.0) @ e3.scaling(5.2, 5.2, 5.2),
                color=(1.0, 0.85, 0.35))                  # 黄色立方体
    # small 3x3 "maze": outer walls plus two internal partitions
    # 六段墙：四段外框 + 两段内部隔断，拼出一个 3x3 的小迷宫
    walls = [
        (15, 2.0, 3.7, 14.0, 4.0, 0.6),
        (15, 2.0, 16.3, 14.0, 4.0, 0.6),
        (8.2, 2.0, 10.0, 0.6, 4.0, 13.0),
        (21.8, 2.0, 10.0, 0.6, 4.0, 13.0),
        (12.0, 2.0, 9.5, 0.6, 4.0, 6.0),
        (17.0, 2.0, 12.5, 0.6, 4.0, 6.0),
    ]
    for x, y, z, sx, sy, sz in walls:
        # 每个墙段 = 单位立方体 + 平移 + 非等比缩放(拉成长条)
        r.draw_mesh(e3.box((1, 1, 1)),
                    e3.translation(x, y, z) @ e3.scaling(sx, sy, sz),
                    color=(0.55, 0.80, 0.95))
    frame = r.end()
    big = pygame.transform.smoothscale(frame, (1000, 650))
    save(big, "00_wireframe_cube_maze.png")
    label("00_wireframe_cube_maze.png",
          "First runnable 3D scene: wireframe cube and a small maze")


def occlusion_demo():
    """Controlled test: the near cube must hide the cubes behind it."""
    # 受控遮挡实验：三个方块前后排开，最近的红色必须挡住后面的黄色和绿色。
    # 这是「证明 z-buffer 真的在工作」的最直接证据。
    r = e3.Renderer(1000, 620)
    cam = e3.Camera(fov=50.0, near=0.5)
    cam.look((0, 9, -26), (0, 1.5, 0))
    r.begin(cam)
    r.add_directional_light((0.5, -0.9, 0.2))     # 加一盏平行光，否则纯色看不出体积感
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(1.2, 2.0, 7.0) @ e3.scaling(6, 6, 6),
                color=(0.20, 0.65, 0.30))          # 最远：绿色(最大的那个)
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(0.0, 2.0, 0.0) @ e3.scaling(4.5, 4.5, 4.5),
                color=(0.95, 0.75, 0.20))          # 中间：黄色
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(-0.6, 1.6, -6.0) @ e3.scaling(3.2, 3.2, 3.2),
                color=(0.90, 0.25, 0.20))          # 最近：红色(应该挡住前两个)
    frame = r.end()
    save(pygame.transform.smoothscale(frame, (1000, 620)), "05_occlusion.png")
    label("05_occlusion.png",
          "Z-buffer test: the near red cube hides the yellow and green cubes behind it")


def aspect_tests():
    # 宽高比测试：同一场景分别用 16:9 和 4:3 渲染，检查方块没有被拉扁或拉长
    # 这验证了投影矩阵里 `f/aspect` 那一项是对的
    for (w, h), name in (((640, 360), "11_resize_16x9.png"),
                         ((640, 480), "12_resize_4x3.png")):
        r = e3.Renderer(w, h, fog_color=(0.55, 0.62, 0.72))
        cam = e3.Camera(fov=55.0, near=0.5)
        cam.look((10, 8, -16), (0, 2, 0))
        r.begin(cam)
        r.add_directional_light((0.4, -1.0, 0.3))
        for i in range(3):
            # 三个方块排成一排，用来目测比例是否正确
            r.draw_mesh(e3.box((2, 2, 2)),
                        e3.translation(i * 2.5 - 2.5, 1.0, 0),
                        color=(0.90, 0.60, 0.20))
        r.draw_mesh(e3.plane(30, 30), e3.translation(0, 0, 0),
                    color=(0.25, 0.40, 0.25), cull=False)     # 地面
        frame = r.end()
        # 统一放大到宽 900，高度按原比例算，两张图并排比较时比例一致
        scaled = pygame.transform.smoothscale(frame, (900, int(900 * h / w)))
        save(scaled, name)
        label(name, f"Aspect-ratio test {w}x{h}: cubes keep their proportions")


def texture_filter_demo():
    # 扩展 2 对比图：同一张黑白棋盘纹理，最近邻 vs 双线性
    # 造 8x8 纹理，按 2x2 分块做黑白交替
    tex = np.zeros((8, 8, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:8, 0:8]
    tex[((yy // 2) + (xx // 2)) % 2 == 0] = 0.06
    tex[((yy // 2) + (xx // 2)) % 2 == 1] = 0.95
    r = e3.Renderer(900, 600, fog_color=(0, 0, 0))
    cam = e3.Camera(fov=42.0, near=0.5)
    cam.look((0, 17, -10), (0, 0, 1))          # 低角度看地面，让纹理被强烈透视压缩
    for mode, name in (("nearest", "09_extension2_nearest.png"),
                       ("bilinear", "10_extension2_bilinear.png")):
        r.begin(cam)
        r.fog_enabled = False
        r.filter_mode = mode                    # 只改过滤方式，其他完全一样
        r.draw_mesh(e3.plane(30, 30, uv=(3, 3)), None,
                    color=(1, 1, 1), texture=tex, cull=False, lighting=False)   # UV 重复 3 次
        frame = r.end()
        save(pygame.transform.smoothscale(frame, (900, 600)), name)
        label(name, f"Extension 2: {mode} sampling -- 8x8 checker, 3 UV repeats")


def main():
    # 前 4 组是纯渲染器实验(不依赖游戏逻辑)，先生成
    mini_wireframe_cube()
    occlusion_demo()
    aspect_tests()
    texture_filter_demo()

    # 后面几张需要真实游戏场景，所以建一个无头游戏实例
    g = g3.Game(headless=True)
    g.rng = random.Random(2026)      # 固定种子 => 每次生成的迷宫/苹果/鸟位置都一样，截图可复现
    g.build_level(first=True)
    g.reset_player(snap_camera=True)
    g.set_quality(3)                 # 用 NATIVE(1.0) 出图，保证截图是满分辨率
    g.renderer.fog_enabled = True

    cx, cz = g3.MAZE_W * 0.5, g3.MAZE_D * 0.5    # 迷宫中心
    eye = (cx - 560.0, 500.0, -360.0)            # 高空斜俯视的机位
    target = (cx, 0.0, cz)

    # ---- 01 / 02: whole maze, both projections
    # 整张迷宫全景：同一机位分别用透视和正交投影各出一张，作为投影方式的对比证据
    g3.VIEW_DISTANCE = 2600.0        # 临时把绘制距离调很大，否则远处的墙会被剔除掉
    g.renderer.fog_enabled = False   # 关雾，全景图要看清所有细节
    set_view(g, eye, target, "perspective")
    g.draw()
    save(g.screen, "01_perspective_full.png")
    label("01_perspective_full.png",
          "Perspective projection: far walls converge, the whole maze fits the frame")

    set_view(g, eye, target, "orthographic", ortho_height=780.0)   # 正交半高要能覆盖整张迷宫
    g.draw()
    save(g.screen, "02_orthographic_full.png")
    label("02_orthographic_full.png",
          "Orthographic projection from the same camera: parallel edges stay parallel")
    g.renderer.fog_enabled = True    # 恢复默认

    # ---- 03 / 04: wireframe and depth buffer, overview
    # 线框图：证明整张迷宫是用三角形拼出来的
    set_view(g, eye, target, "perspective")
    g.renderer.fog_enabled = False
    g.renderer.wireframe = True
    g.draw()
    save(g.screen, "03_wireframe.png")
    label("03_wireframe.png", "Triangle wireframe: every wall is two textured triangles")
    g.renderer.wireframe = False
    g.renderer.fog_enabled = True

    g3.VIEW_DISTANCE = 330.0         # 恢复正常视距

    # ---- 06: gameplay close-up with worm, apple, bird and water
    # 玩法特写：把虫子、苹果、鸟、水潭「摆」到同一镜头里(否则随机生成很难同时入镜)
    apple = g.apples[0]
    ax, az = apple["pos"][0], apple["pos"][2]
    g.pos = np.array([ax - 6.0, g3.PLAYER_Y, az - 84.0])   # 把玩家挪到苹果前方
    g.heading = 0.0                                        # 朝 +Z
    g.trail = [g.pos.copy() for _ in range(12)]            # 轨迹填满，身体才不会挤成一团

    pool = g.water_pools[0]
    # 把第一个水潭整体平移，挪到镜头范围内
    delta = np.array([ax - 4.0, 0.0, az + 68.0]) - pool["pos"]
    pool["pos"] = pool["pos"] + delta
    pool["model"] = e3.translation(*delta) @ pool["model"]   # 网格的模型矩阵同步平移

    bird = g.birds[0]
    bird["pos"] = np.array([ax - 34.0, g3.BIRD_FLY_HEIGHT + 12.0, az - 40.0])   # 挪到左上方
    bird["heading"] = math.pi * 0.5
    bird["phase"] = -14.0          # catch the wings level for this still frame
    # 调整相位让这一帧的翅膀正好是平展的，静态截图更好看

    g3.VIEW_DISTANCE = 700.0         # 特写需要看得更远一点
    set_view(g, (ax + 44.0, 152.0, az - 176.0), (ax + 6.0, 6.0, az + 26.0),
             "perspective")
    g.time = 2.0                     # 固定时间点，保证水波/旗帜动画相位一致、可复现
    g.draw()
    save(g.screen, "06_gameplay.png")
    label("06_gameplay.png",
          "Gameplay frame: segmented worm, hovering apple, swooping bird and water hazard")

    # ---- 04: z-buffer visualisation from the same close camera
    # 用同一个特写机位出深度缓冲图：直接看到「近处亮、远处暗」的深度分布
    g.renderer.depth_debug = True
    g.draw()
    save(g.screen, "04_depth_buffer.png")
    label("04_depth_buffer.png",
          "Z-buffer: bright yellow = near surfaces, blue = far surfaces")
    g.renderer.depth_debug = False

    # ---- 07 / 08: Extension 1 depth fog
    # 扩展 1 对比图：完全同一帧，只切换雾的开关
    g3.VIEW_DISTANCE = 1500.0
    set_view(g, eye, target, "perspective")
    g.renderer.fog_enabled = True
    g.draw()
    save(g.screen, "07_extension1_fog_on.png")
    label("07_extension1_fog_on.png", "Extension 1: depth fog ON (1 - exp(-(d*z)^2))")

    g.renderer.fog_enabled = False
    g.draw()
    save(g.screen, "08_extension1_fog_off.png")
    label("08_extension1_fog_off.png", "Extension 1: identical frame with fog OFF")
    g.renderer.fog_enabled = True

    pygame.quit()
    print("all screenshots written to", OUT)


if __name__ == "__main__":
    main()      # 直接运行本文件即生成全部截图
