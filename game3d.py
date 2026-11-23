"""
game3d.py -- Worm's Garden Adventure 3D
=================================================================
A fully interactive 3D version of the lab9 2D game (Start2Dto3D.py),
re-themed into a worm-versus-birds adventure.

【中文说明】
把 Lab9 的 2D 迷宫游戏(Start2Dto3D.py)完整升级成可交互的 3D 版本，
主题改成「毛毛虫 vs 小鸟」的冒险。
本文件负责「游戏逻辑 + 场景搭建 + 画面物体摆放」，真正的渲染算法在 engine3d.py。
文件分工：game3d.py(玩法/场景) -> engine3d.py(渲染器) -> textures.py(贴图)。

Controls (all prompts are in English):
    W A S D / Arrow Keys   Move (relative to camera direction)   移动(相对摄像机方向)
    Space                  Physical jump (vertical velocity + gravity)   物理跳跃(初速度+重力)
    Right Mouse Drag       Rotate camera        鼠标右键拖拽 -> 旋转摄像机
    Mouse Wheel            Zoom camera          滚轮 -> 缩放/拉近拉远
    Left Click             Place a 3D firework at that location   左键 -> 在点击处放 3D 烟花
    V / P                  Toggle perspective / orthographic projection  切换透视/正交投影
    T                      Toggle textures on/off    纹理总开关
    F                      Extension 1: depth fog on/off   扩展 1：深度雾开关
    B                      Extension 2: bilinear filtering on/off   扩展 2：双线性过滤开关
    F1 / G                 Wireframe overlay     线框叠加显示
    F3                     Depth-buffer (z-buffer) debug view   深度缓冲调试视图
    F5                     Cycle quality preset (FAST / BALANCED / HIGH / NATIVE)  循环画质档位
    H                      Show technical help panel    显示技术说明面板
    R                      Restart        重开一局
    ESC / Q                Quit           退出
"""

import os
# 在创建窗口之前告诉 SDL 启用「按显示器」的 DPI 感知，否则高分屏上画面会被系统拉伸变糊
os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")

import gc
import math
import random

import numpy as np
import pygame

import engine3d as e3          # 我们自己的软件渲染器(别名 e3，后面到处用)
import textures                # 程序化贴图生成

try:
    # Windows 专用：设置 DPI 感知级别，让窗口按真实像素渲染而不是被系统缩放
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        # 旧版 Windows 没有 shcore，退化成老 API
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass                   # 非 Windows 平台直接忽略，不影响运行

# ============================================================
# Basic Settings  基本设置
# ============================================================
WIDTH, HEIGHT = 1250, 900      # 窗口初始尺寸
TITLE = "Worm's Garden Adventure 3D"   # 窗口标题

# Quality presets: (name, internal render scale). The final frame is always
# drawn into this NumPy framebuffer and then resized to the 1250x900 window.
# 画质档位：(名字, 内部渲染比例)。软件渲染很慢，所以先在小分辨率画布上渲染，
# 再平滑放大到窗口大小 —— 这是本项目的核心渲染优化之一。
QUALITY_LEVELS = (
    ("FAST", 0.20),            # 只画窗口的 20% 像素，最快
    ("BALANCED", 0.30),        # 默认档
    ("HIGH", 0.50),
    ("NATIVE", 1.00),          # 原生分辨率，用于截图/分辨率实验
)
RENDER_SCALE = QUALITY_LEVELS[1][1]     # 启动时用 BALANCED，即 0.30
# Upper bound on shaded pixels for interactive play. The window can be resized
# or maximised freely; above this budget the internal render resolution stops
# growing so that fill rate (the real bottleneck) stays bounded.
# 互动游玩时的像素预算上限：窗口可以随便放大，但内部渲染像素数不能超过这个值，
# 否则填充率(真正的瓶颈)会爆掉。
# 【性能优化】自动降档的下限：设为 1 表示最多降到 BALANCED(0.30)，不会掉回 FAST(0.20)。
# 原因是 BALANCED 每帧约 105~125 ms，必然超过 Q_SLOW(≈83 ms)，旧逻辑一玩就自动降到
# FAST 的 250×180，再被放大 5 倍贴到 1250×900 —— 这就是"画面糊"的真正原因。
QUALITY_FLOOR = 1
PIXEL_BUDGET = 320000
# 【性能优化】像素预算从 132000 提到 320000：HIGH 档(0.50)本来被压到 428×308，
# 提到 320000 后四档才是 250×180 / 375×270 / 625×450 / 1250×900，画质分档符合设计
SHARP_UPSCALE = False          # True 用硬边放大(更快)，False 用平滑放大(更好看)
RENDER_W = int(WIDTH * RENDER_SCALE)    # 初始内部渲染宽
RENDER_H = int(HEIGHT * RENDER_SCALE)   # 初始内部渲染高

FPS = 60                       # 目标帧率(实际远达不到，见画质调节器)
MAX_PARTICLES = 320            # 粒子数量上限，防止烟花开太多把帧率打穿

# Quality governor  自动画质调节器的参数
Q_WINDOW = 50                  # 用最近 50 帧的耗时做判断窗口
Q_COOLDOWN = 2.5               # 切档后 2.5 秒内不再切(冷却)
Q_GRACE = 0.8                  # 启动/重建后的 0.8 秒保护期，统计不参与判断
Q_CONFIRM = 3                  # 连续 3 次判断都超阈值才真正切档(防抖)
Q_SLOW = 1 / 12                # 中位帧时间 > 1/12 秒(约 12 FPS) 视为太慢 -> 降档
Q_FAST = 1 / 34                # 中位帧时间 < 1/34 秒(约 34 FPS) 视为很快 -> 升档

# World size  世界尺寸(单位: 世界单位，1 格 = CELL)
CELL = 50                      # 一个迷宫格的边长
WALL_T = 8                     # 墙的厚度
WALL_H = 40                    # 墙的高度
COLS, ROWS = 23, 15            # 迷宫 23 列 × 15 行
MAZE_W = COLS * CELL           # 迷宫总宽
MAZE_D = ROWS * CELL           # 迷宫总深(Z 方向)

PLAYER_R = 11                  # 玩家(毛毛虫)碰撞半径
PLAYER_Y = 12.0            # sphere-centre height above the ground plane 球心离地高度
PLAYER_SPEED = 190.0           # 移动速度(世界单位/秒)

# --- Physical jump (world units = original pixels, CELL = 50 px = 1 unit) ---
# 物理跳跃：用真实的初速度 + 重力积分，而不是「按一下就上升固定高度」的假跳跃
GRAVITY = 520.0            # world units / s^2   重力加速度
JUMP_SPEED = 168.0         # initial vertical velocity, world units / s   起跳初速度
JUMP_COOLDOWN = 0.35       # 两次跳跃之间的冷却时间(秒)
# Analytical values used in the report: t_flight = 2*v0/g, h_max = v0^2/2g
# 报告里用到的解析解：滞空时间与最高点高度(由初速度和重力直接算出)
JUMP_FLIGHT = 2.0 * JUMP_SPEED / GRAVITY
JUMP_PEAK = JUMP_SPEED * JUMP_SPEED / (2.0 * GRAVITY)
WATER_SAFE_CLEARANCE = 14.0  # feet above this height clear the water surface  脚高于此值可安全越过水面

BIRD_R = 11                    # 小鸟碰撞半径
BIRD_SPEED = 70.0              # 小鸟飞行速度
BIRD_FLY_HEIGHT = 26.0         # 小鸟离地的飞行高度

WATER_R = 15.0                 # 水潭半径
WATER_POOLS = 8                # 水潭数量
TOTAL_APPLES = 4               # 每关苹果数(3 银 + 1 金，见 build_level)
SILVER_VALUE = 50              # 银苹果分数
GOLD_VALUE = 100               # 金苹果分数

VIEW_DISTANCE = 330.0          # 墙体的绘制距离(超出此距离的格子不画，这是距离剔除优化)

C_SKY_TOP = (0.24, 0.42, 0.63)     # 天空顶部颜色(偏蓝)
C_SKY_BOT = (0.80, 0.84, 0.80)     # 天空底部颜色(偏白)
C_FOG = (0.74, 0.79, 0.79)         # 雾色
C_GOLD = (1.00, 0.78, 0.16)        # 金色
C_SILVER = (0.92, 0.30, 0.30)      # 银苹果实际用偏红色(视觉上更好认)

# Worm palette (green, segmented body)  毛毛虫配色(绿色、分节身体)
WORM_GREEN = (0.30, 0.72, 0.26)
WORM_GREEN_DARK = (0.16, 0.42, 0.14)
WORM_GREEN_LIGHT = (0.48, 0.86, 0.34)
WORM_BELLY = (0.72, 0.86, 0.42)
WORM_EYE = (0.96, 0.97, 0.92)
WORM_PUPIL = (0.06, 0.07, 0.06)

# Bird palette (dark body, orange beak)  小鸟配色(深色身体、橙色嘴)
BIRD_BODY = (0.12, 0.13, 0.18)
BIRD_BODY_DARK = (0.06, 0.07, 0.10)
BIRD_BELLY = (0.22, 0.24, 0.30)
BIRD_WING = (0.34, 0.37, 0.48)
BIRD_HORN = (1.00, 0.68, 0.16)   # beak  当作嘴(角)
BIRD_EYE = (1.00, 0.92, 0.40)

# 四个方向的增量与对面方向：N=北(dy=-1)、E=东(dx=+1)、S=南、W=西
DIRS = {"N": (0, -1, "S"), "E": (1, 0, "W"), "S": (0, 1, "N"), "W": (-1, 0, "E")}

def make_font(size, bold=False):
    # 取系统字体；如果 arial 不存在(比如 Linux/精简系统)就退回 pygame 内置字体
    try:
        return pygame.font.SysFont("arial", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)

# ============================================================
# Maze  迷宫
# ============================================================
class Maze:
    def __init__(self, cols, rows, rng):
        # 每个格子记录四面墙是否存在：True = 有墙
        self.cols = cols
        self.rows = rows
        self.rng = rng              # 共用外部传入的随机数发生器，方便复现
        self.cells = [[{"N": True, "E": True, "S": True, "W": True}
                       for _ in range(cols)] for _ in range(rows)]   # 一开始所有格子四面全是墙
        self.start = (0, rows // 2)             # 起点在左边中间
        self.finish = (cols - 1, rows // 2)     # 终点在右边中间
        self.generate()

    def generate(self):
        # 深度优先搜索(DFS)挖迷宫：随机选一个没访问过的邻居，打通之间的墙
        rng = self.rng
        stack = [self.start]                    # 用栈实现回溯式 DFS
        visited = {self.start}
        while stack:
            x, y = stack[-1]                    # 当前所在格子
            nbrs = []                           # 可选的未访问邻居
            for d, (dx, dy, opp) in DIRS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in visited:
                    nbrs.append((nx, ny, d, opp))
            if nbrs:
                nx, ny, d, opp = rng.choice(nbrs)   # 随机挑一个邻居
                self.cells[y][x][d] = False         # 拆掉我这边的墙
                self.cells[ny][nx][opp] = False     # 拆掉邻居那边对应的墙(对称)
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()                     # 走不通就回溯
        # 起点左侧、终点右侧各开一个口，表示「入口」和「出口」
        self.cells[self.start[1]][self.start[0]]["W"] = False
        self.cells[self.finish[1]][self.finish[0]]["E"] = False

    def neighbors(self, cell):
        # 返回该格子四邻中「没有墙隔开」的格子，即实际可以走过去的邻居
        x, y = cell
        out = []
        for d, (dx, dy, opp) in DIRS.items():
            if not self.cells[y][x][d]:         # 这面没墙 -> 可通行
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    out.append((nx, ny))
        return out

    def center(self, cell):
        # 把「格子索引」换算成世界坐标(取格子中心点，y=0 表示地面)
        x, y = cell
        return np.array([x * CELL + CELL * 0.5, 0.0, y * CELL + CELL * 0.5])

    def wall_boxes(self):
        # 遍历所有格子，把每一段墙转成 (中心x, 中心z, 宽, 深) 的矩形盒
        out = []
        for y in range(self.rows):
            for x in range(self.cols):
                w = self.cells[y][x]
                px, pz = x * CELL, y * CELL          # 该格左上角的世界坐标
                if w["N"]:
                    # 北墙：横跨整格宽，中心在格子上边缘
                    out.append((px + CELL * 0.5, pz, CELL + WALL_T, WALL_T))
                if w["W"]:
                    # 西墙：竖直方向一整格高(在 XZ 平面上就是深度方向)
                    out.append((px, pz + CELL * 0.5, WALL_T, CELL + WALL_T))
                # 最右/最下一列/行的墙只由该格自己生成一次，避免重复建墙
                if x == self.cols - 1 and w["E"]:
                    out.append((px + CELL, pz + CELL * 0.5, WALL_T, CELL + WALL_T))
                if y == self.rows - 1 and w["S"]:
                    out.append((px + CELL * 0.5, pz + CELL, CELL + WALL_T, WALL_T))
        return out

    def collision_rects(self):
        # 把墙盒(中心+尺寸)转成 AABB(轴对齐包围盒)的 (x0, z0, x1, z1) 形式，供碰撞检测用
        rects = []
        for cx, cz, sx, sz in self.wall_boxes():
            rects.append((cx - sx * 0.5, cz - sz * 0.5, cx + sx * 0.5, cz + sz * 0.5))
        return rects

def sharpen_surface(surf, amount=0.55):
    """Unsharp mask applied to the small framebuffer before it is upscaled.

    The software renderer runs at a fraction of the window size, so the final
    smoothscale blurs edges. A cheap 4-neighbour high-pass pass restores some
    apparent detail without extra 3D work (cost is proportional to the small
    framebuffer, not to the window).
    """
    # 【中文】USM 锐化(Unsharp Mask)：放大之前先在小帧缓冲上做一次高通增强。
    # 原理：模糊图代表「低频」，原图减去模糊图得到「高频边缘」，把高频加回去就变锐利。
    # 代价只与小画布尺寸成正比，与窗口大小无关，所以很便宜。
    arr = pygame.surfarray.array3d(surf).astype(np.float32)    # 取出像素数据
    # 用 np.roll 把图上下左右各平移 1 像素再平均，得到一张 4 邻域模糊图
    blur = (np.roll(arr, 1, 0) + np.roll(arr, -1, 0)
            + np.roll(arr, 1, 1) + np.roll(arr, -1, 1)) * 0.25
    # 原图 + amount*(原图-模糊图) = 增强边缘，再夹回 0..255 转成 uint8
    out = np.clip(arr + amount * (arr - blur), 0.0, 255.0).astype(np.uint8)
    pygame.surfarray.blit_array(surf, out)      # 写回 surface
    return surf

def shift_mesh(mesh, dx, dy, dz):
    # 把网格整体平移(dx,dy,dz)后返回一个新网格：只改顶点，索引/UV/平滑标志沿用
    return e3.Mesh(mesh.vertices + np.array([dx, dy, dz]), mesh.index, mesh.uvs,
                   smooth=mesh.smooth)

def flag_grid(cols=5, rows=3, width=30.0, height=20.0):
    # 生成一面「软旗子」网格：cols×rows 的细分网格，这样可以对顶点做波浪动画
    # (如果用一整片四边形就没法让它飘起来)
    verts, uvs, faces = [], [], []
    for j in range(rows + 1):
        for i in range(cols + 1):
            verts.append((i / cols * width, -j / rows * height, 0.0))   # 从左上角往下铺
            uvs.append((i / cols, j / rows))                            # UV 均匀铺满
    stride = cols + 1
    for j in range(rows):
        for i in range(cols):
            a = j * stride + i
            b = a + 1
            c = a + stride + 1
            d = a + stride
            faces.append((a, b, c))         # 每个小格拆两个三角形
            faces.append((a, c, d))
    return e3.Mesh(verts, faces, uvs, smooth=False)

# ============================================================
# Game  游戏主体
# ============================================================
class Game:
    def __init__(self, headless=False):
        # headless=True 时不显示窗口，用于自动截图/录像
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)   # 可缩放窗口
        self.clock = pygame.time.Clock()
        self.headless = headless

        self.font = make_font(22)                 # 顶部 HUD 用
        self.small = make_font(16)                # 小字/提示条用
        self.title_font = make_font(34, bold=True)   # 大字横幅用

        # 创建软件渲染器：内部用缩小的分辨率，但天空/雾/环境光参数与窗口一致
        self.renderer = e3.Renderer(
            RENDER_W, RENDER_H,
            sky_top=C_SKY_TOP, sky_bottom=C_SKY_BOT,
            fog_color=C_FOG, fog_density=0.0016,
            ambient=(0.22, 0.24, 0.29),
        )
        self.renderer.filter_mode = "nearest"     # Extension 2 switch  扩展2：默认最近邻
        self.renderer.depth_debug = False         # F3  深度调试默认关
        self.quality = 1            # 默认 BALANCED
        self.camera = e3.Camera(fov=62.0, near=1.0, far=1400.0)   # 环绕式摄像机
        self.scaled = pygame.Surface((WIDTH, HEIGHT)).convert()   # 放大后的中间画布
        self.fullscreen = False
        self._windowed_size = (WIDTH, HEIGHT)      # 记住窗口模式下的尺寸，退出全屏时恢复
        self.sharpen = True               # C: unsharp mask before upscaling  默认开锐化

        # 自动画质调节器的运行时状态
        self._dt_hist = []           # 最近若干帧的耗时
        self._slow_run = 0           # 连续「太慢」计数
        self._fast_run = 0           # 连续「太快」计数
        self._q_cool = 0.0           # 冷却计时
        self._q_grace = 0.0          # 保护期计时

        # ---- Mesh resources ----  预先建好所有网格，避免每帧重复构造
        self.box_h = e3.box((CELL + WALL_T, WALL_H, WALL_T))   # 横向墙段
        self.box_v = e3.box((WALL_T, WALL_H, CELL + WALL_T))   # 纵向墙段
        self.box_unit = e3.box((1.0, 1.0, 1.0))                # 单位立方体(翅膀用，靠缩放变形)
        self.sphere_lo = e3.sphere(1.0, 8, 4, smooth=True)     # 低精度球(身体/苹果)
        self.sphere_md = e3.sphere(1.0, 10, 5, smooth=True)    # 中精度球
        self.sphere_sm = e3.sphere(1.0, 5, 3, smooth=True)     # 低精度小球(眼睛/珠子)
        self.cone_sm = e3.cone(1.0, 1.0, 6)                    # 圆锥(嘴/角/尾巴)
        self.cyl_sm = e3.cylinder(1.0, 1.0, 6, smooth=True)    # 圆柱(旗杆/苹果梗)
        self.disc_shadow = e3.disc(1.0, 12, 1)                 # 圆盘(接触阴影)
        self.leaf_mesh = e3.quad((0, 0, 0), (9, 5), normal=(0, 0.4, 1))   # 叶片(单片四边形)

        # ---- Textures ----  贴图全部程序化生成，首次 get 时才真正计算
        self.tex_stone = textures.get("stone")
        self.tex_floor = textures.get("floor")
        self.tex_grass = textures.get("grass")
        self.tex_checker = textures.get("checker")
        self.tex_wood = textures.get("wood")
        self.tex_water = textures.get("water")
        self.tex_cloth = textures.get("cloth")

        # ---- Camera / State ----  摄像机轨道参数(球坐标：偏航/俯仰/距离)
        self.cam_yaw = 0.0            # 水平环绕角
        self.cam_pitch = -0.92        # 俯视角(负值表示从上往下看)
        self.cam_dist = 138.0         # 摄像机到目标的距离
        self.cam_target = np.array([0.0, 18.0, 0.0])   # 摄像机注视点
        self.projection = "perspective"   # 当前投影方式
        self.show_help = False            # 是否显示技术面板

        self.rng = random.Random()        # Python 标准库随机数(游戏逻辑用)
        self.score = 0                    # 总分
        self.level = 1                    # 关卡号
        self.time = 0.0                   # 本关计时
        self.banner = ""                  # 屏幕中央横幅文字
        self.banner_t = 0.0               # 横幅剩余显示时间
        self.running = True               # 主循环开关

        self.particles = []               # 粒子列表
        self.firework_lights = []         # 烟花临时点光源
        self.dying = 0.0                  # 死亡后的短暂停顿计时

        self.leg_amp = 0.0                # 行走动画幅度(0~1，用于身体摆动)
        self.vspeed = 0.0                 # 竖直速度(跳跃用)
        self.trail = []                   # 头部历史轨迹，用于让身体分节跟随

        # Caches  各种缓存：重建时清空、每帧复用，避免重复计算
        self._cells = []                  # (网格, 模型矩阵, 中心) 的墙体列表
        self._cell_centers = np.zeros((0, 3))   # 各墙体格子的中心坐标(用于距离剔除)
        self._ground_models = []          # 地面四块的模型矩阵
        self._topbar = None               # 顶部 HUD 条缓存
        self._hint_bar = None             # 底部提示条背景缓存
        self._hint_txt = None             # 底部提示文字缓存
        self._banner_cache = None         # 横幅渲染结果缓存
        self._banner_cache_key = None     # 缓存对应的横幅文字(变了才重画)
        self._minimap_static = None       # 小地图静态部分(墙体/水潭)缓存
        self._minimap_dirty = True        # 静态部分是否需要重建

        self.build_level(first=True)      # 生成第一关
        self.reset_player(snap_camera=True)   # 把玩家放到起点
        self.time = 0.0

        gc.collect()      # 先做一次完整垃圾回收
        gc.freeze()       # 然后把现有对象冻结，后续 GC 不再扫描它们 —— 减少卡顿

    # --------------------------------------------------------
    def build_level(self, first=False):
        # 生成一关的全部内容：迷宫、墙网格、地面、水潭、苹果、鸟、旗子
        rng = self.rng
        self.maze = Maze(COLS, ROWS, rng)
        self.start_cell = self.maze.start
        self.finish_cell = self.maze.finish

        self.collision = self.maze.collision_rects()      # 所有墙的 AABB
        # 碰撞网格索引：把墙按它覆盖的格子分类，检测时只需查附近 9 个格子，避免遍历全部墙
        self.collision_grid = {}
        for rect in self.collision:
            gx0 = int(math.floor(rect[0] / CELL))
            gx1 = int(math.floor(rect[2] / CELL))
            gz0 = int(math.floor(rect[1] / CELL))
            gz1 = int(math.floor(rect[3] / CELL))
            for gx in range(gx0, gx1 + 1):
                for gz in range(gz0, gz1 + 1):
                    self.collision_grid.setdefault((gx, gz), []).append(rect)

        # 按格子把墙体合并成「一格一个网格」：这样剔除和绘制都以格为单位，调用次数大幅减少
        self.cell_meshes = {}
        by_cell = {}
        for cx, cz, sx, sz in self.maze.wall_boxes():
            gx = int(cx // CELL)
            gz = int(cz // CELL)
            by_cell.setdefault((gx, gz), []).append((cx, cz, sx, sz))

        for (gx, gz), boxes in by_cell.items():
            cc = self.maze.center((gx, gz))     # 该格中心(合并后作为局部原点)
            parts = []
            for cx, cz, sx, sz in boxes:
                src = self.box_h if sx > sz else self.box_v      # 横墙用横向盒，竖墙用纵向盒
                # 把墙段移到「相对格子中心」的位置，向上抬到墙高一半(盒是以中心为原点的)
                parts.append(shift_mesh(src, cx - cc[0], WALL_H * 0.5, cz - cc[2]))
            mesh = e3.merge(parts) if len(parts) > 1 else parts[0]   # 多段就合并成一个网格
            self.cell_meshes[(gx, gz)] = (mesh, cc)

        # 存成 (网格, 平移矩阵, 中心) 三元组，绘制时矩阵直接可用
        self._cells = [(mesh, e3.translation(cc[0], 0.0, cc[2]), cc)
                       for (mesh, cc) in self.cell_meshes.values()]
        if self._cells:
            self._cell_centers = np.array([c[2] for c in self._cells])
        else:
            self._cell_centers = np.zeros((0, 3))

        # 地面：迷宫本体用细分的贴图平面，外面再用 4 块大平面把视野铺满(避免看到世界边缘)
        self.ground = e3.plane(1.0, 1.0)       # 单位平面，靠缩放矩阵变成任意大小
        self.ground_tiles = [
            (MAZE_W * 0.5, -1800.0, MAZE_W + 3600.0, 3600.0),   # 北侧
            (MAZE_W * 0.5, MAZE_D + 1800.0, MAZE_W + 3600.0, 3600.0),   # 南侧
            (-1800.0, MAZE_D * 0.5, 3600.0, MAZE_D),           # 西侧
            (MAZE_W + 1800.0, MAZE_D * 0.5, 3600.0, MAZE_D),   # 东侧
        ]
        # 每块地面的模型矩阵 = 平移到指定位置 + 缩放到指定尺寸(y 缩放为 1，保持水平)
        self._ground_models = [
            (e3.translation(cx, -1.0, cz) @ e3.scaling(w, 1.0, d), w, d)
            for (cx, cz, w, d) in self.ground_tiles
        ]
        self.floor_mesh = self.build_tiled_floor(8, 5)      # 迷宫地面细分成 8×5 块
        self.leaf_mesh = e3.quad((0, 0, 0), (9, 5), normal=(0, 0.4, 1))

        # ---- Water pools ----  水潭：随机挑若干格子(不能是起点/终点)
        self.water_pools = []
        cells = [(x, y) for y in range(ROWS) for x in range(COLS)
                 if (x, y) not in (self.start_cell, self.finish_cell)]
        rng.shuffle(cells)                  # 打乱后取前 N 个
        for cell in cells[:WATER_POOLS]:
            p = self.maze.center(cell)
            p[1] = 0.6                      # 水面临微微高于地面，避免共面闪烁(z-fighting)
            self.water_pools.append({
                "pos": p, "cell": cell, "phase": rng.uniform(0, 6.28),   # phase 让各水潭波动不同步
                "mesh": e3.disc(1.0, 12, 2),                             # 圆盘网格(顶点可上下动)
                "model": e3.translation(*p) @ e3.scaling(WATER_R, 1.0, WATER_R),
            })
        water_cells = {p["cell"] for p in self.water_pools}   # 记下水潭所在格子，苹果不能放上面

        # ---- Apples ----  苹果：3 个银 + 1 个金，随机放在非水潭格子
        self.apples = []
        cells = [c for c in cells if c not in water_cells]
        rng.shuffle(cells)
        kinds = [("silver", SILVER_VALUE)] * 3 + [("gold", GOLD_VALUE)]
        for cell, (kind, value) in zip(cells, kinds):
            p = self.maze.center(cell)
            self.apples.append({
                "pos": p, "type": kind, "value": value,
                "collected": False, "phase": rng.uniform(0, 6.28),   # phase 控制上下浮动/自转的相位
            })
        self.medals = len(self.apples)      # 需要收集的苹果总数(集齐才能通关)

        # ---- Birds ----  小鸟：难度随关卡增加(2+关卡，最多 5 只)
        self.birds = []
        count = min(2 + self.level, 5)
        # 只在离起点较远的格子放鸟，避免一开局就被撞死
        far_cells = [c for c in cells
                     if abs(c[0] - self.start_cell[0]) + abs(c[1] - self.start_cell[1]) > 5]
        rng.shuffle(far_cells)
        for cell in far_cells[:count]:
            nbrs = self.maze.neighbors(cell)
            if not nbrs:
                continue                    # 该格没有可走邻居，跳过
            target = rng.choice(nbrs)       # 初始目标：相邻的一个格子
            p = self.maze.center(cell)
            p[1] = BIRD_FLY_HEIGHT          # 抬到飞行高度
            self.birds.append({
                "cell": cell, "target": target, "prev": cell,        # 当前位置/目标/上一格(用于避免立刻折返)
                "pos": p, "speed": BIRD_SPEED + rng.uniform(-10, 18),   # 每只鸟速度略有差异
                "heading": rng.uniform(0, 6.28), "phase": rng.uniform(0, 6.28),   # 朝向/翅膀相位
            })

        # ---- Flags ----  起点绿旗和终点方格旗，位置相对格子中心略微偏移
        self.start_flag = self.maze.center(self.start_cell) + np.array(
            [-CELL * 0.42, 0.0, CELL * 0.40])
        self.finish_flag = self.maze.center(self.finish_cell) + np.array(
            [CELL * 0.56, 0.0, 0.0])
        self.flag_start_mesh = flag_grid()     # 两面旗各自独立网格(因为顶点动画会改坐标)
        self.flag_finish_mesh = flag_grid()

        self.collected = 0                     # 本关已收集苹果数
        self.time = 0.0                        # 本关计时归零
        self._minimap_dirty = True             # 小地图需要重建
        if not first:
            # 不是第一关时弹一个「LEVEL n」横幅
            self.banner = f"LEVEL {self.level}"
            self.banner_t = 2.2

    def build_tiled_floor(self, nx, ny):
        # 把地面细分成 nx*ny 块小四边形：逐块做雾插值，雾的过渡比一整块更自然
        verts, uvs, faces = [], [], []
        for j in range(ny):
            for i in range(nx):
                x0 = MAZE_W * i / nx
                x1 = MAZE_W * (i + 1) / nx
                z0 = MAZE_D * j / ny
                z1 = MAZE_D * (j + 1) / ny
                base = len(verts)
                verts += [(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]   # 四个角
                # UV 按「每格 CELL 一个平铺单位」展开，所以贴图会按格子重复
                uvs += [(x0 / CELL, z0 / CELL), (x1 / CELL, z0 / CELL),
                        (x1 / CELL, z1 / CELL), (x0 / CELL, z1 / CELL)]
                faces += [(base, base + 2, base + 1), (base, base + 3, base + 2)]   # 两个三角形，法线朝上
        return e3.Mesh(verts, faces, uvs, smooth=False)

    # --------------------------------------------------------
    def _make_renderer(self):
        """(Re)build the framebuffer renderer at the current quality level."""
        # 按当前画质档位重建渲染器(换分辨率)，并继承旧的各个开关状态
        s = QUALITY_LEVELS[self.quality][1]        # 目标缩放比例
        # NATIVE is kept exact (screenshots / resolution experiments); every
        # other preset is clamped so a maximised window cannot multiply the
        # shaded pixel count.
        # NATIVE 保持精确(用于截图和分辨率实验)；其他档位要按像素预算压住比例，
        # 这样把窗口最大化也不会让着色像素数成倍增长。
        if self.quality < len(QUALITY_LEVELS) - 1:
            cap = math.sqrt(PIXEL_BUDGET / float(WIDTH * HEIGHT))   # 预算允许的最大比例
            s = max(0.16, min(s, cap))             # 下限 0.16，避免太小看不清
        old = getattr(self, "renderer", None)       # 旧渲染器(第一次调用时为 None)
        r = e3.Renderer(
            max(160, int(WIDTH * s)), max(120, int(HEIGHT * s)),   # 内部帧缓冲尺寸
            sky_top=C_SKY_TOP, sky_bottom=C_SKY_BOT,
            fog_color=C_FOG, fog_density=0.0016,
            ambient=(0.22, 0.24, 0.29),
        )
        if old is not None:
            # 迁移用户设置：纹理/雾/线框/过滤/深度调试，否则切档位会把开关重置
            r.textures_enabled = old.textures_enabled
            r.fog_enabled = old.fog_enabled
            r.wireframe = old.wireframe
            r.filter_mode = old.filter_mode
            r.depth_debug = old.depth_debug
        self.renderer = r

    def set_quality(self, q):
        # 切换画质档位：夹到合法范围，重建渲染器，并重置调节器状态
        q = max(0, min(len(QUALITY_LEVELS) - 1, q))
        if q == self.quality:
            return                       # 档位没变就不做任何事
        self.quality = q
        self._make_renderer()
        self._dt_hist.clear()            # 清空耗时统计
        self._slow_run = 0
        self._fast_run = 0
        self._q_cool = Q_COOLDOWN        # 进入冷却
        self._q_grace = Q_GRACE          # 进入保护期

    # --------------------------------------------------------
    def _sync_layout(self):
        """Follow a new window size: rebuild HUD caches and the framebuffer."""
        # 窗口尺寸变化后调用：更新全局 WIDTH/HEIGHT、重建 HUD 缓存和帧缓冲
        global WIDTH, HEIGHT
        WIDTH, HEIGHT = self.screen.get_size()      # 读取新的真实窗口尺寸
        self.scaled = pygame.Surface((WIDTH, HEIGHT)).convert()   # 新的放大画布
        # 所有与尺寸相关的缓存全部作废
        self._topbar = None
        self._hint_bar = None
        self._hint_txt = None
        self._minimap_static = None
        self._minimap_dirty = True
        self._banner_cache_key = None
        self._make_renderer()            # 按新尺寸重建渲染器
        self._dt_hist.clear()
        self._q_cool = max(self._q_cool, 0.6)   # 至少冷却 0.6 秒，避免刚改尺寸就误判降档

    def resize_window(self, w, h):
        # 设置窗口大小，但给一个下限(800×600)，否则太小无法看
        self.screen = pygame.display.set_mode((max(800, int(w)), max(600, int(h))),
                                              pygame.RESIZABLE)
        self._sync_layout()

    def toggle_fullscreen(self):
        # 全屏/窗口切换，两种模式都记住各自尺寸
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self._windowed_size = self.screen.get_size()          # 记住窗口尺寸
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)   # (0,0) = 当前分辨率
            self.banner = "Fullscreen (F11 to exit)"
        else:
            self.screen = pygame.display.set_mode(self._windowed_size,
                                                  pygame.RESIZABLE)   # 恢复到原窗口尺寸
            self.banner = "Windowed mode"
        self.banner_t = 1.4
        self._sync_layout()

    def auto_quality(self, dt):
        # 自动画质调节：根据最近帧时间的中位数升降档，保证交互帧率
        if self._q_cool > 0.0:
            self._q_cool -= dt
            return                       # 冷却中，不判断
        if self._q_grace > 0.0:
            self._q_grace -= dt
            return                       # 保护期中，不判断(避免刚重建完就误判)

        self._dt_hist.append(dt)         # 记录这一帧耗时
        if len(self._dt_hist) > Q_WINDOW:
            self._dt_hist.pop(0)         # 保持定长窗口
        if len(self._dt_hist) < Q_WINDOW:
            return                       # 数据不够，先攒满

        med = sorted(self._dt_hist)[Q_WINDOW // 2]     # 取中位数(比平均值抗瞬时毛刺)
        if med > Q_SLOW:
            self._slow_run += 1          # 太慢：累加慢计数，清零快计数
            self._fast_run = 0
        elif med < Q_FAST:
            self._fast_run += 1          # 太快：反过来
            self._slow_run = 0
        else:
            self._slow_run = 0           # 在正常区间：两个计数都清零
            self._fast_run = 0

        # 【性能优化】降档条件是 quality > QUALITY_FLOOR(而不是 > 0)，
        # 即最低只降到 BALANCED，避免画面被放大得过糊
        if self._slow_run >= Q_CONFIRM and self.quality > QUALITY_FLOOR:
            self.set_quality(self.quality - 1)      # 连续够多次就降一档
        elif self._fast_run >= Q_CONFIRM and self.quality < len(QUALITY_LEVELS) - 1:
            self.set_quality(self.quality + 1)      # 连续够多次就升一档

    # --------------------------------------------------------
    def reset_player(self, snap_camera=False):
        # 把玩家状态复位到起点
        p = self.maze.center(self.start_cell)
        self.pos = np.array([p[0], PLAYER_Y, p[2]])   # 位置(球心高度)
        self.heading = math.pi * 0.5                  # 初始朝向
        self.vspeed = 0.0                             # 竖直速度清零
        self.jump_h = 0.0                             # 跳跃高度清零
        self.jump_cd = 0.0                            # 跳跃冷却清零
        self.spark_t = 0.0                            # 脚步粒子发射计时
        self.walk_time = 0.0                          # 行走动画时间
        self.leg_amp = 0.0                            # 行走幅度
        self.trail = [self.pos.copy() for _ in range(10)]   # 轨迹填满初始位置
        if snap_camera:
            # 立刻把摄像机目标对准玩家(不做平滑)，否则开局镜头会从旧位置飞过来
            self.cam_target = self.pos + np.array([0.0, 16.0, 0.0])
        self.dying = 0.0

    def restart_run(self):
        # R 键：完全重开(分数、关卡、场景、粒子全清)
        self.score = 0
        self.level = 1
        self.build_level(first=True)
        self.reset_player(snap_camera=True)
        self.particles.clear()
        self.firework_lights.clear()

    # --------------------------------------------------------
    def handle_event(self, event):
        # 所有输入事件的总入口：键盘、鼠标、窗口事件
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
            self.resize_window(event.w, event.h)      # 用户拖拽窗口大小
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False                  # 退出
            elif event.key == pygame.K_r:
                self.restart_run()                    # 重开
            elif event.key == pygame.K_SPACE:
                # 空格跳跃：只有落地且冷却结束才能跳
                if self.jump_h <= 0.5 and self.jump_cd <= 0:
                    self.vspeed = JUMP_SPEED          # 给一个向上的初速度
                    self.jump_cd = JUMP_COOLDOWN      # 进入冷却
                    self.burst(self.pos + np.array([0, -8, 0]), 10,
                               (0.6, 1.0, 0.5), speed=70, up=90)   # 起跳处喷一圈粒子
            elif event.key in (pygame.K_v, pygame.K_p):
                # V/P 切换投影方式，并提示当前是哪种
                self.projection = ("orthographic" if self.projection == "perspective"
                                   else "perspective")
                self.banner = ("Orthographic Projection" if self.projection == "orthographic"
                               else "Perspective Projection")
                self.banner_t = 1.6
            elif event.key == pygame.K_t:
                # T 开关纹理
                self.renderer.textures_enabled = not self.renderer.textures_enabled
                self.banner = "Textures: " + ("ON" if self.renderer.textures_enabled else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_f:
                # F 开关扩展 1(深度雾)
                self.renderer.fog_enabled = not self.renderer.fog_enabled
                self.banner = "Fog (extension 1): " + ("ON" if self.renderer.fog_enabled else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_b:
                # B 开关扩展 2(双线性过滤)
                r = self.renderer
                r.filter_mode = "bilinear" if r.filter_mode == "nearest" else "nearest"
                self.banner = "Texture filter (extension 2): " + r.filter_mode.upper()
                self.banner_t = 1.4
            elif event.key in (pygame.K_g, pygame.K_F1):
                # 线框模式切换
                self.renderer.wireframe = not self.renderer.wireframe
                self.banner = "Wireframe: " + ("ON" if self.renderer.wireframe else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_F3:
                # 深度缓冲可视化
                self.renderer.depth_debug = not self.renderer.depth_debug
                self.banner = "Depth buffer: " + ("ON" if self.renderer.depth_debug else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_F5:
                # 循环画质档位
                self.set_quality((self.quality + 1) % len(QUALITY_LEVELS))
                self.banner = f"Quality: {QUALITY_LEVELS[self.quality][0]}"
                self.banner_t = 1.2
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help      # 技术面板开关
            elif event.key == pygame.K_F11:
                self.toggle_fullscreen()
            elif event.key == pygame.K_c:
                # C 开关放大前锐化
                self.sharpen = not self.sharpen
                self.banner = "Sharpen: " + ("ON" if self.sharpen else "OFF")
                self.banner_t = 1.2
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # 左键：把屏幕坐标反算成地面点，在那里放烟花
                ground = self.screen_to_ground(event.pos)
                if ground is not None:
                    self.create_firework(ground)
            elif event.button == 4:
                self.cam_dist = max(70.0, self.cam_dist * 0.9)      # 滚轮上滚 -> 拉近
            elif event.button == 5:
                self.cam_dist = min(280.0, self.cam_dist * 1.1)     # 下滚 -> 拉远(有上下限)
        elif event.type == pygame.MOUSEMOTION:
            buttons = pygame.mouse.get_pressed()
            if buttons[2]:
                # 按住右键拖拽 -> 环绕旋转摄像机
                self.cam_yaw -= event.rel[0] * 0.005               # 横向拖动改偏航
                self.cam_pitch = min(-0.30, max(-1.28,
                                                self.cam_pitch - event.rel[1] * 0.004))   # 纵向拖动改俯仰(夹在上下限内)

    # --------------------------------------------------------
    def screen_to_ground(self, mouse):
        # 屏幕点 -> 地面交点：本质是「反投影」，把屏幕坐标反算出一条世界空间射线，再与 y=0 平面求交
        x = (mouse[0] / WIDTH) * 2.0 - 1.0        # 归一化到 NDC 的 x [-1,1]
        y = 1.0 - (mouse[1] / HEIGHT) * 2.0       # 归一化到 NDC 的 y [-1,1]（注意翻转）
        inv = np.linalg.inv(self.renderer.mvp)    # MVP 的逆矩阵
        # 取近裁面(z=-1)和远裁面(z=+1)上各一点，反变换回世界空间得到射线两端
        p0 = inv @ np.array([x, y, -1.0, 1.0])
        p1 = inv @ np.array([x, y, 1.0, 1.0])
        p0 /= p0[3]                               # 齐次坐标归一化(w 除回去)
        p1 /= p1[3]
        d = p1[:3] - p0[:3]                       # 射线方向
        if abs(d[1]) < 1e-6:
            return None                           # 射线与地面平行，无交点
        t = (0.0 - p0[1]) / d[1]                  # 求与 y=0 平面的参数 t
        if t <= 0:
            return None                           # 交点在身后，无效
        hit = p0[:3] + d * t                      # 交点坐标
        # 限制在迷宫附近，防止点到天上点到很远的地方生成一堆粒子
        hit[0] = min(max(hit[0], -200.0), MAZE_W + 200.0)
        hit[2] = min(max(hit[2], -200.0), MAZE_D + 200.0)
        return hit

    def create_firework(self, pos):
        # 生成一束 3D 烟花：60~95 个粒子向四周炸开，同时挂一个短寿命点光源
        colors = [(0.25, 0.6, 1.0), (1.0, 0.35, 0.6), (0.7, 0.3, 1.0), (1.0, 0.8, 0.25)]
        col = self.rng.choice(colors)                     # 每次随机选一种主色
        for _ in range(self.rng.randint(60, 95)):
            ang = self.rng.uniform(0, math.tau)           # 水平角(0..2π)
            sp = self.rng.uniform(40, 190)                # 水平速度
            vel = np.array([math.cos(ang) * sp,
                            self.rng.uniform(120, 260),   # 竖直速度总是向上，形成喷泉状
                            math.sin(ang) * sp])
            life = self.rng.uniform(0.7, 1.4)             # 寿命
            self.add_particle(pos + np.array([0, 2, 0]), vel, life,
                              self.rng.randint(2, 4), col)
        # 记录一个点光源，画的时候按 t 逐渐变暗
        self.firework_lights.append({"pos": pos + np.array([0, 20.0, 0]),
                                     "t": 1.0, "color": col})

    def burst(self, pos, n, color, speed=120.0, up=60.0):
        # 通用小爆炸(起跳/收苹果/死亡时用)：n 个粒子，速度可调
        for _ in range(n):
            ang = self.rng.uniform(0, math.tau)
            sp = self.rng.uniform(speed * 0.3, speed)     # 速度在 30%~100% 之间随机
            self.add_particle(pos, np.array([math.cos(ang) * sp,
                                             self.rng.uniform(up * 0.2, up),
                                             math.sin(ang) * sp]),
                              self.rng.uniform(0.3, 0.7), self.rng.randint(2, 3), color)

    def add_particle(self, pos, vel, life, size, color):
        # 添加一个粒子，并保证总数不超过上限(超了就丢掉最老的)
        self.particles.append({"pos": np.array(pos, float), "vel": np.array(vel, float),
                               "life": life, "max_life": life, "size": size,
                               "color": np.array(color, float)})
        if len(self.particles) > MAX_PARTICLES:
            del self.particles[:len(self.particles) - MAX_PARTICLES]

    # --------------------------------------------------------
    def update(self, dt, keys):
        # 每帧的游戏逻辑更新：时间、跳跃物理、移动、粒子、碰撞、摄像机
        if self.dying > 0:
            self.dying -= dt
            if self.dying <= 0:
                self.reset_player()          # 死亡停顿结束 -> 复活到起点

        self.time += dt
        if self.banner_t > 0:
            self.banner_t -= dt              # 横幅倒计时

        # ---- Physical jump: explicit vertical velocity + gravity (Section 11.3)
        # 物理跳跃：每帧用欧拉积分更新位置，v -= g*dt，h += v*dt
        if self.jump_cd > 0:
            self.jump_cd = max(0.0, self.jump_cd - dt)     # 冷却倒计时
        if self.jump_h > 0.0 or self.vspeed != 0.0:
            self.vspeed -= GRAVITY * dt                     # 速度受重力衰减
            self.jump_h += self.vspeed * dt                 # 高度按速度积分
            if self.jump_h <= 0.0:
                self.jump_h = 0.0                           # 落地：高度和速度都归零
                self.vspeed = 0.0
        self.pos[1] = PLAYER_Y + self.jump_h                # 世界 y = 基础高度 + 跳跃高度

        # 把「摄像机偏航角」转成前进/右方两个水平方向向量，实现「相对镜头移动」
        yaw = self.cam_yaw
        fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        right = np.array([-math.cos(yaw), 0.0, math.sin(yaw)])

        move = np.zeros(3)                    # 本帧的移动向量
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move += fwd                         # 前
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move -= fwd                         # 后
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move += right                       # 右
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move -= right                       # 左

        if keys[pygame.K_e]:
            self.cam_yaw += 1.8 * dt            # E/Q 也可以键盘旋转镜头
        if keys[pygame.K_q]:
            self.cam_yaw -= 1.8 * dt

        moving = np.linalg.norm(move) > 1e-6    # 是否真的在移动(排除零向量)
        if moving and self.dying <= 0:
            move = move / np.linalg.norm(move)      # 归一化，保证斜向移动不会更快
            self.move_player(move, dt)
            self.heading = math.atan2(move[0], move[2])   # 朝向 = 移动方向的水平角
            self.walk_time += dt
            self.leg_amp = min(1.0, self.leg_amp + dt * 8.0)   # 行走幅度平滑升到 1
            self.spark_t += dt
            while self.spark_t >= 0.05:
                # 每 0.05 秒在脚下丢一个尘土粒子，形成走路扬尘
                self.spark_t -= 0.05
                p = self.pos + np.array([0, -12.0 + self.jump_h, 0])
                self.add_particle(p + np.array([self.rng.uniform(-4, 4), 0,
                                                self.rng.uniform(-4, 4)]),
                                  np.array([self.rng.uniform(-15, 15), 24,
                                            self.rng.uniform(-15, 15)]),
                                  0.4, 2, (0.85, 0.82, 0.75))
        else:
            self.leg_amp = max(0.0, self.leg_amp - dt * 8.0)   # 停下来时幅度平滑衰减

        # ---- Worm body trail: segments follow earlier head positions (Section 12.2)
        # 身体拖尾：把当前头位置插到轨迹最前面，身体各节去读历史位置
        self.trail.insert(0, self.pos.copy())
        if len(self.trail) > 14:
            del self.trail[14:]              # 只保留最近 14 个位置

        self.update_birds(dt)
        self.update_particles(dt)
        if self.dying <= 0:
            self.check_pickups()             # 吃苹果判定
            self.check_hazards()             # 水/鸟碰撞判定
            self.check_finish()              # 到达终点判定
        self.update_camera(dt)

    def move_player(self, direction, dt):
        # 移动 + 碰撞：把位移拆成小步，逐步尝试，实现「贴墙滑动」而不是卡住
        movement = direction * PLAYER_SPEED * dt
        # 每步最多移动 3 个世界单位，避免高速时穿过薄墙(tunneling)
        steps = max(1, int(math.ceil(max(abs(movement[0]), abs(movement[2])) / 3.0)))
        sx = movement[0] / steps
        sz = movement[2] / steps
        for _ in range(steps):
            # X 和 Z 分开检测：如果 X 方向撞墙但 Z 方向没撞，仍然可以斜着滑过去
            nx = self.pos[0] + sx
            if not self.circle_hits_wall(nx, self.pos[2], PLAYER_R):
                self.pos[0] = nx
            nz = self.pos[2] + sz
            if not self.circle_hits_wall(self.pos[0], nz, PLAYER_R):
                self.pos[2] = nz
        self.pos[1] = PLAYER_Y + self.jump_h      # 保持竖直位置

    def circle_hits_wall(self, x, z, r):
        # 圆 vs AABB 碰撞：把圆心投影到矩形上取最近点，若距离小于半径则相交
        for x0, z0, x1, z1 in self.collision:
            cx = max(x0, min(x, x1))              # 矩形上离圆心最近的 x
            cz = max(z0, min(z, z1))              # 最近的 z
            dx, dz = x - cx, z - cz
            if dx * dx + dz * dz < r * r:         # 用平方距离比较，省一次开方
                return True
        return False

    def update_birds(self, dt):
        # 小鸟沿迷宫走廊在相邻格子之间飞行
        for d in self.birds:
            target = self.maze.center(d["target"])
            target[1] = BIRD_FLY_HEIGHT           # 保持飞行高度
            delta = target - d["pos"]
            dist = np.linalg.norm(delta)
            if dist <= 2.0:
                # 到达目标格：换下一个目标(尽量不走回头路，走廊尽头则允许折返)
                d["pos"] = target
                d["prev"] = d["cell"]
                d["cell"] = d["target"]
                nbrs = self.maze.neighbors(d["cell"])
                if nbrs:
                    alt = [n for n in nbrs if n != d["prev"]]
                    d["target"] = self.rng.choice(alt if alt else nbrs)
            else:
                step = d["speed"] * dt
                if step >= dist:
                    d["pos"] = target.copy()          # 这一步会越过目标就直接落到目标
                else:
                    d["pos"] = d["pos"] + delta / dist * step   # 朝目标走一步
                d["heading"] = math.atan2(delta[0], delta[2])   # 朝向移动方向
            # 上下轻微浮动：正弦波的相位各不相同，所以每只鸟的节奏不一样
            d["pos"][1] = BIRD_FLY_HEIGHT + math.sin(self.time * 2.2 + d["phase"]) * 5.0

    def update_particles(self, dt):
        # 粒子更新：受重力、按速度移动、生命递减；同时用「就地压缩」的写法清理死亡粒子
        alive = 0
        for p in self.particles:
            p["vel"][1] -= 210.0 * dt           # 粒子也受重力(比玩家轻一些，下落慢)
            p["pos"] += p["vel"] * dt
            p["life"] -= dt
            if p["life"] > 0 and p["pos"][1] > -3:
                self.particles[alive] = p       # 还活着的挪到前面(避免新建列表)
                alive += 1
        if alive < len(self.particles):
            del self.particles[alive:]          # 截断掉后面已死亡的

        for light in self.firework_lights:
            light["t"] -= dt * 1.8              # 烟火光强按时间衰减
        self.firework_lights = [l for l in self.firework_lights if l["t"] > 0]   # 移除已经熄灭的

    def check_pickups(self):
        # 吃苹果判定：玩家与苹果(抬高 16)的距离小于 24 就算吃到
        for apple in self.apples:
            if apple["collected"]:
                continue                        # 已吃过，跳过
            d = self.pos - np.array([apple["pos"][0], 16.0, apple["pos"][2]])
            if np.linalg.norm(d) < 24.0:
                apple["collected"] = True
                self.collected += 1
                self.score += apple["value"]    # 金苹果 100 分，银苹果 50 分
                col = (1.0, 0.75, 0.2) if apple["type"] == "gold" else (1.0, 0.35, 0.35)
                self.burst(apple["pos"] + np.array([0, 16, 0]), 26, col, speed=120, up=110)
                if self.collected >= self.medals:
                    # 苹果集齐 -> 提示可以去终点了
                    self.banner = "Exit unlocked! Head to the finish flag!"
                    self.banner_t = 2.0
                self._minimap_dirty = True      # 小地图上的苹果点要更新

    def check_hazards(self):
        """Height-aware hazard rules (Section 11.4).

        Water only kills when the worm's feet are below the safe clearance.
        Birds use a true 3-D sphere test, so a high jump can pass under a bird
        only if the vertical gap is larger than the sum of the radii.
        """
        # 【中文】带高度判断的危险规则：
        # 1) 水潭：只有在「脚」低于安全高度时才淹死 —— 所以跳起来是可以越过水面的；
        # 2) 小鸟：用真正的 3D 球体距离判定，必须竖直间隙小于两半径之和才会被撞到。
        feet = self.pos[1] - PLAYER_R            # 脚底高度 = 球心高度 - 半径
        for pool in self.water_pools:
            d = self.pos - pool["pos"]
            # 两个条件同时满足才死：脚太低 且 水平距离在 (玩家半径+水潭半径) 内
            if feet < WATER_SAFE_CLEARANCE and \
                    d[0] ** 2 + d[2] ** 2 < (PLAYER_R + WATER_R) ** 2:
                self.kill_player("water")
                return                           # 死了就结束本帧检测
        for b in self.birds:
            # 三维距离判定：跳得足够高时竖直间隙大于两半径之和，就能从鸟下面穿过去
            if np.linalg.norm(self.pos - b["pos"]) < PLAYER_R + BIRD_R:
                self.kill_player("bird")
                return

    def kill_player(self, why):
        # 死亡处理：进入 0.9 秒停顿，喷粒子，弹提示
        self.dying = 0.9
        self.burst(self.pos, 30, (0.4, 0.75, 1.0) if why == "water" else (0.4, 0.9, 1.0),
                   speed=130, up=120)
        self.banner = "Splash! The worm fell into the water!" if why == "water" else "Snatched by a bird!"
        self.banner_t = 1.4

    def check_finish(self):
        # 到达终点判定：水平距离小于 20 且苹果集齐 -> 过关
        d = self.pos - self.finish_flag
        if d[0] ** 2 + d[2] ** 2 < 20.0 ** 2:
            if self.collected >= self.medals:
                self.level += 1                  # 进入下一关
                self.burst(self.finish_flag + np.array([0, 20, 0]), 70,
                           (1.0, 0.85, 0.3), speed=180, up=220)   # 过关烟花
                self.build_level()               # 生成新一关
                self.reset_player()
            else:
                # 苹果没吃完，提示还差几个
                self.banner = f"Need {self.medals - self.collected} more apples"
                self.banner_t = 1.0

    def update_camera(self, dt):
        # The camera aim follows the ground position, not the jump height, so a
        # vertical jump is visible as the worm rising inside the frame.
        # 【中文】镜头瞄准的是「地面位置」而不是「跳跃后的位置」，
        # 这样竖直起跳时能看到毛毛虫在画面里往上移动，而不是镜头跟着一起升。
        target = np.array([self.pos[0], 16.0, self.pos[2]])
        # 指数平滑跟随(每帧向目标靠近 dt*7 的比例)，避免镜头抖动
        self.cam_target = self.cam_target + (target - self.cam_target) * min(1.0, dt * 7.0)

        pitch = self.cam_pitch
        yaw = self.cam_yaw
        # 由俯仰角和偏航角算出摄像机相对目标的「反向视线方向」
        fwd = np.array([math.cos(pitch) * math.sin(yaw),
                        math.sin(pitch),
                        math.cos(pitch) * math.cos(yaw)])
        dist = self.cam_dist
        desired = self.cam_target - fwd * dist   # 期望的摄像机位置
        t = 0.15
        while t < 1.0:
            # 沿目标->期望位置的连线逐点采样，一旦发现点落进了墙里，就把距离收短
            # 这样镜头贴近墙壁时会自动往前拉，不会穿墙看到迷宫外面
            p = self.cam_target + (desired - self.cam_target) * t
            if p[1] < WALL_H + 3.0 and self.point_in_wall(p[0], p[2], 5.0):
                dist = max(45.0, dist * t * 0.92)     # 收短距离，但不小于 45
                break
            t += 0.06
        eye = self.cam_target - fwd * dist
        eye[1] = max(eye[1], 10.0)               # 摄像机不低于地面 10 单位，防止钻到地下

        cam = self.camera
        cam.projection = self.projection         # 同步投影方式(透视/正交切换)
        cam.ortho_height = self.cam_dist * 1.35  # 正交模式下视景体高度随缩放距离变化
        cam.look(eye, self.cam_target)           # 设置摄像机位置与注视点

    def point_in_wall(self, x, z, pad):
        # 判断点(x,z)是否在墙里(带 pad 外扩)：只查周围 9 个格子的碰撞盒，很快
        gx = int(math.floor(x / CELL))
        gz = int(math.floor(z / CELL))
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for x0, z0, x1, z1 in self.collision_grid.get((gx + dx, gz + dz), ()):
                    if x0 - pad <= x <= x1 + pad and z0 - pad <= z <= z1 + pad:
                        return True
        return False

    # --------------------------------------------------------
    def draw(self):
        # 一帧的绘制流程：清屏 -> 加光源 -> 地面 -> 墙 -> 水/苹果/旗/鸟/玩家 -> 粒子 -> 放大 -> HUD
        r = self.renderer
        cam = self.camera
        r.begin(cam)
        # 主光源：平行光(太阳)，方向朝下偏一侧
        r.add_directional_light((0.42, -1.0, 0.30), (1.0, 0.96, 0.86), 0.62)
        # 玩家自带一个暖色点光源，像提着一盏灯，让毛毛虫周围亮起来
        r.add_point_light(self.pos + np.array([0.0, 14.0, 0.0]),
                          (1.0, 0.80, 0.48), 0.62, 210.0)
        # 每个烟花再贡献一个点光源(强度随 t 衰减)
        for light in self.firework_lights:
            r.add_point_light(light["pos"], tuple(light["color"]),
                              0.8 * light["t"], 260.0)

        wf = r.wireframe
        if wf:
            # 线框模式下：先画暗色地面当背景，再切到线框画地面网格，形成"网格纸"效果
            r.wireframe = False
            for model, w, d in self._ground_models:
                r.draw_mesh(self.ground, model,
                            color=(0.045, 0.06, 0.07), lighting=False, cull=False)
            r.draw_mesh(self.floor_mesh, color=(0.09, 0.12, 0.13),
                        lighting=False, cull=False)
            r.wireframe = True
            r.draw_mesh(self.floor_mesh, color=(0.9, 0.95, 1.0), cull=False)
        else:
            # 正常模式：地面改到「全部不透明几何之后」再画，见下方 —— 地面 段。
            # 【性能优化】z-buffer 保证最终画面与原来完全一致，但这样被墙/道具
            # 挡住的地面像素就直接过不了深度测试，不用先着色再被覆盖。
            pass

        if self._cells:
            # 墙体：先按距离剔除(VIEW_DISTANCE)，再按距离近->远排序
            # 排序是为了让近处的墙先画，配合 z-buffer 提前挡住远处的填充，省像素
            eye = cam.eye
            dx = self._cell_centers[:, 0] - eye[0]
            dz = self._cell_centers[:, 2] - eye[2]
            d2 = dx * dx + dz * dz
            mask = d2 < (VIEW_DISTANCE * VIEW_DISTANCE)   # 超出视距的格子直接不画
            idx = np.nonzero(mask)[0]
            if idx.size:
                order = idx[np.argsort(d2[idx])].tolist()   # 由近到远
                for i in order:
                    mesh, model, _ = self._cells[i]
                    r.draw_mesh(mesh, model,
                                texture=self.tex_stone,
                                uv_scale=(1.0 / 44.0, 1.0 / 44.0),   # 贴图按世界单位缩放，墙面纹理连续
                                color=(0.60, 0.59, 0.56) if not r.textures_enabled else (1.0, 1.0, 1.0),
                                fog_vertex=True)

        for pool in self.water_pools:
            self.draw_water(r, pool)              # 水潭
        for apple in self.apples:
            if not apple["collected"]:
                self.draw_apple(r, apple)         # 只画还没吃掉的苹果
        self.draw_flag(r, self.start_flag, self.flag_start_mesh, (0.33, 0.78, 0.28),
                       "start", unlocked=True)
        self.draw_flag(r, self.finish_flag, self.flag_finish_mesh, (1.0, 1.0, 1.0),
                       "finish", unlocked=self.collected >= self.medals)   # 集齐苹果后终点旗才发光
        for d in self.birds:
            self.draw_bird(r, d)
        self.draw_worm(r)

        # ---- 地面 ----
        # 【性能优化】放在全部不透明几何之后：地面是全场最远的表面，靠 z-buffer
        # 保证画面不变，但被墙/道具挡住的地面像素直接过不了深度测试，省掉重复填充
        if not wf:
            # 远处 4 块草地(不带雾，铺满视野边缘)
            for model, w, d in self._ground_models:
                r.draw_mesh(self.ground, model,
                            texture=self.tex_grass, uv_scale=(w / 60.0, d / 60.0),
                            fog_vertex=False, cull=False)
            # 迷宫地面：地砖贴图 + 逐顶点烘焙雾
            r.draw_mesh(self.floor_mesh, texture=self.tex_floor,
                        color=(0.50, 0.50, 0.46) if not r.textures_enabled else (1.0, 1.0, 1.0),
                        fog_vertex=True, cull=False)

        # 粒子：按剩余生命比例决定亮度和大小，加法混合叠加到画面上
        for p in self.particles:
            a = max(0.0, p["life"] / p["max_life"])    # 生命比例 1->0
            r.draw_glow(p["pos"], 1.6 + p["size"] * 0.55, p["color"], 1.1 * a)
        for light in self.firework_lights:
            r.draw_glow(light["pos"], 10.0, light["color"], 0.7 * light["t"])

        surf = r.end()                     # 结束渲染，拿到小尺寸的最终图像
        target = self.screen.get_size()    # 窗口尺寸
        if surf.get_size() == target:
            self.screen.blit(surf, (0, 0))     # 尺寸正好一致(比如 NATIVE 档)直接贴
        else:
            if self.sharpen:
                surf = sharpen_surface(surf, 0.55)     # 先锐化再放大，补回一些细节
            if SHARP_UPSCALE:
                pygame.transform.scale(surf, target, self.scaled)          # 硬边放大(快)
            else:
                pygame.transform.smoothscale(surf, target, self.scaled)    # 平滑放大(好看)
            self.screen.blit(self.scaled, (0, 0))
        self.draw_hud()                    # 最后画 HUD(保证文字在画面最上层)

    def draw_water(self, r, pool):
        # 水面动画：顶点 y 值按「到圆心的距离」做正弦波，形成同心圆波纹向外扩散
        mesh = pool["mesh"]
        v = mesh.vertices
        rad = np.sqrt(v[:, 0] * v[:, 0] + v[:, 2] * v[:, 2])   # 每个顶点到中心的水平距离
        v[:, 1] = np.sin(rad * 9.0 - self.time * 3.2 + pool["phase"]) * 2.2 * rad   # 外圈波动更大
        r.draw_mesh(mesh, pool["model"], color=(0.55, 0.80, 1.00),
                    texture=self.tex_water, emissive=0.22, double_sided=True)   # 双面可见(从下方也能看到)

    def draw_apple(self, r, apple):
        # 苹果 = 球(果身) + 细圆柱(果梗) + 四边形(叶子)，上下浮动并绕 Y 轴自转
        bob = math.sin(self.time * 2.4 + apple["phase"]) * 2.5    # 上下浮动
        p = apple["pos"] + np.array([0.0, 16.0 + bob, 0.0])
        spin = self.time * 1.6 + apple["phase"]                   # 自转角
        color = C_GOLD if apple["type"] == "gold" else C_SILVER
        # 模型矩阵顺序：先缩放 -> 再旋转 -> 最后平移(矩阵右乘，所以写起来是反的)
        model = e3.translation(*p) @ e3.rotation_y(spin) @ e3.scaling(9.0, 8.6, 9.0)
        r.draw_mesh(self.sphere_lo, model, color=color, emissive=0.06)   # 果身略自发光，显得鲜亮
        r.draw_mesh(self.cyl_sm, e3.translation(*p) @ e3.translation(0, 10, 0)
                    @ e3.scaling(0.9, 6.0, 0.9),
                    color=(0.35, 0.22, 0.10), smooth=True)               # 果梗(棕色细柱)
        model = (e3.translation(*p) @ e3.translation(3, 13, 0)
                 @ e3.rotation_y(spin + 0.6) @ e3.rotation_z(0.5))
        r.draw_mesh(self.leaf_mesh, model, color=(0.30, 0.65, 0.25),
                    double_sided=True, smooth=False)                    # 叶子

    def draw_flag(self, r, base, mesh, color, kind, unlocked):
        # 旗子 = 木杆(圆柱) + 顶部金球 + 旗面(可顶点动画的软网格)
        pole_h = 76.0 if kind == "finish" else 58.0     # 终点旗杆更高
        r.draw_mesh(self.cyl_sm,
                    e3.translation(base[0], pole_h * 0.5, base[2])
                    @ e3.scaling(1.2, pole_h, 1.2),
                    color=(0.55, 0.42, 0.26), texture=self.tex_wood, smooth=True)   # 杆
        r.draw_mesh(self.sphere_sm,
                    e3.translation(base[0], pole_h, base[2]) @ e3.scaling(2.6),
                    color=C_GOLD, emissive=0.35, smooth=True)                        # 杆顶金球
        # 旗面波浪：振幅随离杆距离(xs/30)线性变大，像被风吹起来一样
        v = mesh.vertices
        xs = v[:, 0]
        v[:, 2] = np.sin(xs * 0.30 + self.time * 5.0) * 2.4 * (xs / 30.0)
        tex = self.tex_checker if kind == "finish" else self.tex_cloth   # 终点用黑白格，起点用布料
        model = e3.translation(base[0] + 1.0, pole_h - 3.0, base[2]) @ e3.rotation_y(-0.25)
        r.draw_mesh(mesh, model, color=color if kind == "start" else (1.0, 1.0, 1.0),
                    texture=tex, double_sided=True, smooth=False,
                    emissive=0.25 if (kind == "finish" and unlocked) else 0.0)   # 解锁后终点旗自发光
        if kind == "finish" and unlocked:
            r.draw_glow(base + np.array([0.0, pole_h, 0.0]), 8.0, (1.0, 0.9, 0.4), 0.8)   # 加一圈光晕

    def draw_bird(self, r, d):
        # ⚠️ 注意：本函数在第 1084 行还有一份同名定义，Python 会用「后定义的那个」覆盖本定义。
        # 也就是说实际生效的是文件后面那个 draw_bird（更简洁的版本），这一份是早期版本、属于遗留代码。
        # 保留它不影响运行，但如果老师问「为什么有两份」，答案就是：后定义覆盖前定义，实际用的是后者。
        # 鸟 = 多个球拼接 + 锥体做嘴/角/尾巴 + 立方体做翅膀
        p = d["pos"]
        heading = d["heading"]
        flap = math.sin(self.time * 6.0 + d["phase"]) * 0.9   # 翅膀扇动角度
        base = e3.translation(*p) @ e3.rotation_y(-heading)   # 鸟的局部坐标系

        # Body  身体
        r.draw_mesh(self.sphere_lo, base @ e3.scaling(10.5, 6.5, 7.5),
                    color=BIRD_BODY, smooth=True)
        # Belly  腹部(浅色)
        r.draw_mesh(self.sphere_lo, base @ e3.translation(1.0, -1.5, 0)
                    @ e3.scaling(9.0, 3.0, 5.5),
                    color=BIRD_BELLY, smooth=True)
        # Head  头
        r.draw_mesh(self.sphere_sm, base @ e3.translation(9.0, 2.5, 0) @ e3.scaling(5.5),
                    color=BIRD_BODY, smooth=True)
        # Snout  嘴(圆锥旋转 90 度让尖端朝 +X)
        r.draw_mesh(self.cone_sm,
                    base @ e3.translation(13.5, 2.0, 0) @ e3.rotation_z(-math.pi / 2)
                    @ e3.scaling(2.0, 5.5, 2.0),
                    color=BIRD_BODY_DARK, smooth=False)
        # Horns  两只角
        for side in (-1, 1):
            r.draw_mesh(self.cone_sm,
                        base @ e3.translation(8.5, 5.0, side * 1.8)
                        @ e3.rotation_z(-0.4) @ e3.rotation_x(side * 0.3)
                        @ e3.scaling(1.1, 4.0, 1.1),
                        color=BIRD_HORN, smooth=False)
        # Eyes (glowing)  发光的眼睛
        for side in (-1, 1):
            r.draw_mesh(self.sphere_sm,
                        base @ e3.translation(9.8, 3.5, side * 1.8) @ e3.scaling(1.2),
                        color=BIRD_EYE, emissive=0.9, smooth=True)
        # Tail  尾巴
        r.draw_mesh(self.cone_sm,
                    base @ e3.translation(-11.5, 1.0, 0) @ e3.rotation_z(math.pi / 2)
                    @ e3.scaling(2.6, 9.0, 1.4),
                    color=BIRD_BODY_DARK, smooth=False)
        # Tail tip  尾尖(橙色)
        r.draw_mesh(self.cone_sm,
                    base @ e3.translation(-18.0, 1.0, 0) @ e3.rotation_z(math.pi / 2)
                    @ e3.scaling(1.0, 3.5, 0.8),
                    color=BIRD_HORN, smooth=False)
        # Wings (flap around X axis)  翅膀：绕 X 轴扇动
        for side in (-1, 1):
            # 变换顺序：先平移到肩部 -> 绕 X 轴旋转(扇动) -> 再向外移 -> 缩放成扁平的翅膀
            wing = (base @ e3.translation(-1.0, 2.5, side * 4.0)
                    @ e3.rotation_x(side * flap)
                    @ e3.translation(0.0, 0.0, side * 7.0)
                    @ e3.scaling(12.0, 1.0, 13.0))
            r.draw_mesh(self.box_unit, wing, color=BIRD_WING, smooth=False)

    def draw_worm(self, r):
        # 毛毛虫 = 5 节身体球 + 头球 + 两只眼睛 + 腹部浅色带
        # Contact shadow: shrinks as the worm rises, which makes the world-space
        # jump height readable on screen.
        # 【中文】接触阴影：跳得越高阴影越小，这样在屏幕上就能「看出来」跳了多高。
        if self.jump_h > 1.0:
            sc = max(0.45, 1.0 - self.jump_h / 90.0)      # 阴影缩放比例(最低 0.45)
            r.draw_mesh(self.disc_shadow,
                        e3.translation(self.pos[0], 0.35, self.pos[2])
                        @ e3.scaling(13.5 * sc, 1.0, 9.5 * sc),
                        color=(0.08, 0.18, 0.08), lighting=False,   # 阴影不参与光照
                        emissive=0.05, double_sided=True)

        fwd = np.array([math.sin(self.heading), 0.0, math.cos(self.heading)])
        trail = self.trail if self.trail else [self.pos.copy()]
        # Body segments follow a short history of the head position (Section 12.2).
        # 身体各节跟随头部轨迹：在轨迹里按步长取 5 个点作为每节的位置
        if np.linalg.norm(trail[0] - trail[-1]) > 7.0:
            marks = [trail[min(i * 3, len(trail) - 1)] for i in range(5)]
        else:
            # Standing still: lay the segments out behind the head so the worm
            # still reads as a segmented body instead of one blob.
            # 站着不动时：把各节沿身后一字排开，否则 5 节会重叠成一个球看不出分节
            marks = [self.pos - fwd * (7.5 * i) for i in range(5)]

        radii = (10.6, 9.2, 8.2, 7.2, 5.8)          # 每节半径，越往后越细
        shades = (WORM_GREEN, WORM_GREEN, (0.36, 0.80, 0.30),
                  (0.27, 0.66, 0.22), WORM_GREEN_DARK)   # 每节颜色，尾部更暗
        # Draw tail-to-head so the head sphere wins any depth ties.
        # 从尾画到头：深度值相同时后画的覆盖先画的，保证头在最上层
        for k in range(4, -1, -1):
            q = marks[k]
            rad = radii[k]
            r.draw_mesh(self.sphere_lo,
                        e3.translation(q[0], q[1], q[2])
                        @ e3.scaling(rad, rad * 0.80, rad),     # y 方向压扁 0.8，看起来更贴地
                        color=shades[k], smooth=True)

        # Head: bright green sphere with two eyes on the forward/right basis.
        # 头：亮绿色球 + 两只眼睛(位置用「前方向×右方向」两个基向量定位)
        head = np.array([self.pos[0], self.pos[1], self.pos[2]])
        r.draw_mesh(self.sphere_lo, e3.translation(*head)
                    @ e3.scaling(10.8, 8.8, 10.8), color=WORM_GREEN_LIGHT,
                    smooth=True)
        right = np.array([math.cos(self.heading), 0.0, -math.sin(self.heading)])
        for side in (-1, 1):
            # 眼睛位置 = 头中心 + 前方 6 + 左右 3.7 + 上方 4.6
            eye = head + fwd * 6.0 + right * (side * 3.7) + np.array([0.0, 4.6, 0.0])
            r.draw_mesh(self.sphere_sm, e3.translation(*eye) @ e3.scaling(2.3),
                        color=WORM_EYE, emissive=0.30, smooth=True)     # 眼白
            r.draw_mesh(self.sphere_sm,
                        e3.translation(*(eye + fwd * 1.4)) @ e3.scaling(1.1),
                        color=WORM_PUPIL, smooth=True)                  # 瞳孔(稍微往前一点)
        # Light belly stripe under the head.  头下方的浅色腹部条
        r.draw_mesh(self.sphere_lo,
                    e3.translation(head[0], head[1] - 5.0, head[2])
                    @ e3.scaling(8.0, 2.4, 5.2),
                    color=WORM_BELLY, smooth=True)

    def draw_bird(self, r, d):
        # ✅ 这才是真正生效的 draw_bird（它定义在文件更后面，覆盖了前面那份）
        # 造型更简洁：一个深色椭球身体 + 浅色肚皮 + 头 + 嘴 + 两只眼 + 尾巴 + 两片扇动的翅膀
        p = d["pos"]
        heading = d["heading"]
        flap = math.sin(self.time * 7.0 + d["phase"]) * 0.85    # 扇动频率比上一版稍快
        base = e3.translation(*p) @ e3.rotation_y(-heading)     # 局部坐标系：平移到鸟位置并朝向航向

        # Body (dark ellipsoid) and lighter belly.  身体(深色椭球) + 浅色肚皮
        r.draw_mesh(self.sphere_lo, base @ e3.scaling(11.0, 6.0, 7.0),
                    color=BIRD_BODY, smooth=True)
        r.draw_mesh(self.sphere_lo, base @ e3.translation(1.0, -1.8, 0)
                    @ e3.scaling(9.0, 2.8, 5.2), color=BIRD_BELLY, smooth=True)
        # Head.  头
        r.draw_mesh(self.sphere_sm, base @ e3.translation(9.5, 2.8, 0)
                    @ e3.scaling(4.8), color=BIRD_BODY, smooth=True)
        # Beak (cone rotated to point along +X).  嘴：圆锥转 90 度让尖端朝前
        r.draw_mesh(self.cone_sm, base @ e3.translation(14.0, 2.4, 0)
                    @ e3.rotation_z(-math.pi / 2) @ e3.scaling(1.4, 4.4, 1.4),
                    color=BIRD_HORN, smooth=False)
        # Eyes.  眼睛：眼白 + 深色瞳孔
        for side in (-1, 1):
            eye = base @ e3.translation(10.2, 3.9, side * 1.9) @ e3.scaling(1.5)
            r.draw_mesh(self.sphere_sm, eye, color=WORM_EYE, emissive=0.2,
                        smooth=True)
            r.draw_mesh(self.sphere_sm,
                        base @ e3.translation(11.0, 3.9, side * 1.9)
                        @ e3.scaling(0.8), color=BIRD_BODY_DARK, smooth=True)
        # Tail.  尾巴
        r.draw_mesh(self.cone_sm, base @ e3.translation(-12.0, 1.0, 0)
                    @ e3.rotation_z(math.pi / 2) @ e3.scaling(2.2, 8.0, 1.0),
                    color=BIRD_BODY_DARK, smooth=False)
        # Two flat wings that flap around the body axis (X).
        # 两片扁平翅膀：绕身体轴(X)旋转实现扇动
        for side in (-1, 1):
            wing = (base @ e3.translation(-1.0, 2.0, side * 4.5)
                    @ e3.rotation_x(side * flap)
                    @ e3.translation(0.0, 0.0, side * 7.0)
                    @ e3.scaling(11.0, 0.8, 12.0))
            r.draw_mesh(self.box_unit, wing, color=BIRD_WING, smooth=False)

    # --------------------------------------------------------
    def draw_hud(self):
        # HUD(抬头显示)：顶部信息条 + 底部操作提示 + 小地图 + 中央横幅 + 可选技术面板
        s = self.screen

        if self._topbar is None:
            # 顶部半透明黑绿条，只创建一次并缓存(SRCALPHA 表示带透明度)
            bar = pygame.Surface((WIDTH, 56), pygame.SRCALPHA).convert_alpha()
            bar.fill((12, 26, 20, 190))
            pygame.draw.line(bar, (90, 160, 110), (0, 56), (WIDTH, 56), 2)   # 底部一条亮线
            self._topbar = bar
        s.blit(self._topbar, (0, 0))

        # 关卡 / 分数 / 苹果进度 / 计时
        s.blit(self.font.render(f"LEVEL: {self.level}", True, (235, 245, 235)), (18, 15))
        s.blit(self.font.render(f"SCORE: {self.score}", True, (255, 215, 90)), (170, 15))
        s.blit(self.font.render(f"APPLES: {self.collected}/{self.medals}",
                                True, (255, 150, 150)), (340, 15))
        # 计时格式化成 mm:ss（//60 取分钟，%60 取秒）
        s.blit(self.font.render(
            f"TIME: {int(self.time)//60:02d}:{int(self.time)%60:02d}",
            True, (235, 245, 235)), (560, 15))

        fps = int(self.clock.get_fps())      # 实际帧率
        r = self.renderer
        # 性能信息：帧率、三角形数、着色像素数(k)、内部渲染分辨率
        s.blit(self.small.render(
            f"FPS {fps}   TRIS {r.stats['triangles']}"
            f"   PX {r.stats['pixels']//1000}k   RES {r.w}x{r.h}",
            True, (150, 220, 170)), (760, 8))
        # 当前状态：投影模式、纹理/雾/过滤/画质档位
        mode = "PERSP" if self.projection == "perspective" else "ORTHO"
        s.blit(self.small.render(
            f"{mode}  tex:{'on' if r.textures_enabled else 'off'}"
            f"  fog:{'on' if r.fog_enabled else 'off'}"
            f"  filt:{r.filter_mode}"
            f"  q:{QUALITY_LEVELS[self.quality][0]}",
            True, (150, 220, 170)), (760, 30))

        if self._hint_txt is None:
            # 底部操作提示：文字渲染一次就缓存，避免每帧重复渲染文字(很费 CPU)
            hint = ("WASD Move   SPACE Jump   Right-drag Rotate   Wheel Zoom   "
                    "Click Firework   V Projection   T Textures   F Fog   B Filter   "
                    "F1 Wireframe   F3 Depth   F5 Quality   C Sharpen   F11 Fullscreen   "
                    "H Help   R Restart")
            self._hint_txt = self.small.render(hint, True, (230, 240, 230))
            self._hint_bar = pygame.Surface(
                (self._hint_txt.get_width() + 24, self._hint_txt.get_height() + 10),
                pygame.SRCALPHA).convert_alpha()
            self._hint_bar.fill((10, 24, 18, 170))
        hx = WIDTH // 2 - self._hint_bar.get_width() // 2      # 水平居中
        hy = HEIGHT - 38
        s.blit(self._hint_bar, (hx, hy))
        s.blit(self._hint_txt,
               (WIDTH // 2 - self._hint_txt.get_width() // 2, HEIGHT - 33))

        self.draw_minimap()

        if self.banner_t > 0 and self.banner:
            # 横幅：文字+背景只在「文字内容变化」时重新渲染并缓存
            if self._banner_cache_key != self.banner:
                txt = self.title_font.render(self.banner, True, (255, 240, 180))
                bg = pygame.Surface((txt.get_width() + 60, txt.get_height() + 26),
                                    pygame.SRCALPHA).convert_alpha()
                bg.fill((10, 20, 16, 255))
                self._banner_cache = (txt, bg)
                self._banner_cache_key = self.banner
            txt, bg = self._banner_cache
            a = min(1.0, self.banner_t / 0.5)     # 最后 0.5 秒开始淡出
            bg.set_alpha(int(170 * a))
            x = WIDTH // 2 - bg.get_width() // 2
            s.blit(bg, (x, 140))
            s.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 152))

        if self.show_help:
            self.draw_help_panel()

    def draw_minimap(self):
        # 小地图：静态部分(墙/水潭/终点)与动态部分(苹果/鸟/玩家)分开画
        # 分开的原因是静态部分只在关卡重建时变一次，动态部分每帧都在动
        mm_w, mm_h = 230, 150
        x0 = WIDTH - mm_w - 22        # 右上角
        y0 = 72

        if self._minimap_static is None or self._minimap_dirty:
            surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA).convert_alpha()
            surf.fill((8, 20, 14, 185))
            # 世界坐标 -> 小地图像素 的缩放系数
            sx = mm_w / MAZE_W
            sz = mm_h / MAZE_D
            for cx, cz, w, d in self.maze.wall_boxes():
                # 每段墙画成一个小矩形；max(...,1.4) 保证再细的墙至少占 1.4 像素可见
                pygame.draw.rect(surf, (150, 170, 150),
                                 (cx * sx - max(w * sx, 1.4) * 0.5,
                                  cz * sz - max(d * sz, 1.4) * 0.5,
                                  max(w * sx, 1.4), max(d * sz, 1.4)))
            for pool in self.water_pools:
                # 水潭画成橙色小点
                pygame.draw.circle(surf, (255, 120, 40),
                                   (int(pool["pos"][0] * sx),
                                    int(pool["pos"][2] * sz)), 3)
            # 终点画成空心绿圈(线宽 1)
            pygame.draw.circle(surf, (80, 240, 100),
                               (int(self.finish_flag[0] * sx),
                                int(self.finish_flag[2] * sz)), 4, 1)
            self._minimap_static = surf
            self._minimap_dirty = False

        self.screen.blit(self._minimap_static, (x0, y0))

        # 动态部分：每帧新建一张透明图层再叠上去
        surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA).convert_alpha()
        sx = mm_w / MAZE_W
        sz = mm_h / MAZE_D
        for apple in self.apples:
            if not apple["collected"]:
                col = (255, 200, 60) if apple["type"] == "gold" else (240, 80, 80)
                pygame.draw.circle(surf, col,
                                   (int(apple["pos"][0] * sx),
                                    int(apple["pos"][2] * sz)), 3)
        for d in self.birds:
            pygame.draw.circle(surf, (255, 90, 90),
                               (int(d["pos"][0] * sx), int(d["pos"][2] * sz)), 2)
        px, py = int(self.pos[0] * sx), int(self.pos[2] * sz)
        pygame.draw.circle(surf, (120, 200, 255), (px, py), 4)      # 玩家位置
        # 从玩家点画一条长度 9 的短线表示朝向
        pygame.draw.line(surf, (120, 200, 255), (px, py),
                         (px + int(math.sin(self.heading) * 9),
                          py + int(math.cos(self.heading) * 9)), 2)
        self.screen.blit(surf, (x0, y0))

    def draw_help_panel(self):
        # 技术说明面板(H 键)：把 11 个图形学阶段逐条列出来，方便口试时对照讲解
        lines = [
            "3D Graphics Implementation (engine3d.py)",
            "1. 3D coordinates / vertex-edge-face mesh : Mesh, box/sphere/cone/disc",
            "2. 3D transforms (translate/rotate/scale)   : 4x4 matrix multiplication",
            "3. Camera + perspective & orthographic     : look_at + perspective/ortho",
            "4. Clipping                                : near-plane Sutherland-Hodgman",
            "5. Back-face culling                       : signed area in screen space",
            "6. Triangle rasterization                  : bounding box + barycentric coords",
            "7. Depth buffer z-buffer                   : 1/w interpolation (view depth for ortho)",
            "8. Lighting                                : directional + point light + ambient",
            "9. Texture mapping / fog                   : perspective-correct UV + exp fog",
            "10. 3D particles / fireworks               : additive billboards + depth test",
            "11. Animation / collision                  : vertex animation, AABB collision",
            "",
            "V Projection   T Textures   F Fog   G Wireframe   Left-click Firework",
        ]
        w, h = 760, 26 + len(lines) * 22      # 面板尺寸由行数决定
        x = (WIDTH - w) // 2                  # 水平居中
        y = 200
        panel = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
        panel.fill((8, 18, 14, 220))
        pygame.draw.rect(panel, (110, 210, 140), panel.get_rect(), 2)   # 绿色边框
        self.screen.blit(panel, (x, y))
        for i, line in enumerate(lines):
            col = (255, 235, 150) if i == 0 else (215, 235, 220)   # 标题用黄色，其余用浅色
            f = self.font if i == 0 else self.small               # 标题用大字，其余用小字
            self.screen.blit(f.render(line, True, col), (x + 18, y + 14 + i * 22))

    # --------------------------------------------------------
    def run(self):
        # 主循环：限帧 -> 处理事件 -> 更新逻辑 -> 绘制 -> 交换缓冲 -> 自动调画质
        pygame.event.set_grab(False)
        while self.running:
            # tick(FPS) 会按目标帧率休眠，返回距上一帧的毫秒数；
            # min(...,0.05) 把 dt 封顶 50ms，防止卡顿一下后物理积分「跳太远」直接穿墙
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                self.handle_event(event)
            keys = pygame.key.get_pressed()      # 取当前按键状态(持续按键用这个)
            self.update(dt, keys)
            self.draw()
            pygame.display.flip()                # 双缓冲交换，把画面真正显示出来
            self.auto_quality(dt)
        pygame.quit()

    def step_headless(self, keys=None, dt=1 / 60):
        # 供截图/录像工具调用的「单帧推进」：不进入 run() 循环，外部控制节奏
        if keys is None:
            keys = pygame.key.get_pressed()
        self.update(dt, keys)
        self.draw()

if __name__ == "__main__":
    # 直接用 python game3d.py 启动游戏时的入口
    random.seed()          # 用系统时间做随机种子，每次玩迷宫都不一样
    Game().run()
