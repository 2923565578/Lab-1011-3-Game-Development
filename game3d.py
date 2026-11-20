"""
game3d.py -- Worm's Garden Adventure 3D
=================================================================
A fully interactive 3D version of the lab9 2D game (Start2Dto3D.py),
re-themed into a worm-versus-birds adventure.

Controls (all prompts are in English):
    W A S D / Arrow Keys   Move (relative to camera direction)
    Space                  Physical jump (vertical velocity + gravity)
    Right Mouse Drag       Rotate camera
    Mouse Wheel            Zoom camera
    Left Click             Place a 3D firework at that location
    V / P                  Toggle perspective / orthographic projection
    T                      Toggle textures on/off
    F                      Extension 1: depth fog on/off
    B                      Extension 2: bilinear filtering on/off
    F1 / G                 Wireframe overlay
    F3                     Depth-buffer (z-buffer) debug view
    F5                     Cycle quality preset (FAST / BALANCED / HIGH / NATIVE)
    H                      Show technical help panel
    R                      Restart
    ESC / Q                Quit
"""

import os
os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")

import gc
import math
import random

import numpy as np
import pygame

import engine3d as e3
import textures

try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ============================================================
# Basic Settings
# ============================================================
WIDTH, HEIGHT = 1250, 900
TITLE = "Worm's Garden Adventure 3D"

# Quality presets: (name, internal render scale). The final frame is always
# drawn into this NumPy framebuffer and then resized to the 1250x900 window.
QUALITY_LEVELS = (
    ("FAST", 0.20),
    ("BALANCED", 0.30),
    ("HIGH", 0.50),
    ("NATIVE", 1.00),
)
RENDER_SCALE = QUALITY_LEVELS[1][1]
SHARP_UPSCALE = False
RENDER_W = int(WIDTH * RENDER_SCALE)
RENDER_H = int(HEIGHT * RENDER_SCALE)

FPS = 60
MAX_PARTICLES = 320

# Quality governor
Q_WINDOW = 50
Q_COOLDOWN = 2.5
Q_GRACE = 0.8
Q_CONFIRM = 3
Q_SLOW = 1 / 22
Q_FAST = 1 / 58

# World size
CELL = 50
WALL_T = 8
WALL_H = 40
COLS, ROWS = 23, 15
MAZE_W = COLS * CELL
MAZE_D = ROWS * CELL

PLAYER_R = 11
PLAYER_Y = 12.0            # sphere-centre height above the ground plane
PLAYER_SPEED = 190.0

# --- Physical jump (world units = original pixels, CELL = 50 px = 1 unit) ---
GRAVITY = 520.0            # world units / s^2
JUMP_SPEED = 168.0         # initial vertical velocity, world units / s
JUMP_COOLDOWN = 0.35
# Analytical values used in the report: t_flight = 2*v0/g, h_max = v0^2/2g
JUMP_FLIGHT = 2.0 * JUMP_SPEED / GRAVITY
JUMP_PEAK = JUMP_SPEED * JUMP_SPEED / (2.0 * GRAVITY)
WATER_SAFE_CLEARANCE = 14.0  # feet above this height clear the water surface

BIRD_R = 11
BIRD_SPEED = 70.0
BIRD_FLY_HEIGHT = 26.0

WATER_R = 15.0
WATER_POOLS = 8
TOTAL_APPLES = 4
SILVER_VALUE = 50
GOLD_VALUE = 100

VIEW_DISTANCE = 330.0

C_SKY_TOP = (0.24, 0.42, 0.63)
C_SKY_BOT = (0.80, 0.84, 0.80)
C_FOG = (0.74, 0.79, 0.79)
C_GOLD = (1.00, 0.78, 0.16)
C_SILVER = (0.92, 0.30, 0.30)

# Worm palette (green, segmented body)
WORM_GREEN = (0.30, 0.72, 0.26)
WORM_GREEN_DARK = (0.16, 0.42, 0.14)
WORM_GREEN_LIGHT = (0.48, 0.86, 0.34)
WORM_BELLY = (0.72, 0.86, 0.42)
WORM_EYE = (0.96, 0.97, 0.92)
WORM_PUPIL = (0.06, 0.07, 0.06)

# Bird palette (dark body, orange beak)
BIRD_BODY = (0.12, 0.13, 0.18)
BIRD_BODY_DARK = (0.06, 0.07, 0.10)
BIRD_BELLY = (0.22, 0.24, 0.30)
BIRD_WING = (0.34, 0.37, 0.48)
BIRD_HORN = (1.00, 0.68, 0.16)   # beak
BIRD_EYE = (1.00, 0.92, 0.40)

DIRS = {"N": (0, -1, "S"), "E": (1, 0, "W"), "S": (0, 1, "N"), "W": (-1, 0, "E")}

def make_font(size, bold=False):
    try:
        return pygame.font.SysFont("arial", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)

# ============================================================
# Maze
# ============================================================
class Maze:
    def __init__(self, cols, rows, rng):
        self.cols = cols
        self.rows = rows
        self.rng = rng
        self.cells = [[{"N": True, "E": True, "S": True, "W": True}
                       for _ in range(cols)] for _ in range(rows)]
        self.start = (0, rows // 2)
        self.finish = (cols - 1, rows // 2)
        self.generate()

    def generate(self):
        rng = self.rng
        stack = [self.start]
        visited = {self.start}
        while stack:
            x, y = stack[-1]
            nbrs = []
            for d, (dx, dy, opp) in DIRS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in visited:
                    nbrs.append((nx, ny, d, opp))
            if nbrs:
                nx, ny, d, opp = rng.choice(nbrs)
                self.cells[y][x][d] = False
                self.cells[ny][nx][opp] = False
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        self.cells[self.start[1]][self.start[0]]["W"] = False
        self.cells[self.finish[1]][self.finish[0]]["E"] = False

    def neighbors(self, cell):
        x, y = cell
        out = []
        for d, (dx, dy, opp) in DIRS.items():
            if not self.cells[y][x][d]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    out.append((nx, ny))
        return out

    def center(self, cell):
        x, y = cell
        return np.array([x * CELL + CELL * 0.5, 0.0, y * CELL + CELL * 0.5])

    def wall_boxes(self):
        out = []
        for y in range(self.rows):
            for x in range(self.cols):
                w = self.cells[y][x]
                px, pz = x * CELL, y * CELL
                if w["N"]:
                    out.append((px + CELL * 0.5, pz, CELL + WALL_T, WALL_T))
                if w["W"]:
                    out.append((px, pz + CELL * 0.5, WALL_T, CELL + WALL_T))
                if x == self.cols - 1 and w["E"]:
                    out.append((px + CELL, pz + CELL * 0.5, WALL_T, CELL + WALL_T))
                if y == self.rows - 1 and w["S"]:
                    out.append((px + CELL * 0.5, pz + CELL, CELL + WALL_T, WALL_T))
        return out

    def collision_rects(self):
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
    arr = pygame.surfarray.array3d(surf).astype(np.float32)
    blur = (np.roll(arr, 1, 0) + np.roll(arr, -1, 0)
            + np.roll(arr, 1, 1) + np.roll(arr, -1, 1)) * 0.25
    out = np.clip(arr + amount * (arr - blur), 0.0, 255.0).astype(np.uint8)
    pygame.surfarray.blit_array(surf, out)
    return surf

def shift_mesh(mesh, dx, dy, dz):
    return e3.Mesh(mesh.vertices + np.array([dx, dy, dz]), mesh.index, mesh.uvs,
                   smooth=mesh.smooth)

def flag_grid(cols=5, rows=3, width=30.0, height=20.0):
    verts, uvs, faces = [], [], []
    for j in range(rows + 1):
        for i in range(cols + 1):
            verts.append((i / cols * width, -j / rows * height, 0.0))
            uvs.append((i / cols, j / rows))
    stride = cols + 1
    for j in range(rows):
        for i in range(cols):
            a = j * stride + i
            b = a + 1
            c = a + stride + 1
            d = a + stride
            faces.append((a, b, c))
            faces.append((a, c, d))
    return e3.Mesh(verts, faces, uvs, smooth=False)

# ============================================================
# Game
# ============================================================
class Game:
    def __init__(self, headless=False):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.headless = headless

        self.font = make_font(22)
        self.small = make_font(16)
        self.title_font = make_font(34, bold=True)

        self.renderer = e3.Renderer(
            RENDER_W, RENDER_H,
            sky_top=C_SKY_TOP, sky_bottom=C_SKY_BOT,
            fog_color=C_FOG, fog_density=0.0016,
            ambient=(0.22, 0.24, 0.29),
        )
        self.renderer.filter_mode = "nearest"     # Extension 2 switch
        self.renderer.depth_debug = False         # F3
        self.quality = 1
        self.camera = e3.Camera(fov=62.0, near=1.0, far=1400.0)
        self.scaled = pygame.Surface((WIDTH, HEIGHT)).convert()
        self.fullscreen = False
        self._windowed_size = (WIDTH, HEIGHT)
        self.sharpen = True               # C: unsharp mask before upscaling

        self._dt_hist = []
        self._slow_run = 0
        self._fast_run = 0
        self._q_cool = 0.0
        self._q_grace = 0.0

        # ---- Mesh resources ----
        self.box_h = e3.box((CELL + WALL_T, WALL_H, WALL_T))
        self.box_v = e3.box((WALL_T, WALL_H, CELL + WALL_T))
        self.box_unit = e3.box((1.0, 1.0, 1.0))
        self.sphere_lo = e3.sphere(1.0, 8, 4, smooth=True)
        self.sphere_md = e3.sphere(1.0, 10, 5, smooth=True)
        self.sphere_sm = e3.sphere(1.0, 5, 3, smooth=True)
        self.cone_sm = e3.cone(1.0, 1.0, 6)
        self.cyl_sm = e3.cylinder(1.0, 1.0, 6, smooth=True)
        self.disc_shadow = e3.disc(1.0, 12, 1)
        self.leaf_mesh = e3.quad((0, 0, 0), (9, 5), normal=(0, 0.4, 1))

        # ---- Textures ----
        self.tex_stone = textures.get("stone")
        self.tex_floor = textures.get("floor")
        self.tex_grass = textures.get("grass")
        self.tex_checker = textures.get("checker")
        self.tex_wood = textures.get("wood")
        self.tex_water = textures.get("water")
        self.tex_cloth = textures.get("cloth")

        # ---- Camera / State ----
        self.cam_yaw = 0.0
        self.cam_pitch = -0.92
        self.cam_dist = 138.0
        self.cam_target = np.array([0.0, 18.0, 0.0])
        self.projection = "perspective"
        self.show_help = False

        self.rng = random.Random()
        self.score = 0
        self.level = 1
        self.time = 0.0
        self.banner = ""
        self.banner_t = 0.0
        self.running = True

        self.particles = []
        self.firework_lights = []
        self.dying = 0.0

        self.leg_amp = 0.0
        self.vspeed = 0.0
        self.trail = []

        # Caches
        self._cells = []
        self._cell_centers = np.zeros((0, 3))
        self._ground_models = []
        self._topbar = None
        self._hint_bar = None
        self._hint_txt = None
        self._banner_cache = None
        self._banner_cache_key = None
        self._minimap_static = None
        self._minimap_dirty = True

        self.build_level(first=True)
        self.reset_player(snap_camera=True)
        self.time = 0.0

        gc.collect()
        gc.freeze()

    # --------------------------------------------------------
    def build_level(self, first=False):
        rng = self.rng
        self.maze = Maze(COLS, ROWS, rng)
        self.start_cell = self.maze.start
        self.finish_cell = self.maze.finish

        self.collision = self.maze.collision_rects()
        self.collision_grid = {}
        for rect in self.collision:
            gx0 = int(math.floor(rect[0] / CELL))
            gx1 = int(math.floor(rect[2] / CELL))
            gz0 = int(math.floor(rect[1] / CELL))
            gz1 = int(math.floor(rect[3] / CELL))
            for gx in range(gx0, gx1 + 1):
                for gz in range(gz0, gz1 + 1):
                    self.collision_grid.setdefault((gx, gz), []).append(rect)

        self.cell_meshes = {}
        by_cell = {}
        for cx, cz, sx, sz in self.maze.wall_boxes():
            gx = int(cx // CELL)
            gz = int(cz // CELL)
            by_cell.setdefault((gx, gz), []).append((cx, cz, sx, sz))

        for (gx, gz), boxes in by_cell.items():
            cc = self.maze.center((gx, gz))
            parts = []
            for cx, cz, sx, sz in boxes:
                src = self.box_h if sx > sz else self.box_v
                parts.append(shift_mesh(src, cx - cc[0], WALL_H * 0.5, cz - cc[2]))
            mesh = e3.merge(parts) if len(parts) > 1 else parts[0]
            self.cell_meshes[(gx, gz)] = (mesh, cc)

        self._cells = [(mesh, e3.translation(cc[0], 0.0, cc[2]), cc)
                       for (mesh, cc) in self.cell_meshes.values()]
        if self._cells:
            self._cell_centers = np.array([c[2] for c in self._cells])
        else:
            self._cell_centers = np.zeros((0, 3))

        self.ground = e3.plane(1.0, 1.0)
        self.ground_tiles = [
            (MAZE_W * 0.5, -1800.0, MAZE_W + 3600.0, 3600.0),
            (MAZE_W * 0.5, MAZE_D + 1800.0, MAZE_W + 3600.0, 3600.0),
            (-1800.0, MAZE_D * 0.5, 3600.0, MAZE_D),
            (MAZE_W + 1800.0, MAZE_D * 0.5, 3600.0, MAZE_D),
        ]
        self._ground_models = [
            (e3.translation(cx, -1.0, cz) @ e3.scaling(w, 1.0, d), w, d)
            for (cx, cz, w, d) in self.ground_tiles
        ]
        self.floor_mesh = self.build_tiled_floor(8, 5)
        self.leaf_mesh = e3.quad((0, 0, 0), (9, 5), normal=(0, 0.4, 1))

        # ---- Water pools ----
        self.water_pools = []
        cells = [(x, y) for y in range(ROWS) for x in range(COLS)
                 if (x, y) not in (self.start_cell, self.finish_cell)]
        rng.shuffle(cells)
        for cell in cells[:WATER_POOLS]:
            p = self.maze.center(cell)
            p[1] = 0.6
            self.water_pools.append({
                "pos": p, "cell": cell, "phase": rng.uniform(0, 6.28),
                "mesh": e3.disc(1.0, 12, 2),
                "model": e3.translation(*p) @ e3.scaling(WATER_R, 1.0, WATER_R),
            })
        water_cells = {p["cell"] for p in self.water_pools}

        # ---- Apples ----
        self.apples = []
        cells = [c for c in cells if c not in water_cells]
        rng.shuffle(cells)
        kinds = [("silver", SILVER_VALUE)] * 3 + [("gold", GOLD_VALUE)]
        for cell, (kind, value) in zip(cells, kinds):
            p = self.maze.center(cell)
            self.apples.append({
                "pos": p, "type": kind, "value": value,
                "collected": False, "phase": rng.uniform(0, 6.28),
            })
        self.medals = len(self.apples)

        # ---- Birds ----
        self.birds = []
        count = min(2 + self.level, 5)
        far_cells = [c for c in cells
                     if abs(c[0] - self.start_cell[0]) + abs(c[1] - self.start_cell[1]) > 5]
        rng.shuffle(far_cells)
        for cell in far_cells[:count]:
            nbrs = self.maze.neighbors(cell)
            if not nbrs:
                continue
            target = rng.choice(nbrs)
            p = self.maze.center(cell)
            p[1] = BIRD_FLY_HEIGHT
            self.birds.append({
                "cell": cell, "target": target, "prev": cell,
                "pos": p, "speed": BIRD_SPEED + rng.uniform(-10, 18),
                "heading": rng.uniform(0, 6.28), "phase": rng.uniform(0, 6.28),
            })

        # ---- Flags ----
        self.start_flag = self.maze.center(self.start_cell) + np.array(
            [-CELL * 0.42, 0.0, CELL * 0.40])
        self.finish_flag = self.maze.center(self.finish_cell) + np.array(
            [CELL * 0.56, 0.0, 0.0])
        self.flag_start_mesh = flag_grid()
        self.flag_finish_mesh = flag_grid()

        self.collected = 0
        self.time = 0.0
        self._minimap_dirty = True
        if not first:
            self.banner = f"LEVEL {self.level}"
            self.banner_t = 2.2

    def build_tiled_floor(self, nx, ny):
        verts, uvs, faces = [], [], []
        for j in range(ny):
            for i in range(nx):
                x0 = MAZE_W * i / nx
                x1 = MAZE_W * (i + 1) / nx
                z0 = MAZE_D * j / ny
                z1 = MAZE_D * (j + 1) / ny
                base = len(verts)
                verts += [(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]
                uvs += [(x0 / CELL, z0 / CELL), (x1 / CELL, z0 / CELL),
                        (x1 / CELL, z1 / CELL), (x0 / CELL, z1 / CELL)]
                faces += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
        return e3.Mesh(verts, faces, uvs, smooth=False)

    # --------------------------------------------------------
    def _make_renderer(self):
        """(Re)build the framebuffer renderer at the current quality level."""
        s = QUALITY_LEVELS[self.quality][1]
        old = getattr(self, "renderer", None)
        r = e3.Renderer(
            max(160, int(WIDTH * s)), max(120, int(HEIGHT * s)),
            sky_top=C_SKY_TOP, sky_bottom=C_SKY_BOT,
            fog_color=C_FOG, fog_density=0.0016,
            ambient=(0.22, 0.24, 0.29),
        )
        if old is not None:
            r.textures_enabled = old.textures_enabled
            r.fog_enabled = old.fog_enabled
            r.wireframe = old.wireframe
            r.filter_mode = old.filter_mode
            r.depth_debug = old.depth_debug
        self.renderer = r

    def set_quality(self, q):
        q = max(0, min(len(QUALITY_LEVELS) - 1, q))
        if q == self.quality:
            return
        self.quality = q
        self._make_renderer()
        self._dt_hist.clear()
        self._slow_run = 0
        self._fast_run = 0
        self._q_cool = Q_COOLDOWN
        self._q_grace = Q_GRACE

    # --------------------------------------------------------
    def _sync_layout(self):
        """Follow a new window size: rebuild HUD caches and the framebuffer."""
        global WIDTH, HEIGHT
        WIDTH, HEIGHT = self.screen.get_size()
        self.scaled = pygame.Surface((WIDTH, HEIGHT)).convert()
        self._topbar = None
        self._hint_bar = None
        self._hint_txt = None
        self._minimap_static = None
        self._minimap_dirty = True
        self._banner_cache_key = None
        self._make_renderer()
        self._dt_hist.clear()
        self._q_cool = max(self._q_cool, 0.6)

    def resize_window(self, w, h):
        self.screen = pygame.display.set_mode((max(800, int(w)), max(600, int(h))),
                                              pygame.RESIZABLE)
        self._sync_layout()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self._windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.banner = "Fullscreen (F11 to exit)"
        else:
            self.screen = pygame.display.set_mode(self._windowed_size,
                                                  pygame.RESIZABLE)
            self.banner = "Windowed mode"
        self.banner_t = 1.4
        self._sync_layout()

    def auto_quality(self, dt):
        if self._q_cool > 0.0:
            self._q_cool -= dt
            return
        if self._q_grace > 0.0:
            self._q_grace -= dt
            return

        self._dt_hist.append(dt)
        if len(self._dt_hist) > Q_WINDOW:
            self._dt_hist.pop(0)
        if len(self._dt_hist) < Q_WINDOW:
            return

        med = sorted(self._dt_hist)[Q_WINDOW // 2]
        if med > Q_SLOW:
            self._slow_run += 1
            self._fast_run = 0
        elif med < Q_FAST:
            self._fast_run += 1
            self._slow_run = 0
        else:
            self._slow_run = 0
            self._fast_run = 0

        if self._slow_run >= Q_CONFIRM and self.quality > 0:
            self.set_quality(self.quality - 1)
        elif self._fast_run >= Q_CONFIRM and self.quality < len(QUALITY_LEVELS) - 1:
            self.set_quality(self.quality + 1)

    # --------------------------------------------------------
    def reset_player(self, snap_camera=False):
        p = self.maze.center(self.start_cell)
        self.pos = np.array([p[0], PLAYER_Y, p[2]])
        self.heading = math.pi * 0.5
        self.vspeed = 0.0
        self.jump_h = 0.0
        self.jump_cd = 0.0
        self.spark_t = 0.0
        self.walk_time = 0.0
        self.leg_amp = 0.0
        self.trail = [self.pos.copy() for _ in range(10)]
        if snap_camera:
            self.cam_target = self.pos + np.array([0.0, 16.0, 0.0])
        self.dying = 0.0

    def restart_run(self):
        self.score = 0
        self.level = 1
        self.build_level(first=True)
        self.reset_player(snap_camera=True)
        self.particles.clear()
        self.firework_lights.clear()

    # --------------------------------------------------------
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
            self.resize_window(event.w, event.h)
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            elif event.key == pygame.K_r:
                self.restart_run()
            elif event.key == pygame.K_SPACE:
                if self.jump_h <= 0.5 and self.jump_cd <= 0:
                    self.vspeed = JUMP_SPEED
                    self.jump_cd = JUMP_COOLDOWN
                    self.burst(self.pos + np.array([0, -8, 0]), 10,
                               (0.6, 1.0, 0.5), speed=70, up=90)
            elif event.key in (pygame.K_v, pygame.K_p):
                self.projection = ("orthographic" if self.projection == "perspective"
                                   else "perspective")
                self.banner = ("Orthographic Projection" if self.projection == "orthographic"
                               else "Perspective Projection")
                self.banner_t = 1.6
            elif event.key == pygame.K_t:
                self.renderer.textures_enabled = not self.renderer.textures_enabled
                self.banner = "Textures: " + ("ON" if self.renderer.textures_enabled else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_f:
                self.renderer.fog_enabled = not self.renderer.fog_enabled
                self.banner = "Fog (extension 1): " + ("ON" if self.renderer.fog_enabled else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_b:
                r = self.renderer
                r.filter_mode = "bilinear" if r.filter_mode == "nearest" else "nearest"
                self.banner = "Texture filter (extension 2): " + r.filter_mode.upper()
                self.banner_t = 1.4
            elif event.key in (pygame.K_g, pygame.K_F1):
                self.renderer.wireframe = not self.renderer.wireframe
                self.banner = "Wireframe: " + ("ON" if self.renderer.wireframe else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_F3:
                self.renderer.depth_debug = not self.renderer.depth_debug
                self.banner = "Depth buffer: " + ("ON" if self.renderer.depth_debug else "OFF")
                self.banner_t = 1.2
            elif event.key == pygame.K_F5:
                self.set_quality((self.quality + 1) % len(QUALITY_LEVELS))
                self.banner = f"Quality: {QUALITY_LEVELS[self.quality][0]}"
                self.banner_t = 1.2
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
            elif event.key == pygame.K_F11:
                self.toggle_fullscreen()
            elif event.key == pygame.K_c:
                self.sharpen = not self.sharpen
                self.banner = "Sharpen: " + ("ON" if self.sharpen else "OFF")
                self.banner_t = 1.2
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                ground = self.screen_to_ground(event.pos)
                if ground is not None:
                    self.create_firework(ground)
            elif event.button == 4:
                self.cam_dist = max(70.0, self.cam_dist * 0.9)
            elif event.button == 5:
                self.cam_dist = min(280.0, self.cam_dist * 1.1)
        elif event.type == pygame.MOUSEMOTION:
            buttons = pygame.mouse.get_pressed()
            if buttons[2]:
                self.cam_yaw -= event.rel[0] * 0.005
                self.cam_pitch = min(-0.30, max(-1.28,
                                                self.cam_pitch - event.rel[1] * 0.004))

    # --------------------------------------------------------
    def screen_to_ground(self, mouse):
        x = (mouse[0] / WIDTH) * 2.0 - 1.0
        y = 1.0 - (mouse[1] / HEIGHT) * 2.0
        inv = np.linalg.inv(self.renderer.mvp)
        p0 = inv @ np.array([x, y, -1.0, 1.0])
        p1 = inv @ np.array([x, y, 1.0, 1.0])
        p0 /= p0[3]
        p1 /= p1[3]
        d = p1[:3] - p0[:3]
        if abs(d[1]) < 1e-6:
            return None
        t = (0.0 - p0[1]) / d[1]
        if t <= 0:
            return None
        hit = p0[:3] + d * t
        hit[0] = min(max(hit[0], -200.0), MAZE_W + 200.0)
        hit[2] = min(max(hit[2], -200.0), MAZE_D + 200.0)
        return hit

    def create_firework(self, pos):
        colors = [(0.25, 0.6, 1.0), (1.0, 0.35, 0.6), (0.7, 0.3, 1.0), (1.0, 0.8, 0.25)]
        col = self.rng.choice(colors)
        for _ in range(self.rng.randint(60, 95)):
            ang = self.rng.uniform(0, math.tau)
            sp = self.rng.uniform(40, 190)
            vel = np.array([math.cos(ang) * sp,
                            self.rng.uniform(120, 260),
                            math.sin(ang) * sp])
            life = self.rng.uniform(0.7, 1.4)
            self.add_particle(pos + np.array([0, 2, 0]), vel, life,
                              self.rng.randint(2, 4), col)
        self.firework_lights.append({"pos": pos + np.array([0, 20.0, 0]),
                                     "t": 1.0, "color": col})

    def burst(self, pos, n, color, speed=120.0, up=60.0):
        for _ in range(n):
            ang = self.rng.uniform(0, math.tau)
            sp = self.rng.uniform(speed * 0.3, speed)
            self.add_particle(pos, np.array([math.cos(ang) * sp,
                                             self.rng.uniform(up * 0.2, up),
                                             math.sin(ang) * sp]),
                              self.rng.uniform(0.3, 0.7), self.rng.randint(2, 3), color)

    def add_particle(self, pos, vel, life, size, color):
        self.particles.append({"pos": np.array(pos, float), "vel": np.array(vel, float),
                               "life": life, "max_life": life, "size": size,
                               "color": np.array(color, float)})
        if len(self.particles) > MAX_PARTICLES:
            del self.particles[:len(self.particles) - MAX_PARTICLES]

    # --------------------------------------------------------
    def update(self, dt, keys):
        if self.dying > 0:
            self.dying -= dt
            if self.dying <= 0:
                self.reset_player()

        self.time += dt
        if self.banner_t > 0:
            self.banner_t -= dt

        # ---- Physical jump: explicit vertical velocity + gravity (Section 11.3)
        if self.jump_cd > 0:
            self.jump_cd = max(0.0, self.jump_cd - dt)
        if self.jump_h > 0.0 or self.vspeed != 0.0:
            self.vspeed -= GRAVITY * dt
            self.jump_h += self.vspeed * dt
            if self.jump_h <= 0.0:
                self.jump_h = 0.0
                self.vspeed = 0.0
        self.pos[1] = PLAYER_Y + self.jump_h

        yaw = self.cam_yaw
        fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        right = np.array([-math.cos(yaw), 0.0, math.sin(yaw)])

        move = np.zeros(3)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move += fwd
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move -= fwd
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move += right
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move -= right

        if keys[pygame.K_e]:
            self.cam_yaw += 1.8 * dt
        if keys[pygame.K_q]:
            self.cam_yaw -= 1.8 * dt

        moving = np.linalg.norm(move) > 1e-6
        if moving and self.dying <= 0:
            move = move / np.linalg.norm(move)
            self.move_player(move, dt)
            self.heading = math.atan2(move[0], move[2])
            self.walk_time += dt
            self.leg_amp = min(1.0, self.leg_amp + dt * 8.0)
            self.spark_t += dt
            while self.spark_t >= 0.05:
                self.spark_t -= 0.05
                p = self.pos + np.array([0, -12.0 + self.jump_h, 0])
                self.add_particle(p + np.array([self.rng.uniform(-4, 4), 0,
                                                self.rng.uniform(-4, 4)]),
                                  np.array([self.rng.uniform(-15, 15), 24,
                                            self.rng.uniform(-15, 15)]),
                                  0.4, 2, (0.85, 0.82, 0.75))
        else:
            self.leg_amp = max(0.0, self.leg_amp - dt * 8.0)

        # ---- Worm body trail: segments follow earlier head positions (Section 12.2)
        self.trail.insert(0, self.pos.copy())
        if len(self.trail) > 14:
            del self.trail[14:]

        self.update_birds(dt)
        self.update_particles(dt)
        if self.dying <= 0:
            self.check_pickups()
            self.check_hazards()
            self.check_finish()
        self.update_camera(dt)

    def move_player(self, direction, dt):
        movement = direction * PLAYER_SPEED * dt
        steps = max(1, int(math.ceil(max(abs(movement[0]), abs(movement[2])) / 3.0)))
        sx = movement[0] / steps
        sz = movement[2] / steps
        for _ in range(steps):
            nx = self.pos[0] + sx
            if not self.circle_hits_wall(nx, self.pos[2], PLAYER_R):
                self.pos[0] = nx
            nz = self.pos[2] + sz
            if not self.circle_hits_wall(self.pos[0], nz, PLAYER_R):
                self.pos[2] = nz
        self.pos[1] = PLAYER_Y + self.jump_h

    def circle_hits_wall(self, x, z, r):
        for x0, z0, x1, z1 in self.collision:
            cx = max(x0, min(x, x1))
            cz = max(z0, min(z, z1))
            dx, dz = x - cx, z - cz
            if dx * dx + dz * dz < r * r:
                return True
        return False

    def update_birds(self, dt):
        for d in self.birds:
            target = self.maze.center(d["target"])
            target[1] = BIRD_FLY_HEIGHT
            delta = target - d["pos"]
            dist = np.linalg.norm(delta)
            if dist <= 2.0:
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
                    d["pos"] = target.copy()
                else:
                    d["pos"] = d["pos"] + delta / dist * step
                d["heading"] = math.atan2(delta[0], delta[2])
            d["pos"][1] = BIRD_FLY_HEIGHT + math.sin(self.time * 2.2 + d["phase"]) * 5.0

    def update_particles(self, dt):
        alive = 0
        for p in self.particles:
            p["vel"][1] -= 210.0 * dt
            p["pos"] += p["vel"] * dt
            p["life"] -= dt
            if p["life"] > 0 and p["pos"][1] > -3:
                self.particles[alive] = p
                alive += 1
        if alive < len(self.particles):
            del self.particles[alive:]

        for light in self.firework_lights:
            light["t"] -= dt * 1.8
        self.firework_lights = [l for l in self.firework_lights if l["t"] > 0]

    def check_pickups(self):
        for apple in self.apples:
            if apple["collected"]:
                continue
            d = self.pos - np.array([apple["pos"][0], 16.0, apple["pos"][2]])
            if np.linalg.norm(d) < 24.0:
                apple["collected"] = True
                self.collected += 1
                self.score += apple["value"]
                col = (1.0, 0.75, 0.2) if apple["type"] == "gold" else (1.0, 0.35, 0.35)
                self.burst(apple["pos"] + np.array([0, 16, 0]), 26, col, speed=120, up=110)
                if self.collected >= self.medals:
                    self.banner = "Exit unlocked! Head to the finish flag!"
                    self.banner_t = 2.0
                self._minimap_dirty = True

    def check_hazards(self):
        """Height-aware hazard rules (Section 11.4).

        Water only kills when the worm's feet are below the safe clearance.
        Birds use a true 3-D sphere test, so a high jump can pass under a bird
        only if the vertical gap is larger than the sum of the radii.
        """
        feet = self.pos[1] - PLAYER_R
        for pool in self.water_pools:
            d = self.pos - pool["pos"]
            if feet < WATER_SAFE_CLEARANCE and \
                    d[0] ** 2 + d[2] ** 2 < (PLAYER_R + WATER_R) ** 2:
                self.kill_player("water")
                return
        for b in self.birds:
            if np.linalg.norm(self.pos - b["pos"]) < PLAYER_R + BIRD_R:
                self.kill_player("bird")
                return

    def kill_player(self, why):
        self.dying = 0.9
        self.burst(self.pos, 30, (0.4, 0.75, 1.0) if why == "water" else (0.4, 0.9, 1.0),
                   speed=130, up=120)
        self.banner = "Splash! The worm fell into the water!" if why == "water" else "Snatched by a bird!"
        self.banner_t = 1.4

    def check_finish(self):
        d = self.pos - self.finish_flag
        if d[0] ** 2 + d[2] ** 2 < 20.0 ** 2:
            if self.collected >= self.medals:
                self.level += 1
                self.burst(self.finish_flag + np.array([0, 20, 0]), 70,
                           (1.0, 0.85, 0.3), speed=180, up=220)
                self.build_level()
                self.reset_player()
            else:
                self.banner = f"Need {self.medals - self.collected} more apples"
                self.banner_t = 1.0

    def update_camera(self, dt):
        # The camera aim follows the ground position, not the jump height, so a
        # vertical jump is visible as the worm rising inside the frame.
        target = np.array([self.pos[0], 16.0, self.pos[2]])
        self.cam_target = self.cam_target + (target - self.cam_target) * min(1.0, dt * 7.0)

        pitch = self.cam_pitch
        yaw = self.cam_yaw
        fwd = np.array([math.cos(pitch) * math.sin(yaw),
                        math.sin(pitch),
                        math.cos(pitch) * math.cos(yaw)])
        dist = self.cam_dist
        desired = self.cam_target - fwd * dist
        t = 0.15
        while t < 1.0:
            p = self.cam_target + (desired - self.cam_target) * t
            if p[1] < WALL_H + 3.0 and self.point_in_wall(p[0], p[2], 5.0):
                dist = max(45.0, dist * t * 0.92)
                break
            t += 0.06
        eye = self.cam_target - fwd * dist
        eye[1] = max(eye[1], 10.0)

        cam = self.camera
        cam.projection = self.projection
        cam.ortho_height = self.cam_dist * 1.35
        cam.look(eye, self.cam_target)

    def point_in_wall(self, x, z, pad):
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
        r = self.renderer
        cam = self.camera
        r.begin(cam)
        r.add_directional_light((0.42, -1.0, 0.30), (1.0, 0.96, 0.86), 0.62)
        r.add_point_light(self.pos + np.array([0.0, 14.0, 0.0]),
                          (1.0, 0.80, 0.48), 0.62, 210.0)
        for light in self.firework_lights:
            r.add_point_light(light["pos"], tuple(light["color"]),
                              0.8 * light["t"], 260.0)

        wf = r.wireframe
        if wf:
            r.wireframe = False
            for model, w, d in self._ground_models:
                r.draw_mesh(self.ground, model,
                            color=(0.045, 0.06, 0.07), lighting=False, cull=False)
            r.draw_mesh(self.floor_mesh, color=(0.09, 0.12, 0.13),
                        lighting=False, cull=False)
            r.wireframe = True
            r.draw_mesh(self.floor_mesh, color=(0.9, 0.95, 1.0), cull=False)
        else:
            for model, w, d in self._ground_models:
                r.draw_mesh(self.ground, model,
                            texture=self.tex_grass, uv_scale=(w / 60.0, d / 60.0),
                            fog_vertex=False, cull=False)
            r.draw_mesh(self.floor_mesh, texture=self.tex_floor,
                        color=(0.50, 0.50, 0.46) if not r.textures_enabled else (1.0, 1.0, 1.0),
                        fog_vertex=True, cull=False)

        if self._cells:
            eye = cam.eye
            dx = self._cell_centers[:, 0] - eye[0]
            dz = self._cell_centers[:, 2] - eye[2]
            d2 = dx * dx + dz * dz
            mask = d2 < (VIEW_DISTANCE * VIEW_DISTANCE)
            idx = np.nonzero(mask)[0]
            if idx.size:
                order = idx[np.argsort(d2[idx])].tolist()
                for i in order:
                    mesh, model, _ = self._cells[i]
                    r.draw_mesh(mesh, model,
                                texture=self.tex_stone,
                                uv_scale=(1.0 / 44.0, 1.0 / 44.0),
                                color=(0.60, 0.59, 0.56) if not r.textures_enabled else (1.0, 1.0, 1.0),
                                fog_vertex=True)

        for pool in self.water_pools:
            self.draw_water(r, pool)
        for apple in self.apples:
            if not apple["collected"]:
                self.draw_apple(r, apple)
        self.draw_flag(r, self.start_flag, self.flag_start_mesh, (0.33, 0.78, 0.28),
                       "start", unlocked=True)
        self.draw_flag(r, self.finish_flag, self.flag_finish_mesh, (1.0, 1.0, 1.0),
                       "finish", unlocked=self.collected >= self.medals)
        for d in self.birds:
            self.draw_bird(r, d)
        self.draw_worm(r)

        for p in self.particles:
            a = max(0.0, p["life"] / p["max_life"])
            r.draw_glow(p["pos"], 1.6 + p["size"] * 0.55, p["color"], 1.1 * a)
        for light in self.firework_lights:
            r.draw_glow(light["pos"], 10.0, light["color"], 0.7 * light["t"])

        surf = r.end()
        target = self.screen.get_size()
        if surf.get_size() == target:
            self.screen.blit(surf, (0, 0))
        else:
            if self.sharpen:
                surf = sharpen_surface(surf, 0.55)
            if SHARP_UPSCALE:
                pygame.transform.scale(surf, target, self.scaled)
            else:
                pygame.transform.smoothscale(surf, target, self.scaled)
            self.screen.blit(self.scaled, (0, 0))
        self.draw_hud()

    def draw_water(self, r, pool):
        mesh = pool["mesh"]
        v = mesh.vertices
        rad = np.sqrt(v[:, 0] * v[:, 0] + v[:, 2] * v[:, 2])
        v[:, 1] = np.sin(rad * 9.0 - self.time * 3.2 + pool["phase"]) * 2.2 * rad
        r.draw_mesh(mesh, pool["model"], color=(0.55, 0.80, 1.00),
                    texture=self.tex_water, emissive=0.22, double_sided=True)

    def draw_apple(self, r, apple):
        bob = math.sin(self.time * 2.4 + apple["phase"]) * 2.5
        p = apple["pos"] + np.array([0.0, 16.0 + bob, 0.0])
        spin = self.time * 1.6 + apple["phase"]
        color = C_GOLD if apple["type"] == "gold" else C_SILVER
        model = e3.translation(*p) @ e3.rotation_y(spin) @ e3.scaling(9.0, 8.6, 9.0)
        r.draw_mesh(self.sphere_lo, model, color=color, emissive=0.06)
        r.draw_mesh(self.cyl_sm, e3.translation(*p) @ e3.translation(0, 10, 0)
                    @ e3.scaling(0.9, 6.0, 0.9),
                    color=(0.35, 0.22, 0.10), smooth=True)
        model = (e3.translation(*p) @ e3.translation(3, 13, 0)
                 @ e3.rotation_y(spin + 0.6) @ e3.rotation_z(0.5))
        r.draw_mesh(self.leaf_mesh, model, color=(0.30, 0.65, 0.25),
                    double_sided=True, smooth=False)

    def draw_flag(self, r, base, mesh, color, kind, unlocked):
        pole_h = 76.0 if kind == "finish" else 58.0
        r.draw_mesh(self.cyl_sm,
                    e3.translation(base[0], pole_h * 0.5, base[2])
                    @ e3.scaling(1.2, pole_h, 1.2),
                    color=(0.55, 0.42, 0.26), texture=self.tex_wood, smooth=True)
        r.draw_mesh(self.sphere_sm,
                    e3.translation(base[0], pole_h, base[2]) @ e3.scaling(2.6),
                    color=C_GOLD, emissive=0.35, smooth=True)
        v = mesh.vertices
        xs = v[:, 0]
        v[:, 2] = np.sin(xs * 0.30 + self.time * 5.0) * 2.4 * (xs / 30.0)
        tex = self.tex_checker if kind == "finish" else self.tex_cloth
        model = e3.translation(base[0] + 1.0, pole_h - 3.0, base[2]) @ e3.rotation_y(-0.25)
        r.draw_mesh(mesh, model, color=color if kind == "start" else (1.0, 1.0, 1.0),
                    texture=tex, double_sided=True, smooth=False,
                    emissive=0.25 if (kind == "finish" and unlocked) else 0.0)
        if kind == "finish" and unlocked:
            r.draw_glow(base + np.array([0.0, pole_h, 0.0]), 8.0, (1.0, 0.9, 0.4), 0.8)

    def draw_bird(self, r, d):
        p = d["pos"]
        heading = d["heading"]
        flap = math.sin(self.time * 6.0 + d["phase"]) * 0.9
        base = e3.translation(*p) @ e3.rotation_y(-heading)

        # Body
        r.draw_mesh(self.sphere_lo, base @ e3.scaling(10.5, 6.5, 7.5),
                    color=BIRD_BODY, smooth=True)
        # Belly
        r.draw_mesh(self.sphere_lo, base @ e3.translation(1.0, -1.5, 0)
                    @ e3.scaling(9.0, 3.0, 5.5),
                    color=BIRD_BELLY, smooth=True)
        # Head
        r.draw_mesh(self.sphere_sm, base @ e3.translation(9.0, 2.5, 0) @ e3.scaling(5.5),
                    color=BIRD_BODY, smooth=True)
        # Snout
        r.draw_mesh(self.cone_sm,
                    base @ e3.translation(13.5, 2.0, 0) @ e3.rotation_z(-math.pi / 2)
                    @ e3.scaling(2.0, 5.5, 2.0),
                    color=BIRD_BODY_DARK, smooth=False)
        # Horns
        for side in (-1, 1):
            r.draw_mesh(self.cone_sm,
                        base @ e3.translation(8.5, 5.0, side * 1.8)
                        @ e3.rotation_z(-0.4) @ e3.rotation_x(side * 0.3)
                        @ e3.scaling(1.1, 4.0, 1.1),
                        color=BIRD_HORN, smooth=False)
        # Eyes (glowing)
        for side in (-1, 1):
            r.draw_mesh(self.sphere_sm,
                        base @ e3.translation(9.8, 3.5, side * 1.8) @ e3.scaling(1.2),
                        color=BIRD_EYE, emissive=0.9, smooth=True)
        # Tail
        r.draw_mesh(self.cone_sm,
                    base @ e3.translation(-11.5, 1.0, 0) @ e3.rotation_z(math.pi / 2)
                    @ e3.scaling(2.6, 9.0, 1.4),
                    color=BIRD_BODY_DARK, smooth=False)
        # Tail tip
        r.draw_mesh(self.cone_sm,
                    base @ e3.translation(-18.0, 1.0, 0) @ e3.rotation_z(math.pi / 2)
                    @ e3.scaling(1.0, 3.5, 0.8),
                    color=BIRD_HORN, smooth=False)
        # Wings (flap around X axis)
        for side in (-1, 1):
            wing = (base @ e3.translation(-1.0, 2.5, side * 4.0)
                    @ e3.rotation_x(side * flap)
                    @ e3.translation(0.0, 0.0, side * 7.0)
                    @ e3.scaling(12.0, 1.0, 13.0))
            r.draw_mesh(self.box_unit, wing, color=BIRD_WING, smooth=False)

    def draw_worm(self, r):
        # Contact shadow: shrinks as the worm rises, which makes the world-space
        # jump height readable on screen.
        if self.jump_h > 1.0:
            sc = max(0.45, 1.0 - self.jump_h / 90.0)
            r.draw_mesh(self.disc_shadow,
                        e3.translation(self.pos[0], 0.35, self.pos[2])
                        @ e3.scaling(13.5 * sc, 1.0, 9.5 * sc),
                        color=(0.08, 0.18, 0.08), lighting=False,
                        emissive=0.05, double_sided=True)

        fwd = np.array([math.sin(self.heading), 0.0, math.cos(self.heading)])
        trail = self.trail if self.trail else [self.pos.copy()]
        # Body segments follow a short history of the head position (Section 12.2).
        if np.linalg.norm(trail[0] - trail[-1]) > 7.0:
            marks = [trail[min(i * 3, len(trail) - 1)] for i in range(5)]
        else:
            # Standing still: lay the segments out behind the head so the worm
            # still reads as a segmented body instead of one blob.
            marks = [self.pos - fwd * (7.5 * i) for i in range(5)]

        radii = (10.6, 9.2, 8.2, 7.2, 5.8)
        shades = (WORM_GREEN, WORM_GREEN, (0.36, 0.80, 0.30),
                  (0.27, 0.66, 0.22), WORM_GREEN_DARK)
        # Draw tail-to-head so the head sphere wins any depth ties.
        for k in range(4, -1, -1):
            q = marks[k]
            rad = radii[k]
            r.draw_mesh(self.sphere_lo,
                        e3.translation(q[0], q[1], q[2])
                        @ e3.scaling(rad, rad * 0.80, rad),
                        color=shades[k], smooth=True)

        # Head: bright green sphere with two eyes on the forward/right basis.
        head = np.array([self.pos[0], self.pos[1], self.pos[2]])
        r.draw_mesh(self.sphere_lo, e3.translation(*head)
                    @ e3.scaling(10.8, 8.8, 10.8), color=WORM_GREEN_LIGHT,
                    smooth=True)
        right = np.array([math.cos(self.heading), 0.0, -math.sin(self.heading)])
        for side in (-1, 1):
            eye = head + fwd * 6.0 + right * (side * 3.7) + np.array([0.0, 4.6, 0.0])
            r.draw_mesh(self.sphere_sm, e3.translation(*eye) @ e3.scaling(2.3),
                        color=WORM_EYE, emissive=0.30, smooth=True)
            r.draw_mesh(self.sphere_sm,
                        e3.translation(*(eye + fwd * 1.4)) @ e3.scaling(1.1),
                        color=WORM_PUPIL, smooth=True)
        # Light belly stripe under the head.
        r.draw_mesh(self.sphere_lo,
                    e3.translation(head[0], head[1] - 5.0, head[2])
                    @ e3.scaling(8.0, 2.4, 5.2),
                    color=WORM_BELLY, smooth=True)

    def draw_bird(self, r, d):
        p = d["pos"]
        heading = d["heading"]
        flap = math.sin(self.time * 7.0 + d["phase"]) * 0.85
        base = e3.translation(*p) @ e3.rotation_y(-heading)

        # Body (dark ellipsoid) and lighter belly.
        r.draw_mesh(self.sphere_lo, base @ e3.scaling(11.0, 6.0, 7.0),
                    color=BIRD_BODY, smooth=True)
        r.draw_mesh(self.sphere_lo, base @ e3.translation(1.0, -1.8, 0)
                    @ e3.scaling(9.0, 2.8, 5.2), color=BIRD_BELLY, smooth=True)
        # Head.
        r.draw_mesh(self.sphere_sm, base @ e3.translation(9.5, 2.8, 0)
                    @ e3.scaling(4.8), color=BIRD_BODY, smooth=True)
        # Beak (cone rotated to point along +X).
        r.draw_mesh(self.cone_sm, base @ e3.translation(14.0, 2.4, 0)
                    @ e3.rotation_z(-math.pi / 2) @ e3.scaling(1.4, 4.4, 1.4),
                    color=BIRD_HORN, smooth=False)
        # Eyes.
        for side in (-1, 1):
            eye = base @ e3.translation(10.2, 3.9, side * 1.9) @ e3.scaling(1.5)
            r.draw_mesh(self.sphere_sm, eye, color=WORM_EYE, emissive=0.2,
                        smooth=True)
            r.draw_mesh(self.sphere_sm,
                        base @ e3.translation(11.0, 3.9, side * 1.9)
                        @ e3.scaling(0.8), color=BIRD_BODY_DARK, smooth=True)
        # Tail.
        r.draw_mesh(self.cone_sm, base @ e3.translation(-12.0, 1.0, 0)
                    @ e3.rotation_z(math.pi / 2) @ e3.scaling(2.2, 8.0, 1.0),
                    color=BIRD_BODY_DARK, smooth=False)
        # Two flat wings that flap around the body axis (X).
        for side in (-1, 1):
            wing = (base @ e3.translation(-1.0, 2.0, side * 4.5)
                    @ e3.rotation_x(side * flap)
                    @ e3.translation(0.0, 0.0, side * 7.0)
                    @ e3.scaling(11.0, 0.8, 12.0))
            r.draw_mesh(self.box_unit, wing, color=BIRD_WING, smooth=False)

    # --------------------------------------------------------
    def draw_hud(self):
        s = self.screen

        if self._topbar is None:
            bar = pygame.Surface((WIDTH, 56), pygame.SRCALPHA).convert_alpha()
            bar.fill((12, 26, 20, 190))
            pygame.draw.line(bar, (90, 160, 110), (0, 56), (WIDTH, 56), 2)
            self._topbar = bar
        s.blit(self._topbar, (0, 0))

        s.blit(self.font.render(f"LEVEL: {self.level}", True, (235, 245, 235)), (18, 15))
        s.blit(self.font.render(f"SCORE: {self.score}", True, (255, 215, 90)), (170, 15))
        s.blit(self.font.render(f"APPLES: {self.collected}/{self.medals}",
                                True, (255, 150, 150)), (340, 15))
        s.blit(self.font.render(
            f"TIME: {int(self.time)//60:02d}:{int(self.time)%60:02d}",
            True, (235, 245, 235)), (560, 15))

        fps = int(self.clock.get_fps())
        r = self.renderer
        s.blit(self.small.render(
            f"FPS {fps}   TRIS {r.stats['triangles']}"
            f"   PX {r.stats['pixels']//1000}k   RES {r.w}x{r.h}",
            True, (150, 220, 170)), (760, 8))
        mode = "PERSP" if self.projection == "perspective" else "ORTHO"
        s.blit(self.small.render(
            f"{mode}  tex:{'on' if r.textures_enabled else 'off'}"
            f"  fog:{'on' if r.fog_enabled else 'off'}"
            f"  filt:{r.filter_mode}"
            f"  q:{QUALITY_LEVELS[self.quality][0]}",
            True, (150, 220, 170)), (760, 30))

        if self._hint_txt is None:
            hint = ("WASD Move   SPACE Jump   Right-drag Rotate   Wheel Zoom   "
                    "Click Firework   V Projection   T Textures   F Fog   B Filter   "
                    "F1 Wireframe   F3 Depth   F5 Quality   C Sharpen   F11 Fullscreen   "
                    "H Help   R Restart")
            self._hint_txt = self.small.render(hint, True, (230, 240, 230))
            self._hint_bar = pygame.Surface(
                (self._hint_txt.get_width() + 24, self._hint_txt.get_height() + 10),
                pygame.SRCALPHA).convert_alpha()
            self._hint_bar.fill((10, 24, 18, 170))
        hx = WIDTH // 2 - self._hint_bar.get_width() // 2
        hy = HEIGHT - 38
        s.blit(self._hint_bar, (hx, hy))
        s.blit(self._hint_txt,
               (WIDTH // 2 - self._hint_txt.get_width() // 2, HEIGHT - 33))

        self.draw_minimap()

        if self.banner_t > 0 and self.banner:
            if self._banner_cache_key != self.banner:
                txt = self.title_font.render(self.banner, True, (255, 240, 180))
                bg = pygame.Surface((txt.get_width() + 60, txt.get_height() + 26),
                                    pygame.SRCALPHA).convert_alpha()
                bg.fill((10, 20, 16, 255))
                self._banner_cache = (txt, bg)
                self._banner_cache_key = self.banner
            txt, bg = self._banner_cache
            a = min(1.0, self.banner_t / 0.5)
            bg.set_alpha(int(170 * a))
            x = WIDTH // 2 - bg.get_width() // 2
            s.blit(bg, (x, 140))
            s.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 152))

        if self.show_help:
            self.draw_help_panel()

    def draw_minimap(self):
        mm_w, mm_h = 230, 150
        x0 = WIDTH - mm_w - 22
        y0 = 72

        if self._minimap_static is None or self._minimap_dirty:
            surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA).convert_alpha()
            surf.fill((8, 20, 14, 185))
            sx = mm_w / MAZE_W
            sz = mm_h / MAZE_D
            for cx, cz, w, d in self.maze.wall_boxes():
                pygame.draw.rect(surf, (150, 170, 150),
                                 (cx * sx - max(w * sx, 1.4) * 0.5,
                                  cz * sz - max(d * sz, 1.4) * 0.5,
                                  max(w * sx, 1.4), max(d * sz, 1.4)))
            for pool in self.water_pools:
                pygame.draw.circle(surf, (255, 120, 40),
                                   (int(pool["pos"][0] * sx),
                                    int(pool["pos"][2] * sz)), 3)
            pygame.draw.circle(surf, (80, 240, 100),
                               (int(self.finish_flag[0] * sx),
                                int(self.finish_flag[2] * sz)), 4, 1)
            self._minimap_static = surf
            self._minimap_dirty = False

        self.screen.blit(self._minimap_static, (x0, y0))

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
        pygame.draw.circle(surf, (120, 200, 255), (px, py), 4)
        pygame.draw.line(surf, (120, 200, 255), (px, py),
                         (px + int(math.sin(self.heading) * 9),
                          py + int(math.cos(self.heading) * 9)), 2)
        self.screen.blit(surf, (x0, y0))

    def draw_help_panel(self):
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
        w, h = 760, 26 + len(lines) * 22
        x = (WIDTH - w) // 2
        y = 200
        panel = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
        panel.fill((8, 18, 14, 220))
        pygame.draw.rect(panel, (110, 210, 140), panel.get_rect(), 2)
        self.screen.blit(panel, (x, y))
        for i, line in enumerate(lines):
            col = (255, 235, 150) if i == 0 else (215, 235, 220)
            f = self.font if i == 0 else self.small
            self.screen.blit(f.render(line, True, col), (x + 18, y + 14 + i * 22))

    # --------------------------------------------------------
    def run(self):
        pygame.event.set_grab(False)
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                self.handle_event(event)
            keys = pygame.key.get_pressed()
            self.update(dt, keys)
            self.draw()
            pygame.display.flip()
            self.auto_quality(dt)
        pygame.quit()

    def step_headless(self, keys=None, dt=1 / 60):
        if keys is None:
            keys = pygame.key.get_pressed()
        self.update(dt, keys)
        self.draw()

if __name__ == "__main__":
    random.seed()
    Game().run()

