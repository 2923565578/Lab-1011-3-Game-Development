"""
tools/capture_screenshots.py

Renders every evidence screenshot required by the handbook (Section 16.3).
Headless: no window is opened.

    screenshots/00_wireframe_cube_maze.png   first wireframe cube + small maze
    screenshots/01_perspective_full.png     perspective view of the whole maze
    screenshots/02_orthographic_full.png    same camera, orthographic projection
    screenshots/03_wireframe.png            triangle wireframe of the maze
    screenshots/04_depth_buffer.png         z-buffer visualisation
    screenshots/05_occlusion.png            controlled z-buffer occlusion test
    screenshots/06_gameplay.png             worm + apple + bird + water in one frame
    screenshots/07_extension1_fog_on.png    Extension 1: fog ON
    screenshots/08_extension1_fog_off.png   Extension 1: fog OFF, same frame
    screenshots/09_extension2_nearest.png   Extension 2: nearest sampling
    screenshots/10_extension2_bilinear.png  Extension 2: bilinear sampling
    screenshots/11_resize_16x9.png          aspect test, 16:9
    screenshots/12_resize_4x3.png           aspect test, 4:3
"""

import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine3d as e3
import game3d as g3

OUT = os.path.join(ROOT, "screenshots")
os.makedirs(OUT, exist_ok=True)


def save(surface, name):
    pygame.image.save(surface, os.path.join(OUT, name))
    print("saved", name, surface.get_size())


def label(name, text):
    """Add a caption strip under a screenshot (for before/after evidence)."""
    path = os.path.join(OUT, name)
    img = Image.open(path).convert("RGB")
    bar = 52
    canvas = Image.new("RGB", (img.width, img.height + bar), (16, 18, 24))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((18, img.height + 12), text, fill=(235, 240, 235), font=font)
    canvas.save(path)


def set_view(g, eye, target, projection="perspective", ortho_height=None):
    g.projection = projection
    g.camera.projection = projection
    if ortho_height is not None:
        g.camera.ortho_height = ortho_height
    g.camera.look(eye, target)


def mini_wireframe_cube():
    """Handbook Appendix E: a cube, a floor and a small handmade maze."""
    r = e3.Renderer(1000, 650, fog_color=(0.10, 0.13, 0.18),
                    sky_top=(0.03, 0.04, 0.07), sky_bottom=(0.07, 0.09, 0.12))
    cam = e3.Camera(fov=55.0, near=0.5)
    cam.look((9, 9, -15), (7, 2.0, 8))
    r.begin(cam)
    r.fog_enabled = False
    r.wireframe = True
    r.draw_mesh(e3.plane(60, 60), e3.translation(10, 0, 10),
                color=(0.45, 0.55, 0.40), cull=False)
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(0.0, 2.6, 3.0) @ e3.scaling(5.2, 5.2, 5.2),
                color=(1.0, 0.85, 0.35))
    # small 3x3 "maze": outer walls plus two internal partitions
    walls = [
        (15, 2.0, 3.7, 14.0, 4.0, 0.6),
        (15, 2.0, 16.3, 14.0, 4.0, 0.6),
        (8.2, 2.0, 10.0, 0.6, 4.0, 13.0),
        (21.8, 2.0, 10.0, 0.6, 4.0, 13.0),
        (12.0, 2.0, 9.5, 0.6, 4.0, 6.0),
        (17.0, 2.0, 12.5, 0.6, 4.0, 6.0),
    ]
    for x, y, z, sx, sy, sz in walls:
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
    r = e3.Renderer(1000, 620)
    cam = e3.Camera(fov=50.0, near=0.5)
    cam.look((0, 9, -26), (0, 1.5, 0))
    r.begin(cam)
    r.add_directional_light((0.5, -0.9, 0.2))
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(1.2, 2.0, 7.0) @ e3.scaling(6, 6, 6),
                color=(0.20, 0.65, 0.30))
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(0.0, 2.0, 0.0) @ e3.scaling(4.5, 4.5, 4.5),
                color=(0.95, 0.75, 0.20))
    r.draw_mesh(e3.box((1, 1, 1)),
                e3.translation(-0.6, 1.6, -6.0) @ e3.scaling(3.2, 3.2, 3.2),
                color=(0.90, 0.25, 0.20))
    frame = r.end()
    save(pygame.transform.smoothscale(frame, (1000, 620)), "05_occlusion.png")
    label("05_occlusion.png",
          "Z-buffer test: the near red cube hides the yellow and green cubes behind it")


def aspect_tests():
    for (w, h), name in (((640, 360), "11_resize_16x9.png"),
                         ((640, 480), "12_resize_4x3.png")):
        r = e3.Renderer(w, h, fog_color=(0.55, 0.62, 0.72))
        cam = e3.Camera(fov=55.0, near=0.5)
        cam.look((10, 8, -16), (0, 2, 0))
        r.begin(cam)
        r.add_directional_light((0.4, -1.0, 0.3))
        for i in range(3):
            r.draw_mesh(e3.box((2, 2, 2)),
                        e3.translation(i * 2.5 - 2.5, 1.0, 0),
                        color=(0.90, 0.60, 0.20))
        r.draw_mesh(e3.plane(30, 30), e3.translation(0, 0, 0),
                    color=(0.25, 0.40, 0.25), cull=False)
        frame = r.end()
        scaled = pygame.transform.smoothscale(frame, (900, int(900 * h / w)))
        save(scaled, name)
        label(name, f"Aspect-ratio test {w}x{h}: cubes keep their proportions")


def texture_filter_demo():
    tex = np.zeros((8, 8, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:8, 0:8]
    tex[((yy // 2) + (xx // 2)) % 2 == 0] = 0.06
    tex[((yy // 2) + (xx // 2)) % 2 == 1] = 0.95
    r = e3.Renderer(900, 600, fog_color=(0, 0, 0))
    cam = e3.Camera(fov=42.0, near=0.5)
    cam.look((0, 17, -10), (0, 0, 1))
    for mode, name in (("nearest", "09_extension2_nearest.png"),
                       ("bilinear", "10_extension2_bilinear.png")):
        r.begin(cam)
        r.fog_enabled = False
        r.filter_mode = mode
        r.draw_mesh(e3.plane(30, 30, uv=(3, 3)), None,
                    color=(1, 1, 1), texture=tex, cull=False, lighting=False)
        frame = r.end()
        save(pygame.transform.smoothscale(frame, (900, 600)), name)
        label(name, f"Extension 2: {mode} sampling -- 8x8 checker, 3 UV repeats")


def main():
    mini_wireframe_cube()
    occlusion_demo()
    aspect_tests()
    texture_filter_demo()

    g = g3.Game(headless=True)
    g.rng = random.Random(2026)
    g.build_level(first=True)
    g.reset_player(snap_camera=True)
    g.set_quality(3)
    g.renderer.fog_enabled = True

    cx, cz = g3.MAZE_W * 0.5, g3.MAZE_D * 0.5
    eye = (cx - 560.0, 500.0, -360.0)
    target = (cx, 0.0, cz)

    # ---- 01 / 02: whole maze, both projections
    g3.VIEW_DISTANCE = 2600.0
    g.renderer.fog_enabled = False
    set_view(g, eye, target, "perspective")
    g.draw()
    save(g.screen, "01_perspective_full.png")
    label("01_perspective_full.png",
          "Perspective projection: far walls converge, the whole maze fits the frame")

    set_view(g, eye, target, "orthographic", ortho_height=780.0)
    g.draw()
    save(g.screen, "02_orthographic_full.png")
    label("02_orthographic_full.png",
          "Orthographic projection from the same camera: parallel edges stay parallel")
    g.renderer.fog_enabled = True

    # ---- 03 / 04: wireframe and depth buffer, overview
    set_view(g, eye, target, "perspective")
    g.renderer.fog_enabled = False
    g.renderer.wireframe = True
    g.draw()
    save(g.screen, "03_wireframe.png")
    label("03_wireframe.png", "Triangle wireframe: every wall is two textured triangles")
    g.renderer.wireframe = False
    g.renderer.fog_enabled = True

    g3.VIEW_DISTANCE = 330.0

    # ---- 06: gameplay close-up with worm, apple, bird and water
    apple = g.apples[0]
    ax, az = apple["pos"][0], apple["pos"][2]
    g.pos = np.array([ax - 6.0, g3.PLAYER_Y, az - 84.0])
    g.heading = 0.0
    g.trail = [g.pos.copy() for _ in range(12)]

    pool = g.water_pools[0]
    delta = np.array([ax - 4.0, 0.0, az + 68.0]) - pool["pos"]
    pool["pos"] = pool["pos"] + delta
    pool["model"] = e3.translation(*delta) @ pool["model"]

    bird = g.birds[0]
    bird["pos"] = np.array([ax - 34.0, g3.BIRD_FLY_HEIGHT + 12.0, az - 40.0])
    bird["heading"] = math.pi * 0.5
    bird["phase"] = -14.0          # catch the wings level for this still frame

    g3.VIEW_DISTANCE = 700.0
    set_view(g, (ax + 44.0, 152.0, az - 176.0), (ax + 6.0, 6.0, az + 26.0),
             "perspective")
    g.time = 2.0
    g.draw()
    save(g.screen, "06_gameplay.png")
    label("06_gameplay.png",
          "Gameplay frame: segmented worm, hovering apple, swooping bird and water hazard")

    # ---- 04: z-buffer visualisation from the same close camera
    g.renderer.depth_debug = True
    g.draw()
    save(g.screen, "04_depth_buffer.png")
    label("04_depth_buffer.png",
          "Z-buffer: bright yellow = near surfaces, blue = far surfaces")
    g.renderer.depth_debug = False

    # ---- 07 / 08: Extension 1 depth fog
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
    main()
