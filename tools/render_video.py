"""
tools/render_video.py

Renders the 2-4 minute demonstration video without opening a window: the same
game code is stepped headlessly, every frame is piped to ffmpeg (bundled with
imageio-ffmpeg) and encoded as H.264.

Timeline (approx. 2 min 30 s at 30 fps):
    0-14    title + orbit around the whole maze (perspective)
    14-26   the same camera in orthographic projection
    26-44   chase camera, walking through the maze
    44-56   physical jump, side camera
    56-74   collecting the four apples (real pickup logic + particles)
    74-86   water hazard and bird hazard
    86-98   all apples collected / exit unlocked
    98-116  Extension 1: fog on/off comparison
    116-132 Extension 2: bilinear filter on/off comparison
    132-146 wireframe and z-buffer views
    146-152 end card

Run:
    python tools/render_video.py
"""

import math
import os
import random
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio_ffmpeg
import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine3d as e3
import game3d as g3

FPS = 30
OUT_W, OUT_H = 1000, 720
OUT = os.path.join(ROOT, "demo_video.mp4")


class Keys:
    def __init__(self, pressed=()):
        self.pressed = set(pressed)

    def __getitem__(self, key):
        return key in self.pressed


def set_view(g, eye, target, projection="perspective", ortho_height=None):
    g.projection = projection
    g.camera.projection = projection
    if ortho_height is not None:
        g.camera.ortho_height = ortho_height
    g.camera.look(eye, target)


def main():
    g = g3.Game(headless=True)
    g.rng = random.Random(7)
    g.build_level(first=True)
    g.reset_player(snap_camera=True)
    g.quality = -1
    g.set_quality(2)                     # HIGH = 625x450 internal buffer
    g.renderer.fog_enabled = True
    g.renderer.textures_enabled = True
    g.check_hazards = lambda: None       # hazard demo is scripted below

    screen = g.screen
    try:
        font = pygame.font.SysFont("arial", 42, bold=True)
        small = pygame.font.SysFont("arial", 24, bold=True)
    except Exception:
        # Some headless Windows setups cannot enumerate system fonts.
        font = pygame.font.Font(None, 46)
        small = pygame.font.Font(None, 28)
    cx, cz = g3.MAZE_W * 0.5, g3.MAZE_D * 0.5

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def subtitle(text, y=64):
        if not text:
            return
        img = small.render(text, True, (255, 250, 235))
        pad = 16
        box = pygame.Surface((img.get_width() + 2 * pad, img.get_height() + 14),
                             pygame.SRCALPHA)
        box.fill((10, 18, 26, 205))
        x = screen.get_width() // 2 - box.get_width() // 2
        screen.blit(box, (x, y))
        screen.blit(img, (x + pad, y + 7))

    def title_card(main, sub):
        overlay = pygame.Surface((screen.get_width(), screen.get_height()),
                                 pygame.SRCALPHA)
        overlay.fill((6, 10, 16, 205))
        screen.blit(overlay, (0, 0))
        t = font.render(main, True, (255, 240, 200))
        s = small.render(sub, True, (200, 220, 235))
        screen.blit(t, (screen.get_width() // 2 - t.get_width() // 2,
                        screen.get_height() // 2 - 70))
        screen.blit(s, (screen.get_width() // 2 - s.get_width() // 2,
                        screen.get_height() // 2 + 6))

    def emit(caption=""):
        subtitle(caption)
        frame = pygame.transform.smoothscale(screen, (OUT_W, OUT_H))
        proc.stdin.write(pygame.image.tostring(frame, "RGB"))

    total = 0

    def run(seconds, fn, caption=""):
        nonlocal total
        for _ in range(int(seconds * FPS)):
            fn(total / FPS)
            total += 1
            emit(caption)

    # ---------------------------------------------------------- 0-14 overview
    g3.VIEW_DISTANCE = 2600.0
    g.renderer.fog_enabled = True

    def orbit(t):
        ang = t * 0.10
        r = 760.0
        set_view(g, (cx + r * math.cos(ang), 520.0 + 40 * math.sin(t * 0.2),
                     cz + r * math.sin(ang)),
                 (cx, 0.0, cz), "perspective")
        g.time = t
        g.draw()

    def intro(t):
        orbit(t)
        title_card("Worm's Garden Adventure 3D",
                   "Software renderer - Lab 10 + 11, Part A + Part B")

    run(3.0, intro)
    run(11.0, orbit, "Part A: the 2D grid game rebuilt as a 3D world")

    # --------------------------------------------------- 14-26 orthographic
    eye = (cx - 560.0, 500.0, -360.0)

    def ortho(t):
        set_view(g, eye, (cx, 0, cz), "orthographic", 780.0)
        g.time = t
        g.draw()

    run(12.0, ortho, "Same camera, orthographic projection (V / P toggles)")

    # ---------------------------------------------------------- 26-44 chase
    g3.VIEW_DISTANCE = 380.0
    g.projection = "perspective"
    g.camera.projection = "perspective"
    g.cam_yaw = 0.0
    g.cam_pitch = -0.62
    g.cam_dist = 155.0
    g.time = 0.0

    def chase(t, i=[0]):
        g.update(1.0 / FPS, Keys((pygame.K_w,)))
        i[0] += 1
        if i[0] % 90 == 0:
            g.cam_yaw = g.cam_yaw + 0.4
        g.update_camera(1.0 / FPS)
        g.draw()

    run(18.0, chase, "Chase camera: camera-relative movement and wall sliding")

    # ------------------------------------------------------------ 44-56 jump
    g.pos = np.array([cx - 120.0, g3.PLAYER_Y, cz - 20.0])
    g.heading = math.pi * 0.5
    g.trail = [g.pos.copy() for _ in range(12)]
    g.vspeed = 0.0
    g.jump_h = 0.0
    jump_t = [0.0]
    set_view(g, (cx - 260.0, 110.0, cz + 40.0), (cx - 120.0, 40.0, cz - 20.0),
             "perspective")

    def jump(t):
        g.time = t
        # Re-trigger a jump every 3.2 seconds; physics integrates the rest.
        if g.jump_h <= 0.5 and g.vspeed == 0.0 and t - jump_t[0] > 2.4:
            g.vspeed = g3.JUMP_SPEED
            jump_t[0] = t
        g.update(1.0 / FPS, Keys())
        g.pos[0] = cx - 120.0
        g.pos[2] = cz - 20.0
        g.draw()

    run(12.0, jump, "Physical jump: vertical velocity + gravity, world-space height")

    # ---------------------------------------------------------- 56-74 apples
    g.jump_h = 0.0
    g.vspeed = 0.0
    g.renderer.fog_enabled = True

    def apples(t):
        idx = min(3, int((t - 56.0) // 4.0)) if t >= 56 else 0
        apple = g.apples[idx]
        if g.pos[0] < apple["pos"][0] - 200 or g.pos[0] > apple["pos"][0] + 200:
            pass
        g.pos = np.array([apple["pos"][0] - 40.0, g3.PLAYER_Y,
                          apple["pos"][2] + 40.0])
        g.heading = math.atan2(1.0, -1.0)
        g.trail = [g.pos.copy() for _ in range(12)]
        set_view(g, (g.pos[0] - 90.0, 105.0, g.pos[2] + 130.0),
                 (apple["pos"][0], 22.0, apple["pos"][2]), "perspective")
        g.time = t
        g.update(1.0 / FPS, Keys((pygame.K_w,)))
        g.check_pickups()
        g.draw()

    run(18.0, apples, "Apples: sphere distance test, score, particle burst")

    # ----------------------------------------------------------- 74-86 hazards
    pool = g.water_pools[0]
    bird = g.birds[0]

    def hazard(t):
        # First half: stand beside the water pool. Second half: bird close-up.
        g.time = t
        if t < 80.0:
            g.pos = np.array([pool["pos"][0] + 80.0, g3.PLAYER_Y,
                              pool["pos"][2] + 30.0])
            g.heading = math.pi
            g.trail = [g.pos.copy() for _ in range(12)]
            set_view(g, (pool["pos"][0] + 110.0, 120.0, pool["pos"][2] + 150.0),
                     (pool["pos"][0], 8.0, pool["pos"][2]), "perspective")
        else:
            bird["pos"] = np.array([cx, g3.BIRD_FLY_HEIGHT + 6, cz])
            bird["heading"] = 0.0
            set_view(g, (cx + 30.0, 70.0, cz - 120.0), (cx, 24.0, cz), "perspective")
        g.update(1.0 / FPS, Keys())
        g.draw()

    run(12.0, hazard, "Hazards: animated water surface, birds flapping in 3D")

    # ----------------------------------------------------- 86-98 level state
    g.pos = np.array([g3.MAZE_W - 20.0, g3.PLAYER_Y, cz])
    g.trail = [g.pos.copy() for _ in range(12)]
    g.banner = "All apples collected - exit unlocked!"
    g.banner_t = 2.0

    def finale(t):
        g.time = t
        set_view(g, (g3.MAZE_W - 180.0, 140.0, cz - 150.0),
                 (g3.MAZE_W - 20.0, 12.0, cz), "perspective")
        g.banner_t = 2.0
        g.update(1.0 / FPS, Keys())
        g.draw()

    run(12.0, finale, "Level logic: apples unlock the finish flag")

    # ------------------------------------------------------------ 98-116 fog
    g3.VIEW_DISTANCE = 1500.0

    def fog(t):
        set_view(g, eye, (cx, 0, cz), "perspective")
        g.renderer.fog_enabled = int(t // 4.5) % 2 == 0
        g.time = t
        g.draw()

    run(18.0, fog, "Extension 1: depth fog off / on (key F)")

    # ------------------------------------------------------- 116-132 filtering
    g3.VIEW_DISTANCE = 330.0
    g.pos = np.array([cx - 40.0, g3.PLAYER_Y, cz])
    g.trail = [g.pos.copy() for _ in range(12)]

    def filt(t):
        set_view(g, (cx - 150.0, 34.0, cz - 120.0), (cx, 4.0, cz + 30.0),
                 "perspective")
        g.renderer.filter_mode = "bilinear" if int(t // 4.0) % 2 else "nearest"
        g.time = t
        g.draw()

    run(16.0, filt, "Extension 2: nearest vs bilinear texture sampling (key B)")

    # -------------------------------------------------------- 132-146 debug
    g3.VIEW_DISTANCE = 2200.0

    def debug(t):
        set_view(g, eye, (cx, 0, cz), "perspective")
        k = int(t // 3.5) % 3
        g.renderer.wireframe = (k == 1)
        g.renderer.depth_debug = (k == 2)
        g.renderer.fog_enabled = True
        g.time = t
        g.draw()

    run(14.0, debug, "Debug views: wireframe and z-buffer (F1 / F3)")
    g.renderer.wireframe = False
    g.renderer.depth_debug = False

    # ----------------------------------------------------------- 146-152 end
    g3.VIEW_DISTANCE = 2600.0

    def endcard(t):
        set_view(g, (cx - 300.0, 620.0, cz - 600.0), (cx, 0, cz), "perspective")
        g.time = t
        g.draw()
        title_card("Worm's Garden Adventure 3D",
                   "github.com/2923565578/Lab-1011-3-Game-Development")

    run(6.0, endcard)

    proc.stdin.close()
    proc.wait()
    pygame.quit()
    print("video written:", OUT)


if __name__ == "__main__":
    main()
