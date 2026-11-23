"""
tools/render_video.py

Renders the 2-4 minute demonstration video without opening a window: the same
game code is stepped headlessly, every frame is piped to ffmpeg (bundled with
imageio-ffmpeg) and encoded as H.264.

【中文说明】
演示视频生成脚本。不打开窗口，而是用「同一套游戏代码」逐帧推进，
每一帧通过管道直接喂给 ffmpeg(由 imageio-ffmpeg 自带的那个可执行文件)，
最终编码成 H.264 的 mp4。整段约 2 分 30 秒，30 fps。
这样做的好处：视频是「真跑出来的」，不是后期剪辑拼的，能证明代码真的能运行。

Timeline (approx. 2 min 30 s at 30 fps):
    0-14    title + orbit around the whole maze (perspective)   片头 + 环绕整张迷宫(透视)
    14-26   the same camera in orthographic projection           同机位换成正交投影
    26-44   chase camera, walking through the maze               跟随视角在迷宫里走动
    44-56   physical jump, side camera                           侧面机位看物理跳跃
    56-74   collecting the four apples (real pickup logic + particles)  吃四个苹果(真实拾取逻辑+粒子)
    74-86   water hazard and bird hazard                         水潭危险与小鸟危险
    86-98   all apples collected / exit unlocked                 苹果集齐 / 出口解锁
    98-116  Extension 1: fog on/off comparison                   扩展1：雾开/关对比
    116-132 Extension 2: bilinear filter on/off comparison       扩展2：双线性过滤开/关对比
    132-146 wireframe and z-buffer views                         线框视图与深度缓冲视图
    146-152 end card                                             片尾字幕

Run:
    python tools/render_video.py
"""

import math
import os
import random
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")      # 无头模式
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio_ffmpeg      # 提供 get_ffmpeg_exe()，拿到内置的 ffmpeg 路径
import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine3d as e3
import game3d as g3

FPS = 30                       # 视频帧率
OUT_W, OUT_H = 1000, 720       # 输出分辨率(比内部渲染分辨率大，靠放大得到)
OUT = os.path.join(ROOT, "demo_video.mp4")


class Keys:
    # 用一个假对象模拟 pygame.key.get_pressed() 的返回值
    # 因为我们要「脚本化」按键(比如一直按住 W)，而真实键盘在无头环境里读不到
    def __init__(self, pressed=()):
        self.pressed = set(pressed)

    def __getitem__(self, key):
        # 支持 keys[pygame.K_w] 这种下标访问，返回 True/False
        return key in self.pressed


def set_view(g, eye, target, projection="perspective", ortho_height=None):
    # 统一设置摄像机的工具函数：投影方式 + 位置/注视点(+ 可选正交半高)
    g.projection = projection
    g.camera.projection = projection
    if ortho_height is not None:
        g.camera.ortho_height = ortho_height
    g.camera.look(eye, target)


def main():
    g = g3.Game(headless=True)
    g.rng = random.Random(7)             # 固定种子：每次生成的视频内容完全一致
    g.build_level(first=True)
    g.reset_player(snap_camera=True)
    g.quality = -1
    g.set_quality(2)                     # HIGH = 625x450 internal buffer  用 HIGH 档内部缓冲
    g.renderer.fog_enabled = True
    g.renderer.textures_enabled = True
    g.check_hazards = lambda: None       # hazard demo is scripted below
    # 把「危险检测」直接替换成空函数：因为这段视频里危险镜头是脚本摆拍的，
    # 不能让真实的碰撞检测把主角撞死导致画面中断

    screen = g.screen
    try:
        font = pygame.font.SysFont("arial", 42, bold=True)      # 大标题字
        small = pygame.font.SysFont("arial", 24, bold=True)     # 小字幕
    except Exception:
        # Some headless Windows setups cannot enumerate system fonts.
        # 某些无头 Windows 环境列不出系统字体，退回 pygame 内置字体(只是不好看，能跑)
        font = pygame.font.Font(None, 46)
        small = pygame.font.Font(None, 28)
    cx, cz = g3.MAZE_W * 0.5, g3.MAZE_D * 0.5    # 迷宫中心，很多镜头围绕它取景

    # 启动 ffmpeg 子进程：从标准输入读原始 RGB 像素流，实时编码成 mp4
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",     # -i - 表示从 stdin 读
           "-an",                       # 不要音频
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",  # H.264，CRF 20 质量不错体积可控
           "-pix_fmt", "yuv420p", OUT]  # yuv420p 保证各种播放器/网页都能播
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def subtitle(text, y=64):
        # 在画面顶部画一条半透明字幕条
        if not text:
            return
        img = small.render(text, True, (255, 250, 235))
        pad = 16
        box = pygame.Surface((img.get_width() + 2 * pad, img.get_height() + 14),
                             pygame.SRCALPHA)
        box.fill((10, 18, 26, 205))
        x = screen.get_width() // 2 - box.get_width() // 2      # 水平居中
        screen.blit(box, (x, y))
        screen.blit(img, (x + pad, y + 7))

    def title_card(main, sub):
        # 全屏压暗 + 居中的大标题/副标题(片头和片尾用)
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
        # 把当前画面加字幕、缩放、转成字节流写给 ffmpeg
        subtitle(caption)
        frame = pygame.transform.smoothscale(screen, (OUT_W, OUT_H))
        proc.stdin.write(pygame.image.tostring(frame, "RGB"))

    total = 0        # 已经输出的总帧数(用于计算时间轴)

    def run(seconds, fn, caption=""):
        # 播放一段 seconds 秒的时长：每帧调用 fn(t) 生成画面，再 emit 出去
        nonlocal total
        for _ in range(int(seconds * FPS)):
            fn(total / FPS)          # 传入「从视频开始算起的绝对秒数」，脚本据此切换动作
            total += 1
            emit(caption)

    # ---------------------------------------------------------- 0-14 overview
    g3.VIEW_DISTANCE = 2600.0        # 全景需要看很远，临时放大绘制距离
    g.renderer.fog_enabled = True

    def orbit(t):
        # 摄像机绕迷宫中心做圆周运动：角度随时间线性增加，半径 760
        ang = t * 0.10
        r = 760.0
        set_view(g, (cx + r * math.cos(ang), 520.0 + 40 * math.sin(t * 0.2),
                     cz + r * math.sin(ang)),
                 (cx, 0.0, cz), "perspective")
        g.time = t               # 同步游戏时间，让水面/旗帜动画跟着走
        g.draw()

    def intro(t):
        # 片头：环绕镜头 + 中间压一层标题卡
        orbit(t)
        title_card("Worm's Garden Adventure 3D",
                   "Software renderer - Lab 10 + 11, Part A + Part B")

    run(3.0, intro)                                                    # 前 3 秒纯标题
    run(11.0, orbit, "Part A: the 2D grid game rebuilt as a 3D world")  # 再 11 秒环绕

    # --------------------------------------------------- 14-26 orthographic
    eye = (cx - 560.0, 500.0, -360.0)     # 后面多段镜头复用的高空机位

    def ortho(t):
        # 同一机位换成正交投影：平行线保持平行，远处不收缩
        set_view(g, eye, (cx, 0, cz), "orthographic", 780.0)
        g.time = t
        g.draw()

    run(12.0, ortho, "Same camera, orthographic projection (V / P toggles)")

    # ---------------------------------------------------------- 26-44 chase
    g3.VIEW_DISTANCE = 380.0          # 追尾视角不需要看太远，缩小视距省性能
    g.projection = "perspective"
    g.camera.projection = "perspective"
    g.cam_yaw = 0.0
    g.cam_pitch = -0.62               # 稍微俯视的追尾镜头
    g.cam_dist = 155.0
    g.time = 0.0

    def chase(t, i=[0]):
        # 一直按住 W 前进：这是真实的移动逻辑(含碰撞与贴墙滑动)
        g.update(1.0 / FPS, Keys((pygame.K_w,)))
        i[0] += 1
        if i[0] % 90 == 0:
            g.cam_yaw = g.cam_yaw + 0.4     # 每 3 秒转一点镜头角度，避免画面太单调
        g.update_camera(1.0 / FPS)
        g.draw()

    run(18.0, chase, "Chase camera: camera-relative movement and wall sliding")

    # ------------------------------------------------------------ 44-56 jump
    # 把主角摆到固定位置，用侧面机位专门展示跳跃
    g.pos = np.array([cx - 120.0, g3.PLAYER_Y, cz - 20.0])
    g.heading = math.pi * 0.5
    g.trail = [g.pos.copy() for _ in range(12)]
    g.vspeed = 0.0
    g.jump_h = 0.0
    jump_t = [0.0]                   # 用列表包一层，闭包里才能修改
    set_view(g, (cx - 260.0, 110.0, cz + 40.0), (cx - 120.0, 40.0, cz - 20.0),
             "perspective")

    def jump(t):
        g.time = t
        # Re-trigger a jump every 3.2 seconds; physics integrates the rest.
        # 每 2.4 秒重新起跳一次，中间的上升/下落完全交给物理积分算
        if g.jump_h <= 0.5 and g.vspeed == 0.0 and t - jump_t[0] > 2.4:
            g.vspeed = g3.JUMP_SPEED
            jump_t[0] = t
        g.update(1.0 / FPS, Keys())
        g.pos[0] = cx - 120.0        # 锁住水平位置，只让竖直方向动(方便观察抛物线)
        g.pos[2] = cz - 20.0
        g.draw()

    run(12.0, jump, "Physical jump: vertical velocity + gravity, world-space height")

    # ---------------------------------------------------------- 56-74 apples
    g.jump_h = 0.0
    g.vspeed = 0.0
    g.renderer.fog_enabled = True

    def apples(t):
        # 每 4 秒换一个苹果：把主角瞬移到苹果附近，然后调用真实的拾取检测
        idx = min(3, int((t - 56.0) // 4.0)) if t >= 56 else 0
        apple = g.apples[idx]
        if g.pos[0] < apple["pos"][0] - 200 or g.pos[0] > apple["pos"][0] + 200:
            pass                     # (这行是遗留的空分支，保留原样不影响运行)
        g.pos = np.array([apple["pos"][0] - 40.0, g3.PLAYER_Y,
                          apple["pos"][2] + 40.0])
        g.heading = math.atan2(1.0, -1.0)       # 朝向苹果
        g.trail = [g.pos.copy() for _ in range(12)]
        set_view(g, (g.pos[0] - 90.0, 105.0, g.pos[2] + 130.0),
                 (apple["pos"][0], 22.0, apple["pos"][2]), "perspective")
        g.time = t
        g.update(1.0 / FPS, Keys((pygame.K_w,)))
        g.check_pickups()            # 关键：调用游戏里真实的拾取判定(球距离测试)
        g.draw()

    run(18.0, apples, "Apples: sphere distance test, score, particle burst")

    # ----------------------------------------------------------- 74-86 hazards
    pool = g.water_pools[0]
    bird = g.birds[0]

    def hazard(t):
        # First half: stand beside the water pool. Second half: bird close-up.
        # 前半段站在水潭边展示水面动画；后半段切到小鸟特写展示扇翅膀
        g.time = t
        if t < 80.0:
            g.pos = np.array([pool["pos"][0] + 80.0, g3.PLAYER_Y,
                              pool["pos"][2] + 30.0])
            g.heading = math.pi
            g.trail = [g.pos.copy() for _ in range(12)]
            set_view(g, (pool["pos"][0] + 110.0, 120.0, pool["pos"][2] + 150.0),
                     (pool["pos"][0], 8.0, pool["pos"][2]), "perspective")
        else:
            bird["pos"] = np.array([cx, g3.BIRD_FLY_HEIGHT + 6, cz])   # 把鸟挪到镜头正前方
            bird["heading"] = 0.0
            set_view(g, (cx + 30.0, 70.0, cz - 120.0), (cx, 24.0, cz), "perspective")
        g.update(1.0 / FPS, Keys())
        g.draw()

    run(12.0, hazard, "Hazards: animated water surface, birds flapping in 3D")

    # ----------------------------------------------------- 86-98 level state
    # 把主角放到终点附近，横幅显示「苹果已集齐、出口解锁」
    g.pos = np.array([g3.MAZE_W - 20.0, g3.PLAYER_Y, cz])
    g.trail = [g.pos.copy() for _ in range(12)]
    g.banner = "All apples collected - exit unlocked!"
    g.banner_t = 2.0

    def finale(t):
        g.time = t
        set_view(g, (g3.MAZE_W - 180.0, 140.0, cz - 150.0),
                 (g3.MAZE_W - 20.0, 12.0, cz), "perspective")
        g.banner_t = 2.0             # 每帧把横幅时间重置，让它一直显示
        g.update(1.0 / FPS, Keys())
        g.draw()

    run(12.0, finale, "Level logic: apples unlock the finish flag")

    # ------------------------------------------------------------ 98-116 fog
    g3.VIEW_DISTANCE = 1500.0

    def fog(t):
        # 每 4.5 秒切换一次雾的开关，形成「开-关-开」的对比节奏
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
        # 每 4 秒切换一次过滤方式，让观众看清地面纹理的差异
        set_view(g, (cx - 150.0, 34.0, cz - 120.0), (cx, 4.0, cz + 30.0),
                 "perspective")
        g.renderer.filter_mode = "bilinear" if int(t // 4.0) % 2 else "nearest"
        g.time = t
        g.draw()

    run(16.0, filt, "Extension 2: nearest vs bilinear texture sampling (key B)")

    # -------------------------------------------------------- 132-146 debug
    g3.VIEW_DISTANCE = 2200.0

    def debug(t):
        # 三种调试视图循环：正常 -> 线框 -> 深度缓冲
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
        # 片尾：远景 + 标题卡 + 仓库地址
        set_view(g, (cx - 300.0, 620.0, cz - 600.0), (cx, 0, cz), "perspective")
        g.time = t
        g.draw()
        title_card("Worm's Garden Adventure 3D",
                   "github.com/2923565578/Lab-1011-3-Game-Development")

    run(6.0, endcard)

    proc.stdin.close()      # 关闭管道，ffmpeg 收到 EOF 后完成封装
    proc.wait()             # 等待编码进程退出
    pygame.quit()
    print("video written:", OUT)


if __name__ == "__main__":
    main()      # 直接运行本文件即生成演示视频
