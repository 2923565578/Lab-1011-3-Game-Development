"""
tools/benchmark.py

Measures the renderer for the four quality presets required by Section 16.2:
internal framebuffer size, triangles submitted, drawn pixels, average frame
time and the equivalent FPS. Also measures the cost of the two Part B
extensions (fog and bilinear filtering) at HIGH quality.

【中文说明】
性能基准测试脚本。手册 16.2 节要求给出四个画质档位的性能数据：
内部帧缓冲尺寸、提交的三角形数、实际着色的像素数、平均帧时间、等效帧率。
另外还要单独测量 Part B 两个扩展（深度雾、双线性过滤）在 HIGH 档下的额外开销。

Outputs:
    docs/performance.csv     供表格软件打开
    docs/performance.json    结构化数据，方便脚本再处理
    docs/performance.md      直接可以粘进报告里的 Markdown 表格

Run:
    python tools/benchmark.py
"""

import json
import math
import os
import random
import statistics
import sys
import time

# 用 dummy 视频驱动：不需要真实窗口就能创建 Pygame 环境(无头运行)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")   # 隐藏 pygame 启动时的英文欢迎语

import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))   # tools/ 目录
ROOT = os.path.dirname(HERE)                        # 项目根目录
sys.path.insert(0, ROOT)                            # 让 import engine3d 能找到模块

import engine3d as e3
import game3d as g3

DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)    # 输出目录不存在就创建

FRAMES = 45                         # 每次测量跑 45 帧


def measure(g, frames=FRAMES):
    # 测量平均帧时间：连续画 frames 帧，用高精度计时器 per_calc 掐表
    times = []
    for i in range(frames):
        g.time += 1.0 / 60.0        # 推进游戏时间，让动画/水面波动处于不同相位
        t0 = time.perf_counter()
        g.draw()                    # 只测绘制耗时(不含逻辑更新)，因为瓶颈在渲染
        times.append((time.perf_counter() - t0) * 1000.0)   # 转成毫秒
    # 丢掉前 5 帧(预热帧，含首次缓存建立/CPU 频率爬升)，再取平均和中位数
    return statistics.mean(times[5:]), statistics.median(times[5:])


def main():
    g = g3.Game(headless=True)      # 无窗口建游戏实例
    g.rng = random.Random(2026)     # 固定随机种子 => 每次跑出来的迷宫完全一样，数据可比
    g.build_level(first=True)
    g.reset_player(snap_camera=True)
    g.time = 1.0

    # Representative gameplay camera, settled before measuring.
    # 用一个「有代表性的游戏视角」，并先跑 30 次摄像机更新让它稳定下来，
    # 否则测量期间镜头还在平滑移动，测到的就不是固定视角的开销。
    g.cam_yaw = 0.35
    g.cam_pitch = -0.72
    g.cam_dist = 190.0
    for _ in range(30):
        g.update_camera(0.05)

    rows = []
    # 打印表头（左对齐/右对齐用格式化说明符控制）
    print(f"{'Preset':<10}{'Buffer':>12}{'Tris':>8}{'Pixels':>10}"
          f"{'avg ms':>10}{'median':>9}{'FPS':>7}")
    for i, (name, scale) in enumerate(g3.QUALITY_LEVELS):
        g.quality = -1                  # 先把档位设成非法值，保证 set_quality 一定触发重建
        g.set_quality(i)                # 切到第 i 档（会按像素预算重建渲染器）
        g.renderer.fog_enabled = True        # 基准测试固定：开雾、最近邻过滤
        g.renderer.filter_mode = "nearest"   # 这样四档之间只有分辨率一个变量
        avg, med = measure(g)
        r = g.renderer
        rows.append({
            "preset": name, "scale": scale,
            "width": r.w, "height": r.h,                      # 内部帧缓冲尺寸
            "triangles": int(r.stats["triangles"]),           # 本帧提交的三角形数
            "pixels": int(r.stats["pixels"]),                 # 实际通过深度测试的像素数
            "avg_ms": round(avg, 2),
            "median_ms": round(med, 2),
            "fps": round(1000.0 / avg, 1),                    # 由平均帧时间反推的等效帧率
        })
        print(f"{name:<10}{r.w:>5}x{r.h:<6}{r.stats['triangles']:>8}"
              f"{r.stats['pixels']:>10}{avg:>10.2f}{med:>9.2f}{1000.0/avg:>7.1f}")

    # Extension overhead at HIGH quality.
    # 扩展开销测试：固定 HIGH 档(索引 2)，把「雾开/关」×「最近邻/双线性」四种组合各测一遍，
    # 用差值就能算出每个扩展单独增加了多少毫秒。
    g.quality = -1
    g.set_quality(2)
    ext = []
    for fog, filt, tag in ((True, "nearest", "fog on, nearest"),
                           (False, "nearest", "fog off, nearest"),
                           (True, "bilinear", "fog on, bilinear"),
                           (False, "bilinear", "fog off, bilinear")):
        g.renderer.fog_enabled = fog
        g.renderer.filter_mode = filt
        avg, med = measure(g, 30)       # 每组只跑 30 帧，省时间
        ext.append({"setting": tag, "avg_ms": round(avg, 2), "fps": round(1000.0 / avg, 1)})
        print(f"  {tag:<20}{avg:>8.2f} ms")

    # 输出 JSON：把四档性能 + 扩展开销一起存成结构化数据
    with open(os.path.join(DOCS, "performance.json"), "w", encoding="utf-8") as f:
        json.dump({"quality": rows, "extensions": ext}, f, indent=2)

    # 输出 CSV：一行一档，字段用逗号分隔，方便 Excel 打开
    with open(os.path.join(DOCS, "performance.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("preset,scale,width,height,triangles,pixels,avg_ms,median_ms,fps\n")
        for r in rows:
            f.write(f"{r['preset']},{r['scale']},{r['width']},{r['height']},"
                    f"{r['triangles']},{r['pixels']},{r['avg_ms']},{r['median_ms']},{r['fps']}\n")

    # 输出 Markdown 表格：直接粘进报告 Section 9 的性能表
    md = ["| Preset | Framebuffer | Triangles | Shaded pixels | Avg frame ms | FPS |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['preset']} | {r['width']}x{r['height']} | {r['triangles']} | "
                  f"{r['pixels']} | {r['avg_ms']} | {r['fps']} |")
    md.append("")
    md.append("| Extension setting (HIGH) | Avg frame ms | FPS |")
    md.append("|---|---|---|")
    for e in ext:
        md.append(f"| {e['setting']} | {e['avg_ms']} | {e['fps']} |")
    with open(os.path.join(DOCS, "performance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("written docs/performance.csv, .json, .md")   # 提示三个文件已写入
    pygame.quit()


if __name__ == "__main__":
    main()      # 直接运行本文件即执行基准测试
