"""
tools/benchmark.py

Measures the renderer for the four quality presets required by Section 16.2:
internal framebuffer size, triangles submitted, drawn pixels, average frame
time and the equivalent FPS. Also measures the cost of the two Part B
extensions (fog and bilinear filtering) at HIGH quality.

Outputs:
    docs/performance.csv
    docs/performance.json
    docs/performance.md   (ready to paste into the report)

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

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine3d as e3
import game3d as g3

DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

FRAMES = 45


def measure(g, frames=FRAMES):
    times = []
    for i in range(frames):
        g.time += 1.0 / 60.0
        t0 = time.perf_counter()
        g.draw()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times[5:]), statistics.median(times[5:])


def main():
    g = g3.Game(headless=True)
    g.rng = random.Random(2026)
    g.build_level(first=True)
    g.reset_player(snap_camera=True)
    g.time = 1.0

    # Representative gameplay camera, settled before measuring.
    g.cam_yaw = 0.35
    g.cam_pitch = -0.72
    g.cam_dist = 190.0
    for _ in range(30):
        g.update_camera(0.05)

    rows = []
    print(f"{'Preset':<10}{'Buffer':>12}{'Tris':>8}{'Pixels':>10}"
          f"{'avg ms':>10}{'median':>9}{'FPS':>7}")
    for i, (name, scale) in enumerate(g3.QUALITY_LEVELS):
        g.quality = -1
        g.set_quality(i)
        g.renderer.fog_enabled = True
        g.renderer.filter_mode = "nearest"
        avg, med = measure(g)
        r = g.renderer
        rows.append({
            "preset": name, "scale": scale,
            "width": r.w, "height": r.h,
            "triangles": int(r.stats["triangles"]),
            "pixels": int(r.stats["pixels"]),
            "avg_ms": round(avg, 2),
            "median_ms": round(med, 2),
            "fps": round(1000.0 / avg, 1),
        })
        print(f"{name:<10}{r.w:>5}x{r.h:<6}{r.stats['triangles']:>8}"
              f"{r.stats['pixels']:>10}{avg:>10.2f}{med:>9.2f}{1000.0/avg:>7.1f}")

    # Extension overhead at HIGH quality.
    g.quality = -1
    g.set_quality(2)
    ext = []
    for fog, filt, tag in ((True, "nearest", "fog on, nearest"),
                           (False, "nearest", "fog off, nearest"),
                           (True, "bilinear", "fog on, bilinear"),
                           (False, "bilinear", "fog off, bilinear")):
        g.renderer.fog_enabled = fog
        g.renderer.filter_mode = filt
        avg, med = measure(g, 30)
        ext.append({"setting": tag, "avg_ms": round(avg, 2), "fps": round(1000.0 / avg, 1)})
        print(f"  {tag:<20}{avg:>8.2f} ms")

    with open(os.path.join(DOCS, "performance.json"), "w", encoding="utf-8") as f:
        json.dump({"quality": rows, "extensions": ext}, f, indent=2)

    with open(os.path.join(DOCS, "performance.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("preset,scale,width,height,triangles,pixels,avg_ms,median_ms,fps\n")
        for r in rows:
            f.write(f"{r['preset']},{r['scale']},{r['width']},{r['height']},"
                    f"{r['triangles']},{r['pixels']},{r['avg_ms']},{r['median_ms']},{r['fps']}\n")

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
    print("written docs/performance.csv, .json, .md")
    pygame.quit()


if __name__ == "__main__":
    main()
