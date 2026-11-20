"""
textures.py -- 用 Pillow + NumPy 程序化生成的纹理 (不依赖外部图片文件)

所有纹理都是 256x256 (或 128x128) 的 float32 数组, 形状 (h, w, 3), 数值 0..1,
尺寸取 2 的幂, 这样光栅化时可以用位运算 `& (n-1)` 做快速 UV 循环寻址。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 256
_cache = {}


def _noise(rng, w, h, cells, blur=0):
    small = rng.random((cells, cells))
    img = Image.fromarray((small * 255).astype(np.uint8), "L")
    img = img.resize((w, h), Image.BICUBIC)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img, dtype=np.float32) / 255.0


def _fbm(rng, w, h, base=4, octaves=4):
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    cells = base
    for _ in range(octaves):
        acc += _noise(rng, w, h, cells) * amp
        total += amp
        amp *= 0.5
        cells *= 2
    return acc / total


def stone(size=SIZE, seed=7):
    """石墙: 错缝砖块 + 灰浆 + 倒角高光 + 噪声."""
    rng = np.random.default_rng(seed)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    bh, bw, mortar = h / 4.0, w / 3.0, 3.0
    row = ((yy // bh).astype(np.int32)) % 4
    offset = (row % 2) * (bw * 0.5)
    col = (((xx + offset) // bw).astype(np.int32)) % 3

    shade = rng.random((4, 3)).astype(np.float32) * 0.22 + 0.34
    base = shade[row, col]

    ly = yy % bh
    lx = (xx + offset) % bw
    is_mortar = (ly < mortar) | (lx < mortar)

    # 倒角: 砖块上/左亮, 下/右暗
    bevel = np.zeros_like(base)
    bevel += np.clip(1.0 - ly / 6.0, 0, 1) * 0.14
    bevel += np.clip(1.0 - lx / 6.0, 0, 1) * 0.10
    bevel -= np.clip(1.0 - (bh - ly) / 7.0, 0, 1) * 0.13
    bevel -= np.clip(1.0 - (bw - lx) / 7.0, 0, 1) * 0.10

    grain = (_fbm(rng, w, h, 16, 3) - 0.5) * 0.22
    val = base + bevel + grain
    val = np.where(is_mortar, 0.20 + grain * 0.12, val)

    val = np.clip(val, 0.03, 1.0)
    col_rgb = np.stack([val * 1.00, val * 0.99, val * 0.95], axis=-1)
    col_rgb *= np.array([1.02, 1.00, 0.94], dtype=np.float32)
    return np.clip(col_rgb, 0, 1).astype(np.float32)


def grass(size=SIZE, seed=11):
    """草地: 多倍频噪声 + 草叶笔触."""
    rng = np.random.default_rng(seed)
    w = h = size
    f = _fbm(rng, w, h, 5, 4)
    base = np.stack([
        0.16 + f * 0.22,
        0.35 + f * 0.30,
        0.16 + f * 0.16,
    ], axis=-1).astype(np.float32)

    img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(900):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        length = rng.integers(3, 9)
        lean = rng.integers(-3, 4)
        c = int(rng.integers(30, 90))
        draw.line((x, y, x + lean, y - length),
                  fill=(c // 2, c + 60, c // 2), width=1)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.astype(np.float32)


def floor_tiles(size=SIZE, seed=23):
    """迷宫地面: 深浅交错的方形石砖 + 泥土噪声, 带青苔."""
    rng = np.random.default_rng(seed)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tile = 64.0
    ty = (yy // tile).astype(np.int32) % 4
    tx = (xx // tile).astype(np.int32) % 4
    checker = (ty + tx) % 2

    shade = 0.42 + checker * 0.10
    shade += rng.random((4, 4)).astype(np.float32)[ty, tx] * 0.10
    grout = (((yy % tile) < 3) | ((xx % tile) < 3))
    f = _fbm(rng, w, h, 8, 4)

    val = shade + (f - 0.5) * 0.22
    val = np.where(grout, val * 0.55, val)
    moss = np.clip((f - 0.55) * 2.2, 0, 1)

    col = np.stack([val * 0.86, val * 0.88, val * 0.72], axis=-1)
    col[..., 0] *= (1 - moss * 0.45)
    col[..., 1] *= (1 + moss * 0.25)
    col[..., 2] *= (1 - moss * 0.35)
    return np.clip(col, 0, 1).astype(np.float32)


def checker(size=128, seed=3):
    """终点黑白方格旗纹理."""
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cell = size // 5
    pattern = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
    val = 0.96 * pattern + 0.06 * (1 - pattern)
    return np.stack([val, val, val], axis=-1).astype(np.float32)


def wood(size=128, seed=5):
    """木杆纹理: 纵向年轮条纹."""
    rng = np.random.default_rng(seed)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rings = 0.5 + 0.5 * np.sin(xx * 0.6 + _fbm(rng, w, h, 4, 2) * 6.0)
    val = 0.34 + rings * 0.16 + (_fbm(rng, w, h, 16, 3) - 0.5) * 0.08
    col = np.stack([val * 1.35, val * 0.92, val * 0.55], axis=-1)
    return np.clip(col, 0, 1).astype(np.float32)


def water(size=128, seed=31):
    """水池表面: 蓝色波纹."""
    rng = np.random.default_rng(seed)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = cy = size * 0.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ripple = np.sin(r * 0.45 - 1.0) * 0.5 + 0.5
    f = _fbm(rng, w, h, 8, 3)
    val = 0.30 + ripple * 0.30 + f * 0.20
    col = np.stack([val * 0.35, val * 0.75, val * 0.95], axis=-1)
    return np.clip(col, 0, 1).astype(np.float32)


def wood_plank(size=128, seed=41):
    """旗帜用的布料纹理 (淡色, 会被颜色染色)."""
    rng = np.random.default_rng(seed)
    w = h = size
    f = _noise(rng, w, h, 8)
    val = 0.78 + (f - 0.5) * 0.16
    return np.stack([val, val, val], axis=-1).astype(np.float32)


_BUILDERS = {
    "stone": stone,
    "grass": grass,
    "floor": floor_tiles,
    "checker": checker,
    "wood": wood,
    "water": water,
    "cloth": wood_plank,
}


def get(name):
    """按名字取纹理 (带缓存)."""
    if name not in _cache:
        _cache[name] = _BUILDERS[name]()
    return _cache[name]
