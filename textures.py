"""
textures.py -- 用 Pillow + NumPy 程序化生成的纹理 (不依赖外部图片文件)

所有纹理都是 256x256 (或 128x128) 的 float32 数组, 形状 (h, w, 3), 数值 0..1,
尺寸取 2 的幂, 这样光栅化时可以用位运算 `& (n-1)` 做快速 UV 循环寻址。
"""
# 【中文说明】
# 本文件负责「造图」：不下载任何贴图文件，全部用随机噪声 + 几何图案现场生成，
# 所以整个项目可以零资源文件提交。返回值的统一约定是：
#   shape = (高, 宽, 3)，dtype = float32，取值范围 0..1
# 宽高都取 2 的幂(256/128)，因为光栅化里用 `& (n-1)` 取模比 `% n` 快。

from __future__ import annotations          # 允许前向引用类型注解

import numpy as np                          # 噪声、fBm、逐像素着色全用 NumPy
from PIL import Image, ImageDraw, ImageFilter  # Pillow 只用来做缩放/模糊/画线等图像操作

SIZE = 256                                  # 默认纹理边长
_cache = {}                                 # 纹理缓存：同一个名字只生成一次


def _noise(rng, w, h, cells, blur=0):
    # 生成一张「格子噪声」：先在 cells×cells 的小网格上取随机值，
    # 再用双三次插值放大到 w×h，得到平滑起伏的灰度噪声
    small = rng.random((cells, cells))                  # 小尺寸随机矩阵(值域 0..1)
    img = Image.fromarray((small * 255).astype(np.uint8), "L")   # 转成 8 位灰度图
    img = img.resize((w, h), Image.BICUBIC)             # 双三次插值放大，天然平滑
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))    # 可选：再高斯模糊一次
    return np.asarray(img, dtype=np.float32) / 255.0    # 转回 0..1 的浮点数组


def _fbm(rng, w, h, base=4, octaves=4):
    # fBm = Fractal Brownian Motion(分形布朗运动)：多层噪声叠加
    # 每一层的格子数翻倍、振幅减半，叠加后既有大块起伏又有细节
    acc = np.zeros((h, w), dtype=np.float32)            # 累加器
    amp = 1.0                                           # 当前层振幅
    total = 0.0                                         # 振幅总和，用于最后归一化
    cells = base                                        # 当前层的格子数
    for _ in range(octaves):
        acc += _noise(rng, w, h, cells) * amp           # 叠加这一层
        total += amp                                    # 记录总振幅
        amp *= 0.5                                      # 下一层振幅减半
        cells *= 2                                      # 下一层格子数翻倍(频率更高)
    return acc / total                                  # 归一化回 0..1 附近


def stone(size=SIZE, seed=7):
    """石墙: 错缝砖块 + 灰浆 + 倒角高光 + 噪声."""
    # 逐像素用 NumPy 算，不用画图 API，因为砖块是规则的几何图案
    rng = np.random.default_rng(seed)                   # 固定随机种子 => 每次生成结果一致
    w = h = size
    # np.mgrid 生成两张坐标图：yy 是每行的 y 值，xx 是每列的 x 值
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    bh, bw, mortar = h / 4.0, w / 3.0, 3.0             # 砖高、砖宽、灰浆缝宽(像素)
    row = ((yy // bh).astype(np.int32)) % 4            # 每一点属于第几行砖(共 4 行循环)
    offset = (row % 2) * (bw * 0.5)                     # 奇数行偏移半块砖，形成「错缝」
    col = (((xx + offset) // bw).astype(np.int32)) % 3 # 属于第几列砖(共 3 列循环)

    shade = rng.random((4, 3)).astype(np.float32) * 0.22 + 0.34   # 每块砖随机的基础亮度
    base = shade[row, col]                             # 用行/列下标取出每块砖的亮度

    ly = yy % bh                                       # 在当前砖内的纵向位置
    lx = (xx + offset) % bw                            # 在当前砖内的横向位置
    is_mortar = (ly < mortar) | (lx < mortar)          # 贴近砖缝的位置就是灰浆

    # 倒角: 砖块上/左亮, 下/右暗
    # 这样砖块看起来有凸起的立体感，而不是一张平纸
    bevel = np.zeros_like(base)
    bevel += np.clip(1.0 - ly / 6.0, 0, 1) * 0.14      # 上边缘提亮
    bevel += np.clip(1.0 - lx / 6.0, 0, 1) * 0.10      # 左边缘提亮
    bevel -= np.clip(1.0 - (bh - ly) / 7.0, 0, 1) * 0.13   # 下边缘压暗
    bevel -= np.clip(1.0 - (bw - lx) / 7.0, 0, 1) * 0.10   # 右边缘压暗

    grain = (_fbm(rng, w, h, 16, 3) - 0.5) * 0.22      # 细颗粒噪声(减去 0.5 让均值回到 0)
    val = base + bevel + grain                          # 亮度 = 砖底色 + 倒角 + 颗粒
    val = np.where(is_mortar, 0.20 + grain * 0.12, val) # 灰浆单独用一个暗淡的亮度

    val = np.clip(val, 0.03, 1.0)                       # 防止出现纯黑或过曝
    col_rgb = np.stack([val * 1.00, val * 0.99, val * 0.95], axis=-1)   # 三通道：略偏暖的灰
    col_rgb *= np.array([1.02, 1.00, 0.94], dtype=np.float32)           # 再乘一次色调，偏黄一点
    return np.clip(col_rgb, 0, 1).astype(np.float32)


def grass(size=SIZE, seed=11):
    """草地: 多倍频噪声 + 草叶笔触."""
    rng = np.random.default_rng(seed)
    w = h = size
    f = _fbm(rng, w, h, 5, 4)                          # 大尺度噪声，作为草地的起伏
    base = np.stack([
        0.16 + f * 0.22,                               # 红通道很低 => 整体偏绿
        0.35 + f * 0.30,                               # 绿通道最高
        0.16 + f * 0.16,                               # 蓝通道低
    ], axis=-1).astype(np.float32)

    img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)                         # 用 Pillow 的画笔在上面加草叶
    for _ in range(900):                               # 画 900 根小草
        x = rng.integers(0, w)                          # 起点坐标
        y = rng.integers(0, h)
        length = rng.integers(3, 9)                     # 草叶长度
        lean = rng.integers(-3, 4)                      # 倾斜量(负=向左倒)
        c = int(rng.integers(30, 90))                   # 亮度
        # 从 (x,y) 画到 (x+lean, y-length)，即一条向上倾斜的短线
        draw.line((x, y, x + lean, y - length),
                  fill=(c // 2, c + 60, c // 2), width=1)   # 颜色偏绿，宽 1 像素
    arr = np.asarray(img, dtype=np.float32) / 255.0    # 转回浮点数组
    return arr.astype(np.float32)


def floor_tiles(size=SIZE, seed=23):
    """迷宫地面: 深浅交错的方形石砖 + 泥土噪声, 带青苔."""
    rng = np.random.default_rng(seed)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tile = 64.0                                        # 每块地砖边长(像素)
    ty = (yy // tile).astype(np.int32) % 4             # 行号(4 行循环)
    tx = (xx // tile).astype(np.int32) % 4             # 列号(4 列循环)
    checker = (ty + tx) % 2                            # 棋盘格：行列奇偶决定深浅

    shade = 0.42 + checker * 0.10                      # 深/浅两档基础亮度
    shade += rng.random((4, 4)).astype(np.float32)[ty, tx] * 0.10   # 每块砖再加一点随机差异
    grout = (((yy % tile) < 3) | ((xx % tile) < 3))    # 砖缝：离砖边界 3 像素内
    f = _fbm(rng, w, h, 8, 4)                          # 泥土噪声

    val = shade + (f - 0.5) * 0.22                     # 叠加噪声
    val = np.where(grout, val * 0.55, val)             # 砖缝压暗到 55%
    moss = np.clip((f - 0.55) * 2.2, 0, 1)             # 噪声高的地方长出青苔(系数放大增强对比)

    col = np.stack([val * 0.86, val * 0.88, val * 0.72], axis=-1)   # 偏土黄的基础色
    col[..., 0] *= (1 - moss * 0.45)                   # 青苔处：红通道降低
    col[..., 1] *= (1 + moss * 0.25)                   # 绿通道提高 => 变绿
    col[..., 2] *= (1 - moss * 0.35)                   # 蓝通道降低
    return np.clip(col, 0, 1).astype(np.float32)


def checker(size=128, seed=3):
    """终点黑白方格旗纹理."""
    # 这个函数不生成随机数，完全是几何图案(国际象棋棋盘)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cell = size // 5                                   # 每格边长
    pattern = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)   # 奇偶得到 0 或 1
    val = 0.96 * pattern + 0.06 * (1 - pattern)        # 亮格 0.96，暗格 0.06(不是纯黑)
    return np.stack([val, val, val], axis=-1).astype(np.float32)       # 三通道相同 => 灰色/黑白


def wood(size=128, seed=5):
    """木杆纹理: 纵向年轮条纹."""
    rng = np.random.default_rng(seed)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 正弦条纹 + 噪声扰动相位，看起来像不规则的年轮
    rings = 0.5 + 0.5 * np.sin(xx * 0.6 + _fbm(rng, w, h, 4, 2) * 6.0)
    val = 0.34 + rings * 0.16 + (_fbm(rng, w, h, 16, 3) - 0.5) * 0.08   # 基础亮度 + 年轮 + 细噪声
    col = np.stack([val * 1.35, val * 0.92, val * 0.55], axis=-1)      # 偏棕色：红高蓝低
    return np.clip(col, 0, 1).astype(np.float32)


def water(size=128, seed=31):
    """水池表面: 蓝色波纹."""
    rng = np.random.default_rng(seed)
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = cy = size * 0.5                               # 纹理中心
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)       # 到中心的距离(同心波纹)
    ripple = np.sin(r * 0.45 - 1.0) * 0.5 + 0.5        # 正弦波纹，映射到 0..1
    f = _fbm(rng, w, h, 8, 3)                          # 噪声打散，避免波纹太机械
    val = 0.30 + ripple * 0.30 + f * 0.20              # 叠加得到最终亮度
    col = np.stack([val * 0.35, val * 0.75, val * 0.95], axis=-1)      # 蓝色：红低蓝高
    return np.clip(col, 0, 1).astype(np.float32)


def wood_plank(size=128, seed=41):
    """旗帜用的布料纹理 (淡色, 会被颜色染色)."""
    # 名字叫 wood_plank，但实际用作布料；颜色接近白色，方便后面乘上旗帜颜色
    rng = np.random.default_rng(seed)
    w = h = size
    f = _noise(rng, w, h, 8)                           # 单层噪声，布料纤维感
    val = 0.78 + (f - 0.5) * 0.16                      # 亮底 + 轻微起伏
    return np.stack([val, val, val], axis=-1).astype(np.float32)


# 纹理名 -> 生成函数 的映射表，get() 用它按名字查函数
_BUILDERS = {
    "stone": stone,          # 墙面石砖
    "grass": grass,          # 草地
    "floor": floor_tiles,    # 迷宫地面
    "checker": checker,      # 终点方格
    "wood": wood,            # 木杆
    "water": water,          # 水面
    "cloth": wood_plank,     # 旗帜布料
}


def get(name):
    """按名字取纹理 (带缓存)."""
    # 第一次用到某个纹理时才生成，之后直接返回缓存，避免重复计算
    if name not in _cache:
        _cache[name] = _BUILDERS[name]()
    return _cache[name]
