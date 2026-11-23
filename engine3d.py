"""
engine3d.py -- 用 NumPy 和 Pygame 从零手写的软件 3D 渲染器。

engine3d.py -- a software 3D renderer written from scratch with NumPy and Pygame.

没有 OpenGL，没有游戏引擎。图形流水线的每一个必需阶段都在本文件里实现：

  1. 3D 坐标与几何        : Mat4 矩阵、顶点列表、三角网格
  2. 3D 变换              : 平移 / 旋转 / 缩放矩阵
  3. 摄像机与投影          : look-at 视图矩阵、透视投影 + 正交投影
  4. 裁剪                 : 保守的视锥剔除 + 近平面 Sutherland-Hodgman 多边形裁剪
  5. 背面剔除             : 屏幕空间有向面积判定
  6. 三角形光栅化          : 包围盒 + 重心坐标边函数
  7. 深度缓冲 (z-buffer)   : 透视用 1/w，正交用视图空间深度
  8. 光照                 : 环境光 + 平行光 + 点光源，平面着色与 Gouraud(逐顶点)着色
  9. 纹理映射             : 透视正确的 UV 插值，最近邻或双线性采样(Part B 扩展 2)
 10. 3D 粒子             : 带深度测试的加法混合公告板
 11. 渲染优化             : 视锥剔除、距离剔除、按单元格批处理、内部低分辨率 + 平滑放大
 12. 调试视图             : 线框、深度缓冲可视化
"""

from __future__ import annotations          # 允许在类型注解里使用前向引用（3.7+ 特性）

import math                                 # 三角函数、floor/ceil 等标量数学

import numpy as np                          # 全部向量化运算靠 NumPy 完成，这是性能关键
import pygame                               # 只用来创建窗口并把最终帧缓冲贴到屏幕上

EPS = 1e-9                                  # 全局极小量：防止除零和法线归一化时溢出


# ============================================================
# 1. 向量与 4x4 齐次矩阵数学
# ============================================================

def normalize(v):
    # 把向量归一化为单位长度（长度为 0 时原样返回，避免除零）
    v = np.asarray(v, dtype=np.float64)     # 转成 float64 数组，保证精度
    n = np.linalg.norm(v)                   # 求欧几里得长度 |v|
    return v / n if n > EPS else v.copy()   # 长度过小就判为零向量，直接返回副本


def mat_identity():
    # 返回 4x4 单位矩阵，是所有变换矩阵的起点
    return np.eye(4, dtype=np.float64)


def translation(x, y, z):
    # 构造平移矩阵：第 4 列放平移量，这样 M @ (x,y,z,1) 就等于逐分量加上偏移
    m = np.eye(4)
    m[0, 3] = x                             # x 方向平移
    m[1, 3] = y                             # y 方向平移
    m[2, 3] = z                             # z 方向平移
    return m


def scaling(x, y=None, z=None):
    # 构造缩放矩阵；只给一个参数时三个轴等比缩放
    if y is None:
        y = x
    if z is None:
        z = x
    m = np.eye(4)
    m[0, 0] = x                             # x 轴缩放系数
    m[1, 1] = y                             # y 轴缩放系数
    m[2, 2] = z                             # z 轴缩放系数
    return m


def rotation_x(a):
    # 绕 X 轴旋转 a 弧度（右手系，逆时针为正）
    c, s = math.cos(a), math.sin(a)         # 预先算 cos/sin，避免重复计算
    m = np.eye(4)
    m[1, 1], m[1, 2] = c, -s                # 第 2 行：y' =  c*y - s*z
    m[2, 1], m[2, 2] = s, c                 # 第 3 行：z' =  s*y + c*z
    return m


def rotation_y(a):
    # 绕 Y 轴旋转 a 弧度
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2] = c, s                 # x' = c*x + s*z
    m[2, 0], m[2, 2] = -s, c                # z' = -s*x + c*z
    return m


def rotation_z(a):
    # 绕 Z 轴旋转 a 弧度
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1] = c, -s                # x' = c*x - s*y
    m[1, 0], m[1, 1] = s, c                 # y' = s*x + c*y
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """构造右手系视图矩阵 (与 gluLookAt 相同)."""
    # 说明：把世界坐标系变换到「摄像机坐标系」，摄像机看 -Z 方向
    eye = np.asarray(eye, dtype=np.float64)         # 摄像机位置
    target = np.asarray(target, dtype=np.float64)   # 注视点位置
    up = np.asarray(up, dtype=np.float64)           # 世界上方向（一般取 +Y）

    f = normalize(target - eye)              # 前方向 f = 视线方向(单位向量)
    s = normalize(np.cross(f, up))           # 右方向 s = f × up（叉积得垂直向量）
    u = np.cross(s, f)                       # 上方向 u = s × f（三轴正交）

    m = np.eye(4)
    m[0, :3] = s                             # 视图矩阵第 1 行是右轴
    m[1, :3] = u                             # 第 2 行是上轴
    m[2, :3] = -f                            # 第 3 行是 -前轴（因为看 -Z）
    m[0, 3] = -float(np.dot(s, eye))         # 平移分量 = -轴·眼睛，把摄像机移到原点
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))          # 第 3 行取了负号，所以这里符号也随之翻转
    return m


def perspective(fov_deg, aspect, near, far):
    """OpenGL 风格透视投影矩阵 (观察空间朝 -Z)."""
    # fov_deg 是对角线视场角(度)；aspect = 宽/高；near/far 是近远裁剪面距离
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)    # 焦距 f = 1/tan(fov/2)，弧度制
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect                     # x 方向按宽高比压缩，保证圆不被拉成椭圆
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)    # 深度映射到 NDC 的 [-1,1]
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0                           # 把 -z 写进 w，这是透视除法的根源
    return m


def orthographic(half_height, aspect, near, far):
    """正交投影矩阵: half_height 是视景体垂直半高(世界单位)."""
    # 正交投影不做透视缩小：平行线保持平行，远处物体不会变小
    half_width = half_height * aspect       # 水平半宽由宽高比决定
    m = np.eye(4)
    m[0, 0] = 1.0 / half_width              # 把 [-half_width, +half_width] 映射到 [-1,1]
    m[1, 1] = 1.0 / half_height
    m[2, 2] = -2.0 / (far - near)           # z 轴同样映射到 [-1,1]
    m[2, 3] = -(far + near) / (far - near)
    return m


def transform_point(m, p):
    # 用 4x4 矩阵变换一个 3D 点：先升到齐次坐标 (x,y,z,1)，乘完再丢掉 w
    v = m @ np.array([p[0], p[1], p[2], 1.0])
    return v[:3]                            # 只返回前三个分量(此处不做除法)


# ============================================================
# 2. 三角网格 (顶点 / 边 / 面)
# ============================================================

class Mesh:
    """顶点 + 三角面 + 可选 UV / 平滑法线."""

    def __init__(self, vertices, faces, uvs=None, smooth=False):
        # vertices: 顶点坐标表 (N,3)；faces: 三角面索引表 (F,3)
        self.vertices = np.asarray(vertices, dtype=np.float64)   # 顶点位置
        self.index = np.asarray(faces, dtype=np.int32)          # (F,3) 每行是一个三角形的三个顶点下标
        self.uvs = None if uvs is None else np.asarray(uvs, dtype=np.float64)  # 纹理坐标(N,2)
        self.smooth = smooth                    # True 表示用逐顶点平滑法线(Gouraud 着色)
        self.normals = self._vertex_normals() if smooth else None   # 平滑法线只算一次并缓存
        # 包围球半径：到原点最远的顶点距离，用于视锥剔除时快速判定
        self.radius = float(np.max(np.linalg.norm(self.vertices, axis=1))) if len(self.vertices) else 0.0

    def _vertex_normals(self):
        # 计算逐顶点法线：把相邻所有三角面的面积加权法线累加后归一化
        n = len(self.vertices)
        acc = np.zeros((n, 3))                  # 每个顶点的法线累加器
        v = self.vertices
        for tri in self.index:
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            fn = np.cross(b - a, c - a)         # 叉积得到该面的法线(长度正比于面积)
            acc[tri[0]] += fn                   # 三个顶点各累加一次
            acc[tri[1]] += fn
            acc[tri[2]] += fn
        length = np.linalg.norm(acc, axis=1, keepdims=True)   # 每行长度
        return acc / np.maximum(length, EPS)    # 归一化，EPS 防止零长度除法


def _fix_winding(verts, faces):
    """让凸体所有三角面法线朝外 (以原点为中心)."""
    # 目的：保证背面剔除的方向一致，否则球/圆柱会有洞
    v = np.asarray(verts, dtype=np.float64)
    out = []
    for tri in faces:
        a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
        n = np.cross(b - a, c - a)              # 面法线
        # 用三角形重心作为「朝外」参考方向；法线与重心点积为负说明面朝内
        if np.dot(n, (a + b + c) / 3.0) < 0:
            out.append((tri[0], tri[2], tri[1]))    # 交换后两个顶点，翻转绕序
        else:
            out.append(tuple(tri))
    return out


def box(size=(1.0, 1.0, 1.0)):
    """轴对齐立方体/长方体. UV 以世界单位展开, 便于墙体连续贴图."""
    # 生成 6 个面，每个面 4 个顶点 2 个三角形
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5   # 三个方向的半尺寸
    # 每个面描述：(法线, U 轴, V 轴, 法线方向半长, U 方向半长, V 方向半长)
    specs = [
        ((1, 0, 0), (0, 0, -1), (0, 1, 0), hx, hz, hy),    # +X 面
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0), hx, hz, hy),    # -X 面
        ((0, 1, 0), (1, 0, 0), (0, 0, -1), hy, hx, hz),    # +Y 面(顶面)
        ((0, -1, 0), (1, 0, 0), (0, 0, 1), hy, hx, hz),    # -Y 面(底面)
        ((0, 0, 1), (1, 0, 0), (0, 1, 0), hz, hx, hy),     # +Z 面
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0), hz, hx, hy),   # -Z 面
    ]
    verts, uvs, faces = [], [], []
    for n, u, v, hn, hu, hv in specs:
        n = np.array(n, float)                  # 面法线
        u = np.array(u, float)                  # 面内 U 方向
        v = np.array(v, float)                  # 面内 V 方向
        base = len(verts)                       # 本面第一个顶点的下标
        # 四个角按 (-1,-1)(1,-1)(1,1)(-1,1) 顺序生成，形成一圈
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(n * hn + u * (su * hu) + v * (sv * hv))    # 位置 = 法线偏移 + 两轴展开
            uvs.append(((su + 1) * hu, (sv + 1) * hv))              # UV 按世界长度展开，墙面可无缝拼接
        faces.append((base, base + 1, base + 2))    # 第一个三角形
        faces.append((base, base + 2, base + 3))    # 第二个三角形
    return Mesh(verts, faces, uvs, smooth=False)    # 立方体要硬边，所以不平滑


def plane(width, depth, uv=(1.0, 1.0)):
    """XZ 平面上的四边形, 法线 +Y."""
    # 用于地面；uv 参数控制贴图重复次数
    hw, hd = width * 0.5, depth * 0.5
    verts = [(-hw, 0, -hd), (hw, 0, -hd), (hw, 0, hd), (-hw, 0, hd)]   # 四个角的 y 都是 0
    uvs = [(0, 0), (uv[0], 0), (uv[0], uv[1]), (0, uv[1])]
    faces = [(0, 2, 1), (0, 3, 2)]              # 绕序保证法线朝 +Y(朝上)
    return Mesh(verts, faces, uvs, smooth=False)


def quad(center=(0, 0, 0), size=(1, 1), normal=(0, 0, 1)):
    """任意朝向的矩形面 (旗帜 / 叶片 / 翅膀)."""
    # 先根据给定法线构造一组正交的 U/V 轴，再在平面上铺四个顶点
    n = normalize(normal)
    # 选一个与法线不平行的辅助向量，避免叉积退化为零向量
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(helper, n))           # U 轴 = 辅助向量 × 法线
    v = np.cross(n, u)                           # V 轴 = 法线 × U 轴
    c = np.array(center, float)
    hw, hh = size[0] * 0.5, size[1] * 0.5        # 半宽半高
    # 以中心为原点，沿 ±U ±V 展开得到四个角
    verts = [c - u * hw - v * hh, c + u * hw - v * hh, c + u * hw + v * hh, c - u * hw + v * hh]
    faces = [(0, 1, 2), (0, 2, 3)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]      # 整张贴图铺满一个矩形
    return Mesh(verts, faces, uvs, smooth=False)


def sphere(radius=1.0, segs=14, rings=9, smooth=True):
    # 生成 UV 球体：纬度分成 rings 段，经度分成 segs 段，共用一个极点环
    verts, uvs, faces = [], [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings              # 极角 0..π（从北极到南极）
        y = math.cos(theta)                      # 高度分量
        r = math.sin(theta)                      # 该纬度圆的半径系数
        for j in range(segs + 1):
            phi = 2 * math.pi * j / segs         # 方位角 0..2π
            verts.append((math.cos(phi) * r * radius, y * radius, math.sin(phi) * r * radius))
            uvs.append((j / segs, 1.0 - i / rings))    # U 沿经度，V 沿纬度
    stride = segs + 1                            # 每一圈的顶点数

    def idx(i, j):
        # 把二维网格下标 (纬度 i, 经度 j) 换算成顶点数组下标
        return i * stride + j

    for i in range(rings):
        for j in range(segs):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)   # 四边形的四个角
            if i != 0:
                faces.append((a, b, c))          # 第一圈会退化成零面积三角形，跳过
            if i != rings - 1:
                faces.append((a, c, d))          # 最后一圈同理跳过
    faces = _fix_winding(verts, faces)           # 统一让法线朝外
    return Mesh(verts, faces, uvs, smooth=smooth)   # 球面平滑，用逐顶点法线


def cylinder(radius=1.0, height=1.0, segs=14, smooth=True, caps=True):
    # 生成圆柱：先做侧面(两圈顶点)，再可选地加两个圆盘盖子
    verts, uvs, faces = [], [], []
    hy = height * 0.5                            # 半高
    for i, y in enumerate((-hy, hy)):             # i=0 是底圈，i=1 是顶圈
        for j in range(segs + 1):
            phi = 2 * math.pi * j / segs
            verts.append((math.cos(phi) * radius, y, math.sin(phi) * radius))
            uvs.append((j / segs, i))            # V 方向用 0/1 区分底和顶
    stride = segs + 1
    for j in range(segs):
        a, b = j, j + 1                          # 底圈相邻两点
        c, d = stride + j + 1, stride + j        # 顶圈对应两点
        faces.append((a, b, c))                  # 侧面拆成两个三角形
        faces.append((a, c, d))
    if caps:
        for y, flip in ((-hy, True), (hy, False)):   # 两个盖子，底盖需要翻转绕序
            base = len(verts)
            center_index = base                  # 盖子圆心顶点
            verts.append((0, y, 0))
            uvs.append((0.5, 0.5))
            for j in range(segs):                # 盖子边缘一圈顶点
                phi = 2 * math.pi * j / segs
                verts.append((math.cos(phi) * radius, y, math.sin(phi) * radius))
                # 圆盘 UV 用极坐标映射到单位圆内
                uvs.append((0.5 + 0.5 * math.cos(phi), 0.5 + 0.5 * math.sin(phi)))
            for j in range(segs):                # 以圆心为中心的扇形三角化
                a = base + 1 + j
                b = base + 1 + (j + 1) % segs    # 取模实现首尾相连
                faces.append((center_index, b, a) if flip else (center_index, a, b))
    return Mesh(verts, faces, uvs, smooth=smooth)


def cone(radius=1.0, height=1.0, segs=12):
    # 生成圆锥：一个顶点 + 底面一圈顶点，再加一个底面圆盘
    verts, uvs, faces = [], [], []
    hy = height * 0.5
    verts.append((0, hy, 0))                     # 锥顶
    uvs.append((0.5, 1.0))
    for j in range(segs):                        # 底面边缘一圈
        phi = 2 * math.pi * j / segs
        verts.append((math.cos(phi) * radius, -hy, math.sin(phi) * radius))
        uvs.append((j / segs, 0.0))
    base_center = len(verts)                     # 底面圆心
    verts.append((0, -hy, 0))
    uvs.append((0.5, 0.5))
    for j in range(segs):                        # 底面圆盘边缘(单独复制一份顶点，便于法线独立)
        phi = 2 * math.pi * j / segs
        verts.append((math.cos(phi) * radius, -hy, math.sin(phi) * radius))
        uvs.append((0.5 + 0.5 * math.cos(phi), 0.5 + 0.5 * math.sin(phi)))
    for j in range(segs):
        a = 1 + j
        b = 1 + (j + 1) % segs
        faces.append((0, a, b))                  # 侧面三角形：锥顶 + 底边两点
        # 底面三角形：圆心 + 相邻两点，绕序反过来让法线朝下
        faces.append((base_center, base_center + 1 + (j + 1) % segs, base_center + 1 + j))
    return Mesh(verts, faces, uvs, smooth=False) # 圆锥有硬边，不平滑


def disc(radius=1.0, segs=20, rings=3):
    """圆盘 (用于水池), 顶点 y=0, 便于逐顶点波动动画."""
    # 同心圆环结构：圆心一个顶点，然后一层层向外扩展
    verts, uvs, faces = [], [], []
    verts.append((0, 0, 0))                      # 圆心
    uvs.append((0.5, 0.5))
    for r_i in range(1, rings + 1):
        r = radius * r_i / rings                 # 当前环的半径
        for j in range(segs):
            phi = 2 * math.pi * j / segs
            verts.append((math.cos(phi) * r, 0, math.sin(phi) * r))   # y 固定为 0，动画时再改 y
            uvs.append((0.5 + 0.5 * math.cos(phi) * r_i / rings,
                        0.5 + 0.5 * math.sin(phi) * r_i / rings))
    for j in range(segs):
        faces.append((0, 1 + j, 1 + (j + 1) % segs))    # 最内圈的扇形三角形
    for r_i in range(1, rings):                  # 相邻两环之间用四边形连接
        inner = 1 + (r_i - 1) * segs
        outer = 1 + r_i * segs
        for j in range(segs):
            a = inner + j
            b = inner + (j + 1) % segs
            c = outer + (j + 1) % segs
            d = outer + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return Mesh(verts, faces, uvs, smooth=False)


def merge(meshes):
    # 把多个网格合并成一个：顶点表拼接，面索引整体偏移
    verts, uvs, faces = [], [], []
    for m in meshes:
        offset = len(verts)                      # 当前已累计的顶点数 = 本次索引偏移量
        verts.extend(m.vertices)
        if m.uvs is not None:
            uvs.extend(m.uvs)
        else:
            uvs.extend([(0, 0)] * len(m.vertices))   # 没有 UV 就补零
        for tri in m.index:
            faces.append((tri[0] + offset, tri[1] + offset, tri[2] + offset))   # 索引加偏移
    return Mesh(verts, faces, uvs, smooth=False)


# ============================================================
# 3. 摄像机
# ============================================================

class Camera:
    def __init__(self, fov=62.0, near=1.0, far=1600.0,
                 projection="perspective", ortho_height=300.0):
        # fov: 视场角(度)；near/far: 近远裁剪面；projection: "perspective" 或 "orthographic"
        self.fov = fov
        self.near = near
        self.far = far
        self.projection = projection              # 当前投影方式，可用 V/P 键切换
        self.ortho_height = ortho_height          # 正交投影的垂直半高
        self.eye = np.array([0.0, 0.0, -10.0])    # 摄像机位置(默认在 -Z 方向)
        self.target = np.array([0.0, 0.0, 0.0])   # 注视点
        self.up = np.array([0.0, 1.0, 0.0])       # 世界上方向
        self.window_aspect = 1.4                  # 宽高比，渲染开始时会按窗口尺寸更新
        self._update_basis()                      # 立刻算一次三个轴向

    def look(self, eye, target, up=(0.0, 1.0, 0.0)):
        # 重新设置摄像机的位置/朝向，然后刷新轴向基
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self._update_basis()

    def _update_basis(self):
        # 由 eye/target/up 推导出一组正交基：前、右、真上
        self.forward = normalize(self.target - self.eye)      # 视线方向
        self.right = normalize(np.cross(self.forward, self.up))   # 右方向
        self.true_up = np.cross(self.right, self.forward)     # 上方向(已被正交化)

    def view(self):
        # 返回视图矩阵（世界 -> 摄像机坐标系）
        return look_at(self.eye, self.target, self.up)

    def projection_matrix(self, aspect):
        # 按当前投影方式返回对应的投影矩阵
        if self.projection == "orthographic":
            return orthographic(self.ortho_height, aspect, self.near, self.far)
        return perspective(self.fov, aspect, self.near, self.far)

    def is_perspective(self):
        # 是否透视投影（很多地方要根据这个分支选不同的深度/z-buffer 公式）
        return self.projection != "orthographic"

    def visible_sphere(self, center, radius):
        """视锥粗剔除: 球体是否可能出现在画面内 (保守检测)."""
        # 思路：用球体的近似包围球与视锥的四个平面对比，宁可多画也不漏画（保守）
        v = np.asarray(center, dtype=np.float64) - self.eye   # 球心相对摄像机的位置
        depth = float(np.dot(v, self.forward))                # 沿视线方向的深度
        if depth + radius < self.near:
            return False                                     # 整个球在近平面之前，不可见
        if depth - radius > self.far:
            return False                                     # 整个球在远平面之后，不可见
        if self.is_perspective():
            tan_half = math.tan(math.radians(self.fov) * 0.5)
            half_h = max(depth, 0.0) * tan_half              # 该深度处的半高
            half_w = half_h * self.window_aspect             # 该深度处的半宽
        else:
            half_h = self.ortho_height                       # 正交投影下视景体是长方体，尺寸恒定
            half_w = half_h * self.window_aspect
        # 球心在右/上方向上的投影超出半宽半高(减去半径做容差)，说明在视锥外
        if abs(float(np.dot(v, self.right))) - radius > half_w:
            return False
        if abs(float(np.dot(v, self.true_up))) - radius > half_h:
            return False
        return True


# ============================================================
# 4. 渲染器
# ============================================================

class Renderer:
    """软件光栅化渲染器: 顶点变换 -> 裁剪 -> 剔除 -> 三角形填充 + z-buffer."""

    def __init__(self, width, height, sky_top=(0.35, 0.52, 0.72),
                 sky_bottom=(0.86, 0.85, 0.76), fog_color=(0.72, 0.78, 0.78),
                 fog_density=0.0026, ambient=(0.32, 0.34, 0.40)):
        # 渲染器持有的全部状态：帧缓冲、深度缓冲、天空渐变、雾参数、环境光
        self.w = width                            # 帧缓冲宽
        self.h = height                           # 帧缓冲高
        self.color = np.zeros((height, width, 3), dtype=np.float32)    # 彩色帧缓冲 RGB
        self.zbuf = np.zeros((height, width), dtype=np.float32)        # 深度缓冲
        self.surface = pygame.Surface((width, height))                 # 最后贴到屏幕的 Pygame 表面
        self.sky_top = np.asarray(sky_top, dtype=np.float32)           # 天空顶部颜色
        self.sky_bottom = np.asarray(sky_bottom, dtype=np.float32)     # 天空底部颜色
        self.fog_color = np.asarray(fog_color, dtype=np.float32)       # 雾的颜色
        self.fog_density = fog_density                                 # 雾浓度
        self.ambient = np.asarray(ambient, dtype=np.float32)           # 环境光颜色

        # 预先算好天空渐变图：每行按 t 在顶色和底色之间线性混合
        t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        sky = self.sky_top[None, None, :] * (1 - t) + self.sky_bottom[None, None, :] * t
        self._sky = np.repeat(sky, width, axis=1)  # 每行横向复制 width 次，得到整幅天空

        # 【性能优化】预计算像素中心网格：光栅化时直接切片复用（视图、零拷贝），
        # 免得每画一个三角形都要新建两个 np.arange 数组
        self._px = (np.arange(width, dtype=np.float32) + np.float32(0.5))[None, :]   # 1×W 行向量
        self._py = (np.arange(height, dtype=np.float32) + np.float32(0.5))[:, None]  # H×1 列向量
        self._persp = True                         # 当前相机是否透视投影(每帧在 begin() 更新)

        self.camera = None                         # 当前摄像机
        self.view = np.eye(4)                      # 视图矩阵
        self.proj = np.eye(4)                      # 投影矩阵
        self.mvp = np.eye(4)                       # 投影 × 视图，即 MVP(不含模型矩阵)
        self.lights = []                           # 光源列表，每帧重置
        self.wireframe = False                     # 线框模式开关
        self.textures_enabled = True               # 纹理总开关(T 键)
        self.fog_enabled = True                    # 雾开关(F 键, Part B 扩展 1)
        self.filter_mode = "nearest"               # "nearest" | "bilinear" (Part B 扩展 2)
        self.depth_debug = False                   # 深度缓冲可视化开关(F3)

        # 统计信息：本帧画的三角形数、着色像素数、绘制调用次数
        self.stats = {"triangles": 0, "pixels": 0, "draws": 0}

    # --------------------------------------------------------
    def begin(self, camera):
        # 每帧开头调用：更新矩阵、清空颜色和深度缓冲
        self.camera = camera
        self.view = camera.view()
        self.proj = camera.projection_matrix(self.w / self.h)   # 用帧缓冲的宽高比
        self.mvp = self.proj @ self.view                         # 投影矩阵乘视图矩阵
        camera.window_aspect = self.w / self.h                   # 回写宽高比供剔除使用
        # 【性能优化】每帧只查一次投影类型并存下来，光栅化时不再逐个三角形去查
        self._persp = camera.is_perspective()
        self.color[:] = self._sky                               # 用天空渐变填满背景
        # 透视投影用 0 表示无穷远(存的是 1/w)；正交投影存负深度，用 -1e9 代表无穷远
        self.zbuf.fill(0.0 if self._persp else -1e9)
        self.stats = {"triangles": 0, "pixels": 0, "draws": 0}   # 重置统计
        self.lights = []                                        # 清空光源

    def add_directional_light(self, direction, color=(1.0, 1.0, 1.0), intensity=1.0):
        # 添加平行光(太阳)：只有方向没有位置
        self.lights.append({
            "type": "dir",
            "dir": normalize(direction),            # 光线方向(单位向量，指向光传播方向)
            "color": np.asarray(color, dtype=np.float64),
            "intensity": intensity,
        })

    def add_point_light(self, position, color=(1.0, 1.0, 1.0), intensity=1.0, radius=200.0):
        # 添加点光源(例如烟花)：有位置，且强度随距离衰减到 radius 处为 0
        self.lights.append({
            "type": "point",
            "pos": np.asarray(position, dtype=np.float64),
            "color": np.asarray(color, dtype=np.float64),
            "intensity": intensity,
            "radius": radius,
        })

    # --------------------------------------------------------
    def shade(self, normals, positions):
        # 对一批法线/位置做光照计算，返回每项的颜色 RGB。全向量化，一次处理 N 个顶点/面
        n = len(normals)
        total = np.tile(self.ambient, (n, 1))        # 先铺一层环境光(无方向，处处相同)
        for light in self.lights:
            if light["type"] == "dir":
                # 平行光：Lambert 余弦定律，法线与光线反向的点积，负值截断为 0
                ndl = np.clip(-(normals @ light["dir"]), 0.0, 1.0)
                total = total + ndl[:, None] * light["color"] * light["intensity"]
            else:
                # 点光源：先算法线朝向光源的方向，再做余弦 + 距离衰减
                v = light["pos"][None, :] - positions
                dist = np.linalg.norm(v, axis=1) + EPS          # 到光源距离
                ndl = np.clip((normals * v).sum(axis=1) / dist, 0.0, 1.0)   # 用未归一化的 v/dist 即余弦
                atten = np.clip(1.0 - dist / light["radius"], 0.0, 1.0) ** 2   # 线性衰减再平方，边缘更柔和
                total = total + (ndl * atten)[:, None] * light["color"] * light["intensity"]
        return total

    def _fog_factor(self, depth):
        # 指数平方雾：f = 1 - exp(-(density*depth)^2)，depth 越大越接近 1(全雾)
        if not self.fog_enabled:
            return np.zeros_like(depth)
        f = 1.0 - np.exp(-((depth * self.fog_density) ** 2))
        return np.clip(f, 0.0, 1.0)

    # --------------------------------------------------------
    def draw_mesh(self, mesh, model=None, color=(1.0, 1.0, 1.0), texture=None,
                  uv_scale=(1.0, 1.0), lighting=True, smooth=None, double_sided=False,
                  fog=True, fog_vertex=False, emissive=0.0, cull=True):
        # 这是整个渲染器的主入口：把一个网格画一帧。参数含义见下面逐段注释
        cam = self.camera
        if model is None:
            model = np.eye(4)                     # 没给模型矩阵就用单位矩阵(不移动)
        if not self.textures_enabled:
            texture = None                        # 全局纹理开关关闭时强制不用贴图

        # ---- 视锥剔除 (先剔除, 避免无谓的顶点变换) ----
        if cull:
            # 用包围球判定：中心取模型空间原点变换后的位置，半径按模型矩阵最大缩放放大
            center = transform_point(model, (0.0, 0.0, 0.0))
            radius = mesh.radius * max(1e-6, float(np.max(np.abs(model[:3, :3]))))
            if not cam.visible_sphere(center, radius * 1.05):    # 1.05 留一点余量
                return                            # 完全看不见，直接跳过整块网格

        v = mesh.vertices
        n = len(v)
        if n == 0:
            return                                # 空网格直接返回
        self.stats["draws"] += 1                  # 统计：一次绘制调用
        self.stats["triangles"] += len(mesh.index)   # 统计：本网格三角形数

        hom = np.concatenate([v, np.ones((n, 1))], axis=1)   # 顶点升到齐次坐标 (N,4)
        clip = hom @ (self.mvp @ model).T                     # (N,4) 裁剪空间坐标

        if smooth is None:
            smooth = mesh.smooth                  # 默认跟随网格自身的平滑标志

        base = np.asarray(color, dtype=np.float64)   # 基础色(材质颜色)

        # 视图空间深度 (雾 / 正交 z-buffer)
        view_pos = hom @ self.view.T              # 变换到摄像机坐标系
        depth_v = -view_pos[:, 2]                 # 摄像机看 -Z，取负号得到正的深度值

        world_h = hom @ model.T                   # 世界坐标(齐次)
        wpos = world_h[:, :3]                      # 取前三维，光照需要世界位置
        vshade = None                             # 逐顶点着色结果(平滑着色时才有)
        if lighting and smooth and mesh.normals is not None:
            # 法线变换：用模型矩阵的逆转置，避免非等比缩放让法线方向变错
            nmat = np.linalg.inv(model).T[:3, :3]
            nrm = mesh.normals @ nmat.T
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), EPS)   # 重新归一化
            vshade = self.shade(nrm, wpos) * base + emissive    # 光照结果乘材质色，emissive 是自发光

        idx = mesh.index
        if lighting and vshade is None:
            # 平面着色：每个三角形算一个面法线，用三角形中心作为采样位置
            tri_pos = wpos[idx]
            face_n = np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0])
            face_n = face_n / np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), EPS)   # 归一化面法线
            fshade = self.shade(face_n, tri_pos.mean(axis=1)) * base + emissive
        else:
            # 不做光照：所有三角形用同一个颜色
            fshade = np.tile(base + emissive, (len(idx), 1))

        # 雾: 按顶点到摄像机的真实距离计算 (更符合物理, 且与投影方式无关)
        vfog = None
        if fog:
            eye = np.asarray(getattr(cam, "eye", (0.0, 0.0, 0.0)), dtype=np.float64)
            vfog = self._fog_factor(np.linalg.norm(wpos - eye, axis=1))   # 每个顶点的雾系数

        # 把雾烘焙进顶点/面颜色 (省去逐像素雾计算, 渲染优化)
        if vfog is not None and fog_vertex:
            if vshade is not None:
                fc = vfog[:, None]                              # 逐顶点的雾系数
                vshade = vshade * (1.0 - fc) + self.fog_color[None, :] * fc    # 颜色向雾色混合
            else:
                ff = vfog[idx].mean(axis=1)[:, None]             # 每个三角形取三个顶点的平均
                fshade = fshade * (1.0 - ff) + self.fog_color[None, :] * ff
            vfog = None                                         # 已经烘焙进颜色，逐像素阶段不再算雾

        if texture is not None and mesh.uvs is not None:
            uv_all = mesh.uvs * np.asarray(uv_scale, dtype=np.float64)   # UV 乘重复次数
        else:
            texture = None                      # 没 UV 就退化为纯色
            uv_all = None

        d = clip[:, 2] + clip[:, 3]                          # >=0 表示在近平面之内
        face_d = d[idx]
        # 必须同时满足 d>0 (近平面之内) 且 w>0 (在眼睛前方), 否则投影会翻转
        in_front = np.all((face_d > 0.0) & (clip[idx][:, :, 3] > 1e-6), axis=1)
        behind = np.all((face_d <= 0.0) | (clip[idx][:, :, 3] <= 1e-6), axis=1)
        full = np.nonzero(in_front)[0]          # 完全可见：无需裁剪，可批量处理
        partial = np.nonzero(~in_front & ~behind)[0]   # 部分跨近平面：需要逐面裁剪

        # ---- 完全在近平面内的三角形: 批量投影 ----
        if len(full):
            ct = clip[idx[full]]                        # (K,3,4) 这批三角形的裁剪坐标
            w = ct[:, :, 3]
            ndc = ct[:, :, :3] / w[:, :, None]          # 透视除法 -> 归一化设备坐标 NDC
            sx = (ndc[:, :, 0] * 0.5 + 0.5) * self.w     # NDC [-1,1] -> 屏幕 x 像素
            sy = (0.5 - ndc[:, :, 1] * 0.5) * self.h     # y 轴要翻转，因为屏幕原点在左上角
            iw = 1.0 / w                                 # 存 1/w 供 z-buffer 与透视校正插值
            dz = depth_v[idx[full]]

            # 用屏幕空间有向面积判断绕序：面积符号决定是正面还是背面
            area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) -
                    (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
            if double_sided:
                keep = np.abs(area) > 0.5                # 双面材质：两面都画，只要不是退化三角形
                flip = area > 0                          # 绕序相反的顶点需要交换
            else:
                keep = area < -0.5                       # 单面材质：只保留正面(面积为负)
                flip = np.zeros(len(area), dtype=bool)
            if not np.any(keep):
                return                                   # 全被背面剔除，本网格画完
            full = full[keep]                            # 同步筛选所有数组
            sx, sy, iw, dz = sx[keep], sy[keep], iw[keep], dz[keep]
            flip = flip[keep]

            tri_uv = uv_all[idx[full]] if uv_all is not None else None       # 每个三角形的 UV
            tri_fog = vfog[idx[full]] if vfog is not None else None          # 每个顶点的雾系数
            tri_shade = vshade[idx[full]] if vshade is not None else None    # 逐顶点颜色
            tex = texture

            if np.any(flip):
                # 翻转绕序：把三个顶点的所有属性都按相反顺序排，保持属性与顶点对应
                sx[flip] = sx[flip][:, ::-1]
                sy[flip] = sy[flip][:, ::-1]
                iw[flip] = iw[flip][:, ::-1]
                dz[flip] = dz[flip][:, ::-1]
                if tri_uv is not None:
                    tri_uv[flip] = tri_uv[flip][:, ::-1]
                if tri_fog is not None:
                    tri_fog[flip] = tri_fog[flip][:, ::-1]
                if tri_shade is not None:
                    tri_shade[flip] = tri_shade[flip][:, ::-1]

            # 【性能优化】循环外把方法绑定成局部变量，省掉每个三角形一次属性查找
            raster = self._rasterize
            for k in range(len(full)):
                # 逐个三角形做光栅化(见 _rasterize)
                fi = full[k]
                raster(
                    sx[k], sy[k], iw[k], dz[k],
                    tri_uv[k] if tri_uv is not None else None,
                    tri_shade[k] if tri_shade is not None else fshade[fi],
                    tri_fog[k] if tri_fog is not None else None,
                    tex,
                )

        # ---- 跨越近平面的三角形: 逐面裁剪 ----
        for fi in partial:
            self._draw_clipped(fi, idx, clip, depth_v, uv_all, vfog, vshade,
                               fshade, texture, double_sided)

    def _draw_clipped(self, fi, idx, clip, depth_v, uv_all, vfog, vshade,
                      fshade, texture, double_sided):
        # Sutherland-Hodgman 近平面裁剪：把三角形被近平面切掉，剩下的多边形再扇形三角化
        tri = idx[fi]
        pts = clip[tri]                          # 三角形的三个裁剪空间顶点
        d = pts[:, 2] + pts[:, 3]                # 每个顶点到近平面的有符号距离
        attrs = []                               # 把各种逐顶点属性打包在一起，便于同步插值
        for k in range(3):
            a = []
            if uv_all is not None:
                a.extend(uv_all[tri[k]])         # 2 个 UV 分量
            if vfog is not None:
                a.append(vfog[tri[k]])           # 1 个雾系数
            if vshade is not None:
                a.extend(vshade[tri[k]])         # 3 个颜色分量
            a.append(depth_v[tri[k]])            # 1 个视图深度
            attrs.append(a)
        attrs = np.array(attrs)

        keep_p, keep_a = [], []                  # 裁剪后保留的顶点和属性
        for i in range(3):
            j = (i + 1) % 3                      # 下一个顶点(取模成环)
            di, dj = d[i], d[j]
            if di > 0:
                keep_p.append(pts[i])            # 该顶点在近平面之内，直接保留
                keep_a.append(attrs[i])
            if (di > 0) != (dj > 0):
                # 这条边跨越了近平面：在交点处插入一个新顶点
                # 参数 t 由线性插值求出，保证交点正好在 d=0 的平面上
                t = di / (di - dj)
                keep_p.append(pts[i] + t * (pts[j] - pts[i]))
                keep_a.append(attrs[i] + t * (attrs[j] - attrs[i]))   # 属性同步插值
        if len(keep_p) < 3:
            return                               # 裁剪后不足三个顶点，什么都不剩

        for k in range(1, len(keep_p) - 1):
            # 把裁剪后的多边形(最多 4 顶点)按扇形拆成三角形
            tri_p = np.array([keep_p[0], keep_p[k], keep_p[k + 1]])
            tri_a = np.array([keep_a[0], keep_a[k], keep_a[k + 1]])
            w = tri_p[:, 3]
            if np.any(w <= 1e-6):
                continue                         # w 太小说明在摄像机后面，跳过避免除零爆炸
            ndc = tri_p[:, :3] / w[:, None]      # 透视除法
            sx = (ndc[:, 0] * 0.5 + 0.5) * self.w
            sy = (0.5 - ndc[:, 1] * 0.5) * self.h
            area = ((sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]))   # 有向面积
            if double_sided:
                if abs(area) <= 0.5:
                    continue                     # 退化三角形
                if area > 0:                     # 绕序反了就整体翻转
                    sx, sy, w = sx[::-1], sy[::-1], w[::-1]
                    tri_a = tri_a[::-1]
            else:
                if area >= -0.5:
                    continue                     # 背面剔除
            iw = 1.0 / w
            off = 0                              # 属性打包时的偏移游标，顺序与上面 attrs 一致
            tri_uv = None
            if uv_all is not None:
                tri_uv = tri_a[:, off:off + 2]   # 取出 UV
                off += 2
            tri_f = None
            if vfog is not None:
                tri_f = tri_a[:, off]            # 取出雾系数
                off += 1
            tri_sh = None
            if vshade is not None:
                tri_sh = tri_a[:, off:off + 3]   # 取出颜色
                off += 3
            dz = tri_a[:, off]                   # 剩下的就是视图深度
            self._rasterize(sx, sy, iw, dz, tri_uv,
                            tri_sh if tri_sh is not None else fshade[fi], tri_f, texture)

    # --------------------------------------------------------
    def _rasterize(self, sx, sy, iw, dz, uv, shade, fogv, tex):
        # 核心：把一个屏幕空间三角形填成像素。用重心坐标 + 深度测试 + 纹理采样
        # sx/sy 本来就是 float32，直接取第 k 个元素就是 Python float，
        # 【性能优化】省掉原来的 float(...) 包装调用（每个三角形 6 次）
        x0, y0 = sx[0], sy[0]
        x1, y1 = sx[1], sy[1]
        x2, y2 = sx[2], sy[2]

        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)   # 有向面积的两倍
        if area >= -0.5:                                   # 背面 / 退化三角形
            return

        # 先用包围盒把可能的像素范围算出来，避免遍历整个屏幕
        minx = int(math.floor(min(x0, x1, x2)))
        maxx = int(math.ceil(max(x0, x1, x2)))
        miny = int(math.floor(min(y0, y1, y2)))
        maxy = int(math.ceil(max(y0, y1, y2)))
        if minx < 0:
            minx = 0                                     # 裁剪到屏幕边界内
        if miny < 0:
            miny = 0
        if maxx >= self.w:
            maxx = self.w - 1
        if maxy >= self.h:
            maxy = self.h - 1
        if maxx < minx or maxy < miny:
            return                                       # 包围盒完全在屏幕外

        if self.wireframe:
            # 线框调试模式：不填充，只画三角形的三条边
            self._wire_tri(x0, y0, x1, y1, x2, y2, shade[0] if shade.ndim > 1 else shade)
            return

        # 【性能优化】复用构造时预计算好的像素中心网格：切片是视图、不拷贝数据，
        # 省掉原来每个三角形都要新建两个 np.arange 数组的开销
        xs = self._px[0:1, minx:maxx + 1]                # 像素中心 x 坐标（1×W 行向量视图）
        ys = self._py[miny:maxy + 1, 0:1]                # 像素中心 y 坐标（H×1 列向量视图）

        # 重心坐标 (边函数); area<0 时内部点的边函数都为负
        # e0/e1/e2 分别对应「对面 0/1/2 的边」，三者同时 <=0 的点就在三角形内部
        e0 = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        e1 = (x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)
        e2 = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
        mask = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)      # True 表示该像素在三角形内
        if not mask.any():
            return                                     # 包围盒里实际没有像素落在三角形内

        inv_area = np.float32(1.0 / area)               # 面积的倒数，用来归一化边函数
        b0 = e0 * inv_area                              # 归一化后的重心权重
        b1 = e1 * inv_area
        b2 = e2 * inv_area

        # 【性能优化】透视/正交开关在 begin() 里每帧查一次就存起来，
        # 不必每画一个三角形都调用一次相机方法
        persp = self._persp
        iw0, iw1, iw2 = np.float32(iw[0]), np.float32(iw[1]), np.float32(iw[2])   # 三个顶点的 1/w
        if persp:
            q = b0 * iw0 + b1 * iw1 + b2 * iw2           # 插值出 1/w，越大表示越近
            inv = 1.0 / np.maximum(q, np.float32(1e-9))  # 透视校正用，夹 1e-9 防除零
        else:
            # 正交：直接线性插值视图空间深度，再取负号，让「越大越近」和透视一致
            q = -(b0 * np.float32(dz[0]) + b1 * np.float32(dz[1]) + b2 * np.float32(dz[2]))
            inv = None                                   # 正交没有透视变形，不需要校正

        sub_depth = self.zbuf[miny:maxy + 1, minx:maxx + 1]   # 取出这块区域的深度缓冲
        mask &= q > sub_depth                            # 深度测试：只有比已有像素更近才通过
        if not mask.any():
            return                                       # 整片都被遮挡
        self.stats["pixels"] += int(mask.sum())          # 统计着色的像素数

        # ---- 逐像素雾因子 ----
        # 【性能优化】雾因子在这里一次算完，下面三条着色分支共用；
        # 原实现是每条分支里各算一遍，同一批像素的雾被重复算了 3 次
        fogf = None                              # 没有顶点雾数据时保持 None
        if fogv is not None:
            if persp:
                # 透视：雾的顶点值同样要乘 1/w 再插值，最后除以插值后的 1/w
                fogf = (b0 * np.float32(fogv[0] * iw0) +
                        b1 * np.float32(fogv[1] * iw1) +
                        b2 * np.float32(fogv[2] * iw2)) * inv
            else:
                # 正交：直接线性插值
                fogf = (b0 * np.float32(fogv[0]) +
                        b1 * np.float32(fogv[1]) +
                        b2 * np.float32(fogv[2]))
            fogf = np.clip(fogf, 0.0, 1.0)       # 雾浓度夹到 [0,1]，防止过饱和

        flat = shade.ndim == 1                   # True = 整面单色，不需要逐像素插值颜色

        # ---- 颜色 ----
        if tex is not None and uv is not None:
            # 透视校正插值：UV 先乘 1/w 再加权，最后除以插值后的 1/w
            if persp:
                u = (b0 * np.float32(uv[0, 0] * iw0) +
                     b1 * np.float32(uv[1, 0] * iw1) +
                     b2 * np.float32(uv[2, 0] * iw2)) * inv
                v = (b0 * np.float32(uv[0, 1] * iw0) +
                     b1 * np.float32(uv[1, 1] * iw1) +
                     b2 * np.float32(uv[2, 1] * iw2)) * inv
            else:
                # 正交投影没有透视变形，直接线性插值即可
                u = b0 * np.float32(uv[0, 0]) + b1 * np.float32(uv[1, 0]) + b2 * np.float32(uv[2, 0])
                v = b0 * np.float32(uv[0, 1]) + b1 * np.float32(uv[1, 1]) + b2 * np.float32(uv[2, 1])
            th, tw = tex.shape[0], tex.shape[1]          # 纹理高/宽
            if not flat:
                # 逐顶点颜色(Gouraud)：按重心权重插值
                cs = (b0[..., None] * np.float32(shade[0]) +
                      b1[..., None] * np.float32(shade[1]) +
                      b2[..., None] * np.float32(shade[2]))
            else:
                cs = np.float32(shade)                   # 平面着色：整面同一个颜色

            # 退化三角形可能算出 NaN/Inf 的 UV，这里取模到 [0,1) 保证下标合法。
            # 【性能优化】去掉了原来的 np.nan_to_num（它内部要扫 isposinf/isneginf/nan
            # 三趟数组）：NaN 取模后仍是 NaN，转 int32 再按位与一样得到合法下标，结果等价
            u = np.mod(u, 1.0)
            v = np.mod(v, 1.0)

            if self.filter_mode == "bilinear":
                # Bilinear filtering: sample the 2x2 texel neighbourhood and
                # blend with the fractional part of the texture coordinate.
                # Part B extension 2 -- see report Section 7.
                # 双线性过滤：取最近 2x2 个纹素，按 UV 的小数部分做两次线性混合
                fx = u * np.float32(tw) - np.float32(0.5)    # 像素中心对齐，偏移半个纹素
                fy = v * np.float32(th) - np.float32(0.5)
                x0 = np.floor(fx).astype(np.int64)           # 左上纹素坐标
                y0 = np.floor(fy).astype(np.int64)
                tx = (fx - x0)[..., None]                    # 小数部分作为横向权重
                ty = (fy - y0)[..., None]                    # 纵向权重
                # 用按位与实现「2 的幂纹理」无缝环绕：x & (tw-1) 等价于 x mod tw
                x0m = x0 & (tw - 1)
                x1m = (x0 + 1) & (tw - 1)
                y0m = y0 & (th - 1)
                y1m = (y0 + 1) & (th - 1)
                c00 = tex[y0m, x0m]                          # 四个相邻纹素
                c10 = tex[y0m, x1m]
                c01 = tex[y1m, x0m]
                c11 = tex[y1m, x1m]
                top = c00 * (1.0 - tx) + c10 * tx            # 上面两个横向混合
                bot = c01 * (1.0 - tx) + c11 * tx            # 下面两个横向混合
                col = (top * (1.0 - ty) + bot * ty) * cs     # 再纵向混合，最后乘光照色
            else:
                # Nearest-neighbour sampling with power-of-two wrap.
                # 最近邻采样：直接取一个纹素，同样用按位与做 2 的幂环绕。
                # NaN 取模后仍是 NaN，转 int32 会报 invalid 告警但下标按位与后依然合法，
                # 所以这里屏蔽掉那条告警（不影响画面）
                with np.errstate(invalid="ignore"):
                    ti = (v * np.float32(th)).astype(np.int32) & (th - 1)
                    tj = (u * np.float32(tw)).astype(np.int32) & (tw - 1)
                col = tex[ti, tj] * cs
        elif not flat:
            # 无纹理但逐顶点着色：按重心权重插值颜色
            col = (b0[..., None] * np.float32(shade[0]) +
                   b1[..., None] * np.float32(shade[1]) +
                   b2[..., None] * np.float32(shade[2]))
        else:
            col = np.float32(shade)                       # 无纹理无插值：整面纯色

        if fogf is not None:
            # 用前面算好的雾因子把颜色往雾色混合（fogf 补成 (...,1) 便于按通道广播）
            col = col * (1.0 - fogf[..., None]) + self.fog_color * fogf[..., None]

        # 用 np.where 按 mask 批量写入：mask 为 True 的像素才更新颜色和深度
        sub_color = self.color[miny:maxy + 1, minx:maxx + 1]
        col2 = col if col.ndim == 3 else col[None, None, :]   # 单色面先补成 1×1×3 再广播
        sub_color[...] = np.where(mask[..., None], col2, sub_color)
        sub_depth[...] = np.where(mask, q, sub_depth)

    # --------------------------------------------------------
    def _wire_tri(self, x0, y0, x1, y1, x2, y2, shade):
        # 线框绘制：沿三条边按参数 t 采样像素，直接写成亮色
        base = np.asarray(shade, dtype=np.float32)
        c = 0.35 + 0.65 * base                      # 让线框颜色带一点材质亮度
        for ax, ay, bx, by in ((x0, y0, x1, y1), (x1, y1, x2, y2), (x2, y2, x0, y0)):
            steps = int(max(abs(bx - ax), abs(by - ay)))   # 采样步数 = 边上像素跨度
            if steps <= 0:
                continue                            # 退化边，跳过
            t = np.linspace(0.0, 1.0, min(steps, 2000))    # 上限 2000，防止超长边太慢
            px = np.clip((ax + (bx - ax) * t).astype(np.int64), 0, self.w - 1)   # 线性插值并夹到屏幕内
            py = np.clip((ay + (by - ay) * t).astype(np.int64), 0, self.h - 1)
            self.color[py, px] = c                  # 逐点写入颜色

    # --------------------------------------------------------
    def draw_glow(self, position, radius, color, intensity=1.0):
        # 画一个加法混合的发光圆斑(用于粒子/烟花)，带深度测试
        cam = self.camera
        p = self.view @ np.array([position[0], position[1], position[2], 1.0])   # 变换到摄像机空间
        depth = -p[2]                                # 正的深度
        if depth <= cam.near:
            return                                   # 在近平面之前，不画
        c = self.proj @ p                            # 投影
        w = c[3]
        if w <= 1e-6:
            return                                   # 在摄像机后面，投影会翻转，直接跳过
        sx = (c[0] / w * 0.5 + 0.5) * self.w         # 屏幕坐标（透视除法后映射）
        sy = (0.5 - c[1] / w * 0.5) * self.h

        if cam.is_perspective():
            f = 1.0 / math.tan(math.radians(cam.fov) * 0.5)
            rpx = f * radius / depth * self.h * 0.5  # 世界半径 -> 屏幕像素半径(随距离缩小)
            q = 1.0 / depth                          # 深度值(1/w 体系)
        else:
            rpx = radius / max(cam.ortho_height, 1e-6) * self.h * 0.5   # 正交：大小不随距离变
            q = -depth
        rpx = min(max(rpx, 0.7), 300.0)              # 限制像素半径范围，避免退化成一点或过大

        x0 = int(sx - rpx)                           # 算出需要处理的正方形区域
        x1 = int(sx + rpx) + 1
        y0 = int(sy - rpx)
        y1 = int(sy + rpx) + 1
        if x1 < 0 or y1 < 0 or x0 >= self.w or y0 >= self.h:
            return                                   # 完全在屏幕外
        x0 = max(x0, 0)                              # 夹到屏幕内
        y0 = max(y0, 0)
        x1 = min(x1, self.w - 1)
        y1 = min(y1, self.h - 1)
        if x1 < x0 or y1 < y0:
            return

        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5     # 像素中心坐标
        ys = np.arange(y0, y1 + 1, dtype=np.float32)[:, None] + 0.5
        dx = xs - sx
        dy = ys - sy
        r2 = dx * dx + dy * dy                       # 到圆心的距离平方
        mask = r2 <= rpx * rpx                       # 圆形范围
        sub = self.zbuf[y0:y1 + 1, x0:x1 + 1]
        mask &= q > sub                              # 深度测试：被更近的物体挡住就不发光
        if not mask.any():
            return
        falloff = np.clip(1.0 - np.sqrt(r2) / rpx, 0.0, 1.0) ** 2   # 从中心到边缘平方衰减
        add = falloff[mask, None] * (np.asarray(color, dtype=np.float32) * intensity)   # 要叠加的亮度
        self.color[y0:y1 + 1, x0:x1 + 1][mask] += add   # 加法混合：在已有颜色上累加(所以会变亮)

    # --------------------------------------------------------
    def end(self):
        # 每帧结束：可选地把深度缓冲可视化，然后把帧缓冲贴到 Pygame 表面
        if self.depth_debug:
            # Visualise the z-buffer: near surfaces are bright/green, far
            # surfaces fade to blue. Helps to prove that occlusion is correct.
            # 深度缓冲可视化：近处亮/偏绿，远处渐变成蓝色，用来证明遮挡关系正确
            q = self.zbuf
            if self.camera.is_perspective():
                depth = np.where(q > 1e-6, 1.0 / np.maximum(q, 1e-6), 2500.0)   # 1/w 还原成深度
            else:
                depth = -q                       # 正交存的是负深度，取反即可
            sky = self.zbuf <= (0.0 if self.camera.is_perspective() else -1e8)   # 背景像素(没画东西)
            # Exponential falloff gives good contrast both in corridors and in
            # wide overview shots: near surfaces warm, far surfaces cold.
            # 指数衰减让近处暖色、远处冷色，走廊和全景截图都能看清层次
            t = np.exp(-np.maximum(depth, 0.0) / 240.0)
            dbg = np.stack([0.10 + 0.90 * t, 0.22 + 0.70 * t, 0.62 - 0.40 * t], axis=-1)
            dbg[sky] = (0.04, 0.05, 0.10)        # 背景统一画成深蓝黑
            self.color[:] = dbg.astype(np.float32)
        np.clip(self.color, 0.0, 1.0, out=self.color)        # 把浮点颜色夹到 [0,1]
        arr = (self.color * 255.0 + 0.5).astype(np.uint8)    # 转成 0-255 的 8 位整数
        # Pygame 的 surface 是 (宽,高,通道) 顺序，而帧缓冲是 (高,宽,通道)，所以要交换前两维
        pygame.surfarray.blit_array(self.surface, np.transpose(arr, (1, 0, 2)))
        return self.surface
