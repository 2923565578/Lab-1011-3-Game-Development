"""
engine3d.py -- a software 3D renderer written from scratch with NumPy and Pygame.

No OpenGL, no game engine. Every required stage of the graphics pipeline is
implemented in this file:

  1. 3D coordinates and geometry      : Mat4, vertex lists, triangle meshes
  2. 3D transforms                    : translation / rotation / scaling matrices
  3. Camera and projections           : look-at view matrix, perspective + orthographic
  4. Clipping                         : conservative frustum cull + near-plane
                                        Sutherland-Hodgman polygon clipping
  5. Back-face culling                : signed area in screen space
  6. Triangle rasterization           : bounding box + barycentric edge functions
  7. Depth buffer (z-buffer)          : perspective uses 1/w, orthographic uses view depth
  8. Lighting                         : ambient + directional + point light,
                                        flat and Gouraud (per-vertex) shading
  9. Texture mapping                  : perspective-correct UV interpolation,
                                        nearest or bilinear sampling (Part B extension 2)
 10. 3D particles                     : additive billboards with depth testing
 11. Render optimizations             : frustum culling, distance culling, per-cell
                                        batches, internal resolution + smooth upscale
 12. Debug views                      : wireframe, depth-buffer visualization
"""

from __future__ import annotations

import math

import numpy as np
import pygame

EPS = 1e-9


# ============================================================
# 1. 向量与 4x4 齐次矩阵数学
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v.copy()


def mat_identity():
    return np.eye(4, dtype=np.float64)


def translation(x, y, z):
    m = np.eye(4)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scaling(x, y=None, z=None):
    if y is None:
        y = x
    if z is None:
        z = x
    m = np.eye(4)
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotation_x(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotation_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotation_z(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """构造右手系视图矩阵 (与 gluLookAt 相同)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = normalize(target - eye)              # 前方向
    s = normalize(np.cross(f, up))           # 右方向
    u = np.cross(s, f)                       # 上方向

    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def perspective(fov_deg, aspect, near, far):
    """OpenGL 风格透视投影矩阵 (观察空间朝 -Z)."""
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(half_height, aspect, near, far):
    """正交投影矩阵: half_height 是视景体垂直半高(世界单位)."""
    half_width = half_height * aspect
    m = np.eye(4)
    m[0, 0] = 1.0 / half_width
    m[1, 1] = 1.0 / half_height
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def transform_point(m, p):
    v = m @ np.array([p[0], p[1], p[2], 1.0])
    return v[:3]


# ============================================================
# 2. 三角网格 (顶点 / 边 / 面)
# ============================================================

class Mesh:
    """顶点 + 三角面 + 可选 UV / 平滑法线."""

    def __init__(self, vertices, faces, uvs=None, smooth=False):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.index = np.asarray(faces, dtype=np.int32)          # (F,3)
        self.uvs = None if uvs is None else np.asarray(uvs, dtype=np.float64)
        self.smooth = smooth
        self.normals = self._vertex_normals() if smooth else None
        self.radius = float(np.max(np.linalg.norm(self.vertices, axis=1))) if len(self.vertices) else 0.0

    def _vertex_normals(self):
        n = len(self.vertices)
        acc = np.zeros((n, 3))
        v = self.vertices
        for tri in self.index:
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            fn = np.cross(b - a, c - a)
            acc[tri[0]] += fn
            acc[tri[1]] += fn
            acc[tri[2]] += fn
        length = np.linalg.norm(acc, axis=1, keepdims=True)
        return acc / np.maximum(length, EPS)


def _fix_winding(verts, faces):
    """让凸体所有三角面法线朝外 (以原点为中心)."""
    v = np.asarray(verts, dtype=np.float64)
    out = []
    for tri in faces:
        a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
        n = np.cross(b - a, c - a)
        if np.dot(n, (a + b + c) / 3.0) < 0:
            out.append((tri[0], tri[2], tri[1]))
        else:
            out.append(tuple(tri))
    return out


def box(size=(1.0, 1.0, 1.0)):
    """轴对齐立方体/长方体. UV 以世界单位展开, 便于墙体连续贴图."""
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    specs = [
        ((1, 0, 0), (0, 0, -1), (0, 1, 0), hx, hz, hy),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0), hx, hz, hy),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1), hy, hx, hz),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1), hy, hx, hz),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0), hz, hx, hy),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0), hz, hx, hy),
    ]
    verts, uvs, faces = [], [], []
    for n, u, v, hn, hu, hv in specs:
        n = np.array(n, float)
        u = np.array(u, float)
        v = np.array(v, float)
        base = len(verts)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(n * hn + u * (su * hu) + v * (sv * hv))
            uvs.append(((su + 1) * hu, (sv + 1) * hv))
        faces.append((base, base + 1, base + 2))
        faces.append((base, base + 2, base + 3))
    return Mesh(verts, faces, uvs, smooth=False)


def plane(width, depth, uv=(1.0, 1.0)):
    """XZ 平面上的四边形, 法线 +Y."""
    hw, hd = width * 0.5, depth * 0.5
    verts = [(-hw, 0, -hd), (hw, 0, -hd), (hw, 0, hd), (-hw, 0, hd)]
    uvs = [(0, 0), (uv[0], 0), (uv[0], uv[1]), (0, uv[1])]
    faces = [(0, 2, 1), (0, 3, 2)]
    return Mesh(verts, faces, uvs, smooth=False)


def quad(center=(0, 0, 0), size=(1, 1), normal=(0, 0, 1)):
    """任意朝向的矩形面 (旗帜 / 叶片 / 翅膀)."""
    n = normalize(normal)
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(helper, n))
    v = np.cross(n, u)
    c = np.array(center, float)
    hw, hh = size[0] * 0.5, size[1] * 0.5
    verts = [c - u * hw - v * hh, c + u * hw - v * hh, c + u * hw + v * hh, c - u * hw + v * hh]
    faces = [(0, 1, 2), (0, 2, 3)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return Mesh(verts, faces, uvs, smooth=False)


def sphere(radius=1.0, segs=14, rings=9, smooth=True):
    verts, uvs, faces = [], [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        y = math.cos(theta)
        r = math.sin(theta)
        for j in range(segs + 1):
            phi = 2 * math.pi * j / segs
            verts.append((math.cos(phi) * r * radius, y * radius, math.sin(phi) * r * radius))
            uvs.append((j / segs, 1.0 - i / rings))
    stride = segs + 1

    def idx(i, j):
        return i * stride + j

    for i in range(rings):
        for j in range(segs):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            if i != 0:
                faces.append((a, b, c))
            if i != rings - 1:
                faces.append((a, c, d))
    faces = _fix_winding(verts, faces)
    return Mesh(verts, faces, uvs, smooth=smooth)


def cylinder(radius=1.0, height=1.0, segs=14, smooth=True, caps=True):
    verts, uvs, faces = [], [], []
    hy = height * 0.5
    for i, y in enumerate((-hy, hy)):
        for j in range(segs + 1):
            phi = 2 * math.pi * j / segs
            verts.append((math.cos(phi) * radius, y, math.sin(phi) * radius))
            uvs.append((j / segs, i))
    stride = segs + 1
    for j in range(segs):
        a, b = j, j + 1
        c, d = stride + j + 1, stride + j
        faces.append((a, b, c))
        faces.append((a, c, d))
    if caps:
        for y, flip in ((-hy, True), (hy, False)):
            base = len(verts)
            center_index = base
            verts.append((0, y, 0))
            uvs.append((0.5, 0.5))
            for j in range(segs):
                phi = 2 * math.pi * j / segs
                verts.append((math.cos(phi) * radius, y, math.sin(phi) * radius))
                uvs.append((0.5 + 0.5 * math.cos(phi), 0.5 + 0.5 * math.sin(phi)))
            for j in range(segs):
                a = base + 1 + j
                b = base + 1 + (j + 1) % segs
                faces.append((center_index, b, a) if flip else (center_index, a, b))
    return Mesh(verts, faces, uvs, smooth=smooth)


def cone(radius=1.0, height=1.0, segs=12):
    verts, uvs, faces = [], [], []
    hy = height * 0.5
    verts.append((0, hy, 0))
    uvs.append((0.5, 1.0))
    for j in range(segs):
        phi = 2 * math.pi * j / segs
        verts.append((math.cos(phi) * radius, -hy, math.sin(phi) * radius))
        uvs.append((j / segs, 0.0))
    base_center = len(verts)
    verts.append((0, -hy, 0))
    uvs.append((0.5, 0.5))
    for j in range(segs):
        phi = 2 * math.pi * j / segs
        verts.append((math.cos(phi) * radius, -hy, math.sin(phi) * radius))
        uvs.append((0.5 + 0.5 * math.cos(phi), 0.5 + 0.5 * math.sin(phi)))
    for j in range(segs):
        a = 1 + j
        b = 1 + (j + 1) % segs
        faces.append((0, a, b))
        faces.append((base_center, base_center + 1 + (j + 1) % segs, base_center + 1 + j))
    return Mesh(verts, faces, uvs, smooth=False)


def disc(radius=1.0, segs=20, rings=3):
    """圆盘 (用于水池), 顶点 y=0, 便于逐顶点波动动画."""
    verts, uvs, faces = [], [], []
    verts.append((0, 0, 0))
    uvs.append((0.5, 0.5))
    for r_i in range(1, rings + 1):
        r = radius * r_i / rings
        for j in range(segs):
            phi = 2 * math.pi * j / segs
            verts.append((math.cos(phi) * r, 0, math.sin(phi) * r))
            uvs.append((0.5 + 0.5 * math.cos(phi) * r_i / rings,
                        0.5 + 0.5 * math.sin(phi) * r_i / rings))
    for j in range(segs):
        faces.append((0, 1 + j, 1 + (j + 1) % segs))
    for r_i in range(1, rings):
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
    verts, uvs, faces = [], [], []
    for m in meshes:
        offset = len(verts)
        verts.extend(m.vertices)
        if m.uvs is not None:
            uvs.extend(m.uvs)
        else:
            uvs.extend([(0, 0)] * len(m.vertices))
        for tri in m.index:
            faces.append((tri[0] + offset, tri[1] + offset, tri[2] + offset))
    return Mesh(verts, faces, uvs, smooth=False)


# ============================================================
# 3. 摄像机
# ============================================================

class Camera:
    def __init__(self, fov=62.0, near=1.0, far=1600.0,
                 projection="perspective", ortho_height=300.0):
        self.fov = fov
        self.near = near
        self.far = far
        self.projection = projection
        self.ortho_height = ortho_height
        self.eye = np.array([0.0, 0.0, -10.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.window_aspect = 1.4
        self._update_basis()

    def look(self, eye, target, up=(0.0, 1.0, 0.0)):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self._update_basis()

    def _update_basis(self):
        self.forward = normalize(self.target - self.eye)
        self.right = normalize(np.cross(self.forward, self.up))
        self.true_up = np.cross(self.right, self.forward)

    def view(self):
        return look_at(self.eye, self.target, self.up)

    def projection_matrix(self, aspect):
        if self.projection == "orthographic":
            return orthographic(self.ortho_height, aspect, self.near, self.far)
        return perspective(self.fov, aspect, self.near, self.far)

    def is_perspective(self):
        return self.projection != "orthographic"

    def visible_sphere(self, center, radius):
        """视锥粗剔除: 球体是否可能出现在画面内 (保守检测)."""
        v = np.asarray(center, dtype=np.float64) - self.eye
        depth = float(np.dot(v, self.forward))
        if depth + radius < self.near:
            return False
        if depth - radius > self.far:
            return False
        if self.is_perspective():
            tan_half = math.tan(math.radians(self.fov) * 0.5)
            half_h = max(depth, 0.0) * tan_half
            half_w = half_h * self.window_aspect
        else:
            half_h = self.ortho_height
            half_w = half_h * self.window_aspect
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
        self.w = width
        self.h = height
        self.color = np.zeros((height, width, 3), dtype=np.float32)
        self.zbuf = np.zeros((height, width), dtype=np.float32)
        self.surface = pygame.Surface((width, height))
        self.sky_top = np.asarray(sky_top, dtype=np.float32)
        self.sky_bottom = np.asarray(sky_bottom, dtype=np.float32)
        self.fog_color = np.asarray(fog_color, dtype=np.float32)
        self.fog_density = fog_density
        self.ambient = np.asarray(ambient, dtype=np.float32)

        t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        sky = self.sky_top[None, None, :] * (1 - t) + self.sky_bottom[None, None, :] * t
        self._sky = np.repeat(sky, width, axis=1)

        self.camera = None
        self.view = np.eye(4)
        self.proj = np.eye(4)
        self.mvp = np.eye(4)
        self.lights = []
        self.wireframe = False
        self.textures_enabled = True
        self.fog_enabled = True
        self.filter_mode = "nearest"     # "nearest" | "bilinear" (Part B extension)
        self.depth_debug = False

        self.stats = {"triangles": 0, "pixels": 0, "draws": 0}

    # --------------------------------------------------------
    def begin(self, camera):
        self.camera = camera
        self.view = camera.view()
        self.proj = camera.projection_matrix(self.w / self.h)
        self.mvp = self.proj @ self.view
        camera.window_aspect = self.w / self.h
        self.color[:] = self._sky
        self.zbuf.fill(0.0 if camera.is_perspective() else -1e9)
        self.stats = {"triangles": 0, "pixels": 0, "draws": 0}
        self.lights = []

    def add_directional_light(self, direction, color=(1.0, 1.0, 1.0), intensity=1.0):
        self.lights.append({
            "type": "dir",
            "dir": normalize(direction),
            "color": np.asarray(color, dtype=np.float64),
            "intensity": intensity,
        })

    def add_point_light(self, position, color=(1.0, 1.0, 1.0), intensity=1.0, radius=200.0):
        self.lights.append({
            "type": "point",
            "pos": np.asarray(position, dtype=np.float64),
            "color": np.asarray(color, dtype=np.float64),
            "intensity": intensity,
            "radius": radius,
        })

    # --------------------------------------------------------
    def shade(self, normals, positions):
        n = len(normals)
        total = np.tile(self.ambient, (n, 1))
        for light in self.lights:
            if light["type"] == "dir":
                ndl = np.clip(-(normals @ light["dir"]), 0.0, 1.0)
                total = total + ndl[:, None] * light["color"] * light["intensity"]
            else:
                v = light["pos"][None, :] - positions
                dist = np.linalg.norm(v, axis=1) + EPS
                ndl = np.clip((normals * v).sum(axis=1) / dist, 0.0, 1.0)
                atten = np.clip(1.0 - dist / light["radius"], 0.0, 1.0) ** 2
                total = total + (ndl * atten)[:, None] * light["color"] * light["intensity"]
        return total

    def _fog_factor(self, depth):
        if not self.fog_enabled:
            return np.zeros_like(depth)
        f = 1.0 - np.exp(-((depth * self.fog_density) ** 2))
        return np.clip(f, 0.0, 1.0)

    # --------------------------------------------------------
    def draw_mesh(self, mesh, model=None, color=(1.0, 1.0, 1.0), texture=None,
                  uv_scale=(1.0, 1.0), lighting=True, smooth=None, double_sided=False,
                  fog=True, fog_vertex=False, emissive=0.0, cull=True):
        cam = self.camera
        if model is None:
            model = np.eye(4)
        if not self.textures_enabled:
            texture = None

        # ---- 视锥剔除 (先剔除, 避免无谓的顶点变换) ----
        if cull:
            center = transform_point(model, (0.0, 0.0, 0.0))
            radius = mesh.radius * max(1e-6, float(np.max(np.abs(model[:3, :3]))))
            if not cam.visible_sphere(center, radius * 1.05):
                return

        v = mesh.vertices
        n = len(v)
        if n == 0:
            return
        self.stats["draws"] += 1
        self.stats["triangles"] += len(mesh.index)

        hom = np.concatenate([v, np.ones((n, 1))], axis=1)
        clip = hom @ (self.mvp @ model).T                     # (N,4) 裁剪空间坐标

        if smooth is None:
            smooth = mesh.smooth

        base = np.asarray(color, dtype=np.float64)

        # 视图空间深度 (雾 / 正交 z-buffer)
        view_pos = hom @ self.view.T
        depth_v = -view_pos[:, 2]

        world_h = hom @ model.T
        wpos = world_h[:, :3]
        vshade = None
        if lighting and smooth and mesh.normals is not None:
            nmat = np.linalg.inv(model).T[:3, :3]
            nrm = mesh.normals @ nmat.T
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), EPS)
            vshade = self.shade(nrm, wpos) * base + emissive

        idx = mesh.index
        if lighting and vshade is None:
            tri_pos = wpos[idx]
            face_n = np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0])
            face_n = face_n / np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), EPS)
            fshade = self.shade(face_n, tri_pos.mean(axis=1)) * base + emissive
        else:
            fshade = np.tile(base + emissive, (len(idx), 1))

        # 雾: 按顶点到摄像机的真实距离计算 (更符合物理, 且与投影方式无关)
        vfog = None
        if fog:
            eye = np.asarray(getattr(cam, "eye", (0.0, 0.0, 0.0)), dtype=np.float64)
            vfog = self._fog_factor(np.linalg.norm(wpos - eye, axis=1))

        # 把雾烘焙进顶点/面颜色 (省去逐像素雾计算, 渲染优化)
        if vfog is not None and fog_vertex:
            if vshade is not None:
                fc = vfog[:, None]
                vshade = vshade * (1.0 - fc) + self.fog_color[None, :] * fc
            else:
                ff = vfog[idx].mean(axis=1)[:, None]
                fshade = fshade * (1.0 - ff) + self.fog_color[None, :] * ff
            vfog = None

        if texture is not None and mesh.uvs is not None:
            uv_all = mesh.uvs * np.asarray(uv_scale, dtype=np.float64)
        else:
            texture = None
            uv_all = None

        d = clip[:, 2] + clip[:, 3]                          # >=0 表示在近平面之内
        face_d = d[idx]
        # 必须同时满足 d>0 (近平面之内) 且 w>0 (在眼睛前方), 否则投影会翻转
        in_front = np.all((face_d > 0.0) & (clip[idx][:, :, 3] > 1e-6), axis=1)
        behind = np.all((face_d <= 0.0) | (clip[idx][:, :, 3] <= 1e-6), axis=1)
        full = np.nonzero(in_front)[0]
        partial = np.nonzero(~in_front & ~behind)[0]

        # ---- 完全在近平面内的三角形: 批量投影 ----
        if len(full):
            ct = clip[idx[full]]
            w = ct[:, :, 3]
            ndc = ct[:, :, :3] / w[:, :, None]
            sx = (ndc[:, :, 0] * 0.5 + 0.5) * self.w
            sy = (0.5 - ndc[:, :, 1] * 0.5) * self.h
            iw = 1.0 / w
            dz = depth_v[idx[full]]

            area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) -
                    (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
            if double_sided:
                keep = np.abs(area) > 0.5
                flip = area > 0
            else:
                keep = area < -0.5
                flip = np.zeros(len(area), dtype=bool)
            if not np.any(keep):
                return
            full = full[keep]
            sx, sy, iw, dz = sx[keep], sy[keep], iw[keep], dz[keep]
            flip = flip[keep]

            tri_uv = uv_all[idx[full]] if uv_all is not None else None
            tri_fog = vfog[idx[full]] if vfog is not None else None
            tri_shade = vshade[idx[full]] if vshade is not None else None
            tex = texture

            if np.any(flip):
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

            for k in range(len(full)):
                fi = full[k]
                self._rasterize(
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
        tri = idx[fi]
        pts = clip[tri]
        d = pts[:, 2] + pts[:, 3]
        attrs = []
        for k in range(3):
            a = []
            if uv_all is not None:
                a.extend(uv_all[tri[k]])
            if vfog is not None:
                a.append(vfog[tri[k]])
            if vshade is not None:
                a.extend(vshade[tri[k]])
            a.append(depth_v[tri[k]])
            attrs.append(a)
        attrs = np.array(attrs)

        keep_p, keep_a = [], []
        for i in range(3):
            j = (i + 1) % 3
            di, dj = d[i], d[j]
            if di > 0:
                keep_p.append(pts[i])
                keep_a.append(attrs[i])
            if (di > 0) != (dj > 0):
                t = di / (di - dj)
                keep_p.append(pts[i] + t * (pts[j] - pts[i]))
                keep_a.append(attrs[i] + t * (attrs[j] - attrs[i]))
        if len(keep_p) < 3:
            return

        for k in range(1, len(keep_p) - 1):
            tri_p = np.array([keep_p[0], keep_p[k], keep_p[k + 1]])
            tri_a = np.array([keep_a[0], keep_a[k], keep_a[k + 1]])
            w = tri_p[:, 3]
            if np.any(w <= 1e-6):
                continue
            ndc = tri_p[:, :3] / w[:, None]
            sx = (ndc[:, 0] * 0.5 + 0.5) * self.w
            sy = (0.5 - ndc[:, 1] * 0.5) * self.h
            area = ((sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]))
            if double_sided:
                if abs(area) <= 0.5:
                    continue
                if area > 0:
                    sx, sy, w = sx[::-1], sy[::-1], w[::-1]
                    tri_a = tri_a[::-1]
            else:
                if area >= -0.5:
                    continue
            iw = 1.0 / w
            off = 0
            tri_uv = None
            if uv_all is not None:
                tri_uv = tri_a[:, off:off + 2]
                off += 2
            tri_f = None
            if vfog is not None:
                tri_f = tri_a[:, off]
                off += 1
            tri_sh = None
            if vshade is not None:
                tri_sh = tri_a[:, off:off + 3]
                off += 3
            dz = tri_a[:, off]
            self._rasterize(sx, sy, iw, dz, tri_uv,
                            tri_sh if tri_sh is not None else fshade[fi], tri_f, texture)

    # --------------------------------------------------------
    def _rasterize(self, sx, sy, iw, dz, uv, shade, fogv, tex):
        x0, y0 = float(sx[0]), float(sy[0])
        x1, y1 = float(sx[1]), float(sy[1])
        x2, y2 = float(sx[2]), float(sy[2])

        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area >= -0.5:                                   # 背面 / 退化三角形
            return

        minx = int(math.floor(min(x0, x1, x2)))
        maxx = int(math.ceil(max(x0, x1, x2)))
        miny = int(math.floor(min(y0, y1, y2)))
        maxy = int(math.ceil(max(y0, y1, y2)))
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx >= self.w:
            maxx = self.w - 1
        if maxy >= self.h:
            maxy = self.h - 1
        if maxx < minx or maxy < miny:
            return

        if self.wireframe:
            self._wire_tri(x0, y0, x1, y1, x2, y2, shade[0] if shade.ndim > 1 else shade)
            return

        xs = np.arange(minx, maxx + 1, dtype=np.float32) + np.float32(0.5)
        ys = (np.arange(miny, maxy + 1, dtype=np.float32) + np.float32(0.5))[:, None]

        # 重心坐标 (边函数); area<0 时内部点的边函数都为负
        e0 = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        e1 = (x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)
        e2 = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
        mask = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
        if not mask.any():
            return

        inv_area = np.float32(1.0 / area)
        b0 = e0 * inv_area
        b1 = e1 * inv_area
        b2 = e2 * inv_area

        persp = self.camera.is_perspective()
        iw0, iw1, iw2 = np.float32(iw[0]), np.float32(iw[1]), np.float32(iw[2])
        invw = b0 * iw0 + b1 * iw1 + b2 * iw2
        inv = 1.0 / np.maximum(invw, np.float32(1e-9)) if persp else None
        if persp:
            q = invw                                        # 越大越近
        else:
            q = -(b0 * np.float32(dz[0]) + b1 * np.float32(dz[1]) + b2 * np.float32(dz[2]))

        sub_depth = self.zbuf[miny:maxy + 1, minx:maxx + 1]
        mask &= q > sub_depth
        if not mask.any():
            return
        self.stats["pixels"] += int(mask.sum())

        # ---- 颜色 ----
        if tex is not None and uv is not None:
            if persp:
                u = (b0 * np.float32(uv[0, 0] * iw0) +
                     b1 * np.float32(uv[1, 0] * iw1) +
                     b2 * np.float32(uv[2, 0] * iw2)) * inv
                v = (b0 * np.float32(uv[0, 1] * iw0) +
                     b1 * np.float32(uv[1, 1] * iw1) +
                     b2 * np.float32(uv[2, 1] * iw2)) * inv
            else:
                u = b0 * np.float32(uv[0, 0]) + b1 * np.float32(uv[1, 0]) + b2 * np.float32(uv[2, 0])
                v = b0 * np.float32(uv[0, 1]) + b1 * np.float32(uv[1, 1]) + b2 * np.float32(uv[2, 1])
            th, tw = tex.shape[0], tex.shape[1]
            if shade.ndim == 2:
                cs = (b0[..., None] * np.float32(shade[0]) +
                      b1[..., None] * np.float32(shade[1]) +
                      b2[..., None] * np.float32(shade[2]))
            else:
                cs = shade

            # Degenerate (zero-area) fragments can produce NaN/Inf UVs. Wrap
            # them into [0,1) so texture indexing never raises a cast warning.
            u = np.mod(np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), 1.0)
            v = np.mod(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 1.0)

            if self.filter_mode == "bilinear":
                # Bilinear filtering: sample the 2x2 texel neighbourhood and
                # blend with the fractional part of the texture coordinate.
                # Part B extension 2 -- see report Section 7.
                fx = u * np.float32(tw) - np.float32(0.5)
                fy = v * np.float32(th) - np.float32(0.5)
                x0 = np.floor(fx).astype(np.int64)
                y0 = np.floor(fy).astype(np.int64)
                tx = (fx - x0)[..., None]
                ty = (fy - y0)[..., None]
                x0m = x0 & (tw - 1)
                x1m = (x0 + 1) & (tw - 1)
                y0m = y0 & (th - 1)
                y1m = (y0 + 1) & (th - 1)
                c00 = tex[y0m, x0m]
                c10 = tex[y0m, x1m]
                c01 = tex[y1m, x0m]
                c11 = tex[y1m, x1m]
                top = c00 * (1.0 - tx) + c10 * tx
                bot = c01 * (1.0 - tx) + c11 * tx
                col = (top * (1.0 - ty) + bot * ty) * cs
            else:
                # Nearest-neighbour sampling with power-of-two wrap.
                ti = (v * np.float32(th)).astype(np.int32) & (th - 1)
                tj = (u * np.float32(tw)).astype(np.int32) & (tw - 1)
                col = tex[ti, tj] * cs
        elif shade.ndim == 2:
            s0, s1, s2 = shade
            col = (b0[..., None] * np.float32(s0) +
                   b1[..., None] * np.float32(s1) +
                   b2[..., None] * np.float32(s2))
        else:
            col = shade

        if fogv is not None:
            if persp:
                f = (b0 * np.float32(fogv[0] * iw0) +
                     b1 * np.float32(fogv[1] * iw1) +
                     b2 * np.float32(fogv[2] * iw2)) * inv
            else:
                f = b0 * np.float32(fogv[0]) + b1 * np.float32(fogv[1]) + b2 * np.float32(fogv[2])
            f = np.clip(f, 0.0, 1.0)[..., None]
            col = col * (1.0 - f) + self.fog_color * f

        sub_color = self.color[miny:maxy + 1, minx:maxx + 1]
        sub_color[...] = np.where(mask[..., None], col, sub_color)
        sub_depth[...] = np.where(mask, q, sub_depth)

    # --------------------------------------------------------
    def _wire_tri(self, x0, y0, x1, y1, x2, y2, shade):
        base = np.asarray(shade, dtype=np.float32)
        c = 0.35 + 0.65 * base
        for ax, ay, bx, by in ((x0, y0, x1, y1), (x1, y1, x2, y2), (x2, y2, x0, y0)):
            steps = int(max(abs(bx - ax), abs(by - ay)))
            if steps <= 0:
                continue
            t = np.linspace(0.0, 1.0, min(steps, 2000))
            px = np.clip((ax + (bx - ax) * t).astype(np.int64), 0, self.w - 1)
            py = np.clip((ay + (by - ay) * t).astype(np.int64), 0, self.h - 1)
            self.color[py, px] = c

    # --------------------------------------------------------
    def draw_glow(self, position, radius, color, intensity=1.0):
        cam = self.camera
        p = self.view @ np.array([position[0], position[1], position[2], 1.0])
        depth = -p[2]
        if depth <= cam.near:
            return
        c = self.proj @ p
        w = c[3]
        if w <= 1e-6:
            return
        sx = (c[0] / w * 0.5 + 0.5) * self.w
        sy = (0.5 - c[1] / w * 0.5) * self.h

        if cam.is_perspective():
            f = 1.0 / math.tan(math.radians(cam.fov) * 0.5)
            rpx = f * radius / depth * self.h * 0.5
            q = 1.0 / depth
        else:
            rpx = radius / max(cam.ortho_height, 1e-6) * self.h * 0.5
            q = -depth
        rpx = min(max(rpx, 0.7), 300.0)

        x0 = int(sx - rpx)
        x1 = int(sx + rpx) + 1
        y0 = int(sy - rpx)
        y1 = int(sy + rpx) + 1
        if x1 < 0 or y1 < 0 or x0 >= self.w or y0 >= self.h:
            return
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, self.w - 1)
        y1 = min(y1, self.h - 1)
        if x1 < x0 or y1 < y0:
            return

        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32)[:, None] + 0.5
        dx = xs - sx
        dy = ys - sy
        r2 = dx * dx + dy * dy
        mask = r2 <= rpx * rpx
        sub = self.zbuf[y0:y1 + 1, x0:x1 + 1]
        mask &= q > sub
        if not mask.any():
            return
        falloff = np.clip(1.0 - np.sqrt(r2) / rpx, 0.0, 1.0) ** 2
        add = falloff[mask, None] * (np.asarray(color, dtype=np.float32) * intensity)
        self.color[y0:y1 + 1, x0:x1 + 1][mask] += add

    # --------------------------------------------------------
    def end(self):
        if self.depth_debug:
            # Visualise the z-buffer: near surfaces are bright/green, far
            # surfaces fade to blue. Helps to prove that occlusion is correct.
            q = self.zbuf
            if self.camera.is_perspective():
                depth = np.where(q > 1e-6, 1.0 / np.maximum(q, 1e-6), 2500.0)
            else:
                depth = -q
            sky = self.zbuf <= (0.0 if self.camera.is_perspective() else -1e8)
            # Exponential falloff gives good contrast both in corridors and in
            # wide overview shots: near surfaces warm, far surfaces cold.
            t = np.exp(-np.maximum(depth, 0.0) / 240.0)
            dbg = np.stack([0.10 + 0.90 * t, 0.22 + 0.70 * t, 0.62 - 0.40 * t], axis=-1)
            dbg[sky] = (0.04, 0.05, 0.10)
            self.color[:] = dbg.astype(np.float32)
        np.clip(self.color, 0.0, 1.0, out=self.color)
        arr = (self.color * 255.0 + 0.5).astype(np.uint8)
        pygame.surfarray.blit_array(self.surface, np.transpose(arr, (1, 0, 2)))
        return self.surface
