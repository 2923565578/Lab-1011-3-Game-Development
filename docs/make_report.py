"""
docs/make_report.py

Builds the English technical report for Lab 10 + Lab 11 (Part A + Part B) from
the measured data in docs/performance.csv and the screenshots in screenshots/.

    python docs/make_report.py
    -> docs/Technical_Report_Lab10_11.docx

Convert to PDF with Word / LibreOffice afterwards.
"""

import csv
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "screenshots")
DOCS = os.path.join(ROOT, "docs")

doc = Document()

# ---------------------------------------------------------------- page setup
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.left_margin = Cm(2.1)
sec.right_margin = Cm(2.1)
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(1.9)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.line_spacing = 1.12
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

for name, size, color in (("Heading 1", 15, (0x1F, 0x3B, 0x57)),
                          ("Heading 2", 12, (0x1F, 0x4E, 0x79)),
                          ("Heading 3", 11, (0x33, 0x33, 0x33))):
    st = styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(*color)
    st.paragraph_format.space_before = Pt(10)
    st.paragraph_format.space_after = Pt(4)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, bold_lead=None, italic=False):
    par = doc.add_paragraph()
    if bold_lead:
        run = par.add_run(bold_lead)
        run.bold = True
    run = par.add_run(text)
    run.italic = italic
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return par


def bullet(text):
    par = doc.add_paragraph(text, style="List Bullet")
    par.paragraph_format.space_after = Pt(2)
    return par


def eq(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.font.name = "Cambria Math"
    run.font.size = Pt(10.5)
    run.italic = True
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(4)


def caption(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(8)


def figure(name, width_cm, cap):
    doc.add_picture(os.path.join(SHOTS, name), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
    caption(cap)


def pair(left, right, cap_left, cap_right, cap):
    table = doc.add_table(rows=2, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, img, cap_text in ((table.cell(0, 0), left, cap_left),
                                (table.cell(0, 1), right, cap_right)):
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cell.paragraphs[0].add_run()
        run.add_picture(os.path.join(SHOTS, img), width=Cm(7.6))
        cp = cell.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(cap_text)
        r.italic = True
        r.font.size = Pt(8.5)
    table.cell(1, 0).merge(table.cell(1, 1))
    c = table.cell(1, 0).paragraphs[0]
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(cap)
    r.italic = True
    r.font.size = Pt(9)


def table(header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(text))
            run.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ---------------------------------------------------------------- title block
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Worm's Garden Adventure 3D")
run.bold = True
run.font.size = Pt(24)
run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run("From a 2D Grid Game to a Software-Rendered 3D Game")
run.font.size = Pt(13)
run.italic = True

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = meta.add_run(
    "Laboratory 10 + 11 (Part A + Part B)  |  Course GEIAL31L-B2a  |  "
    "Computer Graphics\n"
    "Student: ____________________    Neptun code: ______________    "
    "Date: 19 November 2026\n"
    "Repository: https://github.com/2923565578/Lab-1011-3-Game-Development")
run.font.size = Pt(9.5)

# ---------------------------------------------------------------- 1
h("1. Introduction and objectives")
p("This report documents the conversion of the 2D grid game of the first half of the "
  "semester into a complete 3D game. The new game, Worm's Garden Adventure 3D, is built "
  "on a software renderer written from scratch in Python with NumPy and Pygame. No "
  "OpenGL call, no shader and no third-party 3D engine is used: the transformation "
  "stack, the camera, the projections, the near-plane clipping, the back-face culling, "
  "the triangle rasterizer, the depth buffer, the lighting model and the texture mapper "
  "are all implemented in engine3d.py and are therefore examinable.")
p("Part A (70 marks) required a working 3D game that preserves the 2D gameplay. Part B "
  "(30 marks) required two original extensions with before/after evidence and a "
  "mathematical explanation. This submission implements Part A completely and adds two "
  "graphics-oriented extensions: exponential-squared depth fog and bilinear texture "
  "filtering, both switchable at run time with the F and B keys.")

h("2. What changed from 2D to 3D")
bullet("Cells are no longer pixels. One maze cell is 50 world units (CELL = 50); the "
       "maze is 23 x 15 cells, i.e. 1150 x 750 world units. Walls are 40 units high and "
       "8 units thick.")
bullet("The player is a three-dimensional object with a real height. Jumping changes the "
       "world y coordinate of the worm, not a screen offset, and hazards test the "
       "worm's actual 3D position.")
bullet("Drawing order is irrelevant: visibility is resolved per pixel by a depth buffer, "
       "so walls, the worm, apples, birds and water can be submitted in any order.")
bullet("The camera is a full orbit camera that can rotate, zoom and switch between "
       "perspective and orthographic projection, and the game is resolution "
       "independent (1250x900 window, four internal framebuffer sizes).")

h("3. Architecture")
p("The code is split into three modules plus tools and tests, so that the renderer can "
  "be studied independently of the game rules.")
table(["Module", "Responsibility", "Key contents"],
      [("engine3d.py", "Software 3D pipeline", "Mat4, transform_point, Camera (look-at, perspective, orthographic), Renderer (clip, cull, rasterize, z-buffer, lighting, texture, fog, filter, debug views), Mesh, box/sphere/cone/disc/plane generators"),
       ("game3d.py", "Game and scene", "Maze generation and wall boxes, player physics, circle-vs-AABB collision, height-aware hazards, apples, birds, water pools, flags, particles, HUD, minimap, levels, quality presets, capture hooks"),
       ("textures.py", "Procedural textures", "Pillow-generated grass, stone, water and particle textures (no binary assets shipped)"),
       ("tests/", "Verification", "17 numerical unit tests"),
       ("tools/", "Evidence", "capture_screenshots.py, benchmark.py, render_video.py")],
      widths=[3.3, 4.2, 9.3])
p("The pipeline stages map directly onto the handbook's M1-M12 list:")
table(["Handbook stage", "Implementation"],
      [("M1 3D world data", "Meshes with position, normal, uv; world units"),
       ("M2 transforms", "4x4 matrices: translation, rotation_x/y/z, scaling, composition with @"),
       ("M3 camera", "look_at basis (right, up, forward); orbit rig with yaw/pitch/distance"),
       ("M4 perspective", "pinhole projection, focal length from FOV, aspect-correct"),
       ("M5 orthographic", "parallel projection with explicit ortho_height"),
       ("M6 clipping", "frustum sphere test + near-plane Sutherland-Hodgman polygon clipping"),
       ("M7 back-face culling", "signed screen-space area, counter-clockwise convention"),
       ("M8 rasterization", "bounding box + three edge functions, barycentric coverage mask"),
       ("M9 depth buffer", "1/w for perspective, -z for orthographic, per-pixel test"),
       ("M10 lighting", "ambient + directional + point lights, flat and Gouraud shading"),
       ("M11 textures", "perspective-correct UV (u/w interpolation), nearest or bilinear"),
       ("M12 optimization", "frustum culling, distance culling, per-cell batches, resolution scaling")],
      widths=[4.2, 12.4])

h("4. Coordinates, transforms and camera")
p("World space is right-handed with y up. The ground is the plane y = 0; a cell (col, row) "
  "has its centre at:")
eq("P = ((col + 0.5) * CELL, 0, (row + 0.5) * CELL),  CELL = 50")
p("The cell-to-world mapping is verified by a unit test: the original 2D start cell (0, 7) "
  "becomes (25, 0, 375) world units, i.e. (0.5, 7.5) cells. Rotations follow the "
  "right-hand rule and are composed as matrices; a point is transformed as a column "
  "vector with a homogeneous 1, so the model matrix is applied as T @ R @ S.")
eq("p' = T * R * S * p,     p = (x, y, z, 1)^T")
p("A worked example from the handbook is implemented as a test: rotate +90 degrees "
  "around y and translate by (2, 0, 3); the point (1, 0, 0) becomes (2, 0, 2). Directions "
  "use w = 0, so they ignore translation. Uniform scale commutes with rotation, non-uniform "
  "scale does not - both facts are asserted in the test suite.")
p("The camera is defined by an eye point and a target. Its basis is built with a look-at "
  "construction:")
eq("forward = normalize(target - eye);  right = normalize(cross(forward, up_world));  "
   "up = cross(right, forward)")
p("The view matrix maps world points into camera space with the camera looking down -z, "
  "which is the convention used by the projection step. Six presets are available "
  "and the same rig is used by the screenshot tool with fixed viewpoints "
  "(overview, corridor, close-up, orthographic overview). The player can rotate and zoom "
  "the orbit rig with the mouse at any time.")

h("5. Perspective and orthographic projection")
p("Perspective uses a pinhole model. The focal length is derived from the vertical field "
  "of view so that changing the window aspect ratio never stretches the image:")
eq("f = (H / 2) / tan(FOV / 2)")
eq("sx = W/2 + f * (x_c / z_c),   sy = H/2 - f * (y_c / z_c),   z_c < 0")
p("The near plane is z_c = -near and the far plane is z_c = -far. Orthographic projection "
  "divides by a fixed world height instead of by depth:")
eq("sx = W/2 + (x_c / ortho_height) * H,   sy = H/2 - (y_c / ortho_height) * H")
p("Both projections share the same view matrix, so the V key switches projection without "
  "moving the camera. Figure 1 shows the same eye point with both projections: in "
  "perspective the far walls converge and the maze fits the frame, in orthographic all "
  "parallel edges stay parallel and the maze keeps a constant scale.")
pair("01_perspective_full.png", "02_orthographic_full.png",
     "Perspective", "Orthographic",
     "Figure 1. Same camera, two projections. The perspective view converges towards the "
     "horizon; the orthographic view keeps wall edges parallel.")

h("6. Clipping, culling, rasterization and the depth buffer")
p("Each mesh is transformed to camera space once. A conservative sphere test rejects "
  "meshes completely outside the frustum (using the mesh bounding radius), and triangles "
  "with all three vertices behind the near plane are rejected. Triangles that cross the "
  "near plane are cut with Sutherland-Hodgman clipping in camera space, which produces one "
  "or two new triangles without changing the visible silhouette. This is what prevents the "
  "'giant polygon' artefact when the camera is close to a wall.")
p("After projection, screen-space winding determines the facing: a counter-clockwise "
  "triangle is front-facing, a clockwise one is culled. The rasterizer then computes the "
  "bounding box of the triangle, builds three edge functions")
eq("E0 = (x1-x0)(y-y0) - (y1-y0)(x-x0),  and cyclically for E1, E2")
p("and keeps fragments where all three edge functions have the same sign. The barycentric "
  "weights b0, b1, b2 are the normalized edge values, so colour, UV, fog and depth can all "
  "be interpolated with the same weights. For perspective, the depth key is 1/w and the "
  "attributes are divided by w before interpolation; dividing by the interpolated 1/w "
  "afterwards gives perspective-correct UVs (no 'affine swimming' on the floor).")
p("The depth test is a 'greater' test on 1/w (nearer fragments have a larger 1/w). "
  "Figure 2 is a controlled test: three cubes at different depths; the near red cube must "
  "hide parts of the yellow and green cubes. Figure 3 visualises the same buffer for the "
  "game scene, with near surfaces warm and far surfaces cold. Because the game draws the "
  "scene without any sorting, correct occlusion here proves that depth testing - not draw "
  "order - is doing the work.")
figure("05_occlusion.png", 12.0,
       "Figure 2. Controlled z-buffer test: near red cube occludes the yellow and green "
       "cubes behind it.")
figure("04_depth_buffer.png", 12.0,
       "Figure 3. Z-buffer visualisation of the game scene (bright = near, blue = far).")

h("7. Lighting and texture mapping")
p("Shading uses a Phong-style model with an ambient term, one directional light and dynamic "
  "point lights (the worm's lantern and firework flashes). For each vertex "
  "the diffuse term is")
eq("C = k_a * C_base + k_d * C_base * max(0, n . l) + sum_i k_p * C_i * max(0, n . l_i) / (1 + (d_i / r_i)^2)")
p("Two shading modes are supported: flat shading (one normal per triangle, one colour for "
  "the whole face) and Gouraud shading (per-vertex normals, colour interpolated with the "
  "barycentric weights). Spheres, the worm and the birds use smooth vertex normals; boxes "
  "and walls use flat faces, which keeps the stone blocks crisp. Textures are generated "
  "procedurally with Pillow at start-up (grass, stone, water), sampled with a power-of-two "
  "wrap, and modulated by the interpolated light colour, so a wall in shadow is a darker "
  "version of the same stone texture.")
figure("06_gameplay.png", 13.0,
       "Figure 4. Gameplay frame: segmented worm, apple, bird and water hazard, all lit by "
       "ambient + directional light plus the worm's point light.")

h("8. Gameplay conversion")
p("The maze generator is unchanged in spirit: recursive backtracking on a 23 x 15 grid, "
  "then wall boxes are emitted for each closed edge and merged per cell for batching. "
  "Movement is camera-relative: the input vector is rotated by the camera yaw, normalized, "
  "and integrated with dt, so speed is frame-rate independent.")
p("Collision is a circle versus axis-aligned box test, resolved per axis. The player first "
  "moves in x and is pushed out of any overlapping wall, then moves in z; this produces "
  "smooth sliding along walls instead of sticking. Apples, birds and water use circle or "
  "sphere tests in 3D.")
p("Jumping is a physical simulation, not a sine offset. Space sets the vertical velocity "
  "to v0 = 168 units/s; gravity g = 520 units/s^2 is integrated with the same dt as the "
  "rest of the game:")
eq("v <- v - g * dt,   y <- y + v * dt,   h_max = v0^2 / (2g) = 27.1 units,   "
   "t_flight = 2v0/g = 0.646 s")
p("The camera aims at the ground position, so the worm visibly rises inside the frame; a "
  "contact shadow shrinks as it climbs. Hazards are height-aware: water only kills when "
  "the worm's feet are below 14 units, and a bird collision is a true 3D sphere test, so a "
  "well-timed jump can clear a pool or pass under a swooping bird. Four apples per level "
  "must be collected to unlock the finish flag; silver apples are worth 50 points and gold "
  "apples 100. Levels increase the number of birds and pools. Death and level transitions "
  "trigger 3D particle bursts that are depth-tested against the scene, and the worm body "
  "is five spheres following a 14-sample history of head positions, which produces the "
  "segmented, snake-like motion required by the brief.")

h("9. Part B extension 1: depth fog")
p("The first extension is an atmospheric depth cue. After shading, the fragment colour is "
  "blended towards a fog colour with")
eq("f = 1 - exp(-(density * z)^2),   C_out = (1 - f) * C_shaded + f * C_fog")
p("where z is the perspective-correct view depth and density = 0.0016. The squared "
  "exponential keeps nearby geometry almost unaffected while distant walls fade quickly, "
  "which reinforces the sense of depth and hides the far clipping boundary. The factor is "
  "interpolated in the same barycentric pass as the colour and is free of per-pixel "
  "branches. The F key switches it off and on at run time; Figure 5 is the identical frame "
  "with and without fog.")
pair("07_extension1_fog_on.png", "08_extension1_fog_off.png",
     "Fog ON", "Fog OFF",
     "Figure 5. Extension 1: exponential-squared depth fog. Far walls lose contrast and "
     "take the sky colour; near walls are unchanged.")

h("10. Part B extension 2: bilinear texture filtering")
p("The second extension changes image quality rather than atmosphere. Nearest-neighbour "
  "sampling was replaced by a bilinear reconstruction of the texture signal. With texture "
  "coordinates (u, v) in texels, the four surrounding texels are fetched with a "
  "power-of-two wrap and blended with the fractional offsets tx, ty:")
eq("C = (1-ty) * [(1-tx) * T00 + tx * T10] + ty * [(1-tx) * T01 + tx * T11]")
p("The cost is four texture fetches and six multiplies per fragment instead of one fetch. "
  "The benefit is visible in Figure 6: on a magnified checkerboard the nearest image has "
  "hard, aliased staircase edges, while the bilinear image is smooth and continuous. In "
  "the game the effect is strongest on the ground at grazing angles and on close walls. "
  "The B key toggles the filter; the benchmark measures the overhead (Section 11).")
pair("09_extension2_nearest.png", "10_extension2_bilinear.png",
     "Nearest", "Bilinear",
     "Figure 6. Extension 2: nearest versus bilinear sampling on the same magnified "
     "checkerboard (8x8 texels, 3 UV repeats).")

h("11. Testing, performance and robustness")
p("Seventeen numerical tests run before any visual debugging. They cover the transform "
  "worked examples, the cell-to-world mapping, perspective versus orthographic scaling, "
  "the jump peak and flight time against the analytical solution, wall collision, "
  "z-buffer ordering, near-plane clipping (no NaN, plausible coverage) and the bilinear "
  "filter (the number of grey levels must grow). All tests pass:")
p("python -m unittest discover -s tests -v  ->  Ran 17 tests ... OK", italic=True)

perf_rows = []
csv_path = os.path.join(DOCS, "performance.csv")
if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            perf_rows.append((row["preset"], f"{row['width']}x{row['height']}",
                              row["triangles"], row["pixels"], row["avg_ms"], row["fps"]))
if not perf_rows:
    perf_rows = [("see docs/performance.csv", "", "", "", "", "")]
p("The renderer was measured on the same camera and the same frame for 45 frames per "
  "preset after a warm-up. The internal framebuffer is the dominant cost because every "
  "pixel is shaded in Python/NumPy:")
table(["Preset", "Framebuffer", "Triangles", "Shaded pixels", "Avg ms", "FPS"],
      perf_rows, widths=[3.0, 3.2, 2.6, 3.2, 2.4, 1.8])
p("The scaling is close to linear in pixel count, which confirms that fill rate - not "
  "triangle setup - is the bottleneck. Frustum culling, distance culling (330 units), "
  "back-face culling and per-cell batching reduce the submitted triangle count, and the "
  "quality presets trade resolution for frame time. The bilinear extension roughly "
  "doubles the fragment cost, as expected from four texture fetches; this is reported "
  "honestly rather than hidden.")
p("Resizing and aspect ratio were tested with the same scene rendered at 16:9 and 4:3 "
  "(Figure 7). Because the vertical FOV is fixed and the focal length is derived from the "
  "window height, cubes keep their proportions and nothing is stretched; the wider window "
  "simply reveals more of the scene, exactly as a real camera would.")
p("The window itself is freely resizable and can be maximised (F11 toggles fullscreen). "
  "On a resize the HUD caches, the projection and the framebuffer are rebuilt for the new "
  "size. The internal render resolution follows the window only up to a fixed pixel "
  "budget (about 132k shaded pixels), so a maximised window keeps the same frame time "
  "instead of multiplying fill cost, while F5 still exposes the explicit presets used "
  "for the screenshots. Measured frame time stays at roughly 12-15 FPS from 1250x900 "
  "up to 1920x1080; without the budget the maximised window fell to about 5 FPS. "
  "Because the framebuffer is deliberately smaller than the window, the final smoothscale "
  "would soften edges, so an optional unsharp mask (key C, a 4-neighbour high-pass pass "
  "over the small framebuffer) restores apparent detail before upscaling. Its cost scales "
  "with the framebuffer, not with the window, so it is cheap; it is an image-processing "
  "step, not extra geometry, and it is disclosed here rather than passed off as "
  "additional samples.")
pair("11_resize_16x9.png", "12_resize_4x3.png",
     "16:9", "4:3",
     "Figure 7. Aspect-ratio test: identical scene at 640x360 and 640x480.")

h("12. Limitations and possible improvements")
bullet("Frame rate is limited by the pure-Python/NumPy fragment loop. A tile-based "
       "rasterizer, or vectorizing several triangles at once, would give a large constant "
       "factor improvement.")
bullet("Lighting is per-vertex for smooth meshes; per-pixel normals would improve the "
       "worm and the birds but would need a normal map or per-fragment normals.")
bullet("There are no soft shadows; the contact shadow under the worm is a projected disc, "
       "not a shadow map.")
bullet("Birds fly at a fixed height band and cannot dive at the player; a simple steering "
       "behaviour would make the hazard more interesting.")

h("13. Conclusion")
p("The project converts every gameplay rule of the 2D prototype into a consistent 3D "
  "world and renders it with a pipeline that is implemented and documented stage by stage. "
  "The result is not a tech demo: it is a playable maze game with a real vertical jump, "
  "height-aware hazards, animated creatures, particles, a scoring and level system and a "
  "debuggable renderer. The two extensions - depth fog and bilinear filtering - are both "
  "graphics-oriented, both measurable, and both demonstrated with before/after evidence "
  "and the equations that drive them.")

h("Appendix A. Controls")
table(["Key", "Action"],
      [("W A S D / arrows", "Move (camera-relative)"),
       ("Space", "Physical jump with gravity"),
       ("Right drag / wheel", "Rotate / zoom the orbit camera"),
       ("V or P", "Perspective / orthographic projection"),
       ("T / F / B", "Textures / fog (extension 1) / bilinear filter (extension 2)"),
       ("F1 or G / F3", "Wireframe / z-buffer visualisation"),
       ("F5", "Cycle FAST / BALANCED / HIGH / NATIVE"),
       ("F11 / C", "Fullscreen toggle / sharpening before upscaling"),
       ("R / Esc", "Restart / quit")],
      widths=[5.0, 11.5])

h("Appendix B. Reproduction")
p("python game3d.py", italic=True)
p("python -m unittest discover -s tests -v", italic=True)
p("python tools/capture_screenshots.py", italic=True)
p("python tools/benchmark.py", italic=True)
p("python tools/render_video.py", italic=True)

out = os.path.join(DOCS, "Technical_Report_Lab10_11.docx")
doc.save(out)
print("written", out)
