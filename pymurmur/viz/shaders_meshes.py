"""Vertex/index mesh data for 3D rendering — tetrahedron/winged bird
meshes, sky quad, impostor quad, grid lines, HUD quad.

Level 2 — static data, no project imports. Only numpy for mesh arrays.

Split out of shaders.py (file-size split) — GLSL shader program strings
and theme palettes stay in the original / moved to shaders_themes.py.
"""

from __future__ import annotations

import numpy as np

# ── Tetrahedron mesh (4 vertices, 4 triangular faces) ────────────
# Asymmetric: front tip at +Z for visible orientation.
TETRA_VERTICES = np.array([
    [ 0.0,  0.0,  1.0],  # front tip
    [ 0.0,  0.943, -0.333],  # top
    [-0.816, -0.471, -0.333],  # bottom-left
    [ 0.816, -0.471, -0.333],  # bottom-right
], dtype=np.float32)

TETRA_INDICES = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 1],
    [1, 3, 2],
], dtype=np.uint32)

# ── P8.4: Winged mesh — 7 vertices, 6 triangles (body + wings + tail) ──
# Each vertex is (x, y, z, flap_weight).  flap_weight = 0 for body/tail,
# ±0.5 for wing tips — the shader uses this as the oscillation amplitude.
WINGED_VERTICES = np.array([
    # 0: nose tip
    [ 0.0,  0.0,  1.0,  0.0],
    # 1: body top
    [ 0.0,  0.25, -0.2,  0.0],
    # 2: body bottom
    [ 0.0, -0.15, -0.2,  0.0],
    # 3: right wing tip (flaps up/down)
    [ 0.65,  0.05,  0.0,  0.5],
    # 4: left wing tip (flaps opposite)
    [-0.65,  0.05,  0.0, -0.5],
    # 5: tail upper
    [ 0.0,  0.1, -0.7,  0.0],
    # 6: tail lower
    [ 0.0, -0.05, -0.7,  0.0],
], dtype=np.float32)

WINGED_INDICES = np.array([
    # Body (2 triangles)
    [0, 1, 2],   # right body panel
    [0, 2, 1],   # left body panel (opposite winding)
    # Wings (2 triangles)
    [1, 3, 2],   # right wing
    [1, 2, 4],   # left wing
    # Tail (2 triangles)
    [2, 1, 5],   # tail upper
    [1, 2, 6],   # tail lower
], dtype=np.uint32)

# ── Fullscreen quad for gradient sky (2 triangles, clip-space) ────
SKY_QUAD = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
], dtype=np.float32)

SKY_QUAD_INDICES = np.array([
    [0, 1, 2],
    [0, 2, 3],
], dtype=np.uint32)


# ── P8.1: Sphere impostor quad mesh ──────────────────────────────
# Camera-facing unit quad, centred at origin, 2 triangles.
IMPOSTOR_QUAD = np.array([
    [-0.5, -0.5],   # bottom-left
    [ 0.5, -0.5],   # bottom-right
    [ 0.5,  0.5],   # top-right
    [-0.5,  0.5],   # top-left
], dtype=np.float32)

IMPOSTOR_QUAD_INDICES = np.array([
    [0, 1, 2],
    [0, 2, 3],
], dtype=np.uint32)


# ── Grid line vertices (XY plane, centered on origin) ────────────
GRID_VERTICES = np.array([
    # X-axis lines
    [-1000, -1000, 0], [-1000, 1000, 0],
    [-750, -1000, 0], [-750, 1000, 0],
    [-500, -1000, 0], [-500, 1000, 0],
    [-250, -1000, 0], [-250, 1000, 0],
    [0, -1000, 0], [0, 1000, 0],
    [250, -1000, 0], [250, 1000, 0],
    [500, -1000, 0], [500, 1000, 0],
    [750, -1000, 0], [750, 1000, 0],
    [1000, -1000, 0], [1000, 1000, 0],
    # Y-axis lines
    [-1000, -1000, 0], [1000, -1000, 0],
    [-1000, -750, 0], [1000, -750, 0],
    [-1000, -500, 0], [1000, -500, 0],
    [-1000, -250, 0], [1000, -250, 0],
    [-1000, 0, 0], [1000, 0, 0],
    [-1000, 250, 0], [1000, 250, 0],
    [-1000, 500, 0], [1000, 500, 0],
    [-1000, 750, 0], [1000, 750, 0],
    [-1000, 1000, 0], [1000, 1000, 0],
], dtype=np.float32)


# ── HUD quad (2D, 0,0→1,1) ────────────────────────────────────
# Simple unit quad (0,0)→(1,1) in 2D for HUD rect rendering
HUD_QUAD = np.array([
    0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  # tri 1
    0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  # tri 2
], dtype=np.float32)

