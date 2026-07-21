"""S4.4a — Mesh registry with render-mode recommendation and theme materials.

Defines mesh vertex data for ellipsoid, cone, arrow, and points alongside
the existing tetra, winged, and impostor shapes.  Each entry includes the
vertex buffer, index buffer, ModernGL vertex-array format string, shader
attribute names, and OpenGL primitive mode.

``recommend_render_mode(n)`` chooses a mesh based on bird count:
    ≤10K  → "winged"  (instanced geometry — full Blinn-Phong)
    ≤60K  → "impostor" (camera-facing quads with disc fragments)
    >60K  → "points"   (GL_POINTS — fastest for very large N)

``MATERIAL_REGISTRY`` provides per-theme ambient / diffuse / specular
triplets keyed by ``cfg.viz.theme`` (ink | inverse | paper | graphite).
These are the single source of truth — the legacy ``THEMES`` dict in
shaders.py delegates to them during migration and will be retired.
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Mesh vertex data
# ═══════════════════════════════════════════════════════════════════════
# All meshes point in the +Z direction (forward) so that the per-bird
# LookAt rotation matrix in the vertex shader orients them along the
# velocity vector.

# ── Ellipsoid (8-vertex stretched octahedron, 8 triangles) ──────
# Stretched 2:1 along Z (forward axis) so birds look elongated in
# their direction of travel.  Radius 0.5 in XY, ±1.5 in Z.

ELLIPSOID_VERTICES = np.array([
    [0.0, 0.0, 1.5],     # 0: front tip
    [0.5, 0.0, 0.0],     # 1: +X equator
    [0.0, 0.5, 0.0],     # 2: +Y equator
    [-0.5, 0.0, 0.0],    # 3: −X equator
    [0.0, -0.5, 0.0],    # 4: −Y equator
    [0.0, 0.0, -1.5],    # 5: rear tip
], dtype=np.float32)

ELLIPSOID_INDICES = np.array([
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],   # front half
    [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],   # rear half
], dtype=np.uint32)


# ── Cone (7-vertex, 8 triangles) ─────────────────────────────────
# Tip at +Z, hexagonal base at −Z.  Radius 0.4 at base.

CONE_VERTICES = np.array([
    [0.0, 0.0, 1.5],      # 0: cone tip
    [0.4, 0.0, -0.5],      # 1: base ring vertex 0
    [0.2, 0.346, -0.5],    # 2: base ring vertex 1
    [-0.2, 0.346, -0.5],   # 3: base ring vertex 2
    [-0.4, 0.0, -0.5],     # 4: base ring vertex 3
    [-0.2, -0.346, -0.5],  # 5: base ring vertex 4
    [0.2, -0.346, -0.5],   # 6: base ring vertex 5
], dtype=np.float32)

CONE_INDICES = np.array([
    [0, 1, 2], [0, 2, 3], [0, 3, 4],   # tip to base ring
    [0, 4, 5], [0, 5, 6], [0, 6, 1],
    [1, 6, 2], [2, 6, 5], [2, 5, 4],   # base cap
    [2, 4, 3],
], dtype=np.uint32)


# ── Arrow (10-vertex, 12 triangles) ──────────────────────────────
# Arrowhead at +Z, shaft going back.  Clean directional cue.

ARROW_VERTICES = np.array([
    [0.0, 0.0, 2.0],       # 0: arrow tip
    [0.4, 0.0, 1.0],       # 1: head right
    [0.0, 0.3, 1.0],       # 2: head top
    [-0.4, 0.0, 1.0],      # 3: head left
    [0.0, -0.3, 1.0],      # 4: head bottom
    [0.15, 0.0, -1.5],     # 5: shaft right-rear
    [0.0, 0.15, -1.5],     # 6: shaft top-rear
    [-0.15, 0.0, -1.5],    # 7: shaft left-rear
    [0.0, -0.15, -1.5],    # 8: shaft bottom-rear
    [0.0, 0.0, -0.5],      # 9: shaft centre (connection point)
], dtype=np.float32)

ARROW_INDICES = np.array([
    # Arrowhead (tip to each edge)
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
    # Head base ring
    [1, 2, 9], [2, 3, 9], [3, 4, 9], [4, 1, 9],
    # Shaft (base ring to rear)
    [1, 5, 6], [1, 6, 2],   # right-top face
    [2, 6, 7], [2, 7, 3],   # top-left face
    [3, 7, 8], [3, 8, 4],   # left-bottom face
    [4, 8, 5], [4, 5, 1],   # bottom-right face
    # Shaft rear cap
    [5, 8, 7], [5, 7, 6],
], dtype=np.uint32)


# ── Points — no mesh geometry needed (GL_POINTS primitive) ───────
# The vertex shader emits gl_PointSize; the fragment shader draws a
# screen-space disc.  The instance VBO provides per-bird centres.

POINTS_VERTICES = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
POINTS_INDICES = np.array([[0]], dtype=np.uint32)


# ═══════════════════════════════════════════════════════════════════════
# Mesh registry
# ═══════════════════════════════════════════════════════════════════════

# OpenGL primitive constants — deferred import so the module is
# importable without a GPU context.
_GL_TRIANGLES = 4       # GL_TRIANGLES
_GL_POINTS = 0           # GL_POINTS


def _make_entry(
    vertices: np.ndarray,
    indices: np.ndarray,
    vfmt: str,
    attrs: tuple[str, ...],
    primitive: int = _GL_TRIANGLES,
) -> dict:
    return {
        "vertices": vertices,
        "indices": indices,
        "vertex_format": vfmt,
        "attributes": attrs,
        "primitive": primitive,
    }


MESH_REGISTRY: dict[str, dict] = {
    "tetra": _make_entry(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.943, -0.333],
                   [-0.816, -0.471, -0.333], [0.816, -0.471, -0.333]],
                  dtype=np.float32),
        np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.uint32),
        "3f", ("in_position",),
    ),
    "winged": _make_entry(
        np.array([
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.25, -0.2, 0.0],
            [0.0, -0.15, -0.2, 0.0], [0.65, 0.05, 0.0, 0.5],
            [-0.65, 0.05, 0.0, 0.5], [0.0, 0.05, -0.8, 0.0],
            [0.0, 0.0, -0.4, 0.25],
        ], dtype=np.float32),
        np.array([[0, 1, 3], [0, 3, 2], [0, 2, 4], [0, 4, 1],
                   [1, 2, 5], [5, 6, 1]], dtype=np.uint32),
        "3f 1f", ("in_position", "in_flap_weight"),
    ),
    "impostor": _make_entry(
        np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                  dtype=np.float32),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        "2f", ("in_quad_pos",),
    ),
    "ellipsoid": _make_entry(
        ELLIPSOID_VERTICES, ELLIPSOID_INDICES,
        "3f", ("in_position",),
    ),
    "cone": _make_entry(
        CONE_VERTICES, CONE_INDICES,
        "3f", ("in_position",),
    ),
    "arrow": _make_entry(
        ARROW_VERTICES, ARROW_INDICES,
        "3f", ("in_position",),
    ),
    "points": _make_entry(
        POINTS_VERTICES, POINTS_INDICES,
        "3f", ("in_position",),
        primitive=_GL_POINTS,
    ),
}

VALID_MESH_NAMES: frozenset[str] = frozenset(MESH_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════
# Render-mode recommendation
# ═══════════════════════════════════════════════════════════════════════

def recommend_render_mode(n_active: int, *,
                          instanced_threshold: int = 10_000,
                          impostor_threshold: int = 60_000,
                          ) -> str:
    """Pure function: choose a mesh based on active bird count.

    Args:
        n_active: number of active birds.
        instanced_threshold: at or below this count, use instanced
            geometry (winged mesh with full Blinn-Phong lighting).
        impostor_threshold: at or below this count, use camera-facing
            impostor quads (sphere disc fragments).  Above it, fall
            back to GL_POINTS.

    Returns:
        One of ``"winged"``, ``"impostor"``, or ``"points"``.
        These names match keys in :data:`MESH_REGISTRY`.
    """
    if n_active <= instanced_threshold:
        return "winged"
    if n_active <= impostor_threshold:
        return "impostor"
    return "points"


# ═══════════════════════════════════════════════════════════════════════
# Per-theme material registry
# ═══════════════════════════════════════════════════════════════════════
# Each theme defines ambient (shadowed), diffuse (lit), and specular
# (highlight) triplet, plus extended colours used by specific render
# paths (trails, impostor paper/ink, clear colour).

_MATERIAL_ENTRY_KEYS = frozenset({
    "ambient", "diffuse", "spec",
    "slow", "fast", "clear", "trail", "paper", "ink",
})

MATERIAL_REGISTRY: dict[str, dict[str, tuple[float, float, float]]] = {
    "ink": {
        "ambient":  (0.02, 0.04, 0.10),
        "diffuse":  (0.06, 0.12, 0.40),
        "spec": (1.00, 1.00, 1.00),
        "slow":     (0.10, 0.20, 0.50),
        "fast":     (0.40, 0.80, 1.00),
        "clear":    (0.05, 0.05, 0.10),
        "trail":    (0.30, 0.60, 0.90),
        "paper":    (0.12, 0.10, 0.10),
        "ink":      (0.02, 0.02, 0.06),
    },
    "inverse": {
        "ambient":  (0.25, 0.23, 0.20),
        "diffuse":  (0.80, 0.75, 0.65),
        "spec": (0.90, 0.90, 0.90),
        "slow":     (0.70, 0.65, 0.55),
        "fast":     (1.00, 0.95, 0.85),
        "clear":    (0.92, 0.90, 0.88),
        "trail":    (0.60, 0.50, 0.40),
        "paper":    (0.95, 0.93, 0.90),
        "ink":      (0.15, 0.12, 0.08),
    },
    "paper": {
        "ambient":  (0.18, 0.16, 0.12),
        "diffuse":  (0.55, 0.50, 0.40),
        "spec": (0.30, 0.30, 0.30),
        "slow":     (0.50, 0.45, 0.35),
        "fast":     (0.70, 0.65, 0.55),
        "clear":    (0.85, 0.83, 0.80),
        "trail":    (0.40, 0.35, 0.25),
        "paper":    (0.88, 0.85, 0.82),
        "ink":      (0.25, 0.20, 0.15),
    },
    "graphite": {
        "ambient":  (0.08, 0.08, 0.08),
        "diffuse":  (0.35, 0.35, 0.35),
        "spec": (0.60, 0.60, 0.60),
        "slow":     (0.30, 0.30, 0.30),
        "fast":     (0.50, 0.50, 0.50),
        "clear":    (0.15, 0.15, 0.15),
        "trail":    (0.25, 0.25, 0.25),
        "paper":    (0.20, 0.20, 0.20),
        "ink":      (0.05, 0.05, 0.05),
    },
}

VALID_THEMES: frozenset[str] = frozenset(MATERIAL_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════
# Public helpers
# ═══════════════════════════════════════════════════════════════════════

def get_mesh(name: str) -> dict:
    """Return the mesh entry for *name*, falling back to ``"tetra"``."""
    return MESH_REGISTRY.get(name, MESH_REGISTRY["tetra"])


def get_theme_materials(theme: str) -> dict[str, tuple[float, float, float]]:
    """Return the material dict for *theme*, falling back to ``"ink"``."""
    return MATERIAL_REGISTRY.get(theme, MATERIAL_REGISTRY["ink"])


def resolve_bird_mesh(bird_mesh: str, n_active: int) -> str:
    """Resolve the user-facing ``bird_mesh`` config value to a registry key.

    - ``"auto"`` → delegates to :func:`recommend_render_mode`.
    - ``"sphere"`` / ``"impostor"`` → ``"impostor"``.
    - ``"tetra"`` / ``"winged"`` / ``"ellipsoid"`` / ``"cone"`` /
      ``"arrow"`` / ``"points"`` → pass through.
    - Unknown → ``"tetra"`` (safe fallback).

    Args:
        bird_mesh: value of ``config.viz.bird_mesh``.
        n_active: current active bird count (used when ``bird_mesh="auto"``).

    Returns:
        A key present in :data:`MESH_REGISTRY`.
    """
    if bird_mesh == "auto":
        return recommend_render_mode(n_active)
    if bird_mesh == "sphere":
        return "impostor"
    if bird_mesh in VALID_MESH_NAMES:
        return bird_mesh
    return "tetra"
