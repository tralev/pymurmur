"""Theme palettes (4 monochrome themes) + P8.5 material tables for 3D
rendering.

Level 2 — static data, no project imports.

Split out of shaders.py (file-size split) — GLSL shader program strings
and mesh vertex/index data stay in the original / moved to
shaders_meshes.py.
"""

from __future__ import annotations

# ── Theme palettes (4 monochrome) + P8.5 material tables ──────────────────
THEMES: dict[str, dict[str, tuple[float, float, float]]] = {
    "ink": {
        "ambient": (0.02, 0.04, 0.10),
        "diffuse": (0.06, 0.12, 0.40),
        "slow": (0.1, 0.2, 0.5),
        "fast": (0.4, 0.8, 1.0),
        "spec": (1.0, 1.0, 1.0),
        "clear": (0.05, 0.05, 0.1),
        "trail": (0.3, 0.6, 0.9),
        "paper": (0.15, 0.25, 0.55),
        "ink": (0.02, 0.04, 0.12),
    },
    "inverse": {
        "ambient": (0.25, 0.23, 0.20),
        "diffuse": (0.30, 0.28, 0.22),
        "slow": (0.8, 0.75, 0.7),
        "fast": (0.2, 0.15, 0.1),
        "spec": (0.1, 0.1, 0.1),
        "clear": (0.9, 0.88, 0.85),
        "trail": (0.4, 0.35, 0.3),
        "paper": (0.85, 0.82, 0.78),
        "ink": (0.05, 0.05, 0.06),
    },
    "paper": {
        "ambient": (0.18, 0.16, 0.12),
        "diffuse": (0.35, 0.28, 0.16),
        "slow": (0.25, 0.2, 0.15),
        "fast": (0.45, 0.35, 0.2),
        "spec": (1.0, 0.95, 0.8),
        "clear": (0.95, 0.92, 0.85),
        "trail": (0.5, 0.4, 0.25),
        "paper": (0.92, 0.88, 0.82),
        "ink": (0.15, 0.12, 0.10),
    },
    "graphite": {
        "ambient": (0.08, 0.08, 0.08),
        "diffuse": (0.45, 0.45, 0.45),
        "slow": (0.15, 0.15, 0.15),
        "fast": (0.85, 0.85, 0.85),
        "spec": (0.3, 0.3, 0.3),
        "clear": (0.1, 0.1, 0.1),
        "trail": (0.5, 0.5, 0.5),
        "paper": (0.25, 0.25, 0.25),
        "ink": (0.05, 0.05, 0.05),
    },
    # S4.6: heading-hue debug theme — same lighting as "ink"; per-bird
    # hue source (velocity azimuth vs seed) is a renderer-level decision,
    # not a lighting decision.
    "heading": {
        "ambient": (0.02, 0.04, 0.10),
        "diffuse": (0.06, 0.12, 0.40),
        "slow": (0.1, 0.2, 0.5),
        "fast": (0.4, 0.8, 1.0),
        "spec": (1.0, 1.0, 1.0),
        "clear": (0.05, 0.05, 0.1),
        "trail": (0.3, 0.6, 0.9),
        "paper": (0.15, 0.25, 0.55),
        "ink": (0.02, 0.04, 0.12),
    },
}

