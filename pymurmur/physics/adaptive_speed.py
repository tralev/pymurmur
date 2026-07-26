"""Neighbor-count adaptive speed law — pure function.

Level 0 — numpy only, zero pymurmur imports. A vectorized version of
`pymurmur/physics/forces/angle.py`'s deficit-based speed boost
(isolated boids fly faster), re-implemented here rather than importing
from angle.py so the NeighborAdaptiveSpeed extension can reuse the same
law across all 7 modes without touching angle.py's own tested code
path (angle.py keeps its own inline copy, unchanged).
"""

from __future__ import annotations

import numpy as np


def adaptive_speed_bonus(
    deficit: np.ndarray,
    mode: str,
    deficit_cap: np.ndarray | float,
    linear_scale: float = 5.0,
) -> np.ndarray:
    """Additive speed bonus for boids with fewer neighbors than desired.

    Mirrors angle.py's three speed_mode variants:
        "quadratic" -> min(deficit_cap, deficit^2)
        "softened"  -> min(deficit_cap, deficit^2 / 2)
        "linear" (default) -> deficit * linear_scale, uncapped

    deficit <= 0 (at or above the target neighbor count) contributes no
    bonus, matching angle.py's `if deficit > 0` gate.

    Args:
        deficit: (N,) target_neighbor_count - actual_neighbor_count
        mode: "linear" | "quadratic" | "softened"
        deficit_cap: cap applied to quadratic/softened only (scalar or
            per-boid array)
        linear_scale: multiplier for the "linear" mode

    Returns:
        (N,) float32 additive speed bonus, always >= 0
    """
    deficit = np.asarray(deficit, dtype=np.float32)
    positive = np.maximum(deficit, 0.0)
    if mode == "quadratic":
        bonus = np.minimum(deficit_cap, positive * positive)
    elif mode == "softened":
        bonus = np.minimum(deficit_cap, positive * positive / 2.0)
    else:  # "linear"
        bonus = positive * linear_scale
    return bonus.astype(np.float32)
