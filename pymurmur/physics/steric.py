"""Short-range 1/d² steric repulsion — Pearce SI-Appendix.

Level 0 — pure numpy. No project imports.
F_steric = Σ_{d < threshold} r̂ / d² · strength.
Clamped to max_force per P1.6.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Vec3


def steric_force(
    observer_pos: np.ndarray,
    neighbour_positions: np.ndarray,
    strength: float = 0.6,
    threshold: float = 10.0,
    max_force: float | None = None,
) -> Vec3:
    """Compute 1/d² repulsion force from nearby neighbours.

    Args:
        observer_pos: (3,) float32 — bird position.
        neighbour_positions: (M, 3) float32 — neighbour positions.
        strength: φ_s — repulsion strength (0 = off).
        threshold: maximum distance for repulsion effect.
        max_force: if set and |F| > max_force, F ← F·max_force/|F| (P1.6).

    Returns:
        (3,) float32 force vector.
    """
    if strength == 0.0 or len(neighbour_positions) == 0:
        return np.zeros(3, dtype=np.float32)

    diffs = neighbour_positions - observer_pos
    dists = np.linalg.norm(diffs, axis=1)

    close = dists < threshold
    if not close.any():
        return np.zeros(3, dtype=np.float32)

    close_diffs = diffs[close]
    close_dists = dists[close]

    # r̂ / d², summed — push AWAY from neighbour
    # diffs = nbr - obs, so -diffs = obs - nbr (away direction)
    dirs = -close_diffs / (close_dists[:, np.newaxis] + 1e-10)
    forces = dirs / (close_dists[:, np.newaxis] ** 2)

    result = (np.sum(forces, axis=0) * strength).astype(np.float32)

    # P1.6: Clamp to max_force
    if max_force is not None:
        mag = float(np.linalg.norm(result))
        if mag > max_force and mag > 1e-12:
            result *= (max_force / mag)

    return result
