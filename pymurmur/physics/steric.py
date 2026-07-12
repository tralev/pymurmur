"""Short-range 1/d² steric repulsion — Pearce SI-Appendix.

Level 0 — pure numpy. No project imports.
F_steric = Σ_{d < threshold} r̂ / d² · strength.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Vec3


def steric_force(
    observer_pos: np.ndarray,
    neighbour_positions: np.ndarray,
    strength: float = 0.6,
    threshold: float = 10.0,
) -> Vec3:
    """Compute 1/d² repulsion force from nearby neighbours.

    Args:
        observer_pos: (3,) float32 — bird position.
        neighbour_positions: (M, 3) float32 — neighbour positions.
        strength: φ_s — repulsion strength (0 = off).
        threshold: maximum distance for repulsion effect.

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

    return (np.sum(forces, axis=0) * strength).astype(np.float32)
