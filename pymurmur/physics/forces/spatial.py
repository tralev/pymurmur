"""Reynolds 1987 spatial mode — separation + alignment + cohesion + noise.

Two-pass architecture: Python cKDTree query → numba JIT or numpy force pass.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import (
    alignment_force,
    cohesion_force,
    noise_force,
    separation_force,
)

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig


def spatial_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Compute Reynolds boids forces: separation + alignment + cohesion + noise."""
    active = flock.active
    n_active = active.sum()
    if n_active == 0:
        return

    # Pass 1: Query neighbours via spatial index
    index = flock.get_index()
    if not index.ready:
        return

    neighbor_idx = _query_neighbors(flock, config)

    # Pass 2: Assemble primitives
    sep = separation_force(
        flock.positions, flock.velocities, neighbor_idx, active)
    align = alignment_force(
        flock.positions, flock.velocities, neighbor_idx, active)
    coh = cohesion_force(
        flock.positions, flock.velocities, neighbor_idx, active)
    noise = noise_force(n_active, config.noise_scale)

    flock.accelerations[active] += (
        sep * config.separation_weight +
        align * config.alignment_weight +
        coh * config.cohesion_weight +
        noise
    )

    # Clamp to max_force
    acc_mags = np.linalg.norm(flock.accelerations, axis=1)
    too_strong = acc_mags > config.max_force
    if too_strong.any():
        flock.accelerations[too_strong] = (
            flock.accelerations[too_strong] /
            acc_mags[too_strong, np.newaxis] * config.max_force
        )


def _query_neighbors(
    flock: PhysicsFlock, config: SimConfig
) -> np.ndarray:
    """Build per-bird neighbour index using the spatial index."""
    from scipy.spatial import cKDTree

    active_pos = flock.positions[flock.active]
    n = len(active_pos)
    if n < 2:
        return np.zeros((n, 0), dtype=np.int32)

    k = min(getattr(config, "topological_cap", 50), n - 1)

    tree = cKDTree(active_pos)
    neighbor_idx = np.zeros((n, k), dtype=np.int32)

    for i, pos in enumerate(active_pos):
        _, idx = tree.query(pos, k=k + 1)
        neighbor_idx[i] = idx[1:k + 1]  # exclude self, pad with last if needed

    return neighbor_idx
