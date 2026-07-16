"""Reynolds 1987 spatial mode — separation + alignment + cohesion + noise.

Two-pass architecture: Python cKDTree query → numba JIT or numpy force pass.

P2.2: Wrapped in SpatialMode(ForceMode) with @register("spatial").
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
from ._mode import ForceFn, ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


@register("spatial")
class SpatialMode(ForceMode):
    """Reynolds 1987 boids — separation + alignment + cohesion + noise."""

    needs_index = True

    @staticmethod
    def compute(
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        active: np.ndarray,
        index: SpatialIndex | None,
        rng: np.random.Generator,
        last_theta: np.ndarray,
        config: SimConfig,
    ) -> None:
        """Compute Reynolds boids forces: separation + alignment + cohesion + noise."""
        n_active = active.sum()
        if n_active == 0:
            return

        # Pass 1: Query neighbours via spatial index
        if index is None or not index.ready:
            return

        neighbor_idx = _query_neighbors(positions, active, index, config)

        # Pass 2: Assemble primitives
        sep = separation_force(
            positions, velocities, neighbor_idx, active)
        align = alignment_force(
            positions, velocities, neighbor_idx, active)
        coh = cohesion_force(
            positions, velocities, neighbor_idx, active)
        noise = noise_force(n_active, config.noise_scale, rng)
        # Scatter noise into (N_capacity, 3) — all force shapes match now
        noise_full = np.zeros((len(positions), 3), dtype=np.float32)
        noise_full[active] = noise

        # Add to full array (inactive rows of primitives are zero)
        accelerations += (
            sep * config.separation_weight +
            align * config.alignment_weight +
            coh * config.cohesion_weight +
            noise_full
        )

        # Clamp to max_force
        acc_mags = np.linalg.norm(accelerations, axis=1)
        too_strong = acc_mags > config.max_force
        if too_strong.any():
            accelerations[too_strong] = (
                accelerations[too_strong] /
                acc_mags[too_strong, np.newaxis] * config.max_force
            )


# Backward compatibility alias — tests import spatial_forces directly
spatial_forces: ForceFn = SpatialMode.compute  # type: ignore[assignment]
spatial_forces.needs_index = True


def _query_neighbors(
    positions: np.ndarray,
    active: np.ndarray,
    index: SpatialIndex,
    config: SimConfig,
) -> np.ndarray:
    """Build per-bird neighbour index using the shared spatial index.

    Returns (N_capacity, k) int32 array indexed by global bird index.
    Inactive rows are zero-filled. This shape contract ensures force
    primitives can access rows via global indices without remapping.

    Falls back to a private cKDTree for exact k-NN when the shared index
    is SpatialHashGrid (whose 27-cell grid can miss neighbours outside
    the neighbourhood for birds near domain edges or at low density).
    """
    from scipy.spatial import cKDTree

    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    N = len(positions)

    if n_active < 2:
        return np.zeros((N, 0), dtype=np.int32)

    k = min(getattr(config, "topological_cap", 50), n_active - 1)

    neighbor_idx = np.zeros((N, k), dtype=np.int32)

    # Use shared index for KDTreeIndex (exact k-NN), fall back for SpatialHashGrid
    tree = getattr(index, 'tree', None)
    if tree is not None:
        # Shared KDTreeIndex — query_knn returns global indices (I3.3)
        for j, global_i in enumerate(active_idx):
            pos = positions[global_i]
            nbrs = index.query_knn(pos, k)
            if len(nbrs) > 0:
                neighbor_idx[global_i, :len(nbrs)] = nbrs[:k]
        return neighbor_idx

    # Fallback: private cKDTree for exact k-NN (SpatialHashGrid path).
    # cKDTree returns compacted indices — map to global via active_idx.
    active_pos = positions[active_idx]
    tree = cKDTree(active_pos)
    for j, global_i in enumerate(active_idx):
        _, idx = tree.query(active_pos[j], k=k + 1)
        compacted_nbrs = idx[1:k + 1]  # compacted (0 to n_active-1)
        if len(compacted_nbrs) > 0:
            neighbor_idx[global_i, :len(compacted_nbrs)] = active_idx[compacted_nbrs]

    return neighbor_idx
