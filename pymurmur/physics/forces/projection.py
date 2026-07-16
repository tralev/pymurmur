"""Pearce 2014 projection mode — 3D spherical-cap occlusion + alignment.

Computes delta per bird via occlusion.py, then blends with neighbour
alignment. Uses topological sigma for neighbour selection.

I1.3: Uses spherical_cap_occlusion_batched — all observers in one call,
zero Python object allocations in the hot path.

P2.2: Wrapped in ProjectionMode(ForceMode) with @register("projection").
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ..occlusion import spherical_cap_occlusion_batched
from ..steric import steric_force  # P1.10: L0 atom import at module top (no cycle risk)
from ._mode import ForceFn, ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


@register("projection")
class ProjectionMode(ForceMode):
    """Pearce 2014 hybrid projection forces — occlusion + alignment."""

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
        """Compute Pearce hybrid projection forces.

        1. Batch-collect topological sigma neighbours via spatial index
        2. Batch-compute delta and Theta via spherical_cap_occlusion_batched
        3. Blend: v_desired = phi_p * delta + phi_a * align_dir
        4. Clamp steering and apply
        """
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        if n_active == 0:
            return

        sigma = config.sigma
        blind_cos = None
        if config.blind_deg > 0:
            blind_cos = np.cos(np.radians(config.blind_deg / 2))

        # --- Stage 1: collect neighbour indices for all active birds ---
        nbr_idx = _topological_neighbors_batch(positions, index, active_idx, sigma)
        # nbr_idx: (n_active, sigma) int32 — some rows may have -1 sentinels

        # Find birds with at least one valid neighbour
        has_nbrs = (nbr_idx >= 0).any(axis=1)  # (n_active,)
        if not has_nbrs.any():
            return

        # --- Stage 2: gather neighbour positions/velocities in batch ---
        # Clamp -1 sentinels to 0 for safe gather (will be masked out)
        safe_idx = np.maximum(nbr_idx, 0)  # (n_active, sigma)
        nbr_pos = positions[safe_idx]       # (n_active, sigma, 3)
        nbr_vel = velocities[safe_idx]      # (n_active, sigma, 3)

        # --- Stage 3: batched occlusion ---
        valid_mask = nbr_idx >= 0  # (n_active, sigma) — exclude -1 sentinels
        delta, visible_mask, theta = spherical_cap_occlusion_batched(
            positions[active_idx],
            velocities[active_idx],
            nbr_pos,
            nbr_vel,
            boid_size=config.boid_size,
            blind_cos=blind_cos,
            anisotropy=config.anisotropy if config.refinements else 1.0,
            valid_mask=valid_mask,
            n_jobs=config.parallel_workers,
        )
        # delta:     (n_active, 3)
        # visible_mask: (n_active, sigma) bool
        # theta:     (n_active,)

        last_theta[active_idx] = theta

        # --- Stage 4: alignment direction from visible neighbours ---
        # Zero out invisible neighbour velocities
        vis_vel = nbr_vel * visible_mask[:, :, np.newaxis]  # (n_active, sigma, 3)
        n_visible = visible_mask.sum(axis=1, keepdims=True)  # (n_active, 1)

        v_avg = vis_vel.sum(axis=1) / np.maximum(n_visible, 1)  # (n_active, 3)
        v_norm = np.linalg.norm(v_avg, axis=1, keepdims=True)

        align_dir = np.zeros((n_active, 3), dtype=np.float32)
        valid = (v_norm.squeeze() > 1e-6)
        align_dir[valid] = v_avg[valid] / v_norm[valid]

        # --- Stage 5: blend and steer ---
        v_desired = delta * config.phi_p + align_dir * config.phi_a  # (n_active, 3)
        steering = v_desired - velocities[active_idx]

        steer_mag = np.linalg.norm(steering, axis=1)
        too_strong = steer_mag > config.max_force
        if too_strong.any():
            steering[too_strong] = (
                steering[too_strong] / steer_mag[too_strong, np.newaxis] * config.max_force
            )

        accelerations[active_idx] += steering

        # --- Stage 6: steric repulsion (per-bird, lightweight) ---
        if config.refinements and config.steric > 0:
            for j, i in enumerate(active_idx):
                valid_nbrs = nbr_idx[j][nbr_idx[j] >= 0]
                if len(valid_nbrs) > 0:
                    accelerations[i] += steric_force(
                        positions[i], positions[valid_nbrs], config.steric,
                    )


# Backward compatibility alias — tests import projection_forces directly
projection_forces: ForceFn = ProjectionMode.compute  # type: ignore[assignment]
projection_forces.needs_index = True


def _topological_neighbors_batch(
    positions: np.ndarray,
    index: SpatialIndex | None,
    active_idx: np.ndarray,
    sigma: int,
) -> np.ndarray:
    """Collect sigma nearest neighbours for all active birds.

    Returns (n_active, sigma) int32 with -1 sentinels where fewer than
    sigma neighbours exist.
    """
    n_active = len(active_idx)

    if index is None or not index.ready:
        return np.full((n_active, sigma), -1, dtype=np.int32)

    result = np.full((n_active, sigma), -1, dtype=np.int32)
    for j, i in enumerate(active_idx):
        nbrs = index.query_knn(positions[i], k=sigma)
        result[j, :len(nbrs)] = nbrs

    return result
