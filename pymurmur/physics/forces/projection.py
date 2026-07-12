"""Pearce 2014 projection mode — 3D spherical-cap occlusion + alignment.

Computes δ̂ per bird via occlusion.py, then blends with neighbour
alignment. Uses topological σ for neighbour selection.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ..occlusion import spherical_cap_occlusion_soa

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig


def projection_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Compute Pearce hybrid projection forces.

    For each bird:
    1. Find topological σ neighbours via spatial index
    2. Compute δ̂ and Θ via spherical-cap occlusion
    3. Blend: v_desired = φp·δ̂ + φa·⟨v̂⟩_σ + φn·η̂
    4. Return clamp(v_desired − v, max_force)
    """
    active = flock.active
    n_active = active.sum()
    if n_active == 0:
        return

    sigma = config.sigma
    blind_cos = None
    if config.blind_deg > 0:
        blind_cos = np.cos(np.radians(config.blind_deg / 2))

    # Per-bird loop — occlusion is inherently per-observer
    for i in np.where(active)[0]:
        # Find σ nearest neighbours
        nbr_idx = _topological_neighbors(flock, i, sigma)
        if len(nbr_idx) == 0:
            continue

        nbr_pos = flock.positions[nbr_idx]
        nbr_vel = flock.velocities[nbr_idx]

        # Compute occlusion
        delta, visible_idx, theta = spherical_cap_occlusion_soa(
            flock.positions[i], flock.velocities[i],
            nbr_pos, nbr_vel,
            boid_size=config.boid_size,
            blind_cos=blind_cos,
            anisotropy=config.anisotropy if config.refinements else 1.0,
        )
        flock.last_theta[i] = theta

        # Alignment: average heading of visible neighbours
        align_dir = np.zeros(3, dtype=np.float32)
        if len(visible_idx) > 0:
            visible_vel = nbr_vel[visible_idx]
            v_avg = np.mean(visible_vel, axis=0)
            v_norm = np.linalg.norm(v_avg)
            if v_norm > 1e-6:
                align_dir = v_avg / v_norm

        # Blend desired velocity
        v_desired = (
            delta * config.phi_p +
            align_dir * config.phi_a
        )

        # Steering = desired − current, clamped
        steering = v_desired - flock.velocities[i]
        steer_mag = np.linalg.norm(steering)
        if steer_mag > config.max_force:
            steering = steering / steer_mag * config.max_force

        flock.accelerations[i] += steering

        # Steric repulsion
        if config.refinements and config.steric > 0:
            from ..steric import steric_force
            flock.accelerations[i] += steric_force(
                flock.positions[i], nbr_pos, config.steric)


def _topological_neighbors(
    flock: PhysicsFlock, bird_idx: int, sigma: int
) -> np.ndarray:
    """Get σ nearest neighbours for a single bird via spatial index."""
    index = flock.get_index()
    pos = flock.positions[bird_idx]

    from ..flock import KDTreeIndex, SpatialHashGrid

    if isinstance(index, KDTreeIndex) and index.ready:
        return index.query_knn(pos, k=sigma)
    elif isinstance(index, SpatialHashGrid) and index.ready:
        return index.query_knn(pos, k=sigma)
    return np.array([], dtype=np.int32)
