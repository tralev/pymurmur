"""Vicsek 1995 mode — constant-speed angle coupling.

State is angle-only: each bird has a direction on the unit sphere.
Speed is constant (config.vicsek_velocity). Uses cKDTree.query_ball_point
for radius-based neighbour queries (same pattern as spatial.py).
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig


def vicsek_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Compute Vicsek angle-coupling forces — vectorised.

    u_new = normalize(eta * avg_dir + (1-eta) * noise)
    Then set velocity to u_new * v0 (constant speed).
    """
    active = flock.active
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return

    from scipy.spatial import cKDTree

    eta = config.vicsek_couplage
    noise_d = config.vicsek_diffusion
    radius = config.vicsek_radius_influence
    v0 = config.vicsek_velocity

    positions = flock.positions
    velocities = flock.velocities
    active_pos = positions[active_idx]

    if n_active < 2:
        # Single bird or zero: pure noise direction
        noise = flock.rng.normal(scale=noise_d, size=(n_active, 3)).astype(np.float32)
        noise_norms = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-10
        directions = noise / noise_norms
        velocities[active_idx] = directions * v0
        return

    # Build cKDTree on active positions (same pattern as spatial._query_neighbors)
    tree = cKDTree(active_pos)
    neighbour_dirs = np.zeros((n_active, 3), dtype=np.float32)

    for i, pos in enumerate(active_pos):
        nbrs = tree.query_ball_point(pos, radius)
        if len(nbrs) < 2:  # only self or nothing
            continue

        dirs = velocities[active_idx[nbrs]]
        norms = np.linalg.norm(dirs, axis=1)
        valid = norms > 1e-6
        if valid.sum() > 1:
            avg = np.mean(dirs[valid] / norms[valid, np.newaxis], axis=0)
            neighbour_dirs[i] = avg

    # Generate noise for all birds
    noise = np.random.normal(scale=noise_d, size=(n_active, 3)).astype(np.float32)
    noise_norms = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-10
    noise = noise / noise_norms

    # Blend: birds with neighbours use avg_dir, birds without use pure noise
    has_neighbours = np.linalg.norm(neighbour_dirs, axis=1) > 1e-6
    directions = noise.copy()  # default: pure noise

    if has_neighbours.any():
        nd = neighbour_dirs[has_neighbours]
        nd_norms = np.linalg.norm(nd, axis=1, keepdims=True)
        nd = nd / (nd_norms + 1e-10)
        directions[has_neighbours] = eta * nd + (1 - eta) * noise[has_neighbours]

    # Normalise and set constant-speed velocity
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10
    directions = directions / dir_norms
    velocities[active_idx] = directions * v0
