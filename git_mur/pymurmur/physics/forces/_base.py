"""Shared force primitives — mode-agnostic steering behaviours.

Level 0 — pure numpy functions operating on flat arrays.
These don't know about Reynolds, Vicsek, or Pearce. They just compute
force vectors from neighbour indices.
"""
from __future__ import annotations

import numpy as np


def separation_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Per-bird separation: push away from nearby neighbours.

    F_sep = Σ (p_i - p_j) / d²  for j in neighbours(i).
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)

    for i in np.where(active)[0]:
        nbrs = neighbor_idx[i]
        if len(nbrs) == 0:
            continue
        diffs = positions[nbrs] - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        close = dists > 1e-6
        if not close.any():
            continue
        diffs = diffs[close]
        dists = dists[close]
        force[i] = np.sum(-diffs / (dists[:, np.newaxis] ** 2), axis=0)

    return force


def alignment_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Per-bird alignment: steer toward average neighbour heading.

    F_align = normalize(avg(v_j)) - normalize(v_i).
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)

    for i in np.where(active)[0]:
        nbrs = neighbor_idx[i]
        if len(nbrs) == 0:
            continue
        avg_vel = np.mean(velocities[nbrs], axis=0)
        avg_norm = np.linalg.norm(avg_vel)
        if avg_norm > 1e-6:
            force[i] = avg_vel / avg_norm
        vi_norm = np.linalg.norm(velocities[i])
        if vi_norm > 1e-6:
            force[i] -= velocities[i] / vi_norm

    return force


def cohesion_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Per-bird cohesion: steer toward average neighbour position.

    F_coh = limit_length(avg(p_j) - p_i, 1).
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)

    for i in np.where(active)[0]:
        nbrs = neighbor_idx[i]
        if len(nbrs) == 0:
            continue
        center = np.mean(positions[nbrs], axis=0)
        to_center = center - positions[i]
        length = np.linalg.norm(to_center)
        if length > 1e-6:
            force[i] = to_center / min(length, 1.0)

    return force


def noise_force(
    n: int,
    scale: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random perturbation on the unit sphere.

    Returns (N, 3) float32. scale=0 produces all-zero array.
    """
    if scale == 0.0 or n == 0:
        return np.zeros((n, 3), dtype=np.float32)

    rng = rng or np.random.default_rng()
    pts = rng.normal(scale=scale, size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms
