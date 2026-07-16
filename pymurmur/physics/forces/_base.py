"""Shared force primitives — mode-agnostic steering behaviours.

Level 0 — pure numpy functions operating on flat arrays.
These don't know about Reynolds, Vicsek, or Pearce. They just compute
force vectors from neighbour indices.

Primitives use a vectorised gather+reduce path when neighbor_idx is a
dense 2D int array (production).  For ragged object arrays (test
fixtures) they fall back to the per-bird loop.

P2.10: ForceTerm dataclass + composeForces reducer — every force
contribution is a named, typed, runtime-togglable unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...physics.flock import PhysicsFlock
    from ..extensions._base import StepContext
    from ...core.config import SimConfig


# ── P2.10: ForceTerm composition infrastructure ───────────────────

@dataclass
class ForceTerm:
    """A named, typed, runtime-togglable force contribution (P2.10).

    Each term is a pure function (no side effects, no internal state)
    that receives the flock, step context, and config, and returns an
    (N, 3) float32 force array.

    Usage::

        shell = ForceTerm("shell", gain=1.0, fn=shell_term)
        terms = [shell, expand, target]
        F_total = composeForces(flock, ctx, cfg, terms)
    """

    name: str
    """Human-readable identifier, e.g. 'shell', 'tangential', 'drag'."""

    enabled: bool = True
    """Runtime toggle — flip to False mid-run to silence this term."""

    gain: float = 1.0
    """Per-term intensity multiplier applied before summation."""

    fn: Callable[..., np.ndarray] | None = None
    """Force function: (flock, ctx, cfg) → (N, 3) float32 array."""


def composeForces(
    flock: PhysicsFlock,
    ctx: StepContext,
    config: SimConfig,
    terms: list[ForceTerm],
) -> np.ndarray:
    """Linearly sum enabled force terms (P2.10).

    Iterates over *terms*, multiplies each enabled term's output by
    its *gain*, and accumulates into a single (N, 3) float32 array.

    Args:
        flock: PhysicsFlock instance
        ctx: Per-frame StepContext (frame, dt, rng, center, config)
        config: SimConfig
        terms: ordered list of ForceTerm descriptors

    Returns:
        (N_capacity, 3) float32 — total force per bird
    """
    N = len(flock.positions)
    total = np.zeros((N, 3), dtype=np.float32)
    for term in terms:
        if term.enabled and term.fn is not None:
            total += term.gain * term.fn(flock, ctx, config)
    return total


def _is_ragged(neighbor_idx: np.ndarray) -> bool:
    """True if neighbor_idx is an object array (ragged), not dense 2D."""
    return neighbor_idx.dtype == np.dtype('object')


def separation_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Separation: push away from nearby neighbours.

    F_sep[i] = Σ_j −d_ij / |d_ij|²  for j in neighbours(i).
    Magnitude falls as 1/|d_ij| (d_ij is the raw difference vector, not unit).
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return force

    if _is_ragged(neighbor_idx):
        # Ragged object array — per-bird fallback
        for i in active_idx:
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

    # Dense 2D int array — vectorised gather+reduce
    k = neighbor_idx.shape[1] if neighbor_idx.ndim == 2 else 0
    if k == 0:
        return force

    nbr_idx = neighbor_idx[active_idx]           # (n_active, k)
    p_i = positions[active_idx]                  # (n_active, 3)
    p_j = positions[nbr_idx]                     # (n_active, k, 3)

    diffs = p_j - p_i[:, np.newaxis, :]           # (n_active, k, 3)
    dists = np.linalg.norm(diffs, axis=2)         # (n_active, k)

    close = dists > 1e-6
    dists_safe = np.where(close, dists, 1.0)
    contrib = -diffs / (dists_safe[:, :, np.newaxis] ** 2)
    contrib[~close] = 0.0

    force[active_idx] = np.sum(contrib, axis=1).astype(np.float32)
    return force


def alignment_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Alignment: steer toward average neighbour heading.

    F_align[i] = û_avg(j) − û_i.
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return force

    if _is_ragged(neighbor_idx):
        # Ragged object array — per-bird fallback
        for i in active_idx:
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

    # Dense 2D int array — vectorised gather+reduce
    k = neighbor_idx.shape[1] if neighbor_idx.ndim == 2 else 0
    if k == 0:
        return force

    nbr_idx = neighbor_idx[active_idx]           # (n_active, k)
    v_j = velocities[nbr_idx]                    # (n_active, k, 3)

    avg_vel = np.mean(v_j, axis=1)               # (n_active, 3)
    avg_norms = np.linalg.norm(avg_vel, axis=1)
    valid_avg = avg_norms > 1e-6
    if valid_avg.any():
        force[active_idx[valid_avg]] = (
            avg_vel[valid_avg] / avg_norms[valid_avg, np.newaxis]
        )

    v_i = velocities[active_idx]
    vi_norms = np.linalg.norm(v_i, axis=1)
    valid_self = vi_norms > 1e-6
    if valid_self.any():
        force[active_idx[valid_self]] -= (
            v_i[valid_self] / vi_norms[valid_self, np.newaxis]
        )

    return force


def cohesion_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Cohesion: steer toward average neighbour position.

    F_coh[i] = û(p̄_j − p_i)  — bounded unit vector (P1.7 fix).
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return force

    if _is_ragged(neighbor_idx):
        # Ragged object array — per-bird fallback
        for i in active_idx:
            nbrs = neighbor_idx[i]
            if len(nbrs) == 0:
                continue
            center = np.mean(positions[nbrs], axis=0)
            to_center = center - positions[i]
            length = np.linalg.norm(to_center)
            if length < 1e-10:
                continue
            force[i] = to_center / length
        return force

    # Dense 2D int array — vectorised gather+reduce
    k = neighbor_idx.shape[1] if neighbor_idx.ndim == 2 else 0
    if k == 0:
        return force

    nbr_idx = neighbor_idx[active_idx]           # (n_active, k)
    p_i = positions[active_idx]                  # (n_active, 3)
    p_j = positions[nbr_idx]                     # (n_active, k, 3)

    center = np.mean(p_j, axis=1)                # (n_active, 3)
    to_center = center - p_i                     # (n_active, 3)
    lengths = np.linalg.norm(to_center, axis=1)  # (n_active,)

    nonzero = lengths >= 1e-10
    if nonzero.any():
        force[active_idx[nonzero]] = (
            to_center[nonzero] / lengths[nonzero, np.newaxis]
        )

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
