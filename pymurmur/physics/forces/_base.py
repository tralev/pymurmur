"""Shared force primitives — mode-agnostic steering behaviours.

Level 0 — pure numpy functions operating on flat arrays.
These don't know about Reynolds, Vicsek, or Pearce. They just compute
force vectors from neighbour indices.

Primitives use a vectorised gather+reduce path when neighbor_idx is a
dense 2D int array (production).  For ragged object arrays (test
fixtures) they fall back to the per-bird loop.

P2.10/S2.A5: ForceTerm dataclass + composeForces reducer — every force
contribution is a named, typed, runtime-togglable unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# ── P2.10/S2.A5: ForceTerm composition infrastructure ─────────────

@dataclass
class ForceTerm:
    """A named, typed, runtime-togglable force contribution (P2.10/S2.A5).

    Each term is a pure function (no side effects, no internal state)
    that receives a mode-defined per-frame context object and returns
    an (N, 3) float32 force array. composeForces() doesn't inspect the
    context — it's whatever shape the term functions in a given mode's
    term table expect (e.g. field.py's FieldTermContext), which keeps
    this infrastructure reusable across modes without hardcoding to any
    one mode's data (the original P2.10 design hardcoded (flock, ctx,
    cfg), which no mode ever actually used — see S2.A5's C4 audit).

    Usage::

        shell = ForceTerm("shell", gain=1.0, fn=shell_term)
        terms = [shell, expand, target]
        F_total = composeForces(ctx, terms, n=N)
    """

    name: str
    """Human-readable identifier, e.g. 'shell', 'tangential', 'drag'."""

    enabled: bool = True
    """Runtime toggle — flip to False mid-run to silence this term."""

    gain: float = 1.0
    """Per-term intensity multiplier applied before summation."""

    fn: Callable[[Any], np.ndarray] | None = None
    """Force function: (ctx) → (N, 3) float32 array."""


def composeForces(
    ctx: Any,
    terms: list[ForceTerm],
    n: int,
) -> np.ndarray:
    """Linearly sum enabled force terms (P2.10/S2.A5).

    Iterates over *terms*, multiplies each enabled term's output by
    its *gain*, and accumulates into a single (n, 3) float32 array.

    Args:
        ctx: per-frame context object, passed through unchanged to each
             term's fn — composeForces itself is mode-agnostic
        terms: ordered list of ForceTerm descriptors
        n: bird capacity (output shape (n, 3)) — needed so an empty or
           fully-disabled terms list still returns a correctly-shaped
           zero array

    Returns:
        (n, 3) float32 — total force per bird
    """
    total = np.zeros((n, 3), dtype=np.float32)
    for term in terms:
        if term.enabled and term.fn is not None:
            total += term.gain * term.fn(ctx)
    return total


def _is_ragged(neighbor_idx: np.ndarray) -> bool:
    """True if neighbor_idx is an object array (ragged), not dense 2D."""
    return neighbor_idx.dtype == np.dtype('object')


def separation_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
    kernel: str = "sum",
) -> np.ndarray:
    """Separation: push away from nearby neighbours.

    S1.5: Kernel selector — how neighbour contributions are combined.

    kernel="sum"  → F_sep[i] = Σ_j −d_ij / |d_ij|²  (Reynolds default)
    kernel="mean" → F_sep[i] = (1/k) Σ_j −d_ij / |d_ij|²  (density-invariant)
    kernel="unit" → F_sep[i] = Σ_j −û(d_ij)  (unit direction, distance-independent)

    Magnitude falls as 1/|d_ij| for "sum"/"mean" (d_ij is the raw difference vector,
    not unit).  For "unit", all neighbours push with equal strength.
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return force

    if kernel == "unit":
        # Unit-direction kernel: Σ −û(d_ij) — all neighbours push equally
        if _is_ragged(neighbor_idx):
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
                force[i] = np.sum(-diffs / dists[:, np.newaxis], axis=0)
            return force

        k = neighbor_idx.shape[1] if neighbor_idx.ndim == 2 else 0
        if k == 0:
            return force
        nbr_idx = neighbor_idx[active_idx]
        p_i = positions[active_idx]
        p_j = positions[nbr_idx]
        diffs = p_j - p_i[:, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)
        close = dists > 1e-6
        dists_safe = np.where(close, dists, 1.0)
        contrib = -diffs / dists_safe[:, :, np.newaxis]  # unit vectors
        contrib[~close] = 0.0
        force[active_idx] = np.sum(contrib, axis=1).astype(np.float32)
        return force

    # Shared dense path for "sum" and "mean" kernels
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
            # S1.5: Σ r̂/d² = unit-direction / squared-distance
            contrib = -diffs / (dists[:, np.newaxis] ** 3)
            force[i] = np.sum(contrib, axis=0)
            if kernel == "mean" and len(contrib) > 0:
                force[i] /= len(contrib)
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
    # S1.5: Σ r̂/d² = Σ (−Δ/|Δ|) / d² = Σ −Δ / |Δ|³
    # was −Δ/|Δ|² (1/d magnitude); correct is unit-direction / d².
    contrib = -diffs / (dists_safe[:, :, np.newaxis] ** 3)
    contrib[~close] = 0.0

    force[active_idx] = np.sum(contrib, axis=1).astype(np.float32)

    if kernel == "mean":
        # Divide by neighbour count per bird (density-invariant)
        n_neighbors = close.sum(axis=1).astype(np.float32)  # (n_active,)
        n_neighbors[n_neighbors == 0] = 1.0
        force[active_idx] /= n_neighbors[:, np.newaxis]

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
            # S1.5: Reynolds steering — normalize(v̄ − v_i)
            avg_vel = np.mean(velocities[nbrs], axis=0)
            steering = avg_vel - velocities[i]
            s_norm = np.linalg.norm(steering)
            if s_norm > 1e-6:
                force[i] = steering / s_norm
        return force

    # Dense 2D int array — vectorised gather+reduce
    k = neighbor_idx.shape[1] if neighbor_idx.ndim == 2 else 0
    if k == 0:
        return force

    nbr_idx = neighbor_idx[active_idx]           # (n_active, k)
    v_j = velocities[nbr_idx]                    # (n_active, k, 3)

    # S1.5: Reynolds steering — normalize(v̄ − v_i), not normalize(v̄) − normalize(v_i).
    # The subtract-then-normalize avoids unequal-speed vector distortion.
    avg_vel = np.mean(v_j, axis=1)               # (n_active, 3)
    v_i = velocities[active_idx]                 # (n_active, 3)
    steering = avg_vel - v_i                      # (n_active, 3)
    steering_norms = np.linalg.norm(steering, axis=1)
    valid = steering_norms > 1e-6
    if valid.any():
        force[active_idx[valid]] = (
            steering[valid] / steering_norms[valid, np.newaxis]
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
            # S1.5: limit3(to_center, 1.0) — cap at unit, sub-unit passes
            if length > 1.0:
                to_center = to_center / length
            force[i] = to_center
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

    # S1.5: limit3(to_center, 1.0) — cap at unit length.
    # Sub-unit vectors pass through unscaled (don't inflate short vectors).
    force_i = to_center.copy()
    long = lengths > 1.0
    if long.any():
        force_i[long] = to_center[long] / lengths[long, np.newaxis]
    force[active_idx] = force_i

    return force


def curl_flow(
    positions_active: np.ndarray,
    center: np.ndarray,
    seeds: np.ndarray,
    t: float,
    U: float,
) -> np.ndarray:
    """S2.B11: Shared curl-like flow primitive — L0, mode-agnostic.

    Normalized pseudo-curl direction from a deterministic sinusoidal
    field of the domain-relative position q=(p-center)/U, per-bird
    phase-shifted by *seeds*. Base magnitude 0.08 — callers apply their
    own gain(s) on top (FieldMode: flow*flow_pull; SpatialMode: S2.B11
    flow_weight*0.22).

    Originally field-mode-only (physics/forces/field.py::_compute_curl_flow);
    factored out here so SpatialMode can share the exact same primitive.

    Returns (n_active, 3) float32.
    """
    n = len(seeds)
    if n == 0:
        return np.zeros((n, 3), dtype=np.float32)

    q = (positions_active - center) / max(float(U), 1e-6)
    flow_vec = np.column_stack([
        np.sin(q[:, 1] * 2.8 + t * 0.24 + seeds)
        + np.cos(q[:, 2] * 2.1 - t * 0.17),
        np.sin(q[:, 2] * 2.3 + t * 0.20)
        - np.cos(q[:, 0] * 1.9 + t * 0.24),
        np.sin(q[:, 0] * 2.6 - t * 0.16)
        + np.cos(q[:, 1] * 2.2 + t * 0.24),
    ]).astype(np.float32)

    norms = np.linalg.norm(flow_vec, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (flow_vec / norms * 0.08).astype(np.float32)


def noise_force(
    n: int,
    scale: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random perturbation — uniform direction, magnitude = scale (D9).

    Returns (N, 3) float32. scale=0 produces all-zero array.
    scale=1.0 gives unit vectors (backward compatible).
    """
    if scale == 0.0 or n == 0:
        return np.zeros((n, 3), dtype=np.float32)

    rng = rng or np.random.default_rng()
    pts = rng.normal(scale=scale, size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    # D9: multiply by scale so noise_scale actually controls magnitude,
    # not just on/off toggling via the normalisation step.
    return (pts / norms) * scale
