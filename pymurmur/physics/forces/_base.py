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

from . import force_kernels as _kernels

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


def _dispatch_separation_kernel(
    kernel: str,
    diffs: np.ndarray,
    dists: np.ndarray,
    close: np.ndarray,
    radius: float | None = None,
    zone_width: float | None = None,
    closing_speed: np.ndarray | None = None,
    heading: np.ndarray | None = None,
) -> np.ndarray:
    """Kernel-name -> kernels.py function dispatch, shared by the ragged
    and dense paths below. Unrecognized kernel names silently fall back
    to "sum" (matches pre-refactor behavior, explicitly tested by
    test_invalid_kernel_ignored_by_code_structure)."""
    if kernel == "mean":
        return _kernels.kernel_mean(diffs, dists, close)
    if kernel == "unit":
        return _kernels.kernel_unit(diffs, dists, close)
    if kernel == "exp":
        return _kernels.kernel_exp(diffs, dists, close, radius)
    if kernel == "linear_ramp":
        return _kernels.kernel_linear_ramp(diffs, dists, close, radius)
    if kernel == "asymptotic":
        return _kernels.kernel_asymptotic(diffs, dists, close, radius)
    if kernel == "velocity_weighted":
        return _kernels.kernel_velocity_weighted(diffs, dists, close, closing_speed)
    if kernel == "cosine_zone":
        return _kernels.kernel_cosine_zone(diffs, dists, close, heading)
    if kernel == "linear":
        return _kernels.kernel_linear(diffs, dists, close)
    if kernel == "nearest_only":
        return _kernels.kernel_nearest_only(diffs, dists, close)
    if kernel == "bell_zone":
        return _kernels.kernel_bell_zone(diffs, dists, close, radius, zone_width)
    return _kernels.kernel_sum(diffs, dists, close)


def _closing_speed(diffs: np.ndarray, dists: np.ndarray, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
    """Rate of distance decrease between i and each neighbor j (positive
    = approaching). unit = diffs/dist (points i -> j); closing speed is
    the component of (v_i - v_j) along that direction."""
    dists_safe = np.where(dists > 1e-6, dists, 1.0)
    unit = diffs / dists_safe[..., np.newaxis]
    return np.sum(unit * (v_i - v_j), axis=-1)


def separation_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
    kernel: str = "sum",
    kernel_radius: float = 20.0,
    kernel_zone_width: float = 10.0,
) -> np.ndarray:
    """Separation: push away from nearby neighbours.

    S1.5/kernel registry (pymurmur.physics.forces.kernels) — how
    neighbour contributions are combined:

    kernel="sum"    → Σ r̂/d² (Reynolds default; magnitude falls as 1/d²)
    kernel="mean"   → (1/k) Σ r̂/d² (density-invariant)
    kernel="unit"   → Σ −û(d_ij) (unit direction, distance-independent)
    kernel="exp"          → exponential-decay weight on unit direction
    kernel="linear_ramp"  → linear ramp weight on unit direction
    kernel="asymptotic"   → r/d − 1 weight on unit direction
    kernel="velocity_weighted" → "sum" base scaled by closing speed
                                 (receding neighbours contribute nothing)
    kernel="cosine_zone" → "sum" base scaled by a continuous (1+cosθ)/2
                            weight on bearing vs. own heading (§08-style
                            continuous zone, vs. a hard FOV cone cutoff)
    kernel="linear"       → Σ r̂/d (plain 1/d falloff)
    kernel="nearest_only" → only the single closest neighbour contributes
    kernel="bell_zone"    → cosine-bell weight peaking at kernel_radius
                            (zone center), falling off symmetrically on
                            both sides over kernel_zone_width

    kernel_radius is consulted by exp/linear_ramp/asymptotic/bell_zone
    (as the zone center for bell_zone). kernel_zone_width is only
    consulted by bell_zone.
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return force

    needs_closing_speed = kernel == "velocity_weighted"
    needs_heading = kernel == "cosine_zone"

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
            closing_speed = (
                _closing_speed(diffs, dists, velocities[i], velocities[nbrs])
                if needs_closing_speed else None
            )
            heading = velocities[i] if needs_heading else None
            force[i] = _dispatch_separation_kernel(
                kernel, diffs, dists, close,
                radius=kernel_radius, zone_width=kernel_zone_width,
                closing_speed=closing_speed, heading=heading,
            )
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

    closing_speed = None
    heading = None
    if needs_closing_speed:
        v_i = velocities[active_idx]              # (n_active, 3)
        v_j = velocities[nbr_idx]                 # (n_active, k, 3)
        closing_speed = _closing_speed(diffs, dists, v_i[:, np.newaxis, :], v_j)
    if needs_heading:
        heading = velocities[active_idx]          # (n_active, 3)

    contrib = _dispatch_separation_kernel(
        kernel, diffs, dists, close,
        radius=kernel_radius, zone_width=kernel_zone_width,
        closing_speed=closing_speed, heading=heading,
    )
    force[active_idx] = contrib.astype(np.float32)
    return force


def _weighted_avg_vel(
    kernel: str,
    diffs: np.ndarray,
    dists: np.ndarray,
    close: np.ndarray,
    neighbor_vel: np.ndarray,
    heading: np.ndarray,
    fov_min: float,
    zone_center: float,
    zone_width: float,
) -> np.ndarray:
    """Alignment kernel dispatch — mirrors _dispatch_separation_kernel.
    Returns the weighted-average neighbour velocity (..., 3), NOT yet
    turned into a normalized steering vector (caller does that)."""
    if kernel == "fov_weighted":
        return _kernels.kernel_fov_weighted(diffs, dists, close, heading, neighbor_vel, fov_min)
    if kernel == "circular_mean_2d":
        return _kernels.kernel_circular_mean_2d(diffs, dists, close, neighbor_vel)
    if kernel == "bell_zone":
        return _kernels.kernel_bell_zone_alignment(
            diffs, dists, close, neighbor_vel, zone_center, zone_width,
        )
    return np.mean(neighbor_vel, axis=-2)


def alignment_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
    kernel: str = "unweighted",
    fov_min: float = -1.0,
    kernel_radius: float = 20.0,
    kernel_zone_width: float = 10.0,
) -> np.ndarray:
    """Alignment: steer toward average (or weighted) neighbour heading.

    kernel="unweighted" (default) → F_align[i] = û_avg(j) − û_i, plain mean
        of neighbour velocities — pre-existing behavior, unchanged.
    kernel="fov_weighted" → InverseLerp(cos_theta, fov_min, 1.0)-weighted
        average: neighbours near the bearing axis (dead ahead) dominate,
        neighbours near fov_min contribute ~0.
    kernel="circular_mean_2d" → 2D circular mean (atan2(Σsinθ,Σcosθ)) of
        neighbour headings projected onto the XY plane, Z averaged linearly.
    kernel="bell_zone" → cosine-bell-weighted average velocity, peaking at
        kernel_radius (zone center) and falling off symmetrically over
        kernel_zone_width — same distance-zone concept as separation/
        cohesion's "bell_zone".

    fov_min is only consulted by "fov_weighted" (reuses the same
    InverseLerp-lower-bound semantics as the existing angle_align field).
    kernel_radius/kernel_zone_width are only consulted by "bell_zone".
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return force

    needs_dists = kernel in ("fov_weighted", "circular_mean_2d", "bell_zone")

    if _is_ragged(neighbor_idx):
        # Ragged object array — per-bird fallback
        for i in active_idx:
            nbrs = neighbor_idx[i]
            if len(nbrs) == 0:
                continue
            neighbor_vel = velocities[nbrs]
            if needs_dists:
                diffs = positions[nbrs] - positions[i]
                dists = np.linalg.norm(diffs, axis=1)
                close = dists > 1e-6
                if not close.any():
                    continue
                avg_vel = _weighted_avg_vel(
                    kernel, diffs, dists, close, neighbor_vel, velocities[i], fov_min,
                    kernel_radius, kernel_zone_width,
                )
            else:
                avg_vel = np.mean(neighbor_vel, axis=0)
            # S1.5: Reynolds steering — normalize(v̄ − v_i)
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
    v_i = velocities[active_idx]                 # (n_active, 3)

    if needs_dists:
        p_i = positions[active_idx]               # (n_active, 3)
        p_j = positions[nbr_idx]                   # (n_active, k, 3)
        diffs = p_j - p_i[:, np.newaxis, :]         # (n_active, k, 3)
        dists = np.linalg.norm(diffs, axis=2)       # (n_active, k)
        close = dists > 1e-6
        avg_vel = _weighted_avg_vel(
            kernel, diffs, dists, close, v_j, v_i, fov_min,
            kernel_radius, kernel_zone_width,
        )
    else:
        # S1.5: Reynolds steering — normalize(v̄ − v_i), not normalize(v̄) − normalize(v_i).
        # The subtract-then-normalize avoids unequal-speed vector distortion.
        avg_vel = np.mean(v_j, axis=1)             # (n_active, 3)

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
    kernel: str = "unweighted",
    kernel_radius: float = 20.0,
    kernel_zone_width: float = 10.0,
) -> np.ndarray:
    """Cohesion: steer toward average (or weighted) neighbour position.

    kernel="unweighted" (default) → F_coh[i] = û(mean(p_j) − p_i), bounded
        unit vector (P1.7 fix) — pre-existing behavior, unchanged.
    kernel="inverse_distance" → 1/d-weighted center of mass: nearer
        neighbours pull more strongly than farther ones.
    kernel="bell_zone" → cosine-bell-weighted center of mass, peaking at
        kernel_radius (zone center) and falling off symmetrically over
        kernel_zone_width — same distance-zone concept as separation's
        "bell_zone", not the bearing-angle "cosine_zone" kernel.

    kernel_radius/kernel_zone_width are only consulted by "bell_zone".
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
            if kernel == "inverse_distance":
                dists = np.linalg.norm(diffs, axis=1)
                close = dists > 1e-6
                to_center = _kernels.kernel_inverse_distance(diffs, dists, close)
            elif kernel == "bell_zone":
                dists = np.linalg.norm(diffs, axis=1)
                close = dists > 1e-6
                to_center = _kernels.kernel_bell_zone_cohesion(
                    diffs, dists, close, kernel_radius, kernel_zone_width,
                )
            else:
                to_center = _kernels.kernel_unweighted(diffs)
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
    diffs = p_j - p_i[:, np.newaxis, :]          # (n_active, k, 3)

    if kernel == "inverse_distance":
        dists = np.linalg.norm(diffs, axis=2)
        close = dists > 1e-6
        to_center = _kernels.kernel_inverse_distance(diffs, dists, close)
    elif kernel == "bell_zone":
        dists = np.linalg.norm(diffs, axis=2)
        close = dists > 1e-6
        to_center = _kernels.kernel_bell_zone_cohesion(
            diffs, dists, close, kernel_radius, kernel_zone_width,
        )
    else:
        to_center = _kernels.kernel_unweighted(diffs)
    lengths = np.linalg.norm(to_center, axis=1)  # (n_active,)

    # S1.5: limit3(to_center, 1.0) — cap at unit length.
    # Sub-unit vectors pass through unscaled (don't inflate short vectors).
    force_i = to_center.copy()
    long = lengths > 1.0
    if long.any():
        force_i[long] = to_center[long] / lengths[long, np.newaxis]
    force[active_idx] = force_i.astype(np.float32)

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
