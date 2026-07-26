"""Separation/cohesion kernel registry — pluggable per-neighbor weight
functions.

Level 0 — pure numpy, no project imports. Each kernel function takes
precomputed per-neighbor geometry (diffs, dists, a validity mask, and
for velocity-aware kernels, per-neighbor closing speed) and returns the
summed-over-neighbors contribution.

Shape convention: every function operates on the *last two axes* —
diffs is (..., k, 3), dists/close/closing_speed are (..., k) — so the
exact same function serves both the ragged per-bird path (no leading
batch axis) and the dense vectorized path ((n_active, k, ...)) in
_base.py's separation_force/cohesion_force, which is what lets those
two call sites share one implementation per kernel instead of
duplicating ragged+dense math per kernel as before this module existed.

sum/mean/unit reproduce _base.py's pre-existing formulas exactly
(verified by a before/after behavioral diff, not just re-derived from
the docstrings — separation_force's own top-line docstring formula
predates a since-corrected exponent and no longer matches the code, so
these functions were extracted from the actual code, not the prose).
"""

from __future__ import annotations

import numpy as np


def _unit_dirs(diffs: np.ndarray, dists_safe: np.ndarray) -> np.ndarray:
    """Unit vectors pointing away from each neighbor (-diffs normalized)."""
    return -diffs / dists_safe[..., np.newaxis]


def _safe_dists(dists: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Avoid divide-by-zero for masked-out (non-close) slots."""
    return np.where(close, dists, 1.0)


def kernel_sum(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Σ r̂/d² — Reynolds default. Magnitude falls as 1/d² (not 1/d —
    see module docstring)."""
    dists_safe = _safe_dists(dists, close)
    contrib = _unit_dirs(diffs, dists_safe) / dists_safe[..., np.newaxis] ** 2
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_mean(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray) -> np.ndarray:
    """(1/k) Σ r̂/d² — density-invariant version of "sum"."""
    total = kernel_sum(diffs, dists, close)
    n = np.sum(close, axis=-1).astype(np.float32)
    n = np.where(n == 0, 1.0, n)
    return total / n[..., np.newaxis]


def kernel_unit(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Σ −û(d_ij) — unit-direction, distance-independent magnitude."""
    dists_safe = _safe_dists(dists, close)
    contrib = _unit_dirs(diffs, dists_safe)
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_exp(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray, radius: float) -> np.ndarray:
    """Exponential decay: weight = exp(-(d - radius) / radius) on unit direction."""
    dists_safe = _safe_dists(dists, close)
    weight = np.exp(-(dists_safe - radius) / radius)
    contrib = _unit_dirs(diffs, dists_safe) * weight[..., np.newaxis]
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_linear_ramp(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray, radius: float) -> np.ndarray:
    """Linear ramp: weight = max(radius - d, 0) on unit direction."""
    dists_safe = _safe_dists(dists, close)
    weight = np.maximum(radius - dists_safe, 0.0)
    contrib = _unit_dirs(diffs, dists_safe) * weight[..., np.newaxis]
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_asymptotic(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray, radius: float) -> np.ndarray:
    """Asymptotic: weight = max(radius/d - 1, 0) on unit direction."""
    dists_safe = _safe_dists(dists, close)
    weight = np.maximum(radius / dists_safe - 1.0, 0.0)
    contrib = _unit_dirs(diffs, dists_safe) * weight[..., np.newaxis]
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_velocity_weighted(
    diffs: np.ndarray, dists: np.ndarray, close: np.ndarray,
    closing_speed: np.ndarray,
) -> np.ndarray:
    """Separation scaled by closing speed (positive = approaching), on
    top of the "sum" kernel's 1/d² base — receding neighbors contribute
    nothing (weight floored at 0), approaching ones push harder the
    faster they close."""
    dists_safe = _safe_dists(dists, close)
    weight = np.maximum(closing_speed, 0.0)
    contrib = (
        _unit_dirs(diffs, dists_safe) / dists_safe[..., np.newaxis] ** 2
        * weight[..., np.newaxis]
    )
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_cosine_zone(
    diffs: np.ndarray, dists: np.ndarray, close: np.ndarray,
    heading: np.ndarray,
) -> np.ndarray:
    """Continuous cosine-zone weighting on top of the "sum" kernel's
    1/d² base: weight = (1 + cos_theta) / 2, where cos_theta is the
    cosine of the angle between the observer's heading and the bearing
    to each neighbour (1 = directly ahead, 0 = directly behind). Unlike
    the hard FOV cone cutoff (spatial_helpers._maybe_perception_filter),
    this blends continuously rather than including/excluding a neighbour
    outright."""
    dists_safe = _safe_dists(dists, close)
    bearing = diffs / dists_safe[..., np.newaxis]
    heading_norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    heading_unit = heading / np.maximum(heading_norm, 1e-10)
    cos_theta = np.sum(bearing * heading_unit[..., np.newaxis, :], axis=-1)
    weight = (1.0 + cos_theta) / 2.0
    contrib = (
        _unit_dirs(diffs, dists_safe) / dists_safe[..., np.newaxis] ** 2
        * weight[..., np.newaxis]
    )
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_unweighted(diffs: np.ndarray) -> np.ndarray:
    """Cohesion default: plain mean of neighbor positions (no distance
    weighting, no closeness mask) — matches cohesion_force's pre-existing
    "mean(p_j) - p_i" formula exactly, which never filtered neighbors by
    distance (unlike separation_force's "close" mask)."""
    return np.mean(diffs, axis=-2)


def kernel_inverse_distance(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray) -> np.ndarray:
    """1/d-weighted cohesion: nearer neighbors pull more strongly toward
    the (weighted) center than farther ones.

    Note: weighting diffs directly by 1/d would cancel exactly (diffs
    already has magnitude d, so diffs/d is just a unit vector regardless
    of distance) — the weight must apply to the unit direction, i.e.
    contribution = diffs/d² = unit_direction * (1/d), which does retain
    a distance-dependent magnitude (nearer neighbors contribute more).
    """
    dists_safe = _safe_dists(dists, close)
    weight = np.where(close, 1.0 / dists_safe, 0.0)
    total_weight = np.sum(weight, axis=-1)
    total_weight = np.where(total_weight == 0, 1.0, total_weight)
    contrib = diffs / dists_safe[..., np.newaxis] ** 2
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2) / total_weight[..., np.newaxis]


SEPARATION_KERNELS_NEEDING_RADIUS = frozenset({"exp", "linear_ramp", "asymptotic"})
SEPARATION_KERNELS_NEEDING_VELOCITY = frozenset({"velocity_weighted", "cosine_zone"})
VALID_SEPARATION_KERNELS = frozenset({
    "sum", "mean", "unit", "exp", "linear_ramp", "asymptotic",
    "velocity_weighted", "cosine_zone",
})
VALID_COHESION_KERNELS = frozenset({"unweighted", "inverse_distance"})
