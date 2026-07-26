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


def kernel_linear(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Σ r̂/d — plain 1/d falloff (distinct from "sum"'s 1/d²)."""
    dists_safe = _safe_dists(dists, close)
    contrib = _unit_dirs(diffs, dists_safe) / dists_safe[..., np.newaxis]
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_nearest_only(diffs: np.ndarray, dists: np.ndarray, close: np.ndarray) -> np.ndarray:
    """F_sep = r̂_nearest — only the single closest neighbor contributes,
    at unit strength, regardless of how many other neighbors exist.
    Ties (multiple neighbors at the exact same minimum distance) are
    all included, an intentionally simple tie-break rather than an
    arbitrary single winner."""
    dists_safe = _safe_dists(dists, close)
    dists_for_min = np.where(close, dists, np.inf)
    min_dist = np.min(dists_for_min, axis=-1, keepdims=True)
    is_nearest = close & (dists_for_min <= min_dist)
    contrib = _unit_dirs(diffs, dists_safe)
    contrib = np.where(is_nearest[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def _bell_weight(dists_safe: np.ndarray, zone_center: float, zone_width: float) -> np.ndarray:
    """cos(π·clip(|d − center|/width, 0, 1))/2 + 0.5 — peaks at 1.0 when
    d == zone_center, falls symmetrically to 0.0 at |d - center| >=
    width on EITHER side (nearer than the zone AND farther than it both
    reduce weight) — the one qualitative property distinguishing this
    from every distance-monotonic kernel above."""
    t = np.clip(np.abs(dists_safe - zone_center) / zone_width, 0.0, 1.0)
    return np.cos(np.pi * t) / 2.0 + 0.5


def kernel_bell_zone(
    diffs: np.ndarray, dists: np.ndarray, close: np.ndarray,
    zone_center: float, zone_width: float,
) -> np.ndarray:
    """Separation: cosine-bell weight (see _bell_weight) on unit
    direction (push away)."""
    dists_safe = _safe_dists(dists, close)
    weight = _bell_weight(dists_safe, zone_center, zone_width)
    contrib = _unit_dirs(diffs, dists_safe) * weight[..., np.newaxis]
    contrib = np.where(close[..., np.newaxis], contrib, 0.0)
    return np.sum(contrib, axis=-2)


def kernel_bell_zone_cohesion(
    diffs: np.ndarray, dists: np.ndarray, close: np.ndarray,
    zone_center: float, zone_width: float,
) -> np.ndarray:
    """Cohesion: cosine-bell-weighted center of mass — unlike
    inverse_distance's 1/d weight (which must apply to the *unit*
    direction to avoid cancelling against diffs' own d-proportional
    magnitude), the bell weight is not itself proportional to 1/d, so
    weighting raw diffs directly is safe here (no cancellation trap)."""
    dists_safe = _safe_dists(dists, close)
    weight = np.where(close, _bell_weight(dists_safe, zone_center, zone_width), 0.0)
    total_weight = np.sum(weight, axis=-1)
    total_weight = np.where(total_weight == 0, 1.0, total_weight)
    weighted = diffs * weight[..., np.newaxis]
    return np.sum(weighted, axis=-2) / total_weight[..., np.newaxis]


def kernel_fov_weighted(
    diffs: np.ndarray, dists: np.ndarray, close: np.ndarray,
    heading: np.ndarray, neighbor_vel: np.ndarray, fov_min: float,
) -> np.ndarray:
    """Alignment: InverseLerp(cos_theta, fov_min, 1.0)-weighted average
    of neighbor velocities. Neighbors dead ahead (cos_theta -> 1) weight
    fully; neighbors at the edge of the FOV cone (cos_theta -> fov_min)
    weight toward zero. fov_min is typically config.angle_align (already
    an existing cos(θ) alignment-cone threshold, reused here rather than
    inventing a new field)."""
    dists_safe = _safe_dists(dists, close)
    bearing = diffs / dists_safe[..., np.newaxis]
    heading_norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    heading_unit = heading / np.maximum(heading_norm, 1e-10)
    cos_theta = np.sum(bearing * heading_unit[..., np.newaxis, :], axis=-1)
    denom = max(1.0 - fov_min, 1e-6)
    weight = np.clip((cos_theta - fov_min) / denom, 0.0, 1.0)
    weight = np.where(close, weight, 0.0)
    total_weight = np.sum(weight, axis=-1)
    total_weight = np.where(total_weight == 0, 1.0, total_weight)
    weighted = neighbor_vel * weight[..., np.newaxis]
    return np.sum(weighted, axis=-2) / total_weight[..., np.newaxis]


def kernel_circular_mean_2d(
    diffs: np.ndarray, dists: np.ndarray, close: np.ndarray,
    neighbor_vel: np.ndarray,
) -> np.ndarray:
    """Alignment: 2D circular mean of neighbor headings, projected onto
    the XY plane (this engine's ground-plane convention — Z is "up").
    theta_bar = atan2(Σsinθ, Σcosθ), a scalar-angle circular mean, not a
    vector average — matches §16/17 exactly. Z has no circular/
    wraparound semantics (altitude isn't an angle), so it's averaged
    linearly instead. The XY part is scaled by the mean XY speed (not
    left as a unit vector) so it's on the same physical velocity scale
    as the linearly-averaged Z part, rather than one part being ~1 and
    the other being a real speed."""
    vx = neighbor_vel[..., 0]
    vy = neighbor_vel[..., 1]
    vz = neighbor_vel[..., 2]
    xy_speed = np.sqrt(vx ** 2 + vy ** 2)
    valid = close & (xy_speed > 1e-6)

    theta = np.arctan2(np.where(valid, vy, 0.0), np.where(valid, vx, 1.0))
    sin_sum = np.sum(np.where(valid, np.sin(theta), 0.0), axis=-1)
    cos_sum = np.sum(np.where(valid, np.cos(theta), 0.0), axis=-1)
    theta_bar = np.arctan2(sin_sum, cos_sum)

    n_valid = np.sum(valid, axis=-1)
    n_valid_safe = np.where(n_valid == 0, 1, n_valid)
    mean_xy_speed = np.sum(np.where(valid, xy_speed, 0.0), axis=-1) / n_valid_safe

    n_close = np.sum(close, axis=-1)
    n_close_safe = np.where(n_close == 0, 1, n_close)
    z_mean = np.sum(np.where(close, vz, 0.0), axis=-1) / n_close_safe

    result_x = mean_xy_speed * np.cos(theta_bar)
    result_y = mean_xy_speed * np.sin(theta_bar)
    return np.stack([result_x, result_y, z_mean], axis=-1)


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


SEPARATION_KERNELS_NEEDING_RADIUS = frozenset({"exp", "linear_ramp", "asymptotic", "bell_zone"})
SEPARATION_KERNELS_NEEDING_VELOCITY = frozenset({"velocity_weighted", "cosine_zone"})
VALID_SEPARATION_KERNELS = frozenset({
    "sum", "mean", "unit", "exp", "linear_ramp", "asymptotic",
    "velocity_weighted", "cosine_zone", "linear", "nearest_only", "bell_zone",
})
VALID_COHESION_KERNELS = frozenset({"unweighted", "inverse_distance", "bell_zone"})
VALID_ALIGNMENT_KERNELS = frozenset({"unweighted", "fov_weighted", "circular_mean_2d"})
