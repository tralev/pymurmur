"""Neighbour-query and filtering helpers for SpatialMode.

Extracted from spatial.py (file-size split) — depends on
_dispatch_kernels (still in spatial.py, next to the kernel
selection/fallback machinery it's paired with).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


def _query_neighbors(
    positions: np.ndarray,
    active: np.ndarray,
    index: "SpatialIndex",
    config: "SimConfig",
    filter_mode: str = "hybrid",
    is_predator: np.ndarray | None = None,
) -> np.ndarray:
    """Build per-bird neighbour index using the shared spatial index.

    P4.1: Hybrid metric+topological filter.
    - Queries up to topological_cap (default 50) nearest neighbours via k-NN.
    - Filters to only those within visual_range (metric).
    - Caps at influence_count (default 7) accepted neighbours per bird.

    filter_mode: "hybrid" (metric+topological), "metric" (visual_range only),
                 "topological" (influence_count only), "none" (return all).

    C3: predator_perception_boost — when is_predator marks any active bird,
    predator rows get visual_range scaled by
    predator_perception_boost (prey rows are unaffected).

    Returns (N_capacity, k) int32 array indexed by global bird index.
    Inactive rows are zero-filled.
    """
    from scipy.spatial import cKDTree

    from .spatial import _dispatch_kernels

    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    N = len(positions)

    if n_active < 2:
        return np.zeros((N, 0), dtype=np.int32)

    # ── P4.1: Hybrid filter knobs ──
    topological_cap = min(config.topological_cap, n_active - 1)
    # DynamicVisionRange extension bridge (cfg._dynamic_visual_range_mult,
    # mirrors Wander's cfg._wander_heading pattern) — absent/1.0 by default,
    # so disabled runs are byte-identical to before this bridge existed.
    visual_range = config.visual_range * getattr(config, '_dynamic_visual_range_mult', 1.0)
    influence_count = config.influence_count
    perception_boost = getattr(config, 'predator_perception_boost', 1.0)
    has_predators = (
        is_predator is not None
        and perception_boost != 1.0
        and bool((is_predator & active).any())
    )
    # Query enough candidates so we have room to filter down to influence_count.
    # C3: predator_perception_boost only widens the *filter* radius for
    # predator rows below — the shared k-NN candidate pool stays exactly
    # as before so prey rows are numerically unaffected by boost != 1.0.
    k = max(topological_cap, influence_count * 3)
    k = min(k, n_active - 1)

    neighbor_idx = np.zeros((N, k), dtype=np.int32)

    # ── Batch k-NN query (P4.10 opt): single C++ call instead of per-bird loop ──
    tree = getattr(index, 'tree', None)
    if tree is None:
        tree = cKDTree(positions[active_idx])

    active_pos = positions[active_idx]
    # S2.B6: cfg.perf.num_threads (0 = auto/all cores, matching scipy's
    # workers=-1 convention; N>0 pins the worker count) instead of a
    # hardcoded workers=-1.
    num_threads = getattr(getattr(config, 'perf', None), 'num_threads', 0)
    workers = -1 if num_threads == 0 else num_threads
    _, compacted_idx = tree.query(active_pos, k=k + 1, workers=workers)
    neighbor_idx[active_idx] = active_idx[compacted_idx[:, 1:k + 1]]

    # ── Apply filter based on mode ──
    _hybrid_filter, _, _ = _dispatch_kernels(config)
    if filter_mode == "none":
        pass  # return all neighbours unfiltered
    elif filter_mode == "metric":
        # Metric-only: visual_range filter, no topological cap
        _apply_hybrid_filter(
            _hybrid_filter, neighbor_idx, positions, active,
            visual_range, k, is_predator, perception_boost, has_predators,
        )
    elif filter_mode == "topological":
        # Topological-only: cap at influence_count, no distance filter
        _hybrid_filter(
            neighbor_idx, positions, active, 1e9, influence_count,
        )
    else:  # "hybrid" (default)
        _apply_hybrid_filter(
            _hybrid_filter, neighbor_idx, positions, active,
            visual_range, influence_count, is_predator, perception_boost, has_predators,
        )

    return neighbor_idx


def _apply_hybrid_filter(
    hybrid_filter,
    neighbor_idx: np.ndarray,
    positions: np.ndarray,
    active: np.ndarray,
    visual_range: float,
    count_cap: int,
    is_predator: np.ndarray | None,
    perception_boost: float,
    has_predators: bool,
) -> None:
    """Apply the hybrid filter, splitting predator/prey rows by visual_range.

    Kernels only mutate rows where `active` is True, so calling the same
    kernel twice with disjoint predator/prey active masks against the same
    neighbor_idx array is equivalent to a per-bird visual_range — no kernel
    signature change needed.
    """
    if not has_predators:
        hybrid_filter(neighbor_idx, positions, active, visual_range, count_cap)
        return

    assert is_predator is not None
    predator_active = active & is_predator
    prey_active = active & ~is_predator
    if prey_active.any():
        hybrid_filter(neighbor_idx, positions, prey_active, visual_range, count_cap)
    if predator_active.any():
        hybrid_filter(
            neighbor_idx, positions, predator_active,
            visual_range * perception_boost, count_cap,
        )


def _maybe_perception_filter(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
    max_dist: float,
    cos_angle: float,
) -> np.ndarray:
    """P11.5: Filter a neighbour set by max distance and perception cone.

    max_dist ≤ 0 disables the distance filter; cos_angle ≤ −1 disables
    the cone filter (full sphere). When both are disabled the shared
    neighbor_idx is returned untouched (fast path).

    A neighbour j survives the cone when the angle between bird i's
    heading and the bearing to j satisfies cos(θ) ≥ cos_angle — birds
    behind the cone are excluded.

    Returns a ragged object array (per-bird index lists), which the
    _base force functions handle via their per-bird fallback. Padding
    zeros in neighbor_idx rows are treated as empty slots (the shared
    hybrid-filter convention).
    """
    if max_dist <= 0.0 and cos_angle <= -1.0:
        return neighbor_idx

    N = len(positions)
    out = np.empty(N, dtype=object)
    empty = np.empty(0, dtype=np.int32)
    for i in range(N):
        out[i] = empty

    max_dist_sq = max_dist * max_dist
    for i in np.where(active)[0]:
        nbrs = neighbor_idx[i]
        nbrs = nbrs[(nbrs > 0) & (nbrs != i)]
        if len(nbrs) == 0:
            continue
        diffs = positions[nbrs] - positions[i]
        dists_sq = np.sum(diffs * diffs, axis=1)
        keep = dists_sq > 1e-12
        if max_dist > 0.0:
            keep &= dists_sq <= max_dist_sq
        if cos_angle > -1.0:
            v = velocities[i]
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                bearings = diffs / np.sqrt(dists_sq)[:, np.newaxis]
                cos_theta = bearings @ (v / v_norm)
                keep &= cos_theta >= cos_angle
        out[i] = nbrs[keep].astype(np.int32)
    return out


def _predator_escape(
    positions: np.ndarray,
    neighbor_idx: np.ndarray,
    is_predator: np.ndarray,
    threatened: np.ndarray,
    active: np.ndarray,
    config: "SimConfig",
) -> np.ndarray:
    """P4.3: Compute predator escape force for threatened prey.

    For each threatened prey bird, finds the nearest predator among its
    neighbours and produces a repulsive force away from it, scaled by
    predator_escape_factor.

    Returns (N_capacity, 3) float32 force array.
    """
    from .spatial import _dispatch_kernels

    escape = np.zeros((len(positions), 3), dtype=np.float32)
    # Safe fallback for non-SimConfig configs (e.g., test FakeConfig)
    spatial = getattr(config, 'spatial', None)
    escape_factor = (
        getattr(spatial, 'predator_escape_factor', 10_000_000.0)
        if spatial is not None else 10_000_000.0
    )
    accel_boost = (
        getattr(spatial, 'predator_accel_boost', 1.4)
        if spatial is not None else 1.4
    )

    # S2.B3: minimum-image escape distances on toroidal domains — an
    # all-zero box disables wrapping for every other boundary mode.
    boundary_mode = getattr(config, 'boundary_mode', 'toroidal')
    width = getattr(config, 'width', 0.0)
    height = getattr(config, 'height', 0.0)
    depth = getattr(config, 'depth', 0.0)
    if boundary_mode == 'toroidal' and width and height and depth:
        box = np.array([width, height, depth], dtype=np.float32)
    else:
        box = np.zeros(3, dtype=np.float32)

    # Dispatch kernel based on config.perf.use_numba
    _, _, _pred_escape_kernel = _dispatch_kernels(config)
    _pred_escape_kernel(
        escape, positions, neighbor_idx, is_predator, threatened, active,
        escape_factor, accel_boost, box,
    )

    return escape
