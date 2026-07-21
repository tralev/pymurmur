"""Numba-accelerated force kernels for the spatial mode.

P4.10: @njit-decorated functions that replace Python for-loops in:
  - _query_neighbors: hybrid metric+topological filter
  - predator detection: threat mask computation
  - predator escape: escape force from nearest predator

Each kernel produces identical output to its pure-numpy counterpart.
Imported lazily by spatial.py with a pure-numpy fallback when numba
is unavailable.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator when numba is absent."""
        def wrapper(fn):
            return fn
        return wrapper


# ═══════════════════════════════════════════════════════════════════
# Hybrid filter kernel (P4.1)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _numba_hybrid_filter(
    neighbor_idx: np.ndarray,      # (N, k) int32 — mutated in-place
    positions: np.ndarray,          # (N, 3) float32
    active: np.ndarray,             # (N,) bool
    visual_range: float,
    influence_count: int,
) -> None:
    """P4.1/P4.10: Metric+topological neighbour filter — numba-accelerated.

    For each active bird, filters its k-NN neighbours:
      1. Keeps only those within visual_range (metric filter)
      2. Caps at influence_count accepted neighbours (topological)

    Mutates neighbor_idx in-place for zero-allocation performance.
    """
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    k = neighbor_idx.shape[1]
    vr_sq = visual_range * visual_range

    for idx in range(n_active):
        global_i = active_idx[idx]
        nbrs = neighbor_idx[global_i]
        pi = positions[global_i]

        # Single-pass distance filter (P4.10 fix: use continue, not break,
        # since zeros may be scattered after self-removal, not contiguous).
        in_range_count = 0
        in_range_indices = np.zeros(k, dtype=np.int32)

        for ki in range(k):
            nbr_i = nbrs[ki]
            if nbr_i <= 0:
                continue
            pj = positions[nbr_i]
            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            dz = pi[2] - pj[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq <= vr_sq:
                in_range_indices[in_range_count] = nbr_i
                in_range_count += 1

        # Topological cap: keep at most influence_count
        cap = min(in_range_count, influence_count)

        # Zero out the row then write back capped neighbours
        for ki in range(k):
            neighbor_idx[global_i, ki] = 0

        for ni in range(cap):
            neighbor_idx[global_i, ni] = in_range_indices[ni]


# ═══════════════════════════════════════════════════════════════════
# Predator detection kernel (P4.3)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _numba_predator_detect(
    threatened: np.ndarray,         # (N,) bool — mutated in-place
    neighbor_idx: np.ndarray,       # (N, k) int32
    is_predator: np.ndarray,        # (N,) bool
    active: np.ndarray,             # (N,) bool
) -> None:
    """P4.3/P4.10: Predator threat detection — numba-accelerated.

    For each active prey bird, checks if any neighbour is a predator.
    Marks threatened birds in the `threatened` array (mutated in-place).
    """
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    k = neighbor_idx.shape[1]

    for idx in range(n_active):
        global_i = active_idx[idx]

        # Predators don't flee from predators
        if is_predator[global_i]:
            continue

        nbrs = neighbor_idx[global_i]
        # Scan neighbours for any predator (continue, not break — zeros scattered)
        has_predator = False
        for ki in range(k):
            nbr_i = nbrs[ki]
            if nbr_i <= 0:
                continue
            if is_predator[nbr_i]:
                has_predator = True
                break

        threatened[global_i] = has_predator


# ═══════════════════════════════════════════════════════════════════
# Predator escape kernel (P4.3)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _numba_predator_escape(
    escape: np.ndarray,             # (N, 3) float32 — mutated in-place
    positions: np.ndarray,          # (N, 3) float32
    neighbor_idx: np.ndarray,       # (N, k) int32
    is_predator: np.ndarray,        # (N,) bool
    threatened: np.ndarray,         # (N,) bool
    active: np.ndarray,             # (N,) bool
    escape_factor: float,
    accel_boost: float,
) -> None:
    """P4.3/P4.10: Predator escape force — numba-accelerated.

    For each threatened prey bird, finds the nearest predator among
    its neighbours and computes a 1/d² repulsive force away from it.

    Mutates `escape` array in-place for zero-allocation performance.
    """
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    k = neighbor_idx.shape[1]

    for idx in range(n_active):
        global_i = active_idx[idx]

        if not threatened[global_i]:
            continue

        nbrs = neighbor_idx[global_i]
        pi = positions[global_i]

        # Find nearest predator among neighbours (continue, not break)
        nearest_dist_sq = 1e38
        nearest_pred = -1

        for ki in range(k):
            nbr_i = nbrs[ki]
            if nbr_i <= 0:
                continue
            if not is_predator[nbr_i]:
                continue
            pj = positions[nbr_i]
            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            dz = pi[2] - pj[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest_pred = nbr_i

        if nearest_pred < 0 or nearest_dist_sq < 1e-12:
            continue

        # Compute escape direction (away from predator)
        pj = positions[nearest_pred]
        dx = pi[0] - pj[0]
        dy = pi[1] - pj[1]
        dz = pi[2] - pj[2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)

        # 1/d² falloff, scaled by escape_factor * accel_boost
        mag = escape_factor * accel_boost / (d * d)
        escape[global_i, 0] = (dx / d) * mag
        escape[global_i, 1] = (dy / d) * mag
        escape[global_i, 2] = (dz / d) * mag


# ═══════════════════════════════════════════════════════════════════
# NumPy fallback implementations (used when numba is absent)
# ═══════════════════════════════════════════════════════════════════

def _numpy_hybrid_filter(
    neighbor_idx: np.ndarray,
    positions: np.ndarray,
    active: np.ndarray,
    visual_range: float,
    influence_count: int,
) -> None:
    """Pure-numpy fallback for hybrid filter (identical logic)."""
    active_idx = np.where(active)[0]
    vr_sq = visual_range * visual_range

    for global_i in active_idx:
        nbrs = neighbor_idx[global_i]
        valid = nbrs > 0
        if not valid.any():
            continue
        nbr_indices = nbrs[valid]
        diffs = positions[nbr_indices] - positions[global_i]
        dists_sq = np.sum(diffs * diffs, axis=1)
        in_range = dists_sq <= vr_sq
        nbr_indices = nbr_indices[in_range]
        if len(nbr_indices) > influence_count:
            nbr_indices = nbr_indices[:influence_count]
        neighbor_idx[global_i, :] = 0
        neighbor_idx[global_i, :len(nbr_indices)] = nbr_indices


def _numpy_predator_detect(
    threatened: np.ndarray,
    neighbor_idx: np.ndarray,
    is_predator: np.ndarray,
    active: np.ndarray,
) -> None:
    """Pure-numpy fallback for predator detection (identical logic)."""
    active_idx = np.where(active)[0]
    for global_i in active_idx:
        if is_predator[global_i]:
            continue
        nbrs = neighbor_idx[global_i]
        valid_nbrs = nbrs[nbrs > 0]
        if len(valid_nbrs) > 0 and is_predator[valid_nbrs].any():
            threatened[global_i] = True


def _numpy_predator_escape(
    escape: np.ndarray,
    positions: np.ndarray,
    neighbor_idx: np.ndarray,
    is_predator: np.ndarray,
    threatened: np.ndarray,
    active: np.ndarray,
    escape_factor: float,
    accel_boost: float,
) -> None:
    """Pure-numpy fallback for predator escape (identical logic)."""
    active_idx = np.where(active)[0]
    for global_i in active_idx:
        if not threatened[global_i]:
            continue
        nbrs = neighbor_idx[global_i]
        valid_nbrs = nbrs[nbrs > 0]
        if len(valid_nbrs) == 0:
            continue
        predator_mask = is_predator[valid_nbrs]
        if not predator_mask.any():
            continue
        predator_idx = valid_nbrs[predator_mask]
        diffs = positions[predator_idx] - positions[global_i]
        dists_sq = np.sum(diffs * diffs, axis=1)
        nearest = np.argmin(dists_sq)
        to_predator = diffs[nearest]
        d = np.sqrt(dists_sq[nearest])
        if d < 1e-6:
            continue
        direction = -to_predator / d
        escape[global_i] = direction * (escape_factor * accel_boost / (d * d))
