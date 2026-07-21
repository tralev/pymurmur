"""3D spherical-cap occlusion — Pearce et al. 2014.

Level 0 — pure numpy. No project imports beyond core.types.
Extended to 3D from the original 2D model.

Computes delta (boundary-length-weighted projection direction),
visible neighbors (closest-first, occluded birds excluded),
and internal opacity Theta in [0,1] (probabilistic union of solid angles).

Phase 1 (P1.1-P1.5) — scientific correctness:
  - P1.1: True occlusion culling — neighbour j is visible iff
    not blind(j) and for all k already visible: dot(d_j, d_k) < cos(alpha_k)
  - P1.2: Theta as probabilistic union: Omega_j = 2*pi*(1 - cos(alpha_j)),
    Theta = 1 - prod_{j in visible} (1 - Omega_j / (4*pi))
  - P1.3: delta boundary-length weighted: sum(sin(alpha_j) * d_j) / sum(sin(alpha_j))
  - P1.4: Exact alpha = asin(min(b_eff/d, 1)) — replaces small-angle approx
  - P1.5: Candidate cutoff at max_candidates nearest neighbours

I1.3 (array kernel): spherical_cap_occlusion uses pre-allocated numpy arrays
instead of Python lists.  spherical_cap_occlusion_batched processes all
observers at once taking (N, sigma, 3) neighbour tensors — zero Python
object allocations in the hot path.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

MAX_OCCLUSION_CANDIDATES = 64
_MIN_PARALLEL_OBSERVERS = 100  # threshold below which parallel overhead isn't worth it


# ---------------------------------------------------------------------------
# Single-observer (backward compat — uses pre-allocated numpy arrays)
# ---------------------------------------------------------------------------

def spherical_cap_occlusion(
    observer_pos: np.ndarray,
    observer_vel: np.ndarray,
    neighbour_positions: np.ndarray,
    neighbour_velocities: np.ndarray,
    boid_size: float = 9.0,
    blind_cos: Optional[float] = None,
    anisotropy: float = 1.0,
    max_candidates: int = MAX_OCCLUSION_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute delta, visible neighbours, and Theta for one observer.

    Args:
        observer_pos: (3,) float32 — observer position.
        observer_vel: (3,) float32 — observer velocity.
        neighbour_positions: (M, 3) float32 — neighbour positions.
        neighbour_velocities: (M, 3) float32 — neighbour velocities.
        boid_size: body radius for cap size calculation.
        blind_cos: cos(half blind angle), neighbours behind this excluded.
        anisotropy: body axis ratio a/b (1.0 = isotropic).
        max_candidates: cap on nearest neighbours considered.

    Returns:
        delta: (3,) float32 — |delta| in [0,1]; ~1 at edge, →0 when surrounded.
        visible_idx: (K,) int32 — indices of visible neighbours (closest-first).
        theta: float — Theta in [0,1].
    """
    M = len(neighbour_positions)
    if M == 0:
        return np.zeros(3, dtype=np.float32), np.array([], dtype=np.int32), 0.0

    obs_forward = observer_vel / (np.linalg.norm(observer_vel) + 1e-10)

    # 1. Distances, sort closest-first, cap
    diffs = neighbour_positions - observer_pos
    dists = np.linalg.norm(diffs, axis=1)
    order = np.argsort(dists)[:max_candidates]

    # 2. Pre-computed effective radii for all candidates
    b_effs = _compute_effective_radii(
        observer_pos, neighbour_positions, neighbour_velocities,
        order, boid_size, anisotropy,
    )

    # --- Pre-allocated numpy arrays (I1.3: no Python lists) ---
    max_v = min(max_candidates, M)
    _visible = np.empty(max_v, dtype=np.int32)
    _visible_dirs = np.empty((max_v, 3), dtype=np.float32)
    _visible_cos_alpha = np.empty(max_v, dtype=np.float64)
    _visible_sin_alpha = np.empty(max_v, dtype=np.float64)
    n_vis = 0

    for j_idx, j in enumerate(order):
        d = dists[j]
        if d < 1e-6:
            continue  # skip self

        direction = diffs[j] / d

        # Blind angle check
        if blind_cos is not None:
            cos_angle = np.dot(direction, -obs_forward)
            if cos_angle >= blind_cos:
                continue  # behind observer

        # Exact angular radius (P1.4)
        cap_ratio = b_effs[j_idx] / d
        if cap_ratio >= 1.0:
            alpha = math.pi / 2
            cos_alpha = 0.0
            sin_alpha = 1.0
        else:
            alpha = math.asin(cap_ratio)
            cos_alpha = math.cos(alpha)
            sin_alpha = math.sin(alpha)

        # True occlusion culling (P1.1): vectorised over already-visible caps
        if n_vis > 0:
            dots = _visible_dirs[:n_vis] @ direction  # (n_vis,)
            if np.any(dots >= _visible_cos_alpha[:n_vis]):
                continue  # occluded

        # Visible — store in pre-allocated arrays
        _visible[n_vis] = j
        _visible_dirs[n_vis] = direction
        _visible_cos_alpha[n_vis] = cos_alpha
        _visible_sin_alpha[n_vis] = sin_alpha
        n_vis += 1

    # 3. Compute delta and Theta from visible neighbours
    if n_vis == 0:
        return np.zeros(3, dtype=np.float32), np.array([], dtype=np.int32), 0.0

    visible_arr = _visible[:n_vis].copy()

    # delta = sum(sin(alpha_j) * d_j) / sum(sin(alpha_j))  (P1.3)
    sin_alpha_sum = float(np.sum(_visible_sin_alpha[:n_vis]))
    delta = np.zeros(3, dtype=np.float32)
    for k in range(n_vis):
        delta += _visible_sin_alpha[k] * _visible_dirs[k]
    if sin_alpha_sum > 1e-10:
        delta /= sin_alpha_sum

    # Theta = 1 - prod(1 - Omega_j / (4*pi))  (P1.2)
    remaining = 1.0
    for k in range(n_vis):
        cos_a = float(_visible_cos_alpha[k])
        omega = 2.0 * math.pi * (1.0 - cos_a)
        remaining *= (1.0 - omega / (4.0 * math.pi))
    theta = 1.0 - remaining
    theta = max(0.0, min(1.0, theta))

    return delta.astype(np.float32), visible_arr.astype(np.int32), float(theta)


# ---------------------------------------------------------------------------
# Batched (I1.3): all observers at once, zero Python allocations in hot path
# ---------------------------------------------------------------------------

def spherical_cap_occlusion_batched(
    obs_positions: np.ndarray,
    obs_velocities: np.ndarray,
    nbr_positions: np.ndarray,
    nbr_velocities: np.ndarray,
    boid_size: float = 9.0,
    blind_cos: Optional[float] = None,
    anisotropy: float = 1.0,
    max_candidates: int = MAX_OCCLUSION_CANDIDATES,
    valid_mask: Optional[np.ndarray] = None,
    n_jobs: int = 1,
    min_parallel: int = _MIN_PARALLEL_OBSERVERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched spherical-cap occlusion for all observers.

    Args:
        obs_positions: (N, 3) float32 — observer positions.
        obs_velocities: (N, 3) float32 — observer velocities.
        nbr_positions: (N, K, 3) float32 — per-observer neighbour positions.
        nbr_velocities: (N, K, 3) float32 — per-observer neighbour velocities.
        boid_size: body radius for cap size calculation.
        blind_cos: cos(half blind angle), neighbours behind this excluded.
        anisotropy: body axis ratio a/b (1.0 = isotropic).
        max_candidates: cap on nearest neighbours considered.
        valid_mask: (N, K) bool — False entries excluded from sort.
        n_jobs: number of parallel workers for Stage 2 culling
            (1 = sequential, -1 = all cores, default 1).
        min_parallel: skip parallel when N < min_parallel (default 100).

    Returns:
        delta: (N, 3) float32 — |delta| in [0,1].
        visible_mask: (N, K) bool — True where neighbour is visible.
        theta: (N,) float32 — Theta in [0,1].
    """
    N, K = nbr_positions.shape[:2]
    if N == 0 or K == 0:
        return (
            np.zeros((N, 3), dtype=np.float32),
            np.zeros((N, K), dtype=bool),
            np.zeros(N, dtype=np.float32),
        )

    # --- Stage 1: batch distance, direction, blind angle, effective radii ---

    diffs = nbr_positions - obs_positions[:, np.newaxis, :]  # (N, K, 3)
    dists = np.linalg.norm(diffs, axis=2)                      # (N, K)

    # Exclude invalid neighbours (e.g. -1 sentinels) from sort
    if valid_mask is not None:
        dists = np.where(valid_mask, dists, np.inf)

    # Sort neighbours by distance, keep closest max_candidates
    sort_order = np.argsort(dists, axis=1)[:, :max_candidates]  # (N, M)  M=min(K, max_candidates)
    M = sort_order.shape[1]

    # Gather sorted data
    gather_i = np.arange(N)[:, np.newaxis]                     # (N, 1)
    sorted_dists = dists[gather_i, sort_order]                  # (N, M)
    sorted_diffs = diffs[gather_i, sort_order]                  # (N, M, 3)
    sorted_vels = nbr_velocities[gather_i, sort_order]          # (N, M, 3)

    # Directions
    dirs = sorted_diffs / (sorted_dists[:, :, np.newaxis] + 1e-10)  # (N, M, 3)

    # Observer forward vectors
    obs_forward = obs_velocities / (
        np.linalg.norm(obs_velocities, axis=1, keepdims=True) + 1e-10
    )  # (N, 3)

    # Effective radii (batched)
    b_effs = _compute_effective_radii_batched(
        obs_positions, sorted_diffs, sorted_vels, boid_size, anisotropy,
    )  # (N, M)

    # --- Stage 2: per-observer occlusion culling ---
    if n_jobs > 1 and N >= min_parallel:
        delta, visible_mask, theta = _culling_parallel(
            sorted_dists, dirs, b_effs, obs_forward, sort_order,
            blind_cos, K, n_jobs,
        )
    else:
        delta, visible_mask, theta = _culling_sequential(
            sorted_dists, dirs, b_effs, obs_forward, sort_order,
            blind_cos, M, N, K,
        )

    return delta.astype(np.float32), visible_mask, theta.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 2 helpers: sequential culling, parallel dispatch, chunk worker (P4.6)
# ---------------------------------------------------------------------------

def _culling_sequential(
    sorted_dists: np.ndarray,
    dirs: np.ndarray,
    b_effs: np.ndarray,
    obs_forward: np.ndarray,
    sort_order: np.ndarray,
    blind_cos: Optional[float],
    M: int,
    N: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential culling — delegates to _occlusion_culling_chunk with full arrays."""
    return _occlusion_culling_chunk(
        sorted_dists, dirs, b_effs, obs_forward, sort_order, blind_cos, K,
    )


def _occlusion_culling_chunk(
    chunk_sorted_dists: np.ndarray,
    chunk_dirs: np.ndarray,
    chunk_b_effs: np.ndarray,
    chunk_obs_forward: np.ndarray,
    chunk_sort_order: np.ndarray,
    blind_cos: Optional[float],
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process one chunk of observers (module-level for picklability).

    Args are sliced views for this chunk's observers:
        chunk_sorted_dists: (chunk_N, M) float64
        chunk_dirs:          (chunk_N, M, 3) float32
        chunk_b_effs:        (chunk_N, M) float32
        chunk_obs_forward:   (chunk_N, 3) float32
        chunk_sort_order:    (chunk_N, M) int64
        blind_cos: cos(half blind angle) or None
        K: original neighbour count (width of visible_mask)

    Returns:
        delta:        (chunk_N, 3) float32
        visible_mask: (chunk_N, K) bool
        theta:        (chunk_N,) float32
    """
    chunk_N, M = chunk_sorted_dists.shape
    delta = np.zeros((chunk_N, 3), dtype=np.float32)
    theta = np.zeros(chunk_N, dtype=np.float32)
    visible_mask = np.zeros((chunk_N, K), dtype=bool)

    v_dirs = np.empty((M, 3), dtype=np.float32)
    v_cos_a = np.empty(M, dtype=np.float64)
    v_sin_a = np.empty(M, dtype=np.float64)
    v_cols = np.empty(M, dtype=np.int32)

    for i in range(chunk_N):
        n_vis = 0
        fwd_i = chunk_obs_forward[i]

        for j in range(M):
            d = chunk_sorted_dists[i, j]
            if np.isinf(d) or d < 1e-6:
                continue

            direction = chunk_dirs[i, j]

            if blind_cos is not None:
                cos_angle = np.dot(direction, -fwd_i)
                if cos_angle >= blind_cos:
                    continue

            cap_ratio = chunk_b_effs[i, j] / d
            if cap_ratio >= 1.0:
                cos_alpha = 0.0
                sin_alpha = 1.0
            else:
                alpha = math.asin(float(cap_ratio))
                cos_alpha = math.cos(alpha)
                sin_alpha = math.sin(alpha)

            if n_vis > 0:
                dots = v_dirs[:n_vis] @ direction
                if np.any(dots >= v_cos_a[:n_vis]):
                    continue

            v_dirs[n_vis] = direction
            v_cos_a[n_vis] = cos_alpha
            v_sin_a[n_vis] = sin_alpha
            v_cols[n_vis] = chunk_sort_order[i, j]
            n_vis += 1

        if n_vis == 0:
            continue

        sin_sum = float(np.sum(v_sin_a[:n_vis]))
        d_i = np.zeros(3, dtype=np.float32)
        for k in range(n_vis):
            d_i += v_sin_a[k] * v_dirs[k]
        if sin_sum > 1e-10:
            d_i /= sin_sum
        delta[i] = d_i

        remaining = 1.0
        for k in range(n_vis):
            omega = 2.0 * math.pi * (1.0 - float(v_cos_a[k]))
            remaining *= (1.0 - omega / (4.0 * math.pi))
        theta[i] = max(0.0, min(1.0, 1.0 - remaining))

        for k in range(n_vis):
            orig_j = v_cols[k]
            if 0 <= orig_j < K:
                visible_mask[i, orig_j] = True

    return delta, visible_mask, theta


def _culling_parallel(
    sorted_dists: np.ndarray,
    dirs: np.ndarray,
    b_effs: np.ndarray,
    obs_forward: np.ndarray,
    sort_order: np.ndarray,
    blind_cos: Optional[float],
    K: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel Stage 2: split N observers across workers.

    P4.6: Uses concurrent.futures.ProcessPoolExecutor (stdlib, no new deps).
    Each worker gets a contiguous slice of observers and returns its portion
    of delta/visible_mask/theta. Results are concatenated in the main process.
    """
    from concurrent.futures import ProcessPoolExecutor

    N = sorted_dists.shape[0]

    if n_jobs < 1:
        n_jobs = max(1, os.cpu_count() or 1)
    n_jobs = min(n_jobs, N)  # don't use more workers than observers

    # Split indices into roughly equal chunks
    chunk_size = max(1, (N + n_jobs - 1) // n_jobs)
    tasks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tasks.append((
            sorted_dists[start:end],
            dirs[start:end],
            b_effs[start:end],
            obs_forward[start:end],
            sort_order[start:end],
            blind_cos,
            K,
        ))

    with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks))) as executor:
        futures = [
            executor.submit(_occlusion_culling_chunk, *task)
            for task in tasks
        ]
        results = [f.result() for f in futures]

    # Concatenate results from all chunks
    delta = np.concatenate([r[0] for r in results], axis=0)
    visible_mask = np.concatenate([r[1] for r in results], axis=0)
    theta = np.concatenate([r[2] for r in results], axis=0)

    return delta, visible_mask, theta


# ---------------------------------------------------------------------------
# SoA adapter (backward compat)
# ---------------------------------------------------------------------------

def spherical_cap_occlusion_soa(
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
    nbr_positions: np.ndarray,
    nbr_velocities: np.ndarray,
    boid_size: float = 9.0,
    blind_cos: Optional[float] = None,
    anisotropy: float = 1.0,
    max_candidates: int = MAX_OCCLUSION_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Adapter: delegates to spherical_cap_occlusion (single-observer API)."""
    return spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_positions, nbr_velocities,
        boid_size, blind_cos, anisotropy, max_candidates,
    )


# ---------------------------------------------------------------------------
# Effective radii (single + batched)
# ---------------------------------------------------------------------------

def _compute_effective_radii(
    observer_pos: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    order: np.ndarray,
    boid_size: float,
    anisotropy: float,
) -> np.ndarray:
    """Compute effective body radius for each candidate (single-observer)."""
    n = len(order)
    result = np.full(n, boid_size, dtype=np.float32)
    if anisotropy == 1.0:
        return result
    for i, j in enumerate(order):
        v = velocities[j]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            continue
        v_dir = v / v_norm
        d_vec = positions[j] - observer_pos
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-6:
            continue
        d_dir = d_vec / d_norm
        cos_psi = abs(np.dot(d_dir, v_dir))
        sin_psi = math.sqrt(max(0.0, 1.0 - cos_psi * cos_psi))
        result[i] = math.sqrt(
            (boid_size * sin_psi) ** 2 +
            (boid_size / anisotropy * cos_psi) ** 2
        )
    return result


def _compute_effective_radii_batched(
    obs_positions: np.ndarray,
    diffs: np.ndarray,
    velocities: np.ndarray,
    boid_size: float,
    anisotropy: float,
) -> np.ndarray:
    """Compute effective body radius for all observer-neighbour pairs.

    Args:
        obs_positions: (N, 3) — observer positions.
        diffs: (N, M, 3) — vectors from observer to neighbour.
        velocities: (N, M, 3) — neighbour velocities.
        boid_size: base body radius.
        anisotropy: body axis ratio a/b.

    Returns:
        b_effs: (N, M) float32 — effective radii.
    """
    N, M = diffs.shape[:2]
    result = np.full((N, M), boid_size, dtype=np.float32)
    if anisotropy == 1.0:
        return result

    # Normalise neighbour velocities
    vel_norms = np.linalg.norm(velocities, axis=2)  # (N, M)
    vel_valid = vel_norms > 1e-6
    v_dir = np.zeros_like(velocities)
    np.divide(velocities, vel_norms[:, :, np.newaxis], where=vel_valid[:, :, np.newaxis],
              out=v_dir)

    # Normalise line-of-sight directions
    d_norms = np.linalg.norm(diffs, axis=2)  # (N, M)
    d_valid = d_norms > 1e-6
    d_dir = np.zeros_like(diffs)
    np.divide(diffs, d_norms[:, :, np.newaxis], where=d_valid[:, :, np.newaxis],
              out=d_dir)

    # cos(psi) = |d_dir . v_dir|
    cos_psi = np.abs(np.sum(d_dir * v_dir, axis=2))  # (N, M)
    sin_psi = np.sqrt(np.maximum(0.0, 1.0 - cos_psi * cos_psi))

    # b_eff = sqrt((b * sin(psi))^2 + (b/a * cos(psi))^2)
    b_eff = np.sqrt(
        (boid_size * sin_psi) ** 2 +
        (boid_size / anisotropy * cos_psi) ** 2
    )

    # Only update where both velocity and direction are valid
    both_valid = vel_valid & d_valid
    result[both_valid] = b_eff[both_valid]
    return result
