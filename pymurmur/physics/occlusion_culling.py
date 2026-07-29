"""Occlusion culling strategies (P4.6) — sequential dispatch, chunk
worker, and process-pool parallel dispatch for Stage 2 (per-observer
occlusion culling) of spherical_cap_occlusion_batched.

Level 0 — pure numpy/stdlib, no project imports beyond core.types.

Split out of occlusion.py (file-size split) — the per-observer
occlusion math (spherical_cap_occlusion, spherical_cap_occlusion_batched,
effective-radii helpers) stays in the original, which imports
_culling_sequential/_culling_parallel back from here for its Stage 2
dispatch.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

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


