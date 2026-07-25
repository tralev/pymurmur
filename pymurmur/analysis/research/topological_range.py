"""A13 (Young et al. 2013): independent topological interaction-range
estimator, distinct from H2's k-NN Laplacian robustness pipeline.

The paper compares H2-derived m* against an *independently estimated*
topological range n_c (Ballerini et al.'s field-study methodology,
cited at /tmp/1302.3195v1.txt:357) and finds no significant
correlation between the two (r~=-0.24, p~=0.46) -- evidence the two
analyses measure different flock properties. n_c there is a real-data
field measurement, not derivable from the same H2 pipeline; this
module implements the well-established velocity-fluctuation
correlation-function methodology (Cavagna et al.) as a comparable
*independent* estimator computable from this codebase's own simulated
flocks, for cross-checking against A7's H2-derived m_star_sensing.
"""
from __future__ import annotations

import numpy as np


def compute_topological_correlation_range(
    positions: np.ndarray,
    velocities: np.ndarray,
    max_rank: int = 40,
    tree=None,
) -> float | None:
    """Topological velocity-fluctuation correlation range n_c.

    C(k) = mean over birds i of <dv_i . dv_{i's k-th nearest neighbour}> / <|dv|^2>,
    where dv_i = v_i - mean(v) (velocity fluctuation about the flock
    mean). C(0) = 1 exactly by construction. n_c is the first
    (possibly fractional, via linear interpolation) neighbour rank k
    where C(k) crosses zero -- birds within n_c ranks of each other
    still have correlated velocity fluctuations; beyond it they don't.

    Binned by neighbour *rank* (topological distance), not metric
    distance -- matching the "topological not metric" premise this
    whole line of work is built on, and Ballerini et al.'s own
    methodology.

    Args:
        positions: (N, 3) float32 array.
        velocities: (N, 3) float32 array, same ordering as positions.
        max_rank: maximum neighbour rank to search for a zero-crossing.
        tree: optional pre-built cKDTree to avoid rebuild.

    Returns:
        n_c (float, possibly fractional). None if N<3, max_rank<1,
        velocities have no variance (degenerate -- nothing to
        correlate), or C(k) never crosses zero within max_rank
        (observed for highly-polarized flocks in this codebase --
        velocity fluctuations stay correlated across the whole rank
        range tested, consistent with the "scale-free correlation"
        phenomenon reported for real starling murmurations).
    """
    from scipy.spatial import cKDTree

    N = len(positions)
    if N < 3 or max_rank < 1:
        return None

    dv = velocities - velocities.mean(axis=0)
    norm_sq = float(np.mean(np.sum(dv * dv, axis=1)))
    if norm_sq < 1e-12:
        return None

    if tree is None:
        tree = cKDTree(positions)

    k = min(max_rank + 1, N)
    _, idx = tree.query(positions, k=k)  # idx[:, 0] == self

    c_prev = 1.0  # C(0) = 1 exactly: mean(dv_i . dv_i) / mean(|dv|^2)
    for rank in range(1, k):
        nbr = idx[:, rank]
        c = float(np.mean(np.sum(dv * dv[nbr], axis=1)) / norm_sq)
        if c_prev > 0.0 and c <= 0.0:
            frac = c_prev / (c_prev - c)
            return float((rank - 1) + frac)
        c_prev = c

    return None
