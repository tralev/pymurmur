"""Young et al. 2013 consensus-robustness pipeline (H2, r_nodal, r_per_m,
convergence speed, marginal efficiency, cost-optimal m*).

Extracted from metrics.py (file-size split) — the whole H2 graph-
Laplacian pipeline, independent of FlockMetrics/MetricsCollector state.
"""
from __future__ import annotations

import numpy as np

# ── Expensive metrics (Phase 9) ──────────────────────────────────

def _knn_laplacian_matrix(positions: np.ndarray, m: int, tree=None):
    """Shared k-NN graph Laplacian builder for H₂, convergence speed,
    and the Lyapunov cross-validation (A5).

    Builds the m-nearest-neighbour graph, weights edges aᵢⱼ = 1/m (A8,
    Young et al. 2013's linear-consensus averaging — see compute_h2),
    symmetrizes via max(A, Aᵀ) (S1.8 — an edge exists at full weight if
    either endpoint's k-NN includes the other), and returns the graph
    Laplacian L = D − A as a sparse matrix.

    Returns None for the trivial N<2 or m<1 cases (nothing to build),
    or a zero matrix if no edges exist (degenerate k-NN query).
    """
    from scipy.spatial import cKDTree

    N = len(positions)
    if N < 2 or m < 1:
        return None

    k = min(m + 1, N)
    if tree is None:
        tree = cKDTree(positions)

    # Build sparse adjacency matrix
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # A8: uniform aᵢⱼ = 1/m weighting (Young et al. 2013) — not 1.0.
    edge_weight = 1.0 / m
    for i in range(N):
        _, idx = tree.query(positions[i], k=k)
        for j in idx:
            if j != i:
                rows.append(i)
                cols.append(j)
                data.append(edge_weight)

    from scipy.sparse import coo_matrix
    if not rows:
        return coo_matrix((N, N)).tocsr()

    # Build directed adjacency, then symmetrize
    A_dir = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A = A_dir.maximum(A_dir.T)

    # Laplacian L = D - A
    degrees = np.array(A.sum(axis=1)).flatten()
    D_diag = coo_matrix(
        (degrees, (np.arange(N), np.arange(N))), shape=(N, N)
    ).tocsr()
    return D_diag - A


def _knn_laplacian_eigenvalues(
    positions: np.ndarray, m: int, tree=None,
) -> np.ndarray | None:
    """Shared k-NN graph Laplacian eigensystem for H₂ and convergence speed.

    Builds L via `_knn_laplacian_matrix` and returns the ascending
    eigenvalues of L.

    Returns None for the trivial N<2 or m<1 cases (nothing to build).
    λ₀ = 0 always; λ₁ (algebraic connectivity) is 0 iff the graph is
    disconnected — callers interpret that as needed (H₂ treats it as
    +∞ disagreement; convergence speed treats it as literally 0, the
    mathematically correct value).

    Does not catch eigh() failures — callers each wrap their own call
    with the fallback appropriate to their own contract (H₂ and
    convergence speed disagree on what a computation failure should
    return, so a single shared fallback here would be wrong for one
    of them).
    """
    from scipy.linalg import eigh

    L = _knn_laplacian_matrix(positions, m, tree)
    if L is None:
        return None
    N = len(positions)
    if L.nnz == 0:
        return np.zeros(N)

    return eigh(L.toarray(), eigvals_only=True)


def compute_h2(positions: np.ndarray, m: int, tree=None) -> tuple[float, float]:
    """Compute H₂ consensus robustness for a given neighbour count m.

    Builds k-NN graph (k=m+1), symmetrizes adjacency, computes
    graph Laplacian eigenvalues, and returns H₂².

    A8 (Young et al. 2013): edges are weighted aᵢⱼ = 1/m, matching the
    paper's linear-consensus model dxᵢ/dt = Σ_{j∈Nᵢ} aᵢⱼ(xⱼ−xᵢ) + ξᵢ,
    where each bird averages over its m neighbours rather than summing
    them unweighted. The paper shows uniform 1/m weighting gives the
    best robustness — distance-proportional and order-based weights
    are strictly worse (Fig. S1).

    Args:
        positions: (N, 3) float32 array.
        m: number of neighbours per node.
        tree: optional pre-built cKDTree to avoid rebuild.

    Returns:
        (h2_squared, h2) — H₂² and H₂.
    """
    N = len(positions)
    if N < 2 or m < 1:
        return 0.0, 0.0

    try:
        eigenvals = _knn_laplacian_eigenvalues(positions, m, tree)
        if eigenvals is None:
            # "not rows" degenerate case from the helper (shouldn't
            # occur for N>=2, m>=1, kept for parity with the pre-refactor
            # early return).
            return float('inf'), float('inf')
        # Check if graph is disconnected: algebraic connectivity λ₁ ≈ 0
        # (more than one zero eigenvalue means >1 connected component)
        if eigenvals[1] < 1e-10:
            return float('inf'), float('inf')
        # H₂² = (1/2N) * Σ_{i≥2} 1/λ_i  (λ_0 = 0, skip)
        nonzero = eigenvals[1:][eigenvals[1:] > 1e-10]
        h2_sq = float(np.sum(1.0 / nonzero) / (2 * N))
        h2 = float(np.sqrt(h2_sq))
        return h2_sq, h2
    except Exception:
        return 0.0, 0.0


def compute_h2_lyapunov(positions: np.ndarray, m: int, tree=None) -> tuple[float, float]:
    """A5 (Young et al. 2013): H₂ via the Lyapunov/trace route.

    Eq. 1-5: dx/dt = -Lx + ξ, reduced to the (N-1)-dim disagreement
    subspace y = Qx (Q orthonormal, orthogonal to the all-ones
    consensus direction), giving dy/dt = -L̄y + Qξ with L̄ = QLQᵀ.
    The steady-state covariance Σ solves the Lyapunov equation
    L̄Σ + ΣL̄ᵀ = I, and H₂ = √Trace(Σ).

    This is a second, independent derivation of the same quantity
    `compute_h2` computes via the eigenvalue shortcut
    (H₂² = (1/2N)Σ_{i≥2} 1/λᵢ) — verified numerically identical across
    39 seed/N/m combinations (h2_sq == trace(Σ)/N to float precision).
    It exists purely as a cross-validation utility (not wired into
    FlockMetrics/MetricsCollector — `compute_h2` remains the metric
    actually used, since it's far cheaper: no Lyapunov solve needed).

    Note: the reduction/solve steps emit a benign RuntimeWarning
    (divide-by-zero/overflow/invalid-value in matmul, likely a BLAS
    denormal-number quirk) on some well-conditioned inputs with no
    NaN/Inf in the actual output — confirmed by direct inspection and
    by this function's own cross-validation against compute_h2
    matching to float precision, so it's suppressed here rather than
    propagated.

    Args:
        positions: (N, 3) float32 array.
        m: number of neighbours per node.
        tree: optional pre-built cKDTree to avoid rebuild.

    Returns:
        (h2_squared, h2) — same contract as compute_h2: (inf, inf) if
        disconnected, (0.0, 0.0) for N<2/m<1 or on any failure.
    """
    import warnings

    from scipy.linalg import null_space, solve_continuous_lyapunov

    N = len(positions)
    if N < 2 or m < 1:
        return 0.0, 0.0

    try:
        eigenvals = _knn_laplacian_eigenvalues(positions, m, tree)
        if eigenvals is None:
            return float('inf'), float('inf')
        if eigenvals[1] < 1e-10:
            return float('inf'), float('inf')

        L = _knn_laplacian_matrix(positions, m, tree)
        L_dense = L.toarray()

        # Q: (N-1)xN orthonormal basis orthogonal to the all-ones vector
        ones = np.ones((N, 1)) / np.sqrt(N)
        Q = null_space(ones.T).T

        with warnings.catch_warnings():
            # Benign BLAS/matmul RuntimeWarning (divide-by-zero/overflow/
            # invalid-value) observed on well-conditioned inputs with no
            # NaN/Inf in the actual output -- confirmed by direct
            # inspection, and by this function's own cross-validation
            # test agreeing with compute_h2 to float precision across
            # 39+ cases. Not a real numerical issue, so suppressed.
            warnings.simplefilter("ignore", RuntimeWarning)
            L_bar = Q @ L_dense @ Q.T
            Sigma = solve_continuous_lyapunov(L_bar, np.eye(N - 1))

        h2_sq = float(np.trace(Sigma) / N)
        h2 = float(np.sqrt(h2_sq))
        return h2_sq, h2
    except Exception:
        return 0.0, 0.0


def compute_r_nodal(h2: float, N: int) -> float:
    """A6 (Young et al. 2013): H₂ nodal robustness, R_nodal = √N / H₂.

    Normalizes H₂ by √N to remove system-size dependence, then
    inverts so *larger* means *more robust* (H₂ itself is a
    disagreement measure — smaller is better, which A6 finds
    unintuitive for a "robustness" figure).

    Args:
        h2: H₂ (not H₂²) — pass compute_h2's second return value.
        N: flock size.

    Returns:
        R_nodal. 0.0 when h2 is non-finite (disconnected graph — the
        paper's own stated convention) or <=0 (degenerate H₂ from
        N<2/failure, which is this codebase's "nothing to compute"
        sentinel, not a real zero-disagreement state).
    """
    if not np.isfinite(h2) or h2 <= 0.0 or N <= 0:
        return 0.0
    return float(np.sqrt(N) / h2)


def compute_r_per_m(r_nodal: float, m: int) -> float:
    """A7 (Young et al. 2013): robustness per neighbour, R_per_m = R_nodal / m.

    Accounts for sensing cost — each additional neighbour requires
    neurological/sensory effort, so raw R_nodal alone overstates the
    benefit of large m. The paper reports this peaks at an interior
    m*≈6-7 (see find_m_star_by_sensing_cost); empirically, this does
    NOT reproduce in this codebase's H₂ implementation (verified: the
    curve is monotonically decreasing from the connectivity threshold
    onward for both random and simulated flocks) — see
    test_h2.py::TestRPerM for the honestly-xfailed claim.

    Args:
        r_nodal: R_nodal (pass compute_r_nodal's output).
        m: neighbour count the r_nodal value was computed at.

    Returns:
        R_per_m. 0.0 for m<=0 (division guard).
    """
    if m <= 0:
        return 0.0
    return float(r_nodal / m)


def find_m_star_by_sensing_cost(positions: np.ndarray, tree=None) -> tuple[int, float]:
    """A7 (Young et al. 2013): m* = argmax_m R_per_m(m), the sensing-
    cost-optimal neighbour count.

    Distinct from find_optimal_m (which minimizes the linear-penalty
    cost J(m) = H₂(m) + 0.06·m) — a different objective on the same
    underlying H₂(m) curve. Mirrors find_optimal_m's scan structure.

    Args:
        positions: (N, 3) float32 array.
        tree: optional pre-built cKDTree to avoid rebuild.

    Returns:
        (m*, R_per_m at m*). (2, 0.0) if no m in [2, 20] connects.
    """
    N = len(positions)
    best_m = 2
    best_r_per_m = 0.0
    for m in range(2, min(21, N)):
        _, h2 = compute_h2(positions, m, tree)
        if not np.isfinite(h2):
            continue
        r_nodal = compute_r_nodal(h2, N)
        r_per_m = compute_r_per_m(r_nodal, m)
        if r_per_m > best_r_per_m:
            best_r_per_m = r_per_m
            best_m = m
    return best_m, best_r_per_m


def compute_convergence_speed(positions: np.ndarray, m: int, tree=None) -> float:
    """A10 (Young et al. 2013): consensus convergence speed λ₂(L).

    The algebraic connectivity (Fiedler value) of the same m-nearest-
    neighbour graph Laplacian H₂ is built from — the second-smallest
    eigenvalue, λ₂. Since L is always real and symmetric here (S1.8's
    max-form symmetrization), this is exactly Re(λ₂(L)) with no
    complex-number handling needed.

    This is deliberately a *different* quantity from H₂: the paper's
    headline result is that speed and robustness trade off oppositely
    as m grows — λ₂ increases monotonically with m (faster consensus,
    no interior optimum), while H₂ decreases (this file's compute_h2).
    Real flocks are shaped by robustness, not raw convergence speed —
    this metric exists to make that contrast observable, not to change
    which m* gets selected (find_optimal_m is unaffected).

    λ₂ = 0 exactly when the graph is disconnected — the mathematically
    correct value (no special-casing needed, unlike H₂'s +∞).

    Args:
        positions: (N, 3) float32 array.
        m: number of neighbours per node.
        tree: optional pre-built cKDTree to avoid rebuild.

    Returns:
        λ₂(L) — 0.0 for N<2, m<1, a disconnected graph, or on any
        eigendecomposition failure (matches compute_h2's own
        degrade-to-a-safe-scalar policy for that case, though its
        failure value there is 0.0 too — the two happen to agree here
        even though their disconnected-graph values differ).
    """
    try:
        eigenvals = _knn_laplacian_eigenvalues(positions, m, tree)
        if eigenvals is None or len(eigenvals) < 2:
            return 0.0
        return float(max(0.0, eigenvals[1]))
    except Exception:
        return 0.0


def find_optimal_m(positions: np.ndarray, tree=None) -> tuple[int, float]:
    """Find cost-optimal neighbour count m*.

    J(m) = H₂(m) + 0.06·m — minimize over m ∈ [2, 20].
    Optionally pass a pre-built cKDTree to avoid rebuilds.
    Returns (m*, H₂) for reuse.
    """
    best_m = 6
    best_h2 = 0.0
    best_j = float('inf')
    for m in range(2, min(21, len(positions))):
        _, h2 = compute_h2(positions, m, tree)
        # Skip disconnected graphs (returned as inf)
        if not np.isfinite(h2):
            continue
        j = h2 + 0.06 * m
        if j < best_j:
            best_j = j
            best_m = m
            best_h2 = h2
    return best_m, best_h2


def find_connectivity_threshold(
    positions: np.ndarray, m_max: int = 20, tree=None,
) -> int | None:
    """A9 (Young et al. 2013): the smallest m at which the m-nearest-
    neighbour graph is connected (H₂ finite).

    Young et al. report m≥5 as the empirical connectivity threshold
    across 394 field-observed starling-flock snapshots (N=440-2600) --
    almost always connected at m=5, and m=1,2 almost always disconnected.
    This is the corresponding measurement on this codebase's own
    positions (simulated or synthetic), mirroring find_optimal_m's/
    find_m_star_by_sensing_cost's scan structure.

    Args:
        positions: (N, 3) float32 array.
        m_max: largest m to try before giving up.
        tree: optional pre-built cKDTree to avoid rebuild.

    Returns:
        The first m in [1, m_max] where compute_h2 reports a connected
        graph (finite H₂), or None if nothing in range connects.
    """
    for m in range(1, min(m_max + 1, len(positions))):
        _, h2 = compute_h2(positions, m, tree)
        if np.isfinite(h2):
            return m
    return None


# ── P9.6: η(m) Marginal efficiency ────────────────────────────

def _compute_eta_m(
    positions: np.ndarray,
    tree,
    optimal_m: int,
) -> float | None:
    """P9.6: Marginal efficiency η(m) at the optimal m*.

    η(m) = (H₂(m₀) − H₂(m)) / (m − m₀)

    Uses m₀ = max(2, optimal_m − 2) as the baseline neighbour count.
    Returns +inf when m first connects the graph (H₂ drops from inf
    to finite). Returns 0.0 when both m₀ and m are disconnected
    (P0.13 inf fix ensures this).

    Args:
        positions: (N, 3) float32 array.
        tree: pre-built cKDTree.
        optimal_m: the cost-optimal m*.

    Returns:
        η(m*) — marginal improvement in robustness per extra neighbour,
        or None if N < 4.
    """
    N = len(positions)
    if N < 4 or optimal_m < 3:
        return None

    m0 = max(2, optimal_m - 2)
    if m0 >= optimal_m:
        return None

    _, h2_opt = compute_h2(positions, optimal_m, tree)
    _, h2_base = compute_h2(positions, m0, tree)

    # Both disconnected → η = 0
    if not np.isfinite(h2_opt) and not np.isfinite(h2_base):
        return 0.0
    # Transition from disconnected→connected → η = +∞
    if not np.isfinite(h2_base) and np.isfinite(h2_opt):
        return float('inf')
    # Connected→disconnected (shouldn't happen, but guard)
    if np.isfinite(h2_base) and not np.isfinite(h2_opt):
        return 0.0

    return (h2_base - h2_opt) / (optimal_m - m0)


