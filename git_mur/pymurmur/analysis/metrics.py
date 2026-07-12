"""Scientific observables and metrics collection.

Level 1 — 15 observables, split into fast (O(N)) and expensive (O(N²)).
Gated behind config.metrics_detail_level and config.metrics_interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import TYPE_CHECKING

from ..core.types import Vec3

if TYPE_CHECKING:
    from ..physics.flock import PhysicsFlock


@dataclass
class FlockMetrics:
    """Container for all 15 scientific observables.

    Fast metrics (O(N)) computed every frame at detail_level >= 1.
    Expensive metrics (O(N²)) computed every metrics_interval frames at detail_level >= 2.
    """

    # ── Fast (O(N), every frame) ─────────────────────────────────
    alpha: float = 0.0            # order parameter |Σ v̂| / N
    theta: float = 0.0            # internal opacity Θ
    theta_prime: float = 0.0      # external opacity Θ'
    angular_momentum: Vec3 = np.zeros(3, dtype=np.float32)
    dispersion: float = 0.0       # ⟨|r − r_com|⟩
    speed_avg: float = 0.0        # ⟨|v|⟩
    force_avg: float = 0.0        # ⟨|a|⟩
    power_avg: float = 0.0        # ⟨|a·v|⟩
    local_spacing: float = 0.0    # median k=7 neighbour distance

    # ── Expensive (O(N²) or O(N log N), gated) ───────────────────
    h2: float | None = None       # H₂ consensus robustness
    tau_rho: float | None = None  # density autocorrelation time
    msd: float | None = None      # mean squared displacement
    gyration_radius: float | None = None   # trimmed RMS
    aspect_ratio: float | None = None      # flock elongation (PCA)
    thickness_ratio: float | None = None   # flock flatness (PCA)
    optimal_m: float | None = None         # cost-optimal neighbour count m*


class MetricsCollector:
    """Computes and caches flock metrics each frame.

    Expensive metrics (H2, shape, gyration) can optionally be
    computed in a background thread via use_async=True.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self._history: list[FlockMetrics] = []
        self._position_snapshots: list[np.ndarray] = []  # for MSD
        self._density_history: list[np.ndarray] = []     # for tau_rho
        self._detail_level = getattr(config, 'metrics_detail_level', 1) if config else 1
        self._interval = getattr(config, 'metrics_interval', 60) if config else 60
        self._theta_prime_grid = 30  # voxel resolution for external opacity
        self._async_result: object | None = None  # Future from background thread
        self._async_gen: int = 0  # generation counter to detect stale results

    def collect(self, flock: PhysicsFlock, frame: int) -> None:
        """Compute metrics for the current frame."""
        active = flock.active
        n = active.sum()
        if n == 0:
            return

        m = FlockMetrics()

        # ── Fast metrics ──────────────────────────────────────────
        positions = flock.positions[active]
        velocities = flock.velocities[active]

        # Order parameter α
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dirs = velocities / norms
        m.alpha = float(np.linalg.norm(dirs.sum(axis=0)) / n)

        # Internal opacity Θ
        m.theta = float(np.mean(flock.last_theta[active]))

        # External opacity Θ' (fast: O(N) grid rasterization)
        m.theta_prime = compute_theta_prime(positions, self._theta_prime_grid)

        # Centre of mass and dispersion
        com = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - com, axis=1)
        m.dispersion = float(np.mean(dists))

        # Speed / force / power
        speeds = np.linalg.norm(velocities, axis=1)
        m.speed_avg = float(np.mean(speeds))

        accs = flock.accelerations[active]
        acc_mags = np.linalg.norm(accs, axis=1)
        m.force_avg = float(np.mean(acc_mags))
        m.power_avg = float(np.mean(np.abs(np.sum(accs * velocities, axis=1))))

        # Angular momentum: ⟨r × v⟩ / N
        m.angular_momentum = np.mean(np.cross(positions, velocities), axis=0)

        # ── Expensive metrics (gated) ─────────────────────────────

        if self._detail_level >= 2 and frame % self._interval == 0:
            # Pick up completed async result from previous interval
            if self._async_result is not None:
                self._collect_async_result(m)
                # Fire async for current frame
                self._start_async_expensive(positions.copy(), n)
            else:
                # First interval frame: compute synchronously so we have immediate results
                _compute_expensive_metrics(m, positions, n)

            # MSD: compute from accumulated snapshots (fast, sync is fine)
            if len(self._position_snapshots) >= 2:
                m.msd = compute_msd(self._position_snapshots)

        # MSD: snapshot positions every interval frame
        # tau_rho: snapshot density histogram every interval frame
        if frame % self._interval == 0:
            self._position_snapshots.append(positions.copy())
            # Store coarse density histogram for autocorrelation
            if self._detail_level >= 2:
                bounds = np.array([positions.min(axis=0), positions.max(axis=0)])
                hist = _density_histogram(positions, bounds, self._theta_prime_grid)
                self._density_history.append(hist)

        # tau_rho: compute from accumulated density histograms
        if self._detail_level >= 2 and len(self._density_history) >= 4:
            m.tau_rho = compute_tau_rho(self._density_history)

        self._history.append(m)

    def _start_async_expensive(self, positions: np.ndarray, n: int) -> None:
        """Fire expensive metric computation in a background thread.

        Uses a generation counter so that if a previous async is still
        running, its stale result won't overwrite the new one.
        """
        import threading
        self._async_gen += 1
        gen = self._async_gen
        self._async_result = {"done": False, "data": None}

        def _worker() -> None:
            m = FlockMetrics()
            _compute_expensive_metrics(m, positions, n)
            # Only store result if this is still the current generation
            if self._async_gen == gen:
                self._async_result = {"done": True, "data": m}

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _collect_async_result(self, m: FlockMetrics) -> None:
        """Pick up completed async result if ready, skip if still running."""
        result = self._async_result
        if result is None:
            return
        if not result.get("done"):  # type: ignore[union-attr]
            return  # still computing, skip this frame
        async_m = result.get("data")  # type: ignore[union-attr]
        if async_m is not None:
            m.h2 = async_m.h2
            m.optimal_m = async_m.optimal_m
            m.local_spacing = async_m.local_spacing
            m.aspect_ratio = async_m.aspect_ratio
            m.thickness_ratio = async_m.thickness_ratio
            m.gyration_radius = async_m.gyration_radius
        self._async_result = None

    def snapshot(self) -> FlockMetrics:
        """Return the most recent metrics snapshot."""
        return self._history[-1] if self._history else FlockMetrics()

    @property
    def history(self) -> list[FlockMetrics]:
        return self._history


# ── Expensive metrics (Phase 9) ──────────────────────────────────

def compute_h2(positions: np.ndarray, m: int, tree=None) -> tuple[float, float]:
    """Compute H₂ consensus robustness for a given neighbour count m.

    Builds k-NN graph (k=m+1), symmetrizes adjacency, computes
    graph Laplacian eigenvalues, and returns H₂².

    Args:
        positions: (N, 3) float32 array.
        m: number of neighbours per node.
        tree: optional pre-built cKDTree to avoid rebuild.

    Returns:
        (h2_squared, h2) — H₂² and H₂.
    """
    from scipy.spatial import cKDTree
    from scipy.linalg import eigh

    N = len(positions)
    if N < 2 or m < 1:
        return 0.0, 0.0

    k = min(m + 1, N)
    if tree is None:
        tree = cKDTree(positions)

    # Build sparse adjacency matrix
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(N):
        _, idx = tree.query(positions[i], k=k)
        for j in idx:
            if j != i:
                rows.append(i)
                cols.append(j)
                data.append(1.0)

    if not rows:
        return 0.0, 0.0

    # Build directed adjacency, then symmetrize
    from scipy.sparse import coo_matrix
    A_dir = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A = (A_dir + A_dir.T) / 2.0  # symmetrize with average weight

    # Laplacian L = D - A
    degrees = np.array(A.sum(axis=1)).flatten()
    D_diag = coo_matrix(
        (degrees, (np.arange(N), np.arange(N))), shape=(N, N)
    ).tocsr()
    L = D_diag - A

    # Eigenvalues of Laplacian (need dense for eigh)
    try:
        eigenvals = eigh(L.toarray(), eigvals_only=True)
        # H₂² = (1/2N) * Σ_{i≥2} 1/λ_i  (λ_0 = 0, skip)
        nonzero = eigenvals[1:][eigenvals[1:] > 1e-10]
        if len(nonzero) == 0:
            return 0.0, 0.0
        h2_sq = float(np.sum(1.0 / nonzero) / (2 * N))
        h2 = float(np.sqrt(h2_sq))
        return h2_sq, h2
    except Exception:
        return 0.0, 0.0


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
        j = h2 + 0.06 * m
        if j < best_j:
            best_j = j
            best_m = m
            best_h2 = h2
    return best_m, best_h2


def _compute_expensive_metrics(m: FlockMetrics, positions: np.ndarray, n: int) -> None:
    """Fill in expensive FlockMetrics fields (gated)."""
    if n < 2:
        return

    # Build single cKDTree for all queries
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)

    # H₂ robustness (reuses tree via compute_h2)
    optimal_m, h2 = find_optimal_m(positions, tree)
    m.h2 = h2
    m.optimal_m = optimal_m

    # Local spacing: median 7th-neighbour distance (reuses tree)
    k = min(8, n)
    dists, _ = tree.query(positions, k=k)
    if k > 1:
        m.local_spacing = float(np.median(dists[:, -1]))

    # Flock shape PCA
    aspect, thickness = compute_shape(positions)
    m.aspect_ratio = aspect
    m.thickness_ratio = thickness

    # Gyration radius (trimmed)
    m.gyration_radius = compute_gyration(positions)

    # MSD (from collector's snapshots — computed by the collector)


def compute_shape(positions: np.ndarray) -> tuple[float, float]:
    """PCA flock shape analysis via 3×3 covariance.

    Returns (aspect_ratio, thickness_ratio).
    aspect = sqrt(λ₁/λ₃) — elongation (>1 = elongated).
    thickness = sqrt(λ₂/λ₃) — flatness (pancake vs cigar).

    For degenerate cases (line, plane): if λ₃ ≈ 0 but λ₁ ≫ 0,
    returns (inf, 0) or (large, small) rather than (1, 1).
    """
    N = len(positions)
    if N < 3:
        return 1.0, 1.0

    centered = positions - np.mean(positions, axis=0)
    cov = (centered.T @ centered) / N
    eigenvals = np.linalg.eigvalsh(cov)

    # λ₁ ≥ λ₂ ≥ λ₃ (eigvalsh returns ascending, so reverse)
    lambda1, lambda2, lambda3 = eigenvals[2], eigenvals[1], eigenvals[0]

    if lambda3 < 1e-10:
        if lambda1 > 1e-10:
            # Degenerate: flat/linear shape
            aspect = float(np.sqrt(lambda1 / lambda2)) if lambda2 > 1e-10 else 1e6
            return aspect, 0.0
        return 1.0, 1.0

    aspect = float(np.sqrt(lambda1 / lambda3))
    thickness = float(np.sqrt(lambda2 / lambda3))
    return aspect, thickness


def compute_gyration(positions: np.ndarray) -> float:
    """Trimmed gyration radius — RMS of middle 70% distances from CoM.

    Excludes innermost 15% and outermost 15% to reduce outlier sensitivity.
    """
    N = len(positions)
    if N < 3:
        return 0.0

    com = np.mean(positions, axis=0)
    dists = np.sort(np.linalg.norm(positions - com, axis=1))

    lo = int(N * 0.15)
    hi = int(N * 0.85)
    if hi <= lo:
        return 0.0

    trimmed = dists[lo:hi]
    return float(np.sqrt(np.mean(trimmed ** 2)))


def compute_theta_prime(positions: np.ndarray, grid_res: int = 30) -> float:
    """External opacity Θ' — fraction of bounding volume occupied by birds.

    Rasterizes positions onto a coarse 3D grid and returns the
    fraction of voxels containing at least one bird.

    Args:
        positions: (N, 3) float32 array.
        grid_res: voxels per axis (default 30 → 27,000 voxels).

    Returns:
        θ' ∈ [0, 1] — occupied fraction.
    """
    N = len(positions)
    if N == 0:
        return 0.0

    # Compute bounding box with padding
    mins = positions.min(axis=0) - 1e-6
    maxs = positions.max(axis=0) + 1e-6
    span = maxs - mins
    if np.any(span < 1e-10):
        return 1.0 / (grid_res ** 3)

    # Discretize into voxel indices
    indices = ((positions - mins) / span * grid_res).astype(int)
    indices = np.clip(indices, 0, grid_res - 1)

    # Pack 3D indices into 1D linear index
    linear = np.ravel_multi_index(
        (indices[:, 0], indices[:, 1], indices[:, 2]),
        (grid_res, grid_res, grid_res),
    )
    occupied = len(np.unique(linear))
    return occupied / (grid_res ** 3)


def _density_histogram(
    positions: np.ndarray, bounds: np.ndarray, grid_res: int = 30
) -> np.ndarray:
    """Compute flattened 3D density histogram for autocorrelation."""
    mins, maxs = bounds[0], bounds[1]
    span = maxs - mins
    if np.any(span < 1e-10):
        return np.zeros(grid_res ** 3, dtype=np.float32)
    hist, _ = np.histogramdd(
        positions,
        bins=grid_res,
        range=[(mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])],
    )
    return hist.ravel().astype(np.float32)


def compute_tau_rho(density_history: list[np.ndarray]) -> float:
    """Density autocorrelation time τ_ρ via exponential decay fit.

    Computes Pearson r(τ) between density histograms at lag τ,
    fits r(τ) ≈ exp(−τ / τ_ρ) to extract the characteristic timescale.

    Args:
        density_history: list of flattened density histograms, one per snapshot.

    Returns:
        τ_ρ in frame units (≥ 1). Returns -1 if histograms are unchanging.
    """
    if len(density_history) < 4:
        return 0.0

    n = len(density_history)
    max_lag = min(n - 1, 6)

    lags: list[int] = []
    corrs: list[float] = []

    for lag in range(1, max_lag + 1):
        pairs = [(density_history[t], density_history[t + lag]) for t in range(n - lag)]
        if not pairs:
            continue
        # Compute Pearson r across all pairs for this lag
        r_vals: list[float] = []
        for h0, h1 in pairs:
            h0c = h0 - h0.mean()
            h1c = h1 - h1.mean()
            denom = np.sqrt((h0c ** 2).sum() * (h1c ** 2).sum())
            if denom < 1e-10:
                continue  # zero-variance pair: undefined correlation, skip
            else:
                r_vals.append(float((h0c * h1c).sum() / denom))
        lags.append(lag)
        corrs.append(float(np.mean(r_vals)))

    if not lags or all(c <= 0 for c in corrs):
        return 0.0

    # Fit exponential decay: log(r) = -τ / τ_ρ → τ_ρ = -τ / log(r)
    # Only use positive correlations for the fit
    valid = [(l, c) for l, c in zip(lags, corrs) if c > 0.01]
    if len(valid) < 2:
        return 0.0

    # Weighted average of per-lag estimates
    # τ_ρ = -lag / log(r), with corr clamped to <1 to avoid log(1)=0
    tau_estimates = []
    for lag, corr in valid:
        corr_safe = min(corr, 0.999)  # r=1 → large τ, not inf
        tau_estimates.append(-lag / np.log(corr_safe + 1e-10))

    return float(np.median(tau_estimates))


def compute_msd(snapshots: list[np.ndarray]) -> float:
    """Mean squared displacement from position snapshots.

    MSD = mean displacement² over the longest available lag.
    """
    if len(snapshots) < 2:
        return 0.0

    # Compare first and last snapshot
    pos0 = snapshots[0]
    pos1 = snapshots[-1]
    if len(pos0) != len(pos1):
        return 0.0

    disp = pos1 - pos0
    return float(np.mean(np.sum(disp ** 2, axis=1)))
