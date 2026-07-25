"""Flock shape, order, and motion metrics — nematic order, PCA shape
(aspect/thickness), gyration radius, R_max (B3), suggested m* from
shape, robust number density, jamming_index (B14), and P9.8 motion
metrics (boundary overshoot, altitude deviation, normalized angular
momentum).

Extracted from metrics.py (file-size split).
"""
from __future__ import annotations

import numpy as np

# ── P9.1: Nematic order parameter ──────────────────────────────

def compute_nematic_order(dirs: np.ndarray) -> float:
    """Compute nematic order parameter S from the Q-tensor.

    P9.1: Builds the 3×3 traceless Q-tensor from unit direction
    vectors, then returns its maximum eigenvalue λ_max ∈ [0,1].

    Q_αβ = (1/N) Σ_i ((3/2)·û_i^α·û_i^β − (1/2)·δ_αβ)
    S    = λ_max(Q)

    S ≈ 1  → perfect alignment (or anti-alignment — nematic is
              invariant under û → −û, unlike polar α).
    S ≈ 0  → isotropic (uniform on sphere).

    Args:
        dirs: (N, 3) float32 unit direction vectors.

    Returns:
        S ∈ [0, 1] — scalar nematic order parameter.
    """
    N = dirs.shape[0]
    if N == 0:
        return 0.0

    # Q_αβ = (1/N) Σ_i ( (3/2)·ûα·ûβ − (1/2)·δαβ )
    # Outer products: (N,3,1) × (N,1,3) → (N,3,3), then mean over N
    u = dirs.reshape(N, 3, 1)
    uT = dirs.reshape(N, 1, 3)
    outer = u @ uT  # (N, 3, 3)
    Q = np.mean(1.5 * outer, axis=0)  # (3/2) · (1/N) Σ outer
    Q -= 0.5 * np.eye(3, dtype=dirs.dtype)  # − (1/2)·δ

    # S = λ_max(Q)
    eigenvals = np.linalg.eigvalsh(Q)  # ascending: λ₀ ≤ λ₁ ≤ λ₂
    S = float(eigenvals[2])  # λ_max

    # Clamp to [0, 1] (floating-point may produce small negatives)
    return max(0.0, min(1.0, S))


def compute_shape(positions: np.ndarray) -> tuple[float, float]:
    """PCA flock shape analysis via 3×3 covariance.

    Returns (aspect_ratio, thickness_ratio).
    aspect = sqrt(λ₁/λ₃) — elongation (>1 = elongated).
    thickness = sqrt(λ₃/λ₁) ∈ (0,1] — flatness (P1.9 fix: was λ₂/λ₃).

    λ₁ ≥ λ₂ ≥ λ₃.  thickness → 0 for lines/planes, → 1 for spheres.

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
    # P1.9: thickness = sqrt(λ₃/λ₁) ∈ (0,1]
    #   λ₃/λ₁ → 1 for spheres, → 0 for lines/planes
    thickness = float(np.sqrt(lambda3 / lambda1))
    return aspect, thickness


def compute_gyration(positions: np.ndarray) -> float:
    """P9.7: Robust gyration radius — median centroid, top-15% trim.

    Uses the median position as centroid (not mean) for outlier
    resistance. Retains only the innermost 85% of points (trims
    the most distant 15%). Returns RMS of kept distances.

    One 10K-unit outlier moves R_g < 5%.
    """
    N = len(positions)
    if N < 3:
        return 0.0

    # Median centroid (P9.7: resistant to outliers)
    com = np.median(positions, axis=0)
    dists = np.sort(np.linalg.norm(positions - com, axis=1))

    # Top-15% trim: keep innermost 85%
    keep = int(N * 0.85)
    if keep < 2:
        return 0.0
    kept = dists[:keep]
    return float(np.sqrt(np.mean(kept ** 2)))


def compute_r_max(positions: np.ndarray) -> float:
    """B3 (Pearce et al. 2014): R_max(t) = max_{i,j} |rᵢ(t) − rⱼ(t)|.

    The flock's 3D diameter — the largest pairwise Euclidean distance
    between any two birds. Tracks whether the swarm fragments. Key
    empirical result: the swarm does NOT fragment unless φp = 0 — even
    tiny projection coupling (φp > 0) maintains 3D cohesion, stronger
    than local Reynolds models achieve. This metric exists to make
    that fragmentation (or lack of it) directly observable.

    O(N²) time and memory (scipy's condensed pairwise-distance matrix)
    — the same "expensive metrics" tier as this module's other O(N²)/
    O(N log N) gated observables. Cheaper than h2's O(N³) dense
    eigendecomposition at the same gate, so this introduces no new
    scaling bottleneck at the sizes already accepted there.

    Args:
        positions: (N, 3) float32 array.

    Returns:
        R_max ≥ 0. Returns 0.0 for N < 2 (no pairs to compare).
    """
    N = len(positions)
    if N < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    return float(np.max(pdist(positions)))


# P4.4: Convert simulation units to real-world physical units



def compute_jamming_index(force_avg: float, max_force: float) -> float:
    """B14 (Pearce et al. 2014): steering-saturation proxy for the
    {φp, φa} "jammed" corner. 0 = steering fully saturated at
    max_force (turbulent, unconstrained maneuvering). 1 = steering
    has converged near zero (v_desired ≈ v — locked, rigid
    configuration). No formula is given in the source paper (B14
    describes a phenomenon, not a metric) — this is an engineered
    proxy, empirically verified in this codebase: the shipped defaults
    (φp=0.03, φa=0.80) saturate steering at exactly max_force every
    frame (index≈0), while the paper's high-φp/high-φa corner
    desaturates steering to 45-65% of max_force (index≈0.35-0.55).
    """
    if max_force <= 0.0:
        return 0.0
    return float(np.clip(1.0 - force_avg / max_force, 0.0, 1.0))



# ── P9.5: Shape → m* ──────────────────────────────────────────

def compute_suggested_m(aspect: float) -> float:
    """P9.5: Map flock aspect ratio to suggested neighbour count m*.

    m* = 9.78 + clamp((aspect − 1) / 2, 0, 1) · (6.05 − 9.78)

    aspect = 1 (sphere)   → m* = 9.78 (rounder flocks use more neighbours)
    aspect ≥ 3 (elongated) → m* = 6.05 (elongated flocks use fewer)

    Args:
        aspect: PCA aspect ratio sqrt(λ₁/λ₃) ≥ 1.

    Returns:
        m* ∈ [6.05, 9.78] — suggested optimal neighbour count.
    """
    if aspect < 1:
        aspect = 1.0
    t = min(1.0, (aspect - 1.0) / 2.0)  # clamp to [0, 1]
    return 9.78 + t * (6.05 - 9.78)



# ── P9.7: Robust gyration + ideal exponent ────────────────────

def compute_robust_density(
    positions: np.ndarray,
) -> tuple[float, float]:
    """P9.7: Robust gyration radius and number density.

    Uses median centroid + top-15% trim (same as compute_gyration).
    Returns (R_g, ρ) where ρ = N_kept / ((4/3)·π·R_g³).

    Args:
        positions: (N, 3) float32 array.

    Returns:
        (R_g, ρ) — gyration radius and number density. ρ = 0 for
        degenerate flocks.
    """
    R_g = compute_gyration(positions)
    if R_g <= 0:
        return R_g, 0.0
    N_kept = max(int(len(positions) * 0.85), 2)
    rho = N_kept / ((4.0 / 3.0) * np.pi * R_g ** 3)
    return R_g, rho



# ── P9.8: Motion metrics ──────────────────────────────────────

def _compute_boundary_overshoot(
    positions: np.ndarray,
    domain_w: float,
    domain_h: float,
    domain_d: float,
) -> float:
    """P9.8: Total overshoot distance beyond the domain boundary.

    boundary_overshoot = Σ max(0, ‖p − C‖ − R_dom)

    C = domain centre, R_dom = half the domain width.
    """
    centre = np.array([domain_w / 2, domain_h / 2, domain_d / 2], dtype=np.float64)
    R_dom = min(domain_w, domain_h, domain_d) / 2.0
    dists = np.linalg.norm(positions - centre, axis=1)
    overshoot = np.maximum(0, dists - R_dom)
    return float(np.sum(overshoot))


def _compute_altitude_deviation(
    positions: np.ndarray,
    z_target: float | None = None,
) -> float:
    """P9.8: Mean absolute deviation from target altitude.

    altitude_deviation = (1/N)·Σ|z_i − z_target|

    Args:
        positions: (N, 3) float32 array.
        z_target: target Z altitude. Defaults to 500.0.

    Returns:
        Mean absolute altitude deviation.
    """
    N = len(positions)
    if N == 0:
        return 0.0

    if z_target is None:
        z_target = 500.0

    deviations = np.abs(positions[:, 2] - z_target)
    return float(np.mean(deviations))


def compute_normalized_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    v0: float,
    R_g: float,
) -> float:
    """P9.8: Normalized angular momentum about centre of mass.

    L_norm = ‖⟨r × v⟩‖ / (v0 · R_g)

    O(1) quantity, invariant under domain scaling.

    Args:
        positions: (N, 3) float32.
        velocities: (N, 3) float32.
        v0: characteristic speed.
        R_g: gyration radius.

    Returns:
        L_norm ≥ 0 — 0 for purely radial/linear motion, ~1 for
        coherent rotation.
    """
    N = len(positions)
    if N == 0 or v0 <= 0 or R_g <= 0:
        return 0.0

    com = positions.mean(axis=0)
    r_centered = positions - com
    L = np.mean(np.cross(r_centered, velocities), axis=0)
    return float(np.linalg.norm(L) / (v0 * R_g))
