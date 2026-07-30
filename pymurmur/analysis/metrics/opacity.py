"""Pearce et al. 2014 opacity pipeline — internal/external opacity,
silhouette, mean-field Psky, marginal-opacity density, and the
opacity-nonuniformity statistical test.

Extracted from metrics.py (file-size split).
"""
from __future__ import annotations

import numpy as np


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


def compute_opacity_nonuniformity(
    theta_samples: list[float] | np.ndarray, x_min: float = 0.0,
) -> tuple[float, float]:
    """B11 (Pearce et al. 2014): null-hypothesis test that opacity
    samples are NOT uniformly distributed on [x_min, 1].

    The paper's statistical proof of marginal opacity as a universal
    property: real starling-flock opacities cluster around
    intermediate values rather than spreading randomly toward fully
    opaque, rejecting uniformity at 99.99% confidence for all tested
    x_min. This is a general-purpose Kolmogorov-Smirnov test against
    Uniform[x_min, 1] — a thin wrapper around scipy, operating on
    *any* collection of opacity samples gathered over time (not a
    per-frame FlockMetrics field, since it needs a distribution of
    samples, not one instant).

    Args:
        theta_samples: opacity values Θ ∈ [x_min, 1], gathered over
            multiple frames/runs.
        x_min: lower bound of the uniform null hypothesis (default 0).

    Returns:
        (ks_statistic, p_value). Small p_value (e.g. < 0.05) rejects
        the uniform-distribution null hypothesis — samples cluster
        rather than spreading uniformly.
    """
    from scipy import stats

    samples = np.asarray(theta_samples, dtype=np.float64)
    result = stats.kstest(samples, "uniform", args=(x_min, 1.0 - x_min))
    return float(result.statistic), float(result.pvalue)



# ── P9.4: 2D silhouette Θ' ────────────────────────────────────

# S3.6a: Expected marginal-opacity band for a settled, seeded
# projection-mode flock (Pearce 2014 occlusion-avoidance dynamics
# self-regulate silhouette coverage toward a "marginal opacity" —
# neither too sparse to look enclosed nor so dense the flock reads as a
# solid disk). Used as a µ±3σ regression band by the S3.6a acceptance
# test — no new physics, this just documents/validates the existing
# S3.6 silhouette (compute_silhouette_2d) behaviour.
MARGINAL_OPACITY_MEAN = 0.30
MARGINAL_OPACITY_STD = 0.243

# B10: Pearce et al. 2014's *second* empirical opacity dataset — public
# domain starling-flock images (n independent of their own UK-flock
# measurements above), µ_Θ₀=0.41, σ²=0.012. Confirms marginal opacity
# across different flock conditions (size, lighting, location), not
# just the paper's own dataset. Both anchors fall inside the S3.6a
# acceptance band [0.05, 0.55] — see test_marginal_opacity_constants_accessible.
PUBLIC_OPACITY_MEAN = 0.41
PUBLIC_OPACITY_STD = 0.1095  # sqrt(0.012)


def compute_silhouette_2d(
    positions: np.ndarray,
    boid_size: float = 5.0,
    grid_res: int = 100,
    observer_axis: int = 2,
) -> float:
    """P9.4: 2D silhouette — disk rasterization ⊥ observer axis.

    Projects positions onto the plane perpendicular to `observer_axis`
    (default Z, i.e. the XY plane), rasterizes disks of radius
    `boid_size` onto a coarse grid, and returns the union fraction
    (overlaps count once).

    This complements the 3D voxel-based theta_prime and measures
    how much of the observer's field of view the flock covers.

    Args:
        positions: (N, 3) float32 array.
        boid_size: disk radius in world units.
        grid_res: pixels per axis (default 100 → 10,000 cells).
        observer_axis: 0/1/2 — axis the observer looks along (projected
            out); default 2 (Z) preserves prior behaviour.

    Returns:
        silhouette ∈ [0, 1] — fraction of bounding rectangle covered.
    """
    N = len(positions)
    if N == 0:
        return 0.0

    # Project onto the plane perpendicular to observer_axis.
    ax = [i for i in range(3) if i != observer_axis]
    pts = positions[:, ax]  # (N, 2)
    mins = pts.min(axis=0) - boid_size
    maxs = pts.max(axis=0) + boid_size
    span = maxs - mins
    if np.any(span < 1e-10):
        return 0.0

    # Pixel grid
    grid = np.zeros((grid_res, grid_res), dtype=bool)
    # Scale: world → pixel indices
    scale = grid_res / span
    pixel_radius = int(np.ceil(boid_size * max(scale[0], scale[1])))
    pixel_radius = max(pixel_radius, 1)

    for i in range(N):
        px = int((pts[i, 0] - mins[0]) * scale[0])
        py = int((pts[i, 1] - mins[1]) * scale[1])
        # Draw filled circle (cheap; grid is small)
        for dx in range(-pixel_radius, pixel_radius + 1):
            for dy in range(-pixel_radius, pixel_radius + 1):
                if dx * dx + dy * dy <= pixel_radius * pixel_radius:
                    gx = px + dx
                    gy = py + dy
                    if 0 <= gx < grid_res and 0 <= gy < grid_res:
                        grid[gy, gx] = True

    occupied = int(grid.sum())
    return occupied / (grid_res * grid_res)



def compute_psky_meanfield(N: int, b: float, R: float) -> float:
    """B5 (Pearce et al. 2014): mean-field probability a random ray
    through a homogeneous isotropic 3D flock hits sky (unobstructed).

    Psky ≈ exp(−ρ·b²·R), where ρ = N / ((4/3)πR³) is number density,
    b is effective body radius, R is characteristic flock radius (d=3,
    so b^(d−1) = b²). This is the idealized sphere/uniform-density
    approximation the paper's marginal-opacity derivation (B6) is
    built on — distinct from the exact simulated internal opacity Θ
    (spherical-cap occlusion geometry, see occlusion.py), which this
    is a cheap analytical estimate of, not a replacement for.

    Args:
        N: number of birds.
        b: effective body radius (config.boid_size).
        R: characteristic flock radius (e.g. compute_robust_density's R_g).

    Returns:
        Psky ∈ [0, 1] (clipped — the exponential form can't exceed 1,
        but degenerate R<=0 or N<=0 inputs are guarded to 1.0, "empty
        flock, all sky").
    """
    if N <= 0 or R <= 0:
        return 1.0
    rho = N / ((4.0 / 3.0) * np.pi * R ** 3)
    return float(np.clip(np.exp(-rho * b ** 2 * R), 0.0, 1.0))


def compute_marginal_opacity_density(N: int, b: float) -> float:
    """B6 (Pearce et al. 2014): critical number density ρ* at which
    Psky ≈ 1/2 (marginal opacity) for a flock of N birds with
    effective body radius b.

    Derivation: set Psky = exp(−ρ·b²·R) = 1/2 → ρ·b²·R = ln2.
    Substituting R from ρ = N/((4/3)πR³) (i.e. R = (3N/(4πρ))^(1/3))
    and solving for ρ gives ρ* = √((4/3)π·(ln2)³ / (N·b⁶)) — the
    N^(−1/2) scaling law B6 states (verified: ρ*·√N is exactly
    constant across N, algebraically, not just empirically).

    Args:
        N: number of birds.
        b: effective body radius (config.boid_size).

    Returns:
        ρ* — the critical density for marginal opacity. 0.0 for
        degenerate N<=0 or b<=0.
    """
    if N <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt((4.0 / 3.0) * np.pi * (np.log(2.0) ** 3) / (N * b ** 6)))


