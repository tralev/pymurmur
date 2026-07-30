"""Density scaling analysis (Phase 9.6).

Measures how local flock density scales with population size N.
Sweeps N across a range, measures local_spacing as density proxy,
fits power-law rho(N) ~ N^beta, and compares toroidal vs open boundaries.

Also tracks gyration radius (flock size) and 2D silhouette opacity
per N, mirroring the three complementary geometric views murmuration's
own density_scaling.py reports (spacing, size, theta_ext) rather than
spacing alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DensityScalingResult:
    """Container for density scaling sweep results.

    Attributes:
        n_values: 1D array of population sizes tested.
        spacings_toroidal: 1D array of median local_spacing for toroidal boundary.
        spacings_open: 1D array of median local_spacing for open boundary.
        sizes_toroidal: 1D array of median gyration radius (Rg) for toroidal boundary.
        sizes_open: 1D array of median gyration radius (Rg) for open boundary.
        theta_ext_toroidal: 1D array of median 2D silhouette opacity for toroidal boundary.
        theta_ext_open: 1D array of median 2D silhouette opacity for open boundary.
        beta_toroidal: power-law exponent for toroidal (log spacing vs log N slope).
        beta_open: power-law exponent for open boundary.
        r_sq_toroidal: R-squared of the toroidal spacing fit.
        r_sq_open: R-squared of the open spacing fit.
        size_beta_toroidal: power-law exponent for gyration radius (toroidal).
        size_beta_open: power-law exponent for gyration radius (open).
        size_r_sq_toroidal: R-squared of the toroidal size fit.
        size_r_sq_open: R-squared of the open size fit.
        ideal_density_exponent: P9.7 — theoretical ideal = −0.5 for comparison.
        ideal_size_exponent: theoretical ideal = +0.5 (Rg ~ N^(+1/2), the
            3D size-scaling complement to ideal_density_exponent) for comparison.
    """

    n_values: np.ndarray
    spacings_toroidal: np.ndarray
    spacings_open: np.ndarray
    sizes_toroidal: np.ndarray
    sizes_open: np.ndarray
    theta_ext_toroidal: np.ndarray
    theta_ext_open: np.ndarray
    beta_toroidal: float = np.nan
    beta_open: float = np.nan
    r_sq_toroidal: float = np.nan
    r_sq_open: float = np.nan
    size_beta_toroidal: float = np.nan
    size_beta_open: float = np.nan
    size_r_sq_toroidal: float = np.nan
    size_r_sq_open: float = np.nan
    ideal_density_exponent: float = -0.5
    ideal_size_exponent: float = 0.5


def sweep_density_scaling(
    n_values: list[int] | None = None,
    steps: int = 200,
    settle_frac: float = 0.5,
    seed: int = 42,
) -> DensityScalingResult:
    """Sweep population size N, measure density for toroidal and open boundaries.

    For each N, runs a headless spatial-mode simulation under both
    toroidal and open boundary conditions. Measures the median
    local_spacing (7th-neighbour distance), gyration radius, and 2D
    silhouette opacity as complementary geometric views at steady
    state. Fits power-law relationships (spacing, size) via log-log
    regression.

    Args:
        n_values: list of N to sweep (default: [50, 100, 200, 400, 800]).
        steps: simulation steps per N.
        settle_frac: fraction of final frames to average over.
        seed: base random seed.

    Returns:
        DensityScalingResult with measurements and power-law fits.
    """
    from ...core.config import SimConfig
    from ...simulation.engine import SimulationEngine

    if n_values is None:
        n_values = [50, 100, 200, 400, 800]

    n_arr = np.array(n_values, dtype=np.float64)
    n_spacings_t = np.full(len(n_values), np.nan, dtype=np.float64)
    n_spacings_o = np.full(len(n_values), np.nan, dtype=np.float64)
    n_sizes_t = np.full(len(n_values), np.nan, dtype=np.float64)
    n_sizes_o = np.full(len(n_values), np.nan, dtype=np.float64)
    n_theta_ext_t = np.full(len(n_values), np.nan, dtype=np.float64)
    n_theta_ext_o = np.full(len(n_values), np.nan, dtype=np.float64)

    settle_start = int(steps * (1 - settle_frac))
    if settle_start >= steps:
        settle_start = max(0, steps - 10)

    for idx, n in enumerate(n_values):
        domain = max(80.0, n ** (1 / 3) * 15)

        # ── Toroidal boundary ──────────────────────────────────
        cfg_t = SimConfig()
        cfg_t.mode = "spatial"
        cfg_t.num_boids = n
        cfg_t.width = domain
        cfg_t.height = domain
        cfg_t.depth = domain
        cfg_t.boundary_mode = "toroidal"
        # Compute local_spacing ~10 times in the settled phase
        spacing_interval = max(1, steps // 10)

        cfg_t.metrics_detail_level = 2
        cfg_t.metrics_interval = spacing_interval
        cfg_t.seed = seed + idx

        sim_t = SimulationEngine(cfg_t)
        sim_t.run_headless(steps=steps)

        settled_t = sim_t.metrics.history[settle_start:]
        spacings_t = [snap.local_spacing for snap in settled_t if snap.local_spacing > 0]
        if spacings_t:
            n_spacings_t[idx] = float(np.median(spacings_t))
        sizes_t = [
            snap.gyration_radius for snap in settled_t
            if snap.gyration_radius is not None and snap.gyration_radius > 0
        ]
        if sizes_t:
            n_sizes_t[idx] = float(np.median(sizes_t))
        theta_ext_t = [snap.silhouette_2d for snap in settled_t if not np.isnan(snap.silhouette_2d)]
        if theta_ext_t:
            n_theta_ext_t[idx] = float(np.median(theta_ext_t))

        # ── Open boundary ──────────────────────────────────────
        cfg_o = SimConfig()
        cfg_o.mode = "spatial"
        cfg_o.num_boids = n
        cfg_o.width = domain
        cfg_o.height = domain
        cfg_o.depth = domain
        cfg_o.boundary_mode = "open"
        cfg_o.metrics_detail_level = 2
        cfg_o.metrics_interval = spacing_interval
        cfg_o.seed = seed + idx

        sim_o = SimulationEngine(cfg_o)
        sim_o.run_headless(steps=steps)

        settled_o = sim_o.metrics.history[settle_start:]
        spacings_o = [snap.local_spacing for snap in settled_o if snap.local_spacing > 0]
        if spacings_o:
            n_spacings_o[idx] = float(np.median(spacings_o))
        sizes_o = [
            snap.gyration_radius for snap in settled_o
            if snap.gyration_radius is not None and snap.gyration_radius > 0
        ]
        if sizes_o:
            n_sizes_o[idx] = float(np.median(sizes_o))
        theta_ext_o = [snap.silhouette_2d for snap in settled_o if not np.isnan(snap.silhouette_2d)]
        if theta_ext_o:
            n_theta_ext_o[idx] = float(np.median(theta_ext_o))

    result = DensityScalingResult(
        n_values=n_arr,
        spacings_toroidal=n_spacings_t,
        spacings_open=n_spacings_o,
        sizes_toroidal=n_sizes_t,
        sizes_open=n_sizes_o,
        theta_ext_toroidal=n_theta_ext_t,
        theta_ext_open=n_theta_ext_o,
    )

    # Fit power-laws: spacing(N) = C * N^beta -> log(spacing) = beta*log(N) + log(C)
    _fit_power_laws(result)
    return result


def _fit_power_law_pair(
    n_values: np.ndarray, values_t: np.ndarray, values_o: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit y(N) = C * N^beta via log-log linear regression, both boundaries.

    Returns (beta_toroidal, beta_open, r_sq_toroidal, r_sq_open); any
    fit with fewer than 3 valid (non-NaN, positive) points stays NaN.
    """
    log_n = np.log(n_values)
    beta_t = beta_o = r_sq_t = r_sq_o = np.nan

    valid_t = ~np.isnan(values_t) & (values_t > 0)
    if valid_t.sum() >= 3:
        log_y = np.log(values_t[valid_t])
        beta_t, c_t = np.polyfit(log_n[valid_t], log_y, 1)
        pred_t = beta_t * log_n[valid_t] + c_t
        ss_res_t = np.sum((log_y - pred_t) ** 2)
        ss_tot_t = np.sum((log_y - np.mean(log_y)) ** 2)
        r_sq_t = 1 - ss_res_t / max(ss_tot_t, 1e-10)

    valid_o = ~np.isnan(values_o) & (values_o > 0)
    if valid_o.sum() >= 3:
        log_y = np.log(values_o[valid_o])
        beta_o, c_o = np.polyfit(log_n[valid_o], log_y, 1)
        pred_o = beta_o * log_n[valid_o] + c_o
        ss_res_o = np.sum((log_y - pred_o) ** 2)
        ss_tot_o = np.sum((log_y - np.mean(log_y)) ** 2)
        r_sq_o = 1 - ss_res_o / max(ss_tot_o, 1e-10)

    return float(beta_t), float(beta_o), float(r_sq_t), float(r_sq_o)


def _fit_power_laws(result: DensityScalingResult) -> None:
    """Fit power-law exponents (spacing and size) via log-log linear regression."""
    (result.beta_toroidal, result.beta_open,
     result.r_sq_toroidal, result.r_sq_open) = _fit_power_law_pair(
        result.n_values, result.spacings_toroidal, result.spacings_open,
    )
    (result.size_beta_toroidal, result.size_beta_open,
     result.size_r_sq_toroidal, result.size_r_sq_open) = _fit_power_law_pair(
        result.n_values, result.sizes_toroidal, result.sizes_open,
    )


def save_results(result: DensityScalingResult, path: str) -> None:
    """Save density scaling results to a .npz file."""
    np.savez_compressed(
        path,
        n_values=result.n_values,
        spacings_toroidal=result.spacings_toroidal,
        spacings_open=result.spacings_open,
        sizes_toroidal=result.sizes_toroidal,
        sizes_open=result.sizes_open,
        theta_ext_toroidal=result.theta_ext_toroidal,
        theta_ext_open=result.theta_ext_open,
        beta_toroidal=np.array([result.beta_toroidal]),
        beta_open=np.array([result.beta_open]),
        r_sq_toroidal=np.array([result.r_sq_toroidal]),
        r_sq_open=np.array([result.r_sq_open]),
        size_beta_toroidal=np.array([result.size_beta_toroidal]),
        size_beta_open=np.array([result.size_beta_open]),
        size_r_sq_toroidal=np.array([result.size_r_sq_toroidal]),
        size_r_sq_open=np.array([result.size_r_sq_open]),
    )


def load_results(path: str) -> DensityScalingResult:
    """Load density scaling results from a .npz file."""
    data = np.load(path)
    return DensityScalingResult(
        n_values=data["n_values"],
        spacings_toroidal=data["spacings_toroidal"],
        spacings_open=data["spacings_open"],
        sizes_toroidal=data["sizes_toroidal"],
        sizes_open=data["sizes_open"],
        theta_ext_toroidal=data["theta_ext_toroidal"],
        theta_ext_open=data["theta_ext_open"],
        beta_toroidal=float(data["beta_toroidal"][0]),
        beta_open=float(data["beta_open"][0]),
        r_sq_toroidal=float(data["r_sq_toroidal"][0]),
        r_sq_open=float(data["r_sq_open"][0]),
        size_beta_toroidal=float(data["size_beta_toroidal"][0]),
        size_beta_open=float(data["size_beta_open"][0]),
        size_r_sq_toroidal=float(data["size_r_sq_toroidal"][0]),
        size_r_sq_open=float(data["size_r_sq_open"][0]),
    )
