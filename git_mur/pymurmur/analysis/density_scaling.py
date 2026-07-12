"""Density scaling analysis (Phase 9.6).

Measures how local flock density scales with population size N.
Sweeps N across a range, measures local_spacing as density proxy,
fits power-law rho(N) ~ N^beta, and compares toroidal vs open boundaries.
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
        beta_toroidal: power-law exponent for toroidal (log spacing vs log N slope).
        beta_open: power-law exponent for open boundary.
        r_sq_toroidal: R-squared of the toroidal fit.
        r_sq_open: R-squared of the open fit.
    """

    n_values: np.ndarray
    spacings_toroidal: np.ndarray
    spacings_open: np.ndarray
    beta_toroidal: float = np.nan
    beta_open: float = np.nan
    r_sq_toroidal: float = np.nan
    r_sq_open: float = np.nan


def sweep_density_scaling(
    n_values: list[int] | None = None,
    steps: int = 200,
    settle_frac: float = 0.5,
    seed: int = 42,
) -> DensityScalingResult:
    """Sweep population size N, measure density for toroidal and open boundaries.

    For each N, runs a headless spatial-mode simulation under both
    toroidal and open boundary conditions. Measures the median
    local_spacing (7th-neighbour distance) as a density proxy at
    steady state. Fits power-law relationships via log-log regression.

    Args:
        n_values: list of N to sweep (default: [50, 100, 200, 400, 800]).
        steps: simulation steps per N.
        settle_frac: fraction of final frames to average over.
        seed: base random seed.

    Returns:
        DensityScalingResult with measurements and power-law fits.
    """
    from ..core.config import SimConfig
    from ..simulation.engine import SimulationEngine

    if n_values is None:
        n_values = [50, 100, 200, 400, 800]

    n_arr = np.array(n_values, dtype=np.float64)
    n_spacings_t = np.full(len(n_values), np.nan, dtype=np.float64)
    n_spacings_o = np.full(len(n_values), np.nan, dtype=np.float64)

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

        spacings_t = [
            snap.local_spacing
            for snap in sim_t.metrics.history[settle_start:]
            if snap.local_spacing > 0
        ]
        if spacings_t:
            n_spacings_t[idx] = float(np.median(spacings_t))

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

        spacings_o = [
            snap.local_spacing
            for snap in sim_o.metrics.history[settle_start:]
            if snap.local_spacing > 0
        ]
        if spacings_o:
            n_spacings_o[idx] = float(np.median(spacings_o))

    result = DensityScalingResult(
        n_values=n_arr,
        spacings_toroidal=n_spacings_t,
        spacings_open=n_spacings_o,
    )

    # Fit power-laws: spacing(N) = C * N^beta → log(spacing) = beta*log(N) + log(C)
    _fit_power_laws(result)
    return result


def _fit_power_laws(result: DensityScalingResult) -> None:
    """Fit power-law exponents via log-log linear regression.

    spacing(N) = C * N^beta  →  log(spacing) = beta * log(N) + C'
    """
    log_n = np.log(result.n_values)

    # Toroidal fit
    valid_t = ~np.isnan(result.spacings_toroidal)
    if valid_t.sum() >= 3:
        beta_t, c_t = np.polyfit(log_n[valid_t], np.log(result.spacings_toroidal[valid_t]), 1)
        result.beta_toroidal = float(beta_t)
        pred_t = beta_t * log_n[valid_t] + c_t
        ss_res_t = np.sum((np.log(result.spacings_toroidal[valid_t]) - pred_t) ** 2)
        ss_tot_t = np.sum((np.log(result.spacings_toroidal[valid_t]) - np.mean(np.log(result.spacings_toroidal[valid_t]))) ** 2)
        result.r_sq_toroidal = float(1 - ss_res_t / max(ss_tot_t, 1e-10))

    # Open fit
    valid_o = ~np.isnan(result.spacings_open)
    if valid_o.sum() >= 3:
        beta_o, c_o = np.polyfit(log_n[valid_o], np.log(result.spacings_open[valid_o]), 1)
        result.beta_open = float(beta_o)
        pred_o = beta_o * log_n[valid_o] + c_o
        ss_res_o = np.sum((np.log(result.spacings_open[valid_o]) - pred_o) ** 2)
        ss_tot_o = np.sum((np.log(result.spacings_open[valid_o]) - np.mean(np.log(result.spacings_open[valid_o]))) ** 2)
        result.r_sq_open = float(1 - ss_res_o / max(ss_tot_o, 1e-10))


def save_results(result: DensityScalingResult, path: str) -> None:
    """Save density scaling results to a .npz file."""
    np.savez_compressed(
        path,
        n_values=result.n_values,
        spacings_toroidal=result.spacings_toroidal,
        spacings_open=result.spacings_open,
        beta_toroidal=np.array([result.beta_toroidal]),
        beta_open=np.array([result.beta_open]),
        r_sq_toroidal=np.array([result.r_sq_toroidal]),
        r_sq_open=np.array([result.r_sq_open]),
    )


def load_results(path: str) -> DensityScalingResult:
    """Load density scaling results from a .npz file."""
    data = np.load(path)
    return DensityScalingResult(
        n_values=data["n_values"],
        spacings_toroidal=data["spacings_toroidal"],
        spacings_open=data["spacings_open"],
        beta_toroidal=float(data["beta_toroidal"][0]),
        beta_open=float(data["beta_open"][0]),
        r_sq_toroidal=float(data["r_sq_toroidal"][0]),
        r_sq_open=float(data["r_sq_open"][0]),
    )
