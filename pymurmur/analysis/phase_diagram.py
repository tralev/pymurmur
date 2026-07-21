"""Phase diagram sweep for Vicsek mode (Phase 9.4).

Grid search over (eta, D) space to map the noise → order transition.
Measures steady-state order parameter (polar α or nematic S) for each
parameter pair, identifies the phase boundary where order crosses 0.5.

P9.1: ``order_type`` option — ``"polar"`` (default) reads ``alpha``,
``"nematic"`` reads ``nematic_S``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhaseDiagramResult:
    """Container for phase diagram sweep results.

    Attributes:
        eta_grid: 1D array of vicsek_couplage values.
        d_grid: 1D array of vicsek_diffusion values.
        alpha_grid: 2D array [len(d_grid), len(eta_grid)] of steady-state alpha.
        boundary_eta: 1D array [len(d_grid)] of eta at the alpha=0.5 transition,
                       or NaN if no crossing found.
    """

    eta_grid: np.ndarray
    d_grid: np.ndarray
    alpha_grid: np.ndarray  # order values (polar α or nematic S, per order_type)
    order_type: str = "polar"  # P9.1: which order parameter was measured
    boundary_eta: np.ndarray = field(default_factory=lambda: np.array([]))


def sweep_vicsek_phase(
    eta_range: tuple[float, float] = (0.0, 1.0),
    d_range: tuple[float, float] = (0.0, 4.0),
    n_eta: int = 10,
    n_d: int = 8,
    n_boids: int = 100,
    steps: int = 200,
    settle_frac: float = 0.5,
    seed: int = 42,
    order_type: str = "polar",   # P9.1: "polar" → alpha, "nematic" → nematic_S
) -> PhaseDiagramResult:
    """Sweep (eta, D) parameter space for the Vicsek phase transition.

    For each (eta_i, d_j) pair, runs a headless Vicsek simulation and
    measures the steady-state order parameter (polar α or nematic S)
    averaged over the final settle_frac fraction of frames.

    Args:
        eta_range: (min, max) for vicsek_couplage η ∈ [0, 1].
        d_range: (min, max) for vicsek_diffusion D.
        n_eta: number of eta grid points.
        n_d: number of D grid points.
        n_boids: flock size.
        steps: total simulation steps per (eta, D) point.
        settle_frac: fraction of final frames to average order over.
        seed: base random seed (incremented per sweep point).
        order_type: "polar" (default) uses polar α; "nematic" uses nematic S.

    Returns:
        PhaseDiagramResult with grids and measured order values.
    """
    if order_type not in ("polar", "nematic"):
        raise ValueError(
            f"order_type must be 'polar' or 'nematic', got '{order_type}'"
        )
    from ..core.config import SimConfig
    from ..simulation.engine import SimulationEngine

    eta_grid = np.linspace(eta_range[0], eta_range[1], n_eta, dtype=np.float64)
    d_grid = np.linspace(d_range[0], d_range[1], n_d, dtype=np.float64)
    alpha_grid = np.full((n_d, n_eta), np.nan, dtype=np.float64)

    settle_start = int(steps * (1 - settle_frac))
    if settle_start >= steps:
        settle_start = max(0, steps - 10)  # ensure at least some frames for averaging

    # Small domain so birds stay close even at low density
    domain = max(50.0, n_boids ** (1 / 3) * 10)

    for j, d_val in enumerate(d_grid):
        for i, eta_val in enumerate(eta_grid):
            cfg = SimConfig()
            cfg.mode = "vicsek"
            cfg.num_boids = n_boids
            cfg.width = domain
            cfg.height = domain
            cfg.depth = domain
            cfg.vicsek_couplage = float(eta_val)
            cfg.vicsek_diffusion = float(d_val)
            cfg.vicsek_radius_influence = domain * 0.3  # large radius for phase transition
            cfg.vicsek_velocity = 1.0
            cfg.metrics_detail_level = 1
            cfg.boundary_mode = "toroidal"
            cfg.seed = seed + j * n_eta + i

            sim = SimulationEngine(cfg)
            sim.run_headless(steps=steps)

            # Average order over the settled portion
            if order_type == "nematic":
                values = [
                    snap.nematic_S
                    for snap in sim.metrics.history[settle_start:]
                    if snap.nematic_S >= 0
                ]
            else:  # polar
                values = [
                    snap.alpha
                    for snap in sim.metrics.history[settle_start:]
                    if snap.alpha >= 0
                ]
            if values:
                alpha_grid[j, i] = float(np.mean(values))

    result = PhaseDiagramResult(
        eta_grid=eta_grid,
        d_grid=d_grid,
        alpha_grid=alpha_grid,
        order_type=order_type,
    )
    result.boundary_eta = _find_phase_boundary(result)
    return result


def _find_phase_boundary(result: PhaseDiagramResult) -> np.ndarray:
    """Extract eta at the alpha=0.5 transition for each D slice.

    Uses linear interpolation between the two grid points that bracket
    alpha=0.5. Returns NaN where no crossing is found or alpha never
    reaches 0.5.
    """
    boundary = np.full(len(result.d_grid), np.nan, dtype=np.float64)

    for j in range(len(result.d_grid)):
        alphas = result.alpha_grid[j]
        if np.all(np.isnan(alphas)):
            continue

        # Find first index where alpha crosses above 0.5
        above = alphas >= 0.5
        if not above.any():
            continue

        idx = int(np.argmax(above))
        if idx == 0:
            # Already ordered at lowest eta
            boundary[j] = result.eta_grid[0]
        else:
            # Linear interpolation between idx-1 and idx
            a0, a1 = alphas[idx - 1], alphas[idx]
            if a1 - a0 > 1e-10:
                frac = (0.5 - a0) / (a1 - a0)
                e0, e1 = result.eta_grid[idx - 1], result.eta_grid[idx]
                boundary[j] = e0 + frac * (e1 - e0)
            else:
                boundary[j] = result.eta_grid[idx]

    return boundary


def save_results(result: PhaseDiagramResult, path: str) -> None:
    """Save phase diagram results to a .npz file."""
    np.savez_compressed(
        path,
        eta_grid=result.eta_grid,
        d_grid=result.d_grid,
        alpha_grid=result.alpha_grid,
        boundary_eta=result.boundary_eta,
        order_type=np.array([result.order_type], dtype='U10'),
    )


def load_results(path: str) -> PhaseDiagramResult:
    """Load phase diagram results from a .npz file."""
    data = np.load(path)
    order_type = str(data.get("order_type", np.array(["polar"]))[0])
    return PhaseDiagramResult(
        eta_grid=data["eta_grid"],
        d_grid=data["d_grid"],
        alpha_grid=data["alpha_grid"],
        order_type=order_type,
        boundary_eta=data.get("boundary_eta", np.array([])),
    )
