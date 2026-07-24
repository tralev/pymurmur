"""Scientific observables and metrics collection.

Level 1 — 15+ observables, split into fast (O(N)) and expensive (O(N^2)).
Gated behind config.metrics_detail_level and config.metrics_interval.

This module is now a thin re-export shim (file-size split) — the
actual implementation lives across flock_metrics.py, collector.py,
consensus_robustness.py, opacity.py, shape_motion.py, and
dynamics_curves.py. Every previously-public name from this module
is re-exported here unchanged, so existing
`from pymurmur.analysis.metrics import X` call sites keep working.
"""
from __future__ import annotations

from .collector import (
    MetricsCollector,
    _compute_expensive_metrics,
    _compute_physical_metrics,
    _density_histogram,
)
from .consensus_robustness import (
    _compute_eta_m,
    _knn_laplacian_eigenvalues,
    _knn_laplacian_matrix,
    compute_convergence_speed,
    compute_h2,
    compute_h2_lyapunov,
    compute_r_nodal,
    compute_r_per_m,
    find_m_star_by_sensing_cost,
    find_optimal_m,
)
from .dynamics_curves import (
    compute_convex_hull_density,
    compute_msd,
    compute_msd_curve,
    compute_tau_rho,
    compute_tau_rho_hull,
    compute_theta_accel_correlation,
)
from .flock_metrics import FlockMetrics
from .opacity import (
    MARGINAL_OPACITY_MEAN,
    MARGINAL_OPACITY_STD,
    PUBLIC_OPACITY_MEAN,
    PUBLIC_OPACITY_STD,
    compute_marginal_opacity_density,
    compute_opacity_nonuniformity,
    compute_psky_meanfield,
    compute_silhouette_2d,
    compute_theta_prime,
)
from .shape_motion import (
    _compute_altitude_deviation,
    _compute_boundary_overshoot,
    compute_gyration,
    compute_jamming_index,
    compute_nematic_order,
    compute_normalized_angular_momentum,
    compute_r_max,
    compute_robust_density,
    compute_shape,
    compute_suggested_m,
)

__all__ = [
    "FlockMetrics",
    "MetricsCollector",
    "MARGINAL_OPACITY_MEAN",
    "MARGINAL_OPACITY_STD",
    "PUBLIC_OPACITY_MEAN",
    "PUBLIC_OPACITY_STD",
    "_compute_altitude_deviation",
    "_compute_boundary_overshoot",
    "_compute_eta_m",
    "_compute_expensive_metrics",
    "_compute_physical_metrics",
    "_density_histogram",
    "_knn_laplacian_eigenvalues",
    "_knn_laplacian_matrix",
    "compute_convergence_speed",
    "compute_convex_hull_density",
    "compute_gyration",
    "compute_h2",
    "compute_h2_lyapunov",
    "compute_jamming_index",
    "compute_marginal_opacity_density",
    "compute_msd",
    "compute_msd_curve",
    "compute_nematic_order",
    "compute_normalized_angular_momentum",
    "compute_opacity_nonuniformity",
    "compute_psky_meanfield",
    "compute_r_max",
    "compute_r_nodal",
    "compute_r_per_m",
    "compute_robust_density",
    "compute_shape",
    "compute_silhouette_2d",
    "compute_suggested_m",
    "compute_tau_rho",
    "compute_tau_rho_hull",
    "compute_theta_accel_correlation",
    "compute_theta_prime",
    "find_m_star_by_sensing_cost",
    "find_optimal_m",
]
