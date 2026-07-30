"""FlockMetrics — the container dataclass for all scientific observables.

Extracted from metrics.py (file-size split).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.types import Vec3


@dataclass
class FlockMetrics:
    """Container for all 15 scientific observables.

    Fast metrics (O(N)) computed every frame at detail_level >= 1.
    Expensive metrics (O(N²)) computed every metrics_interval frames at detail_level >= 2.
    """

    # ── Fast (O(N), every frame) ─────────────────────────────────
    alpha: float = 0.0            # polar order parameter |Σ v̂| / N
    nematic_S: float = 0.0        # P9.1: nematic order S = λ_max(Q) ∈ [0,1]
    theta: float = 0.0            # internal opacity Θ
    theta_prime: float = 0.0      # external opacity Θ' (3D voxel)
    silhouette_2d: float = 0.0    # P9.4: 2D silhouette Θ' (disk rasterization)
    angular_momentum: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    normalized_angular_momentum: float = 0.0  # P9.8: ‖⟨r×v⟩‖ / (v0·R_g)
    dispersion: float = 0.0       # ⟨|r − r_com|⟩
    speed_avg: float = 0.0        # ⟨|v|⟩
    force_avg: float = 0.0        # ⟨|a|⟩
    jamming_index: float = 0.0    # B14: steering-saturation proxy — 0=saturated, 1=locked
    power_avg: float = 0.0        # ⟨|a·v|⟩
    local_spacing: float = 0.0    # median k=7 neighbour distance
    # P9.8: Motion metrics
    velocity_deviation: float = 0.0  # (1/N)Σ‖v̄ − v_i‖
    boundary_overshoot: float = 0.0  # Σ max(0, ‖p−C‖ − R_dom)
    altitude_deviation: float = 0.0  # (1/N)·Σ|z_i − z_target|

    # S6.4: Obstacle collision counter
    collisions_this_step: int = 0    # per-step collision count from ObstacleScene

    # S2.E5: influencer-mode target-distance diagnostics (None outside influencer mode)
    target_dist_min: float | None = None   # min ‖p − T‖ this frame
    target_dist_max: float | None = None   # max ‖p − T‖ this frame

    # ── P4.4: Physical metrics (real-world units) ────────────────
    speed_real_ms: float = 0.0      # mean speed in m/s
    accel_real_ms2: float = 0.0     # mean acceleration in m/s²
    force_real_N: float = 0.0       # mean force in newtons
    power_real_W: float = 0.0       # mean mechanical power in watts (mean |a·v| per bird)
    energy_J: float = 0.0           # S2.B4: work done this frame (power * dt), in joules

    # ── Expensive (O(N²) or O(N log N), gated) ───────────────────
    h2: float | None = None       # H₂ consensus robustness
    r_nodal: float | None = None  # A6: nodal robustness √N/H₂ (larger = more robust)
    r_per_m: float | None = None       # A7: robustness per neighbour, R_nodal/m at m_star_sensing
    m_star_sensing: int | None = None  # A7: argmax_m R_per_m(m) — sensing-cost-optimal m*
    tau_rho: float | None = None  # density autocorrelation time (frames)
    theta_accel_correlation: list[float] | None = None  # B9: C(δt), horizontal accel vs opacity
    theta_accel_peak_lag: int | None = None  # B9: δt (frames) where |C(δt)| peaks
    theta_accel_correlation_3d: list[float] | None = None  # full-3D-accel sibling of theta_accel_correlation
    theta_accel_peak_lag_3d: int | None = None  # δt (frames) where |C_3d(δt)| peaks
    hull_volume: float | None = None  # P9.3: convex hull volume
    density_rho: float | None = None  # P9.3: N / hull_volume
    msd: float | None = None      # P9.2: mean squared displacement (longest lag)
    msd_slope: float | None = None   # P9.2: log-log slope (ballistic≈2, diffusive≈1)
    msd_crossover: int | None = None # P9.2: lag where slope drops below 1.5
    msd_curve: list[float] | None = None  # P9.2: MSD values per log-spaced lag
    gyration_radius: float | None = None   # P9.7: robust gyration (median CoM, top-15% trim)
    r_max: float | None = None             # B3: max pairwise 3D distance — swarm fragmentation
    psky_meanfield: float | None = None    # B5: mean-field P(ray hits sky), homogeneous-sphere approx
    marginal_opacity_density: float | None = None  # B6: critical density ρ* for Psky≈0.5
    aspect_ratio: float | None = None      # flock elongation (PCA)
    thickness_ratio: float | None = None   # flock flatness (PCA)
    optimal_m: float | None = None         # cost-optimal neighbour count m*
    suggested_m: float | None = None       # P9.5: shape→m* from aspect ratio
    eta_m: float | None = None             # P9.6: marginal efficiency η(m)
    convergence_speed: float | None = None  # A10: algebraic connectivity λ₂(L) — contrasts with h2's robustness

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (I6.5).

        ndarray → list, numpy scalar → Python scalar, NaN → null,
        inf → null, None → null.
        """
        import math

        result: dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, np.ndarray):
                result[field_name] = value.tolist()
            elif isinstance(value, np.floating) and np.isnan(value):
                result[field_name] = None
            elif isinstance(value, (np.floating, np.integer)):
                result[field_name] = value.item()
            elif isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    result[field_name] = None
                else:
                    result[field_name] = value
            elif value is None:
                result[field_name] = None
            else:
                result[field_name] = value
        return result

    def summary(self, mode: str = "", N_active: int = 0, fps: float = 0.0,
                phi_p: float = 0.0, phi_a: float = 0.0, sigma: int = 0) -> str:
        """P10.2: One-line formatted readout for window title.

        Format: mode | N=… | φp/φa/σ | α Θ Θ′ | L σr | τρ | FPS
        Includes physical units where available.
        """
        parts: list[str] = []
        if mode:
            parts.append(f"{mode} N={N_active}")
        else:
            parts.append(f"N={N_active}")
        # P10.2: φp/φa/σ readout
        if phi_p > 0 or phi_a > 0 or sigma > 0:
            parts.append(f"φp={phi_p:.2f}/φa={phi_a:.2f}/σ={sigma}")
        parts.append(f"α={self.alpha:.3f}")
        if not np.isnan(self.theta):
            parts.append(f"Θ={self.theta:.3f}")
        if not np.isnan(self.theta_prime):
            parts.append(f"Θ′={self.theta_prime:.3f}")
        if self.normalized_angular_momentum > 0:
            parts.append(f"L={self.normalized_angular_momentum:.2f}")
        if self.local_spacing > 0:
            parts.append(f"σr={self.local_spacing:.1f}")
        # P10.2: Physical units for speed and energy
        if self.speed_real_ms > 0:
            parts.append(f"{self.speed_real_ms:.1f}m/s")
        if self.energy_J > 0:
            parts.append(f"{self.energy_J:.2f}J")
        if self.tau_rho is not None and self.tau_rho > 0:
            parts.append(f"τρ={self.tau_rho:.0f}")
        # S2.E5: influencer-mode target-distance readout
        if self.target_dist_min is not None and self.target_dist_max is not None:
            parts.append(f"dT=[{self.target_dist_min:.0f},{self.target_dist_max:.0f}]")
        if fps > 0:
            parts.append(f"{fps:.0f}fps")
        return " | ".join(parts)

