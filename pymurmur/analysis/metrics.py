"""Scientific observables and metrics collection.

Level 1 — 15 observables, split into fast (O(N)) and expensive (O(N²)).
Gated behind config.metrics_detail_level and config.metrics_interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.types import Vec3

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..physics.flock import PhysicsFlock


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
    tau_rho: float | None = None  # density autocorrelation time (frames)
    hull_volume: float | None = None  # P9.3: convex hull volume
    density_rho: float | None = None  # P9.3: N / hull_volume
    msd: float | None = None      # P9.2: mean squared displacement (longest lag)
    msd_slope: float | None = None   # P9.2: log-log slope (ballistic≈2, diffusive≈1)
    msd_crossover: int | None = None # P9.2: lag where slope drops below 1.5
    msd_curve: list[float] | None = None  # P9.2: MSD values per log-spaced lag
    gyration_radius: float | None = None   # P9.7: robust gyration (median CoM, top-15% trim)
    aspect_ratio: float | None = None      # flock elongation (PCA)
    thickness_ratio: float | None = None   # flock flatness (PCA)
    optimal_m: float | None = None         # cost-optimal neighbour count m*
    suggested_m: float | None = None       # P9.5: shape→m* from aspect ratio
    eta_m: float | None = None             # P9.6: marginal efficiency η(m)

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


class MetricsCollector:
    """Computes and caches flock metrics each frame.

    Expensive metrics (H2, shape, gyration) can optionally be
    computed in a background thread via use_async=True.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self._history: list[FlockMetrics] = []
        self._position_snapshots: list[np.ndarray] = []  # for MSD
        self._density_history: list[np.ndarray] = []     # for tau_rho (histogram)
        self._hull_density_ring: list[float] = []        # P9.3: hull density ring buffer
        self._hull_density_maxlen: int = 500             # P9.3: max ring buffer slots
        self._hull_density_interval: int = 10            # P9.3: sample every N frames
        self._detail_level = config.metrics_detail_level if config else 1
        self._interval = config.metrics_interval if config else 60
        # D19: History cap — ring-buffer truncation prevents unbounded growth
        self._history_cap = config.history_cap if config else 10000
        self._mode = config.mode if config else 'projection'
        # S3.11: EMA readout smoothing — display-only, raw history untouched
        self._readout_smooth = config.readout_smooth if config else 0.04
        self._ema_metrics: FlockMetrics = FlockMetrics()  # EMA-smoothed display snapshot
        self._theta_prime_grid = 30  # voxel resolution for external opacity
        self._async_result: object | None = None  # Future from background thread
        self._async_gen: int = 0  # generation counter to detect stale results
        # P4.4: Physical metrics conversion factors
        self._bird_mass_kg = config.bird_mass_kg if config else 0.075
        self._cruise_speed_ms = config.cruise_speed_ms if config else 8.94
        self._acc_peak_ms2 = config.acc_peak_ms2 if config else 40.0
        self._v0 = config.v0 if config else 4.0
        self._max_force = config.max_force if config else 0.15
        # S2.B4: dt_phys for energy_J = power_real_W * dt (work this frame)
        self._dt_phys = config.dt_phys if config else 1.0 / 60.0
        # P9.2: domain size for MSD unwrapping
        self._domain_w = config.width if config else 1000.0
        self._domain_h = config.height if config else 1000.0
        self._domain_d = config.depth if config else 1000.0
        # P9.8: Roost target altitude
        self._roost_z_target: float | None = None
        if config is not None:
            self._roost_z_target = config.roost.z_target
        # G7: Fastmath × metrics-export warning flag
        self._fastmath: bool = config.perf.fastmath if config else False
        self._warned_fastmath: bool = False
        # S2.E5: kept as a live reference (not a snapshot) — influencer
        # mode writes _target_dist_min/max onto this same config object
        # every frame via InfluencerMode.compute().
        self._config = config

    def collect(self, flock: PhysicsFlock, frame: int,
                collisions_this_step: int = 0) -> None:
        """Compute metrics for the current frame."""
        # G7: Fastmath × metrics-export warning — emit once, on first frame
        if self._fastmath and not self._warned_fastmath:
            import warnings
            warnings.warn(
                "Metrics exported with perf.fastmath=True — "
                "floating-point determinism not guaranteed. "
                "Set perf.fastmath=False for reproducible observables.",
                RuntimeWarning, stacklevel=2,
            )
            self._warned_fastmath = True
        # S2.B3/S2.D3: flock observables (alpha, dispersion, etc.) are
        # computed over prey only wherever a species column is populated —
        # a predator's presence shouldn't count toward the prey's own
        # order/cohesion signal. is_predator is always a real (N,) bool
        # array (all-False when n_predators=0), so this is a no-op unless
        # predators are actually configured.
        active = flock.active & ~flock.is_predator
        n = active.sum()
        if n == 0:
            return

        m = FlockMetrics()

        # ── Fast metrics ──────────────────────────────────────────
        positions = flock.positions[active]
        velocities = flock.velocities[active]

        # Order parameter α (polar)
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dirs = velocities / norms
        m.alpha = float(np.linalg.norm(dirs.sum(axis=0)) / n)

        # P9.1: Nematic order parameter S (traceless Q-tensor)
        m.nematic_S = float(compute_nematic_order(dirs))

        # Internal opacity Θ — NaN in non-projection modes (P1.10)
        if self._mode == 'projection':
            m.theta = float(np.mean(flock.last_theta[active]))
        else:
            m.theta = float('nan')

        # External opacity Θ' (fast: O(N) grid rasterization)
        m.theta_prime = compute_theta_prime(positions, self._theta_prime_grid)

        # P9.4: 2D silhouette Θ' — disk rasterization ⊥ observer axis
        m.silhouette_2d = compute_silhouette_2d(positions)

        # Centre of mass and dispersion
        com = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - com, axis=1)
        m.dispersion = float(np.mean(dists))

        # Speed / force / power
        speeds = np.linalg.norm(velocities, axis=1)
        m.speed_avg = float(np.mean(speeds))

        # D18: Read from last_accelerations stash — accelerations are
        # zeroed by integrate() before metrics.collect() runs.
        accs = flock.last_accelerations[active]
        acc_mags = np.linalg.norm(accs, axis=1)
        m.force_avg = float(np.mean(acc_mags))
        m.power_avg = float(np.mean(np.abs(np.sum(accs * velocities, axis=1))))

        # Angular momentum about the centre of mass: ⟨(r-CoM) × v⟩ = Σ(r-CoM)×v / N.
        # S3.9: CoM-centered (not origin-centered) so its magnitude is
        # exactly the reward module's angular-momentum penalty term
        # ‖Σᵢ(pᵢ−CoM)×vᵢ‖/N, and so it matches
        # compute_normalized_angular_momentum's own CoM-centering below.
        m.angular_momentum = np.mean(np.cross(positions - com, velocities), axis=0)

        # P9.8: Motion metrics
        m.velocity_deviation = float(
            np.mean(np.linalg.norm(np.mean(velocities, axis=0) - velocities, axis=1))
        )
        m.boundary_overshoot = _compute_boundary_overshoot(
            positions, self._domain_w, self._domain_h, self._domain_d
        )
        m.altitude_deviation = _compute_altitude_deviation(
            positions, z_target=self._roost_z_target
        )

        # S6.4: Obstacle collision counter
        m.collisions_this_step = collisions_this_step

        # S2.E5: influencer-mode target-distance diagnostics — read off the
        # config object InfluencerMode.compute() writes onto each frame.
        if self._mode == 'influencer' and self._config is not None:
            m.target_dist_min = getattr(self._config, '_target_dist_min', None)
            m.target_dist_max = getattr(self._config, '_target_dist_max', None)

        # P9.8: Normalized angular momentum (uses R_g when available)
        # Uses a fast R_g estimate for real-time display; expensive R_g
        # from async may overwrite it later.
        _Rg_fast = compute_gyration(positions)
        m.normalized_angular_momentum = compute_normalized_angular_momentum(
            positions, velocities, self._v0, _Rg_fast
        )

        # ── P4.4/S2.B4: Physical metrics — real-world unit conversions ──
        _compute_physical_metrics(m, speeds, acc_mags, velocities, accs,
                                   self._bird_mass_kg,
                                   self._cruise_speed_ms, self._acc_peak_ms2,
                                   self._v0, self._max_force, self._dt_phys)

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
            if len(self._position_snapshots) >= 3:
                m.msd = compute_msd(self._position_snapshots)
                msd_vals, lags, slope, crossover = compute_msd_curve(
                    self._position_snapshots,
                    domain_size=(self._domain_w, self._domain_h, self._domain_d),
                )
                m.msd_slope = slope
                m.msd_crossover = crossover
                m.msd_curve = msd_vals

        # MSD: snapshot positions every interval frame
        # tau_rho: snapshot density histogram every interval frame
        # P9.3: hull density sampled every 10 frames into ring buffer
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

        # P9.3: Hull density ring buffer — sample every 10 frames
        if self._detail_level >= 2 and frame % self._hull_density_interval == 0:
            rho = compute_convex_hull_density(positions)
            if rho > 0:
                self._hull_density_ring.append(rho)
                if len(self._hull_density_ring) > self._hull_density_maxlen:
                    self._hull_density_ring.pop(0)

        # S3.5: Hull autocorrelation time from ring buffer
        if self._detail_level >= 2 and len(self._hull_density_ring) >= 4:
            m.tau_rho = compute_tau_rho_hull(
                self._hull_density_ring,
                interval=self._hull_density_interval,
                buffer_size=self._hull_density_maxlen,
            )
            # Also set hull-derived fields from latest sample
            if self._hull_density_ring:
                rho_latest = self._hull_density_ring[-1]
                m.density_rho = rho_latest
                if rho_latest > 0:
                    m.hull_volume = n / rho_latest

        self._history.append(m)
        # S3.11: EMA readout smoothing (display-only, raw history untouched).
        # Uses an EMA factor α = readout_smooth; 0 = passthrough.
        # smoothed(t) = (1 − α) · smoothed(t−1) + α · raw(t)
        if self._readout_smooth > 0.0:
            self._apply_ema_readout(m)

        # D19: History cap — ring-buffer truncation prevents unbounded growth.
        # Snapshots are collected every _interval frames, so their cap is
        # proportionally smaller (history_cap // interval).
        if len(self._history) > self._history_cap:
            self._history = self._history[-self._history_cap:]
            snap_cap = max(1, self._history_cap // self._interval)
            if len(self._position_snapshots) > snap_cap:
                self._position_snapshots = self._position_snapshots[-snap_cap:]
            if len(self._density_history) > snap_cap:
                self._density_history = self._density_history[-snap_cap:]

    def _start_async_expensive(self, positions: np.ndarray, n: int) -> None:
        """Fire expensive metric computation in a background thread.

        Uses a generation counter so that if a previous async is still
        running, its stale result won't overwrite the new one.
        """
        import threading
        self._async_gen += 1
        gen = self._async_gen
        self._async_result = {"done": False, "data": None, "gen": -1}

        def _worker() -> None:
            m = FlockMetrics()
            _compute_expensive_metrics(m, positions, n)
            # Only store result if this is still the current generation
            if self._async_gen == gen:
                self._async_result = {"done": True, "data": m, "gen": gen}

        t = threading.Thread(target=_worker, daemon=True)
        self._async_thread = t  # stored for testability (join in tests)
        t.start()

    def _collect_async_result(self, m: FlockMetrics) -> None:
        """Pick up completed async result if ready, skip if still running."""
        result = self._async_result
        if result is None:
            return
        if not result.get("done"):  # type: ignore[attr-defined]
            return  # still computing, skip this frame
        async_m = result.get("data")  # type: ignore[attr-defined]
        if async_m is not None:
            m.h2 = async_m.h2
            m.optimal_m = async_m.optimal_m
            m.local_spacing = async_m.local_spacing
            m.aspect_ratio = async_m.aspect_ratio
            m.thickness_ratio = async_m.thickness_ratio
            m.gyration_radius = async_m.gyration_radius
            m.suggested_m = async_m.suggested_m
            m.eta_m = async_m.eta_m
        self._async_result = None

    def snapshot(self) -> FlockMetrics:
        """Return the most recent metrics snapshot."""
        return self._history[-1] if self._history else FlockMetrics()

    def smoothed(self) -> FlockMetrics:
        """S3.11: Return EMA-smoothed display snapshot.

        When readout_smooth > 0, returns the EMA-blended FlockMetrics;
        when readout_smooth = 0, falls back to the raw snapshot.
        Always returns a FlockMetrics (never None).
        """
        if self._readout_smooth > 0.0:
            return self._ema_metrics
        return self.snapshot()

    def _apply_ema_readout(self, raw: FlockMetrics) -> None:
        """S3.11: Apply EMA (exponential moving average) to display metrics.

        EMA formula:  smoothed(t) = (1 − α)·smoothed(t−1) + α·raw(t)
        where α = self._readout_smooth.

        Only applies to scalar fast-metrics (alpha, nematic_S, theta,
        theta_prime, silhouette_2d, normalized_angular_momentum,
        dispersion, speed_avg, force_avg, power_avg, local_spacing,
        speed_real_ms, accel_real_ms2, force_real_N, power_real_W,
        energy_J, velocity_deviation, boundary_overshoot,
        altitude_deviation).

        Expensive fields (h2, tau_rho, msd, shape, gyration) are
        gated and change infrequently — they are passed through
        raw (snapshot-on-change) to avoid stale display reads.

        _ema_metrics is initialized to FlockMetrics() (all zeros) in __init__;
        EMA blends from zero on frame 1, converging to true values in ~1/α frames.
        No aliasing with history — _ema_metrics is a distinct object.
        """
        alpha = self._readout_smooth
        if alpha <= 0.0:
            return  # passthrough — smoothed() falls back to raw snapshot

        # Blend scalar fast-metrics fields
        for field_name in (
            "alpha", "nematic_S", "theta", "theta_prime", "silhouette_2d",
            "normalized_angular_momentum", "dispersion", "speed_avg",
            "force_avg", "power_avg", "local_spacing",
            "speed_real_ms", "accel_real_ms2", "force_real_N",
            "power_real_W", "energy_J",
            "velocity_deviation", "boundary_overshoot", "altitude_deviation",
        ):
            raw_val = getattr(raw, field_name)
            if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                continue  # skip NaN / None — keep previous smoothed value
            ema_val = getattr(self._ema_metrics, field_name)
            smoothed_val = (1.0 - alpha) * float(ema_val) + alpha * float(raw_val)
            object.__setattr__(self._ema_metrics, field_name, smoothed_val)

        # Angular momentum (ndarray Vec3) — blend per-component
        raw_L = raw.angular_momentum
        if raw_L is not None and len(raw_L) == 3:
            ema_L = self._ema_metrics.angular_momentum
            smoothed_L = (1.0 - alpha) * ema_L + alpha * raw_L
            object.__setattr__(self._ema_metrics, "angular_momentum", smoothed_L)

        # Expensive fields — pass through raw when they change (snapshot-on-update)
        for field_name in (
            "h2", "tau_rho", "hull_volume", "density_rho",
            "msd", "msd_slope", "msd_crossover", "msd_curve",
            "gyration_radius", "aspect_ratio", "thickness_ratio",
            "optimal_m", "suggested_m", "eta_m",
        ):
            raw_val = getattr(raw, field_name)
            if raw_val is not None:
                object.__setattr__(self._ema_metrics, field_name, raw_val)

    @property
    def history(self) -> list[FlockMetrics]:
        return self._history


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
    from scipy.linalg import eigh
    from scipy.spatial import cKDTree

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
        return float('inf'), float('inf')

    # Build directed adjacency, then symmetrize
    from scipy.sparse import coo_matrix
    A_dir = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    # S1.8: max-form symmetrization — an edge exists at full weight if
    # either endpoint's k-NN includes the other (was averaging, which
    # halved the weight of one-directional k-NN edges).
    A = A_dir.maximum(A_dir.T)

    # Laplacian L = D - A
    degrees = np.array(A.sum(axis=1)).flatten()
    D_diag = coo_matrix(
        (degrees, (np.arange(N), np.arange(N))), shape=(N, N)
    ).tocsr()
    L = D_diag - A

    # Eigenvalues of Laplacian (need dense for eigh)
    try:
        eigenvals = eigh(L.toarray(), eigvals_only=True)
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

    # P9.5: Suggested m* from shape aspect ratio
    m.suggested_m = compute_suggested_m(aspect)

    # P9.6: Marginal efficiency η(m)
    m.eta_m = _compute_eta_m(positions, tree, optimal_m)

    # Gyration radius (P9.7: robust — median CoM, top-15% trim)
    m.gyration_radius = compute_gyration(positions)

    # MSD (from collector's snapshots — computed by the collector)


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


# P4.4: Convert simulation units to real-world physical units


def _compute_physical_metrics(
    m: FlockMetrics,
    speeds: np.ndarray,
    acc_mags: np.ndarray,
    velocities: np.ndarray,
    accs: np.ndarray,
    bird_mass_kg: float,
    cruise_speed_ms: float,
    acc_peak_ms2: float,
    v0: float,
    max_force: float,
    dt: float,
) -> None:
    """P4.4/S2.B4: Convert simulation quantities to real-world physical units.

    speed_real_ms  = mean(|v|) * cruise_speed_ms / v0
    accel_real_ms2 = mean(|a|) * acc_peak_ms2 / max_force
    force_real_N   = accel_real_ms2 * bird_mass_kg

    S2.B4: power and energy corrected —
    power_real_W = m * mean(|k_a*a_i · k_v*v_i|)  (mean of PER-BIRD dot
                   products, not force_real_N * speed_real_ms — a product
                   of means loses the correlation between each bird's own
                   acceleration and velocity direction)
    energy_J     = power_real_W * dt  (work done this frame — one term of
                   the Σ P·Δt integral, not instantaneous kinetic ½mv².
                   Deliberately per-frame rather than a lifetime-running
                   total: energy_J feeds MARL's per-step dense reward
                   (gym_env.py::_compute_reward → rewards.py), which needs
                   a bounded, current-behaviour signal — a monotonically
                   growing total would saturate the reward regardless of
                   policy quality. A caller wanting total accumulated work
                   over an episode can sum energy_J over collector history.
    """
    if v0 <= 0 or max_force <= 0:
        return
    k_v = cruise_speed_ms / v0
    k_a = acc_peak_ms2 / max_force
    # Mean simulated speed → real m/s
    m.speed_real_ms = float(np.mean(speeds)) * k_v
    # Mean simulated acceleration → real m/s^2
    m.accel_real_ms2 = float(np.mean(acc_mags)) * k_a
    # Force: F = m * a
    m.force_real_N = m.accel_real_ms2 * bird_mass_kg
    # S2.B4: mean of per-bird |dot(k_a*a_i, k_v*v_i)|, scaled by mass
    per_bird_power = np.abs(np.sum((accs * k_a) * (velocities * k_v), axis=1))
    m.power_real_W = bird_mass_kg * float(np.mean(per_bird_power))
    # S2.B4: work done this frame (one Σ P·Δt term), not ½mv²
    m.energy_J = m.power_real_W * dt


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
    valid = [(lag_val, _c) for lag_val, _c in zip(lags, corrs) if _c > 0.01]
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


# ── P9.2: MSD(τ) curve ────────────────────────────────────────

def compute_msd_curve(
    snapshots: list[np.ndarray],
    domain_size: tuple[float, float, float] = (1000.0, 1000.0, 1000.0),
    max_lag: int = 64,
) -> tuple[list[float], list[int], float, int | None]:
    """Compute MSD(τ) curve with unwrapped positions and log-spaced lags.

    P9.2: Unwraps positions across toroidal boundaries via min_image,
    computes MSD over log-spaced lags {1, 2, 4, …, max_lag}, fits
    a log-log slope, and detects the ballistic→diffusive crossover.

    Args:
        snapshots: list of (N, 3) position snapshots at evenly-spaced frames.
        domain_size: (W, H, D) for min-image unwrapping.
        max_lag: maximum lag in snapshot units (must be < len(snapshots)).

    Returns:
        (msd_vals, lags, slope, crossover) where:
        - msd_vals: MSD value at each lag.
        - lags: the log-spaced lag list.
        - slope: log-log slope over the first 3 lags (ballistic regime).
        - crossover: first lag where the per-lag exponent drops below 1.5,
          or None if never crosses.
    """
    T = len(snapshots)
    if T < 3:
        return [0.0], [1], 0.0, None

    N = snapshots[0].shape[0]
    if N == 0:
        return [0.0], [1], 0.0, None

    W, H, D = domain_size
    box = np.array([W, H, D], dtype=np.float32)

    # Build unwrapped trajectory: p_unwrap[0] = p[0],
    #   p_unwrap[t] = p_unwrap[t−1] + min_image(p[t] − p[t−1])
    unwrapped = [snapshots[0].copy()]
    for t in range(1, T):
        delta = snapshots[t] - snapshots[t - 1]
        # min_image per-axis: Δx − W·round(Δx/W)
        delta_unwrapped = delta - box * np.round(delta / box)
        unwrapped.append(unwrapped[-1] + delta_unwrapped)
    traj = np.stack(unwrapped, axis=0)  # (T, N, 3)

    # Log-spaced lags: 1, 2, 4, 8, …, max_lag
    lags: list[int] = []
    lag = 1
    while lag <= max_lag and lag < T:
        lags.append(lag)
        lag *= 2
    if not lags:
        lags.append(1)

    msd_vals: list[float] = []
    for lag in lags:
        count = T - lag
        if count < 1:
            msd_vals.append(0.0)
            continue
        # MSD[lag] = (1/(T−lag))·Σ_t ‖p_unwrap(t+lag) − p_unwrap(t)‖²
        diffs = traj[lag:] - traj[:count]  # (count, N, 3)
        sq_disp = np.sum(diffs * diffs, axis=2)  # (count, N)
        msd_vals.append(float(np.mean(sq_disp)))

    # Log-log slope: linear fit to log(MSD) vs log(lag) over first 3 lags
    if len(lags) >= 2:
        n_fit = min(3, len(lags))
        log_lags = np.log(np.array(lags[:n_fit], dtype=np.float64))
        log_msd = np.log(np.maximum(np.array(msd_vals[:n_fit], dtype=np.float64), 1e-12))
        slope, _ = np.polyfit(log_lags, log_msd, 1)
    else:
        slope = 0.0

    # Crossover: first lag where per-lag exponent drops below 1.5
    # Per-lag exponent: d(log MSD) / d(log lag) between consecutive lags
    crossover: int | None = None
    for i in range(1, len(lags)):
        if lags[i - 1] == 0:
            continue
        local_slope = (np.log(max(msd_vals[i], 1e-12))
                       - np.log(max(msd_vals[i - 1], 1e-12))) / np.log(lags[i] / lags[i - 1])
        if local_slope < 1.5 and crossover is None:
            crossover = lags[i]

    return msd_vals, lags, float(slope), crossover


# ── P9.3: Hull-volume density + autocorrelation time ───────────

def compute_convex_hull_density(positions: np.ndarray) -> float:
    """P9.3: Compute flock density via convex hull volume.

    ρ = N / ConvexHull(positions).volume

    Returns 0.0 if the hull is degenerate (coplanar, colinear, or
    fewer than 4 non-coplanar points).

    Args:
        positions: (N, 3) float32 array of bird positions.

    Returns:
        ρ ≥ 0 — density in birds per unit volume.
    """
    N = len(positions)
    if N < 4:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(positions)
        vol = hull.volume
        if vol <= 0:
            return 0.0
        return N / vol
    except Exception:
        return 0.0


def compute_tau_rho_hull(
    density_ring: list[float],
    interval: int = 10,
    buffer_size: int = 500,
) -> float:
    """S3.5: Density autocorrelation time from hull-density ring buffer.

    τ = interval · (0.5 + Σ_{lag≥1} r(lag))
    Stops summation at the first lag where r(lag) ≤ 0, **or** at
    lag = 0.25·buffer_size, whichever comes first — the cap keeps τ
    finite on slowly-varying series that never cross zero (S3.5).

    Uses the ring buffer convention where index 0 is the oldest sample
    and index -1 is the newest (reverse of the original spec, but both
    work identically for autocorrelation).

    Args:
        density_ring: list of hull-density samples ρ(t).
        interval: frames between consecutive samples (default 10).
        buffer_size: capacity of the ring buffer the samples were drawn
            from (default 500, matching MetricsCollector's
            `_hull_density_maxlen`) — the 0.25·buffer_size stop cap is
            relative to this capacity, not the current sample count.

    Returns:
        τ_ρ in frame units. Returns 0 if insufficient data or
        no positive correlations.
    """
    n = len(density_ring)
    if n < 4:
        return 0.0

    series = np.array(density_ring, dtype=np.float64)
    mean = series.mean()
    var = np.var(series)
    if var < 1e-12:
        return 0.0  # constant series → τ = 0

    # Compute autocorrelation r(lag) for lags 1..max_lag
    max_lag = min(n - 1, max(1, int(0.25 * buffer_size)))
    tau_sum = 0.5  # from the formula: 0.5 + Σ r(lag)

    for lag in range(1, max_lag + 1):
        # r(lag) = ⟨(ρ_t - ρ̄)(ρ_{t+lag} - ρ̄)⟩ / var
        # Using the oldest-to-newest convention:
        # series[0] is oldest, series[-1] is newest
        # lag=1: compare adjacent pairs (0,1), (1,2), ..., (n-2,n-1)
        head = series[:n - lag]
        tail = series[lag:]
        r = float(np.mean((head - mean) * (tail - mean)) / var)

        if r <= 0:
            break
        tau_sum += r

    return float(tau_sum * interval)


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


def compute_silhouette_2d(
    positions: np.ndarray,
    boid_size: float = 5.0,
    grid_res: int = 100,
) -> float:
    """P9.4: 2D silhouette — disk rasterization ⊥ observer axis.

    Projects positions onto the XY plane (⊥ Z = observer axis),
    rasterizes disks of radius `boid_size` onto a coarse grid,
    and returns the union fraction (overlaps count once).

    This complements the 3D voxel-based theta_prime and measures
    how much of the observer's field of view the flock covers.

    Args:
        positions: (N, 3) float32 array.
        boid_size: disk radius in world units.
        grid_res: pixels per axis (default 100 → 10,000 cells).

    Returns:
        silhouette ∈ [0, 1] — fraction of bounding rectangle covered.
    """
    N = len(positions)
    if N == 0:
        return 0.0

    # Project to XY (observer axis = Z)
    pts = positions[:, :2]  # (N, 2)
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
