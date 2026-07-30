"""MetricsCollector — orchestrates per-frame metrics collection.

Extracted from metrics.py (file-size split). Owns the ring-buffer
state (position/density snapshots, hull-density and B9 accel/theta
rings) and dispatches to the compute_* functions split across
consensus_robustness.py, opacity.py, shape_motion.py, and
dynamics_curves.py. _compute_expensive_metrics/_compute_physical_metrics/
_density_histogram stay here (not split further) since they're
private orchestration helpers with no callers outside this class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .consensus_robustness import (
    _compute_eta_m,
    compute_convergence_speed,
    compute_r_nodal,
    find_m_star_by_sensing_cost,
    find_optimal_m,
)
from .dynamics_curves import (
    compute_convex_hull_density,
    compute_msd,
    compute_msd_curve,
    compute_tau_rho,
    compute_tau_rho_hull,
    compute_theta_accel_correlation_3d,
    compute_theta_horizontal_accel_correlation,
)
from .flock_metrics import FlockMetrics
from .opacity import (
    compute_marginal_opacity_density,
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

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...physics.flock import PhysicsFlock


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
        # B9: COM velocity + theta ring buffers for the accel/opacity
        # cross-correlation — same cadence/cap as the hull-density ring
        # (projection-mode only, since theta is NaN elsewhere).
        self._accel_com_vel_ring: list[np.ndarray] = []
        self._accel_theta_ring: list[float] = []
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
        # S3.6: silhouette_2d's disk radius — was hardcoded to the
        # function default (5.0) regardless of the actual bird size.
        self._boid_size = config.boid_size if config else 5.0
        # S3.8: altitude_deviation's z_target. RoostConfig.z_target is a
        # static dataclass default (500.0) unaware of domain depth; the
        # spec wants the default to be the domain-centre z when the
        # field hasn't been explicitly overridden away from that shared
        # sentinel default — use domain depth/2 in that case, the user's
        # value otherwise.
        self._roost_z_target: float | None = None
        if config is not None:
            from ...core.config import RoostConfig
            if config.roost.z_target == RoostConfig().z_target:
                self._roost_z_target = config.depth / 2.0
            else:
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
        m.silhouette_2d = compute_silhouette_2d(positions, boid_size=self._boid_size)

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
        m.jamming_index = compute_jamming_index(m.force_avg, self._max_force)
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
                _compute_expensive_metrics(m, positions, n, self._boid_size)

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

        # B9: COM velocity + theta ring buffers, same cadence as the hull
        # density ring — projection-mode only, since theta is NaN elsewhere.
        if (self._detail_level >= 2 and frame % self._hull_density_interval == 0
                and self._mode == 'projection' and np.isfinite(m.theta)):
            self._accel_com_vel_ring.append(velocities.mean(axis=0))
            self._accel_theta_ring.append(m.theta)
            if len(self._accel_com_vel_ring) > self._hull_density_maxlen:
                self._accel_com_vel_ring.pop(0)
                self._accel_theta_ring.pop(0)

        if self._detail_level >= 2 and len(self._accel_theta_ring) >= 6:
            curve, peak_lag = compute_theta_horizontal_accel_correlation(
                self._accel_com_vel_ring,
                self._accel_theta_ring,
                interval=self._hull_density_interval,
                buffer_size=self._hull_density_maxlen,
            )
            m.theta_accel_correlation = curve
            m.theta_accel_peak_lag = peak_lag

            curve_3d, peak_lag_3d = compute_theta_accel_correlation_3d(
                self._accel_com_vel_ring,
                self._accel_theta_ring,
                interval=self._hull_density_interval,
                buffer_size=self._hull_density_maxlen,
            )
            m.theta_accel_correlation_3d = curve_3d
            m.theta_accel_peak_lag_3d = peak_lag_3d

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

        boid_size = self._boid_size

        def _worker() -> None:
            m = FlockMetrics()
            _compute_expensive_metrics(m, positions, n, boid_size)
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
            m.convergence_speed = async_m.convergence_speed
            m.r_max = async_m.r_max
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
        dispersion, speed_avg, force_avg, jamming_index, power_avg,
        local_spacing, speed_real_ms, accel_real_ms2, force_real_N,
        power_real_W, energy_J, velocity_deviation, boundary_overshoot,
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
            "force_avg", "jamming_index", "power_avg", "local_spacing",
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
            "optimal_m", "suggested_m", "eta_m", "convergence_speed", "r_max",
            "theta_accel_correlation", "theta_accel_peak_lag",
            "theta_accel_correlation_3d", "theta_accel_peak_lag_3d",
        ):
            raw_val = getattr(raw, field_name)
            if raw_val is not None:
                object.__setattr__(self._ema_metrics, field_name, raw_val)

    @property
    def history(self) -> list[FlockMetrics]:
        return self._history




def _compute_expensive_metrics(
    m: FlockMetrics, positions: np.ndarray, n: int, boid_size: float = 5.0,
) -> None:
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
    m.r_nodal = compute_r_nodal(h2, n)
    m.m_star_sensing, m.r_per_m = find_m_star_by_sensing_cost(positions, tree)

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

    # A10: convergence speed λ₂(L) at the same m* found for H₂ above —
    # computed at the robustness-optimal neighbour count so the two
    # numbers are directly comparable (robustness vs speed distinction).
    m.convergence_speed = compute_convergence_speed(positions, optimal_m, tree)

    # Gyration radius (P9.7: robust — median CoM, top-15% trim)
    m.gyration_radius = compute_gyration(positions)

    # B3: max pairwise 3D distance — swarm fragmentation tracking
    m.r_max = compute_r_max(positions)

    # B5/B6: mean-field Psky + marginal-opacity critical density,
    # reusing the sphere-density assumption compute_robust_density
    # already implements (R_g, ρ) — distinct from the hull-based
    # density_rho field, which assumes the flock's actual convex-hull
    # shape rather than an idealized sphere.
    R_g_robust, _rho_robust = compute_robust_density(positions)
    if R_g_robust > 0:
        m.psky_meanfield = compute_psky_meanfield(n, boid_size, R_g_robust)
    m.marginal_opacity_density = compute_marginal_opacity_density(n, boid_size)

    # MSD (from collector's snapshots — computed by the collector)



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


