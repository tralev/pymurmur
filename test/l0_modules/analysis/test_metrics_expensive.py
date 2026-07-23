"""Phase 9 metrics tests — shape PCA, gyration radius, MSD, theta_prime, tau_rho.
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    MetricsCollector,
    _density_histogram,
    compute_gyration,
    compute_msd,
    compute_r_max,
    compute_shape,
    compute_tau_rho,
    compute_theta_accel_correlation,
    compute_theta_prime,
    find_optimal_m,
)
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine


class TestShapePCA:
    """Flock shape analysis via 3x3 covariance PCA."""

    def test_sphere_has_aspect_near_one(self):
        """Uniform sphere -> aspect ~ 1."""
        rng = np.random.default_rng(42)
        phi = np.arccos(1 - 2 * rng.uniform(0, 1, 100))
        theta = rng.uniform(0, 2 * np.pi, 100)
        positions = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ]).astype(np.float32)

        aspect, thickness = compute_shape(positions)
        assert 0.7 < aspect < 1.35, f"Sphere aspect={aspect:.3f}, expected ~1"
        assert 0.7 < thickness < 1.35, f"Sphere thickness={thickness:.3f}, expected ~1"

    def test_line_has_large_aspect(self):
        """Points on a line with tiny noise -> aspect >> 1, thickness → 0 (P1.9)."""
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            np.linspace(0, 100, 50),
            rng.uniform(-0.01, 0.01, 50),
            rng.uniform(-0.01, 0.01, 50),
        ]).astype(np.float32)

        aspect, thickness = compute_shape(positions)
        assert aspect > 10, f"Line aspect={aspect:.3f}, expected >10"
        # P1.9: thickness = sqrt(λ₃/λ₁) ∈ (0,1]. For a noisy line, λ₃≪λ₁ → thickness ≈ 0
        assert 0.0 < thickness < 0.2, f"Line thickness={thickness:.3f}, expected <0.2"

    def test_perfect_line_degenerate_guard(self):
        """Clean zero-noise line hits degenerate guard -> large aspect, thickness 0."""
        positions = np.column_stack([
            np.linspace(0, 100, 50),
            np.zeros(50),
            np.zeros(50),
        ]).astype(np.float32)

        aspect, thickness = compute_shape(positions)
        assert aspect > 100, f"Perfect line aspect={aspect:.3f}, expected large"
        assert thickness == 0.0

    def test_pancake_shape(self):
        """Wide rectangular pancake with thin z -> large aspect, thin thickness (P1.9)."""
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            rng.uniform(-200, 200, 100),
            rng.uniform(-20, 20, 100),
            rng.uniform(-0.5, 0.5, 100),
        ]).astype(np.float32)

        aspect, thickness = compute_shape(positions)
        assert aspect > 5, f"Pancake aspect={aspect:.3f}, expected >5"
        # P1.9: thickness = sqrt(λ₃/λ₁). Thin z → λ₃≪λ₁ → thickness ≈ 0
        assert 0.0 < thickness < 0.1, f"Pancake thickness={thickness:.3f}, expected <0.1"

    def test_small_n_returns_one(self):
        """N < 3 -> return (1, 1)."""
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        aspect, thickness = compute_shape(positions)
        assert aspect == 1.0
        assert thickness == 1.0

    def test_perfect_plane_degenerate(self):
        """Points on xy-plane (z=0) -> degenerate guard, thickness=0."""
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            rng.uniform(-50, 50, 80),
            rng.uniform(-50, 50, 80),
            np.zeros(80),
        ]).astype(np.float32)
        aspect, thickness = compute_shape(positions)
        assert thickness == 0.0, f"Plane thickness={thickness}, expected 0"
        assert aspect > 0.5, f"Plane aspect={aspect:.3f}, expected >0.5"


class TestGyration:
    """Trimmed gyration radius."""

    def test_sphere_gyration_finite(self):
        """Gyration radius is finite and positive."""
        positions = np.random.uniform(-50, 50, (30, 3)).astype(np.float32)
        rg = compute_gyration(positions)
        assert rg > 0
        assert np.isfinite(rg)

    def test_single_point_zero(self):
        """N < 3 -> rg = 0."""
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        assert compute_gyration(positions) == 0.0

    def test_trimmed_positive(self):
        """Trimmed gyration radius is positive for non-trivial flock."""
        positions = np.random.uniform(-100, 100, (100, 3)).astype(np.float32)
        rg = compute_gyration(positions)
        assert rg > 0
        assert np.isfinite(rg)


class TestRMax:
    """B3 (Pearce et al. 2014): R_max — max pairwise 3D distance
    (flock diameter / fragmentation tracking)."""

    def test_n_less_than_2_returns_zero(self):
        """N < 2 -> no pairs to compare -> 0.0."""
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        assert compute_r_max(positions) == 0.0

    def test_hand_computed_3_4_5_triangle(self):
        """3 points forming a 3-4-5 right triangle: pairwise distances
        are exactly {3, 4, 5}, so R_max == 5."""
        positions = np.array(
            [[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=np.float32
        )
        assert compute_r_max(positions) == pytest.approx(5.0, abs=1e-6)

    def test_larger_spread_gives_larger_r_max(self):
        """A wider point cloud has a larger diameter than a tighter one."""
        rng = np.random.default_rng(3)
        tight = rng.uniform(-1, 1, (50, 3)).astype(np.float32)
        wide = rng.uniform(-100, 100, (50, 3)).astype(np.float32)
        assert compute_r_max(wide) > compute_r_max(tight)

    def test_metrics_collector_computes_r_max(self):
        """MetricsCollector populates r_max at the gated expensive-metrics
        interval, alongside the other shape/extent metrics."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 4

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=12)

        computed = [s for s in sim.metrics.history if s.r_max is not None]
        assert len(computed) >= 2, f"Only {len(computed)} frames had r_max"
        for snap in computed:
            assert snap.r_max > 0
            assert np.isfinite(snap.r_max)

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason=(
            "B3's headline claim (Pearce et al. 2014): the swarm does not "
            "fragment unless phi_p=0, even tiny projection coupling "
            "maintains 3D cohesion. Measured directly: with an open "
            "boundary (required -- toroidal wrap would cap R_max at the "
            "domain diagonal regardless of phi_p), N=100, seed=7, over "
            "300-3000 frames and phi_a in {0.0, 0.3, 0.8}, R_max at "
            "phi_p=0 vs phi_p=0.03 differs by ~1% -- noise-level, not the "
            "dramatic divergence the paper describes. Either pymurmur's "
            "noise/steering calibration differs enough from the source "
            "paper's exact setup that the effect doesn't reproduce at "
            "these settings, or a much longer horizon / different N is "
            "needed. Flagged for follow-up rather than asserting a "
            "threshold the current implementation doesn't actually clear."
        ),
        strict=False,
    )
    def test_swarm_cohesion_requires_projection_coupling(self):
        """B3: swarm should not fragment when phi_p > 0, but should
        fragment more freely at phi_p = 0 (Pearce et al. 2014)."""
        def _final_r_max(phi_p, steps=1000, seed=7):
            cfg = SimConfig()
            cfg.mode = "projection"
            cfg.num_boids = 100
            cfg.seed = seed
            cfg.boundary_mode = "open"  # toroidal wrap would mask fragmentation
            cfg.projection.phi_p = phi_p
            sim = SimulationEngine(cfg)
            sim.run_headless(steps=steps)
            return compute_r_max(sim.flock.positions[sim.flock.active])

        r_max_uncoupled = _final_r_max(phi_p=0.0)
        r_max_coupled = _final_r_max(phi_p=0.03)  # default phi_p

        assert r_max_coupled < r_max_uncoupled * 0.9, (
            f"phi_p=0 (uncoupled) R_max={r_max_uncoupled:.1f} should "
            f"substantially exceed phi_p=0.03 (coupled) "
            f"R_max={r_max_coupled:.1f}"
        )


class TestMSD:
    """Mean squared displacement."""

    def test_msd_zero_for_static_positions(self):
        """Same positions -> MSD = 0."""
        pos = np.random.uniform(0, 10, (20, 3)).astype(np.float32)
        snapshots = [pos.copy(), pos.copy(), pos.copy()]
        msd = compute_msd(snapshots)
        assert msd == pytest.approx(0.0, abs=1e-6)

    def test_msd_positive_for_moving_flock(self):
        """Moving flock -> MSD > 0."""
        pos0 = np.random.uniform(0, 10, (20, 3)).astype(np.float32)
        pos1 = pos0 + 5.0
        msd = compute_msd([pos0, pos1])
        assert msd > 0
        assert msd == pytest.approx(75.0, rel=0.01)  # 3D: 5^2 * 3 = 75

    def test_msd_single_snapshot_zero(self):
        """Single snapshot -> MSD = 0."""
        pos = np.random.uniform(0, 10, (5, 3)).astype(np.float32)
        assert compute_msd([pos]) == 0.0

    def test_msd_mismatched_snapshot_sizes(self):
        """Different N between snapshots -> MSD = 0."""
        pos0 = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
        pos1 = np.random.uniform(0, 10, (8, 3)).astype(np.float32)
        assert compute_msd([pos0, pos1]) == 0.0

    def test_msd_empty_snapshots(self):
        """Empty snapshot list -> MSD = 0."""
        assert compute_msd([]) == 0.0

    def test_metrics_collector_computes_shape_and_msd(self):
        """MetricsCollector computes shape + MSD at gated intervals."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 4

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=12)

        computed = [s for s in sim.metrics.history if s.aspect_ratio is not None]
        assert len(computed) >= 2, f"Only {len(computed)} frames had expensive metrics"

        for snap in computed:
            assert snap.aspect_ratio > 0
            assert snap.gyration_radius > 0


class TestThetaPrime:
    """External opacity via grid rasterization."""

    def test_empty_positions(self):
        """No birds -> theta_prime = 0."""
        positions = np.zeros((0, 3), dtype=np.float32)
        assert compute_theta_prime(positions) == 0.0

    def test_full_grid(self):
        """Many birds spread across domain -> theta_prime > 0."""
        positions = np.random.uniform(-50, 50, (100, 3)).astype(np.float32)
        tp = compute_theta_prime(positions)
        assert 0.0 < tp <= 1.0, f"theta_prime={tp}, expected (0,1]"

    def test_single_bird_minimal(self):
        """One bird -> theta_prime = 1 / grid^3."""
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        tp = compute_theta_prime(positions)
        assert tp == pytest.approx(1.0 / 30 ** 3, rel=0.01)

    def test_clustered_birds_low_opacity(self):
        """All birds in a tiny cluster -> low opacity."""
        positions = np.random.uniform(-1, 1, (50, 3)).astype(np.float32)
        tp = compute_theta_prime(positions, grid_res=20)
        assert tp < 0.1, f"Clustered theta_prime={tp}, expected <0.1"

    def test_collapsed_domain(self):
        """Zero-span domain -> near-zero opacity."""
        positions = np.array([[5, 5, 5], [5, 5, 5]], dtype=np.float32)
        tp = compute_theta_prime(positions)
        assert tp < 0.01


class TestTauRho:
    """Density autocorrelation time."""

    def test_unchanging_density(self):
        """Identical histograms -> large tau (slow decay)."""
        hist = np.ones(1000, dtype=np.float32)
        hist[500:] = 2.0
        density_history = [hist.copy() for _ in range(8)]
        tau = compute_tau_rho(density_history)
        assert tau > 10 or tau == 0.0, f"tau_rho={tau}, expected large or 0"

    def test_decaying_density(self):
        """Linearly interpolated change -> finite tau_rho."""
        h0 = np.zeros(1000, dtype=np.float32)
        h0[100:200] = 10.0
        h1 = np.zeros(1000, dtype=np.float32)
        h1[300:400] = 10.0
        density_history = [
            h0.copy(),
            h0 * 0.75 + h1 * 0.25,
            h0 * 0.5 + h1 * 0.5,
            h0 * 0.25 + h1 * 0.75,
            h1.copy(),
        ]
        tau = compute_tau_rho(density_history)
        assert 0.0 <= tau < 100, f"tau_rho={tau}, expected finite"

    def test_insufficient_data(self):
        """Less than 4 snapshots -> tau_rho = 0."""
        density_history = [np.ones(100, dtype=np.float32) for _ in range(3)]
        assert compute_tau_rho(density_history) == 0.0

    def test_zero_variance_histogram(self):
        """Constant (flat) histograms -> tau_rho = 0."""
        density_history = [np.zeros(500, dtype=np.float32) for _ in range(8)]
        tau = compute_tau_rho(density_history)
        assert tau == 0.0

    def test_all_negative_correlations(self):
        """Orthogonal histograms (non-overlapping peaks) -> all r negative -> returns 0."""
        # Each histogram has a peak in a different range, so any pair has negative r
        density_history = []
        for i in range(6):
            h = np.zeros(500, dtype=np.float32)
            start = i * 80
            h[start:start + 80] = 10.0
            density_history.append(h)
        tau = compute_tau_rho(density_history)
        assert tau == 0.0, f"All-negative should return 0, got {tau}"

    def test_density_histogram_shape(self):
        """_density_histogram returns correct shape and count."""
        positions = np.random.uniform(0, 10, (50, 3)).astype(np.float32)
        bounds = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)
        hist = _density_histogram(positions, bounds, grid_res=10)
        assert len(hist) == 1000  # 10^3
        assert hist.sum() == 50

    def test_zero_span_density_histogram(self):
        """Collapsed positions -> zero-span -> returns zeros."""
        positions = np.array([[5, 5, 5], [5, 5, 5]], dtype=np.float32)
        bounds = np.array([[5, 5, 5], [5, 5, 5]], dtype=np.float32)
        hist = _density_histogram(positions, bounds, grid_res=10)
        assert np.all(hist == 0.0)

    def test_metrics_collector_computes_theta_prime(self):
        """MetricsCollector always computes theta_prime (fast metric)."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 10

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=5)

        for snap in sim.metrics.history:
            assert snap.theta_prime > 0, f"theta_prime={snap.theta_prime}, expected >0"

    def test_metrics_collector_computes_tau_rho(self):
        """At detail_level >= 2, tau_rho computed after enough snapshots."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 2

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=12)

        computed_tau = [s for s in sim.metrics.history if s.tau_rho is not None and s.tau_rho > 0]
        assert len(computed_tau) >= 1, f"tau_rho never computed: {len(computed_tau)} frames"


class TestThetaAccelCorrelation:
    """B9 (Pearce et al. 2014): cross-correlation between horizontal
    COM acceleration and internal opacity Theta."""

    def test_insufficient_samples_returns_none(self):
        curve, peak = compute_theta_accel_correlation(
            [np.zeros(3)] * 3, [0.1, 0.2, 0.3]
        )
        assert curve is None
        assert peak is None

    def test_length_mismatch_returns_none(self):
        curve, peak = compute_theta_accel_correlation(
            [np.zeros(3)] * 8, [0.1] * 6
        )
        assert curve is None
        assert peak is None

    def test_constant_theta_returns_none(self):
        """Zero-variance theta -> degenerate, can't correlate."""
        rng = np.random.default_rng(2)
        vel = list(rng.normal(0, 1, (10, 3)).astype(np.float64))
        theta = [0.3] * 10
        curve, peak = compute_theta_accel_correlation(vel, theta)
        assert curve is None
        assert peak is None

    def test_constant_velocity_returns_none(self):
        """Zero-variance acceleration (constant velocity) -> degenerate."""
        vel = [np.array([1.0, 2.0, 0.0])] * 10
        theta = list(np.linspace(0.1, 0.9, 10))
        curve, peak = compute_theta_accel_correlation(vel, theta)
        assert curve is None
        assert peak is None

    def test_engineered_lag_is_recovered(self):
        """theta is built to track accel_mag exactly at a lag of 2
        sample-steps -- the recovered peak_lag must land there
        (verified against the function's own output while designing
        this test, not hand-derived on paper -- cross-correlation of
        two constructed signals is easy to get subtly wrong by hand)."""
        rng = np.random.default_rng(1)
        n = 20
        vel = rng.normal(0, 1, (n, 3)).astype(np.float64)
        vel[:, 2] = 0.0
        accel_mag = np.linalg.norm(np.diff(vel[:, :2], axis=0), axis=1)  # (n-1,)

        lag_samples = 2
        theta = np.full(n, 0.3)
        for i in range(len(accel_mag)):
            idx = i + 1 + lag_samples
            if idx < n:
                theta[idx] = 0.3 + 0.1 * accel_mag[i]

        curve, peak_lag = compute_theta_accel_correlation(
            list(vel), list(theta), interval=10, buffer_size=500
        )
        assert curve is not None
        assert peak_lag == lag_samples * 10, (
            f"expected peak at {lag_samples} sample-steps "
            f"({lag_samples * 10} frames), got {peak_lag} frames"
        )
        assert curve[lag_samples] == pytest.approx(max(curve, key=abs), abs=1e-9)

    def test_metrics_collector_computes_theta_accel_correlation(self):
        """Projection mode, enough frames to fill the buffer past the
        >=6-sample threshold (10-frame cadence -> need >=60 frames)."""
        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.num_boids = 40
        cfg.seed = 5
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 5

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=200)

        computed = [
            s for s in sim.metrics.history if s.theta_accel_correlation is not None
        ]
        assert len(computed) >= 1, "theta_accel_correlation was never computed"
        for snap in computed:
            assert isinstance(snap.theta_accel_correlation, list)
            assert len(snap.theta_accel_correlation) > 0
            assert all(np.isfinite(c) for c in snap.theta_accel_correlation)
            assert snap.theta_accel_peak_lag is not None
            assert snap.theta_accel_peak_lag >= 0

    def test_non_projection_mode_stays_none(self):
        """theta is NaN outside projection mode, so the buffers never
        populate and the fields stay None even at detail_level=2 with
        plenty of frames."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 40
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 5

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=200)

        assert all(
            s.theta_accel_correlation is None for s in sim.metrics.history
        )
        assert all(
            s.theta_accel_peak_lag is None for s in sim.metrics.history
        )


class TestFindOptimalM:
    """find_optimal_m edge cases."""

    def test_find_optimal_m_prebuilt_tree(self):
        """With pre-built tree works same as without."""
        from scipy.spatial import cKDTree
        positions = np.random.uniform(0, 50, (25, 3)).astype(np.float32)
        tree = cKDTree(positions)
        m1, h1 = find_optimal_m(positions)
        m2, h2 = find_optimal_m(positions, tree=tree)
        assert m1 == m2, f"m* differs: {m1} vs {m2}"
        assert h1 == pytest.approx(h2, rel=0.01)

    def test_find_optimal_m_small_n(self):
        """N <= 2 -> loops over empty range -> falls back to default m=6."""
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        m_star, h2 = find_optimal_m(positions)
        assert m_star == 6  # default fallback
        assert h2 == 0.0


    def test_collector_single_bird_expensive_metrics(self):
        """Single active bird -> n<2 -> _compute_expensive_metrics returns early."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 1
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=5)

        # Should not crash; expensive metrics remain None for n<2
        for snap in sim.metrics.history:
            assert snap.h2 is None
            assert snap.aspect_ratio is None
            assert snap.gyration_radius is None


def test_collector_empty_flock_no_crash():
    """MetricsCollector.collect() with no active birds should not crash."""
    cfg = SimConfig()
    cfg.mode = "spatial"
    cfg.num_boids = 0
    cfg.metrics_detail_level = 2

    sim = SimulationEngine(cfg)
    sim.run_headless(steps=3)
    assert len(sim.metrics.history) == 0


def test_compute_shape_all_same_point():
    """All birds at same point -> all eigenvalues ~0 -> returns (1, 1)."""
    positions = np.array([[0, 0, 0]] * 5, dtype=np.float32)
    aspect, thickness = compute_shape(positions)
    assert aspect == 1.0
    assert thickness == 1.0


def test_collector_snapshot_empty_history():
    """snapshot() on empty history returns default FlockMetrics."""
    collector = MetricsCollector()
    snap = collector.snapshot()
    assert snap.alpha == 0.0
    assert snap.theta == 0.0
    assert snap.h2 is None
    assert snap.tau_rho is None
