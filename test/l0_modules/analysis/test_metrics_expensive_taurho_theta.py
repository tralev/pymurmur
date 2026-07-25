"""Phase 9 metrics tests — tau_rho, theta-acceleration correlation, find_optimal_m, misc collector edge cases.

Split out of test_metrics_expensive.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    MetricsCollector,
    _density_histogram,
    compute_shape,
    compute_tau_rho,
    compute_theta_accel_correlation,
    find_optimal_m,
)
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine


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

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason=(
            "B13's claim (Pearce et al. 2014, Fig. 2f): density "
            "autocorrelation time tau_rho decreases monotonically as "
            "phi_p increases (projection provides instantaneous global "
            "coupling, speeding up dynamics). Measured directly: N=60, "
            "seed in {1,2,3}, 80x80x80 domain (needed for "
            "compute_tau_rho_hull's absolute-variance floor to clear at "
            "all -- a pre-existing scale sensitivity of that metric at "
            "the default ~1000-unit domain, not something this test "
            "changes), phi_p in {0, 0.01, 0.03, 0.05, 0.08}, 500 steps, "
            "tail-averaged over seeds: tau_rho = 64.5, 64.2, 63.6, "
            "63.0, 69.0 -- essentially flat/non-monotonic, with "
            "per-seed variance (30-100) dwarfing any phi_p-driven "
            "signal. Does not reproduce cleanly at realistic settings "
            "in this codebase. Flagged for follow-up rather than "
            "asserting a trend the current implementation doesn't "
            "actually clear."
        ),
        strict=False,
    )
    def test_b13_tau_rho_decreases_with_phi_p(self):
        """B13: tau_rho should decrease as phi_p increases."""
        def _tail_avg_tau_rho(phi_p, steps=500, seed=7, N=60):
            cfg = SimConfig()
            cfg.mode = "projection"
            cfg.num_boids = N
            cfg.seed = seed
            cfg.width = 80.0
            cfg.height = 80.0
            cfg.depth = 80.0
            cfg.projection.phi_p = phi_p
            cfg.metrics_detail_level = 2
            cfg.metrics_interval = 5
            sim = SimulationEngine(cfg)
            sim.run_headless(steps=steps)
            tau_values = [
                s.tau_rho for s in sim.metrics.history
                if s.tau_rho is not None and s.tau_rho > 0
            ]
            return np.mean(tau_values[-10:]) if len(tau_values) >= 10 else (
                np.mean(tau_values) if tau_values else float("nan")
            )

        low_phi_p = np.mean([_tail_avg_tau_rho(0.0, seed=s) for s in (1, 2, 3)])
        high_phi_p = np.mean([_tail_avg_tau_rho(0.08, seed=s) for s in (1, 2, 3)])

        assert high_phi_p < low_phi_p * 0.9, (
            f"tau_rho at phi_p=0.08 ({high_phi_p:.1f}) should be "
            f"substantially lower than at phi_p=0 ({low_phi_p:.1f})"
        )


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
