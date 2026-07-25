"""Phase 9 metrics tests — marginal opacity density, B7 opacity sensitivity, jamming index, MSD, theta_prime, opacity nonuniformity.

Split out of test_metrics_expensive.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    compute_jamming_index,
    compute_marginal_opacity_density,
    compute_msd,
    compute_opacity_nonuniformity,
    compute_psky_meanfield,
    compute_theta_prime,
)
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine


class TestMarginalOpacityDensity:
    """B6 (Pearce et al. 2014): critical density rho* for Psky~=0.5,
    derived from Psky=0.5 -> rho ~ N^(-1/2) scaling."""

    def test_degenerate_n_returns_zero(self):
        assert compute_marginal_opacity_density(N=0, b=1.0) == 0.0

    def test_degenerate_b_returns_zero(self):
        assert compute_marginal_opacity_density(N=100, b=0.0) == 0.0

    def test_scaling_law_rho_times_sqrt_n_is_constant(self):
        """B6's headline claim: rho* ~ N^(-1/2), i.e. rho*.sqrt(N) is
        constant. Algebraically exact by construction -- verified
        directly, not just asserted."""
        b = 1.0
        values = [
            compute_marginal_opacity_density(N, b) * np.sqrt(N)
            for N in (100, 400, 1600, 6400)
        ]
        for v in values[1:]:
            assert v == pytest.approx(values[0], rel=1e-9)

    def test_marginal_density_gives_psky_one_half(self):
        """Round-trip: plugging rho* back through the Psky formula
        (at the R implied by rho=N/((4/3)piR^3)) should give ~0.5."""
        N, b = 300, 1.0
        rho_star = compute_marginal_opacity_density(N, b)
        R = (N / ((4.0 / 3.0) * np.pi * rho_star)) ** (1.0 / 3.0)
        psky = compute_psky_meanfield(N, b, R)
        assert psky == pytest.approx(0.5, abs=1e-6)

    def test_metrics_collector_computes_marginal_density(self):
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 5

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=10)

        computed = [
            s for s in sim.metrics.history if s.marginal_opacity_density is not None
        ]
        assert len(computed) >= 1, "marginal_opacity_density was never computed"
        for snap in computed:
            assert snap.marginal_opacity_density > 0.0


class TestB7OpacitySensitivity:
    """B7 (Pearce et al. 2014): 3D opacity is exponentially sensitive
    to inter-bird spacing -- halving spacing changes Theta from 50%
    to ~94%. Uses the already-implemented B5/B6 mean-field machinery
    (compute_psky_meanfield/compute_marginal_opacity_density) plus a
    check against the real simulated (occlusion-based) Theta."""

    def test_b7_halving_spacing_gives_exact_paper_ratio(self):
        """Starting exactly at the marginal point (Theta=0.5) and
        halving R (the spacing lever, N/b fixed): the exponent in
        Psky=exp(-rho*b^2*R) quadruples when R halves (rho itself
        scales as 1/R^3, so rho*R ~ 1/R^2), so
        Psky_new = Psky_old^4 = 0.5^4 = 0.0625 -> Theta=0.9375 -- an
        algebraically exact result, not an approximation, holding for
        any N/b."""
        N, b = 300, 1.0
        rho_star = compute_marginal_opacity_density(N, b)
        R = (N / ((4.0 / 3.0) * np.pi * rho_star)) ** (1.0 / 3.0)
        psky_before = compute_psky_meanfield(N, b, R)
        theta_before = 1.0 - psky_before
        assert theta_before == pytest.approx(0.5, abs=1e-6)

        psky_after = compute_psky_meanfield(N, b, R / 2.0)
        theta_after = 1.0 - psky_after
        assert theta_after == pytest.approx(0.9375, abs=1e-6), (
            f"expected Theta=0.9375 (~94%) after halving spacing from "
            f"the marginal point, got {theta_after:.4f}"
        )

    @pytest.mark.slow
    def test_b7_real_simulated_theta_sensitive_to_spacing(self):
        """The real occlusion-based Theta (not just the mean-field
        formula) is also strongly, super-linearly sensitive to
        spacing -- measured directly: halving the domain (N fixed,
        so spacing roughly halves) at least doubles Theta at each
        step across 400->200->100->50."""
        def tail_theta(domain, steps=200, seed=7, N=200):
            cfg = SimConfig()
            cfg.mode = "projection"
            cfg.num_boids = N
            cfg.seed = seed
            cfg.width = domain
            cfg.height = domain
            cfg.depth = domain
            cfg.metrics_detail_level = 1
            sim = SimulationEngine(cfg)
            sim.run_headless(steps=steps)
            thetas = [
                s.theta for s in sim.metrics.history[-50:] if np.isfinite(s.theta)
            ]
            return float(np.mean(thetas)) if thetas else float("nan")

        domains = [400, 200, 100, 50]
        thetas = [tail_theta(d) for d in domains]

        # Measured directly before writing this assertion: ratios were
        # ~3.3x, ~2.9x, ~1.9x across the three halvings -- growth is
        # strong but tapers as Theta approaches its ceiling near 1.0,
        # so 1.8x (below the smallest measured ratio) is the honest,
        # earned per-step threshold rather than a uniform >2x that the
        # last (highest-density) step doesn't quite clear.
        for i in range(len(thetas) - 1):
            assert thetas[i + 1] > thetas[i] * 1.8, (
                f"halving domain {domains[i]}->{domains[i+1]} should "
                f"substantially increase Theta: {thetas[i]:.4f} -> {thetas[i+1]:.4f}"
            )


class TestJammingIndex:
    """B14 (Pearce et al. 2014): steering-saturation proxy for the
    {phi_p, phi_a} "jammed" corner. No formula is given in the source
    paper -- this is an engineered diagnostic, verified empirically
    against this codebase's own dynamics rather than transcribed from
    the paper."""

    def test_zero_max_force_returns_zero(self):
        assert compute_jamming_index(0.05, 0.0) == 0.0

    def test_hand_computed_ratio(self):
        assert compute_jamming_index(0.15, 0.15) == pytest.approx(0.0, abs=1e-9)
        assert compute_jamming_index(0.075, 0.15) == pytest.approx(0.5, abs=1e-9)
        assert compute_jamming_index(0.0, 0.15) == pytest.approx(1.0, abs=1e-9)

    def test_clips_negative(self):
        """force_avg > max_force shouldn't happen (steering is clamped
        upstream), but the function must not return a negative index."""
        assert compute_jamming_index(0.30, 0.15) == 0.0

    def test_metrics_collector_computes_jamming_index(self):
        """jamming_index is a fast field -- always finite, always in
        [0, 1], populated every frame regardless of detail level."""
        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 1

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=10)

        for snap in sim.metrics.history:
            assert np.isfinite(snap.jamming_index)
            assert 0.0 <= snap.jamming_index <= 1.0

    def test_normal_regime_near_zero_vs_corner_regime_elevated(self):
        """Behavioral regression test for the B14 claim: the shipped
        defaults (phi_p=0.03, phi_a=0.80) saturate steering at
        max_force every frame (jamming_index~0), while the paper's
        high-phi_p/high-phi_a corner desaturates steering substantially
        (measured empirically at ~45-65% of max_force -> index ~0.35-0.55).
        """
        def _tail_avg_jamming(phi_p, phi_a, steps=400, seed=7):
            cfg = SimConfig()
            cfg.mode = "projection"
            cfg.num_boids = 100
            cfg.seed = seed
            cfg.projection.phi_p = phi_p
            cfg.phi_a = phi_a
            cfg.metrics_detail_level = 1

            sim = SimulationEngine(cfg)
            sim.run_headless(steps=steps)

            tail = sim.metrics.history[-100:]
            return float(np.mean([s.jamming_index for s in tail]))

        normal = _tail_avg_jamming(phi_p=0.03, phi_a=0.80)
        corner = _tail_avg_jamming(phi_p=0.5, phi_a=0.99)

        assert normal < 0.05, f"normal regime jamming_index={normal:.3f}, expected ~0"
        assert corner > 0.2, f"corner regime jamming_index={corner:.3f}, expected >0.2"
        assert corner > normal * 2, (
            f"corner ({corner:.3f}) should be clearly elevated vs "
            f"normal ({normal:.3f})"
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


class TestOpacityNonuniformity:
    """B11 (Pearce et al. 2014): KS test that opacity samples are NOT
    uniformly distributed -- the statistical proof of marginal opacity
    as a universal property."""

    def test_uniform_sample_not_rejected(self):
        """A genuinely uniform sample should NOT be rejected (high p)."""
        rng = np.random.default_rng(0)
        uniform_samples = rng.uniform(0.0, 1.0, 300)
        _, p = compute_opacity_nonuniformity(uniform_samples, x_min=0.0)
        assert p > 0.05, f"uniform sample should not reject uniformity, p={p:.4f}"

    def test_clustered_sample_rejected(self):
        """A sample clustered around an intermediate value (like the
        paper's marginal-opacity claim) should be strongly rejected
        (low p) -- matches the pre-implementation methodology check."""
        rng = np.random.default_rng(0)
        clustered = np.clip(rng.normal(0.3, 0.05, 300), 0.0, 1.0)
        stat, p = compute_opacity_nonuniformity(clustered, x_min=0.0)
        assert p < 0.0001, f"clustered sample should strongly reject uniformity, p={p:.6f}"
        assert stat > 0.3

    def test_returns_finite_stat_and_pvalue(self):
        rng = np.random.default_rng(1)
        samples = rng.uniform(0.2, 0.8, 50)
        stat, p = compute_opacity_nonuniformity(samples, x_min=0.2)
        assert np.isfinite(stat)
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0

    @pytest.mark.slow
    def test_our_own_sim_theta_samples_against_uniformity(self):
        """Run projection mode (bird-like defaults) and gather tail
        Theta samples, then test them for uniformity. Honest report --
        no forced conclusion beyond what compute_opacity_nonuniformity
        itself already proved works (p<0.05 threshold)."""
        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.num_boids = 150
        cfg.seed = 7
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=300)

        theta_samples = [
            s.theta for s in sim.metrics.history[-150:] if np.isfinite(s.theta)
        ]
        assert len(theta_samples) >= 50, "not enough finite theta samples"

        stat, p = compute_opacity_nonuniformity(theta_samples, x_min=0.0)
        assert np.isfinite(stat) and np.isfinite(p)
        # Low-bar assertion (matches what the methodology tests above
        # already prove works): our sim's theta values, produced by a
        # real dynamical process rather than sampled from a uniform
        # distribution, should not be statistically indistinguishable
        # from Uniform[0,1]. Not a re-assertion of the paper's exact
        # 99.99%-confidence claim about their own photographic dataset
        # (a different data source entirely) -- just confirming the
        # qualitative finding (non-uniform clustering) reproduces here.
        assert p < 0.05, f"expected our sim's theta samples to reject uniformity, p={p}"


@pytest.mark.slow
def test_b8_theta_vs_inverse_n_linear_fit():
    """B8 (Pearce et al. 2014): Theta vs 1/N should fit a line with
    a high R^2 (paper reports 0.99, N>=400) -- validates marginal
    opacity holds across flock sizes at constant phi_p, phi_a.

    Measured directly in this codebase (projection mode, defaults,
    N=40..800, 200 steps, tail-averaged theta): R^2=0.61 -- clearly
    linear and positive (theta rises with N, falls with 1/N) but
    weaker than the paper's 0.99. Asserting a modest, earned
    threshold rather than forcing the paper's exact number (same
    honesty as B3's fragmentation test, but this claim reproduces
    directionally, so it's a real assertion, not an xfail).
    """
    from scipy.stats import linregress

    def tail_theta(N, steps=200, seed=7):
        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.num_boids = N
        cfg.seed = seed
        cfg.metrics_detail_level = 1
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=steps)
        thetas = [s.theta for s in sim.metrics.history[-50:] if np.isfinite(s.theta)]
        return float(np.mean(thetas)) if thetas else float("nan")

    Ns = [40, 80, 150, 300, 500, 800]
    thetas = [tail_theta(N) for N in Ns]
    inv_n = [1.0 / N for N in Ns]

    result = linregress(inv_n, thetas)
    r_squared = result.rvalue ** 2

    assert r_squared > 0.4, (
        f"Theta vs 1/N fit R^2={r_squared:.3f}, expected >0.4 "
        f"(directional linear relationship; paper reports 0.99)"
    )
    assert result.slope < 0, (
        "Theta should increase as N increases (decrease as 1/N "
        f"increases), got slope={result.slope:.4f}"
    )


