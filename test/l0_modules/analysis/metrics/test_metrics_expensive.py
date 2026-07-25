"""Phase 9 metrics tests — shape PCA, M* thickness sweep, gyration radius, R_max, P_sky mean-field.

Split out of test_metrics_expensive.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    compute_gyration,
    compute_psky_meanfield,
    compute_r_max,
    compute_shape,
    find_m_star_by_sensing_cost,
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


class TestMStarThickness:
    """A12 (Young et al. 2013): m* (cost-optimal neighbour count)
    decreases as 3D flock thickness increases -- thin flocks need
    more neighbours (thickness ~0.15 -> m*~6-7), thick flocks fewer
    (thickness ~0.4 -> m*~5-6). Verified empirically before writing
    this test (unlike B3's fragmentation claim, this one reproduces
    cleanly): synthetic squashed-Gaussian point clouds at thickness
    0.11-0.95 gave m*=7 (thinnest) dropping to and staying at m*=5
    for thickness>=0.18 -- matching the paper's qualitative trend."""

    @staticmethod
    def _make_flock(N, thickness_scale, rng):
        """3D Gaussian squashed along z relative to x,y -- thickness_scale
        controls how flat (small) vs spherical (~1.0) the cloud is."""
        xy = rng.normal(0, 100, (N, 2))
        z = rng.normal(0, 100 * thickness_scale, (N, 1))
        return np.hstack([xy, z]).astype(np.float32)

    def test_thin_flock_has_higher_m_star_than_thick(self):
        rng = np.random.default_rng(11)
        thin = self._make_flock(300, 0.1, rng)
        thick = self._make_flock(300, 0.6, rng)

        thin_aspect, thin_thickness = compute_shape(thin)
        thick_aspect, thick_thickness = compute_shape(thick)
        assert thin_thickness < 0.2, f"thin fixture thickness={thin_thickness:.3f}, expected <0.2"
        assert thick_thickness > 0.4, f"thick fixture thickness={thick_thickness:.3f}, expected >0.4"

        m_star_thin, _ = find_optimal_m(thin)
        m_star_thick, _ = find_optimal_m(thick)
        assert m_star_thin >= m_star_thick, (
            f"thin flock (thickness={thin_thickness:.3f}) m*={m_star_thin} "
            f"should be >= thick flock (thickness={thick_thickness:.3f}) "
            f"m*={m_star_thick}"
        )

    def test_m_star_settles_in_paper_reported_range_for_thick_flock(self):
        """Thick 3D flocks: paper reports m*~5-6."""
        rng = np.random.default_rng(12)
        thick = self._make_flock(300, 0.6, rng)
        m_star, _ = find_optimal_m(thick)
        assert 5 <= m_star <= 6, f"thick flock m*={m_star}, expected 5-6"


class TestA14ThicknessSweepN1200:
    """A14 (Young et al. 2013): uniform random 3D flocks, N=1200 (the
    paper's literal figure), varying thickness. Paper claims: m*
    decreases then plateaus with thickness; peak robustness-per-
    neighbour increases sigmoidally with thickness. Uses
    find_m_star_by_sensing_cost (A7's paper-correct m* definition)."""

    @staticmethod
    def _make_flock(N, thickness_scale, rng):
        xy = rng.normal(0, 100, (N, 2))
        z = rng.normal(0, 100 * thickness_scale, (N, 1))
        return np.hstack([xy, z]).astype(np.float32)

    @pytest.mark.slow
    def test_a14_peak_robustness_increases_with_thickness_at_n1200(self):
        """Real, passing test: measured directly before writing this
        assertion -- peak R_per_m rose 7.2 -> 9.0 -> 8.6 -> 9.9 ->
        10.0 -> 10.1 across thickness 0.1 -> 0.92. Not perfectly
        monotonic step-to-step, but the thickest setting clearly and
        substantially exceeds the thinnest."""
        rng = np.random.default_rng(7)
        thin = self._make_flock(1200, 0.1, rng)
        thick = self._make_flock(1200, 0.8, rng)

        _, peak_thin = find_m_star_by_sensing_cost(thin)
        _, peak_thick = find_m_star_by_sensing_cost(thick)

        assert peak_thick > peak_thin * 1.2, (
            f"thick flock peak R_per_m={peak_thick:.2f} should "
            f"substantially exceed thin flock peak R_per_m={peak_thin:.2f}"
        )

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason=(
            "A14's other sub-claim (Young et al. 2013): m* decreases "
            "then plateaus with thickness at N=1200. Measured directly: "
            "m* bounces noisily around 3-4 regardless of thickness "
            "(single-seed pairs can go either direction by chance -- "
            "e.g. thin=5/thick=3 for one seed, thin=3/thick=4 for "
            "another), consistent with A7's own finding that this "
            "codebase's R_per_m argmax sits at the connectivity floor "
            "regardless of flock structure, not a real thickness-driven "
            "trend. Averaged over 3 seeds this test happens to show a "
            "small directional gap (thin mean 3.67 vs thick mean 3.0) "
            "but it's noise, not the paper's clean decreasing trend -- "
            "strict=False since the sign of this margin isn't reliable "
            "run to run. The peak-VALUE trend (tested above, passing) "
            "and this m*-trend are different sub-claims with different "
            "outcomes here."
        ),
        strict=False,
    )
    def test_a14_m_star_decreases_with_thickness_at_n1200(self):
        thin_vals, thick_vals = [], []
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            thin = self._make_flock(1200, 0.1, rng)
            thick = self._make_flock(1200, 0.8, rng)
            m_thin, _ = find_m_star_by_sensing_cost(thin)
            m_thick, _ = find_m_star_by_sensing_cost(thick)
            thin_vals.append(m_thin)
            thick_vals.append(m_thick)

        mean_thin = np.mean(thin_vals)
        mean_thick = np.mean(thick_vals)
        assert mean_thin > mean_thick + 1.0, (
            f"thin flock mean m*={mean_thin:.2f} ({thin_vals}) should "
            f"substantially exceed thick flock mean m*={mean_thick:.2f} "
            f"({thick_vals})"
        )


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


class TestPskyMeanfield:
    """B5 (Pearce et al. 2014): mean-field probability a random ray
    through the flock hits sky, Psky = exp(-rho*b^2*R)."""

    def test_degenerate_n_returns_one(self):
        assert compute_psky_meanfield(N=0, b=1.0, R=10.0) == 1.0

    def test_degenerate_r_returns_one(self):
        assert compute_psky_meanfield(N=100, b=1.0, R=0.0) == 1.0

    def test_hand_computed(self):
        # rho = 500 / ((4/3)pi*20^3) = 0.0149207759...
        # Psky = exp(-rho*1^2*20)
        N, b, R = 500, 1.0, 20.0
        rho = N / ((4.0 / 3.0) * np.pi * R ** 3)
        expected = np.exp(-rho * b ** 2 * R)
        assert compute_psky_meanfield(N, b, R) == pytest.approx(expected, rel=1e-9)

    def test_denser_flock_lower_psky(self):
        """More birds in the same radius -> higher density -> more
        occluded -> lower probability of hitting sky."""
        sparse = compute_psky_meanfield(N=50, b=1.0, R=20.0)
        dense = compute_psky_meanfield(N=500, b=1.0, R=20.0)
        assert dense < sparse

    def test_bounded_in_unit_interval(self):
        for N in (10, 1000, 100000):
            p = compute_psky_meanfield(N, b=1.0, R=5.0)
            assert 0.0 <= p <= 1.0

    def test_metrics_collector_computes_psky(self):
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 5

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=10)

        computed = [s for s in sim.metrics.history if s.psky_meanfield is not None]
        assert len(computed) >= 1, "psky_meanfield was never computed"
        for snap in computed:
            assert 0.0 <= snap.psky_meanfield <= 1.0


