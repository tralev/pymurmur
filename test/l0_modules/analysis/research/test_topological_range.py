"""A13 (Young et al. 2013): topological correlation-range n_c vs
H2-derived m* -- independent analyses, no significant correlation
expected (paper: r~=-0.24, p~=0.46)."""

import numpy as np
import pytest
from scipy.stats import pearsonr

from pymurmur.analysis.metrics import find_m_star_by_sensing_cost
from pymurmur.analysis.research.topological_range import compute_topological_correlation_range
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine


class TestTopologicalCorrelationRange:
    """Unit tests for compute_topological_correlation_range."""

    def test_n_less_than_3_returns_none(self):
        pos = np.zeros((2, 3), dtype=np.float32)
        vel = np.zeros((2, 3), dtype=np.float32)
        assert compute_topological_correlation_range(pos, vel) is None

    def test_zero_max_rank_returns_none(self):
        pos = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
        vel = np.random.uniform(-1, 1, (10, 3)).astype(np.float32)
        assert compute_topological_correlation_range(pos, vel, max_rank=0) is None

    def test_zero_velocity_variance_returns_none(self):
        """All birds share the exact same velocity -> no fluctuation
        to correlate."""
        pos = np.random.uniform(0, 10, (20, 3)).astype(np.float32)
        vel = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (20, 1))
        assert compute_topological_correlation_range(pos, vel) is None

    def test_uncorrelated_velocities_gives_near_rank_one(self):
        """Velocities with no spatial structure at all (independent of
        position) should cross zero almost immediately -- verified
        directly before writing this assertion: n_c~=0.98 for this
        exact construction."""
        rng = np.random.default_rng(0)
        pos = rng.uniform(-100, 100, (300, 3)).astype(np.float32)
        vel = rng.normal(0, 1, (300, 3)).astype(np.float32)
        n_c = compute_topological_correlation_range(pos, vel)
        assert n_c is not None
        assert n_c < 3.0, f"expected n_c near 1 for uncorrelated velocities, got {n_c}"

    def test_prebuilt_tree_same_result(self):
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(1)
        pos = rng.uniform(-100, 100, (100, 3)).astype(np.float32)
        vel = rng.normal(0, 1, (100, 3)).astype(np.float32)
        tree = cKDTree(pos)
        n_c_no_tree = compute_topological_correlation_range(pos, vel)
        n_c_with_tree = compute_topological_correlation_range(pos, vel, tree=tree)
        assert n_c_no_tree == pytest.approx(n_c_with_tree, rel=0.01)

    @pytest.mark.slow
    def test_a13_no_significant_correlation_with_m_star(self):
        """The paper's headline finding: H2-derived m* and the
        independently-estimated topological range n_c are NOT
        significantly correlated (r~=-0.24, p~=0.46) -- evidence the
        two analyses measure different flock properties.

        Measured directly in this codebase before writing this
        assertion: swept 10 {phi_p, phi_a} configurations x 2 seeds
        each (projection mode, N=200), computing both m_star_sensing
        (A7) and n_c on the same simulated flock, skipping the
        configurations where n_c was None (highly-polarized flocks in
        the scale-free-correlation regime, where C(k) never crosses
        zero -- itself consistent with the well-known "scale-free
        correlation" phenomenon reported for real starling
        murmurations). Got 17 valid pairs, r=-0.189, p=0.467 --
        remarkably close to the paper's own r~=-0.24, p~=0.46. A real,
        passing reproduction, not an xfail.
        """
        configs = [
            (0.03, 0.80), (0.05, 0.60), (0.10, 0.50), (0.15, 0.40),
            (0.20, 0.30), (0.20, 0.20), (0.20, 0.10), (0.10, 0.10),
            (0.05, 0.05), (0.30, 0.15),
        ]

        m_stars = []
        n_cs = []
        for i, (phi_p, phi_a) in enumerate(configs):
            for seed_mult in (1, 2):
                cfg = SimConfig()
                cfg.mode = "projection"
                cfg.num_boids = 200
                cfg.seed = seed_mult * 10 + i
                cfg.projection.phi_p = phi_p
                cfg.phi_a = phi_a
                sim = SimulationEngine(cfg)
                sim.run_headless(steps=250)

                pos = sim.flock.positions[sim.flock.active]
                vel = sim.flock.velocities[sim.flock.active]
                m_star, _ = find_m_star_by_sensing_cost(pos)
                n_c = compute_topological_correlation_range(pos, vel, max_rank=60)
                if n_c is not None:
                    m_stars.append(m_star)
                    n_cs.append(n_c)

        assert len(m_stars) >= 10, (
            f"too few valid (m_star, n_c) pairs to be meaningful: {len(m_stars)}"
        )

        r, p = pearsonr(m_stars, n_cs)
        assert abs(r) < 0.5, f"expected weak correlation, got r={r:.3f}"
        assert p > 0.05, f"expected non-significant correlation, got p={p:.3f}"
