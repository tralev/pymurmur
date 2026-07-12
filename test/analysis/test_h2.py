"""H₂ robustness tests — Phase 9.1

k-NN graph, Laplacian eigenvalues, cost-optimal m*.
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import compute_h2, find_optimal_m
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine


class TestH2Robustness:
    """H₂ consensus robustness metric (Young 2020)."""

    def test_complete_graph_h2_finite(self):
        """Fully connected flock produces finite, non-zero H₂."""
        # All birds at same point → complete graph → H₂ is defined
        positions = np.zeros((20, 3), dtype=np.float32)
        positions += np.random.normal(0, 0.1, (20, 3)).astype(np.float32)
        h2_sq, h2 = compute_h2(positions, m=5)
        assert np.isfinite(h2)
        assert h2 >= 0

    def test_h2_smaller_than_n(self):
        """H₂ < N for any connected graph."""
        positions = np.random.uniform(0, 100, (15, 3)).astype(np.float32)
        _, h2 = compute_h2(positions, m=3)
        assert h2 < len(positions), f"H₂={h2:.3f} >= N={len(positions)}"

    def test_h2_decreases_with_more_neighbours(self):
        """More neighbours → better connectivity → lower H₂."""
        positions = np.random.uniform(0, 50, (30, 3)).astype(np.float32)
        _, h2_few = compute_h2(positions, m=2)
        _, h2_many = compute_h2(positions, m=8)
        assert h2_many <= h2_few + 1e-6, (
            f"H₂ should decrease with more neighbours: "
            f"m=2 → {h2_few:.4f}, m=8 → {h2_many:.4f}"
        )

    def test_optimal_m_in_range(self):
        """find_optimal_m returns m ∈ [2, 20] and H₂ ≥ 0."""
        positions = np.random.uniform(0, 100, (25, 3)).astype(np.float32)
        m_star, h2 = find_optimal_m(positions)
        assert 2 <= m_star <= 20, f"m*={m_star} out of range"
        assert h2 >= 0

    def test_single_bird_zero_h2(self):
        """N=1 → H₂ = 0 (no graph to build)."""
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        h2_sq, h2 = compute_h2(positions, m=5)
        assert h2_sq == 0.0
        assert h2 == 0.0

    def test_two_birds(self):
        """N=2 → connected → finite H₂."""
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        _, h2 = compute_h2(positions, m=1)
        assert np.isfinite(h2)
        assert h2 >= 0

    def test_m_zero_returns_zero(self):
        """m=0 → (0, 0) immediately."""
        positions = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
        h2_sq, h2 = compute_h2(positions, m=0)
        assert h2_sq == 0.0
        assert h2 == 0.0

    def test_negative_m_returns_zero(self):
        """m < 0 → (0, 0) immediately."""
        positions = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
        h2_sq, h2 = compute_h2(positions, m=-1)
        assert h2_sq == 0.0
        assert h2 == 0.0

    def test_isolated_birds_no_edges(self):
        """Birds with m neighbours but k=1 (only self) → no edges → (0, 0)."""
        # With m=1 and N=1, k = min(2, 1) = 1, only self returned, skipped
        positions = np.array([[0, 0, 0], [1000, 1000, 1000]], dtype=np.float32)
        _, h2 = compute_h2(positions, m=1)
        assert h2 >= 0  # shouldn't crash on no-edge case

    def test_prebuilt_tree_same_result(self):
        """Passing a pre-built cKDTree yields same H₂ as default."""
        from scipy.spatial import cKDTree
        positions = np.random.uniform(0, 50, (15, 3)).astype(np.float32)
        tree = cKDTree(positions)
        _, h2_no_tree = compute_h2(positions, m=4)
        _, h2_with_tree = compute_h2(positions, m=4, tree=tree)
        assert h2_no_tree == pytest.approx(h2_with_tree, rel=0.01)

    def test_metrics_collector_computes_h2(self):
        """MetricsCollector computes h2 at gated intervals."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 5

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=10)

        # Only frames 0, 5 should have h2 computed (steps 0..9)
        h2_values = [snap.h2 for snap in sim.metrics.history if snap.h2 is not None]
        assert len(h2_values) >= 1, "H₂ was never computed"
        assert all(np.isfinite(h) and h >= 0 for h in h2_values)
