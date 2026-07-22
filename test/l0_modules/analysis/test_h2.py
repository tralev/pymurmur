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
        """H₂ < N for any connected graph (virtually guaranteed connected).

        Uses a tight cluster in a small domain with m=10 to guarantee
        connectivity — random positions in a large domain can produce
        disconnected k-NN graphs even with seeding.
        """
        rng = np.random.default_rng(42)
        # 15 birds tightly packed in [0, 10]³ → virtually guaranteed connected
        positions = rng.uniform(0, 10, (15, 3)).astype(np.float32)
        _, h2 = compute_h2(positions, m=10)
        assert np.isfinite(h2), f"H₂ should be finite for connected graph, got {h2}"
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
        """Birds at single point — all edges built, graph is connected → finite H₂."""
        positions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        _, h2 = compute_h2(positions, m=2)
        # All birds co-located → complete graph → connected → finite H₂
        assert np.isfinite(h2)

    def test_h2_inf_when_disconnected(self):
        """Disconnected k-NN graph returns (inf, inf) — P0.13."""
        import math
        # Four birds in two isolated clusters — m=1 means each bird only
        # connects within its cluster, graph is disconnected
        pts = np.array([[0, 0, 0], [1, 0, 0], [1000, 0, 0], [1001, 0, 0]], dtype=np.float32)
        _, h2 = compute_h2(pts, m=1)
        assert math.isinf(h2), f"Expected inf for disconnected graph, got {h2}"

    def test_prebuilt_tree_same_result(self):
        """Passing a pre-built cKDTree yields same H₂ as default."""
        from scipy.spatial import cKDTree
        positions = np.random.uniform(0, 50, (15, 3)).astype(np.float32)
        tree = cKDTree(positions)
        _, h2_no_tree = compute_h2(positions, m=4)
        _, h2_with_tree = compute_h2(positions, m=4, tree=tree)
        assert h2_no_tree == pytest.approx(h2_with_tree, rel=0.01)

    def test_hand_3node_max_form_symmetrization(self):
        """S1.8: 3-node directed k-NN graph symmetrizes via max(A, Aᵀ),
        not (A + Aᵀ)/2.

        Points at x=0,1,3 with m=1 (k=2): node 0's nearest other point
        is node 1 (0→1); node 1's nearest is node 0 (1→0); node 2's
        nearest is node 1 (2→1), but node 1's nearest is node 0, not
        node 2 — so the raw k-NN graph has a one-directional edge
        1↔2. Under max-form symmetrization this becomes a *full-weight*
        edge, producing the unweighted path graph P3 (0-1-2), whose
        Laplacian eigenvalues are analytically {0, 1, 3}
        (2 − 2·cos(kπ/3) for k=0,1,2). Under the old average-form
        ((A+Aᵀ)/2) the 1-2 edge would have weight 0.5 and this exact
        value would not hold.
        """
        positions = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=np.float32)
        h2_sq, h2 = compute_h2(positions, m=1)
        # h2_sq = (1/2N) Σ 1/λ_i over nonzero λ = (1/6)(1/1 + 1/3) = 2/9
        assert h2_sq == pytest.approx(2.0 / 9.0, abs=1e-6)
        assert h2 == pytest.approx(np.sqrt(2.0 / 9.0), abs=1e-6)

    def test_uniform_1m_weighting_scales_h2(self):
        """A8: edges are weighted aᵢⱼ = 1/m (Young et al. 2013), not 1.0.

        3 collinear birds at m=2 (k=3): every bird's 2 nearest others
        are the other two birds, so the k-NN graph is the complete
        graph K3 — already symmetric, isolating the weighting question
        from the symmetrization question (covered separately by
        test_hand_3node_max_form_symmetrization above).

        Weighted (aᵢⱼ = 1/m = 0.5): the K3 Laplacian is exactly 0.5×
        the unweighted K3 Laplacian, so its eigenvalues are exactly
        0.5× the standard K3 result {0, 3, 3} → {0, 1.5, 1.5}.
        h2_sq = (1/2N) Σ 1/λ_i = (1/6)(1/1.5 + 1/1.5) = 2/9 → h2 ≈ 0.4714.

        Under the old unweighted form (aᵢⱼ = 1.0) this same graph gives
        eigenvalues {0, 3, 3} → h2_sq = (1/6)(1/3 + 1/3) = 1/9 → h2 ≈
        0.3333 — a different, smaller value. This test would fail under
        the old unweighted adjacency.
        """
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        h2_sq, h2 = compute_h2(positions, m=2)
        assert h2_sq == pytest.approx(2.0 / 9.0, abs=1e-6)
        assert h2 == pytest.approx(np.sqrt(2.0 / 9.0), abs=1e-6)
        # Explicitly rule out the old unweighted value, so a regression
        # back to aᵢⱼ = 1.0 fails loudly rather than silently.
        assert h2 != pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_metrics_collector_computes_h2(self):
        """MetricsCollector computes h2 at gated intervals."""

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
