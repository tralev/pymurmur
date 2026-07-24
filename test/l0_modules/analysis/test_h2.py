"""H₂ robustness tests — Phase 9.1

k-NN graph, Laplacian eigenvalues, cost-optimal m*.
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    compute_convergence_speed,
    compute_h2,
    compute_h2_lyapunov,
    compute_r_nodal,
    find_optimal_m,
)
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


class TestConvergenceSpeed:
    """A10: consensus convergence speed λ₂(L) — the algebraic
    connectivity / Fiedler value of the same k-NN Laplacian H₂ is
    built from. Distinct from H₂: increases monotonically with m
    (no interior optimum), where H₂ decreases."""

    def test_n_less_than_2_returns_zero(self):
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        assert compute_convergence_speed(positions, m=5) == 0.0

    def test_m_zero_returns_zero(self):
        positions = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
        assert compute_convergence_speed(positions, m=0) == 0.0

    def test_negative_m_returns_zero(self):
        positions = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
        assert compute_convergence_speed(positions, m=-1) == 0.0

    def test_disconnected_graph_returns_exactly_zero(self):
        """Unlike H₂ (which returns +inf on disconnection — infinite
        disagreement), λ₂ = 0 exactly is the mathematically correct
        value for a disconnected graph — no special-casing needed.
        Reuses the same two-cluster fixture as
        TestH2Robustness.test_h2_inf_when_disconnected, so both
        metrics are shown disagreeing about the *same* graph.
        """
        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [1000, 0, 0], [1001, 0, 0]], dtype=np.float32
        )
        _, h2 = compute_h2(pts, m=1)
        speed = compute_convergence_speed(pts, m=1)
        assert not np.isfinite(h2), "sanity check: this graph must be disconnected"
        assert speed == 0.0

    def test_hand_3node_k3_lambda2(self):
        """Same 3-node m=2 complete-graph K3 setup as
        test_uniform_1m_weighting_scales_h2: weighted (aᵢⱼ=0.5)
        eigenvalues are exactly {0, 1.5, 1.5}, so λ₂ = 1.5 exactly.
        """
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        speed = compute_convergence_speed(positions, m=2)
        assert speed == pytest.approx(1.5, abs=1e-6)

    def test_a10_robustness_vs_speed_contrast(self):
        """The headline A10 claim: as m grows, H₂ (robustness) falls
        while λ₂ (convergence speed) rises — opposite trends on the
        exact same graph sequence, which is why a finite m* exists at
        all (nature trades some speed for robustness, not the other
        way around).
        """
        rng = np.random.default_rng(7)
        positions = rng.uniform(0, 100, (60, 3)).astype(np.float32)
        ms = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]

        h2_values = []
        speed_values = []
        for m in ms:
            _, h2 = compute_h2(positions, m)
            h2_values.append(h2)
            speed_values.append(compute_convergence_speed(positions, m))

        # H2: monotonically non-increasing (skip the disconnected m=2
        # inf, which trivially "decreases" to the next finite value).
        finite_h2 = [h for h in h2_values if np.isfinite(h)]
        assert all(
            finite_h2[i] >= finite_h2[i + 1] - 1e-9
            for i in range(len(finite_h2) - 1)
        ), f"H2 should be non-increasing in m: {finite_h2}"

        # Convergence speed: monotonically non-decreasing, no interior
        # optimum, over the identical m range and positions.
        assert all(
            speed_values[i] <= speed_values[i + 1] + 1e-9
            for i in range(len(speed_values) - 1)
        ), f"convergence speed should be non-decreasing in m: {speed_values}"

        # The contrast itself: speed's last value is well above its
        # first (finite) value, while H2's last value is well below
        # its first — genuinely opposite trends, not just "both flat".
        assert speed_values[-1] > speed_values[2] * 2, (
            "convergence speed should rise substantially over the m range"
        )
        assert finite_h2[-1] < finite_h2[0] * 0.5, (
            "H2 should fall substantially over the m range"
        )

    def test_prebuilt_tree_same_result(self):
        from scipy.spatial import cKDTree
        positions = np.random.uniform(0, 50, (15, 3)).astype(np.float32)
        tree = cKDTree(positions)
        speed_no_tree = compute_convergence_speed(positions, m=4)
        speed_with_tree = compute_convergence_speed(positions, m=4, tree=tree)
        assert speed_no_tree == pytest.approx(speed_with_tree, rel=0.01)


class TestH2Lyapunov:
    """A5 (Young et al. 2013): H₂ via the Lyapunov/trace route —
    a second, independent derivation that must agree numerically
    with compute_h2's eigenvalue-shortcut result."""

    def test_n_less_than_2_returns_zero(self):
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        assert compute_h2_lyapunov(positions, m=5) == (0.0, 0.0)

    def test_m_zero_returns_zero(self):
        positions = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
        assert compute_h2_lyapunov(positions, m=0) == (0.0, 0.0)

    def test_disconnected_graph_returns_inf(self):
        """Same fixture as test_h2_inf_when_disconnected -- both
        derivations must agree it's disconnected."""
        import math
        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [1000, 0, 0], [1001, 0, 0]], dtype=np.float32
        )
        _, h2_lyap = compute_h2_lyapunov(pts, m=1)
        assert math.isinf(h2_lyap)

    def test_hand_3node_k3_matches_compute_h2(self):
        """Same 3-node K3 (m=2) hand-computed case as A8's
        test_uniform_1m_weighting_scales_h2: h2_sq = 2/9 exactly."""
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        h2_sq, h2 = compute_h2_lyapunov(positions, m=2)
        assert h2_sq == pytest.approx(2.0 / 9.0, abs=1e-6)
        assert h2 == pytest.approx(np.sqrt(2.0 / 9.0), abs=1e-6)

    def test_matches_compute_h2_across_random_graphs(self):
        """The two independent derivations (eigenvalue shortcut vs.
        Lyapunov/trace) must agree on connected random graphs -- this
        is the paper's own claim (A5: 'shows both paths ... same
        answer'), verified numerically here rather than assumed."""
        rng = np.random.default_rng(3)
        checked = 0
        for N in (8, 15, 30):
            for m in (3, 5):
                if m >= N:
                    continue
                positions = rng.uniform(-50, 50, (N, 3)).astype(np.float32)
                h2_sq_eigen, _ = compute_h2(positions, m)
                if not np.isfinite(h2_sq_eigen):
                    continue  # skip disconnected graphs
                h2_sq_lyap, _ = compute_h2_lyapunov(positions, m)
                assert h2_sq_lyap == pytest.approx(h2_sq_eigen, rel=1e-6), (
                    f"N={N} m={m}: eigen={h2_sq_eigen} lyapunov={h2_sq_lyap}"
                )
                checked += 1
        assert checked >= 4, "too few connected graphs to be a meaningful check"

    def test_prebuilt_tree_same_result(self):
        from scipy.spatial import cKDTree
        positions = np.random.uniform(0, 50, (15, 3)).astype(np.float32)
        tree = cKDTree(positions)
        _, h2_no_tree = compute_h2_lyapunov(positions, m=4)
        _, h2_with_tree = compute_h2_lyapunov(positions, m=4, tree=tree)
        assert h2_no_tree == pytest.approx(h2_with_tree, rel=0.01)


class TestRNodal:
    """A6 (Young et al. 2013): H₂ nodal robustness R_nodal = √N/H₂ --
    normalized so *larger* means *more robust*, unlike H₂ itself."""

    def test_disconnected_h2_gives_zero(self):
        """Per A6's own spec: 'Zero when the graph is disconnected.'"""
        assert compute_r_nodal(float('inf'), N=10) == 0.0

    def test_degenerate_h2_gives_zero(self):
        """h2<=0 is this codebase's N<2/failure sentinel, not a real
        zero-disagreement state -- must not divide-by-zero to inf."""
        assert compute_r_nodal(0.0, N=10) == 0.0

    def test_hand_computed(self):
        # R_nodal = sqrt(N)/H2
        assert compute_r_nodal(h2=2.0, N=16) == pytest.approx(2.0, abs=1e-9)

    def test_larger_h2_gives_smaller_r_nodal(self):
        """H2 is a disagreement measure (smaller=better); R_nodal
        inverts it, so robustness ordering flips."""
        r_low_disagreement = compute_r_nodal(h2=0.5, N=20)
        r_high_disagreement = compute_r_nodal(h2=2.0, N=20)
        assert r_low_disagreement > r_high_disagreement

    def test_metrics_collector_computes_r_nodal(self):
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 5

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=10)

        computed = [s for s in sim.metrics.history if s.r_nodal is not None]
        assert len(computed) >= 1, "r_nodal was never computed"
        for snap in computed:
            assert snap.r_nodal >= 0.0
            assert np.isfinite(snap.r_nodal)
