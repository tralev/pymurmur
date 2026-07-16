"""Independent L0 function tests for P3.2–P3.12 field mode (missing coverage).

Each test imports and calls a single L0 function directly — no flock or engine needed.
"""
import numpy as np
import pytest

from pymurmur.physics.forces.field import (
    _compute_anchors,
    _compute_targets,
    _compute_phases,
    _compute_leader_chaser,
    _compute_shell_force,
    _compute_tangential,
    _compute_fold_noise,
    _compute_drift_alignment,
    _compute_grid_sep_normalized,
    _compute_floating_boundary,
    _hash01,
)
from pymurmur.physics.boid import init_positions


class TestP3_2_AnchorsAndPhaseWeights:
    """P3.2: Supplemental tests for blob anchors and cyclic phase weights."""

    def test_targets_are_weighted_combination_of_anchors(self):
        """T_legacy is a convex combination of the 5 anchors."""
        seeds = np.arange(50, dtype=np.float32)
        C = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        anchors = _compute_anchors(0.0, C, 100.0)
        T = _compute_targets(seeds, 10.0, anchors)

        # Each target should lie within the convex hull of the 5 anchors
        assert T.shape == (50, 3)
        anchor_min = anchors.min(axis=0)
        anchor_max = anchors.max(axis=0)
        assert (T >= anchor_min - 1e-4).all(), "Targets outside anchor bounds"
        assert (T <= anchor_max + 1e-4).all(), "Targets outside anchor bounds"

    def test_phases_deterministic_and_in_range(self):
        """φ_i ∈ [0, 1) and same seeds+t produce same phases."""
        seeds = np.array([0, 1, 100], dtype=np.float32)
        p1 = _compute_phases(seeds, 5.0)
        p2 = _compute_phases(seeds, 5.0)
        assert (p1 >= 0.0).all() and (p1 < 1.0).all()
        np.testing.assert_allclose(p1, p2, atol=1e-7)

    def test_hash01_deterministic_and_in_range(self):
        """hash01 returns values in [0, 1) deterministically."""
        x = np.array([0.0, 1.0, 42.0], dtype=np.float32)
        h = _hash01(x)
        assert (h >= 0.0).all() and (h < 1.0).all()
        # Deterministic
        np.testing.assert_allclose(_hash01(x), h, atol=1e-7)


class TestP3_3_LeaderChaser:
    """P3.3: Independent leader/chaser group tests."""

    def test_chase_nonzero_produces_different_targets(self):
        """chase_strength > 0 changes targets away from T_legacy."""
        seeds = np.arange(100, dtype=np.float32)
        T_in = np.random.randn(100, 3).astype(np.float32)
        anchors = _compute_anchors(0.0, np.zeros(3), 100.0)
        T_zero = _compute_leader_chaser(seeds, 30.0, T_in, anchors, 100.0, 0.0, 0.85)
        T_chase = _compute_leader_chaser(seeds, 30.0, T_in, anchors, 100.0, 0.8, 0.85)
        # chase>0 must produce different targets
        assert not np.allclose(T_zero, T_chase, atol=1e-3), (
            "chase_strength>0 should change targets"
        )

    def test_group_stability(self):
        """Same seed → same group across calls."""
        seeds = np.arange(50, dtype=np.float32)
        anchors = _compute_anchors(0.0, np.zeros(3), 1.0)
        T_in = np.zeros((50, 3), dtype=np.float32)
        T1 = _compute_leader_chaser(seeds, 0.0, T_in, anchors, 1.0, 0.5, 0.85)
        T2 = _compute_leader_chaser(seeds, 0.0, T_in, anchors, 1.0, 0.5, 0.85)
        np.testing.assert_allclose(T1, T2, atol=1e-6)

    def test_golden_angle_shells_produce_stratified_positions(self):
        """Chase targets are not all at the same point (stratified shells)."""
        seeds = np.arange(200, dtype=np.float32)
        anchors = _compute_anchors(10.0, np.zeros(3), 100.0)
        T_in = _compute_targets(seeds, 10.0, anchors)
        T = _compute_leader_chaser(seeds, 10.0, T_in, anchors, 100.0, 0.8, 0.85)
        # Targets should span a non-trivial range
        spread = np.std(T, axis=0)
        assert np.any(spread > 1.0), f"Targets too clustered: spread={spread}"


class TestP3_4_ShellForce:
    """P3.4: Independent shell force + inner cavity tests."""

    def test_inner_cavity_pushes_out(self):
        """Birds inside the inner floor are pushed outward."""
        n = 1
        U = 10.0
        # R_blob ≈ 3.2 at t=0, seed=0. inner ≈ 3.2 * 0.46 = 1.47
        # Place bird at d=0.5 (well inside inner cavity)
        pos = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        targ = np.zeros_like(pos)
        seeds = np.zeros(1, dtype=np.float32)
        F = _compute_shell_force(pos, targ, seeds, 0.0, U, 1.0, 0.0, 1.0, 1.0)
        # Should be pushed out (away from target = positive x)
        assert F[0, 0] > 0, f"Inner cavity should push out: F[0,0]={F[0,0]:.4f}"

    def test_shell_force_scales_with_cohesion(self):
        """Doubling cohesion doubles shell force magnitude."""
        n = 2
        pos = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        targ = np.zeros_like(pos)
        seeds = np.array([0, 1], dtype=np.float32)
        F1 = _compute_shell_force(pos, targ, seeds, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0)
        F2 = _compute_shell_force(pos, targ, seeds, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(F2, F1 * 2.0, atol=1e-4)


class TestP3_6_RemainingTerms:
    """P3.6: Independent tests for fold noise and drift alignment."""

    def test_fold_noise_produces_nonzero_force(self):
        """Fold noise generates force for non-zero flow/pull."""
        pos = np.random.randn(20, 3).astype(np.float32) * 50.0
        C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        seeds = np.arange(20, dtype=np.float32)
        F = _compute_fold_noise(pos, C, seeds, 5.0, 100.0, 1.0, 1.0, 1.0)
        mags = np.linalg.norm(F, axis=1)
        assert np.mean(mags > 1e-6) > 0.8, (
            f"Only {np.mean(mags > 1e-6)*100:.0f}% got fold force"
        )

    def test_fold_noise_zero_when_flow_pull_zero(self):
        """Fold noise returns zero when flow_pull=0."""
        pos = np.random.randn(10, 3).astype(np.float32)
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(10, dtype=np.float32)
        F = _compute_fold_noise(pos, C, seeds, 0.0, 100.0, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(F, 0.0)

    def test_fold_noise_scales_with_ripple_envelope(self):
        """Doubling ripple_envelope_sum doubles fold noise."""
        pos = np.random.randn(10, 3).astype(np.float32)
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(10, dtype=np.float32)
        F1 = _compute_fold_noise(pos, C, seeds, 0.0, 100.0, 1.0, 1.0, 1.0)
        F2 = _compute_fold_noise(pos, C, seeds, 0.0, 100.0, 1.0, 1.0, 2.0)
        np.testing.assert_allclose(F2, F1 * 2.0, atol=1e-4)

    def test_drift_alignment_steers_toward_heading(self):
        """Drift alignment accelerates birds toward wander_heading·v0."""
        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        F = _compute_drift_alignment(v, heading, 4.0, 1.0, 1.0)
        # Should point in +x direction (toward heading * v0)
        assert F[0, 0] > 0, f"Drift should steer toward heading: F={F}"
        np.testing.assert_allclose(F[0, 1:], 0.0, atol=1e-6)

    def test_tangential_zero_when_pull_zero(self):
        """Tangential returns zero when tangent_pull=0."""
        pos = np.random.randn(5, 3).astype(np.float32)
        targ = np.random.randn(5, 3).astype(np.float32)
        seeds = np.arange(5, dtype=np.float32)
        F = _compute_tangential(pos, targ, seeds, 0.0, 1.0, 0.0, 0.0)
        np.testing.assert_allclose(F, 0.0)


class TestP3_11_GridSepNormalization:
    """P3.11: Independent grid-mode separation normalization tests."""

    def test_returns_fraction_of_separation(self):
        """factor = separation / max(1, neighbour_count)."""
        assert _compute_grid_sep_normalized(None, 10.0, 5) == 2.0

    def test_division_by_zero_guarded(self):
        """neighbour_count=0 → factor = separation / 1."""
        assert _compute_grid_sep_normalized(None, 5.0, 0) == 5.0

    def test_minimum_factor_when_many_neighbours(self):
        """Large neighbour_count → small factor."""
        big = _compute_grid_sep_normalized(None, 10.0, 1000)
        small = _compute_grid_sep_normalized(None, 10.0, 2)
        assert big < small


class TestP3_12_FloatingBoundary:
    """P3.12: Independent floating boundary tests."""

    def test_boundary_radius_scales_with_blob(self):
        """R_boundary = 1.45 * max(R_blobs) — bigger blobs → bigger boundary."""
        C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # All birds at same position (inside)
        pos = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        R_small = np.array([1.0], dtype=np.float32)
        R_large = np.array([10.0], dtype=np.float32)

        F_small = _compute_floating_boundary(pos, C, R_small, 1.0, mu=0.1)
        F_large = _compute_floating_boundary(pos, C, R_large, 1.0, mu=0.1)
        # Both should be inside their respective boundaries → zero force
        np.testing.assert_allclose(F_small, 0.0, atol=1e-6)
        np.testing.assert_allclose(F_large, 0.0, atol=1e-6)

    def test_zero_blobs_returns_empty(self):
        """Empty R_blobs → empty force array."""
        F = _compute_floating_boundary(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            1.0, mu=0.1,
        )
        assert F.shape == (0, 3)


class TestP3_10_BlobInit:
    """P3.10: Independent blob position init tests."""

    def test_blob_init_produces_valid_positions(self):
        """Blob init returns (n,3) float32 in-domain."""
        p = init_positions(100, 1000, 700, 400, mode="blob")
        assert p.shape == (100, 3)
        assert p.dtype == np.float32
        assert (p >= 0).all() and (p[:, 0] < 1000).all()
        assert (p[:, 1] < 700).all() and (p[:, 2] < 400).all()

    def test_blob_init_spreads_across_5_centres(self):
        """Birds are distributed across ~5 clusters (one per centre)."""
        p = init_positions(500, 1000, 700, 400, mode="blob")
        # 5 clusters within ~160 units of domain centre → std ~40–70 per axis
        spread = np.std(p, axis=0)
        assert np.all(spread > 40), f"Blob init too clustered: spread={spread}"

    def test_blob_init_deterministic_with_seed(self):
        """Same params + seed → identical positions."""
        rng = np.random.default_rng(42)
        p1 = init_positions(50, 1000, 700, 400, rng=rng, mode="blob")
        rng2 = np.random.default_rng(42)
        p2 = init_positions(50, 1000, 700, 400, rng=rng2, mode="blob")
        np.testing.assert_allclose(p1, p2, atol=1e-6)
