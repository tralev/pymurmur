"""Phase 1 acceptance-criterion tests (P1.1-P1.5).

Split out of test_phase1.py (file-size split).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pymurmur.physics.occlusion import spherical_cap_occlusion


@pytest.mark.phase1
class TestCollinearOcclusion:
    """P1.1: True occlusion culling — collinear birds: only nearest visible."""

    def test_collinear_birds_only_nearest_visible(self) -> None:
        """Place 3 birds along same line-of-sight; only the nearest is visible."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 3 neighbours along +x axis at distances 10, 20, 30
        nbr_pos = np.array([
            [30.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],  # unsorted on purpose — occlusion sorts internally
            [10.0, 0.0, 0.0],
        ], dtype=np.float32)
        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        delta, visible_idx, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=9.0, blind_cos=None, anisotropy=1.0,
        )

        # Only the nearest bird (global index 2, at distance 10) should be visible
        assert len(visible_idx) == 1, (
            f"Expected 1 visible neighbour, got {len(visible_idx)}"
        )
        assert visible_idx[0] == 2, (
            f"Expected nearest bird (index 2), got index {visible_idx[0]}"
        )

    def test_empty_neighbours_returns_sensible_defaults(self) -> None:
        """No neighbours → empty visible_idx, zero theta, zero delta."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        empty = np.empty((0, 3), dtype=np.float32)

        delta, visible_idx, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, empty, empty,
        )

        assert len(visible_idx) == 0
        assert theta == 0.0
        assert np.allclose(delta, np.zeros(3), atol=1e-12)


# ── Θ sub-additive ∈ [0,1] (P1.2) ─────────────────────────────────

@pytest.mark.phase1
class TestThetaSubadditive:
    """P1.2: Internal opacity Θ ∈ [0,1] and sub-additive."""

    def test_theta_between_zero_and_one(self) -> None:
        """Θ is always between 0 and 1 for any valid configuration."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(1, 50)
            nbr_pos = rng.uniform(-50, 50, size=(n, 3)).astype(np.float32)
            nbr_vel = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
            obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            _, _, theta = spherical_cap_occlusion(
                obs_pos, obs_vel, nbr_pos, nbr_vel,
                boid_size=9.0,
            )

            assert 0.0 <= theta <= 1.0, f"Θ = {theta} not in [0,1]"

    def test_theta_subadditive(self) -> None:
        """Θ(A ∪ B) ≤ Θ(A) + Θ(B) — probabilistic-union is sub-additive."""
        rng = np.random.default_rng(99)
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        for _ in range(20):
            # Group A: 5 birds in +x hemisphere
            a_pos = rng.uniform([5, -30, -30], [50, 30, 30], size=(5, 3)).astype(np.float32)
            a_vel = rng.uniform(-1, 1, size=(5, 3)).astype(np.float32)

            # Group B: 5 birds in −x hemisphere (behind observer)
            b_pos = rng.uniform([-50, -30, -30], [-5, 30, 30], size=(5, 3)).astype(np.float32)
            b_vel = rng.uniform(-1, 1, size=(5, 3)).astype(np.float32)

            _, _, theta_a = spherical_cap_occlusion(
                obs_pos, obs_vel, a_pos, a_vel, boid_size=9.0,
            )
            _, _, theta_b = spherical_cap_occlusion(
                obs_pos, obs_vel, b_pos, b_vel, boid_size=9.0,
            )

            # Union
            ab_pos = np.vstack([a_pos, b_pos]).astype(np.float32)
            ab_vel = np.vstack([a_vel, b_vel]).astype(np.float32)
            _, _, theta_ab = spherical_cap_occlusion(
                obs_pos, obs_vel, ab_pos, ab_vel, boid_size=9.0,
            )

            assert theta_a >= 0.0 and theta_a <= 1.0
            assert theta_b >= 0.0 and theta_b <= 1.0
            assert theta_ab >= 0.0 and theta_ab <= 1.0
            assert theta_ab <= theta_a + theta_b + 1e-10, (
                f"Sub-additivity violated: Θ(A∪B)={theta_ab:.6f} "
                f"> Θ(A)={theta_a:.6f} + Θ(B)={theta_b:.6f}"
            )


# ── |δ̂| cancellation in fully-surrounded centre (P1.3) ────────────

@pytest.mark.phase1
class TestDeltaHatMagnitude:
    """P1.3: Boundary-length-weighted δ̂ — |δ̂| → 0 in centre, → 1 at edge."""

    def test_delta_hat_vanishes_when_fully_surrounded(self) -> None:
        """6 neighbours on ±x, ±y, ±z axes → weighted sum cancels, |δ̂| ≈ 0."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Symmetrically placed at equal distance on each axis
        nbr_pos = np.array([
            [10.0, 0.0, 0.0],    # +x
            [-10.0, 0.0, 0.0],   # −x
            [0.0, 10.0, 0.0],    # +y
            [0.0, -10.0, 0.0],   # −y
            [0.0, 0.0, 10.0],    # +z
            [0.0, 0.0, -10.0],   # −z
        ], dtype=np.float32)
        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        delta, _, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=9.0, blind_cos=None, anisotropy=1.0,
        )

        # All 6 directions cancel → |δ̂| < 1e-2
        delta_mag = float(np.linalg.norm(delta))
        assert delta_mag < 1e-2, (
            f"|δ̂| = {delta_mag:.6f}, expected < 1e-2 when fully surrounded"
        )
        # Θ should be > 0 since neighbours occupy solid angle
        assert theta > 0.0

    def test_delta_hat_at_flock_edge_near_one(self) -> None:
        """Neighbours on one side only → weighted sum nearly unit, |δ̂| ≈ 1."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # All neighbours in +x direction → |δ̂| ≈ 1
        nbr_pos = np.array([
            [10.0, 2.0, 1.0],
            [12.0, -1.0, 0.0],
            [9.0, 0.0, -2.0],
        ], dtype=np.float32)
        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        delta, _, _ = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=9.0, blind_cos=None, anisotropy=1.0,
        )

        delta_mag = float(np.linalg.norm(delta))
        assert delta_mag > 0.95, (
            f"|δ̂| = {delta_mag:.6f}, expected ≈ 1 at flock edge"
        )
        assert delta_mag <= 1.0, (
            f"|δ̂| = {delta_mag:.6f}, should not exceed 1.0"
        )

# ── Exact asin α vs small-angle approximation (P1.4) ──────────

@pytest.mark.phase1
class TestExactAsinAlpha:
    """P1.4: Exact α = asin(b_eff/d) replaces small-angle α ≈ b_eff/d.

    asin(x) > x for x > 0, so the exact cap is larger and occludes more.
    This test uses a borderline configuration where the small-angle
    approximation would leave B visible but exact asin blocks it.
    """

    def test_exact_asin_occludes_where_small_angle_would_not(self) -> None:
        """Bird B at 8.6° off-axis: occluded by exact asin, not by approx."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Bird A: directly ahead at d=10, boid_size=4 → cap_ratio = 0.4
        #   α_exact = asin(0.4) ≈ 0.4115 rad, cos_α_exact ≈ 0.9165
        #   α_small  = 0.4,              cos_α_small  ≈ 0.9211
        # Bird B: slightly off-axis so d̂_A·d̂_B between the two thresholds
        nbr_pos = np.array([
            [10.0, 0.0, 0.0],     # A (index 0) — directly ahead
            [20.0, 8.6, 0.0],      # B (index 1) — offset by ~8.6°
        ], dtype=np.float32)
        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        boid_size = 4.0
        delta, visible_idx, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=boid_size, blind_cos=None, anisotropy=1.0,
        )

        # Only bird A (index 0) should be visible — B is occluded
        assert len(visible_idx) == 1, (
            f"Expected 1 visible (A only), got {len(visible_idx)}"
        )
        assert visible_idx[0] == 0, (
            f"Expected bird A (index 0) visible, got index {visible_idx[0]}"
        )

        # Verify that exact asin matters: exact cos_α < small-angle cos_α
        cap_ratio = boid_size / 10.0  # b_eff / d_A
        cos_alpha_exact = math.cos(math.asin(cap_ratio))
        cos_alpha_small = math.cos(cap_ratio)  # small-angle approx

        assert cos_alpha_exact < cos_alpha_small, (
            f"Exact cos_α={cos_alpha_exact:.6f} should be < "
            f"small-angle cos_α={cos_alpha_small:.6f} (exact cap is bigger)"
        )

        # d̂_A · d̂_B must lie between exact and small-angle thresholds
        d_a = np.array([1.0, 0.0, 0.0])  # A direction
        d_b = np.array([20.0, 8.6, 0.0])
        d_b /= np.linalg.norm(d_b)
        dot_ab = float(np.dot(d_a, d_b))

        assert cos_alpha_exact <= dot_ab < cos_alpha_small, (
            f"d̂_A·d̂_B = {dot_ab:.6f} should be in "
            f"[{cos_alpha_exact:.6f}, {cos_alpha_small:.6f})"
        )

    def test_small_cap_ratio_exact_approx_nearly_equal(self) -> None:
        """At very small cap_ratio the exact asin ≈ small-angle approx.

        This is a mathematical-sanity check; it does not exercise
        spherical_cap_occlusion directly.
        """
        # cap_ratio = 0.01: asin(0.01) ≈ 0.010000167, cos ≈ 0.99995
        # small-angle: 0.01, cos ≈ 0.99995 — differences < 1e-7
        cap_ratio = 0.01
        cos_exact = math.cos(math.asin(cap_ratio))
        cos_small = math.cos(cap_ratio)

        # They should be very close but exact should still be slightly smaller
        assert abs(cos_exact - cos_small) < 1e-7, (
            f"For cap_ratio=0.01, exact and approx should nearly match: "
            f"{cos_exact:.12f} vs {cos_small:.12f}"
        )
        assert cos_exact < cos_small, (
            "Exact cos_α should still be ≤ small-angle (larger cap)"
        )

    def test_large_cap_ratio_exact_asin_diverges_strongly(self) -> None:
        """At cap_ratio=0.8 the exact asin diverges ~17% from small-angle.

        This is a mathematical-sanity check; it does not exercise
        spherical_cap_occlusion directly.
        """
        cap_ratio = 0.8
        alpha_exact = math.asin(cap_ratio)  # ≈ 0.9273 rad ≈ 53.1°
        alpha_small = cap_ratio            # 0.8 rad ≈ 45.8°

        relative_diff = (alpha_exact - alpha_small) / alpha_small
        assert relative_diff > 0.15, (
            f"Expected >15% relative difference at cap_ratio=0.8, "
            f"got {relative_diff*100:.1f}%"
        )

    def test_cascade_exact_asin_changes_occlusion_chain(self) -> None:
        """P1.1+P1.4: 3-bird cascade where exact asin on nearest bird
        determines whether the second bird is visible, which determines
        whether the third bird is occluded by the chain.

        Bird A at [10,0,0] (nearest), B at [20,8.6,0], C at [27.56,11.85,0]
        (C on same ray as B). boid_size=4 → cap_ratio_A=0.4.

        Exact asin: A occludes both B and C → only A visible.
        Small-angle: A is too small to occlude B → B visible → B
        occludes C → visible = [A, B].
        """
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        boid_size = 4.0

        # B and C are on the same ray from the observer:
        #   d̂ = normalize([20, 8.6, 0]) ≈ [0.918642, 0.395128, 0]
        # B at d≈21.77, C at d=30 along that same direction
        nbr_pos = np.array([
            [10.0,   0.0,    0.0],     # A (index 0)
            [20.0,   8.6,    0.0],     # B (index 1)
            [27.559, 11.854, 0.0],     # C (index 2) — same ray as B
        ], dtype=np.float32)
        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        delta, visible_idx, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=boid_size, blind_cos=None, anisotropy=1.0,
        )

        # Exact asin → A's cap is large enough to occlude B,
        # so B is never visible → B can't occlude C → A also occludes C.
        # Only A (index 0) is visible.
        assert len(visible_idx) == 1, (
            f"Expected 1 visible (A only), got {len(visible_idx)}"
        )
        assert visible_idx[0] == 0, (
            f"Expected A (index 0), got index {visible_idx[0]}"
        )

        # Verify the math: small-angle would give a different chain.
        cap_ratio = boid_size / 10.0  # b_eff / d_A
        cos_alpha_exact = math.cos(math.asin(cap_ratio))   # ≈ 0.9165
        cos_alpha_small = math.cos(cap_ratio)              # ≈ 0.9211

        # B and C share the same direction d̂
        d_a = np.array([1.0, 0.0, 0.0])
        d_bc = np.array([20.0, 8.6, 0.0])
        d_bc /= np.linalg.norm(d_bc)
        dot_abc = float(np.dot(d_a, d_bc))

        # Exact: A's cap blocks both B and C
        assert dot_abc >= cos_alpha_exact, (
            f"d̂_A·d̂_BC = {dot_abc:.6f} should be ≥ exact cos_α={cos_alpha_exact:.6f}"
        )

        # Small-angle: A's cap is too small → B and C pass A
        assert dot_abc < cos_alpha_small, (
            f"d̂_A·d̂_BC = {dot_abc:.6f} should be < small-angle cos_α={cos_alpha_small:.6f}"
        )

        # If B were visible, its cap would occlude C:
        #   α_B = asin(boid_size / d_B), cos_α_B for occlusion test
        d_b = float(np.linalg.norm(nbr_pos[1]))  # ~21.77
        cos_alpha_b = math.cos(math.asin(boid_size / d_b))  # ≈ 0.9830
        # C is on same ray as B → d̂_B·d̂_C = 1.0 → always occluded
        assert 1.0 >= cos_alpha_b > 0.0, (
            f"B's cap should be non-degenerate: cos_α_B={cos_alpha_b:.6f}"
        )


# ── 64-neighbour candidate cutoff (P1.5) ──────────────────

@pytest.mark.phase1
class TestCandidateCutoff:
    """P1.5: Only the nearest 64 neighbours are considered as candidates."""

    def test_128_collinear_only_nearest_visible(self) -> None:
        """128 collinear birds → only 64 considered, nearest occludes all."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 128 birds along +x at distances 5, 10, 15, ..., 640
        n = 128
        nbr_pos = np.zeros((n, 3), dtype=np.float32)
        nbr_pos[:, 0] = np.arange(1, n + 1, dtype=np.float32) * 5.0
        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        delta, visible_idx, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=9.0, blind_cos=None, anisotropy=1.0,
        )

        # Nearest bird (index 0 at d=5) has cap_ratio = 9/5 > 1
        # → covers entire forward view, occludes all behind it
        assert len(visible_idx) == 1, (
            f"Expected 1 visible (nearest occludes all), got {len(visible_idx)}"
        )
        assert visible_idx[0] == 0, (
            f"Expected nearest bird (index 0) visible, got index {visible_idx[0]}"
        )

    def test_128_spread_out_visible_capped_at_64(self) -> None:
        """128 spread-out birds, tiny boid_size → ≤64 visible (cutoff)."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Fibonacci-sphere-like distribution on the forward
        # hemisphere (x ≥ 0). Points with y → −0.2 may dip
        # slightly behind the observer but blind_cos=None so
        # all are processed identically.
        n = 128
        nbr_pos = np.zeros((n, 3), dtype=np.float32)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        for i in range(n):
            # Map i to [0, 1] for elevation: only forward hemisphere (y >= -0.2)
            t = i / (n - 1)
            # Elevation from nearly-horizontal (-0.2) to zenith (1.0)
            y = 1.0 - t * 1.2  # y ∈ [-0.2, 1.0]
            radius_at_y = np.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * i

            dist = 10.0 + i * 0.1  # deterministic distance ordering: i=0 nearest
            nbr_pos[i, 0] = dist * radius_at_y * np.cos(theta)
            nbr_pos[i, 1] = dist * y
            nbr_pos[i, 2] = dist * radius_at_y * np.sin(theta)

        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        delta, visible_idx, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=0.1, blind_cos=None, anisotropy=1.0,
        )

        # With boid_size=0.1 and distances ≥10, cap_ratio ≤ 0.01
        # → angular radius < 0.6° → birds don't occlude each other.
        # But only 64 nearest are considered.
        assert len(visible_idx) == 64, (
            f"Expected exactly 64 visible (cutoff at 64), got {len(visible_idx)}"
        )

    def test_cutoff_respects_max_candidates_parameter(self) -> None:
        """Explicit max_candidates=10 limits candidates even further."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 50 spread-out birds at varying distances
        n = 50
        rng = np.random.default_rng(77)
        nbr_pos = rng.uniform(1, 100, size=(n, 3)).astype(np.float32)
        # Ensure all are in forward hemisphere
        nbr_pos[:, 0] = np.abs(nbr_pos[:, 0])
        nbr_vel = np.ones_like(nbr_pos, dtype=np.float32)

        delta, visible_idx, theta = spherical_cap_occlusion(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
            boid_size=0.01, blind_cos=None, anisotropy=1.0,
            max_candidates=10,
        )

        # With tiny boid_size nothing occludes, but only 10 candidates checked
        assert len(visible_idx) == 10, (
            f"Expected exactly 10 visible (max_candidates=10), got {len(visible_idx)}"
        )


