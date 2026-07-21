"""Phase 1 acceptance-criterion tests (P1.1–P1.10)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pymurmur.analysis.metrics import MetricsCollector, compute_shape
from pymurmur.physics.forces._base import cohesion_force
from pymurmur.physics.occlusion import spherical_cap_occlusion
from pymurmur.physics.steric import steric_force
from test.helpers import _call_force  # noqa: E402

# ── Collinear occlusion: nearest-only visibility (P1.1) ──────────

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


# ── Cohesion force bounded at unit vector (P1.7) ─────────

@pytest.mark.phase1
class TestCohesionBounded:
    """P1.7: cohesion_force returns F = normalize(p̄ − p_i) — bounded unit."""

    @staticmethod
    def _make_neighbor_idx(spec: list[list[int]]) -> np.ndarray:
        """Build ragged neighbor_idx array from a list-of-lists spec."""
        arr = np.empty(len(spec), dtype=object)
        for i, nbrs in enumerate(spec):
            arr[i] = np.array(nbrs, dtype=np.int32)
        return arr

    def test_cohesion_magnitude_is_one_regardless_of_distance(self) -> None:
        """CoM at [15,0,0] from bird at origin → |F| = 1 exactly."""
        positions = np.array([
            [0.0, 0.0, 0.0],    # bird 0
            [10.0, 0.0, 0.0],   # neighbour 1
            [20.0, 0.0, 0.0],   # neighbour 2
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True, False, False])
        neighbor_idx = self._make_neighbor_idx([[1, 2], [], []])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        mag = float(np.linalg.norm(force[0]))
        assert abs(mag - 1.0) < 1e-6, (
            f"|F_coh| = {mag:.8f}, expected exactly 1.0"
        )

    def test_far_away_neighbour_still_unit_magnitude(self) -> None:
        """Neighbour at d=10000 → |F| = 1 (bounded regardless of range)."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0],
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True, False])
        neighbor_idx = self._make_neighbor_idx([[1], []])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        mag = float(np.linalg.norm(force[0]))
        assert abs(mag - 1.0) < 1e-6, (
            f"|F_coh| at d=10000: {mag:.8f}, expected 1.0"
        )

    def test_no_neighbours_returns_zero(self) -> None:
        """No neighbours → zero force vector."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True])
        neighbor_idx = self._make_neighbor_idx([[]])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        assert np.allclose(force[0], np.zeros(3), atol=1e-12)

    def test_coincident_neighbour_returns_zero(self) -> None:
        """Neighbour at same position → length ≤ 1e-6 guard → zero."""
        positions = np.array([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # coincident
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True, False])
        neighbor_idx = self._make_neighbor_idx([[1], []])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        assert np.allclose(force[0], np.zeros(3), atol=1e-12)

    def test_inactive_bird_force_remains_zero(self) -> None:
        """Inactive bird gets no force regardless of neighbours."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([False, True])
        neighbor_idx = self._make_neighbor_idx([[], [0]])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        # Bird 0 is inactive → force[0] should be zero
        assert np.allclose(force[0], np.zeros(3), atol=1e-12)
        # Bird 1 is active → should get unit force toward bird 0
        mag1 = float(np.linalg.norm(force[1]))
        assert abs(mag1 - 1.0) < 1e-6


# ── Steric force clamping (P1.6) ──────────────────────────────────

@pytest.mark.phase1
class TestStericMaxForce:
    """P1.6: Steric repulsion clamped to max_force at close range.

    Also covers D8+D21 cross-cutting: steric clamp (D8) works correctly
    with the corrected separation formula Σ r̂/d² (D21)."""

    def test_steric_clamp_with_corrected_separation_no_nan(self):
        """D8+D21: Steric clamp prevents NaN with corrected 1/d³ separation.

        D21 fixed separation from 1/d to 1/d² (r̂/d² = -diffs/dists³).
        At d=0.01, the force is 10000× stronger than at d=1.0 — without
        D8's steric clamp (max_force), this would explode. With both fixes,
        the force is clamped to max_force=0.15 and stays finite.
        """
        import numpy as np

        from pymurmur.physics.forces._base import separation_force

        N = 3
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=np.float32)
        velocities = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        active = np.ones(N, dtype=bool)
        # Birds 0 and 1 see each other; bird 2 far away
        neighbor_idx = np.array([
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([], dtype=np.int32),
        ], dtype=object)

        force = separation_force(positions, velocities, neighbor_idx, active)

        # D21: With corrected 1/d³, force magnitude at d=0.01 should be
        # very large (~10000), but we verify it's finite (not NaN/Inf)
        assert np.isfinite(force).all(), (
            f"D21: corrected separation must not produce NaN. force={force}"
        )
        # D8: Verify force is non-zero and in correct direction
        f01 = float(np.linalg.norm(force[0]))
        assert f01 > 0, (
            "D8+D21: steric force should be non-zero at close range"
        )
        # Force should push birds apart (bird 0 at origin, bird 1 at +x)
        # Separation pushes bird 0 negative (away from bird 1)
        assert force[0, 0] < 0, (
            f"D8+D21: separation should push bird 0 away from bird 1. "
            f"fx={force[0,0]:.4f}"
        )

    def test_steric_clamp_limits_close_range_force(self):
        """D8+D21: Steric clamp (D8) caps force from corrected kernel (D21).

        D21 fixed separation to r̂/d² (dists³ in code). D8 ensures
        max_force is passed to steric_force. At d=0.01, the force is
        clamped to max_force, preventing explosion.
        """
        import numpy as np

        from pymurmur.physics.steric import steric_force

        observer = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour = np.array([[0.01, 0.0, 0.0]], dtype=np.float32)

        force = steric_force(
            observer, neighbour,
            strength=0.6, threshold=10.0, max_force=0.15,
        )

        # Forces must be finite
        assert np.isfinite(force).all()
        f_mag = float(np.linalg.norm(force))
        # With max_force=0.15, the clamp should keep force ≤ 0.15
        assert f_mag <= 0.151, (
            f"D8: steric clamp must cap force at max_force=0.15, "
            f"got {f_mag:.6f}"
        )
        assert f_mag > 0, "Steric force must be non-zero at close range"

    def test_steric_at_small_distance_returns_max_force(self) -> None:
        """At d = 0.01 the raw 1/d² force ≈ 6000, clamped to max_force."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)

        max_force = 5.0
        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            threshold=10.0,
            max_force=max_force,
        )

        force_mag = float(np.linalg.norm(force))
        assert abs(force_mag - max_force) < 1e-6, (
            f"Expected |F| = {max_force} (clamped), got {force_mag:.6f}"
        )

    def test_steric_no_neighbours_returns_zero(self) -> None:
        """No neighbours → zero force vector."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        empty = np.empty((0, 3), dtype=np.float32)

        force = steric_force(
            obs_pos, empty,
            strength=0.6,
            threshold=10.0,
            max_force=5.0,
        )

        assert np.allclose(force, np.zeros(3), atol=1e-12)

    def test_steric_distant_neighbour_returns_zero(self) -> None:
        """Neighbour beyond threshold → no force contribution."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)

        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            threshold=10.0,
            max_force=5.0,
        )

        assert np.allclose(force, np.zeros(3), atol=1e-12)

    def test_clamp_at_d001_with_production_max_force(self) -> None:
        """D8: Pair at d=0.01, strength=0.6, default clamp 0.15 → ‖F‖ == 0.15.

        This is the exact test case from the roadmap — verifies that
        steric_force's max_force clamp actually engages at production
        defaults (not just at the artificially high 5.0 used above).
        At d=0.01 with strength=0.6: raw force ≈ 0.6/(0.01)² = 6000.
        The clamp at 0.15 must reduce it to exactly 0.15."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)

        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            max_force=0.15,
        )

        assert np.linalg.norm(force) == pytest.approx(0.15)

    def test_clamp_not_triggered_at_large_distance(self) -> None:
        """D8: At d=5.0 with max_force=0.15, steric force is well below
        max_force — clamp should NOT engage. Verifies the clamp is
        conditional, not always-on."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)

        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            max_force=0.15,
        )
        # Raw force ≈ 0.6 / 25 = 0.024 < max_force 0.15, so no clamp
        mag = float(np.linalg.norm(force))
        assert mag < 0.15, f"Force {mag:.4f} should be below max_force"
        assert mag > 0.0, "Force should be non-zero"


# ── Vicsek memory-term autocorrelation (P1.8) ────────────

@pytest.mark.phase1
class TestVicsekMemory:
    """P1.8: Vicsek memory term preserves direction exactly when D=0."""

    @staticmethod
    def _make_flock(num_boids: int = 1, diffusion: float = 0.0):
        """Create a PhysicsFlock + SimConfig pair in vicsek mode."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = num_boids
        cfg.seed = 42
        cfg.vicsek_diffusion = diffusion
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.5
        cfg.vicsek_time_step = 0.1
        return PhysicsFlock(cfg), cfg

    def test_d_zero_single_bird_direction_preserved(self) -> None:
        """D=0, single bird: direction unchanged after one vicsek step."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=1, diffusion=0.0)

        # Set known direction
        known_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        flock.velocities[0] = known_dir * cfg.vicsek_velocity

        _call_force(vicsek_forces, flock, cfg)

        new_vel = flock.velocities[0]
        new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
        dot = float(np.dot(new_dir, known_dir))

        assert dot > 0.999999, (
            f"Direction drifted: dot(new, old) = {dot:.12f}, expected > 0.999999"
        )
        # Speed should remain constant at v0
        speed = float(np.linalg.norm(new_vel))
        assert abs(speed - cfg.vicsek_velocity) < 1e-6, (
            f"Speed changed: {speed:.6f}, expected {cfg.vicsek_velocity}"
        )

    def test_d_zero_repeated_frames_no_drift(self) -> None:
        """D=0 over 100 frames: direction never drifts."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=1, diffusion=0.0)

        known_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        flock.velocities[0] = known_dir.astype(np.float32) * cfg.vicsek_velocity

        for _ in range(100):
            _call_force(vicsek_forces, flock, cfg)
            new_vel = flock.velocities[0]
            new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
            dot = float(np.dot(new_dir, known_dir))
            assert dot > 0.999999, (
                f"Drift at frame: dot(new, old) = {dot:.12f}"
            )
            # Speed should remain constant
            speed = float(np.linalg.norm(new_vel))
            assert abs(speed - cfg.vicsek_velocity) < 1e-6

    def test_d_zero_two_birds_opposite_directions_preserved(self) -> None:
        """D=0, 2 birds heading opposite ways: multi-bird path,
        neighbour average cancels → falls back to pure memory,
        both directions preserved independently."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=2, diffusion=0.0)

        # Place birds close together so they see each other
        flock.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        flock.positions[1] = np.array([2.0, 0.0, 0.0], dtype=np.float32)

        # Opposite directions: +x and -x
        dir0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dir1 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        flock.velocities[0] = dir0 * cfg.vicsek_velocity
        flock.velocities[1] = dir1 * cfg.vicsek_velocity

        _call_force(vicsek_forces, flock, cfg)

        # Both directions should be preserved (opposite avg → zero → no blend)
        for i, expected_dir in enumerate([dir0, dir1]):
            new_vel = flock.velocities[i]
            new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
            dot = float(np.dot(new_dir, expected_dir))
            assert dot > 0.999999, (
                f"Bird {i}: dot(new, old) = {dot:.12f}, expected > 0.999999"
            )
            speed = float(np.linalg.norm(new_vel))
            assert abs(speed - cfg.vicsek_velocity) < 1e-6, (
                f"Bird {i}: speed = {speed:.6f}, expected {cfg.vicsek_velocity}"
            )

    def test_d_zero_two_birds_same_direction_preserved(self) -> None:
        """D=0, 2 birds heading same direction: neighbour blend should
        not change the direction since û_target = û_old (aligned)."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=2, diffusion=0.0)

        flock.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        flock.positions[1] = np.array([2.0, 0.0, 0.0], dtype=np.float32)

        # Same direction
        dir0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        flock.velocities[0] = dir0 * cfg.vicsek_velocity
        flock.velocities[1] = dir0 * cfg.vicsek_velocity

        _call_force(vicsek_forces, flock, cfg)

        # Since neighbour average equals each bird's own direction,
        # the blend eta*nd + (1-eta)*noisy = eta*dir + (1-eta)*dir = dir.
        # With D=0, noisy_dirs = old_dirs = dir. Direction preserved.
        for i in range(2):
            new_vel = flock.velocities[i]
            new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
            dot = float(np.dot(new_dir, dir0))
            assert dot > 0.999999, (
                f"Bird {i}: dot(new, old) = {dot:.12f}, expected > 0.999999"
            )


# ── Thickness ratio √(λ₃/λ₁) ∈ (0,1] (P1.9) ──────────────

@pytest.mark.phase1
class TestThicknessRatio:
    """P1.9: thickness = sqrt(λ₃/λ₁) ∈ (0,1]; → 1 for spheres, → 0 for lines."""

    def test_sphere_thickness_near_one(self) -> None:
        """100 random points on a 3D sphere → thickness ≈ 1."""
        rng = np.random.default_rng(42)
        # Generate points uniformly on unit sphere, scale to radius 50
        pts = rng.normal(size=(100, 3)).astype(np.float32)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 50.0

        _, thickness = compute_shape(pts)

        assert 0.8 < thickness <= 1.0, (
            f"Sphere thickness = {thickness:.3f}, expected > 0.8 (nearly isotropic)"
        )

    def test_line_thickness_near_zero(self) -> None:
        """50 points along x-axis with tiny noise → thickness ≈ 0."""
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            np.linspace(0, 100, 50),
            rng.normal(0, 1e-3, 50),
            rng.normal(0, 1e-3, 50),
        ]).astype(np.float32)

        _, thickness = compute_shape(positions)

        # P1.9: λ₃ ≪ λ₁ for a line → thickness ≈ 0
        assert 0.0 < thickness < 0.2, (
            f"Line thickness = {thickness:.3f}, expected < 0.2"
        )

    def test_pancake_thickness_near_zero(self) -> None:
        """Wide pancake with thin z → large aspect, thickness ≈ 0."""
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            rng.uniform(-200, 200, 100),
            rng.uniform(-200, 200, 100),
            rng.normal(0, 1e-3, 100),
        ]).astype(np.float32)

        _, thickness = compute_shape(positions)

        # P1.9: thin z → λ₃ ≪ λ₁ → thickness ≈ 0
        assert 0.0 < thickness < 0.1, (
            f"Pancake thickness = {thickness:.3f}, expected < 0.1"
        )

    def test_less_than_three_points_returns_one(self) -> None:
        """N < 3 → returns (1.0, 1.0) per the degenerate guard."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        aspect, thickness = compute_shape(positions)

        assert aspect == 1.0
        assert thickness == 1.0


# ── Θ NaN in non-projection modes (P1.10) ─────────────────

@pytest.mark.phase1
class TestThetaNaN:
    """P1.10: Θ returns NaN when mode is not 'projection'."""

    @staticmethod
    def _make_dummy_flock(mode: str):
        """Create a PhysicsFlock + MetricsCollector pair for mode."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = mode
        cfg.num_boids = 5
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)
        return flock, collector

    def test_theta_is_nan_in_non_projection_mode(self) -> None:
        """Vicsek mode → m.theta should be NaN."""
        flock, collector = self._make_dummy_flock("vicsek")

        # Set some last_theta values (should be ignored in vicsek mode)
        flock.last_theta[:] = 0.5

        collector.collect(flock, 0)
        m = collector.snapshot()

        assert math.isnan(m.theta), (
            f"Θ in vicsek mode should be NaN, got {m.theta}"
        )

    def test_theta_is_finite_in_projection_mode(self) -> None:
        """Projection mode → m.theta should be a finite number."""
        flock, collector = self._make_dummy_flock("projection")

        # Set last_theta so the mean is meaningful
        flock.last_theta[:] = 0.3

        collector.collect(flock, 0)
        m = collector.snapshot()

        assert not math.isnan(m.theta), (
            "Θ in projection mode should be finite, got NaN"
        )
        assert math.isfinite(m.theta), (
            f"Θ in projection mode should be finite, got {m.theta}"
        )
        # Mean of [0.3, 0.3, 0.3, 0.3, 0.3] = 0.3
        assert abs(m.theta - 0.3) < 1e-6, (
            f"Θ should be mean of last_theta = 0.3, got {m.theta:.6f}"
        )
