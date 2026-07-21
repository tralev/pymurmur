"""I1.7 — Property tests for each Level 0 force primitive.

Fuzzy-tests separation_force, alignment_force, cohesion_force, noise_force
for documented formula correctness, output shape, bound enforcement,
rng determinism, and vectorised/ragged path equivalence.

Uses hypothesis-style assertions: generate random inputs, assert invariants.
"""

from typing import Optional

import numpy as np
import pytest

from pymurmur.physics.forces._base import (
    _is_ragged,
    alignment_force,
    cohesion_force,
    noise_force,
    separation_force,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rng():
    """Fresh seeded RNG — avoids shared state across tests."""
    return np.random.default_rng(42)


def _make_dense_and_ragged(
    positions: np.ndarray, k: int, rng: Optional[np.random.Generator] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense_idx, ragged_idx) with the same k random neighbours per bird.

    k is clamped to N-1 so every bird always has enough distinct neighbours;
    no -1 sentinel padding is needed.
    """
    rng = rng or _make_rng()
    N = len(positions)
    k = min(k, N - 1)
    dense = np.zeros((N, k), dtype=np.int32)
    ragged = np.empty(N, dtype=object)
    for i in range(N):
        candidates = [j for j in range(N) if j != i]
        if len(candidates) > k:
            chosen = rng.choice(candidates, size=k, replace=False)
        else:
            chosen = candidates
        for j, nbr in enumerate(chosen):
            dense[i, j] = nbr
        ragged[i] = np.array(chosen, dtype=np.int32)
    return dense, ragged


# ===================================================================
# separation_force
# ===================================================================


class TestSeparationForceProperties:
    """Property tests for separation_force."""

    def test_output_shape_and_dtype(self):
        """Returns (N, 3) float32."""
        N = 10
        pos = np.random.default_rng(1).normal(size=(N, 3)).astype(np.float32)
        vel = np.random.default_rng(2).normal(size=(N, 3)).astype(np.float32)
        active = np.ones(N, dtype=bool)
        dense, _ = _make_dense_and_ragged(pos, k=3)
        f = separation_force(pos, vel, dense, active)
        assert f.shape == (N, 3)
        assert f.dtype == np.float32

    def test_no_neighbours_is_zero(self):
        """All birds have empty neighbour lists → force is zero."""
        rng = _make_rng()
        N = 5
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.ones(N, dtype=bool)
        empty = np.zeros((N, 0), dtype=np.int32)
        f = separation_force(pos, vel, empty, active)
        assert np.allclose(f, 0.0)

    def test_all_inactive_is_zero(self):
        """No active birds → force is all zero."""
        rng = _make_rng()
        N = 5
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.zeros(N, dtype=bool)
        dense, _ = _make_dense_and_ragged(pos, k=2, rng=rng)
        f = separation_force(pos, vel, dense, active)
        assert np.allclose(f, 0.0)

    def test_inactive_rows_are_zero(self):
        """Inactive birds get zero force in their slots."""
        rng = _make_rng()
        N = 10
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.array([True] * 6 + [False] * 4)
        dense, _ = _make_dense_and_ragged(pos, k=3, rng=rng)
        f = separation_force(pos, vel, dense, active)
        assert np.allclose(f[~active], 0.0)

    def test_no_nan_or_inf(self):
        """Output never contains NaN or inf for random inputs."""
        rng = _make_rng()
        for _ in range(20):
            N = rng.integers(4, 30)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            dense, ragged = _make_dense_and_ragged(pos, k=rng.integers(1, 5), rng=rng)
            for idx in (dense, ragged):
                f = separation_force(pos, vel, idx, active)
                assert np.all(np.isfinite(f)), (
                    f"NaN/inf in output with "
                    f"{'ragged' if _is_ragged(idx) else 'dense'} idx"
                )

    def test_force_points_away_from_neighbour(self):
        """For a single neighbour, force on bird i points away from j.

        F_sep[i] = −d̂_ij / |d_ij|², where d_ij = p_j − p_i.
        So −d_ij points from j toward i (away from j). ✓
        """
        rng = _make_rng()
        N = 6
        pos = rng.normal(scale=50, size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.ones(N, dtype=bool)
        # Each bird has exactly one distinct neighbour
        dense = np.zeros((N, 1), dtype=np.int32)
        for i in range(N):
            dense[i, 0] = (i + 1) % N
        f = separation_force(pos, vel, dense, active)
        for i in range(N):
            if np.allclose(f[i], 0.0):
                continue
            j = dense[i, 0]
            d_ij = pos[j] - pos[i]  # vector from i to j
            # Force should point opposite to d_ij (away from j)
            if np.linalg.norm(d_ij) > 1e-6:
                dot = np.dot(f[i], d_ij)
                assert dot <= 0, (
                    f"Bird {i}: force dot d_ij = {dot:.4f}, "
                    f"force should point away from neighbour {j}"
                )

    def test_magnitude_decreases_with_distance(self):
        """S1.5: |F_sep| = 1/d² for a single neighbour (Σ r̂/d²).

        With one neighbour at distance d, unit direction toward bird 1
        gives r̂ = (1,0,0), r̂/d² = (1/d², 0, 0). Magnitude = 1/d².
        """
        pos = np.zeros((2, 3), dtype=np.float32)
        vel = np.zeros((2, 3), dtype=np.float32)
        active = np.ones(2, dtype=bool)
        idx = np.array([[1], [0]], dtype=np.int32)

        for dist in [1.0, 2.0, 5.0, 10.0]:
            pos[1] = [dist, 0.0, 0.0]
            f = separation_force(pos, vel, idx, active)
            # S1.5: Σ r̂/d² = 1/d² for a single neighbour
            expected_mag = 1.0 / (dist ** 2)
            actual_mag = np.linalg.norm(f[0])
            assert np.isclose(actual_mag, expected_mag, rtol=1e-5), (
                f"d={dist}: expected |F|=1/d²={expected_mag:.6f}, got {actual_mag:.6f}"
            )

    def test_vectorised_and_ragged_equivalent(self):
        """Dense and ragged paths produce identical output for the same neighbours."""
        rng = _make_rng()
        for _ in range(20):
            N = rng.integers(4, 20)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            k = rng.integers(1, min(5, N))
            dense, ragged = _make_dense_and_ragged(pos, k=k, rng=rng)
            f_dense = separation_force(pos, vel, dense, active)
            f_ragged = separation_force(pos, vel, ragged, active)
            assert np.allclose(f_dense, f_ragged, atol=1e-5), (
                f"Max diff: {np.abs(f_dense - f_ragged).max():.6f}"
            )

    def test_zero_distance_handled(self):
        """Co-located birds don't crash; zero-distance pairs are skipped."""
        pos = np.array([[50, 0, 0], [50, 0, 0], [100, 0, 0]], dtype=np.float32)
        vel = np.zeros((3, 3), dtype=np.float32)
        active = np.ones(3, dtype=bool)
        idx = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        f = separation_force(pos, vel, idx, active)
        assert np.all(np.isfinite(f))
        # Birds 0 and 1 are colocated — neither's force contribution from the
        # other should be NaN or inf
        assert not np.any(np.isnan(f))
        assert not np.any(np.isinf(f))

    def test_zero_distance_ragged(self):
        """Ragged path also handles colocated birds without NaN/inf."""
        pos = np.array([[50, 0, 0], [50, 0, 0], [100, 0, 0]], dtype=np.float32)
        vel = np.zeros((3, 3), dtype=np.float32)
        active = np.ones(3, dtype=bool)
        # Ragged: each bird's neighbour list is an object array
        ragged = np.empty(3, dtype=object)
        ragged[0] = np.array([1, 2], dtype=np.int32)
        ragged[1] = np.array([0, 2], dtype=np.int32)
        ragged[2] = np.array([0, 1], dtype=np.int32)
        f = separation_force(pos, vel, ragged, active)
        assert np.all(np.isfinite(f))
        assert not np.any(np.isnan(f))
        assert not np.any(np.isinf(f))


# ===================================================================
# alignment_force
# ===================================================================


class TestAlignmentForceProperties:
    """Property tests for alignment_force."""

    def test_output_shape_and_dtype(self):
        """Returns (N, 3) float32."""
        N = 10
        pos = np.random.default_rng(3).normal(size=(N, 3)).astype(np.float32)
        vel = np.random.default_rng(4).normal(size=(N, 3)).astype(np.float32)
        active = np.ones(N, dtype=bool)
        dense, _ = _make_dense_and_ragged(pos, k=3)
        f = alignment_force(pos, vel, dense, active)
        assert f.shape == (N, 3)
        assert f.dtype == np.float32

    def test_no_neighbours_is_zero(self):
        """Birds with no neighbours get zero alignment force."""
        rng = _make_rng()
        N = 5
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.ones(N, dtype=bool)
        empty = np.zeros((N, 0), dtype=np.int32)
        f = alignment_force(pos, vel, empty, active)
        assert np.allclose(f, 0.0)

    def test_all_inactive_is_zero(self):
        """No active birds → force is all zero."""
        rng = _make_rng()
        N = 5
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.zeros(N, dtype=bool)
        dense, _ = _make_dense_and_ragged(pos, k=2, rng=rng)
        f = alignment_force(pos, vel, dense, active)
        assert np.allclose(f, 0.0)

    def test_inactive_rows_are_zero(self):
        """Inactive birds get zero force in their slots."""
        rng = _make_rng()
        N = 10
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.array([True] * 6 + [False] * 4)
        dense, _ = _make_dense_and_ragged(pos, k=3, rng=rng)
        f = alignment_force(pos, vel, dense, active)
        assert np.allclose(f[~active], 0.0)

    def test_magnitude_bounded(self):
        """|F_align| ≤ 2 (difference of two unit vectors)."""
        rng = _make_rng()
        for _ in range(30):
            N = rng.integers(2, 15)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            dense, ragged = _make_dense_and_ragged(
                pos, k=rng.integers(1, min(5, N)), rng=rng
            )
            for idx in (dense, ragged):
                f = alignment_force(pos, vel, idx, active)
                norms = np.linalg.norm(f[active], axis=1)
                assert np.all(norms <= 2.0 + 1e-5), (
                    f"Max norm {norms.max():.4f} exceeds bound 2.0"
                )

    def test_identical_velocities_is_zero(self):
        """All neighbours share the bird's velocity → force ≈ 0."""
        rng = _make_rng()
        pos = rng.normal(scale=50, size=(4, 3)).astype(np.float32)
        v = rng.normal(size=(3,)).astype(np.float32)
        vel = np.tile(v, (4, 1))
        active = np.ones(4, dtype=bool)
        dense = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32)
        f = alignment_force(pos, vel, dense, active)
        assert np.allclose(f[active], 0.0, atol=1e-5)

    def test_opposite_velocities_gives_max(self):
        """S1.5: Reynolds steering — normalize(v̄ − v_i).

        v̄=(−1,0,0), v_i=(1,0,0) → v̄−v_i=(−2,0,0) → normalize → (−1,0,0).
        """
        pos = np.array([[0, 0, 0], [5, 0, 0], [6, 0, 0]], dtype=np.float32)
        vel = np.array([[1, 0, 0], [-1, 0, 0], [-1, 0, 0]], dtype=np.float32)
        active = np.ones(3, dtype=bool)
        idx = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        f = alignment_force(pos, vel, idx, active)
        # S1.5: normalize(v̄ − v_i) = normalize((−1,0,0) − (1,0,0))
        #   = normalize((−2,0,0)) = (−1,0,0)
        assert np.allclose(f[0], [-1.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(np.linalg.norm(f[0]), 1.0, atol=1e-5)

    def test_no_nan_or_inf(self):
        """Output never contains NaN or inf for random inputs."""
        rng = _make_rng()
        for _ in range(20):
            N = rng.integers(4, 30)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            dense, ragged = _make_dense_and_ragged(
                pos, k=rng.integers(1, 5), rng=rng
            )
            for idx in (dense, ragged):
                f = alignment_force(pos, vel, idx, active)
                assert np.all(np.isfinite(f))

    def test_vectorised_and_ragged_equivalent(self):
        """Dense and ragged paths produce identical output."""
        rng = _make_rng()
        for _ in range(20):
            N = rng.integers(4, 20)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            k = rng.integers(1, min(5, N))
            dense, ragged = _make_dense_and_ragged(pos, k=k, rng=rng)
            f_dense = alignment_force(pos, vel, dense, active)
            f_ragged = alignment_force(pos, vel, ragged, active)
            assert np.allclose(f_dense, f_ragged, atol=1e-5), (
                f"Max diff: {np.abs(f_dense - f_ragged).max():.6f}"
            )

    def test_zero_velocity_handled(self):
        """Birds with zero velocity are handled gracefully (no div-by-zero)."""
        rng = _make_rng()
        pos = rng.normal(scale=50, size=(4, 3)).astype(np.float32)
        vel = np.zeros((4, 3), dtype=np.float32)
        active = np.ones(4, dtype=bool)
        dense = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32)
        f = alignment_force(pos, vel, dense, active)
        assert np.all(np.isfinite(f))
        # All velocities zero → average is zero → no valid heading → force remains 0
        assert np.allclose(f, 0.0)


# ===================================================================
# cohesion_force
# ===================================================================


class TestCohesionForceProperties:
    """Property tests for cohesion_force."""

    def test_output_shape_and_dtype(self):
        """Returns (N, 3) float32."""
        N = 10
        pos = np.random.default_rng(5).normal(size=(N, 3)).astype(np.float32)
        vel = np.random.default_rng(6).normal(size=(N, 3)).astype(np.float32)
        active = np.ones(N, dtype=bool)
        dense, _ = _make_dense_and_ragged(pos, k=3)
        f = cohesion_force(pos, vel, dense, active)
        assert f.shape == (N, 3)
        assert f.dtype == np.float32

    def test_no_neighbours_is_zero(self):
        """Birds with no neighbours get zero cohesion force."""
        rng = _make_rng()
        N = 5
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.ones(N, dtype=bool)
        empty = np.zeros((N, 0), dtype=np.int32)
        f = cohesion_force(pos, vel, empty, active)
        assert np.allclose(f, 0.0)

    def test_all_inactive_is_zero(self):
        """No active birds → force is all zero."""
        rng = _make_rng()
        N = 5
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.zeros(N, dtype=bool)
        dense, _ = _make_dense_and_ragged(pos, k=2, rng=rng)
        f = cohesion_force(pos, vel, dense, active)
        assert np.allclose(f, 0.0)

    def test_inactive_rows_are_zero(self):
        """Inactive birds get zero force in their slots."""
        rng = _make_rng()
        N = 10
        pos = rng.normal(size=(N, 3)).astype(np.float32)
        vel = rng.normal(size=(N, 3)).astype(np.float32)
        active = np.array([True] * 6 + [False] * 4)
        dense, _ = _make_dense_and_ragged(pos, k=3, rng=rng)
        f = cohesion_force(pos, vel, dense, active)
        assert np.allclose(f[~active], 0.0)

    def test_force_is_unit_vector(self):
        """S1.5: cohesion returns limit3(to_center, 1.0) — capped at unit.

        Magnitude ≤ 1.0 always.  Short centroids (< 1) pass through unscaled;
        long centroids (> 1) are normalized to unit length.
        """
        rng = _make_rng()
        for _ in range(30):
            N = rng.integers(2, 20)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            dense, ragged = _make_dense_and_ragged(
                pos, k=rng.integers(1, min(5, N)), rng=rng
            )
            for idx in (dense, ragged):
                f = cohesion_force(pos, vel, idx, active)
                norms = np.linalg.norm(f[active], axis=1)
                nonzero = norms > 1e-10
                if nonzero.any():
                    # S1.5: limit3 → magnitude ≤ 1.0
                    assert np.all(norms[nonzero] <= 1.0 + 1e-5), (
                        f"Norms exceed 1.0: {norms[nonzero]}"
                    )

    def test_force_points_toward_center_of_mass(self):
        """Force on bird i should have non-negative dot with (center − p_i)."""
        rng = _make_rng()
        for _ in range(20):
            N = rng.integers(3, 15)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            dense, ragged = _make_dense_and_ragged(
                pos, k=rng.integers(1, min(5, N)), rng=rng
            )
            for idx in (dense, ragged):
                f = cohesion_force(pos, vel, idx, active)
                for i in range(N):
                    if np.allclose(f[i], 0.0):
                        continue
                    nbrs = idx[i] if not _is_ragged(idx) else idx[i]
                    if len(nbrs) == 0:
                        continue
                    center = np.mean(pos[nbrs], axis=0)
                    to_center = center - pos[i]
                    if np.linalg.norm(to_center) > 1e-10:
                        dot = np.dot(f[i], to_center)
                        assert dot >= -1e-6, (
                            f"Bird {i}: force dot (center−p_i) = {dot:.4f} < 0; "
                            f"force should point toward center of mass"
                        )

    def test_single_neighbour_exact_direction(self):
        """With one neighbour, force is the exact unit vector toward it."""
        rng = _make_rng()
        for _ in range(20):
            pos = rng.normal(scale=100, size=(2, 3)).astype(np.float32)
            vel = rng.normal(size=(2, 3)).astype(np.float32)
            active = np.ones(2, dtype=bool)
            idx = np.array([[1], [0]], dtype=np.int32)
            f = cohesion_force(pos, vel, idx, active)
            d_01 = pos[1] - pos[0]
            expected = d_01 / np.linalg.norm(d_01)
            assert np.allclose(f[0], expected, atol=1e-5)

    def test_no_nan_or_inf(self):
        """Output never contains NaN or inf for random inputs."""
        rng = _make_rng()
        for _ in range(20):
            N = rng.integers(4, 30)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            dense, ragged = _make_dense_and_ragged(
                pos, k=rng.integers(1, 5), rng=rng
            )
            for idx in (dense, ragged):
                f = cohesion_force(pos, vel, idx, active)
                assert np.all(np.isfinite(f))

    def test_vectorised_and_ragged_equivalent(self):
        """Dense and ragged paths produce identical output."""
        rng = _make_rng()
        for _ in range(20):
            N = rng.integers(4, 20)
            pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
            vel = rng.normal(size=(N, 3)).astype(np.float32)
            active = np.ones(N, dtype=bool)
            k = rng.integers(1, min(5, N))
            dense, ragged = _make_dense_and_ragged(pos, k=k, rng=rng)
            f_dense = cohesion_force(pos, vel, dense, active)
            f_ragged = cohesion_force(pos, vel, ragged, active)
            assert np.allclose(f_dense, f_ragged, atol=1e-5), (
                f"Max diff: {np.abs(f_dense - f_ragged).max():.6f}"
            )


# ===================================================================
# noise_force
# ===================================================================


class TestNoiseForceProperties:
    """Property tests for noise_force."""

    def test_output_shape_and_dtype(self):
        """Returns (N, 3) float32."""
        for n in [0, 1, 10, 100]:
            f = noise_force(n, 0.5)
            assert f.shape == (n, 3)
            assert f.dtype == np.float32

    def test_zero_scale_is_zero(self):
        """scale=0 produces all-zero array."""
        for n in [0, 5, 50]:
            f = noise_force(n, 0.0)
            assert np.allclose(f, 0.0)

    def test_noise_magnitude_equals_scale(self):
        """D9: noise_force(N, scale) produces vectors with mean magnitude ≈ scale."""
        rng = np.random.default_rng(42)
        for scale in [0.1, 0.5, 2.0, 5.0]:
            f = noise_force(10000, scale, rng)
            mags = np.linalg.norm(f, axis=1)
            # With 10⁴ draws, mean magnitude should converge to scale
            assert np.mean(mags) == pytest.approx(scale, rel=0.03), (
                f"scale={scale}: expected mean mag ≈ {scale}, got {np.mean(mags):.4f}"
            )

    def test_unit_norm_at_scale_one(self):
        """scale=1.0 still produces unit vectors (backward compatible)."""
        f = noise_force(500, 1.0)
        norms = np.linalg.norm(f, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_with_same_seed(self):
        """Same seed → bit-identical output."""
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        f1 = noise_force(100, 1.0, rng1)
        f2 = noise_force(100, 1.0, rng2)
        assert np.array_equal(f1, f2)

    def test_different_with_different_seeds(self):
        """Different seeds → different output (probabilistic, but very likely)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        f1 = noise_force(200, 1.0, rng1)
        f2 = noise_force(200, 1.0, rng2)
        assert not np.allclose(f1, f2)

    def test_default_rng_works(self):
        """Calling without rng uses default generator."""
        f = noise_force(50, 0.5)
        assert f.shape == (50, 3)
        assert not np.allclose(f, 0.0)

    def test_no_nan_or_inf(self):
        """Output never contains NaN or inf."""
        rng = _make_rng()
        for _ in range(10):
            f = noise_force(rng.integers(1, 200), rng.uniform(0.1, 5.0))
            assert np.all(np.isfinite(f))

    def test_n_zero(self):
        """n=0 returns shape (0, 3) array."""
        f = noise_force(0, 1.0)
        assert f.shape == (0, 3)
        assert f.dtype == np.float32

    def test_uniform_on_sphere(self):
        """For large N, mean direction should be near zero (uniform on sphere)."""
        N = 10000
        f = noise_force(N, 1.0, np.random.default_rng(0))
        mean = np.mean(f, axis=0)
        # Expected |mean| ≈ 0; with N=10000, should be very small
        assert np.linalg.norm(mean) < 0.05, (
            f"Mean direction norm = {np.linalg.norm(mean):.4f}, expected near 0"
        )
