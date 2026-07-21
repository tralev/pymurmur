"""Unit tests for physics.occlusion — spherical_cap_occlusion."""

import numpy as np
import pytest

from pymurmur.physics.occlusion import (
    _occlusion_culling_chunk,
    spherical_cap_occlusion,
    spherical_cap_occlusion_batched,
    spherical_cap_occlusion_soa,
)


def test_occlusion_no_neighbors():
    """Empty neighbour list returns zero delta, empty visible, theta=0."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    nbr_pos = np.zeros((0, 3), dtype=np.float32)
    nbr_vel = np.zeros((0, 3), dtype=np.float32)

    delta, visible, theta = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    assert np.allclose(delta, [0, 0, 0])
    assert len(visible) == 0
    assert theta == 0.0


def test_occlusion_single_neighbor():
    """One neighbour: visible includes it, theta > 0."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    nbr_pos = np.array([[50, 0, 0]], dtype=np.float32)
    nbr_vel = np.array([[1, 0, 0]], dtype=np.float32)

    delta, visible, theta = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    assert len(visible) == 1
    assert theta > 0.0


def test_occlusion_delta_magnitude():
    """|delta| ∈ [0, 1]."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    nbr_pos = np.random.uniform(-100, 100, (20, 3)).astype(np.float32)
    nbr_vel = np.random.uniform(-1, 1, (20, 3)).astype(np.float32)

    delta, _, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    mag = np.linalg.norm(delta)
    assert 0.0 <= mag <= 1.0


def test_occlusion_theta_range():
    """theta ∈ [0, 1]."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    nbr_pos = np.random.uniform(-100, 100, (20, 3)).astype(np.float32)
    nbr_vel = np.random.uniform(-1, 1, (20, 3)).astype(np.float32)

    _, _, theta = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    assert 0.0 <= theta <= 1.0


def test_occlusion_soa_adapter():
    """SoA adapter returns same shape results."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    nbr_pos = np.array([[50, 0, 0]], dtype=np.float32)
    nbr_vel = np.array([[1, 0, 0]], dtype=np.float32)

    delta, visible, theta = spherical_cap_occlusion_soa(
        obs_pos, obs_vel, nbr_pos, nbr_vel)

    assert delta.shape == (3,)
    assert theta >= 0.0


# ── Edge case and refinement tests ────────────────────────────────


def test_occlusion_self_skip():
    """Neighbour at identical position is skipped (d < 1e-6 guard)."""
    obs_pos = np.array([100, 100, 100], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # One neighbour at exactly the same position
    nbr_pos = np.array([[100, 100, 100], [150, 100, 100]], dtype=np.float32)
    nbr_vel = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    delta, visible, theta = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    # Self-skip: bird at identical position should not be in visible
    # visible only contains index 1 (the far bird)
    assert len(visible) == 1
    assert visible[0] == 1


def test_occlusion_blind_angle_excludes():
    """Neighbour directly behind observer is excluded with blind angle."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)  # facing +x
    # Neighbour at (-50, 0, 0) — directly behind
    nbr_pos = np.array([[-50, 0, 0]], dtype=np.float32)
    nbr_vel = np.array([[0, 0, 0]], dtype=np.float32)
    # blind_cos = cos(half_angle). half_angle=45° → cos45≈0.707
    blind_cos = np.cos(np.radians(45.0))

    _, visible, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel, blind_cos=blind_cos)
    # Direction to neighbour is (-1,0,0). -obs_forward is (-1,0,0).
    # dot(direction, -obs_forward) = dot((-1,0,0), (-1,0,0)) = 1.0
    # 1.0 >= blind_cos=0.707 → excluded
    assert len(visible) == 0


def test_occlusion_blind_angle_forward_visible():
    """Neighbour in front of observer is still visible with blind angle."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)  # facing +x
    # Neighbour at (50, 0, 0) — directly ahead
    nbr_pos = np.array([[50, 0, 0]], dtype=np.float32)
    nbr_vel = np.array([[0, 0, 0]], dtype=np.float32)
    blind_cos = np.cos(np.radians(45.0))  # half_angle=45°

    _, visible, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel, blind_cos=blind_cos)
    # Direction is (1,0,0), -obs_forward=(-1,0,0). dot=-1.0 < 0.707 → visible
    assert len(visible) == 1


def test_occlusion_anisotropic_body():
    """Anisotropic bodies produce different cap sizes based on viewing angle."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # Neighbour at (50, 0, 50) viewed at 45° from its forward direction
    nbr_pos = np.array([[50, 0, 50]], dtype=np.float32)
    nbr_vel = np.array([[0, 0, 1]], dtype=np.float32)  # neighbour facing +z

    # Isotropic (anisotropy=1.0)
    _, _, theta_iso = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel, anisotropy=1.0)
    # Anisotropic (anisotropy=3.0 — body longer than wide)
    _, _, theta_aniso = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel, anisotropy=3.0)

    # Theta should differ because effective radius changes with viewing angle
    assert theta_iso != theta_aniso
    assert 0.0 <= theta_iso <= 1.0
    assert 0.0 <= theta_aniso <= 1.0


def test_occlusion_interior_bird():
    """Bird surrounded by many neighbours → theta → 1, |delta| → 0."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # Create neighbours in all directions (sphere of radius 20)
    n = 30
    nbr_pos = np.random.default_rng(42).normal(size=(n, 3)).astype(np.float32)
    nbr_pos = nbr_pos / np.linalg.norm(nbr_pos, axis=1, keepdims=True) * 20
    nbr_vel = np.random.default_rng(42).normal(size=(n, 3)).astype(np.float32)

    delta, _, theta = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel, boid_size=9.0)

    # Surrounded bird should have high opacity
    # cap_radius = boid_size/d ≈ 9/20 = 0.45 per bird
    # 30 birds × 0.45 = 13.5, but clamped to 1.0
    assert theta >= 0.5  # should be high
    assert theta <= 1.0


def test_occlusion_edge_bird():
    """Bird at edge of flock → |delta| ≈ 1."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # All neighbours on one side (+x direction)
    nbr_pos = np.array([
        [20, 0, 0], [25, 5, 0], [22, -5, 0],
        [18, 0, 5], [21, 0, -5],
    ], dtype=np.float32)
    nbr_vel = np.ones((5, 3), dtype=np.float32)

    delta, _, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)

    # Edge bird — delta should have non-trivial magnitude
    delta_mag = np.linalg.norm(delta)
    assert delta_mag > 0.0
    assert delta_mag <= 1.0
    # Delta should point generally toward the neighbours (+x)
    assert delta[0] > 0


def test_occlusion_closest_first():
    """Visible list is ordered by distance (closest first).

    P1.1: With true occlusion culling, collinear birds beyond the closest
    are occluded. Use non-collinear positions to verify closest-first ordering.
    """
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # Non-collinear neighbours at various distances — all separated so
    # no occlusion between them. Closest visible first.
    nbr_pos = np.array([
        [80, 10, 0],   # farthest — offset in y
        [30, -10, 0],  # middle — offset in y (opposite side)
        [10, 0, 10],   # closest — offset in z
    ], dtype=np.float32)
    nbr_vel = np.ones((3, 3), dtype=np.float32)

    _, visible, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)

    # P1.1: With true occlusion and separated positions, all 3 should be visible.
    # Order should be closest-first: [2, 1, 0]
    assert len(visible) == 3, f"Expected 3 visible with separated birds, got {len(visible)}"
    assert list(visible) == [2, 1, 0]


def test_occlusion_delta_normalized():
    """|delta| is clamped to ≤ 1 when it would exceed 1."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # Very close neighbours → large cap radii → potentially large delta
    nbr_pos = np.array([
        [1, 0, 0], [2, 0, 0], [3, 0, 0],
        [0, 1, 0], [0, 2, 0],
    ], dtype=np.float32)
    nbr_vel = np.ones((5, 3), dtype=np.float32)

    delta, _, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel, boid_size=20.0)

    mag = np.linalg.norm(delta)
    assert mag <= 1.0 + 1e-6


def test_occlusion_soa_empty():
    """SoA adapter with empty neighbours returns zeros."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    nbr_pos = np.zeros((0, 3), dtype=np.float32)
    nbr_vel = np.zeros((0, 3), dtype=np.float32)

    delta, visible, theta = spherical_cap_occlusion_soa(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    assert np.allclose(delta, [0, 0, 0])
    assert len(visible) == 0
    assert theta == 0.0


def test_occlusion_soa_bit_identical():
    """SoA adapter produces identical results to original for same input."""
    obs_pos = np.array([10, 20, 30], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    rng = np.random.default_rng(123)
    nbr_pos = rng.normal(0, 30, (10, 3)).astype(np.float32) + obs_pos
    nbr_vel = rng.normal(0, 1, (10, 3)).astype(np.float32)

    d1, v1, t1 = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    d2, v2, t2 = spherical_cap_occlusion_soa(
        obs_pos, obs_vel, nbr_pos, nbr_vel)

    # SoA adapter re-sorts arrays, so visible indices may differ
    # but delta and theta must be identical
    assert np.allclose(d1, d2)
    assert t1 == pytest.approx(t2)


def test_occlusion_marginal_opacity():
    """With 150 birds and Pearce defaults, Θ emerges in expected range."""
    rng = np.random.default_rng(42)
    obs_pos = np.array([500, 350, 200], dtype=np.float32)
    obs_vel = np.array([4, 0, 0], dtype=np.float32)
    # 150 birds spread in domain
    nbr_pos = rng.uniform(0, 1000, (150, 3)).astype(np.float32)
    nbr_vel = rng.normal(0, 1, (150, 3)).astype(np.float32)

    _, _, theta = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel, boid_size=9.0)

    # Marginal opacity should be in a reasonable range
    # With birds spread across full domain, occlusion should be partial
    assert 0.01 <= theta <= 1.0


def test_occlusion_self_excluded():
    """Observer's own position in neighbour list is skipped via d < 1e-6 guard."""
    obs_pos = np.array([500, 350, 200], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # Neighbour list includes observer's exact position
    nbr_pos = np.array([
        [500, 350, 200],  # self — should be skipped
        [550, 350, 200],  # real neighbour
    ], dtype=np.float32)
    nbr_vel = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    _, visible, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)
    # Self should not appear in visible list
    assert 0 not in visible
    assert len(visible) == 1
    assert visible[0] == 1


# ===================================================================
# I1.3 — Batched occlusion tests
# ===================================================================


def test_batched_output_shapes():
    """spherical_cap_occlusion_batched returns correct shapes."""
    N, K = 3, 4
    obs_pos = np.random.default_rng(0).normal(size=(N, 3)).astype(np.float32)
    obs_vel = np.random.default_rng(1).normal(size=(N, 3)).astype(np.float32)
    nbr_pos = np.random.default_rng(2).normal(size=(N, K, 3)).astype(np.float32)
    nbr_vel = np.random.default_rng(3).normal(size=(N, K, 3)).astype(np.float32)

    delta, visible_mask, theta = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel,
    )
    assert delta.shape == (N, 3)
    assert delta.dtype == np.float32
    assert visible_mask.shape == (N, K)
    assert visible_mask.dtype == bool
    assert theta.shape == (N,)
    assert theta.dtype == np.float32


def test_batched_empty_inputs():
    """Zero observers or zero neighbours → correct zero-filled outputs."""
    # N=0
    d, v, t = spherical_cap_occlusion_batched(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.float32),
    )
    assert d.shape == (0, 3)
    assert v.shape == (0, 4)
    assert t.shape == (0,)

    # K=0
    d, v, t = spherical_cap_occlusion_batched(
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 0, 3), dtype=np.float32),
        np.zeros((2, 0, 3), dtype=np.float32),
    )
    assert d.shape == (2, 3)
    assert v.shape == (2, 0)
    assert t.shape == (2,)
    assert np.allclose(d, 0.0)
    assert np.allclose(t, 0.0)


def test_batched_single_observer_equivalent():
    """Batched with N=1 produces identical delta/theta to single-observer."""
    rng = np.random.default_rng(42)
    K = 10
    obs_pos = rng.normal(scale=50, size=(3,)).astype(np.float32)
    obs_vel = rng.normal(size=(3,)).astype(np.float32)
    nbr_pos = rng.normal(scale=100, size=(K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(K, 3)).astype(np.float32)

    # Single-observer
    d1, v1, t1 = spherical_cap_occlusion(obs_pos, obs_vel, nbr_pos, nbr_vel)

    # Batched with N=1
    d2, v2, t2 = spherical_cap_occlusion_batched(
        obs_pos[np.newaxis, :],
        obs_vel[np.newaxis, :],
        nbr_pos[np.newaxis, :, :],
        nbr_vel[np.newaxis, :, :],
    )

    assert np.allclose(d1, d2[0], atol=1e-5)
    assert t1 == pytest.approx(float(t2[0]), abs=1e-5)
    # Visible sets match (batched returns bool mask, single returns indices)
    batched_visible = set(np.where(v2[0])[0])
    assert batched_visible == set(v1)


def test_batched_valid_mask_excludes_sentinels():
    """Neighbours marked invalid in valid_mask are never visible."""
    N, K = 2, 3
    obs_pos = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
    obs_vel = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    # All neighbours close to observer 0, far from observer 1
    nbr_pos = np.array([
        [[20, 0, 0], [30, 0, 0], [40, 0, 0]],
        [[20, 0, 0], [30, 0, 0], [40, 0, 0]],
    ], dtype=np.float32)
    nbr_vel = np.ones((N, K, 3), dtype=np.float32)

    # Mask out all neighbours for observer 1
    valid_mask = np.array([[True, True, True], [False, False, False]])

    _, visible_mask, _ = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, valid_mask=valid_mask,
    )

    # Observer 0: neighbours should be visible (close, forward)
    assert visible_mask[0].any()
    # Observer 1: all neighbours masked out → none visible
    assert not visible_mask[1].any()


def test_batched_partial_valid_mask():
    """Partially masked neighbours: only valid ones can become visible."""
    _N, K = 1, 3
    obs_pos = np.array([[0, 0, 0]], dtype=np.float32)
    obs_vel = np.array([[1, 0, 0]], dtype=np.float32)
    # Close neighbours, all in front
    nbr_pos = np.array([[[10, 0, 0], [15, 0, 0], [20, 0, 0]]], dtype=np.float32)
    nbr_vel = np.ones((1, K, 3), dtype=np.float32)

    # Only neighbour 0 and 2 are valid
    valid_mask = np.array([[True, False, True]])

    _, visible_mask, _ = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, valid_mask=valid_mask,
    )

    # Neighbour 1 (masked) must not be visible
    assert not visible_mask[0, 1]
    # Neighbour 0 should be visible (closest, in front)
    assert visible_mask[0, 0]


def test_batched_no_nan_or_inf():
    """Batched output never contains NaN or inf."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        N = rng.integers(2, 10)
        K = rng.integers(2, 8)
        obs_pos = rng.normal(scale=200, size=(N, 3)).astype(np.float32)
        obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
        nbr_pos = rng.normal(scale=200, size=(N, K, 3)).astype(np.float32)
        nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

        d, v, t = spherical_cap_occlusion_batched(
            obs_pos, obs_vel, nbr_pos, nbr_vel,
        )
        assert np.all(np.isfinite(d))
        assert np.all(np.isfinite(t))


def test_batched_delta_and_theta_bounds():
    """Batched delta in [0,1], theta in [0,1]."""
    rng = np.random.default_rng(12)
    N, K = 10, 20
    obs_pos = rng.normal(scale=100, size=(N, 3)).astype(np.float32)
    obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
    nbr_pos = rng.normal(scale=100, size=(N, K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

    delta, _, theta = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel,
    )

    delta_mags = np.linalg.norm(delta, axis=1)
    assert np.all(delta_mags >= 0.0)
    assert np.all(delta_mags <= 1.0 + 1e-5)
    assert np.all(theta >= 0.0)
    assert np.all(theta <= 1.0)


def test_batched_blind_angle():
    """Batched occlusion respects blind angle for all observers."""
    N, K = 2, 2
    obs_pos = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    obs_vel = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    # Same neighbour set for both observers
    nbr_pos = np.array([
        [[50, 0, 0], [-50, 0, 0]],
        [[50, 0, 0], [-50, 0, 0]],
    ], dtype=np.float32)
    nbr_vel = np.ones((N, K, 3), dtype=np.float32)
    blind_cos = np.cos(np.radians(45.0))

    _, visible_mask, _ = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, blind_cos=blind_cos,
    )

    # Observer 0 faces +x: neighbour at +x visible, -x blind
    assert visible_mask[0, 0]   # (50, 0, 0) in front
    assert not visible_mask[0, 1]  # (-50, 0, 0) behind

    # Observer 1 faces -x: neighbour at -x visible, +x blind
    assert not visible_mask[1, 0]  # (50, 0, 0) behind
    assert visible_mask[1, 1]   # (-50, 0, 0) in front


def test_batched_anisotropy():
    """Batched anisotropy produces different thetas than isotropic."""
    rng = np.random.default_rng(99)
    N, K = 4, 10
    obs_pos = rng.normal(scale=50, size=(N, 3)).astype(np.float32)
    obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
    nbr_pos = rng.normal(scale=100, size=(N, K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

    _, _, t_iso = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, anisotropy=1.0,
    )
    _, _, t_aniso = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, anisotropy=3.0,
    )

    # At least some observers should differ
    assert not np.allclose(t_iso, t_aniso, atol=1e-6)
    assert np.all(t_iso >= 0.0) and np.all(t_iso <= 1.0)
    assert np.all(t_aniso >= 0.0) and np.all(t_aniso <= 1.0)


def test_batched_multiple_observers_independent():
    """Each observer's occlusion is independent — results differ per bird."""
    rng = np.random.default_rng(55)
    N, K = 5, 8
    obs_pos = rng.normal(scale=200, size=(N, 3)).astype(np.float32)
    obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
    nbr_pos = rng.normal(scale=200, size=(N, K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

    delta, visible_mask, theta = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel,
    )

    # Each observer should have different delta (not all identical)
    assert not np.allclose(delta[0], delta[1:], atol=1e-6)


# ===================================================================
# P4.6 — Parallel occlusion tests
# ===================================================================


def test_parallel_identical_to_sequential():
    """Parallel culling (n_jobs=2) produces bit-identical results to sequential."""
    rng = np.random.default_rng(42)
    # Use N >= min_parallel default (100) so parallel path actually activates
    N, K = 150, 64
    obs_pos = rng.normal(scale=200, size=(N, 3)).astype(np.float32)
    obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
    nbr_pos = rng.normal(scale=200, size=(N, K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

    # Sequential (n_jobs=1, default)
    d_seq, v_seq, t_seq = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, n_jobs=1,
    )

    # Parallel (n_jobs=2)
    d_par, v_par, t_par = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, n_jobs=2,
    )

    assert np.array_equal(d_seq, d_par), "delta differs between sequential and parallel"
    assert np.array_equal(v_seq, v_par), "visible_mask differs between sequential and parallel"
    assert np.array_equal(t_seq, t_par), "theta differs between sequential and parallel"


def test_parallel_auto_workers():
    """n_jobs=-1 uses all available cores and matches sequential."""
    rng = np.random.default_rng(77)
    N, K = 200, 32
    obs_pos = rng.normal(scale=200, size=(N, 3)).astype(np.float32)
    obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
    nbr_pos = rng.normal(scale=200, size=(N, K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

    d_seq, v_seq, t_seq = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, n_jobs=1,
    )
    d_par, v_par, t_par = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, n_jobs=-1,
    )

    assert np.array_equal(d_seq, d_par)
    assert np.array_equal(v_seq, v_par)
    assert np.array_equal(t_seq, t_par)


def test_parallel_below_threshold_stays_sequential():
    """Small N (< min_parallel) uses sequential path even with n_jobs=4."""
    rng = np.random.default_rng(99)
    N, K = 10, 8  # way below _MIN_PARALLEL_OBSERVERS=100
    obs_pos = rng.normal(scale=200, size=(N, 3)).astype(np.float32)
    obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
    nbr_pos = rng.normal(scale=200, size=(N, K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

    d_seq, v_seq, t_seq = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, n_jobs=1,
    )
    d_par, v_par, t_par = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, n_jobs=4, min_parallel=50,
    )

    # Should be identical (both use sequential path since N=10 < 50)
    assert np.array_equal(d_seq, d_par)
    assert np.array_equal(v_seq, v_par)
    assert np.array_equal(t_seq, t_par)


def test_parallel_chunk_worker_bit_identical():
    """_occlusion_culling_chunk produces identical results to the full batch for its slice."""
    rng = np.random.default_rng(33)
    N, K = 100, 16
    obs_pos = rng.normal(scale=200, size=(N, 3)).astype(np.float32)
    obs_vel = rng.normal(size=(N, 3)).astype(np.float32)
    nbr_pos = rng.normal(scale=200, size=(N, K, 3)).astype(np.float32)
    nbr_vel = rng.normal(size=(N, K, 3)).astype(np.float32)

    # Full batch
    d_full, v_full, t_full = spherical_cap_occlusion_batched(
        obs_pos, obs_vel, nbr_pos, nbr_vel, n_jobs=1,
    )

    # Manually compute the Stage 1 inputs for the first chunk (observers 0:50)
    diffs = nbr_pos - obs_pos[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=2)
    sort_order = np.argsort(dists, axis=1)[:, :64]
    M = sort_order.shape[1]
    gather_i = np.arange(N)[:, np.newaxis]
    sorted_dists = dists[gather_i, sort_order]
    sorted_diffs = diffs[gather_i, sort_order]
    dirs = sorted_diffs / (sorted_dists[:, :, np.newaxis] + 1e-10)
    obs_forward = obs_vel / (np.linalg.norm(obs_vel, axis=1, keepdims=True) + 1e-10)

    # Run chunk worker on first half
    half = N // 2
    d_chunk, v_chunk, t_chunk = _occlusion_culling_chunk(
        sorted_dists[:half], dirs[:half],
        np.full((half, M), 9.0, dtype=np.float32),
        obs_forward[:half], sort_order[:half],
        None, K,
    )

    # Compare against the full-batch output's first half
    assert np.array_equal(d_full[:half], d_chunk)
    assert np.array_equal(v_full[:half], v_chunk)
    assert np.array_equal(t_full[:half], t_chunk)
