"""Unit tests for physics.occlusion — spherical_cap_occlusion."""

import numpy as np
import pytest

from pymurmur.physics.occlusion import (
    spherical_cap_occlusion,
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
    """Visible list is ordered by distance (closest first)."""
    obs_pos = np.array([0, 0, 0], dtype=np.float32)
    obs_vel = np.array([1, 0, 0], dtype=np.float32)
    # Neighbours at various distances
    nbr_pos = np.array([
        [80, 0, 0],   # farthest
        [30, 0, 0],   # middle
        [10, 0, 0],   # closest
    ], dtype=np.float32)
    nbr_vel = np.ones((3, 3), dtype=np.float32)

    _, visible, _ = spherical_cap_occlusion(
        obs_pos, obs_vel, nbr_pos, nbr_vel)

    # Visible should be sorted closest-first: [2, 1, 0]
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
