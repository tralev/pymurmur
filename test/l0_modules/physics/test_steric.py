"""Unit tests for physics.steric — steric_force()."""

import numpy as np

from pymurmur.physics.steric import steric_force


def test_steric_zero_strength():
    """steric_force(strength=0) returns zero vector."""
    obs = np.array([0, 0, 0], dtype=np.float32)
    nbrs = np.array([[5, 0, 0]], dtype=np.float32)
    force = steric_force(obs, nbrs, strength=0.0)
    assert np.allclose(force, [0, 0, 0])


def test_steric_no_neighbors():
    """Empty neighbour list returns zero vector."""
    obs = np.array([0, 0, 0], dtype=np.float32)
    nbrs = np.zeros((0, 3), dtype=np.float32)
    force = steric_force(obs, nbrs, strength=10.0)
    assert np.allclose(force, [0, 0, 0])


def test_steric_direction_away():
    """Force points away from neighbour."""
    obs = np.array([0, 0, 0], dtype=np.float32)
    nbrs = np.array([[5, 0, 0]], dtype=np.float32)
    force = steric_force(obs, nbrs, strength=1.0)
    # Force should have negative x component (push away from +x neighbour)
    assert force[0] < 0


def test_steric_falls_with_distance():
    """Force magnitude decreases as neighbour distance increases."""
    obs = np.array([0, 0, 0], dtype=np.float32)
    f_near = steric_force(obs, np.array([[2, 0, 0]], dtype=np.float32), strength=1.0)
    f_far = steric_force(obs, np.array([[8, 0, 0]], dtype=np.float32), strength=1.0)
    assert np.linalg.norm(f_near) > np.linalg.norm(f_far)


def test_steric_close_range_only():
    """Neighbour at distance > threshold produces no force."""
    obs = np.array([0, 0, 0], dtype=np.float32)
    nbrs = np.array([[100, 0, 0]], dtype=np.float32)
    force = steric_force(obs, nbrs, strength=1.0, threshold=10.0)
    assert np.allclose(force, [0, 0, 0])


def test_steric_radius_scaling():
    """A wider threshold reaches farther neighbours, a narrower one doesn't."""
    obs = np.array([0, 0, 0], dtype=np.float32)
    nbrs = np.array([[30, 0, 0]], dtype=np.float32)
    narrow = steric_force(obs, nbrs, strength=1.0, threshold=10.0)
    wide = steric_force(obs, nbrs, strength=1.0, threshold=40.0)
    assert np.allclose(narrow, [0, 0, 0])
    assert not np.allclose(wide, [0, 0, 0])


def test_steric_radius_matches_boid_size_scaling_convention():
    """sim_new.md's STERIC_RADIUS = boid_size * 4.0 convention: a neighbour
    just inside that radius contributes, just outside does not."""
    boid_size = 9.0
    steric_radius = boid_size * 4.0  # 36.0
    obs = np.array([0, 0, 0], dtype=np.float32)
    just_inside = steric_force(
        obs, np.array([[steric_radius - 1.0, 0, 0]], dtype=np.float32),
        strength=1.0, threshold=steric_radius,
    )
    just_outside = steric_force(
        obs, np.array([[steric_radius + 1.0, 0, 0]], dtype=np.float32),
        strength=1.0, threshold=steric_radius,
    )
    assert not np.allclose(just_inside, [0, 0, 0])
    assert np.allclose(just_outside, [0, 0, 0])
