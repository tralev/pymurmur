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
