"""Unit tests for physics.forces._base — force primitives."""

import numpy as np

from pymurmur.physics.forces._base import (
    separation_force,
    alignment_force,
    cohesion_force,
    noise_force,
)


def test_separation_force_no_neighbors(known_positions, known_velocities, neighbor_idx):
    """Birds with no neighbours get zero separation force."""
    N = len(known_positions)
    active = np.ones(N, dtype=bool)
    empty_idx = np.zeros((N, 0), dtype=np.int32)
    force = separation_force(known_positions, known_velocities, empty_idx, active)
    assert np.allclose(force, 0.0)


def test_separation_force_direction(known_positions, known_velocities):
    """Force points away from neighbour."""
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = separation_force(known_positions, known_velocities, idx, active)
    # Bird 0 pushed away from bird 1 at (10, 0, 0) → negative x
    assert force[0, 0] < 0


def test_alignment_force_no_neighbors(known_positions, known_velocities, neighbor_idx):
    """Birds with no neighbours get zero alignment force."""
    N = len(known_positions)
    active = np.ones(N, dtype=bool)
    empty_idx = np.zeros((N, 0), dtype=np.int32)
    force = alignment_force(known_positions, known_velocities, empty_idx, active)
    assert np.allclose(force, 0.0)


def test_cohesion_force_toward_center(known_positions, known_velocities):
    """Force points toward the centre of mass of neighbours."""
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = cohesion_force(known_positions, known_velocities, idx, active)
    # Bird 0's lone neighbour at (10, 0, 0) → cohesion toward +x
    assert force[0, 0] > 0


def test_noise_force_shape():
    """noise_force(N, s) returns (N, 3) float32."""
    f = noise_force(50, 0.5)
    assert f.shape == (50, 3)
    assert f.dtype == np.float32


def test_noise_force_zero_scale():
    """scale=0 produces all-zero array."""
    f = noise_force(20, 0.0)
    assert np.allclose(f, 0.0)


def test_force_primitives_inactive_rows(known_positions, known_velocities, neighbor_idx):
    """All primitives return zero force for inactive birds."""
    N = len(known_positions)
    active = np.array([True, True, False, False])
    sep = separation_force(known_positions, known_velocities, neighbor_idx, active)
    assert np.allclose(sep[~active], 0.0)
    align = alignment_force(known_positions, known_velocities, neighbor_idx, active)
    assert np.allclose(align[~active], 0.0)
    coh = cohesion_force(known_positions, known_velocities, neighbor_idx, active)
    assert np.allclose(coh[~active], 0.0)


def test_separation_force_zero_distance():
    """Two birds at identical positions — handled gracefully (no div-by-zero crash)."""
    pos = np.array([[100, 100, 100], [100, 100, 100]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    active = np.ones(2, dtype=bool)
    force = separation_force(pos, vel, idx, active)
    # Zero-distance neighbours are skipped (1e-6 guard), so force is zero
    assert np.all(np.isfinite(force))
    assert not np.any(np.isnan(force))


def test_separation_force_falls_with_distance(known_positions, known_velocities):
    """Force magnitude decreases as neighbour distance increases."""
    # Bird 0 sees bird 1 at (10, 0, 0) — far
    idx_far = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force_far = separation_force(known_positions, known_velocities, idx_far, active)

    # Move bird 1 closer to bird 0
    pos_near = known_positions.copy()
    pos_near[1] = np.array([1, 0, 0], dtype=np.float32)
    idx_near = np.array([[1], [0], [3], [2]], dtype=np.int32)
    force_near = separation_force(pos_near, known_velocities, idx_near, active)

    # Force should be stronger at close range
    assert np.linalg.norm(force_near[0]) > np.linalg.norm(force_far[0])


def test_separation_force_inactive_ignored():
    """Inactive birds get zero separation force computed for them."""
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    vel = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    # Bird 1 is inactive, bird 0 is active
    active = np.array([True, False])
    force = separation_force(pos, vel, idx, active)
    # Bird 0 is active and sees bird 1 — gets a force (neighbour filtering
    # by active mask is the caller's responsibility).
    assert np.isfinite(force[0]).all()
    # Bird 1 is inactive → force slot stays at initialised zero
    assert np.allclose(force[1], 0.0)


def test_alignment_force_parallel():
    """Two birds with identical velocities → alignment force is zero."""
    pos = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    active = np.ones(2, dtype=bool)
    force = alignment_force(pos, vel, idx, active)
    # Identical velocities → avg/norm == vi/norm → force ≈ 0
    assert np.allclose(force[0], 0.0, atol=1e-6)


def test_alignment_force_opposite():
    """Two birds with opposite velocities → alignment force is nonzero."""
    pos = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    active = np.ones(2, dtype=bool)
    force = alignment_force(pos, vel, idx, active)
    assert not np.allclose(force[0], 0.0)
    assert np.isfinite(force).all()


def test_cohesion_force_single_neighbor(known_positions, known_velocities):
    """With one neighbour, cohesion force points directly toward it."""
    # Bird 0 has only bird 1 at (10, 0, 0) → cohesion should point toward +x
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = cohesion_force(known_positions, known_velocities, idx, active)
    # Force toward neighbour should have positive x component
    assert force[0, 0] > 0
    # Force should be along x axis only (neighbour is on x axis)
    assert abs(force[0, 1]) < 1e-6
    assert abs(force[0, 2]) < 1e-6


def test_cohesion_force_no_neighbors(known_positions, known_velocities):
    """Birds with no neighbours get zero cohesion force."""
    N = len(known_positions)
    active = np.ones(N, dtype=bool)
    empty_idx = np.zeros((N, 0), dtype=np.int32)
    force = cohesion_force(known_positions, known_velocities, empty_idx, active)
    assert np.allclose(force, 0.0)


def test_noise_force_unit_scale():
    """noise_force(N, 1.0) produces vectors with unit norm."""
    N = 500
    f = noise_force(N, 1.0)
    assert f.shape == (N, 3)
    norms = np.linalg.norm(f, axis=1)
    # All norms should be ~1.0 (noise is normalized after gaussian sampling)
    assert np.allclose(norms, 1.0, atol=1e-5)
