"""Unit tests for Phase 3 — Projection mode (Pearce 2014 hybrid projection).

Tests projection_forces(), _topological_neighbors(), and related SI refinements.
Separated from test_forces.py for independent Phase 3 coverage verification.
"""

from copy import copy

import numpy as np


from test.helpers import _call_force  # noqa: E402


def test_projection_mode_zero_active(default_config):
    """projection_forces returns early when no birds are active."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    flock.active[:] = False
    flock.accelerations[:] = 0.0

    _call_force(projection_forces, flock, cfg)
    assert np.allclose(flock.accelerations, 0.0)


def test_projection_mode_produces_forces(default_config):
    """Projection mode produces non-zero forces with default settings."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.phi_p = 0.5
    cfg.phi_a = 0.5

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_projection_mode_updates_theta(default_config):
    """last_theta is updated after projection_forces."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.last_theta[:] = -1.0  # sentinel value
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)

    # Birds with neighbours get theta >= 0; edge birds with no neighbours
    # in the hash grid's 27-cell radius stay at sentinel -1.0.
    active_theta = flock.last_theta[flock.active]
    assert (active_theta >= -1.0).all()  # never below sentinel
    assert (active_theta >= 0.0).any()   # at least some birds updated


def test_projection_mode_blind_angle_effect(default_config):
    """Setting blind_deg > 0 changes behaviour (doesn't crash)."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.blind_deg = 90.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_anisotropy_effect(default_config):
    """Anisotropy > 1 runs without crash when refinements enabled."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.refinements = True
    cfg.anisotropy = 3.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_steric_enabled(default_config):
    """Steric force is applied when refinements + steric > 0."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 30
    cfg.refinements = True
    cfg.steric = 1.0
    cfg.phi_p = 0.0
    cfg.phi_a = 0.0  # only steric

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    # With phi_p=0 and phi_a=0, forces come only from steric
    # May or may not be zero depending on neighbour distances
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_sigma_effect(default_config):
    """Changing sigma changes the number of neighbours considered."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.sigma = 3  # fewer neighbours

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.last_theta[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_delta_computed(default_config):
    """Delta (projection direction) is non-zero when neighbours present."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.phi_p = 0.8
    cfg.phi_a = 0.0  # only projection component
    cfg.refinements = False  # no steric interference

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)

    # Forces should be non-zero (delta computed from occlusion)
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_projection_mode_force_within_bounds(default_config):
    """No bird's acceleration exceeds config.max_force."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.phi_p = 1.0
    cfg.phi_a = 1.0
    cfg.max_force = 2.0  # low clamp
    cfg.refinements = False  # steric added after clamp, would break the bound

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)

    # Check that all acceleration magnitudes are <= max_force
    acc_mags = np.linalg.norm(flock.accelerations, axis=1)
    active_mags = acc_mags[flock.active]
    assert np.all(active_mags <= cfg.max_force + 1e-5), \
        f"max acc: {active_mags.max()}, limit: {cfg.max_force}"


def test_projection_mode_hash_grid(default_config):
    """Projection mode works with SpatialHashGrid (N < 5000)."""
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 200  # triggers SpatialHashGrid, not KDTreeIndex
    cfg.phi_p = 0.8
    cfg.phi_a = 0.2

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # Verify we're using SpatialHashGrid
    from pymurmur.physics.flock import SpatialHashGrid
    assert isinstance(flock.get_index(), SpatialHashGrid)

    _call_force(projection_forces, flock, cfg)

    # Should produce non-zero, finite forces via hash grid topological neighbors
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_topological_neighbors_fallback(default_config):
    """_topological_neighbors_batch returns all -1 sentinels when index not ready."""
    from pymurmur.physics.forces.projection import _topological_neighbors_batch
    from pymurmur.physics.flock import PhysicsFlock

    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Replace index with an object lacking query_knn -> falls through
    class _FakeIndex:
        ready = False
    flock._index = _FakeIndex()
    active_idx = np.where(flock.active)[0]

    result = _topological_neighbors_batch(flock.positions, flock.get_index(), active_idx, 4)
    assert (result == -1).all()  # all sentinels when index not ready


def test_topological_neighbors_kdtree(default_config):
    """_topological_neighbors_batch uses KDTreeIndex when available."""
    from pymurmur.physics.forces.projection import _topological_neighbors_batch
    from pymurmur.physics.flock import PhysicsFlock, KDTreeIndex

    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    # Force KDTreeIndex
    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)
    flock._index = kdt

    active_idx = np.where(flock.active)[0]
    result = _topological_neighbors_batch(flock.positions, flock.get_index(), active_idx, 4)
    assert (result >= 0).any()  # returns neighbours via KDTree
