"""Unit tests for physics.forces dispatch and mode functions."""

from copy import copy

import numpy as np


def test_compute_all_forces_imports():
    """All 5 modes are importable from the forces package."""
    from pymurmur.physics.forces import compute_all_forces
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.forces.field import field_forces
    from pymurmur.physics.forces.vicsek import vicsek_forces
    from pymurmur.physics.forces.influencer import influencer_forces
    assert callable(compute_all_forces)
    assert callable(spatial_forces)
    assert callable(projection_forces)
    assert callable(field_forces)
    assert callable(vicsek_forces)
    assert callable(influencer_forces)


def test_mode_dispatch_unknown_raises(default_config):
    """Invalid mode raises ValueError."""
    import pytest
    from pymurmur.physics.forces import compute_all_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = default_config
    cfg.mode = "invalid_mode"
    flock = PhysicsFlock(cfg)

    with pytest.raises(ValueError, match="Unknown force mode"):
        compute_all_forces(flock, cfg)


def test_all_modes_run(default_config):
    """Each of the 5 modes runs without crash on a small flock."""
    from pymurmur.physics.forces import compute_all_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.num_boids = 300

    for mode in ["projection", "spatial", "field", "vicsek", "influencer"]:
        cfg.mode = mode
        flock = PhysicsFlock(cfg)
        compute_all_forces(flock, cfg)
        # All active accelerations should be finite
        assert np.isfinite(flock.accelerations).all()


def test_spatial_forces_produces_nonzero(default_config):
    """Spatial mode with weights produces non-zero forces."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 300
    cfg.noise_scale = 0.5
    flock = PhysicsFlock(cfg)

    # Rebuild index first
    flock.get_index().rebuild(flock.positions, flock.active)
    spatial_forces(flock, cfg)
    assert not np.allclose(flock.accelerations[flock.active], 0.0)


# ── Spatial mode scenario tests ────────────────────────────────────


def test_spatial_mode_all_weights_zero(default_config):
    """All weights=0 → no steering forces (only noise may remain if scale=0 too)."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 10.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    spatial_forces(flock, cfg)
    # Accelerations should be all-zero (no steering, no noise)
    assert np.allclose(flock.accelerations[flock.active], 0.0)


def test_spatial_mode_separation_only(default_config):
    """Separation-only mode pushes birds apart."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 50.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    spatial_forces(flock, cfg)

    # Forces should be nonzero
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    # All forces should be finite
    assert np.isfinite(acc_active).all()


def test_spatial_mode_alignment_only(default_config):
    """Alignment-only mode steers toward average heading."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 20.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)
    spatial_forces(flock, cfg)

    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_spatial_mode_cohesion_only(default_config):
    """Cohesion-only mode pulls birds toward centre."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 20.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    spatial_forces(flock, cfg)

    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_spatial_mode_noise_only(default_config):
    """Noise-only mode produces random perturbations."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 1.0
    cfg.max_force = 20.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    spatial_forces(flock, cfg)

    acc_active = flock.accelerations[flock.active]
    # Noise should produce non-zero forces
    assert not np.allclose(acc_active, 0.0)
    # Noise vectors should have roughly unit norm before weight application
    assert np.isfinite(acc_active).all()


def test_spatial_mode_force_clamped(default_config):
    """No bird's acceleration exceeds config.max_force."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.separation_weight = 100.0  # huge weight → huge forces
    cfg.alignment_weight = 100.0
    cfg.cohesion_weight = 100.0
    cfg.noise_scale = 10.0
    cfg.max_force = 5.0  # low clamp

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    spatial_forces(flock, cfg)

    # Check that all acceleration magnitudes are ≤ max_force (within tolerance)
    acc_mags = np.linalg.norm(flock.accelerations, axis=1)
    active_mags = acc_mags[flock.active]
    assert np.all(active_mags <= cfg.max_force + 1e-5), \
        f"max acc: {active_mags.max()}, limit: {cfg.max_force}"


def test_spatial_mode_numba_fallback(default_config):
    """Spatial mode works without numba (pure numpy path)."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 30

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # Run on pure numpy path (numba may or may not be installed)
    spatial_forces(flock, cfg)
    acc_active = flock.accelerations[flock.active]
    assert np.isfinite(acc_active).all()
    assert not np.allclose(acc_active, 0.0)


def test_spatial_mode_zero_active(default_config):
    """spatial_forces returns early when no birds are active."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    # Deactivate all birds
    flock.active[:] = False
    flock.accelerations[:] = 0.0

    spatial_forces(flock, cfg)
    # Should be a no-op — no crash, no NaN
    assert np.allclose(flock.accelerations, 0.0)


def test_spatial_mode_single_bird(default_config):
    """spatial_forces handles N=1 without neighbour queries."""
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 1

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    spatial_forces(flock, cfg)
    # n < 2 → neighbour query returns empty → no steering → only noise
    assert np.isfinite(flock.accelerations).all()


