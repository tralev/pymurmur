"""Unit tests for physics.flock — PhysicsFlock init/state (core ops).

Remaining core after the file-size split (spatial index tests moved to
test_spatial_index.py; species/stash/max-speed to test_flock_state.py;
spawn_at to test_flock_spawn_at.py/test_flock_spawn_at_engine.py; D6
seed semantics to test_flock_seed_semantics.py; P0.4 determinism tests
to test_flock_determinism.py; P0.5 smoothed swarm centre tests to
test_flock_swarm_centre.py).
"""

import numpy as np

from pymurmur.physics.flock import PhysicsFlock
from test.helpers import _step_flock  # noqa: E402 — shared test helper


def test_flock_init_creates_birds(default_config):
    """PhysicsFlock(config) has N_active == config.num_boids."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    assert flock.N_active == 50


def test_flock_init_positions_in_domain(default_config):
    """All positions within domain bounds."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    pos = flock.positions[flock.active]
    assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= cfg.width).all()


def test_flock_init_velocities_nonzero(default_config):
    """All velocities have non-zero norm."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
    assert (speeds > 0).all()


def test_flock_init_accelerations_zero(default_config):
    """All accelerations are zero after init."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    assert (flock.accelerations == 0.0).all()


def test_flock_add_boids(default_config):
    """add_boids(5) increases N_active by 5."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    initial = flock.N_active
    flock.add_boids(5, cfg)
    assert flock.N_active == initial + 5


def test_flock_remove_boids(default_config):
    """remove_boids(5) decreases N_active by 5."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    initial = flock.N_active
    removed = flock.remove_boids(5)
    assert removed == 5
    assert flock.N_active == initial - 5


def test_flock_remove_boids_deactivates(default_config):
    """Removed birds have active[i] = False."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    # Find an active bird
    active_idx = np.where(flock.active)[0]
    target = active_idx[-1]
    flock.remove_boids(1)
    assert not flock.active[target]


def test_flock_step_runs(default_config):
    """flock.step(config, dt) completes without error."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    _step_flock(flock, cfg, 1.0 / 60.0)
    assert flock.N_active == 10


def test_flock_step_positions_change(default_config):
    """Positions change after step() with non-zero forces."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    pos_before = flock.positions[flock.active].copy()
    _step_flock(flock, cfg, 1.0 / 60.0)
    pos_after = flock.positions[flock.active]
    assert not np.allclose(pos_before, pos_after)


def test_flock_add_boids_initializes(default_config):
    """Added birds have non-zero positions and velocities."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    added = flock.add_boids(5, cfg)
    assert added == 5
    # New birds are at the end of the active array
    active_idx = np.where(flock.active)[0]
    new_birds = active_idx[-5:]
    for i in new_birds:
        assert not np.allclose(flock.positions[i], 0.0)
        assert np.linalg.norm(flock.velocities[i]) > 0


def test_flock_add_beyond_capacity(default_config):
    """add_boids() extends arrays when all slots filled."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    cap_before = flock.N_capacity
    # Try to add more than capacity
    added = flock.add_boids(cap_before + 100, cfg)
    assert added > 0
    assert flock.N_capacity > cap_before


def test_flock_remove_all(default_config):
    """Removing all birds leaves N_active = 0."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    flock.remove_boids(20)
    assert flock.N_active == 0


def test_flock_seeded_reproducible(default_config):
    """Same seed + same config → identical flock state."""
    from copy import copy

    cfg1 = copy(default_config)
    cfg1.seed = 42
    cfg1.num_boids = 30
    flock1 = PhysicsFlock(cfg1)

    cfg2 = copy(default_config)
    cfg2.seed = 42
    cfg2.num_boids = 30
    flock2 = PhysicsFlock(cfg2)

    assert np.allclose(flock1.positions, flock2.positions)
    assert np.allclose(flock1.velocities, flock2.velocities)
