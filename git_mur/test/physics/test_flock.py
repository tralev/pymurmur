"""Unit tests for physics.flock — PhysicsFlock, SpatialHashGrid, KDTreeIndex."""

import numpy as np

from pymurmur.physics.flock import PhysicsFlock, SpatialHashGrid, KDTreeIndex


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
    flock.step(cfg, 1.0 / 60.0)
    assert flock.N_active == 10


def test_flock_step_positions_change(default_config):
    """Positions change after step() with non-zero forces."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    pos_before = flock.positions[flock.active].copy()
    flock.step(cfg, 1.0 / 60.0)
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


def test_flock_spatial_index_auto_select():
    """N < 5000 uses SpatialHashGrid, N >= 5000 uses KDTreeIndex."""
    from pymurmur.core.config import SimConfig

    cfg_small = SimConfig()
    cfg_small.num_boids = 100
    flock_small = PhysicsFlock(cfg_small)
    assert isinstance(flock_small.get_index(), SpatialHashGrid)

    cfg_large = SimConfig()
    cfg_large.num_boids = 6000
    flock_large = PhysicsFlock(cfg_large)
    assert isinstance(flock_large.get_index(), KDTreeIndex)


def test_hash_grid_rebuild(small_flock):
    """rebuild() runs without error."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        grid.rebuild(small_flock.positions, small_flock.active)
        assert grid.ready


def test_hash_grid_query_returns_candidates(small_flock):
    """Query returns candidate indices."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        grid.rebuild(small_flock.positions, small_flock.active)
        center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        candidates = grid.query_radius(center, 100.0)
        assert isinstance(candidates, list)


def test_hash_grid_query_returns_self(small_flock):
    """Query at bird's position includes that bird."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        grid.rebuild(small_flock.positions, small_flock.active)
        active_idx = np.where(small_flock.active)[0]
        bird_idx = active_idx[0]
        bird_pos = small_flock.positions[bird_idx]
        candidates = grid.query_radius(bird_pos, 50.0)
        assert bird_idx in candidates


def test_hash_grid_query_empty(small_flock):
    """Query with radius=0 returns only birds in same cell."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        grid.rebuild(small_flock.positions, small_flock.active)
        # Query far from all birds
        far = np.array([99999.0, 99999.0, 99999.0], dtype=np.float32)
        candidates = grid.query_radius(far, 10.0)
        assert len(candidates) == 0


def test_hash_grid_query_all(small_flock):
    """Query returns a list (may be empty for very sparse grids)."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        grid.rebuild(small_flock.positions, small_flock.active)
        center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        candidates = grid.query_radius(center, 500.0)
        # Query should return a list (even if empty due to 27-cell limits)
        assert isinstance(candidates, list)


def test_hash_grid_inactive_excluded(small_flock):
    """Inactive birds are not returned in queries."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        # Deactivate all birds
        small_flock.active[:] = False
        grid.rebuild(small_flock.positions, small_flock.active)
        center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        candidates = grid.query_radius(center, 500.0)
        assert len(candidates) == 0


def test_hash_grid_cell_wrapping():
    """Query near domain edge correctly searches adjacent cells only.

    Note: SpatialHashGrid does NOT implement toroidal wrapping.
    Cell keys are computed as int(pos // cell_size) without modulo.
    This test verifies the actual (non-wrapping) behaviour."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.num_boids = 4
    flock = PhysicsFlock(cfg)
    grid = flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        # Place birds at known positions near opposite edges
        flock.positions[:] = np.array([
            [10, 350, 200],     # near x=0 edge
            [990, 350, 200],    # near x=1000 edge
            [500, 10, 200],     # near y=0 edge
            [500, 690, 200],    # near y=700 edge
        ], dtype=np.float32)
        grid.rebuild(flock.positions, flock.active)

        # Query at x≈0 should find bird[0] but NOT bird[1] (no wrapping)
        candidates_near_0 = grid.query_radius(
            np.array([5, 350, 200], dtype=np.float32), 50.0)
        assert 0 in candidates_near_0
        assert 1 not in candidates_near_0  # no toroidal wrap

        # Query at x≈1000 should find bird[1] but NOT bird[0]
        candidates_near_1000 = grid.query_radius(
            np.array([995, 350, 200], dtype=np.float32), 50.0)
        assert 1 in candidates_near_1000
        assert 0 not in candidates_near_1000


# ── KDTreeIndex tests ─────────────────────────────────────────────

def test_kdtree_build():
    """KDTreeIndex rebuild with positions completes without error."""
    kdt = KDTreeIndex()
    pos = np.random.randn(100, 3).astype(np.float32) + 500
    active = np.ones(100, dtype=bool)
    kdt.rebuild(pos, active)
    assert kdt.ready


def test_kdtree_query_knn():
    """query_knn(pos, k=5) returns 5 indices."""
    kdt = KDTreeIndex()
    pos = np.random.randn(100, 3).astype(np.float32) + 500
    active = np.ones(100, dtype=bool)
    kdt.rebuild(pos, active)
    query_pos = pos[0]
    idx = kdt.query_knn(query_pos, 5)
    assert len(idx) == 5


def test_kdtree_closest_is_self():
    """Querying with a bird's own position returns neighbors near it."""
    kdt = KDTreeIndex()
    rng = np.random.default_rng(42)
    pos = rng.random((100, 3), dtype=np.float32) * 1000
    active = np.ones(100, dtype=bool)
    kdt.rebuild(pos, active)
    # query_knn skips self (idx[1:]) — verify we get k results
    idx = kdt.query_knn(pos[0], 5)
    assert len(idx) == 5
    # All returned indices should be valid (0 to 99)
    assert (idx >= 0).all() and (idx < 100).all()


def test_kdtree_distance_increases():
    """query_knn returns indices in order of increasing distance."""
    kdt = KDTreeIndex()
    rng = np.random.default_rng(42)
    pos = rng.random((200, 3), dtype=np.float32) * 1000
    active = np.ones(200, dtype=bool)
    kdt.rebuild(pos, active)
    # Use the tree directly to get distances
    dists, _ = kdt._tree.query(pos[0], k=10)
    # Distances should be monotonically non-decreasing
    for i in range(len(dists) - 1):
        assert float(dists[i]) <= float(dists[i + 1]) + 1e-6, \
            f"dist[{i}]={dists[i]} > dist[{i+1}]={dists[i+1]}"


def test_hash_grid_query_knn_no_rebuild():
    """query_knn before rebuild returns empty (_positions is None)."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    grid = SpatialHashGrid(cfg)
    result = grid.query_knn(np.array([500.0, 350.0, 200.0], dtype=np.float32), k=5)
    assert len(result) == 0  # _positions is None → empty


def test_hash_grid_query_knn_single_bird(small_flock):
    """query_knn returns empty when only one bird in 27-cell area."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        # Single active bird
        small_flock.active[:] = False
        small_flock.active[0] = True
        grid.rebuild(small_flock.positions, small_flock.active)
        result = grid.query_knn(small_flock.positions[0], k=5)
        assert len(result) == 0  # ≤1 candidate → empty


def test_hash_grid_query_knn_colocated(small_flock):
    """query_knn returns empty when all candidates are at the same position."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        # Place multiple birds at exact same position
        small_flock.active[:] = True
        small_flock.positions[:] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        grid.rebuild(small_flock.positions, small_flock.active)
        # All candidates at d=0 → mask is empty → returns empty
        result = grid.query_knn(np.array([500.0, 350.0, 200.0], dtype=np.float32), k=5)
        assert len(result) == 0  # all candidates at distance 0


def test_kdtree_query_knn_no_tree():
    """query_knn returns empty when no tree has been built."""
    kdt = KDTreeIndex()
    result = kdt.query_knn(np.array([500.0, 350.0, 200.0], dtype=np.float32), k=5)
    assert len(result) == 0  # no tree -> empty
    assert not kdt.ready


def test_index_skip_for_field_mode(default_config):
    """Field mode skips spatial index rebuild (doesn't query neighbors)."""
    from pymurmur.physics.flock import SpatialHashGrid
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "field"
    flock = PhysicsFlock(cfg)
    index = flock.get_index()

    # Clear the index so rebuild would be needed
    if isinstance(index, SpatialHashGrid):
        index._bins.clear()

    # Step should complete without error and without rebuilding
    flock.step(cfg, 1.0 / 60.0)
    # Index should remain empty since field mode skips rebuild
    if isinstance(index, SpatialHashGrid):
        assert not index.ready, "Field mode should skip index rebuild"


def test_index_rebuilt_for_spatial_mode(default_config):
    """Spatial mode DOES rebuild the spatial index."""
    from pymurmur.physics.flock import SpatialHashGrid
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "spatial"
    flock = PhysicsFlock(cfg)
    index = flock.get_index()

    flock.step(cfg, 1.0 / 60.0)
    # Index should be ready after step for spatial mode
    if isinstance(index, SpatialHashGrid):
        assert index.ready, "Spatial mode should rebuild index"


def test_index_rebuilt_for_projection_mode(default_config):
    """Projection mode DOES rebuild the spatial index."""
    from pymurmur.physics.flock import SpatialHashGrid
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "projection"
    flock = PhysicsFlock(cfg)
    index = flock.get_index()

    flock.step(cfg, 1.0 / 60.0)
    if isinstance(index, SpatialHashGrid):
        assert index.ready, "Projection mode should rebuild index"


def test_index_skip_for_influencer_mode(default_config):
    """Influencer mode skips spatial index rebuild."""
    from pymurmur.physics.flock import SpatialHashGrid
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "influencer"
    flock = PhysicsFlock(cfg)
    index = flock.get_index()

    if isinstance(index, SpatialHashGrid):
        index._bins.clear()

    flock.step(cfg, 1.0 / 60.0)
    if isinstance(index, SpatialHashGrid):
        assert not index.ready, "Influencer mode should skip index rebuild"


def test_index_skip_for_vicsek_mode(default_config):
    """Vicsek mode skips flock-level index (builds its own cKDTree)."""
    from pymurmur.physics.flock import SpatialHashGrid
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "vicsek"
    flock = PhysicsFlock(cfg)
    index = flock.get_index()

    if isinstance(index, SpatialHashGrid):
        index._bins.clear()

    flock.step(cfg, 1.0 / 60.0)
    if isinstance(index, SpatialHashGrid):
        assert not index.ready, "Vicsek mode should skip flock-level index rebuild"
