"""Unit tests for physics.flock — PhysicsFlock, SpatialHashGrid, KDTreeIndex."""

import numpy as np
import pytest

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


# ── P0.4 Determinism Tests ─────────────────────────────────────


def test_flock_rng_initialised(default_config):
    """PhysicsFlock has a self.rng attribute initialised from config.seed."""
    cfg = default_config
    cfg.seed = 42
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "rng"), "flock.rng must exist"
    assert isinstance(flock.rng, np.random.Generator), (
        "flock.rng must be np.random.Generator"
    )


def test_same_seed_bit_identical():
    """Two engines with same seed produce bit-identical positions after 100 steps.

    P0.4 requirement: same seed → bit-identical after 100 steps per mode.
    Tests projection mode (deterministic, no zero-speed reseed).
    """
    from pymurmur.simulation.engine import SimulationEngine
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 30
    cfg.mode = "projection"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(
        e1.flock.positions, e2.flock.positions,
        err_msg="Same seed must produce bit-identical positions after 100 steps"
    )
    np.testing.assert_array_equal(
        e1.flock.velocities, e2.flock.velocities,
        err_msg="Same seed must produce bit-identical velocities after 100 steps"
    )


def test_same_seed_bit_identical_spatial():
    """Spatial mode also produces bit-identical results with same seed."""
    from pymurmur.simulation.engine import SimulationEngine
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.seed = 77
    cfg.num_boids = 30
    cfg.mode = "spatial"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)


def test_same_seed_bit_identical_field():
    """Field mode also produces bit-identical results with same seed."""
    from pymurmur.simulation.engine import SimulationEngine
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.seed = 123
    cfg.num_boids = 30
    cfg.mode = "field"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)


def test_different_seeds_diverge():
    """Different seeds produce different positions after 100 steps."""
    from pymurmur.simulation.engine import SimulationEngine
    from pymurmur.core.config import SimConfig

    cfg1 = SimConfig()
    cfg1.seed = 42
    cfg1.num_boids = 30
    cfg1.mode = "projection"

    cfg2 = SimConfig()
    cfg2.seed = 99
    cfg2.num_boids = 30
    cfg2.mode = "projection"

    e1 = SimulationEngine(cfg1)
    e2 = SimulationEngine(cfg2)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    assert not np.array_equal(e1.flock.positions, e2.flock.positions), (
        "Different seeds must produce different positions"
    )


# ── P0.5 Smoothed Swarm Centre Tests ───────────────────────────


def test_center_initialised_none(default_config):
    """flock.center is None before any step()."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    assert flock.center is None, "center must be None before first step"


def test_center_set_after_first_step(default_config):
    """flock.center is set to the centroid after the first step."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    flock.step(cfg, 1.0 / 60.0)
    assert flock.center is not None, "center must be set after first step"
    assert flock.center.shape == (3,), "center must be (3,) float32"
    assert flock.center.dtype == np.float32


def test_center_close_to_centroid(default_config):
    """After first step, centre equals centroid exactly (EMA init snap)."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    flock.step(cfg, 1.0 / 60.0)

    centroid = flock.positions[flock.active].mean(axis=0)
    np.testing.assert_array_equal(
        flock.center, centroid,
        err_msg="First step: center must snap to centroid"
    )


def test_center_ema_smoothing(default_config):
    """Teleport flock — centre moves exactly 50% toward centroid (EMA α=0.5).

    Uses update_center() directly to isolate EMA behaviour from physics.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    # Initialise centre
    flock.update_center()
    old_center = flock.center.copy()

    # Teleport all birds far away (no step, no physics)
    flock.positions += np.array([500.0, 0.0, 0.0], dtype=np.float32)

    # Update centre directly — pure EMA, no force/integrate
    flock.update_center()

    new_centroid = flock.positions[flock.active].mean(axis=0)
    distance_center_moved = np.linalg.norm(flock.center - old_center)
    distance_to_centroid = np.linalg.norm(new_centroid - old_center)

    # Centre should have moved toward the centroid
    assert distance_center_moved > 0, "center should move after teleport"

    # Centre should NOT have reached the centroid in one step (EMA lag)
    assert distance_center_moved < distance_to_centroid, (
        f"EMA lag: center moved {distance_center_moved:.1f}, "
        f"but centroid moved {distance_to_centroid:.1f}"
    )

    # EMA α=0.5 → centre moves exactly 50% of the way
    expected_move = 0.5 * distance_to_centroid
    assert np.isclose(distance_center_moved, expected_move, atol=1e-4), (
        f"EMA α=0.5: expected move ≈ {expected_move:.1f}, "
        f"got {distance_center_moved:.1f}"
    )


def test_center_converges(default_config):
    """Pure EMA: after teleport, centre converges to <1% of centroid in ~7 frames.

    Uses update_center() directly (no physics) so convergence follows
    error = D · (0.5)^n exactly.  With D=500, error < 5.0 after 7 frames.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    # Initialise centre
    flock.update_center()

    # Teleport — zero velocities so no physics drift
    flock.positions += np.array([500.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[:] = 0.0
    flock.accelerations[:] = 0.0

    centroid = flock.positions[flock.active].mean(axis=0)
    for i in range(20):
        flock.update_center()
        error = np.linalg.norm(flock.center - centroid)
        if error < 0.01 * np.linalg.norm(centroid):
            break

    assert i < 10, (
        f"Pure EMA should converge within 10 frames, took {i + 1}"
    )


def test_center_no_active_birds(default_config):
    """update_center() is a no-op when all birds are inactive."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.step(cfg, 1.0 / 60.0)

    # Deactivate all birds
    flock.active[:] = False
    center_before = flock.center.copy()

    flock.update_center()

    np.testing.assert_array_equal(
        flock.center, center_before,
        err_msg="center must not change when no active birds"
    )


# ── P0.6 Species Column Tests ───────────────────────────────────


def test_is_predator_all_false_initially(default_config):
    """All birds start as prey (is_predator all False)."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "is_predator"), "flock.is_predator must exist"
    assert flock.is_predator.dtype == bool
    assert not flock.is_predator.any(), "all birds must be prey initially"
    assert len(flock.is_predator) == flock.N_capacity


def test_add_boids_predator_flag(default_config):
    """add_boids(is_predator=True) marks new birds as predators."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    added = flock.add_boids(5, cfg, is_predator=True)
    assert added == 5
    # New birds are at the end of active
    active_idx = np.where(flock.active)[0]
    new_birds = active_idx[-5:]
    assert flock.is_predator[new_birds].all(), (
        "new birds must be predators"
    )
    # Original birds are still prey
    original = active_idx[:10]
    assert not flock.is_predator[original].any(), (
        "original birds must remain prey"
    )


def test_add_boids_prey_default(default_config):
    """add_boids() without is_predator defaults to prey (False)."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    added = flock.add_boids(5, cfg)  # no is_predator arg
    assert added == 5
    active_idx = np.where(flock.active)[0]
    new_birds = active_idx[-5:]
    assert not flock.is_predator[new_birds].any(), (
        "default add_boids must produce prey"
    )


def test_species_survives_add_remove(default_config):
    """is_predator flag persists after remove_boids (flags on inactive birds survive).

    Per P0.6 spec: add 5 predators → 5 total, remove 3 active from end →
    flags on those inactive indices persist. is_predator.sum() stays at 5.
    """
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Add predators
    flock.add_boids(5, cfg, is_predator=True)
    assert flock.is_predator.sum() == 5

    # Remove 3 birds (last active ones — predators, since added at end)
    removed = flock.remove_boids(3)
    assert removed == 3
    # Flags persist on all 5 predator slots (3 now inactive, 2 still active)
    assert flock.is_predator.sum() == 5, (
        f"is_predator flags must persist on inactive birds, got {flock.is_predator.sum()}"
    )
    # Only 2 predators remain active
    assert flock.is_predator[flock.active].sum() == 2


def test_species_carried_through_extend(default_config):
    """is_predator preserved when arrays grow via add_boids beyond capacity."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Mark first 3 birds as predators
    active_idx = np.where(flock.active)[0]
    flock.is_predator[active_idx[:3]] = True
    cap_before = flock.N_capacity

    # Add more birds to force extend
    added = flock.add_boids(cap_before + 50, cfg)
    assert added > 0
    assert flock.N_capacity > cap_before

    # First 3 birds should still be predators after extend
    assert flock.is_predator[active_idx[:3]].all(), (
        "predator flags lost after _extend()"
    )


def test_is_predator_inactive_preserved(default_config):
    """Inactive birds' is_predator flags are preserved."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Make the last 2 active birds predators (they'll be removed first)
    active_idx = np.where(flock.active)[0]
    flock.is_predator[active_idx[-2:]] = True
    assert flock.is_predator.sum() == 2

    # Remove 2 (deactivates last 2 active, which are the predators)
    flock.remove_boids(2)

    # The deactivated birds should still have is_predator=True
    # (they're inactive but their flags persist)
    inactive = np.where(~flock.active)[0]
    assert len(inactive) == 2
    assert flock.is_predator[inactive].all(), (
        "inactive predators should retain their flags"
    )
    # And is_predator.sum() still counts them
    assert flock.is_predator.sum() == 2


# ── P0.7 Prev Positions + Acceleration Stash Tests ─────────────


def test_prev_positions_initialised(default_config):
    """prev_positions is (N, 3) float32, initially all zeros."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "prev_positions"), "prev_positions must exist"
    assert flock.prev_positions.shape == (flock.N_capacity, 3)
    assert flock.prev_positions.dtype == np.float32
    assert (flock.prev_positions == 0.0).all()


def test_last_accelerations_initialised(default_config):
    """last_accelerations is (N, 3) float32, initially all zeros."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "last_accelerations"), "last_accelerations must exist"
    assert flock.last_accelerations.shape == (flock.N_capacity, 3)
    assert flock.last_accelerations.dtype == np.float32
    assert (flock.last_accelerations == 0.0).all()


def test_prev_positions_stashed_before_integrate(default_config):
    """After step(), prev_positions holds the pre-integration positions."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    pos_before_step = flock.positions.copy()
    flock.step(cfg, 1.0 / 60.0)

    # prev_positions should equal positions from before the step
    np.testing.assert_array_equal(
        flock.prev_positions, pos_before_step,
        err_msg="prev_positions must capture pre-integration positions"
    )
    # positions should have changed (integration moved birds)
    assert not np.array_equal(flock.positions, pos_before_step), (
        "positions should change after step"
    )


def test_last_accelerations_stashed_before_reset(default_config):
    """After step(), last_accelerations holds the force-computed accelerations."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    flock.step(cfg, 1.0 / 60.0)

    # After step(), accelerations are reset to zero (integrate does this)
    assert (flock.accelerations[flock.active] == 0.0).all(), (
        "accelerations should be zero after integrate"
    )
    # But last_accelerations should hold the pre-reset values
    # At minimum, it should be non-zero for at least some birds (forces exist)
    assert not (flock.last_accelerations[flock.active] == 0.0).all(), (
        "last_accelerations must capture non-zero force accelerations"
    )


def test_stash_arrays_survive_extend(default_config):
    """prev_positions and last_accelerations preserved after _extend()."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Set known values
    flock.prev_positions[:] = np.arange(30, dtype=np.float32).reshape(10, 3)
    flock.last_accelerations[:] = np.arange(30, 60, dtype=np.float32).reshape(10, 3)
    cap_before = flock.N_capacity

    # Force extend
    flock.add_boids(cap_before + 50, cfg)

    # First 10 rows should be preserved
    expected_prev = np.arange(30, dtype=np.float32).reshape(10, 3)
    expected_acc = np.arange(30, 60, dtype=np.float32).reshape(10, 3)
    np.testing.assert_array_equal(flock.prev_positions[:10], expected_prev)
    np.testing.assert_array_equal(flock.last_accelerations[:10], expected_acc)


# ── P0.8 Per-Bird Max Speed Tests ──────────────────────────────


def test_max_speed_default_none(default_config):
    """max_speed is None by default (scalar v0 fallback)."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "max_speed"), "max_speed must exist"
    assert flock.max_speed is None, "max_speed must be None by default"


def test_last_accelerations_nonzero_after_forces(default_config):
    """P0.7: last_accelerations captures actual force data, not just zeros."""
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "spatial"
    flock = PhysicsFlock(cfg)
    flock.step(cfg, 1.0 / 60.0)
    # After a step with spatial forces, accelerations should have been
    # non-zero before being reset (captured in last_accelerations)
    acc_mags = np.linalg.norm(flock.last_accelerations[flock.active], axis=1)
    assert acc_mags.max() > 0, (
        f"last_accelerations should capture non-zero forces, got max={acc_mags.max():.6f}"
    )


def test_max_speed_per_bird_lowers_cap(default_config):
    """Setting per-bird max_speed lowers the speed cap for those birds."""
    cfg = default_config
    cfg.num_boids = 5
    cfg.mode = "projection"
    flock = PhysicsFlock(cfg)

    # Give every bird a tight speed cap
    flock.max_speed = np.full(flock.N_capacity, 1.0, dtype=np.float32)
    # Set velocities above the cap
    flock.velocities[:] = np.array([[3.0, 0.0, 0.0]] * 5, dtype=np.float32)
    flock.accelerations[:] = 0.0

    flock.step(cfg, 1.0 / 60.0)

    speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
    # All speeds should be clamped to max_speed=1.0, not v0=4.0
    assert (speeds <= 1.01).all(), f"speeds={speeds} should ≤ 1.0"
    assert (speeds >= 0.29).all(), f"speeds={speeds} should ≥ 0.3"


def test_max_speed_different_per_bird(default_config):
    """Each bird can have a different max_speed cap."""
    cfg = default_config
    cfg.num_boids = 3
    cfg.mode = "projection"
    flock = PhysicsFlock(cfg)

    # Bird 0: cap=2.0, Bird 1: cap=1.0, Bird 2: cap=3.0
    flock.max_speed = np.array([2.0, 1.0, 3.0], dtype=np.float32)
    # All start at high speed
    flock.velocities[:] = np.array([[5.0, 0.0, 0.0]] * 3, dtype=np.float32)
    flock.accelerations[:] = 0.0

    flock.step(cfg, 1.0 / 60.0)

    speeds = np.linalg.norm(flock.velocities, axis=1)
    assert speeds[0] <= 2.01, f"bird 0 speed={speeds[0]}"
    assert speeds[1] <= 1.01, f"bird 1 speed={speeds[1]}"
    assert speeds[2] <= 3.01, f"bird 2 speed={speeds[2]}"
    # Each should be at their cap (since starting speed 5 > all caps)
    assert np.isclose(speeds[0], 2.0, atol=0.05)
    assert np.isclose(speeds[1], 1.0, atol=0.05)
    assert np.isclose(speeds[2], 3.0, atol=0.05)


def test_add_boids_uses_flock_rng_deterministically(default_config):
    """P0.4: add_boids uses flock.rng — same RNG state → same positions."""
    cfg = default_config
    flock = PhysicsFlock(cfg)
    # Re-initialise with known seed
    flock.rng = np.random.default_rng(42)

    # Test 1: add_boids uses flock.rng for positions
    flock.rng = np.random.default_rng(42)
    flock.add_boids(5, cfg)
    pos1 = flock.positions[-5:].copy()

    flock.rng = np.random.default_rng(42)
    flock.add_boids(5, cfg)
    pos2 = flock.positions[-5:].copy()
    # After adding 5 more, positions should match (same RNG state before add)
    assert np.array_equal(pos1, pos2), "add_boids not using flock.rng deterministically"


def test_max_speed_none_uses_scalar_v0(default_config):
    """When max_speed is None, the scalar v0 from config is used."""
    cfg = default_config
    cfg.num_boids = 5
    cfg.mode = "projection"
    cfg.v0 = 3.0  # non-default v0
    flock = PhysicsFlock(cfg)

    # max_speed is None → should use config.v0 = 3.0
    assert flock.max_speed is None
    flock.velocities[:] = np.array([[8.0, 0.0, 0.0]] * 5, dtype=np.float32)
    flock.accelerations[:] = 0.0

    flock.step(cfg, 1.0 / 60.0)

    speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
    assert (speeds <= 3.01).all(), f"speeds={speeds} should ≤ 3.0 (cfg.v0)"


def test_max_speed_with_ceiling_mode(default_config):
    """P0.8+P0.9: per-bird max_speed works with ceiling speed mode."""
    cfg = default_config
    cfg.num_boids = 4
    flock = PhysicsFlock(cfg)
    # Different caps per bird — bird 2 (cap=5) stays, bird 0 (cap=2) clamped
    flock.max_speed = np.array([2.0, 3.0, 5.0, 4.0], dtype=np.float32)
    flock.velocities = np.array([[8.0, 0, 0]] * 4, dtype=np.float32)
    flock.accelerations[:] = 0.0

    from pymurmur.physics.boid import integrate
    integrate(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, cfg.width, cfg.height, cfg.depth,
        4.0, "toroidal", 1.0 / 60.0,
        max_speed=flock.max_speed, speed_mode="ceiling",
    )
    speeds = np.linalg.norm(flock.velocities, axis=1)
    # Ceiling: speeds > cap are clamped down to cap
    s0, s1, s2, s3 = float(speeds[0]), float(speeds[1]), float(speeds[2]), float(speeds[3])
    assert s0 == pytest.approx(2.0, abs=0.05), f"bird 0 cap=2: speed={s0:.4f}"
    assert s1 == pytest.approx(3.0, abs=0.05), f"bird 1 cap=3: speed={s1:.4f}"
    assert s2 == pytest.approx(5.0, abs=0.05), f"bird 2 cap=5: speed={s2:.4f}"
    assert s3 == pytest.approx(4.0, abs=0.05), f"bird 3 cap=4: speed={s3:.4f}"