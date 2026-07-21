"""Unit tests for physics.flock — PhysicsFlock, SpatialHashGrid, KDTreeIndex."""

import numpy as np
import pytest

from pymurmur.physics.flock import KDTreeIndex, PhysicsFlock, SpatialHashGrid
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
    """Query in empty cell returns no candidates when no birds present."""
    grid = small_flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        # Place all birds at a known cell
        small_flock.positions[:] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        grid.rebuild(small_flock.positions, small_flock.active)

        # Query in a cell far from the birds' cell
        # With cell_size=70 and cols=15 (1000/70), bird cell = (7, 5, 2)
        # Query at cell (4, 5, 2) — neighbor cells {3,4,5} don't include 7
        far = np.array([300.0, 350.0, 200.0], dtype=np.float32)
        candidates = grid.query_radius(far, 50.0)
        # With cell_size=70, 27-cell neighborhood from cell (4,5,2)
        # doesn't reach cell (7,5,2) → no candidates
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
    """Query near domain edge finds birds across the toroidal seam (P2.5).

    Modulo-wrapped cell keys + min-image distances enable correct
    cross-boundary neighbour queries."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.num_boids = 4
    cfg.width = 1000
    cfg.height = 700
    cfg.depth = 400
    cfg.visual_range = 100  # small enough to separate birds
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

        # P2.5: query at x≈0 finds bird[0] AND cross-seam bird[1]
        candidates_near_0 = grid.query_radius(
            np.array([5, 350, 200], dtype=np.float32), 50.0)
        assert 0 in candidates_near_0, "bird at x=10 should be found"
        assert 1 in candidates_near_0, (
            "P2.5: bird at x=990 should be found cross-seam via wrapped cells"
        )

        # P2.5: query at x≈1000 finds bird[1] AND cross-seam bird[0]
        candidates_near_1000 = grid.query_radius(
            np.array([995, 350, 200], dtype=np.float32), 50.0)
        assert 1 in candidates_near_1000, "bird at x=990 should be found"
        assert 0 in candidates_near_1000, (
            "P2.5: bird at x=10 should be found cross-seam via wrapped cells"
        )


def test_hash_grid_toroidal_distance():
    """P2.5: query_knn uses min-image distances for correct toroidal ranking.

    Two birds near opposite X boundaries: bird at x=10, bird at x=990.
    Toroidal distance between them is ~20 (across the seam), not ~980.
    query_knn should rank by toroidal distance."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.num_boids = 4
    cfg.width = 1000
    cfg.height = 700
    cfg.depth = 400
    cfg.visual_range = 200
    flock = PhysicsFlock(cfg)
    grid = flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        flock.positions[:] = np.array([
            [10, 350, 200],     # bird 0
            [990, 350, 200],    # bird 1: ~20 away toroidally
            [500, 350, 200],    # bird 2: ~490 away
            [500, 600, 200],    # bird 3: far
        ], dtype=np.float32)
        grid.rebuild(flock.positions, flock.active)

        # Query from bird 0
        result = grid.query_knn(flock.positions[0], k=3)

        # bird 1 (toroidal neighbor) should be closest
        assert result[0] == 1, (
            f"Expected bird 1 closest (toroidal dist ~20), got {list(result)}"
        )


def test_hash_grid_toroidal_yz_axes():
    """P2.5: cross-seam queries work for Y and Z axes too.

    Birds near opposite Y boundaries (y=10, y=690) and Z boundaries
    (z=10, z=390) should find each other across the seam."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.num_boids = 6
    cfg.width = 1000
    cfg.height = 700
    cfg.depth = 400
    cfg.visual_range = 150
    flock = PhysicsFlock(cfg)
    grid = flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        flock.positions[:] = np.array([
            [500, 10, 200],     # bird 0 — near y=0
            [500, 690, 200],    # bird 1 — near y=700, toroidal ~20
            [500, 350, 10],     # bird 2 — near z=0
            [500, 350, 390],    # bird 3 — near z=400, toroidal ~20
            [500, 350, 200],    # bird 4 — centre
            [500, 500, 200],    # bird 5 — far
        ], dtype=np.float32)
        grid.rebuild(flock.positions, flock.active)

        # Y-axis: bird 0 at y=10 queries → bird 1 should be found cross-seam
        candidates_y = grid.query_radius(flock.positions[0], 50.0)
        assert 1 in candidates_y, (
            f"P2.5 Y-axis: bird at y=690 should be found cross-seam, got {candidates_y}"
        )

        # Z-axis: bird 2 at z=10 queries → bird 3 should be found cross-seam
        candidates_z = grid.query_radius(flock.positions[2], 50.0)
        assert 3 in candidates_z, (
            f"P2.5 Z-axis: bird at z=390 should be found cross-seam, got {candidates_z}"
        )


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
    _step_flock(flock, cfg, 1.0 / 60.0)
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

    _step_flock(flock, cfg, 1.0 / 60.0)
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

    _step_flock(flock, cfg, 1.0 / 60.0)
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

    _step_flock(flock, cfg, 1.0 / 60.0)
    if isinstance(index, SpatialHashGrid):
        assert not index.ready, "Influencer mode should skip index rebuild"


def test_index_rebuilt_for_vicsek_mode(default_config):
    """Vicsek mode now uses the flock-level spatial index (I3.1)."""
    from pymurmur.physics.flock import SpatialHashGrid
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "vicsek"
    flock = PhysicsFlock(cfg)
    index = flock.get_index()

    _step_flock(flock, cfg, 1.0 / 60.0)
    if isinstance(index, SpatialHashGrid):
        assert index.ready, "Vicsek mode should rebuild shared index (I3.1)"


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
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

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
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

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


def test_same_seed_bit_identical_spatial_with_noise():
    """Spatial mode with noise_scale > 0 is deterministic (I1.5 regression).

    This guards against noise_force being called without the seeded rng —
    a missing rng argument causes np.random.default_rng() which produces
    different values on every call.  Fixed by passing rng to noise_force.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 99
    cfg.num_boids = 40
    cfg.mode = "spatial"
    cfg.noise_scale = 1.5

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(50):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(
        e1.flock.positions, e2.flock.positions,
        err_msg="spatial mode with noise_scale > 0 must be deterministic (I1.5)"
    )
    np.testing.assert_array_equal(
        e1.flock.velocities, e2.flock.velocities,
        err_msg="spatial mode with noise_scale > 0 must be deterministic (I1.5)"
    )


def test_same_seed_bit_identical_field():
    """Field mode also produces bit-identical results with same seed."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

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


def test_same_seed_bit_identical_vicsek():
    """Vicsek mode produces bit-identical results with same seed (I1.5)."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 50
    cfg.mode = "vicsek"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)
    np.testing.assert_array_equal(e1.flock.velocities, e2.flock.velocities)


def test_same_seed_bit_identical_influencer():
    """Influencer mode produces bit-identical results with same seed (I1.5)."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 77
    cfg.num_boids = 30
    cfg.mode = "influencer"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)
    np.testing.assert_array_equal(e1.flock.velocities, e2.flock.velocities)


def test_all_modes_deterministic():
    """Parametric: every force mode produces bit-identical results with same seed."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    for mode in ["projection", "spatial", "field", "vicsek", "influencer"]:
        cfg = SimConfig()
        cfg.seed = 123
        cfg.num_boids = 20
        cfg.mode = mode
        if mode == "spatial":
            cfg.noise_scale = 1.0  # exercise the noise RNG path (I1.5 regression guard)

        e1 = SimulationEngine(cfg)
        e2 = SimulationEngine(cfg)

        for _ in range(50):
            e1.step(1.0 / 60.0)
            e2.step(1.0 / 60.0)

        np.testing.assert_array_equal(
            e1.flock.positions, e2.flock.positions,
            err_msg=f"{mode}: same seed must produce bit-identical positions"
        )


@pytest.mark.parametrize("mode", ["projection", "spatial", "field", "vicsek", "influencer"])
def test_different_seeds_diverge(mode):
    """Different seeds produce different positions after 100 steps.

    Parametrized across all 5 force modes. Verifies that seed-based
    RNG pipeline works for every mode — different seed → different
    trajectory, which is the complement of the bit-identical test.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg1 = SimConfig()
    cfg1.seed = 42
    cfg1.num_boids = 30
    cfg1.mode = mode

    cfg2 = SimConfig()
    cfg2.seed = 99
    cfg2.num_boids = 30
    cfg2.mode = mode

    e1 = SimulationEngine(cfg1)
    e2 = SimulationEngine(cfg2)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    assert not np.array_equal(e1.flock.positions, e2.flock.positions), (
        f"{mode}: different seeds must produce different positions"
    )


# ── P0.5 Smoothed Swarm Centre Tests ───────────────────────────


def test_center_initialised_none(default_config):
    """D1: flock.center is initialised to domain centre (not None).

    Before D1: center was None before any step (snap-to-centroid on frame 0).
    After D1:  center starts at (W/2, H/2, D/2) so sphere boundary is
               always centred on domain centre from frame 0.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    assert flock.center is not None, "D1: center initialised to domain centre"
    assert flock.center.shape == (3,), "center must be (3,) float32"
    assert flock.center.dtype == np.float32
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    np.testing.assert_array_equal(flock.center, C)


def test_center_set_after_first_step(default_config):
    """D1: flock.center is initialised to domain centre before first step,
    and EMA-drifts toward the centroid after step().

    Before D1: center snapped to centroid on frame 0 (None → centroid).
    After D1:  center starts at domain centre, then EMA blends toward
               centroid: center ← center + 0.5·(centroid − center).
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    C_initial = flock.center.copy()
    _step_flock(flock, cfg, 1.0 / 60.0)
    assert flock.center is not None, "center must be set after first step"
    assert flock.center.shape == (3,), "center must be (3,) float32"
    assert flock.center.dtype == np.float32
    # D1: After EMA, center moves toward centroid (not identity snap)
    # It should differ from initial domain centre after the first step
    assert not np.allclose(flock.center, C_initial), (
        "center should EMA-drift from domain centre toward centroid"
    )


def test_center_close_to_centroid(default_config):
    """D1: After first step, centre is halfway between domain centre and centroid.

    Before D1: center snapped to centroid (EMA init snap, exact match).
    After D1:  center starts at domain centre, EMA: center += 0.5·(centroid − center).
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    C_initial = flock.center.copy()
    _step_flock(flock, cfg, 1.0 / 60.0)

    centroid = flock.positions[flock.active].mean(axis=0)
    # D1: After EMA, center = (C_initial + centroid) / 2
    expected = (C_initial + centroid) / 2.0
    np.testing.assert_allclose(
        flock.center, expected, rtol=0.05, atol=5.0,
        err_msg="center should be halfway between domain centre and centroid"
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
    for i in range(20):  # noqa: B007
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
    _step_flock(flock, cfg, 1.0 / 60.0)

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
    _step_flock(flock, cfg, 1.0 / 60.0)

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
    _step_flock(flock, cfg, 1.0 / 60.0)

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
    _step_flock(flock, cfg, 1.0 / 60.0)
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

    _step_flock(flock, cfg, 1.0 / 60.0)

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

    _step_flock(flock, cfg, 1.0 / 60.0)

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

    _step_flock(flock, cfg, 1.0 / 60.0)

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


# ── P0.4 Determinism — AST scan for module-level np.random ───────


def test_no_module_level_np_random():
    """P0.4: No module-level np.random.* calls remain in pymurmur/.

    Scans every .py file under pymurmur/ for bare `np.random.` calls
    (i.e. calls on the module-level RNG, not on a local Generator instance).
    P0.4 requires all stochastic sites to use flock.rng or a local Generator.
    """
    import ast
    from pathlib import Path

    violations = []
    pymurmur_root = Path("pymurmur")

    for py_file in sorted(pymurmur_root.rglob("*.py")):
        if py_file.name == "__init__.py" and py_file.stat().st_size < 10:
            continue
        tree = ast.parse(py_file.read_text())

        for node in ast.walk(tree):
            # Match: np.random.<method>(...) — bare np.random call
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == "np"
                        and node.func.value.attr == "random"):
                    # Exclude: np.random.default_rng(...) — that's allowed (creates Generator)
                    if node.func.attr != "default_rng":
                        violations.append(
                            f"{py_file}:{node.lineno}: np.random.{node.func.attr}(...)"
                        )

    assert not violations, (
        f"P0.4 violation: {len(violations)} module-level np.random.* call(s) found:\n"
        + "\n".join(violations)
        + "\n\nAll stochastic sites must use flock.rng (or a local Generator)."
    )


# ── P10.4: spawn_at tests — cube-velocity law + v0/rng plumbing ─

class TestSpawnAt:
    """P10.4: spawn_at() — position, velocity law, v0/rng plumbing."""

    def test_spawn_position_exact(self, default_config):
        """Spawned bird has the exact position passed."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        target = (123.0, 456.0, 789.0)
        idx = flock.spawn_at(target)
        assert idx >= 0
        np.testing.assert_array_equal(
            flock.positions[idx], np.array(target, dtype=np.float32),
        )

    def test_spawn_velocity_magnitude_bounded_by_v0(self, default_config):
        """Cube-velocity law: |v| ≤ v0 always (limit3 clamp)."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.v0 = 4.0
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        # Spawn many birds, check all velocities are within v0
        for _ in range(50):
            idx = flock.spawn_at((500, 350, 200), v0=cfg.v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            assert speed <= cfg.v0 + 1e-6, (
                f"spawn velocity {speed:.4f} exceeds v0={cfg.v0}"
            )
            assert speed >= 0.0, "spawn velocity should be non-negative"

    def test_spawn_velocity_components_in_range(self, default_config):
        """Cube-velocity law: each component ∈ [-v0, v0]."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.v0 = 3.0
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(123)

        for _ in range(100):
            idx = flock.spawn_at((500, 350, 200), v0=cfg.v0, rng=rng)
            vel = flock.velocities[idx]
            assert (-cfg.v0 - 0.01 <= vel[0] <= cfg.v0 + 0.01), (
                f"vx={vel[0]:.4f} outside [-v0, v0]"
            )
            assert (-cfg.v0 - 0.01 <= vel[1] <= cfg.v0 + 0.01)
            assert (-cfg.v0 - 0.01 <= vel[2] <= cfg.v0 + 0.01)

    def test_spawn_v0_scales_velocity(self, default_config):
        """Higher v0 produces larger velocities on average."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        # Spawn with v0=2.0
        speeds_small = []
        for _ in range(30):
            idx = flock.spawn_at((500, 350, 200), v0=2.0, rng=rng)
            speeds_small.append(float(np.linalg.norm(flock.velocities[idx])))

        rng2 = np.random.default_rng(42)
        flock2 = PhysicsFlock(cfg)
        speeds_large = []
        for _ in range(30):
            idx = flock2.spawn_at((500, 350, 200), v0=4.0, rng=rng2)
            speeds_large.append(float(np.linalg.norm(flock2.velocities[idx])))

        # On average, v0=4.0 should produce ~2× the speed of v0=2.0
        mean_small = np.mean(speeds_small)
        mean_large = np.mean(speeds_large)
        ratio = mean_large / max(mean_small, 0.01)
        assert ratio > 1.6, (
            f"v0=4 mean={mean_large:.3f} vs v0=2 mean={mean_small:.3f}, "
            f"ratio={ratio:.2f} should be >1.6 (~2.0 expected)"
        )

    def test_spawn_rng_deterministic(self, default_config):
        """Same rng seed → identical velocity for spawned bird."""
        cfg = default_config
        cfg.num_boids = 5

        rng1 = np.random.default_rng(42)
        flock1 = PhysicsFlock(cfg)
        idx1 = flock1.spawn_at((100, 200, 300), v0=4.0, rng=rng1)

        rng2 = np.random.default_rng(42)
        flock2 = PhysicsFlock(cfg)
        idx2 = flock2.spawn_at((100, 200, 300), v0=4.0, rng=rng2)

        np.testing.assert_array_equal(
            flock1.velocities[idx1], flock2.velocities[idx2],
            err_msg="Same rng seed must produce identical spawn velocity",
        )

    def test_spawn_rng_default_uses_flock_rng(self, default_config):
        """When rng is not passed, flock.rng is used."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.seed = 42
        flock = PhysicsFlock(cfg)

        # Reset rng to known state and spawn
        flock.rng = np.random.default_rng(42)
        idx1 = flock.spawn_at((100, 200, 300), v0=4.0)  # no rng=...

        # Same state should produce same result
        flock2 = PhysicsFlock(cfg)
        flock2.rng = np.random.default_rng(42)
        idx2 = flock2.spawn_at((100, 200, 300), v0=4.0)

        np.testing.assert_array_equal(
            flock.velocities[idx1], flock2.velocities[idx2],
        )

    def test_spawn_rng_advances_state(self, default_config):
        """Each spawn advances the RNG — two consecutive spawns differ."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        idx1 = flock.spawn_at((500, 350, 200), v0=4.0, rng=rng)
        idx2 = flock.spawn_at((500, 350, 200), v0=4.0, rng=rng)

        # Two spawns from same rng should give different velocities
        assert not np.array_equal(
            flock.velocities[idx1], flock.velocities[idx2],
        ), "Consecutive spawns should produce different velocities"

    def test_spawn_predator_flag(self, default_config):
        """Spawned predator gets is_predator=True."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200), is_predator=True)
        assert bool(flock.is_predator[idx]) is True

    def test_spawn_prey_flag_default(self, default_config):
        """Spawned bird defaults to is_predator=False."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200))
        assert bool(flock.is_predator[idx]) is False

    def test_spawn_reuses_inactive_slot(self, default_config):
        """spawn_at activates an inactive slot before extending."""
        cfg = default_config
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        # Deactivate bird at index 3
        flock.active[3] = False
        cap_before = flock.N_capacity

        idx = flock.spawn_at((500, 350, 200))
        assert idx == 3, f"Should reuse inactive slot 3, got {idx}"
        assert flock.N_capacity == cap_before, "No extension needed"

    def test_spawn_extends_capacity(self, default_config):
        """spawn_at extends arrays when all slots are active."""
        cfg = default_config
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        cap_before = flock.N_capacity

        idx = flock.spawn_at((500, 350, 200))
        assert idx >= 0
        assert flock.N_capacity > cap_before, (
            "Capacity should extend when all slots active"
        )

    def test_spawn_acceleration_zero(self, default_config):
        """Spawned bird starts with zero acceleration."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200))
        np.testing.assert_array_equal(
            flock.accelerations[idx], np.zeros(3, dtype=np.float32),
        )

    def test_spawn_seed_assigned(self, default_config):
        """Spawned bird gets a seed value in [0, 1)."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200))
        s = float(flock.seeds[idx])
        assert 0.0 <= s < 1.0, f"seed {s} not in [0, 1)"

    def test_spawn_velocity_not_all_same_direction(self, default_config):
        """Cube-velocity produces varied directions, not just one axis."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        directions = []
        for _ in range(20):
            idx = flock.spawn_at((500, 350, 200), v0=4.0, rng=rng)
            vel = flock.velocities[idx]
            mag = np.linalg.norm(vel)
            if mag > 0:
                directions.append(vel / mag)

        # Check there's variation in at least one component
        dirs = np.array(directions)
        for axis in range(3):
            std = float(np.std(dirs[:, axis]))
            assert std > 0.05, (
                f"Axis {axis}: std={std:.4f} — all birds facing same direction?"
            )


# ── P10.4: Cube-velocity law — exact formula verification ─────

class TestCubeVelocityLaw:
    """P10.4: limit3((U³ − 0.5) · 2v0, v0) — exact formula, clamping, distribution."""

    def test_exact_formula_reproduction(self, default_config):
        """spawn_at velocity equals manual limit3((U−0.5)·2v0, v0) computation."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        v0 = 3.0

        # Snapshot RNG state, then manually compute what spawn_at should produce
        rng_snap = np.random.default_rng(42)
        U = rng_snap.uniform(0, 1, 3).astype(np.float32)
        raw_vel = (U ** 3 - 0.5) * 2.0 * v0
        mag = float(np.linalg.norm(raw_vel))
        if mag > v0:
            raw_vel *= v0 / mag
        expected = raw_vel.copy()

        # Now call spawn_at with the same RNG state
        rng2 = np.random.default_rng(42)
        idx = flock.spawn_at((100, 200, 300), v0=v0, rng=rng2)
        actual = flock.velocities[idx]

        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Cube-velocity law mismatch: expected={expected}, got={actual}"
        )

    def test_formula_with_v0_one(self, default_config):
        """With v0=1.0, raw velocity is in [-1,1]³ before clamp."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(99)

        # Spawn several birds with v0=1.0
        for _ in range(50):
            idx = flock.spawn_at((500, 350, 200), v0=1.0, rng=rng)
            vel = flock.velocities[idx]
            # Each component must be in [-1, 1] (before clamp it's in [-v0, v0])
            assert -1.01 <= vel[0] <= 1.01
            assert -1.01 <= vel[1] <= 1.01
            assert -1.01 <= vel[2] <= 1.01
            # Magnitude must be ≤ 1 (after limit3 clamp)
            assert float(np.linalg.norm(vel)) <= 1.01

    def test_limit3_clamping_fires(self, default_config):
        """limit3 clamp is exercised — some velocities reach exactly |v|=v0."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        v0 = 2.0

        # The cube [-v0,v0]³ has diagonal sqrt(3)*v0 ≈ 3.46 > v0=2.
        # ~43% of uniform-cube samples will exceed v0 and get clamped.
        # After clamping, their magnitude is exactly v0.
        n_clamped = 0
        for _ in range(200):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            if abs(speed - v0) < 0.001:
                n_clamped += 1

        # At least some should be clamped (probability of none in 200 is < 1e-50)
        assert n_clamped > 0, (
            "limit3 clamp never fired in 200 spawns — "
            "expected ~86 clamps (43%%), got 0"
        )

    def test_clamped_velocity_magnitude_is_exactly_v0(self, default_config):
        """When raw_vel exceeds v0, the clamped velocity has |v| ≈ v0 (float32)."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(777)
        v0 = 1.5

        # Spawn many, collect clamped ones
        clamped_speeds = []
        for _ in range(500):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            if abs(speed - v0) < 0.01:
                clamped_speeds.append(speed)

        assert len(clamped_speeds) > 10, (
            f"Expected many clamped velocities, got {len(clamped_speeds)}"
        )
        # All clamped speeds should be very close to v0 (allow float32 epsilon)
        for s in clamped_speeds:
            assert abs(s - v0) < 0.01, (
                f"Clamped speed {s:.6f} not close enough to v0={v0}"
            )

    def test_unclamped_velocity_below_v0(self, default_config):
        """When raw_vel mag ≤ v0, the velocity is left unchanged (no clamp)."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        v0 = 10.0  # Large v0 makes clamping rare (cube [-10,10]³, diag≈17.3)

        unclamped_count = 0
        for _ in range(200):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            # Not at exactly v0 → was not clamped
            if abs(speed - v0) > 0.01:
                unclamped_count += 1
                # Verify the components are each within [-v0, v0]
                vel = flock.velocities[idx]
                assert -v0 - 0.01 <= vel[0] <= v0 + 0.01
                assert -v0 - 0.01 <= vel[1] <= v0 + 0.01
                assert -v0 - 0.01 <= vel[2] <= v0 + 0.01

        assert unclamped_count > 0, (
            "With large v0, some velocities should NOT be clamped"
        )

    def test_distribution_is_cube_law_before_clamp(self, default_config):
        """Pre-clamp raw_vel is (U³−0.5)·2v0, bounded by [-v0, v0]³ (D20).

        We can verify this externally by spawning with a known RNG,
        capturing the raw uniform values, and applying the same transform.
        """
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        v0 = 4.0

        # Reconstruct: spawn_at calls r.uniform(0,1,3), then transforms.
        # We intercept by using the same RNG and comparing the pre-transform.
        rng = np.random.default_rng(42)
        U = rng.uniform(0, 1, 3).astype(np.float32)
        raw = (U ** 3 - 0.5) * 2.0 * v0  # D20 cube law, in [-v0, v0]³

        # Now spawn with the same rng
        rng2 = np.random.default_rng(42)
        idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng2)
        vel = flock.velocities[idx]

        # If raw was not clamped (mag ≤ v0), vel == raw exactly
        raw_mag = float(np.linalg.norm(raw))
        if raw_mag <= v0:
            np.testing.assert_array_almost_equal(vel, raw, decimal=6)
        else:
            # If clamped, vel = raw * (v0 / raw_mag)
            np.testing.assert_array_almost_equal(vel, raw * (v0 / raw_mag), decimal=6)

    def test_cube_law_mean_bias(self, default_config):
        """Cube-law mean per component is approximately −0.5·v0.

        With (U³−0.5)·2v0, each component has theoretical mean
        (0.25−0.5)·2v0 = −0.5·v0.  This is the cube-law's systematic
        bias: pushing mass toward ±v0 concentrates at −v0 end.
        """
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        v0 = 5.0

        vels = []
        for _ in range(500):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            vels.append(flock.velocities[idx])

        mean_vel = np.mean(vels, axis=0)
        # Cube-law mean per component: (0.25−0.5)·2v0 = −0.5·v0 = −2.5 for v0=5.
        # limit3 clamping shifts mean toward 0 — observed ≈ −2.0 for v0=5.
        # Use wide bound to accommodate clamp + sampling variance.
        for axis in range(3):
            assert -3.5 < float(mean_vel[axis]) < -0.5, (
                f"Axis {axis}: mean={float(mean_vel[axis]):.3f} "
                f"should be negative (cube-law bias per component)"
            )


# ── P10.4: Engine plumbing — v0/rng flow from engine to spawn_at ─

class TestSpawnAtEnginePlumbing:
    """P10.4: engine.enqueue_spawn + drain_commands passes v0 and rng correctly."""

    @pytest.fixture
    def _engine(self) -> tuple:
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.v0 = 3.0
        engine = SimulationEngine(cfg)
        return engine, cfg

    def test_engine_spawn_uses_config_v0(self, _engine):
        """Engine passes config.v0 to spawn_at — velocity bounded by v0."""
        engine, cfg = _engine
        n_before = engine.flock.N_active

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 1
        # Find the newly spawned bird (highest active index)
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        speed = float(np.linalg.norm(engine.flock.velocities[new_bird]))
        assert speed <= cfg.v0 + 1e-6, (
            f"Spawned velocity {speed:.4f} exceeds config.v0={cfg.v0}"
        )

    def test_engine_spawn_position_is_exact(self, _engine):
        """Engine spawn places bird at exact enqueued position."""
        engine, cfg = _engine
        target = (123.0, 456.0, 789.0)
        engine.enqueue_spawn(target)
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[new_bird],
            np.array(target, dtype=np.float32),
        )

    def test_engine_spawn_predator_flag(self, _engine):
        """Engine spawn with is_predator=True sets the flag."""
        engine, cfg = _engine
        engine.enqueue_spawn((500, 350, 200), is_predator=True)
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert bool(engine.flock.is_predator[new_bird]) is True

    def test_config_v0_change_affects_spawn(self, _engine):
        """Changing config.v0 before drain_commands affects spawn velocity."""
        engine, cfg = _engine

        # Spawn with v0=3.0
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_3 = active_idx[-1]
        speed_3 = float(np.linalg.norm(engine.flock.velocities[bird_v0_3]))

        # Change config.v0 to a much lower value
        cfg.v0 = 0.5
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_05 = active_idx[-1]
        speed_05 = float(np.linalg.norm(engine.flock.velocities[bird_v0_05]))

        # Second spawn should have lower velocity (bounded by new v0=0.5)
        assert speed_05 <= 0.51, (
            f"After v0→0.5, spawned speed {speed_05:.4f} should be ≤ 0.5"
        )
        # First spawn should have higher velocity (bounded by old v0=3.0)
        assert speed_3 <= 3.01

    def test_engine_spawn_uses_flock_rng(self, _engine):
        """Engine drain_commands passes flock.rng to spawn_at."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        # Two engines with same seed spawn same velocity
        cfg1 = SimConfig()
        cfg1.num_boids = 10
        cfg1.seed = 42
        e1 = SimulationEngine(cfg1)
        e1.enqueue_spawn((500, 350, 200))
        e1.drain_commands()

        cfg2 = SimConfig()
        cfg2.num_boids = 10
        cfg2.seed = 42
        e2 = SimulationEngine(cfg2)
        e2.enqueue_spawn((500, 350, 200))
        e2.drain_commands()

        # Both should produce the same velocity (same seed → same flock.rng state)
        a1 = np.where(e1.flock.active)[0][-1]
        a2 = np.where(e2.flock.active)[0][-1]
        np.testing.assert_array_equal(
            e1.flock.velocities[a1], e2.flock.velocities[a2],
            err_msg="Same seed must produce identical spawn velocity via engine"
        )

    def test_engine_spawn_different_seeds_diverge(self, _engine):
        """Different seeds → different spawn velocities through engine."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg1 = SimConfig()
        cfg1.num_boids = 10
        cfg1.seed = 42
        e1 = SimulationEngine(cfg1)
        e1.enqueue_spawn((500, 350, 200))
        e1.drain_commands()

        cfg2 = SimConfig()
        cfg2.num_boids = 10
        cfg2.seed = 99
        e2 = SimulationEngine(cfg2)
        e2.enqueue_spawn((500, 350, 200))
        e2.drain_commands()

        a1 = np.where(e1.flock.active)[0][-1]
        a2 = np.where(e2.flock.active)[0][-1]
        assert not np.array_equal(
            e1.flock.velocities[a1], e2.flock.velocities[a2],
        ), "Different seeds must produce different spawn velocities"

    def test_engine_multiple_spawns_in_one_drain(self, _engine):
        """Multiple enqueued spawns are all processed in one drain_commands."""
        engine, cfg = _engine
        n_before = engine.flock.N_active

        engine.enqueue_spawn((100, 200, 300))
        engine.enqueue_spawn((400, 500, 600))
        engine.enqueue_spawn((700, 800, 900))
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 3
        # Last 3 active birds should have the enqueued positions
        active_idx = np.where(engine.flock.active)[0]
        b1, b2, b3 = active_idx[-3], active_idx[-2], active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[b1], np.array([100, 200, 300], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[b2], np.array([400, 500, 600], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[b3], np.array([700, 800, 900], dtype=np.float32),
        )

    def test_engine_spawn_rng_advances_per_spawn(self, _engine):
        """Each spawn via engine advances flock.rng — two spawns differ."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.enqueue_spawn((500, 350, 200))  # same position
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        b1, b2 = active_idx[-2], active_idx[-1]

        assert not np.array_equal(
            engine.flock.velocities[b1], engine.flock.velocities[b2],
        ), "Two consecutive spawns through engine must differ (RNG advances)"

    def test_engine_spawn_updates_num_boids(self, _engine):
        """After drain_commands, config.num_boids reflects new N_active."""
        engine, cfg = _engine
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        assert cfg.num_boids == engine.flock.N_active


# ── D6 + D20: Seed=0 spawning determinism (cross-cutting) ─────


def test_same_seed_zero_with_spawning_deterministic():
    """D6+D20: Two engines with seed=0 + identical spawns → bit-identical.

    Before D6: seed=0 was conflated with None, causing non-deterministic
    spawning across runs. After D6+D20: seed=0 is a valid deterministic
    seed, and spawn_at uses the cube-velocity law deterministically.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg1 = SimConfig()
    cfg1.seed = 0
    cfg1.num_boids = 10
    cfg1.mode = "spatial"

    cfg2 = SimConfig()
    cfg2.seed = 0
    cfg2.num_boids = 10
    cfg2.mode = "spatial"

    e1 = SimulationEngine(cfg1)
    e2 = SimulationEngine(cfg2)

    # Run a few steps then spawn birds (D20 cube-law velocity)
    for _ in range(5):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    # Identical spawn operations on both engines
    e1.enqueue_spawn((100, 200, 300))
    e2.enqueue_spawn((100, 200, 300))
    e1.enqueue_spawn((400, 500, 600))
    e2.enqueue_spawn((400, 500, 600))
    e1.drain_commands()
    e2.drain_commands()

    # Both engines must be bit-identical after spawning
    np.testing.assert_array_equal(
        e1.flock.positions, e2.flock.positions,
        err_msg="D6+D20: seed=0 with spawning must be deterministic"
    )
    np.testing.assert_array_equal(
        e1.flock.velocities, e2.flock.velocities,
        err_msg="D6+D20: seed=0 spawn velocities must be bit-identical"
    )


# ── D6: Seed semantics (0 ≠ None) ────────────────────────────────


class TestD6SeedSemantics:
    """D6: seed=0 determinism is distinct from seed=None (fresh entropy).

    The bug (now fixed) was:
        default_rng(config.seed if config.seed else 0)
    which conflated seed=0 with seed=None because 0 is falsy.
    The fix is:
        default_rng(config.seed)
    numpy interprets None correctly as "fresh entropy" and 0 as
    "deterministic seed 0".
    """

    def test_seed_zero_is_deterministic(self, default_config):
        """D6: Two flocks with seed=0 produce identical state."""
        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 30
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = 0
        cfg2.num_boids = 30
        flock2 = PhysicsFlock(cfg2)

        # Seed 0 should be honoured — both flocks must be bit-identical.
        np.testing.assert_array_equal(flock1.positions, flock2.positions)
        np.testing.assert_array_equal(flock1.velocities, flock2.velocities)
        np.testing.assert_array_equal(flock1.seeds, flock2.seeds)

    def test_seed_zero_diverges_from_seed_none(self, default_config):
        """D6: seed=0 produces different output than seed=None.

        If seed=None were being replaced with seed=0 (the original bug),
        both flocks would be identical.  They must differ."""
        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 30
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = None
        cfg2.num_boids = 30
        flock2 = PhysicsFlock(cfg2)

        # seed=None must produce fresh entropy, not seed=0.
        # The probability of two 90-dimensional random draws colliding
        # is astronomically small, so a single comparison suffices.
        assert not np.array_equal(flock1.positions, flock2.positions), (
            "seed=0 and seed=None must produce different positions"
        )
        assert not np.array_equal(flock1.velocities, flock2.velocities), (
            "seed=0 and seed=None must produce different velocities"
        )

    def test_seed_none_is_nondeterministic(self, default_config):
        """D6: Two flocks with seed=None produce different state."""
        cfg1 = default_config
        cfg1.seed = None
        cfg1.num_boids = 30
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = None
        cfg2.num_boids = 30
        flock2 = PhysicsFlock(cfg2)

        # seed=None should give fresh entropy each time.
        assert not np.array_equal(flock1.positions, flock2.positions), (
            "seed=None must produce fresh entropy each call"
        )

    def test_seed_zero_determinism_persists_after_steps(
        self, default_config,
    ):
        """D6: seed=0 remains deterministic after multiple integration steps."""
        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 20
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = 0
        cfg2.num_boids = 20
        flock2 = PhysicsFlock(cfg2)

        # Step both flocks the same number of times
        for _ in range(5):
            _step_flock(flock1, cfg1, 1.0 / 60.0)
            _step_flock(flock2, cfg2, 1.0 / 60.0)

        np.testing.assert_array_equal(flock1.positions, flock2.positions)

    def test_seed_via_engine_zero_vs_none_diverge(
        self, default_config,
    ):
        """D6: At engine level, seed=0 and seed=None diverge after stepping."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 20
        eng1 = SimulationEngine(cfg1)

        cfg2 = default_config
        cfg2.seed = None
        cfg2.num_boids = 20
        eng2 = SimulationEngine(cfg2)

        # Step both 10 times in headless mode
        eng1.step(1.0 / 60.0)
        eng2.step(1.0 / 60.0)

        # seed=0 and seed=None must produce different trajectories
        assert not np.array_equal(
            eng1.flock.positions, eng2.flock.positions,
        ), "seed=0 and seed=None must diverge at engine level"


def test_boundary_radius_factor_scales_sphere_clamp(default_config):
    """C3: boundary_radius_factor scales the effective sphere boundary."""
    cfg = default_config
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 100.0
    cfg.num_boids = 1
    cfg.boundary_radius_factor = 2.0

    flock = PhysicsFlock(cfg)
    center = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    # Place the bird far outside even the scaled radius, at rest.
    flock.positions[0] = center + np.array([500.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[0] = 0.0
    flock.accelerations[0] = 0.0

    flock.integrate(cfg, dt=1.0 / 60.0)

    dist = float(np.linalg.norm(flock.positions[0] - center))
    assert dist == pytest.approx(200.0, abs=1e-3), (
        f"Expected hard clamp at radius*factor=200, got {dist:.3f}"
    )


def test_boundary_radius_factor_default_is_noop(default_config):
    """C3: boundary_radius_factor=1.0 (default) matches unscaled behaviour."""
    cfg = default_config
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 100.0
    cfg.num_boids = 1
    assert cfg.boundary_radius_factor == 1.0

    flock = PhysicsFlock(cfg)
    center = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    flock.positions[0] = center + np.array([500.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[0] = 0.0
    flock.accelerations[0] = 0.0

    flock.integrate(cfg, dt=1.0 / 60.0)

    dist = float(np.linalg.norm(flock.positions[0] - center))
    assert dist == pytest.approx(100.0, abs=1e-3)
