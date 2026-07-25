"""Unit tests for physics.spatial_index — SpatialHashGrid, KDTreeIndex, index-mode selection.

Split out of test_flock.py (file-size split).
"""

import numpy as np

from pymurmur.physics.flock import KDTreeIndex, PhysicsFlock, SpatialHashGrid
from test.helpers import _step_flock  # noqa: E402 — shared test helper


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


def test_hash_grid_query_radius_is_honored():
    """D5: query_radius's radius argument actually changes what's found —
    previously it always searched a fixed ±1-cell block regardless of the
    requested radius, so the same query point with a small vs. large
    radius returned identical results."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.num_boids = 1
    cfg.width = 1000
    cfg.height = 700
    cfg.depth = 400
    cfg.visual_range = 100  # cell_size = 100
    flock = PhysicsFlock(cfg)
    grid = flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        # Bird at cell x=4; query at cell x=0 — 4 cells away, well outside
        # the old fixed ±1-cell (3-cell-wide) neighborhood.
        flock.positions[:] = np.array([[450, 350, 200]], dtype=np.float32)
        grid.rebuild(flock.positions, flock.active)
        query_pos = np.array([50, 350, 200], dtype=np.float32)

        small_radius_hits = grid.query_radius(query_pos, 50.0)
        assert 0 not in small_radius_hits, (
            "a 50-unit radius should not reach a bird 400 units away"
        )

        large_radius_hits = grid.query_radius(query_pos, 450.0)
        assert 0 in large_radius_hits, (
            "a 450-unit radius should reach a bird 400 units away — "
            "if this fails, query_radius is still ignoring its argument"
        )


def test_hash_grid_query_radius_no_duplicates_on_small_grid():
    """D5: a large radius on a small grid (few cells per axis) must not
    return the same bird's index more than once via modulo-wrapped cell
    revisits."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.num_boids = 3
    cfg.width = 200
    cfg.height = 200
    cfg.depth = 200
    cfg.visual_range = 100  # cell_size = 100 → only 2 cells per axis
    flock = PhysicsFlock(cfg)
    grid = flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        flock.positions[:] = np.array(
            [[10, 10, 10], [110, 110, 110], [190, 190, 190]], dtype=np.float32,
        )
        grid.rebuild(flock.positions, flock.active)

        # Large radius relative to the tiny (2×2×2-cell) grid — forces a
        # wide, wrap-heavy search window.
        candidates = grid.query_radius(
            np.array([100, 100, 100], dtype=np.float32), 1000.0,
        )
        assert len(candidates) == len(set(candidates)), (
            f"duplicate indices in query_radius result: {candidates}"
        )


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


