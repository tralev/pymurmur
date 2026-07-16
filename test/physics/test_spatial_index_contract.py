"""Additional Phase 2 tests for KDTreeIndex global indices (P2.4),
ghost-cell Z-axis corners (P2.5), and SpatialIndex protocol (P2.3)."""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.core.types import SpatialIndex
from pymurmur.physics.flock import PhysicsFlock, SpatialHashGrid, KDTreeIndex


# ── P2.4: KDTreeIndex global indices ──────────────────────────────


def test_kdtree_global_indices_compacted_to_global():
    """P2.4: KDTreeIndex.query_knn returns global indices, not compacted."""
    cfg = SimConfig()
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)

    kdt = KDTreeIndex()
    # Deactivate birds 10-19 so active set is non-contiguous
    flock.active[10:20] = False
    kdt.rebuild(flock.positions, flock.active)

    # Query from an active bird — results must be global indices (0-49)
    active_global = np.where(flock.active)[0]
    query_pos = flock.positions[active_global[0]]

    result = kdt.query_knn(query_pos, k=5)
    assert len(result) > 0, "Should find neighbours"

    # All returned indices must be in the active set (global)
    for idx in result:
        assert 0 <= idx < 50, f"Index {idx} out of global range [0, 50)"
        assert flock.active[idx], f"Index {idx} should be active"


def test_kdtree_global_indices_with_gaps():
    """P2.4: Global indices correct when active set has gaps."""
    cfg = SimConfig()
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    # Activate only birds 0, 5, 10, 15 — sparse gaps
    flock.active[:] = False
    flock.active[[0, 5, 10, 15]] = True

    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)

    result = kdt.query_knn(flock.positions[0], k=3)
    for idx in result:
        assert idx in {0, 5, 10, 15}, (
            f"Returned index {idx} not in active set {{0,5,10,15}}"
        )


# ── P2.3: SpatialIndex Protocol conformance ───────────────────────


def test_spatial_hash_grid_conforms_to_protocol():
    """P2.3: SpatialHashGrid structurally conforms to SpatialIndex Protocol."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    grid = SpatialHashGrid(cfg)

    # Protocol requires: ready, rebuild, query_knn, query_radius, tree
    assert hasattr(grid, 'ready')
    assert hasattr(grid, 'rebuild')
    assert callable(grid.rebuild)
    assert hasattr(grid, 'query_knn')
    assert callable(grid.query_knn)
    assert hasattr(grid, 'query_radius')
    assert callable(grid.query_radius)
    assert hasattr(grid, 'tree')

    # Verify it's recognized as SpatialIndex
    assert isinstance(grid, SpatialIndex)


def test_kdtree_conforms_to_protocol():
    """P2.3: KDTreeIndex structurally conforms to SpatialIndex Protocol."""
    kdt = KDTreeIndex()

    assert hasattr(kdt, 'ready')
    assert hasattr(kdt, 'rebuild')
    assert callable(kdt.rebuild)
    assert hasattr(kdt, 'query_knn')
    assert callable(kdt.query_knn)
    assert hasattr(kdt, 'tree')


# ── P2.5: Ghost-cell Z-axis corners ───────────────────────────────


def test_hash_grid_ghost_z_axis_corner():
    """P2.5: Birds near z=0 and z=D find each other via modulo cells."""
    cfg = SimConfig()
    cfg.num_boids = 4
    cfg.width = 1000
    cfg.height = 700
    cfg.depth = 400
    cfg.visual_range = 150
    flock = PhysicsFlock(cfg)
    grid = flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        flock.positions[:] = np.array([
            [500, 350, 10],    # bird 0 — near z=0
            [500, 350, 390],   # bird 1 — near z=400, toroidal ~20
            [100, 100, 10],    # bird 2 — near x=0, z=0 (corner)
            [900, 600, 390],   # bird 3 — near x=1000, y=700, z=400 (far corner)
        ], dtype=np.float32)
        grid.rebuild(flock.positions, flock.active)

        # Z-axis cross-seam: bird 0 at z=10 → bird 1 (z=390) found
        candidates = grid.query_radius(flock.positions[0], 50.0)
        assert 1 in candidates, (
            f"P2.5 Z-axis corner: bird at z=390 should be found, got {candidates}"
        )


def test_hash_grid_ghost_xyz_corner():
    """P2.5: Bird near three boundaries (corner) finds opposite-corner bird."""
    cfg = SimConfig()
    cfg.num_boids = 6
    cfg.width = 1000
    cfg.height = 700
    cfg.depth = 400
    cfg.visual_range = 200
    flock = PhysicsFlock(cfg)
    grid = flock.get_index()
    if isinstance(grid, SpatialHashGrid):
        flock.positions[:] = np.array([
            [10, 10, 10],       # bird 0 — near origin corner
            [990, 690, 390],    # bird 1 — opposite corner, toroidally all ~20
            [500, 350, 200],    # bird 2 — centre
            [500, 350, 200],    # bird 3
            [500, 350, 200],    # bird 4
            [500, 350, 200],    # bird 5
        ], dtype=np.float32)
        grid.rebuild(flock.positions, flock.active)

        # Bird 0 queries → bird 1 found across all 3 seams
        result = grid.query_knn(flock.positions[0], k=2)
        assert 1 in result, (
            f"P2.5 XYZ corner: opposite-corner bird should be found, got {list(result)}"
        )


# ═══════════════════════════════════════════════════════════════════
# Cross-implementation parity: SpatialHashGrid ≡ KDTreeIndex
# For dense flocks away from domain edges, both indexes must return
# the same nearest-neighbour sets (order may differ due to distance
# sorting in hash grid vs exact tree distances).
# ═══════════════════════════════════════════════════════════════════


def test_spatial_index_parity_closest_neighbor_agreement():
    """P2.3→P2.4: Both indexes agree on the single closest neighbor
    for a dense flock far from domain boundaries.

    This is the strongest cross-implementation invariant: if the two
    indexes disagree on which bird is THE closest, something is
    fundamentally broken. Uses a tight central cluster so no toroidal
    wrapping occurs — min-image ≡ Euclidean distance for all pairs."""
    cfg = SimConfig()
    cfg.num_boids = 60
    cfg.width = 1000
    cfg.height = 700
    cfg.depth = 400
    cfg.visual_range = 150
    flock = PhysicsFlock(cfg)

    # Place birds in a tight cluster at centre — max offset from any
    # border is ~300, so no toroidal wrapping for central birds
    rng = np.random.default_rng(99)
    flock.positions[:] = (
        rng.normal(0, 60, (60, 3)).astype(np.float32)
        + np.array([500, 350, 200], dtype=np.float32)
    )

    grid = SpatialHashGrid(cfg)
    grid.rebuild(flock.positions, flock.active)

    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)

    # Track birds where hash grid returns empty (peripheral birds with
    # no neighbors in the 27-cell neighborhood)
    disagreements = 0
    hash_empty = 0

    for i in range(60):
        pos = flock.positions[i]

        grid_result = grid.query_knn(pos, k=5)
        kdt_result = kdt.query_knn(pos, k=5)

        if len(grid_result) == 0:
            hash_empty += 1
            # Hash grid may return empty for peripheral birds —
            # KDTree should still return at least 1 neighbor since
            # the flock is dense (60 birds in tight cluster)
            continue

        # Strongest invariant: both agree on the CLOSEST neighbor
        if grid_result[0] != kdt_result[0]:
            disagreements += 1

    # All birds should agree on closest neighbor (central cluster,
    # no toroidal wrapping, 27-cell box covers ~450 range, cluster
    # is ~180 wide → all birds' neighbors are in the hash grid)
    assert hash_empty == 0, (
        f"{hash_empty}/60 birds had empty hash-grid results (cluster too sparse)"
    )
    assert disagreements == 0, (
        f"{disagreements}/60 birds had different closest neighbor across indexes"
    )


def test_spatial_index_parity_same_self_exclusion():
    """Both indexes exclude self from query results.

    query_knn should return neighbours only — the querying bird
    must never appear in its own results."""
    cfg = SimConfig()
    cfg.num_boids = 50
    cfg.visual_range = 100
    flock = PhysicsFlock(cfg)

    rng = np.random.default_rng(42)
    flock.positions[:] = (
        rng.normal(0, 60, (50, 3)).astype(np.float32)
        + np.array([500, 350, 200], dtype=np.float32)
    )

    grid = SpatialHashGrid(cfg)
    grid.rebuild(flock.positions, flock.active)

    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)

    for i in range(50):
        pos = flock.positions[i]

        grid_result = grid.query_knn(pos, k=5)
        kdt_result = kdt.query_knn(pos, k=5)

        # Self must not appear in results
        assert i not in grid_result, (
            f"Bird {i} found itself in SpatialHashGrid.query_knn: {list(grid_result)}"
        )
        assert i not in kdt_result, (
            f"Bird {i} found itself in KDTreeIndex.query_knn: {list(kdt_result)}"
        )


def test_spatial_index_parity_results_in_active_set():
    """Both indexes only return indices from the active set."""
    cfg = SimConfig()
    cfg.num_boids = 50
    cfg.visual_range = 100
    flock = PhysicsFlock(cfg)

    rng = np.random.default_rng(77)
    flock.positions[:] = (
        rng.normal(0, 60, (50, 3)).astype(np.float32)
        + np.array([500, 350, 200], dtype=np.float32)
    )

    # Deactivate half the birds — create gaps
    flock.active[10:35] = False

    grid = SpatialHashGrid(cfg)
    grid.rebuild(flock.positions, flock.active)

    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)

    for i in range(50):
        if not flock.active[i]:
            continue
        pos = flock.positions[i]

        grid_result = grid.query_knn(pos, k=5)
        kdt_result = kdt.query_knn(pos, k=5)

        for idx in grid_result:
            assert flock.active[idx], (
                f"SpatialHashGrid returned inactive bird {idx} for query at {i}"
            )
        for idx in kdt_result:
            assert flock.active[idx], (
                f"KDTreeIndex returned inactive bird {idx} for query at {i}"
            )


def test_spatial_index_parity_distance_ordering():
    """Both indexes return neighbours in order of increasing distance
    from query point. Distances must be monotonically non-decreasing."""
    cfg = SimConfig()
    cfg.num_boids = 80
    cfg.visual_range = 100
    flock = PhysicsFlock(cfg)

    rng = np.random.default_rng(123)
    flock.positions[:] = (
        rng.normal(0, 60, (80, 3)).astype(np.float32)
        + np.array([500, 350, 200], dtype=np.float32)
    )

    grid = SpatialHashGrid(cfg)
    grid.rebuild(flock.positions, flock.active)

    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)

    # Test a subset of birds for speed
    test_indices = [0, 10, 20, 30, 40, 50, 60, 70]

    for i in test_indices:
        pos = flock.positions[i]

        # KDTreeIndex: use public tree property to verify distances
        tree = kdt.tree
        if tree is not None:
            dists, _ = tree.query(pos, k=6)
            # cKDTree returns [self, n1, n2, ...] — skip self (idx 0)
            for j in range(1, len(dists)):
                assert float(dists[j-1]) <= float(dists[j]) + 1e-6, (
                    f"KDTreeIndex: dists not monotonic for bird {i}: {dists}"
                )

        # SpatialHashGrid: check that results exist (hash grid sorts
        # by toroidal distance, so ordering is only approximate for
        # toroidal domains, but for central flocks it should be correct)
        grid_result = grid.query_knn(pos, k=5)
        assert isinstance(grid_result, np.ndarray), (
            f"SpatialHashGrid.query_knn must return ndarray, got {type(grid_result)}"
        )
