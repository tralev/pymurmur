"""P4 — Spatial Variants tests.

Covers _query_neighbors, _maybe_perception_filter, and _predator_escape
from pymurmur/physics/forces/spatial.py.
"""

import numpy as np

# ── Helpers ──────────────────────────────────────────────────────

def _make_fake_index(positions):
    """Create a minimal SpatialIndex-compatible object wrapping a KDTree."""
    from scipy.spatial import cKDTree

    class FakeIndex:
        ready = True

        def __init__(self, tree):
            self.tree = tree

    return FakeIndex(cKDTree(positions))


def _make_config(**overrides):
    """Build a minimal SimConfig-like object for spatial variant tests."""
    class FakeConfig:
        topological_cap = 50
        visual_range = 70.0
        influence_count = 7
        noise_scale = 0.05
        separation_weight = 0.1
        alignment_weight = 0.1
        cohesion_weight = 0.1
        max_force = 0.15
        predator_escape_factor = 100.0
        predator_accel_boost = 1.4
        v0 = 4.0

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    return FakeConfig(**overrides)


# ── _query_neighbors ────────────────────────────────────────────

class TestQueryNeighbors:
    """_query_neighbors — hybrid metric+topological neighbour queries."""

    def test_returns_zero_array_with_few_active(self):
        """n_active < 2 → returns (N, 0) empty array."""
        from pymurmur.physics.forces.spatial import _query_neighbors

        positions = np.random.default_rng(0).uniform(0, 500, (10, 3)).astype(np.float32)
        active = np.zeros(10, dtype=bool)
        active[0] = True  # only 1 active
        active_idx = np.where(active)[0]
        index = _make_fake_index(positions[active_idx])
        cfg = _make_config()

        result = _query_neighbors(positions, active, index, cfg)
        assert result.shape == (10, 0)

    def test_returns_valid_neighbor_indices(self):
        """10 active birds → per-bird neighbor idx with correct capacity."""
        from pymurmur.physics.forces.spatial import _query_neighbors

        positions = np.random.default_rng(1).uniform(0, 50, (10, 3)).astype(np.float32)
        active = np.ones(10, dtype=bool)
        active_idx = np.where(active)[0]
        index = _make_fake_index(positions[active_idx])
        cfg = _make_config()

        result = _query_neighbors(positions, active, index, cfg)
        assert result.shape[0] == 10  # N_capacity rows
        assert result.shape[1] > 0    # at least one neighbour column
        # Active rows should have non-zero entries (neighbour global indices)
        assert (result[active] > 0).any()

    def test_inactive_rows_are_zero_filled(self):
        """Inactive birds have zero-filled neighbour rows."""
        from pymurmur.physics.forces.spatial import _query_neighbors

        positions = np.random.default_rng(2).uniform(0, 500, (10, 3)).astype(np.float32)
        active = np.array([True] * 5 + [False] * 5, dtype=bool)
        active_idx = np.where(active)[0]
        index = _make_fake_index(positions[active_idx])
        cfg = _make_config()

        result = _query_neighbors(positions, active, index, cfg)
        assert np.all(result[~active] == 0)

    def test_influence_count_caps_neighbors(self):
        """influence_count=2 → each bird has max 2 accepted neighbours."""
        from pymurmur.physics.forces.spatial import _query_neighbors

        positions = np.random.default_rng(3).uniform(0, 500, (20, 3)).astype(np.float32)
        active = np.ones(20, dtype=bool)
        active_idx = np.where(active)[0]
        index = _make_fake_index(positions[active_idx])
        cfg = _make_config(influence_count=2, topological_cap=20, visual_range=1000.0)

        result = _query_neighbors(positions, active, index, cfg)
        # After hybrid filter, each active row should have ≤ influence_count
        # non-zero entries (excl. self-reference)
        for i in np.where(active)[0]:
            nonzero = (result[i] > 0).sum()
            assert nonzero <= 2, f"bird {i} has {nonzero} neighbours, expected ≤ 2"

    def test_visual_range_filters_distant_birds(self):
        """visual_range=1.0 → only very close birds appear as neighbours."""
        from pymurmur.physics.forces.spatial import _query_neighbors

        # Tight cluster: 5 birds within 10 units, 5 birds far away
        positions = np.zeros((10, 3), dtype=np.float32)
        positions[:5] = np.random.default_rng(4).uniform(-5, 5, (5, 3)).astype(np.float32)
        positions[5:] = np.random.default_rng(5).uniform(400, 500, (5, 3)).astype(np.float32)
        active = np.ones(10, dtype=bool)
        active_idx = np.where(active)[0]
        index = _make_fake_index(positions[active_idx])
        cfg = _make_config(visual_range=20.0, influence_count=5)

        result = _query_neighbors(positions, active, index, cfg)
        # Birds 0-4 should only have neighbours among themselves (indices 0-4)
        for i in range(5):
            nbrs = result[i]
            nonzero = nbrs[nbrs > 0]
            assert len(nonzero) > 0, f"bird {i} should have close neighbours"
            assert all(j < 5 for j in nonzero), (
                f"bird {i} has distant neighbour: {nonzero}"
            )


# ── _maybe_perception_filter ────────────────────────────────────

class TestPerceptionFilter:
    """_maybe_perception_filter — distance + cone perception gating."""

    def test_disabled_filters_return_original_unchanged(self):
        """max_dist=0 and cos_angle=-1 → returns neighbor_idx unchanged."""
        from pymurmur.physics.forces.spatial import _maybe_perception_filter

        positions = np.random.default_rng(6).uniform(0, 500, (5, 3)).astype(np.float32)
        velocities = np.random.default_rng(7).uniform(-1, 1, (5, 3)).astype(np.float32)
        active = np.ones(5, dtype=bool)
        neighbor_idx = np.array([[2, 3, 4, 0, 0], [0, 3, 4, 0, 0],
                                  [0, 1, 4, 0, 0], [0, 1, 2, 0, 0],
                                  [0, 1, 2, 0, 0]], dtype=np.int32)

        result = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            max_dist=0.0, cos_angle=-1.0,
        )
        # Should be the same array (fast path, identity)
        assert result is neighbor_idx

    def test_max_distance_filters_distant_neighbors(self):
        """max_dist=5 excludes neighbours beyond the distance threshold."""
        from pymurmur.physics.forces.spatial import _maybe_perception_filter

        # Bird 0 at origin, birds 1-4 at various distances
        positions = np.array([
            [0, 0, 0],
            [3, 0, 0],   # within 5
            [7, 0, 0],   # beyond 5
            [10, 0, 0],  # beyond 5
            [4, 3, 0],   # dist=5, at boundary
        ], dtype=np.float32)
        velocities = np.ones((5, 3), dtype=np.float32)
        active = np.ones(5, dtype=bool)
        # Bird 0 sees birds 1-4 (indices)
        neighbor_idx = np.array([[1, 2, 3, 4, 0], [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]], dtype=np.int32)

        result = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            max_dist=5.0, cos_angle=-1.0,
        )
        # Bird 0's filtered neighbours: bird 1 (dist=3) and bird 4 (dist=5)
        # Birds 2 (dist=7) and 3 (dist=10) excluded
        bird0_nbrs = set(result[0])
        assert 1 in bird0_nbrs, "bird at dist=3 should be included"
        assert 4 in bird0_nbrs, "bird at dist=5 should be included"
        assert 2 not in bird0_nbrs, "bird at dist=7 should be excluded"
        assert 3 not in bird0_nbrs, "bird at dist=10 should be excluded"

    def test_cone_angle_filters_behind_birds(self):
        """cos_angle=0 excludes neighbours behind the heading (90° cone)."""
        from pymurmur.physics.forces.spatial import _maybe_perception_filter

        # Bird 0 at origin, heading +X. Bird 1 ahead (+X), bird 2 behind (-X).
        positions = np.array([
            [0, 0, 0],
            [5, 0, 0],   # ahead (+X)
            [-5, 0, 0],  # behind (-X)
        ], dtype=np.float32)
        velocities = np.array([
            [1, 0, 0],   # heading +X
            [0, 1, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        active = np.ones(3, dtype=bool)
        neighbor_idx = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)

        result = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            max_dist=0.0, cos_angle=0.0,  # 90° cone (cos 90° = 0)
        )
        bird0_nbrs = set(result[0])
        assert 1 in bird0_nbrs, "bird ahead (+X) should be inside 90° cone"
        assert 2 not in bird0_nbrs, "bird behind (-X) should be outside 90° cone"

    def test_both_filters_applied_simultaneously(self):
        """Both max_dist and cos_angle applied → intersection of both filters."""
        from pymurmur.physics.forces.spatial import _maybe_perception_filter

        positions = np.array([
            [0, 0, 0],
            [3, 0, 0],    # ahead, within range
            [8, 0, 0],    # ahead, beyond range
            [-3, 0, 0],   # behind, within range (excluded by cone)
        ], dtype=np.float32)
        velocities = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        active = np.ones(4, dtype=bool)
        neighbor_idx = np.array([[1, 2, 3, 0], [0, 0, 0, 0],
                                  [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

        result = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            max_dist=5.0, cos_angle=0.0,
        )
        bird0_nbrs = set(result[0])
        # Bird 1: ahead, dist=3 → passes both → included
        assert 1 in bird0_nbrs
        # Bird 2: ahead, dist=8 → fails distance → excluded
        assert 2 not in bird0_nbrs
        # Bird 3: behind, dist=3 → fails cone → excluded
        assert 3 not in bird0_nbrs

    def test_stationary_bird_no_cone_filter(self):
        """Zero-velocity bird → cone filter passes all (no heading to test)."""
        from pymurmur.physics.forces.spatial import _maybe_perception_filter

        positions = np.array([
            [0, 0, 0],
            [5, 0, 0],
            [-5, 0, 0],
        ], dtype=np.float32)
        velocities = np.array([
            [0, 0, 0],   # stationary — no heading
            [0, 1, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        active = np.array([True, True, True], dtype=bool)
        neighbor_idx = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)

        result = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            max_dist=0.0, cos_angle=-0.5,  # restrictive cone
        )
        # Stationary bird: velocity norm < 1e-10 → cone filter skipped → both pass
        bird0_nbrs = set(result[0])
        assert 1 in bird0_nbrs
        assert 2 in bird0_nbrs

    def test_inactive_birds_get_empty(self):
        """Inactive birds get empty neighbour lists."""
        from pymurmur.physics.forces.spatial import _maybe_perception_filter

        positions = np.random.default_rng(8).uniform(0, 500, (5, 3)).astype(np.float32)
        velocities = np.ones((5, 3), dtype=np.float32)
        active = np.array([True, True, False, False, False], dtype=bool)
        neighbor_idx = np.ones((5, 3), dtype=np.int32)

        result = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            max_dist=1000.0, cos_angle=-1.0,
        )
        for i in range(2, 5):
            assert len(result[i]) == 0, f"inactive bird {i} should have empty list"


# ── _predator_escape ────────────────────────────────────────────

class TestPredatorEscape:
    """_predator_escape — threat-driven escape force computation."""

    def test_no_threatened_birds_returns_zero_force(self):
        """Zero threatened birds → all-zero escape force array."""
        from pymurmur.physics.forces.spatial import _predator_escape

        positions = np.random.default_rng(9).uniform(0, 500, (10, 3)).astype(np.float32)
        is_predator = np.zeros(10, dtype=bool)
        threatened = np.zeros(10, dtype=bool)
        active = np.ones(10, dtype=bool)
        neighbor_idx = np.zeros((10, 5), dtype=np.int32)
        cfg = _make_config()

        result = _predator_escape(
            positions, neighbor_idx, is_predator, threatened, active, cfg,
        )
        assert result.shape == (10, 3)
        assert np.all(result == 0.0)

    def test_threatened_bird_gets_nonzero_escape_force(self):
        """Prey near predator → non-zero escape force away from predator."""
        from pymurmur.physics.forces.spatial import _predator_escape

        # Bird 0 is prey, bird 1 is predator (close by)
        positions = np.array([
            [0, 0, 0],
            [10, 0, 0],
        ], dtype=np.float32)
        is_predator = np.array([False, True], dtype=bool)
        threatened = np.array([True, False], dtype=bool)
        active = np.ones(2, dtype=bool)
        neighbor_idx = np.array([[1, 0], [0, 0]], dtype=np.int32)
        cfg = _make_config(predator_escape_factor=500.0)

        result = _predator_escape(
            positions, neighbor_idx, is_predator, threatened, active, cfg,
        )
        # Bird 0 should have non-zero escape force (away from predator at +X)
        assert np.linalg.norm(result[0]) > 0, "threatened bird must get escape force"
        # Escape force should point away from predator (negative X)
        assert result[0, 0] < 0, f"escape force should point away from predator, got {result[0]}"

    def test_non_threatened_active_bird_gets_zero_force(self):
        """Active bird not threatened → zero escape force."""
        from pymurmur.physics.forces.spatial import _predator_escape

        positions = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [50, 0, 0],  # far from predator, not threatened
        ], dtype=np.float32)
        is_predator = np.array([False, True, False], dtype=bool)
        threatened = np.array([True, False, False], dtype=bool)
        active = np.ones(3, dtype=bool)
        neighbor_idx = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
        cfg = _make_config()

        result = _predator_escape(
            positions, neighbor_idx, is_predator, threatened, active, cfg,
        )
        assert np.all(result[2] == 0.0), "non-threatened bird 2 must have zero escape"
