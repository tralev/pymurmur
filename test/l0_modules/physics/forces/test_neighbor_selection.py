"""Unit tests for physics.forces.neighbor_selection — NeighborSelector
ABC, NEIGHBOR_SELECTOR_REGISTRY, and dispatch-vs-direct-call equality
for the 3 registered strategies.

Modularity pass 2: formalises SpatialMode/ProjectionMode/VicsekMode's
neighbor-query strategies behind a registry mirroring ForceMode's
proven pattern. These tests verify the registry itself plus that each
strategy's registry dispatch reproduces its underlying function's
direct-call behavior exactly (pure extraction, no behavior change).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.physics.flock import KDTreeIndex, PhysicsFlock
from pymurmur.physics.forces.neighbor_selection import (
    NEIGHBOR_SELECTOR_REGISTRY,
    BallTreeRadiusSelector,
    HybridSelector,
    NeighborSelector,
    TopologicalVisibilitySelector,
)
from pymurmur.physics.forces.projection import _topological_neighbors_batch
from pymurmur.physics.forces.spatial_helpers import _query_neighbors


class TestNeighborSelectorRegistry:
    def test_all_three_strategies_registered(self):
        assert set(NEIGHBOR_SELECTOR_REGISTRY.keys()) == {
            "hybrid", "topological_visibility", "ball_tree_radius",
        }

    def test_registered_classes_are_neighbor_selector_subclasses(self):
        for cls in NEIGHBOR_SELECTOR_REGISTRY.values():
            assert issubclass(cls, NeighborSelector)

    def test_registry_maps_to_expected_classes(self):
        assert NEIGHBOR_SELECTOR_REGISTRY["hybrid"] is HybridSelector
        assert NEIGHBOR_SELECTOR_REGISTRY["topological_visibility"] is TopologicalVisibilitySelector
        assert NEIGHBOR_SELECTOR_REGISTRY["ball_tree_radius"] is BallTreeRadiusSelector


def _build_flock(config, n=40, seed=3):
    config.num_boids = n
    config.seed = seed
    flock = PhysicsFlock(config)
    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)
    flock._index = kdt
    return flock


class TestHybridSelectorDispatchMatchesDirectCall:
    def test_dispatch_matches_direct_call(self, default_config):
        flock = _build_flock(default_config)
        index = flock.get_index()

        direct = _query_neighbors(
            flock.positions, flock.active, index, default_config,
            filter_mode="hybrid",
        )
        via_registry = NEIGHBOR_SELECTOR_REGISTRY["hybrid"].select(
            flock.positions, flock.velocities, flock.active, index, default_config,
            filter_mode="hybrid",
        )
        np.testing.assert_array_equal(direct, via_registry)

    @pytest.mark.parametrize("filter_mode", ["hybrid", "metric", "topological", "none"])
    def test_dispatch_matches_direct_call_all_filter_modes(self, default_config, filter_mode):
        flock = _build_flock(default_config)
        index = flock.get_index()

        direct = _query_neighbors(
            flock.positions, flock.active, index, default_config,
            filter_mode=filter_mode,
        )
        via_registry = NEIGHBOR_SELECTOR_REGISTRY["hybrid"].select(
            flock.positions, flock.velocities, flock.active, index, default_config,
            filter_mode=filter_mode,
        )
        np.testing.assert_array_equal(direct, via_registry)


class TestTopologicalVisibilitySelectorDispatchMatchesDirectCall:
    def test_dispatch_matches_direct_call(self, default_config):
        flock = _build_flock(default_config)
        index = flock.get_index()
        active_idx = np.where(flock.active)[0]
        sigma = 4

        direct = _topological_neighbors_batch(flock.positions, index, active_idx, sigma)
        via_registry = NEIGHBOR_SELECTOR_REGISTRY["topological_visibility"].select(
            flock.positions, flock.velocities, flock.active, index, default_config,
            sigma=sigma,
        )
        np.testing.assert_array_equal(direct, via_registry)


class TestBallTreeRadiusSelectorDispatchMatchesDirectCall:
    def test_dispatch_matches_manually_replicated_original_logic(self, default_config):
        """Reproduces the exact former inline block in VicsekMode.compute()
        (pre-extraction), given the same inputs, and compares against the
        registry dispatch."""
        from scipy.sparse import coo_matrix

        flock = _build_flock(default_config)
        index = flock.get_index()
        active_idx = np.where(flock.active)[0]
        active_pos = flock.positions[active_idx]
        radius = default_config.vicsek_radius_influence

        # ── Manually replicated pre-extraction logic ──
        tree = getattr(index, 'tree', None) if index is not None else None
        if tree is None:
            from scipy.spatial import cKDTree
            tree = cKDTree(active_pos)

        vel_norms = np.linalg.norm(flock.velocities[active_idx], axis=1)
        valid_mask = vel_norms > 1e-6

        all_nbrs = tree.query_ball_tree(tree, radius)
        rows: list[int] = []
        cols: list[int] = []
        for i, nbrs in enumerate(all_nbrs):
            for j in nbrs:
                if valid_mask[j]:
                    rows.append(i)
                    cols.append(j)

        n_active = len(active_idx)
        expected_nbr_counts = np.zeros(n_active, dtype=np.float32)
        expected_adj = None
        if rows:
            expected_adj = coo_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_active, n_active),
            ).tocsr()
            expected_nbr_counts = np.array(expected_adj.sum(axis=1)).flatten()

        # ── Registry dispatch ──
        adj, nbr_counts = NEIGHBOR_SELECTOR_REGISTRY["ball_tree_radius"].select(
            flock.positions, flock.velocities, flock.active, index, default_config,
            radius=radius,
        )

        np.testing.assert_array_equal(nbr_counts, expected_nbr_counts)
        if expected_adj is None:
            assert adj is None
        else:
            np.testing.assert_array_equal(adj.toarray(), expected_adj.toarray())

    def test_returns_none_adjacency_when_no_valid_velocities(self, default_config):
        """All-zero velocities -> valid_mask is all-False -> every
        candidate edge is filtered out -> adj is None, regardless of
        how close birds are positioned."""
        default_config.num_boids = 5
        default_config.seed = 1
        flock = PhysicsFlock(default_config)
        flock.positions[:] = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
        ], dtype=np.float32)
        flock.velocities[:] = 0.0
        kdt = KDTreeIndex()
        kdt.rebuild(flock.positions, flock.active)
        flock._index = kdt

        adj, nbr_counts = NEIGHBOR_SELECTOR_REGISTRY["ball_tree_radius"].select(
            flock.positions, flock.velocities, flock.active, flock.get_index(), default_config,
            radius=1000.0,
        )
        assert adj is None
        np.testing.assert_array_equal(nbr_counts, np.zeros(5, dtype=np.float32))
