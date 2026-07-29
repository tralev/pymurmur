"""Unit tests for physics.plugins.spatial_index_strategy — SPATIAL_INDEX_STRATEGY_REGISTRY
and AUTO_INDEX_THRESHOLD, mirroring test_boundary_registry.py's shape
(registry-membership assertions for a modularity-pass registry).

Modularity pass 5: extracts the 4-way if/elif index-selection chain from
PhysicsFlock.__init__() behind a registry of plain callables (NOT an ABC
— the strategies have no uniform constructor signature, documented in
the module's own docstring). These tests verify registry contents and
the "auto" strategy's threshold behavior; PhysicsFlock's actual index
selection is covered by test_flock.py / test_spatial_index_contract.py.
"""

from __future__ import annotations

from pymurmur.core.config import SimConfig
from pymurmur.physics.spatial_index import KDTreeIndex, SpatialHashGrid
from pymurmur.physics.plugins.spatial_index_strategy import (
    AUTO_INDEX_THRESHOLD,
    SPATIAL_INDEX_STRATEGY_REGISTRY,
)


class TestSpatialIndexStrategyRegistry:
    def test_all_four_strategies_registered(self):
        assert set(SPATIAL_INDEX_STRATEGY_REGISTRY.keys()) == {
            "kdtree", "hash_grid", "none", "auto",
        }

    def test_entries_are_callable(self):
        for strategy in SPATIAL_INDEX_STRATEGY_REGISTRY.values():
            assert callable(strategy)

    def test_kdtree_strategy_returns_kdtree_index(self):
        strategy = SPATIAL_INDEX_STRATEGY_REGISTRY["kdtree"]
        result = strategy(SimConfig(), N_active=10, kdtree_box=None)
        assert isinstance(result, KDTreeIndex)

    def test_hash_grid_strategy_returns_spatial_hash_grid(self):
        strategy = SPATIAL_INDEX_STRATEGY_REGISTRY["hash_grid"]
        result = strategy(SimConfig(), N_active=10, kdtree_box=None)
        assert isinstance(result, SpatialHashGrid)

    def test_none_strategy_returns_none(self):
        strategy = SPATIAL_INDEX_STRATEGY_REGISTRY["none"]
        result = strategy(SimConfig(), N_active=10, kdtree_box=None)
        assert result is None

    def test_auto_strategy_uses_hash_grid_below_threshold(self):
        strategy = SPATIAL_INDEX_STRATEGY_REGISTRY["auto"]
        result = strategy(SimConfig(), N_active=AUTO_INDEX_THRESHOLD - 1, kdtree_box=None)
        assert isinstance(result, SpatialHashGrid)

    def test_auto_strategy_uses_kdtree_at_and_above_threshold(self):
        strategy = SPATIAL_INDEX_STRATEGY_REGISTRY["auto"]
        result = strategy(SimConfig(), N_active=AUTO_INDEX_THRESHOLD, kdtree_box=None)
        assert isinstance(result, KDTreeIndex)


class TestAutoIndexThreshold:
    def test_threshold_is_5000(self):
        assert AUTO_INDEX_THRESHOLD == 5000
