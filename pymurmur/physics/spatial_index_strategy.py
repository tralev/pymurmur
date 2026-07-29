"""SpatialIndexStrategy ABC, SPATIAL_INDEX_STRATEGY_REGISTRY, and
@register decorator.

Modularity pass 5: extracts the 4-way if/elif index-selection chain from
PhysicsFlock.__init__() (kdtree, hash_grid, none, auto) behind a
registry, mirroring physics/forces/_mode.py's ForceMode/MODE_REGISTRY
pattern.

Comparison.md's Spatial Acceleration taxonomy lists grid cells, KDTree,
hashed grid, k-NN, and parallel reduction across the surveyed
implementations; this codebase implements grid + KDTree with adaptive
auto-selection (N < 5K → hash_grid, N >= 5K → kdtree).

Each strategy returns a SpatialIndex | None and can be re-evaluated
when N_active crosses the threshold (auto mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..core.types import SpatialIndex

SPATIAL_INDEX_STRATEGY_REGISTRY: dict[str, type["SpatialIndexStrategy"]] = {}

# Shared with flock.py::_reevaluate_index() — single source of truth for
# the N_active threshold that decides SpatialHashGrid vs. KDTreeIndex in
# "auto" mode. Previously duplicated as two independently-hardcoded
# literals; fixed to avoid silent drift between the two call sites.
AUTO_INDEX_THRESHOLD = 5000


class SpatialIndexStrategy:
    """Protocol for spatial-index selection strategies.

    Unlike the other ABC registries (ForceMode, BoundaryMode,
    SpeedModel), this one is NOT an ABC because the strategies share
    NO uniform signature — they have different constructor needs
    (KDTree needs a box, SpatialHashGrid needs a SimConfig, None
    returns None, auto delegates to N-heuristic).  The registry exists
    solely to replace the hardcoded if/elif in PhysicsFlock.__init__()
    with a dispatch table, which is sufficient for the current set of
    4 strategies.

    Each subclass is a callable that takes (config, N_active, kdtree_box)
    and returns SpatialIndex | None.

    Usage::

        @register("auto")
        def auto_strategy(config, N_active, kdtree_box):
            if N_active >= 5000:
                return KDTreeIndex(box=kdtree_box)
            return SpatialHashGrid(config)
    """

    pass  # no base class — strategies are plain functions registered by name


def register(name: str):
    """Decorator to register a SpatialIndexStrategy function in
    SPATIAL_INDEX_STRATEGY_REGISTRY.

    Usage::

        @register("kdtree")
        def kdtree_strategy(config, N_active, kdtree_box):
            from ..spatial_index import KDTreeIndex
            return KDTreeIndex(box=kdtree_box)
    """

    def decorator(fn):
        SPATIAL_INDEX_STRATEGY_REGISTRY[name] = fn
        return fn

    return decorator


# ── Registered strategies ─────────────────────────────────────────


@register("kdtree")
def _kdtree_strategy(config, N_active, kdtree_box):
    """Always use KDTreeIndex (toroidal-aware when box is provided)."""
    from .spatial_index import KDTreeIndex

    return KDTreeIndex(box=kdtree_box)


@register("hash_grid")
def _hash_grid_strategy(config, N_active, kdtree_box):
    """Always use SpatialHashGrid."""
    from .spatial_index import SpatialHashGrid

    return SpatialHashGrid(config)


@register("none")
def _none_strategy(config, N_active, kdtree_box):
    """No spatial index — modes that need one must build their own."""
    return None


@register("auto")
def _auto_strategy(config, N_active, kdtree_box):
    """Auto-select: SpatialHashGrid for N < AUTO_INDEX_THRESHOLD, KDTreeIndex above."""
    from .spatial_index import KDTreeIndex, SpatialHashGrid

    if N_active >= AUTO_INDEX_THRESHOLD:
        return KDTreeIndex(box=kdtree_box)
    return SpatialHashGrid(config)
