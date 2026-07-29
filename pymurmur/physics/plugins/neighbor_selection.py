"""NeighborSelector ABC, NEIGHBOR_SELECTOR_REGISTRY, and @register decorator.

Modularity pass: formalises the neighbor-selection strategies used by
SpatialMode, ProjectionMode, and VicsekMode behind a registry, mirroring
_mode.py's ForceMode/MODE_REGISTRY pattern. AngleMode's neighbor banding
is fused into its own per-bird steering loop and is NOT included here —
extracting it cleanly would require restructuring that loop, which this
pure-extraction pass deliberately avoids (see the modularity-pass-2 plan).
InfluencerMode/MarlMode/FieldMode don't do index-based neighbor queries
at all (global/O(N) formulas or bespoke targeting) and have nothing to
register here either.

Deliberately minimal ABC (one abstract method, no class-level metadata)
— unlike ForceMode's 7 modes sharing one exact compute() signature, these
3 strategies have genuinely different needs (config-driven filter_mode
vs. sigma-count vs. radius), so `select()` takes **kwargs for each
strategy's own extra parameters rather than forcing a shared parameter
list that doesn't fit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex

NEIGHBOR_SELECTOR_REGISTRY: dict[str, type["NeighborSelector"]] = {}


class NeighborSelector(ABC):
    """Protocol for neighbor-selection strategies.

    Usage::

        @register("hybrid")
        class HybridSelector(NeighborSelector):
            @staticmethod
            def select(positions, velocities, active, index, config, **kwargs):
                ...  # neighbor query + filter
    """

    @staticmethod
    @abstractmethod
    def select(
        positions: "np.ndarray",
        velocities: "np.ndarray",
        active: "np.ndarray",
        index: "SpatialIndex | None",
        config: "SimConfig",
        **kwargs: Any,
    ) -> Any:
        """Select neighbors for all active birds.

        Args:
            positions: (N, 3) float32
            velocities: (N, 3) float32
            active: (N,) bool
            index: spatial index or None
            config: SimConfig
            **kwargs: strategy-specific extras (e.g. filter_mode, sigma,
                radius) — return shape is strategy-specific too (a dense
                (N, k) neighbor_idx array, a ragged/-1-padded array, or a
                sparse adjacency + degree-count pair, depending on what
                the calling mode's own downstream math consumes).
        """
        ...


def register(name: str):
    """Decorator to register a NeighborSelector subclass in
    NEIGHBOR_SELECTOR_REGISTRY.

    Usage::

        @register("hybrid")
        class HybridSelector(NeighborSelector):
            ...
    """

    def decorator(cls: type[NeighborSelector]) -> type[NeighborSelector]:
        NEIGHBOR_SELECTOR_REGISTRY[name] = cls
        return cls

    return decorator


@register("hybrid")
class HybridSelector(NeighborSelector):
    """Wraps spatial_helpers.py's _query_neighbors — batched cKDTree kNN
    + metric/topological/hybrid/none filter_mode dispatch, predator-
    perception-boost aware. Used by SpatialMode.

    kwargs: filter_mode (default "hybrid"), is_predator (default None).
    Returns (N_capacity, k) int32 dense neighbor_idx array, zero-filled
    on inactive rows.
    """

    @staticmethod
    def select(positions, velocities, active, index, config, **kwargs):
        from ..forces.spatial_helpers import _query_neighbors

        return _query_neighbors(positions, active, index, config, **kwargs)


@register("topological_visibility")
class TopologicalVisibilitySelector(NeighborSelector):
    """Wraps projection.py's _topological_neighbors_batch — per-bird
    index.query_knn(k=sigma) loop, -1-sentinel padded. Used by
    ProjectionMode.

    kwargs: sigma (required, topological neighbor count).
    Returns (n_active, sigma) int32 array with -1 sentinels.
    """

    @staticmethod
    def select(positions, velocities, active, index, config, **kwargs):
        from ..forces.projection import _topological_neighbors_batch

        active_idx = np.where(active)[0]
        sigma = kwargs["sigma"]
        return _topological_neighbors_batch(positions, index, active_idx, sigma)


def _vicsek_ball_tree_adjacency(
    active_pos: np.ndarray,
    valid_mask: np.ndarray,
    radius: float,
    tree,
) -> tuple[Any, np.ndarray]:
    """Ball-tree radius query -> valid-neighbor sparse adjacency + degree
    counts. Moved verbatim from VicsekMode.compute()'s former inline block.

    Returns (adj, nbr_counts): adj is a scipy.sparse.csr_matrix of shape
    (n_active, n_active) with a 1 at [i,j] iff j is within `radius` of i
    AND valid_mask[j] is True (None if no edges); nbr_counts is the
    per-bird degree (row sum of adj, zeros if no edges).
    """
    n_active = len(active_pos)
    all_nbrs = tree.query_ball_tree(tree, radius)

    rows: list[int] = []
    cols: list[int] = []
    for i, nbrs in enumerate(all_nbrs):
        for j in nbrs:
            if valid_mask[j]:
                rows.append(i)
                cols.append(j)

    nbr_counts = np.zeros(n_active, dtype=np.float32)
    if not rows:
        return None, nbr_counts

    from scipy.sparse import coo_matrix

    adj = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_active, n_active),
    ).tocsr()
    nbr_counts = np.array(adj.sum(axis=1)).flatten()
    return adj, nbr_counts


@register("ball_tree_radius")
class BallTreeRadiusSelector(NeighborSelector):
    """Wraps VicsekMode's former inline ball-tree radius query — builds
    a sparse adjacency of velocity-valid neighbors within `radius`. Used
    by VicsekMode.

    kwargs: radius (required).
    Returns (adj, nbr_counts) — see _vicsek_ball_tree_adjacency.
    """

    @staticmethod
    def select(positions, velocities, active, index, config, **kwargs):
        active_idx = np.where(active)[0]
        active_pos = positions[active_idx]
        radius = kwargs["radius"]

        tree = getattr(index, 'tree', None) if index is not None else None
        if tree is None:
            from scipy.spatial import cKDTree

            tree = cKDTree(active_pos)

        vel_norms = np.linalg.norm(velocities[active_idx], axis=1)
        valid_mask = vel_norms > 1e-6

        return _vicsek_ball_tree_adjacency(active_pos, valid_mask, radius, tree)
