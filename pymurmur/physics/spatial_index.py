"""Spatial index implementations — SpatialHashGrid, KDTreeIndex.

Extracted from flock.py (file-size split) — PhysicsFlock owns and
selects between these, but they're independent, stateless-per-query
index structures with no dependency on PhysicsFlock itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.types import min_image

if TYPE_CHECKING:
    from ..core.config import SimConfig


class SpatialHashGrid:
    """Uniform 3D grid with 27-cell queries.

    O(N) rebuild, O(1) per-query. Best for N < 5,000 where per-cell
    density is low.

    P2.5: Modulo-wrapped cell keys + min-image distances for correct
    toroidal boundary behaviour.
    """

    def __init__(self, config: SimConfig) -> None:
        self._cell_size: float = config.visual_range
        self._bins: dict[tuple[int, int, int], list[int]] = {}
        self._positions: np.ndarray | None = None
        # P2.5: Domain dimensions for toroidal wrapping
        self._box: tuple[float, float, float] = (
            float(config.width), float(config.height), float(config.depth),
        )
        # C3: use_toroidal_distance — only apply min-image wrapping when
        # both the flag is set AND the boundary is actually toroidal.
        # Previously wrapped unconditionally, which was wrong for
        # open/margin/sphere/sphere_soft boundaries.
        self._toroidal: bool = (
            bool(getattr(config, 'use_toroidal_distance', True))
            and config.boundary_mode == "toroidal"
        )
        self._cols: int = max(1, int(config.width / self._cell_size))
        self._rows: int = max(1, int(config.height / self._cell_size))
        self._slices: int = max(1, int(config.depth / self._cell_size))

    @property
    def ready(self) -> bool:
        return len(self._bins) > 0

    @property
    def tree(self) -> None:
        """SpatialHashGrid has no underlying tree — returns None."""
        return None

    def rebuild(self, positions: np.ndarray, active: np.ndarray) -> None:
        """Populate grid bins from active bird positions.

        P2.5: Cell keys are modulo-wrapped so birds near one boundary
        appear in cells at the opposite boundary, enabling cross-seam
        neighbour queries.
        """
        self._bins.clear()
        self._positions = positions  # stored for query_knn
        cell_size = self._cell_size
        cols, rows, slices = self._cols, self._rows, self._slices
        for i in np.where(active)[0]:
            x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
            key = (
                int(x // cell_size) % cols,
                int(y // cell_size) % rows,
                int(z // cell_size) % slices,
            )
            self._bins.setdefault(key, []).append(i)

    def incremental_rebuild(
        self, positions: np.ndarray, active: np.ndarray,
        last_cell: np.ndarray,
    ) -> int:
        """P5.6: Update bins only for birds that crossed cell boundaries.

        Returns number of birds touched (removed from old cell or added
        to new cell). Birds whose cell key hasn't changed are left
        in-place in the bin dict.

        Args:
            positions: (N, 3) float32 positions
            active: (N,) bool active mask
            last_cell: (N, 3) int32 array of last-frame cell coords;
                       -1 sentinel means "not yet filed"

        Returns:
            Number of birds that were re-filed (touch count).
        """
        self._positions = positions
        cell_size = self._cell_size
        cols, rows, slices = self._cols, self._rows, self._slices
        touched = 0

        for i in np.where(active)[0]:
            x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
            new_key = (
                int(x // cell_size) % cols,
                int(y // cell_size) % rows,
                int(z // cell_size) % slices,
            )

            old_key: tuple | None = None
            if last_cell[i, 0] >= 0:
                old_key = (
                    int(last_cell[i, 0]),
                    int(last_cell[i, 1]),
                    int(last_cell[i, 2]),
                )

            if old_key is not None and old_key != new_key:
                # Remove from old bin
                if old_key in self._bins and i in self._bins[old_key]:
                    self._bins[old_key].remove(i)
                touched += 1

            if old_key is None or old_key != new_key:
                # Add to new bin
                self._bins.setdefault(new_key, []).append(i)
                touched += 1

            last_cell[i] = np.array(new_key, dtype=np.int32)

        return touched

    def query_radius(self, pos: np.ndarray, radius: float) -> list[int]:
        """Return candidate indices within the cell neighborhood covering
        ``radius``.

        D5: the neighborhood half-width scales with the requested radius
        (``ceil(radius / cell_size)`` cells each direction) instead of a
        hardcoded ±1 — a caller asking for a radius smaller or larger
        than the grid's own cell_size previously got the wrong candidate
        set (always exactly the fixed 27-cell block regardless of what
        was asked for). Capped at 10 cells/direction (2 121 cells) so a
        pathologically large radius can't turn one query into an
        effective full-grid scan.

        Cell-block membership, not an exact circle — callers that need
        exact-radius filtering (e.g. query_knn) re-filter by real
        distance afterward; this is deliberately a safe superset.

        P2.5: Neighbour cell keys are modulo-wrapped for toroidal
        cross-seam queries.
        """
        cell_size = self._cell_size
        cols, rows, slices = self._cols, self._rows, self._slices
        cx = int(pos[0] // cell_size) % cols
        cy = int(pos[1] // cell_size) % rows
        cz = int(pos[2] // cell_size) % slices
        reach = max(1, min(10, int(np.ceil(radius / cell_size))))
        # Dedup wrapped keys, not just raw offsets: on a small grid
        # (cols/rows/slices <= 2*reach), modulo-wrapping can revisit the
        # same cell from different (dx,dy,dz) offsets — without this, a
        # bird's index would appear multiple times in candidates and
        # corrupt query_knn's downstream top-k selection.
        visited_keys: set[tuple[int, int, int]] = set()
        candidates: list[int] = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    key = (
                        (cx + dx) % cols,
                        (cy + dy) % rows,
                        (cz + dz) % slices,
                    )
                    if key in visited_keys:
                        continue
                    visited_keys.add(key)
                    if key in self._bins:
                        candidates.extend(self._bins[key])
        return candidates

    def query_knn(self, pos: np.ndarray, k: int) -> np.ndarray:
        """Return indices of k nearest neighbors via radius query + sort.

        Uses the 27-cell neighbourhood from query_radius, computes
        toroidal (min-image) distances, and returns the k closest
        (excluding self).

        P2.5: Min-image distances for correct toroidal boundary behaviour.
        """
        if self._positions is None:
            return np.array([], dtype=np.int32)

        candidates = self.query_radius(pos, self._cell_size)
        if len(candidates) <= 1:
            return np.array([], dtype=np.int32)

        # Collect candidate positions via numpy advanced indexing
        candidate_pos = self._positions[candidates]
        diffs = candidate_pos - pos

        # P2.5/C3: min-image distances only for a toroidal boundary with
        # use_toroidal_distance enabled — otherwise plain euclidean.
        if self._toroidal:
            box_arr = np.array(self._box, dtype=np.float32)
            diffs = min_image(diffs, box_arr)
        dists = np.linalg.norm(diffs, axis=1)

        # Exclude self (d < 1e-6) and sort by distance
        mask = dists > 1e-6
        if not mask.any():
            return np.array([], dtype=np.int32)

        valid_dists = dists[mask]
        valid_indices = np.array(candidates, dtype=np.int32)[mask]

        # Use argpartition for top k (faster than full argsort)
        n = min(k, len(valid_dists))
        part_idx = np.argpartition(valid_dists, n - 1)[:n]
        # Sort the top k results
        sorted_local = np.argsort(valid_dists[part_idx])
        return valid_indices[part_idx[sorted_local]]


class KDTreeIndex:
    """scipy.spatial.cKDTree wrapper.

    O(N log N) build, O(k log N) per query. Required for N >= 5,000.
    Returns global indices — compacted→global mapping applied in rebuild.
    """

    def __init__(self, box: tuple[float, float, float] | None = None) -> None:
        self._tree: Any | None = None   # scipy.spatial.cKDTree
        self._active_map: np.ndarray = np.array([], dtype=np.int32)
        # C3: use_toroidal_distance — periodic boxsize for toroidal parity
        # with SpatialHashGrid. None (default) preserves prior behaviour
        # (plain, non-periodic tree) for every existing caller.
        self._box: tuple[float, float, float] | None = box

    @property
    def ready(self) -> bool:
        return self._tree is not None

    def rebuild(self, positions: np.ndarray, active: np.ndarray) -> None:
        """Build cKDTree from active bird positions.

        Stores a compacted→global index map so query_knn returns
        global indices consistent with SpatialHashGrid.

        C3: when constructed with a box, uses scipy's periodic `boxsize`
        for toroidal-consistent neighbour queries. Falls back to a plain
        (non-periodic) tree if positions fall outside [0, box) — scipy
        requires periodic coordinates in-bounds.
        """
        from scipy.spatial import cKDTree
        self._active_map = np.where(active)[0].astype(np.int32)
        active_pos = positions[active]
        if len(active_pos) == 0:
            self._tree = None
            return
        if self._box is not None:
            try:
                self._tree = cKDTree(
                    active_pos, boxsize=np.array(self._box, dtype=np.float64),
                )
                return
            except ValueError:
                pass  # positions out of [0, box) — fall back below
        self._tree = cKDTree(active_pos)

    @property
    def tree(self) -> object | None:
        """Raw scipy cKDTree — for batch operations like query_ball_tree.

        Note: the raw tree returns compacted indices; callers using
        this property directly are responsible for index-space conversion.
        """
        return self._tree

    def query_knn(self, pos: np.ndarray, k: int) -> np.ndarray:
        """Return global indices of k nearest neighbors (excluding self)."""
        if self._tree is None:
            return np.array([], dtype=np.int32)
        _, idx = self._tree.query(pos, k=k + 1)
        # Exclude self (idx[0]) and map compacted→global
        compacted = idx[1:] if len(idx) > 1 else idx[:0]
        if len(compacted) > 0 and len(self._active_map) > 0:
            return self._active_map[compacted]
        return np.array([], dtype=np.int32)
