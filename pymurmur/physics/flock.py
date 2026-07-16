"""PhysicsFlock — flock state container with spatial indexing.

Level 1 — assembles FlockArrays + spatial index + integration kernel.
Auto-selects SpatialHashGrid (N < 5K) or KDTreeIndex (N >= 5K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .boid import integrate, random_unit_sphere, init_positions, init_velocities_blob
from ..core.types import SpatialIndex, min_image

if TYPE_CHECKING:
    from ..core.config import SimConfig


# ── PhysicsFlock ──────────────────────────────────────────────────

class PhysicsFlock:
    """Owns the flock state arrays and provides the step() method.

    Uses Structure-of-Arrays for flat, vectorised operations.
    Force computation is delegated to physics.forces (not owned here).
    """

    def __init__(self, config: SimConfig) -> None:
        N = config.num_boids
        self.rng = np.random.default_rng(
            config.seed if config.seed else 0
        )

        position_mode = getattr(config, 'position_init', 'box')
        self.positions = init_positions(
            N, config.width, config.height, config.depth, self.rng,
            mode=position_mode,
            separation=getattr(config, 'boid_size', 9.0),
        )
        if position_mode == 'blob':
            self.velocities = init_velocities_blob(N, config.v0, self.rng)
        else:
            self.velocities = random_unit_sphere(N, self.rng) * config.v0 * 0.8
        self.accelerations = np.zeros((N, 3), dtype=np.float32)
        self.seeds = self.rng.uniform(0.0, 1.0, N).astype(np.float32)
        self.last_theta = np.zeros(N, dtype=np.float32)
        self.active = np.ones(N, dtype=bool)

        # Smoothed swarm centre — EMA centroid (P0.5)
        self.center: np.ndarray | None = None

        # Wander state — set by Wander extension before force computation (P3.1)
        self.wander_center: np.ndarray | None = None
        self.wander_heading: np.ndarray | None = None

        # Species column — predator/prey flag (P0.6)
        self.is_predator: np.ndarray = np.zeros(N, dtype=bool)

        # Previous-frame stash — render interpolation + MSD (P0.7)
        self.prev_positions: np.ndarray = np.zeros((N, 3), dtype=np.float32)
        self.last_accelerations: np.ndarray = np.zeros((N, 3), dtype=np.float32)

        # Per-bird max speed — None uses scalar v0 (P0.8)
        self.max_speed: np.ndarray | None = None

        # Spatial index selection — honor explicit config, fall back to N heuristic
        self._index: SpatialIndex | None
        self._spatial_index_mode = config.spatial_index
        self._visual_range = config.visual_range
        self._index_threshold = 5000
        index_choice = config.spatial_index
        if index_choice == "kdtree":
            self._index = KDTreeIndex()
        elif index_choice == "hash_grid":
            self._index = SpatialHashGrid(config)
        elif index_choice == "none":
            # No spatial index — modes that need it must build their own
            self._index = None
        else:  # "auto"
            if N >= self._index_threshold:
                self._index = KDTreeIndex()
            else:
                self._index = SpatialHashGrid(config)

    def integrate(self, config: SimConfig, dt: float) -> None:
        """Stash pre-integration state, integrate, reset accelerations, update centre.

        Pure state operation — never imports forces. The caller (engine)
        is responsible for force computation before calling this method.
        """
        # 1. Stash pre-integration state (P0.7)
        self.last_accelerations[:] = self.accelerations
        self.prev_positions[:] = self.positions

        # 2. Integrate
        integrate(
            self.positions, self.velocities, self.accelerations,
            self.active,
            config.width, config.height, config.depth,
            config.v0, config.boundary_mode, dt,
            config.boundary_sphere_radius, config.boundary_avoidance_factor,
            rng=self.rng,
            max_speed=self.max_speed,
            center=self.center,
        )

        # 3. Update smoothed centre (EMA centroid)
        self.update_center()

    def _reevaluate_index(self) -> None:
        """Migrate index on N threshold crossing when spatial_index is 'auto'.

        When N_active crosses 5,000 during add_boids/remove_boids,
        switches between SpatialHashGrid and KDTreeIndex.
        Does nothing when spatial_index is explicitly set.
        """
        if self._spatial_index_mode != "auto" or self._index is None:
            return

        n = self.N_active
        is_kdtree = isinstance(self._index, KDTreeIndex)

        if n >= self._index_threshold and not is_kdtree:
            self._index = KDTreeIndex()
        elif n < self._index_threshold and is_kdtree:
            # Need config-like object for SpatialHashGrid — fake it
            from ..core.config import SimConfig
            cfg = SimConfig()
            cfg.visual_range = self._visual_range
            self._index = SpatialHashGrid(cfg)

    def add_boids(
        self, count: int, config: SimConfig, is_predator: bool = False
    ) -> int:
        """Flip inactive slots to active. Extends arrays if needed.

        Returns number actually added.
        """
        inactive = np.where(~self.active)[0]
        added = min(count, len(inactive))

        if added == 0 and count > 0:
            # Extend arrays
            self._extend(count)
            inactive = np.where(~self.active)[0]
            added = min(count, len(inactive))

        if added > 0:
            idx = inactive[:added]
            self.active[idx] = True
            self.is_predator[idx] = is_predator
            position_mode = getattr(config, 'position_init', 'box')
            self.positions[idx] = init_positions(
                added, config.width, config.height, config.depth, self.rng,
                mode=position_mode,
                separation=getattr(config, 'boid_size', 9.0),
            )
            if position_mode == 'blob':
                self.velocities[idx] = init_velocities_blob(added, config.v0, self.rng)
            else:
                self.velocities[idx] = random_unit_sphere(added, self.rng) * config.v0 * 0.8
            self.accelerations[idx] = 0.0

        if added > 0:
            self._reevaluate_index()

        return added

    def remove_boids(self, count: int) -> int:
        """Deactivate last N active birds. Returns number actually removed."""
        active_idx = np.where(self.active)[0]
        removed = min(count, len(active_idx))
        if removed > 0:
            self.active[active_idx[-removed:]] = False
            self._reevaluate_index()
        return removed

    def _extend(self, count: int) -> None:
        """Grow arrays to accommodate more birds."""
        N = self.N_capacity
        new_size = N + max(count, 1000)

        for attr in (
            "positions", "velocities", "accelerations",
            "prev_positions", "last_accelerations",
        ):
            arr = getattr(self, attr)
            extended = np.zeros((new_size, 3), dtype=np.float32)
            extended[:N] = arr
            setattr(self, attr, extended)

        for attr in ("seeds", "last_theta"):
            arr = getattr(self, attr)
            extended = np.zeros(new_size, dtype=np.float32)
            extended[:N] = arr
            setattr(self, attr, extended)

        # Extend bool arrays
        for attr in ("is_predator",):
            arr = getattr(self, attr)
            extended = np.zeros(new_size, dtype=bool)
            extended[:N] = arr
            setattr(self, attr, extended)

        active = np.zeros(new_size, dtype=bool)
        active[:N] = self.active
        self.active = active

    @property
    def N_active(self) -> int:
        return int(self.active.sum())

    @property
    def N_capacity(self) -> int:
        return len(self.active)

    def get_index(self) -> SpatialIndex | None:
        return self._index

    def update_center(self) -> None:
        """Update smoothed swarm centre via EMA.

        centroid = mean of active positions.
        If centre is uninitialised, snap to centroid.
        Otherwise, EMA: centre ← centre + 0.5 · (centroid − centre).
        """
        if self.active.sum() == 0:
            return

        centroid = self.positions[self.active].mean(axis=0)

        if self.center is None:
            self.center = centroid.copy()
        else:
            self.center += 0.5 * (centroid - self.center)


# ── Spatial index implementations ─────────────────────────────────

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

    def query_radius(self, pos: np.ndarray, radius: float) -> list[int]:
        """Return candidate indices within the 27-cell neighborhood.

        P2.5: Neighbour cell keys are modulo-wrapped for toroidal
        cross-seam queries.
        """
        cell_size = self._cell_size
        cols, rows, slices = self._cols, self._rows, self._slices
        cx = int(pos[0] // cell_size) % cols
        cy = int(pos[1] // cell_size) % rows
        cz = int(pos[2] // cell_size) % slices
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (
                        (cx + dx) % cols,
                        (cy + dy) % rows,
                        (cz + dz) % slices,
                    )
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

        # P2.5: Min-image distances for toroidal wrapping
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

    def __init__(self) -> None:
        self._tree: object = None   # scipy.spatial.cKDTree
        self._active_map: np.ndarray = np.array([], dtype=np.int32)

    @property
    def ready(self) -> bool:
        return self._tree is not None

    def rebuild(self, positions: np.ndarray, active: np.ndarray) -> None:
        """Build cKDTree from active bird positions.

        Stores a compacted→global index map so query_knn returns
        global indices consistent with SpatialHashGrid.
        """
        from scipy.spatial import cKDTree
        self._active_map = np.where(active)[0].astype(np.int32)
        active_pos = positions[active]
        self._tree = cKDTree(active_pos) if len(active_pos) > 0 else None

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
