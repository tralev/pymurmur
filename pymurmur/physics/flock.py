"""PhysicsFlock — flock state container with spatial indexing.

Level 1 — assembles FlockArrays + spatial index + integration kernel.
Auto-selects SpatialHashGrid (N < 5K) or KDTreeIndex (N >= 5K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .boid import integrate, random_positions, random_unit_sphere

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

        self.positions = random_positions(
            N, config.width, config.height, config.depth, self.rng)
        self.velocities = random_unit_sphere(N, self.rng) * config.v0 * 0.8
        self.accelerations = np.zeros((N, 3), dtype=np.float32)
        self.seeds = self.rng.uniform(0.0, 1.0, N).astype(np.float32)
        self.last_theta = np.zeros(N, dtype=np.float32)
        self.active = np.ones(N, dtype=bool)

        # Smoothed swarm centre — EMA centroid (P0.5)
        self.center: np.ndarray | None = None

        # Species column — predator/prey flag (P0.6)
        self.is_predator: np.ndarray = np.zeros(N, dtype=bool)

        # Previous-frame stash — render interpolation + MSD (P0.7)
        self.prev_positions: np.ndarray = np.zeros((N, 3), dtype=np.float32)
        self.last_accelerations: np.ndarray = np.zeros((N, 3), dtype=np.float32)

        # Per-bird max speed — None uses scalar v0 (P0.8)
        self.max_speed: np.ndarray | None = None

        # Auto-select spatial index based on flock size
        self._index: SpatialHashGrid | KDTreeIndex
        if N >= 5000:
            self._index = KDTreeIndex()
        else:
            self._index = SpatialHashGrid(config)

    _INDEX_MODES: frozenset[str] = frozenset({"spatial", "projection"})

    def step(self, config: SimConfig, dt: float) -> None:
        """One simulation tick: rebuild index -> compute forces -> integrate."""
        # 1. Rebuild spatial index (only modes that query neighbours)
        if config.mode in self._INDEX_MODES:
            self._index.rebuild(self.positions, self.active)

        # 2. Compute forces - dispatched by mode (physics.forces)
        from .forces import compute_all_forces
        compute_all_forces(self, config)

        # 3. Stash pre-integration state (P0.7)
        self.last_accelerations[:] = self.accelerations
        self.prev_positions[:] = self.positions

        # 4. Integrate
        integrate(
            self.positions, self.velocities, self.accelerations,
            self.active,
            config.width, config.height, config.depth,                    config.v0, config.boundary_mode, dt,
                    config.boundary_sphere_radius, config.boundary_avoidance_factor,
                    rng=self.rng,
                    max_speed=self.max_speed,
                )

        # 5. Update smoothed centre (EMA centroid)
        self.update_center()

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
            self.positions[idx] = random_positions(
                added, config.width, config.height, config.depth, self.rng)
            self.velocities[idx] = random_unit_sphere(added, self.rng) * config.v0 * 0.8
            self.accelerations[idx] = 0.0

        return added

    def remove_boids(self, count: int) -> int:
        """Deactivate last N active birds. Returns number actually removed."""
        active_idx = np.where(self.active)[0]
        removed = min(count, len(active_idx))
        if removed > 0:
            self.active[active_idx[-removed:]] = False
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

    def get_index(self) -> SpatialHashGrid | KDTreeIndex:
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
    """

    def __init__(self, config: SimConfig) -> None:
        self._cell_size: float = config.visual_range
        self._bins: dict[tuple[int, int, int], list[int]] = {}
        self._positions: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return len(self._bins) > 0

    def rebuild(self, positions: np.ndarray, active: np.ndarray) -> None:
        """Populate grid bins from active bird positions."""
        self._bins.clear()
        self._positions = positions  # stored for query_knn
        cell_size = self._cell_size
        for i in np.where(active)[0]:
            key = (
                int(positions[i, 0] // cell_size),
                int(positions[i, 1] // cell_size),
                int(positions[i, 2] // cell_size),
            )
            self._bins.setdefault(key, []).append(i)

    def query_radius(self, pos: np.ndarray, radius: float) -> list[int]:
        """Return candidate indices within the 27-cell neighborhood."""
        cell_size = self._cell_size
        cx, cy, cz = int(pos[0] // cell_size), int(pos[1] // cell_size), int(pos[2] // cell_size)
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (cx + dx, cy + dy, cz + dz)
                    if key in self._bins:
                        candidates.extend(self._bins[key])
        return candidates

    def query_knn(self, pos: np.ndarray, k: int) -> np.ndarray:
        """Return indices of k nearest neighbors via radius query + sort.

        Uses the 27-cell neighbourhood from query_radius, computes
        distances, and returns the k closest (excluding self).
        """
        if self._positions is None:
            return np.array([], dtype=np.int32)

        candidates = self.query_radius(pos, self._cell_size)
        if len(candidates) <= 1:
            return np.array([], dtype=np.int32)

        # Collect candidate positions via numpy advanced indexing
        candidate_pos = self._positions[candidates]
        diffs = candidate_pos - pos
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
    """

    def __init__(self) -> None:
        self._tree: object = None   # scipy.spatial.cKDTree

    @property
    def ready(self) -> bool:
        return self._tree is not None

    def rebuild(self, positions: np.ndarray, active: np.ndarray) -> None:
        """Build cKDTree from active bird positions."""
        from scipy.spatial import cKDTree
        active_pos = positions[active]
        self._tree = cKDTree(active_pos) if len(active_pos) > 0 else None

    def query_knn(self, pos: np.ndarray, k: int) -> np.ndarray:
        """Return indices of k nearest neighbors (excluding self)."""
        if self._tree is None:
            return np.array([], dtype=np.int32)
        _, idx = self._tree.query(pos, k=k + 1)
        return idx[1:] if len(idx) > 1 else idx[:0]
