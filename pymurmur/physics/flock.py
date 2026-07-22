"""PhysicsFlock — flock state container with spatial indexing.

Level 1 — assembles FlockArrays + spatial index + integration kernel.
Auto-selects SpatialHashGrid (N < 5K) or KDTreeIndex (N >= 5K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.types import SpatialIndex, min_image
from .boid import (
    init_positions,
    init_velocities,
    integrate,
)

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
        self.rng = np.random.default_rng(config.seed)

        position_mode = config.position_init
        velocity_mode = config.velocity_init
        self.positions = init_positions(
            N, config.width, config.height, config.depth, self.rng,
            mode=position_mode,
            separation=config.boid_size,
        )
        # P4.9: Unified velocity-init dispatch — 6 modes
        self.velocities = init_velocities(
            N, config.v0, self.rng,
            mode=velocity_mode,
            positions=self.positions,
        )
        self.accelerations = np.zeros((N, 3), dtype=np.float32)
        self.seeds = self.rng.uniform(0.0, 1.0, N).astype(np.float32)
        self.last_theta = np.zeros(N, dtype=np.float32)
        self.active = np.ones(N, dtype=bool)

        # Smoothed swarm centre — EMA centroid (P0.5).
        # D1: Initialised to domain centre so sphere/sphere_soft boundary is
        # always centred on C, never origin, even on frame 0 before update_center().
        self.center: np.ndarray = np.array(
            [config.width / 2, config.height / 2, config.depth / 2],
            dtype=np.float32,
        )

        # Wander state — set by Wander extension before force computation (P3.1)
        self.wander_center: np.ndarray | None = None
        self.wander_heading: np.ndarray | None = None

        # Species column — predator/prey flag (P0.6)
        self.is_predator: np.ndarray = np.zeros(N, dtype=bool)
        # C4: n_predators activates the first N birds as predators
        n_pred = config.n_predators
        if n_pred > 0:
            self.is_predator[:min(n_pred, N)] = True

        # Previous-frame stash — render interpolation + MSD (P0.7)
        self.prev_positions: np.ndarray = np.zeros((N, 3), dtype=np.float32)
        self.last_accelerations: np.ndarray = np.zeros((N, 3), dtype=np.float32)

        # P4.8: Coherence gate factor — set by ecology extension at runtime.
        # Default 1.0 = full alignment/cohesion; <1.0 reduces for small flocks.
        # Spatial mode reads this via the force bridge (engine copies to
        # config._coherence_factor before compute_all_forces).
        self.coherence_factor: float = 1.0

        # P8.3: Ring trail position history — rolling buffer (N, K, 3)
        self.position_history: np.ndarray | None = None

        # Per-bird max speed — None uses scalar v0 (P0.8)
        self.max_speed: np.ndarray | None = None

        # Spatial index selection — honor explicit config, fall back to N heuristic
        self._index: SpatialIndex | None
        self._spatial_index_mode = config.spatial_index
        self._visual_range = config.visual_range
        self._index_threshold = 5000
        index_choice = config.spatial_index
        # C3: use_toroidal_distance — periodic boxsize for KDTreeIndex,
        # mirroring SpatialHashGrid's min-image wrapping. None unless
        # the boundary is actually toroidal.
        kdtree_box: tuple[float, float, float] | None = None
        if config.use_toroidal_distance and config.boundary_mode == "toroidal":
            kdtree_box = (float(config.width), float(config.height), float(config.depth))
        if index_choice == "kdtree":
            self._index = KDTreeIndex(box=kdtree_box)
        elif index_choice == "hash_grid":
            self._index = SpatialHashGrid(config)
        elif index_choice == "none":
            # No spatial index — modes that need it must build their own
            self._index = None
        else:  # "auto"
            if N >= self._index_threshold:
                self._index = KDTreeIndex(box=kdtree_box)
            else:
                self._index = SpatialHashGrid(config)

    def integrate(self, config: SimConfig, dt: float,
                  speed_mode: str = "band", move: bool = True,
                  inertia: float = 0.0) -> None:
        """Stash pre-integration state, integrate, reset accelerations, update centre.

        Pure state operation — never imports forces. The caller (engine)
        is responsible for force computation before calling this method.
        D2: speed_mode is wired from the active mode's own declaration,
        falling back to config.spatial.speed_mode if the mode doesn't
        declare one.
        D11: move=False skips position update (modes that own their positions).
        D12: inertia is wired from config.field_inertia.
        """
        # 1. Stash pre-integration state (P0.7)
        self.last_accelerations[:] = self.accelerations
        self.prev_positions[:] = self.positions

        # C3: predator_speed_boost — predator rows get a faster speed cap.
        # Golden-safe fast path: default n_predators=0 means is_predator
        # is all-False, so self.max_speed passes through unchanged.
        max_speed = self.max_speed
        predator_boost = config.spatial.predator_speed_boost
        if predator_boost != 1.0 and self.is_predator.any():
            base = max_speed if max_speed is not None else np.full(
                len(self.positions), config.v0, dtype=np.float32,
            )
            max_speed = np.where(
                self.is_predator, base * predator_boost, base,
            ).astype(np.float32)

        # 2. Integrate
        # C3: boundary_radius_factor — scales the effective sphere radius
        # (default 1.0 is a no-op).
        effective_sphere_radius = (
            config.boundary_sphere_radius * config.boundary_radius_factor
        )
        integrate(
            self.positions, self.velocities, self.accelerations,
            self.active,
            config.width, config.height, config.depth,
            config.v0, config.boundary_mode, dt,
            effective_sphere_radius, config.boundary_avoidance_factor,
            rng=self.rng,
            max_speed=max_speed,
            speed_mode=speed_mode,
            speed_min_factor=config.speed_min_factor,  # P11.5
            # D1: boundary sphere is centred on the DOMAIN centre, not
            # self.center (the EMA centroid tracker used by extensions) —
            # a centroid-centred boundary would follow the flock instead
            # of bounding it. center=None lets integrate() default to C.
            center=None,
            move=move,
            inertia=inertia,
            # S2.B2: one-shot velocity-domain noise set by SpatialMode
            # when spatial.noise_mode=="velocity"; cleared right after
            # use so a stale array can't leak into a later mode switch.
            velocity_noise=getattr(config, '_spatial_velocity_noise', None),
        )
        if hasattr(config, '_spatial_velocity_noise'):
            config._spatial_velocity_noise = None

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
            position_mode = config.position_init
            velocity_mode = config.velocity_init
            self.positions[idx] = init_positions(
                added, config.width, config.height, config.depth, self.rng,
                mode=position_mode,
                separation=config.boid_size,
            )
            # P4.9: Unified velocity-init dispatch for added boids
            self.velocities[idx] = init_velocities(
                added, config.v0, self.rng,
                mode=velocity_mode,
                positions=self.positions[idx],
            )
            self.accelerations[idx] = 0.0

        if added > 0:
            self._reevaluate_index()

        return added

    def spawn_at(
        self, position: tuple[float, float, float],
        is_predator: bool = False,
        v0: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> int:
        """P10.4: Spawn a single boid at a specific world position.

        Finds an inactive slot (extends arrays if needed), activates it,
        sets its position and a cube-distributed velocity:
            limit3((U³ − 0.5) · 2·v0, v0)
        where U³ is a random unit-cube vector [0,1]³.

        The engine passes v0=self.config.v0; the default 4.0 is a safety
        net for direct callers (D6).

        Returns the index, or -1 if no slot could be allocated.
        """
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0:
            self._extend(100)
            inactive = np.where(~self.active)[0]
            if len(inactive) == 0:
                return -1

        idx = inactive[0]
        self.active[idx] = True
        self.is_predator[idx] = is_predator
        self.positions[idx] = np.array(position, dtype=np.float32)
        # P10.4: Cube-distributed velocity — limit3((U³ − 0.5)·2v0, v0)
        # Uniform in [-0.5v0, 0.5v0]³ → scale to cap at v0
        # D20: Cube-velocity law — limit3((U³−0.5)·2v0, v0).
        # Cubing the uniform pushes more probability mass toward ±v0,
        # giving spawned birds a plausible cruising-speed profile.
        rng_obj = rng if rng is not None else self.rng
        raw_vel = (rng_obj.uniform(0, 1, 3).astype(np.float32) ** 3 - 0.5) * 2.0 * v0
        # Clamp magnitude to v0 (limit3)
        mag = np.linalg.norm(raw_vel)
        if mag > v0:
            raw_vel *= v0 / mag
        self.velocities[idx] = raw_vel
        self.accelerations[idx] = 0.0
        self.seeds[idx] = self.rng.uniform(0.0, 1.0)
        self.last_theta[idx] = 0.0
        self._reevaluate_index()
        return int(idx)

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

        # P8.3: Extend position_history if active
        if self.position_history is not None:
            K = self.position_history.shape[1]
            extended_hist = np.zeros((new_size, K, 3), dtype=np.float32)
            extended_hist[:N] = self.position_history
            self.position_history = extended_hist

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
        centre is always initialised to the domain centre (D1).
        EMA: centre ← centre + 0.5 · (centroid − centre).
        """
        if self.active.sum() == 0:
            return

        centroid = self.positions[self.active].mean(axis=0)
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
