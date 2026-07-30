"""PhysicsFlock — flock state container with spatial indexing.

Level 1 — assembles FlockArrays + spatial index + integration kernel.
Auto-selects SpatialHashGrid (N < 5K) or KDTreeIndex (N >= 5K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.types import SpatialIndex
from .boid import (
    init_positions,
    init_velocities,
    integrate,
)
from .spatial_index import KDTreeIndex, SpatialHashGrid
from .plugins.spatial_index_strategy import (
    AUTO_INDEX_THRESHOLD,
    SPATIAL_INDEX_STRATEGY_REGISTRY,
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

        # Generic per-boid behavioral state, written by the
        # BoidStateMachine extension (0 = normal/unassigned when the
        # extension is disabled — see boid_state_machine.py for the
        # state-id constants). Eager, fixed-length, matching is_predator's
        # convention rather than the lazy-None pattern used by optional
        # per-boid multiplier arrays, since state is a persistent species-
        # like property once the extension is active.
        self.boid_state: np.ndarray = np.zeros(N, dtype=np.int8)

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

        # Per-bird speed-cap multiplier from the SpeedNoise extension —
        # None when disabled. Separate from max_speed (Predator's boost
        # array) so the two compose multiplicatively rather than one
        # clobbering the other; see integrate().
        self.speed_noise_mult: np.ndarray | None = None

        # Per-bird speed-cap multiplier from the NeighborAdaptiveSpeed
        # extension — None when disabled. Composes multiplicatively with
        # speed_noise_mult/max_speed the same way, chained after
        # speed_noise_mult in integrate().
        self.neighbor_adaptive_speed_mult: np.ndarray | None = None

        # Per-bird speed-cap multiplier from the BoidStateMachine
        # extension — None when disabled. Same multiplicative
        # composition, chained after neighbor_adaptive_speed_mult.
        self.boid_state_speed_mult: np.ndarray | None = None

        # Per-bird predator-threat force, isolated from flock.accelerations
        # when priority_stack_enabled — lets the priority stack budget it
        # as its own tier rather than fusing it into the flocking term.
        # None when the Predator extension is inactive or the priority
        # stack is disabled (default path unaffected).
        self.predator_priority_accel: np.ndarray | None = None

        # Spatial index selection — dispatch through registry
        self._index: SpatialIndex | None
        self._spatial_index_mode = config.spatial_index
        self._visual_range = config.visual_range

        # C3: toroidal-distance box for KDTreeIndex
        kdtree_box: tuple[float, float, float] | None = None
        if config.use_toroidal_distance and config.boundary_mode == "toroidal":
            kdtree_box = (float(config.width), float(config.height), float(config.depth))

        # Dispatch: unrecognised modes fall back to "auto"
        strategy = SPATIAL_INDEX_STRATEGY_REGISTRY.get(
            config.spatial_index,
            SPATIAL_INDEX_STRATEGY_REGISTRY["auto"],
        )
        self._index = strategy(config, N, kdtree_box)

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

        # Modularity pass 10/11: mode-computed per-bird target speed
        # (vicsek's predator/prey v0-vs-v_pred distinction, angle's
        # adaptive deficit-based speed law) -- takes priority as the
        # max_speed BASE that the extension multipliers below compose
        # onto, instead of defaulting to flat config.v0 and silently
        # discarding what the mode itself computed (a real bug this
        # channel fixes: speed_mode="fixed" previously always
        # renormalised to flat v0 regardless of what vicsek/angle wrote
        # to velocities, since nothing ever populated self.max_speed for
        # them). One-shot: cleared after read so a stale array can't
        # leak into a later frame or mode switch.
        max_speed = self.max_speed
        mode_target_speed = getattr(config, '_mode_target_speed', None)
        if mode_target_speed is not None:
            max_speed = mode_target_speed
            config._mode_target_speed = None

        # C3: predator_speed_boost — predator rows get a faster speed cap.
        # Golden-safe fast path: default n_predators=0 means is_predator
        # is all-False, so self.max_speed passes through unchanged.
        predator_boost = config.spatial.predator_speed_boost
        if predator_boost != 1.0 and self.is_predator.any():
            base = max_speed if max_speed is not None else np.full(
                len(self.positions), config.v0, dtype=np.float32,
            )
            max_speed = np.where(
                self.is_predator, base * predator_boost, base,
            ).astype(np.float32)

        # SpeedNoise extension: multiplicative per-boid cap modulation,
        # composes with the predator boost above rather than replacing it.
        if self.speed_noise_mult is not None:
            base = max_speed if max_speed is not None else np.full(
                len(self.positions), config.v0, dtype=np.float32,
            )
            max_speed = (base * self.speed_noise_mult).astype(np.float32)

        # NeighborAdaptiveSpeed extension: same multiplicative composition,
        # chained after speed_noise_mult.
        if self.neighbor_adaptive_speed_mult is not None:
            base = max_speed if max_speed is not None else np.full(
                len(self.positions), config.v0, dtype=np.float32,
            )
            max_speed = (base * self.neighbor_adaptive_speed_mult).astype(np.float32)

        # BoidStateMachine extension: same multiplicative composition,
        # chained after neighbor_adaptive_speed_mult.
        if self.boid_state_speed_mult is not None:
            base = max_speed if max_speed is not None else np.full(
                len(self.positions), config.v0, dtype=np.float32,
            )
            max_speed = (base * self.boid_state_speed_mult).astype(np.float32)

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
            # General velocity damping/friction — mode-agnostic (unlike
            # field_inertia, not gated to one mode at the engine level).
            damping=config.velocity_damping,
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

        if n >= AUTO_INDEX_THRESHOLD and not is_kdtree:
            self._index = KDTreeIndex()
        elif n < AUTO_INDEX_THRESHOLD and is_kdtree:
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

