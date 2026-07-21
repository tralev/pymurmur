"""SimulationEngine — step loop and flock lifecycle.

Level 2 — assembles PhysicsFlock + ExtensionManager + MetricsCollector.
NEVER imports viz or capture. Pure numpy/scipy — fully headless-capable.

P8.10: Fixed-timestep accumulator — decouples physics from render
framerate.  Accumulates variable frame_dt into a fixed dt_phys bucket,
stepping physics in fixed-size ticks, and computes lerped
render_positions for smooth display at any framerate.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..physics.extensions._base import StepContext
from ..physics.forces import (
    MODE_REGISTRY,
    compute_all_forces,
    mode_needs_index,
)

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..physics.extensions import ExtensionManager
    from ..physics.flock import PhysicsFlock
    from ..physics.obstacles import ObstacleScene


def _apply_influencer_density_init(flock: PhysicsFlock, config: SimConfig) -> None:
    """C4: position_init="influencer_density" composer.

    physics.flock/physics.boid (L0/L1) can't import physics.forces (would
    create an import cycle — see test_architecture.py's forbidden edges),
    so the density-scaled Gaussian init (P7.4) is applied here instead,
    overwriting PhysicsFlock's default position init post-construction.
    """
    if config.position_init != "influencer_density":
        return
    from ..physics.forces.influencer import InfluencerMode
    flock.positions = InfluencerMode.density_init_positions(
        config.num_boids, config.width, config.height, config.depth,
        config, flock.rng,
    )


class CommandQueue:
    """Pending live mutations — drained by engine.step() before integration."""

    def __init__(self) -> None:
        self.pending_add: int = 0
        self.pending_remove: int = 0
        self.pending_reset: bool = False
        # P10.4: Cursor-ray spawning
        self.pending_spawn_bird: list[tuple[float, float, float]] = []
        self.pending_spawn_predator: list[tuple[float, float, float]] = []
        self.pending_clear: bool = False


class SimulationEngine:
    """Orchestrates one simulation timestep.

    Owns the flock, extensions, metrics, and a CommandQueue for live
    mutations. step() is the atomic unit — drains commands, then runs
    extensions → index → forces → integrate → metrics.
    """

    def __init__(self, config: SimConfig) -> None:
        from ..analysis.metrics import MetricsCollector
        from ..physics.extensions import ExtensionManager
        from ..physics.flock import PhysicsFlock

        config.validate()
        self.config = config

        # C6: Apply numba settings once at engine startup
        if config.perf.fastmath:
            try:
                import numba
                numba.config.FASTMATH = True
            except ImportError:
                pass
        if config.perf.num_threads > 0:
            try:
                import numba
                numba.set_num_threads(config.perf.num_threads)
            except ImportError:
                pass

        self.flock: PhysicsFlock = PhysicsFlock(config)
        _apply_influencer_density_init(self.flock, config)
        self.extensions: ExtensionManager = ExtensionManager(config)
        self.metrics: MetricsCollector = MetricsCollector(config)
        self.commands: CommandQueue = CommandQueue()
        self.frame: int = 0
        self._perf = None  # set by visualizer or manually
        self._obstacle_scene: ObstacleScene | None = None  # S6.4: optional ObstacleScene

        # P8.10: Fixed-timestep accumulator
        self._accumulator: float = 0.0
        self.render_positions: np.ndarray | None = None

    @property
    def perf(self):
        """Optional PerfDiagnostics instance (set externally)."""
        return self._perf

    @perf.setter
    def perf(self, value) -> None:
        self._perf = value

    @property
    def obstacle_scene(self):
        """S6.4: Optional ObstacleScene for collision detection/avoidance."""
        return self._obstacle_scene

    @obstacle_scene.setter
    def obstacle_scene(self, value) -> None:
        self._obstacle_scene = value

    # ── Command queue ─────────────────────────────────────────

    def enqueue_add(self, count: int) -> None:
        """Queue boids to be added on the next step()."""
        self.commands.pending_add += count

    def enqueue_remove(self, count: int) -> None:
        """Queue boids to be removed on the next step()."""
        self.commands.pending_remove += count

    def enqueue_reset(self) -> None:
        """Queue a full simulation reset on the next step()."""
        self.commands.pending_reset = True

    def enqueue_spawn(self, position: tuple[float, float, float],
                      is_predator: bool = False) -> None:
        """P10.4: Queue a boid spawn at a specific world position."""
        if is_predator:
            self.commands.pending_spawn_predator.append(position)
        else:
            self.commands.pending_spawn_bird.append(position)

    def enqueue_clear(self) -> None:
        """P10.4: Queue clearing all active boids."""
        self.commands.pending_clear = True

    def drain_commands(self) -> None:
        """Execute all pending add/remove/reset commands.

        Called at the start of step() for headless users, and also
        called by the viz loop on every frame (including paused) so
        that +/- mutations take effect immediately.
        """
        cq = self.commands

        if cq.pending_reset:
            cq.pending_reset = False
            cq.pending_add = 0
            cq.pending_remove = 0
            self.reset()
            return

        if cq.pending_add > 0:
            added = self.flock.add_boids(cq.pending_add, self.config)
            self.config.num_boids = self.flock.N_active
            cq.pending_add -= added

        if cq.pending_remove > 0:
            removed = self.flock.remove_boids(cq.pending_remove)
            self.config.num_boids = self.flock.N_active
            cq.pending_remove -= removed

        # P10.4: Drain cursor-ray spawns
        for pos in cq.pending_spawn_bird:
            self.flock.spawn_at(pos, is_predator=False,
                               v0=self.config.v0, rng=self.flock.rng)
        self.config.num_boids = self.flock.N_active
        cq.pending_spawn_bird.clear()

        for pos in cq.pending_spawn_predator:
            self.flock.spawn_at(pos, is_predator=True,
                               v0=self.config.v0, rng=self.flock.rng)
        self.config.num_boids = self.flock.N_active
        cq.pending_spawn_predator.clear()

        # P10.4: Clear all boids
        if cq.pending_clear:
            self.flock.active[:] = False
            self.config.num_boids = 0
            cq.pending_clear = False

    # ── Step ──────────────────────────────────────────────────

    def step(self, frame_dt: float = 1.0 / 60.0) -> None:
        """Accumulate frame time and step physics in fixed dt_phys ticks.

        P8.10: Clamps frame_dt at 1/20s to avoid spiral-of-death,
        then drains the accumulator in dt_phys-sized physics steps.
        After the physics loop, computes lerped render_positions for
        smooth display at any framerate via the visualizer.
        """
        if self._perf is not None:
            t0 = time.perf_counter()

        dt_phys = max(self.config.dt_phys, 1e-6)  # P8.10: guard against zero

        # Drain command queue (add/remove/reset before physics)
        self.drain_commands()

        # P8.10: Clamp frame_dt and accumulate
        self._accumulator += min(frame_dt, 1.0 / 20.0)

        # Also cap accumulator to avoid spiral if dt_phys is tiny
        self._accumulator = min(self._accumulator, dt_phys * 10.0)

        # Step physics in fixed dt_phys ticks
        while self._accumulator >= dt_phys:
            self._step_physics(dt_phys)
            self._accumulator -= dt_phys

        # P8.10: Compute lerped render positions for smooth display
        alpha = self._accumulator / dt_phys if self._accumulator > 0.0 else 0.0
        if alpha > 0.0 and self.flock.prev_positions is not None:
            self.render_positions = (
                self.flock.prev_positions
                + alpha * (self.flock.positions - self.flock.prev_positions)
            )
        else:
            # Accumulator drained cleanly — positions == prev_positions.
            # Set None so visualizer reads flock.positions directly,
            # saving an array copy per frame.
            self.render_positions = None

        if self._perf is not None:
            ms = (time.perf_counter() - t0) * 1000.0
            self._perf.record_physics(ms)

    def _step_physics(self, dt: float) -> None:
        """Fixed-timestep physics tick (P8.10).

        Extracted from the original step() body — one discrete physics
        integration tick at the fixed dt_phys rate. Called repeatedly
        by step() until the accumulator is drained.
        """
        # P3.2: Per-config field mode time — set from frame counter.
        self.config._field_time = float(self.frame) * dt

        # P7.1: Per-engine influencer tick
        self.config._influencer_tick = (
            float(self.frame)
            * self.config.influencer_tick_rate
            * self.config.influencer_substeps
        )

        # 1. Extensions — pass per-frame context (I5)
        ctx = StepContext(
            frame=self.frame,
            dt=dt,
            rng=self.flock.rng,
            center=self.flock.center,
            config=self.config,
        )
        self.extensions.pre_step(self.flock, ctx)

        # 2. Rebuild spatial index
        if self.flock._index is not None and mode_needs_index(self.config.mode):
            self.flock._index.rebuild(self.flock.positions, self.flock.active)

        # 3. Compute forces
        compute_all_forces(self.flock, self.config)

        # 4. Integrate (stash + physics + centre update)
        # Wire speed_mode from config — spatial.speed_mode controls
        # how speeds are clamped after force application.
        # D11: honour owns_positions — if the mode claims ownership
        # of position updates, pass move=False so integrate() skips
        # the position step (mode handles its own positions).
        mode_cls = MODE_REGISTRY.get(self.config.mode)
        mode_owns_positions = (
            getattr(mode_cls, 'owns_positions', False)
            if mode_cls is not None else False
        )
        # D12: field_inertia is a field-mode parameter — the raw/clamped
        # velocity lerp softens the speed clamp, which would violate the
        # hard speed-band contract of the other modes (P4 acceptance).
        mode_inertia = (
            self.config.field_inertia if self.config.mode == "field" else 0.0
        )
        self.flock.integrate(self.config, dt,
                            speed_mode=self.config.spatial.speed_mode,
                            move=not mode_owns_positions,
                            inertia=mode_inertia)

        # 4b. P6.3: Species collision resolution
        if self.config.mode == "vicsek":
            from ..physics.forces.vicsek import resolve_species_collisions
            resolve_species_collisions(
                self.flock.positions, self.flock.is_predator, self.config,
                self.flock.active,
            )
            # Re-wrap after collision pushes
            W, H, D = self.config.width, self.config.height, self.config.depth
            pos = self.flock.positions
            active = self.flock.active
            for dim, domain in enumerate([W, H, D]):
                col = pos[active, dim]
                col[col < 0] += domain
                col[col >= domain] -= domain
                pos[active, dim] = col

        # 4c. S6.4: Obstacle collision detection + kinematic correction
        collisions_this_step = 0
        if self._obstacle_scene is not None:
            scene = self._obstacle_scene
            if scene.n_shapes > 0:
                active_mask = self.flock.active
                if active_mask.any():
                    act_idx = np.where(active_mask)[0]
                    prev_collisions = scene.collision_count
                    corrected, collided = scene.resolve(
                        self.flock.prev_positions[act_idx],
                        self.flock.positions[act_idx],
                    )
                    collisions_this_step = scene.collision_count - prev_collisions
                    self.flock.positions[act_idx] = corrected

                    # S6.4: Obstacle avoidance steering (applied to velocities)
                    avoid = scene.avoidance_accel(
                        self.flock.positions[act_idx],
                        self.flock.velocities[act_idx],
                        static_weight=self.config.spatial.static_avoid_weight,
                        predictive_weight=self.config.spatial.predictive_avoid_weight,
                        fly_away_max_dist=self.config.spatial.fly_away_max_dist,
                        min_time_to_collide=self.config.spatial.min_time_to_collide,
                    )
                    self.flock.velocities[act_idx] += avoid

        # 5. Metrics
        self.metrics.collect(self.flock, self.frame,
                            collisions_this_step=collisions_this_step)
        self.frame += 1

    def run_headless(
        self,
        steps: int | None = None,
        callback: Callable | None = None,
    ) -> None:
        """Run the simulation loop.

        Args:
            steps: number of steps to run (None = infinite).
            callback: called with (engine) after each step (for Recorder).
        """
        if steps is None:
            while True:
                self.step()
                if callback:
                    callback(self)
        else:
            for _ in range(steps):
                self.step()
                if callback:
                    callback(self)

    def reset(self) -> None:
        """Create a fresh flock and metrics with the same config."""
        from ..analysis.metrics import MetricsCollector
        from ..physics.flock import PhysicsFlock

        self.flock = PhysicsFlock(self.config)
        _apply_influencer_density_init(self.flock, self.config)
        self.metrics = MetricsCollector(self.config)
        self.commands = CommandQueue()
        self.frame = 0
        if self._perf is not None:
            self._perf.reset()
