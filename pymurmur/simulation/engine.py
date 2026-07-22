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
    """C4/S2.E4: position_init="influencer_density" composer.

    physics.flock/physics.boid (L0/L1) can't import physics.forces (would
    create an import cycle — see test_architecture.py's forbidden edges),
    so the density-scaled Gaussian init (P7.4) is applied here instead,
    overwriting PhysicsFlock's default position init post-construction.

    S2.E4: also triggers automatically when config.mode == "influencer"
    and influencer.density_scaled_init is set — a preset shouldn't have
    to separately set position_init="influencer_density" AND
    influencer.density_scaled_init:true for one behaviour.

    S2.E4 "zero initial directions": the auto-trigger path also zeroes
    initial velocities so the first compute() call's blend is driven
    purely by the target pull rather than an arbitrary initial heading.
    Explicit position_init="influencer_density" (C4) is documented to
    override positions only, so an explicit velocity_init (e.g. "drift")
    survives when triggered that way.
    """
    auto_trigger = (
        config.mode == "influencer" and config.influencer_density_scaled_init
    )
    if config.position_init != "influencer_density" and not auto_trigger:
        return
    from ..physics.forces.influencer import InfluencerMode
    flock.positions = InfluencerMode.density_init_positions(
        config.num_boids, config.width, config.height, config.depth,
        config, flock.rng,
    )
    if auto_trigger:
        flock.velocities[:] = 0.0


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
        # S2.E6: pilotable-flock — accumulated per-axis move directions
        # (camera-frame or world-frame, caller's choice) since the last drain.
        self.pending_pilot_move: list[tuple[float, float, float]] = []
        self.pending_pilot_toggle: bool | None = None  # None = no change queued


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

    def enqueue_pilot_move(self, direction: tuple[float, float, float]) -> None:
        """S2.E6: Queue a pilot-point displacement (unit direction vector).

        Scaled by influencer_pilot_speed * unit-scale U * dt when drained.
        A no-op unless config.mode == "influencer" and pilot is enabled.
        """
        self.commands.pending_pilot_move.append(direction)

    def enqueue_pilot_toggle(self, enabled: bool) -> None:
        """S2.E6: Queue enabling/disabling pilot mode on the next step()."""
        self.commands.pending_pilot_toggle = enabled

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

        # S2.E6: Pilotable flock — drain toggle + accumulated move commands
        if self.config.mode == "influencer" and self.config.influencer_pilot_enabled:
            from ..physics.forces.influencer import InfluencerMode, PilotTarget

            pilot = InfluencerMode._pilot
            if pilot is None:
                C = np.array(
                    [self.config.width / 2.0, self.config.height / 2.0,
                     self.config.depth / 2.0],
                    dtype=np.float32,
                )
                pilot = PilotTarget(position=C)
                pilot.active = True  # influencer_pilot_enabled implies "on" by default
                InfluencerMode.set_pilot(pilot)

            # pilot.active only changes on an explicit toggle command —
            # it must persist across frames, not reset every drain.
            if cq.pending_pilot_toggle is not None:
                pilot.active = cq.pending_pilot_toggle
                cq.pending_pilot_toggle = None

            if cq.pending_pilot_move:
                U = 0.4 * min(self.config.width, self.config.height, self.config.depth)
                step_dist = self.config.influencer_pilot_speed * U * self.config.dt_phys
                for direction in cq.pending_pilot_move:
                    d = np.array(direction, dtype=np.float32)
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-10:
                        d_hat = d / d_norm
                        pilot.position = pilot.position + d_hat * step_dist
                        pilot.heading = d_hat
                cq.pending_pilot_move.clear()
        elif cq.pending_pilot_move or cq.pending_pilot_toggle is not None:
            # Queued while pilot mode wasn't active/applicable — drop silently
            # (mirrors other command-queue no-ops, e.g. spawn with N=0 config).
            cq.pending_pilot_move.clear()
            cq.pending_pilot_toggle = None

    # ── Step ──────────────────────────────────────────────────

    def step(self, frame_dt: float = 1.0 / 60.0,
              control: np.ndarray | None = None) -> None:
        """Accumulate frame time and step physics in fixed dt_phys ticks.

        P8.10: Clamps frame_dt at 1/20s to avoid spiral-of-death,
        then drains the accumulator in dt_phys-sized physics steps.
        After the physics loop, computes lerped render_positions for
        smooth display at any framerate via the visualizer.

        D8: control is an optional (N, 3) external per-bird action,
        applied by marl mode (the only mode that currently reads it) —
        the prerequisite seam for the MARL bridge (S7). One-shot: only
        touches config._marl_action when control is explicitly passed
        (set before this step, cleared after), so direct manual mutation
        of config._marl_action by other callers between step() calls is
        left alone. Formalises what MurmurationEnv previously did by
        reaching into engine.config._marl_action directly, and now also
        auto-clears it (the old code had to remember to do that itself).
        """
        if self._perf is not None:
            t0 = time.perf_counter()

        if control is not None:
            self.config._marl_action = control

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

        # D8: one-shot control — clear after use so a stale action isn't
        # silently re-applied if step() is next called without one.
        if control is not None:
            self.config._marl_action = None

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
        # D11: honour owns_positions — if the mode claims ownership
        # of position updates, pass move=False so integrate() skips
        # the position step (mode handles its own positions).
        mode_cls = MODE_REGISTRY.get(self.config.mode)
        mode_owns_positions = (
            getattr(mode_cls, 'owns_positions', False)
            if mode_cls is not None else False
        )
        # D2: each mode may declare its own speed_mode (e.g. vicsek/angle
        # already set exact per-bird velocities directly — "fixed" makes
        # the generic clamp below a documented no-op rather than an
        # implicit one; marl clamps on a different unit scale (v_cap, not
        # v0) and needs "none" so this step doesn't re-clamp it against
        # the wrong reference). Modes that don't declare one (spatial,
        # projection, field) keep the historical, still-configurable
        # config.spatial.speed_mode default.
        mode_speed_mode = (
            getattr(mode_cls, 'speed_mode', None) if mode_cls is not None else None
        ) or self.config.spatial.speed_mode
        # D12: field_inertia is a field-mode parameter — the raw/clamped
        # velocity lerp softens the speed clamp, which would violate the
        # hard speed-band contract of the other modes (P4 acceptance).
        mode_inertia = (
            self.config.field_inertia if self.config.mode == "field" else 0.0
        )
        self.flock.integrate(self.config, dt,
                            speed_mode=mode_speed_mode,
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
        controller: Callable[["SimulationEngine"], np.ndarray | None] | None = None,
    ) -> None:
        """Run the simulation loop.

        Args:
            steps: number of steps to run (None = infinite).
            callback: called with (engine) after each step (for Recorder).
            controller: D8/S7 — called with (engine) *before* each step to
                produce this step's external control action (or None),
                which is passed straight through to step(control=...).
                The MARL bridge scripts use this to drive an external
                policy without the engine importing gymnasium or any
                RL dependency.
        """
        if steps is None:
            while True:
                control = controller(self) if controller else None
                self.step(control=control)
                if callback:
                    callback(self)
        else:
            for _ in range(steps):
                control = controller(self) if controller else None
                self.step(control=control)
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
