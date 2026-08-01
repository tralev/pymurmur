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
from ..physics.plugins.obstacle_avoidance import OBSTACLE_AVOIDANCE_REGISTRY
from ..physics.priority_stack import allocate_priority_budget
from .command_queue import CommandQueue, _CommandQueueMixin

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..physics.extensions import ExtensionManager
    from ..physics.flock import PhysicsFlock
    from ..physics.obstacles import ObstacleScene


def _apply_influencer_density_init(flock: PhysicsFlock, config: SimConfig) -> None:
    """C4/S2.E4: position_init="influencer_density" composer.

    physics.flock/physics.boid (L0/L1) can't import physics.forces (would
    create an import cycle — see test_architecture_edges_data.py's
    forbidden edges),
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


class SimulationEngine(_CommandQueueMixin):
    """Orchestrates one simulation timestep.

    Owns the flock, extensions, metrics, and a CommandQueue for live
    mutations. step() is the atomic unit — drains commands, then runs
    extensions → index → forces → integrate → metrics.

    enqueue_*/drain_commands (the command-queue API) are provided by
    _CommandQueueMixin (file-size split — see .command_queue).
    """

    def __init__(self, config: SimConfig) -> None:
        from ..analysis.metrics import MetricsCollector
        from ..physics.extensions import ExtensionManager
        from ..physics.flock import PhysicsFlock

        config.validate()
        self.config = config

        # C6: Apply numba settings once at engine startup. fastmath is
        # wired per-kernel at dispatch time (spatial.py's _dispatch_kernels,
        # vicsek_predator.py's resolve_species_collisions) since numba
        # compiles fastmath in at @njit decoration time, not globally.
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

    def _drain_pilot_commands(self) -> None:
        """S2.E6: Pilotable flock — drain toggle + accumulated move commands.

        Force-mode-aware (reaches into InfluencerMode's pilot target), so
        this stays on SimulationEngine itself rather than
        _CommandQueueMixin (see command_queue.py's docstring) — only
        simulation.engine may import both physics.flock and
        physics.forces (I4.2 M3 architecture guard).
        """
        cq = self.commands
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

    def _priority_obstacle_accel(
        self, positions: np.ndarray, velocities: np.ndarray,
    ) -> np.ndarray:
        """priority_stack_enabled: obstacle-avoidance steering force
        (tier 1), computed pre-integrate from this tick's *starting*
        state — not the post-move, post-resolve positions the default
        post-integrate path (4c below) uses. This is a deliberate,
        documented predictive-vs-reactive semantic shift, consistent
        with every other force in the tick being computed from
        tick-start state. Intentionally NOT a refactor of the default
        4c block (no code shared) — duplicating this is a smaller risk
        than touching that proven-stable path via extraction.
        """
        n = len(positions)
        result = np.zeros((n, 3), dtype=np.float32)
        scene = self._obstacle_scene
        if scene is None or scene.n_shapes == 0:
            return result
        active_mask = self.flock.active
        if not active_mask.any():
            return result
        act_idx = np.where(active_mask)[0]
        avoid = OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"].compute_accel(
            scene, positions[act_idx], velocities[act_idx],
            static_weight=self.config.spatial.static_avoid_weight,
            predictive_weight=self.config.spatial.predictive_avoid_weight,
            fly_away_max_dist=self.config.spatial.fly_away_max_dist,
            min_time_to_collide=self.config.spatial.min_time_to_collide,
        )
        result[act_idx] = avoid
        return result

    def _resolve_priority_budget(self, mode_cls, config: SimConfig) -> float:
        """priority_stack_enabled: max_force (0.15) is only the correct
        budget scale for accel-based modes that already self-clamp to
        it (spatial/field/projection). Velocity-direct "fixed"-speed
        modes (vicsek/angle/influencer) turn on a v0-relative scale
        instead. marl explicitly avoids both max_force and v0 — its own
        code notes v0 would be a scale mismatch — using
        marl_velocity_cap * U instead; mirrored here for the same
        reason.
        """
        speed_mode = getattr(mode_cls, 'speed_mode', None)
        if speed_mode == "fixed":
            return config.v0
        if speed_mode == "none":
            U = min(config.width, config.height, config.depth) / 6.0
            return config.marl.marl_velocity_cap * U
        return config.max_force

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

        # priority_stack_enabled: snapshot velocities before pre_step so
        # tier 3 can be derived as a delta after compute_all_forces runs.
        # flock.accelerations is provably zero at this point (integrate()
        # zeroed it at the end of the previous tick, and nothing else
        # touches it before pre_step) — capturing v_before here (not
        # after pre_step) means Wander/Ecology/Ripple's pre_step
        # contributions correctly fold into tier 3 via the delta below,
        # rather than being silently dropped.
        priority_stack_on = self.config.priority_stack_enabled
        v_before = self.flock.velocities.copy() if priority_stack_on else None

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

        # priority_stack_enabled: tier 1 (obstacle avoidance), computed
        # pre-integrate from this tick's starting state — see
        # _priority_obstacle_accel().
        tier1 = (
            self._priority_obstacle_accel(self.flock.positions, self.flock.velocities)
            if priority_stack_on else None
        )

        # 3. Compute forces
        compute_all_forces(self.flock, self.config)

        if priority_stack_on:
            # tier 3: whatever the active mode produced this tick,
            # whether via accelerations (spatial/field/projection) or a
            # direct velocity write (vicsek/angle/influencer/marl) — both
            # are one-frame velocity deltas in this engine's Euler
            # integration convention, so one formula captures either
            # uniformly with no per-mode branching.
            #
            # Known limitation: for owns_positions modes (influencer),
            # compute_all_forces() has already committed this tick's
            # position move by the time we get here — the priority stack
            # can only reallocate the exit velocity carried into next
            # tick, not protect this tick's movement into an obstacle.
            tier3 = self.flock.accelerations + (self.flock.velocities - v_before)
            tier2 = (
                self.flock.predator_priority_accel
                if self.flock.predator_priority_accel is not None
                else np.zeros_like(tier3)
            )
            budget_mode_cls = MODE_REGISTRY.get(self.config.mode)
            budget = self._resolve_priority_budget(budget_mode_cls, self.config)
            assert tier1 is not None  # priority_stack_on (unchanged since tier1 was set) guarantees this
            allocated = allocate_priority_budget(tier1, tier2, tier3, budget)
            self.flock.velocities[:] = v_before
            self.flock.accelerations[:] = allocated

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
            from ..physics.forces.vicsek_predator import resolve_species_collisions
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

                    if not priority_stack_on:
                        # S6.4: Obstacle avoidance steering (applied to
                        # velocities). Skipped when priority_stack_on —
                        # tier 1 already applied this tick's avoidance
                        # pre-integrate; re-applying here would double it.
                        avoid = OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"].compute_accel(
                            scene,
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
