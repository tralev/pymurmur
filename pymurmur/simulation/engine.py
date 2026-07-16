"""SimulationEngine — step loop and flock lifecycle.

Level 2 — assembles PhysicsFlock + ExtensionManager + MetricsCollector.
NEVER imports viz or capture. Pure numpy/scipy — fully headless-capable.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..physics.forces import compute_all_forces, mode_needs_index
from ..physics.extensions._base import StepContext

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..physics.flock import PhysicsFlock
    from ..physics.extensions import ExtensionManager


class CommandQueue:
    """Pending live mutations — drained by engine.step() before integration."""

    def __init__(self) -> None:
        self.pending_add: int = 0
        self.pending_remove: int = 0
        self.pending_reset: bool = False


class SimulationEngine:
    """Orchestrates one simulation timestep.

    Owns the flock, extensions, metrics, and a CommandQueue for live
    mutations. step() is the atomic unit — drains commands, then runs
    extensions → index → forces → integrate → metrics.
    """

    def __init__(self, config: SimConfig) -> None:
        from ..physics.flock import PhysicsFlock
        from ..physics.extensions import ExtensionManager
        from ..analysis.metrics import MetricsCollector

        config.validate()
        self.config = config
        self.flock: PhysicsFlock = PhysicsFlock(config)
        self.extensions: ExtensionManager = ExtensionManager(config)
        self.metrics: MetricsCollector = MetricsCollector(config)
        self.commands: CommandQueue = CommandQueue()
        self.frame: int = 0
        self._perf = None  # set by visualizer or manually

    @property
    def perf(self):
        """Optional PerfDiagnostics instance (set externally)."""
        return self._perf

    @perf.setter
    def perf(self, value) -> None:
        self._perf = value

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

    # ── Step ──────────────────────────────────────────────────

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """One simulation tick: drain commands → extensions → index → forces → integrate → metrics."""
        if self._perf is not None:
            t0 = time.perf_counter()

        # 0. Drain command queue (add/remove/reset before physics)
        self.drain_commands()

        # P3.2: Per-config field mode time — set from frame counter.
        # Each engine has its own frame, so two engines with the same
        # seed produce identical results (no shared class variable).
        self.config._field_time = float(self.frame) * dt

        # 1. Extensions — pass per-frame context (I5)
        ctx = StepContext(
            frame=self.frame,
            dt=dt,
            rng=self.flock.rng,
            center=self.flock.center,
            config=self.config,
        )
        self.extensions.pre_step(self.flock, ctx)

        # 2. Rebuild spatial index (only modes that need it, skip if index is "none")
        if self.flock._index is not None and mode_needs_index(self.config.mode):
            self.flock._index.rebuild(self.flock.positions, self.flock.active)

        # 3. Compute forces — dispatched by mode (physics.forces)
        compute_all_forces(self.flock, self.config)

        # 4. Integrate (stash + physics + centre update)
        self.flock.integrate(self.config, dt)

        # 5. Metrics
        self.metrics.collect(self.flock, self.frame)
        self.frame += 1

        if self._perf is not None:
            ms = (time.perf_counter() - t0) * 1000.0
            self._perf.record_physics(ms)

    def run_headless(
        self,
        steps: int | None = None,
        callback: callable | None = None,
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
        from ..physics.flock import PhysicsFlock
        from ..analysis.metrics import MetricsCollector

        self.flock = PhysicsFlock(self.config)
        self.metrics = MetricsCollector(self.config)
        self.commands = CommandQueue()
        self.frame = 0
        if self._perf is not None:
            self._perf.reset()
