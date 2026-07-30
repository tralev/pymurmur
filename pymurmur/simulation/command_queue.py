"""CommandQueue and SimulationEngine's command-queue mixin.

File-size split from engine.py — pure extraction, no behavior change.
CommandQueue holds pending live mutations (add/remove/reset/spawn/
clear/pilot); _CommandQueueMixin provides the enqueue_*/drain_commands
public API, mixed into SimulationEngine via multiple inheritance
(mirroring the existing Renderer3D(_RendererVAOMixin, _RendererDrawMixin)
pattern in viz/renderer.py). Expects self.commands (a CommandQueue),
self.flock, and self.config to already be set by
SimulationEngine.__init__ before any of these methods are called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..physics.flock import PhysicsFlock


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


class _CommandQueueMixin:
    """SimulationEngine's enqueue_*/drain_commands methods."""

    # These are set by SimulationEngine.__init__; declared here for mypy.
    commands: CommandQueue
    flock: "PhysicsFlock"
    config: "SimConfig"

    if TYPE_CHECKING:
        # Provided by SimulationEngine itself (engine.py), not this mixin.
        def reset(self) -> None: ...
        def _drain_pilot_commands(self) -> None: ...

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

        # S2.E6: Pilotable flock — force-mode-aware, stays on SimulationEngine
        # itself (not this mixin) so this module never needs to import
        # physics.forces — only simulation.engine may import both
        # physics.flock and physics.forces (I4.2 M3 architecture guard).
        self._drain_pilot_commands()
