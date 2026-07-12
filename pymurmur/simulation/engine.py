"""SimulationEngine — step loop and flock lifecycle.

Level 2 — assembles PhysicsFlock + ExtensionManager + MetricsCollector.
NEVER imports viz or capture. Pure numpy/scipy — fully headless-capable.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..physics.flock import PhysicsFlock
    from ..physics.extensions import ExtensionManager


class SimulationEngine:
    """Orchestrates one simulation timestep.

    Owns the flock, extensions, and metrics. step() is the atomic unit —
    every frame is identical, no special first-frame logic.
    """

    def __init__(self, config: SimConfig) -> None:
        from ..physics.flock import PhysicsFlock
        from ..physics.extensions import ExtensionManager
        from ..analysis.metrics import MetricsCollector

        self.config = config
        self.flock: PhysicsFlock = PhysicsFlock(config)
        self.extensions: ExtensionManager = ExtensionManager(config)
        self.metrics: MetricsCollector = MetricsCollector(config)
        self.frame: int = 0
        self._perf = None  # set by visualizer or manually

    @property
    def perf(self):
        """Optional PerfDiagnostics instance (set externally)."""
        return self._perf

    @perf.setter
    def perf(self, value) -> None:
        self._perf = value

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """One simulation tick: extensions → forces → integrate → metrics."""
        if self._perf is not None:
            t0 = time.perf_counter()

        self.extensions.pre_step(self.flock)
        self.flock.step(self.config, dt)
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
        self.frame = 0
        if self._perf is not None:
            self._perf.reset()
