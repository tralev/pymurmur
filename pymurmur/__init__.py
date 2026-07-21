"""pymurmur — 3D murmuration simulation and visualisation.

Simulate starling flocks at any scale (150 → 300 000 birds)
using interchangeable physics models with optional real-time 3D rendering.

Public API (I7.2):
    from pymurmur import SimConfig, SimulationEngine, Recorder, Simulation
"""

from .capture.recorder import Recorder  # noqa: F401
from .core.config import SimConfig  # noqa: F401
from .simulation.engine import SimulationEngine  # noqa: F401


# P10.5: Simulation facade — single-class entry point for headless runs
class Simulation:
    """Convenience facade for headless simulation and benchmarking.

    Usage:
        sim = pymurmur.Simulation(num_boids=500, mode="spatial", seed=42)
        sim.run(steps=100)
        metrics = sim.metrics_history
        timings = sim.benchmark(flock_size=2000, num_steps=50)
    """

    def __init__(self, **kwargs) -> None:
        cfg = SimConfig(**kwargs)
        self.config = cfg
        self._engine = SimulationEngine(cfg)

    def run(self, steps: int = 0, *, callback=None) -> None:
        """Run headless simulation for `steps` frames."""
        self._engine.run_headless(steps=steps, callback=callback)

    @property
    def metrics_history(self) -> list:
        """Return the list of collected FlockMetrics."""
        return self._engine.metrics.history

    def benchmark(self, flock_size: int = 2000, num_steps: int = 50) -> list[float]:
        """P10.5: Run benchmark and return per-frame timings in seconds.

        Returns list of `num_steps` float values — one wall-clock
        timing per frame (seconds).
        """
        import time
        from copy import copy

        cfg = copy(self.config)
        cfg.num_boids = flock_size
        from .simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)

        # Warm-up: 10 steps
        engine.run_headless(steps=10)

        timings: list[float] = []
        for _ in range(num_steps):
            t0 = time.perf_counter()
            engine.step()
            timings.append(time.perf_counter() - t0)
        return timings
