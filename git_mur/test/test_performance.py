"""Performance benchmarks — validate FPS/memory budgets per mode and scale.

All tests marked @pytest.mark.slow — run on PR merge or nightly only.
"""

import pytest

pytestmark = pytest.mark.slow


class TestPerformanceBenchmarks:
    """FPS and memory benchmarks for each mode at target scales."""

    def test_bench_150_projection(self, default_config):
        """Projection mode at N=150 within budget (< 16 ms)."""
        from pymurmur.simulation.engine import SimulationEngine
        import time

        sim = SimulationEngine(default_config)
        t0 = time.perf_counter()
        sim.run_headless(steps=10)
        elapsed = (time.perf_counter() - t0) / 10 * 1000  # ms per step
        assert elapsed < 16, f"Projection N=150: {elapsed:.1f} ms > 16 ms budget"

    def test_bench_200_spatial(self):
        """Spatial mode at N=200 within budget (< 16 ms)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 200
        sim = SimulationEngine(cfg)
        t0 = time.perf_counter()
        sim.run_headless(steps=10)
        elapsed = (time.perf_counter() - t0) / 10 * 1000
        assert elapsed < 16, f"Spatial N=200: {elapsed:.1f} ms > 16 ms budget"

    def test_bench_16k_field(self):
        """Field mode at N=16K within budget (< 16 ms)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 16_000
        sim = SimulationEngine(cfg)
        t0 = time.perf_counter()
        sim.run_headless(steps=5)
        elapsed = (time.perf_counter() - t0) / 5 * 1000
        assert elapsed < 16, f"Field N=16K: {elapsed:.1f} ms > 16 ms budget"

    def test_bench_100_vicsek(self):
        """Vicsek mode at N=100 within budget (< 16 ms)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 100
        sim = SimulationEngine(cfg)
        t0 = time.perf_counter()
        sim.run_headless(steps=10)
        elapsed = (time.perf_counter() - t0) / 10 * 1000
        assert elapsed < 16, f"Vicsek N=100: {elapsed:.1f} ms > 16 ms budget"

    def test_bench_200_influencer(self):
        """Influencer mode at N=200 within budget (< 16 ms)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time

        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 200
        sim = SimulationEngine(cfg)
        t0 = time.perf_counter()
        sim.run_headless(steps=10)
        elapsed = (time.perf_counter() - t0) / 10 * 1000
        assert elapsed < 16, f"Influencer N=200: {elapsed:.1f} ms > 16 ms budget"

    def test_memory_150(self, default_config):
        """Memory at N=150 (< 10 MB)."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        # Rough estimate: sum of array nbytes
        total = sum(
            arr.nbytes for arr in [
                sim.flock.positions, sim.flock.velocities,
                sim.flock.accelerations, sim.flock.seeds,
                sim.flock.last_theta, sim.flock.active,
            ]
        )
        mb = total / (1024 * 1024)
        assert mb < 10, f"Memory N=150: {mb:.1f} MB > 10 MB budget"

    def test_memory_16k(self):
        """Memory at N=16K (< 50 MB)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 16_000
        sim = SimulationEngine(cfg)
        total = sum(
            arr.nbytes for arr in [
                sim.flock.positions, sim.flock.velocities,
                sim.flock.accelerations, sim.flock.seeds,
                sim.flock.last_theta, sim.flock.active,
            ]
        )
        mb = total / (1024 * 1024)
        assert mb < 50, f"Memory N=16K: {mb:.1f} MB > 50 MB budget"

    def test_300k_allocation_and_step(self):
        """300K birds: allocates without crash, memory < 30 MB, runs steps."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        modes = ["spatial", "field", "influencer"]  # modes that work at 300K
        for mode in modes:
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = 300_000
            cfg.metrics_detail_level = 0  # no metrics overhead
            sim = SimulationEngine(cfg)

            # Verify memory budget
            total = sum(
                arr.nbytes for arr in [
                    sim.flock.positions, sim.flock.velocities,
                    sim.flock.accelerations, sim.flock.seeds,
                    sim.flock.last_theta, sim.flock.active,
                ]
            )
            mb = total / (1024 * 1024)
            assert mb < 30, f"{mode} N=300K: {mb:.1f} MB > 30 MB budget"

            # Verify can step without crash
            sim.run_headless(steps=2)
            assert sim.flock.N_active > 0

    def test_bit_reproducibility(self):
        """Same seed + same config → identical metrics after 100 steps."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.seed = 42
        cfg.num_boids = 20
        sim1 = SimulationEngine(cfg)
        sim1.run_headless(steps=100)

        cfg2 = SimConfig()
        cfg2.seed = 42
        cfg2.num_boids = 20
        sim2 = SimulationEngine(cfg2)
        sim2.run_headless(steps=100)

        import numpy as np
        assert np.allclose(sim1.flock.positions, sim2.flock.positions)
        assert np.allclose(sim1.flock.velocities, sim2.flock.velocities)
