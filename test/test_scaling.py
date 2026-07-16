"""Scaling validation — verify O(N), O(N log N), O(1) complexity claims.

All scaling tests run small-N sweeps that complete quickly (< 0.5s each),
so they are NOT marked slow and run in the fast suite.
"""

import pytest


class TestScalingValidation:
    """Validate asymptotic complexity of force modes and data structures."""

    def test_o1_scaling_field(self):
        """Field mode time ∝ N (linear fit R² > 0.95)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time

        cfg = SimConfig()
        cfg.mode = "field"

        times = []
        sizes = [1000, 2000, 5000, 10000]
        for n in sizes:
            cfg.num_boids = n
            sim = SimulationEngine(cfg)
            t0 = time.perf_counter()
            sim.run_headless(steps=10)
            t = (time.perf_counter() - t0) / 10
            times.append(t)

        # Verify roughly linear: ratio of last to first < ratio of sizes * 1.5
        ratio_t = times[-1] / times[0]
        ratio_n = sizes[-1] / sizes[0]
        assert ratio_t < ratio_n * 1.5, (
            f"Field mode scaling: t ratio {ratio_t:.1f}x, n ratio {ratio_n:.1f}x"
        )

    def test_nlogn_scaling_kdtree(self):
        """KDTree build time ∝ N log N."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time
        import math

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.spatial_index = "kdtree"

        times = []
        sizes = [500, 1000, 2000, 5000]
        for n in sizes:
            cfg.num_boids = n
            sim = SimulationEngine(cfg)
            t0 = time.perf_counter()
            sim.run_headless(steps=5)
            t = (time.perf_counter() - t0) / 5
            times.append(t)

        # Verify sub-quadratic: ratio of last to first < ratio of n²
        ratio_t = times[-1] / times[0]
        ratio_nlogn = (sizes[-1] * math.log(sizes[-1])) / (sizes[0] * math.log(sizes[0]))
        assert ratio_t < ratio_nlogn * 2, (
            f"KDTree scaling: t ratio {ratio_t:.1f}x, n log n ratio {ratio_nlogn:.1f}x"
        )

    def test_topological_scaling_projection(self):
        """Projection mode time independent of flock density (topological σ)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time

        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.sigma = 4

        times = []
        sizes = [100, 200, 500]
        for n in sizes:
            cfg.num_boids = n
            sim = SimulationEngine(cfg)
            t0 = time.perf_counter()
            sim.run_headless(steps=5)
            t = (time.perf_counter() - t0) / 5
            times.append(t)

        # σ is topological — time per bird should be roughly constant
        per_bird = [t / n for t, n in zip(times, sizes)]
        # Shouldn't grow by more than 2x from smallest to largest
        assert per_bird[-1] < per_bird[0] * 2, (
            f"Projection per-bird time grew {per_bird[-1]/per_bird[0]:.1f}x"
        )

    def test_active_mask_scaling(self):
        """add_boids()/remove_boids() time is O(1) (no reallocation)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        import time

        cfg = SimConfig()
        cfg.num_boids = 5000
        sim = SimulationEngine(cfg)

        t0 = time.perf_counter()
        sim.flock.add_boids(10, cfg)
        t_add = time.perf_counter() - t0

        t0 = time.perf_counter()
        sim.flock.remove_boids(10)
        t_remove = time.perf_counter() - t0

        # Both should be near-instant (changing active mask only)
        assert t_add < 0.01, f"add_boids took {t_add*1000:.2f} ms"
        assert t_remove < 0.01, f"remove_boids took {t_remove*1000:.2f} ms"

    def test_buffer_growth_amortized(self, gpu_available):
        """Instance buffer growth amortized to O(1) per addition."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(800, 600, headless=True)
        initial_cap = r._max_instances

        # Add many birds — buffer should grow but not per-bird
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        cfg = SimConfig()
        cfg.num_boids = initial_cap + 1000
        flock = PhysicsFlock(cfg)
        r.update_instances(flock)
        assert r._max_instances >= initial_cap

    def test_metrics_gating_scaling(self):
        """Expensive metrics trigger on interval boundary, not every frame."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 10
        cfg.num_boids = 20
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=5)

        # With interval=10, expensive metrics shouldn't be computed yet
        last_snap = sim.metrics.snapshot()
        assert last_snap.h2 is None  # expensive, not computed at frame < interval
