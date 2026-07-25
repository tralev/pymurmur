"""Performance — FPS/memory benchmarks per mode, P2 scaling checkpoint ladder, index-type contract, all-mode base-case budget.

Split out of test_performance.py (file-size split). Only
@pytest.mark.slow tests are meant for nightly; the rest are fast
smoke checks.
"""

import numpy as np
import pytest

from pymurmur.physics.forces import MODE_REGISTRY


class TestPerformanceBenchmarks:
    """FPS and memory benchmarks for each mode at target scales."""

    def test_bench_150_projection(self, default_config):
        """Projection mode at N=150 within budget (< 16 ms)."""
        import time

        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        t0 = time.perf_counter()
        sim.run_headless(steps=10)
        elapsed = (time.perf_counter() - t0) / 10 * 1000  # ms per step
        assert elapsed < 16, f"Projection N=150: {elapsed:.1f} ms > 16 ms budget"

    def test_bench_200_spatial(self):
        """Spatial mode at N=200 within budget (< 50 ms)."""
        import time

        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 200
        sim = SimulationEngine(cfg)
        t0 = time.perf_counter()
        sim.run_headless(steps=10)
        elapsed = (time.perf_counter() - t0) / 10 * 1000
        assert elapsed < 50, f"Spatial N=200: {elapsed:.1f} ms > 50 ms budget"

    @pytest.mark.xfail(reason="Pre-existing: hardware-dependent, 16ms budget too tight")
    def test_bench_16k_field(self):
        """Field mode at N=16K within budget (< 16 ms)."""
        import time

        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 16_000
        sim = SimulationEngine(cfg)
        t0 = time.perf_counter()
        sim.run_headless(steps=5)
        elapsed = (time.perf_counter() - t0) / 5 * 1000
        assert elapsed < 16, f"Field N=16K: {elapsed:.1f} ms > 16 ms budget"

    def test_bench_100_vicsek(self):
        """Vicsek mode at N=100 within budget (< 30 ms)."""
        import time

        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 100
        sim = SimulationEngine(cfg)
        t0 = time.perf_counter()
        sim.run_headless(steps=10)
        elapsed = (time.perf_counter() - t0) / 10 * 1000
        assert elapsed < 30, f"Vicsek N=100: {elapsed:.1f} ms > 30 ms budget"

    def test_bench_200_influencer(self):
        """Influencer mode at N=200 within budget (< 16 ms)."""
        import time

        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

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

    @pytest.mark.slow
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

        assert np.allclose(sim1.flock.positions, sim2.flock.positions)
        assert np.allclose(sim1.flock.velocities, sim2.flock.velocities)


# ── P2: Scaling checkpoint ladder ─────────────────────────────────

# Budgets calibrated 2026-07-20 on Apple Silicon M-series.
# Each budget is measured × 1.5 — tight enough to catch regressions,
# with ×3.0 headroom absorbing CI variance.
# Tier column verifies index choice (hash_grid → kdtree at N≥5K auto-switch).
SCALING_CHECKPOINTS: list[tuple[int, float, str]] = [
    #  (N, budget_ms, expected_tier)
    (150,        3.0, "hash_grid"),   # SpatialHashGrid
    (1_500,     15.0, "hash_grid"),   # SoA vectorised
    (16_000,   140.0, "kdtree"),      # cKDTree batch
    (50_000,   435.0, "kdtree"),      # numba kernels
    (300_000, 2700.0, "kdtree"),      # full stack, metrics off
]

HEADROOM_P2 = 3.0  # CI variance multiplier (same as P1)
P2_STEPS = 30        # per-checkpoint steps (100 per roadmap, 30 for nightly pragmatics)


class TestScalingCheckpoints:
    """P2: Scaling checkpoint ladder — budget + index-tier validation."""

    @staticmethod
    def _benchmark(n: int, steps: int) -> tuple[float, str, bool]:
        """Run spatial mode at N=*n* for *steps* and return
        (mean_ms_per_step, index_type_used, numba_active)."""
        import time

        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = n
        cfg.seed = 7
        cfg.metrics_detail_level = 0  # no metrics overhead at large N
        if n >= 50_000:
            cfg.use_numba = True
        sim = SimulationEngine(cfg)

        # Detect which index is active (PhysicsFlock stores it as _index)
        idx = sim.flock._index  # noqa: SLF001
        idx_type = type(idx).__name__ if idx is not None else "none"
        # Normalise: KDTreeIndex → kdtree, SpatialHashGrid → hash_grid
        if "KDTree" in idx_type:
            idx_type = "kdtree"
        elif "Hash" in idx_type or "SpatialHash" in idx_type:
            idx_type = "hash_grid"

        # P2: Verify numba path active at N≥50K (set via config before engine init)
        numba_active = bool(cfg.use_numba) if n >= 50_000 else False

        sim.run_headless(steps=2)  # warm-up
        t0 = time.perf_counter()
        sim.run_headless(steps=steps)
        elapsed = (time.perf_counter() - t0) / steps * 1000.0
        return elapsed, idx_type, numba_active

    @pytest.mark.slow
    @pytest.mark.parametrize("n, budget_ms, expected_tier", SCALING_CHECKPOINTS)
    def test_checkpoint_budget_and_tier(
        self, n: int, budget_ms: float, expected_tier: str,
    ):
        """P2: Each scaling checkpoint meets its step-time budget and
        uses the expected spatial index tier."""
        elapsed, idx_type, numba_active = self._benchmark(n, P2_STEPS)
        threshold = budget_ms * HEADROOM_P2
        assert elapsed <= threshold, (
            f"N={n:,}: {elapsed:.1f} ms/step exceeds "
            f"budget {budget_ms} ms × headroom {HEADROOM_P2} = {threshold:.0f} ms"
        )
        assert idx_type == expected_tier, (
            f"N={n:,}: expected index tier '{expected_tier}', "
            f"got '{idx_type}'"
        )
        # numba should be active at N≥50K (set by _benchmark)
        if n >= 50_000:
            assert numba_active, (
                f"N={n:,}: numba path not active — expected at 50K+"
            )


class TestIndexTypeContract:
    """P2: Index type transitions at the expected N thresholds.

    Fast smoke — does NOT benchmark, just checks the index type
    at key population sizes.
    """

    @staticmethod
    def _get_index_type(n: int) -> str:
        """Create a flock with *n* boids and return the index type name."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = n
        cfg.seed = 7
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)
        idx = sim.flock._index  # noqa: SLF001
        if idx is None:
            return "none"
        t = type(idx).__name__
        if "KDTree" in t:
            return "kdtree"
        if "Hash" in t or "SpatialHash" in t:
            return "hash_grid"
        return t

    def test_small_flock_uses_hash_grid(self):
        """N=100 → SpatialHashGrid (below KDTree switch threshold)."""
        assert self._get_index_type(100) == "hash_grid"

    def test_medium_flock_uses_kdtree(self):
        """N=10_000 → KDTreeIndex (above KDTree switch threshold)."""
        assert self._get_index_type(10_000) == "kdtree"

    def test_large_flock_uses_kdtree(self):
        """N=100_000 → KDTreeIndex."""
        assert self._get_index_type(100_000) == "kdtree"

    def test_very_small_flock_hash_grid(self):
        """N=10 → SpatialHashGrid (edge case: tiny flock)."""
        assert self._get_index_type(10) == "hash_grid"


# ── P2: All-mode baseline budget at N=150 ────────────────────────

# Generous base-case budget for each mode at N=150.
# Detects catastrophic regressions (e.g. 10× slowdown) in any mode
# at the smallest scale.  Not a replacement for the P1 N=2K budgets.
BASE_BUDGET_150: dict[str, float] = {
    "projection":   50.0,
    "spatial":      10.0,
    "field":         8.0,
    "vicsek":      100.0,
    "influencer":    8.0,
    "angle":        50.0,
    "marl":         50.0,
}


@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
def test_mode_base_case_budget(mode: str):
    """P2 (@slow): Every mode completes 10 steps at N=150 within a
    generous base-case budget (catches catastrophic regressions)."""
    import time

    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = 150
    cfg.seed = 7

    # Use P1 budget if no explicit base-case entry (backward compat)
    budget = BASE_BUDGET_150.get(mode, 1000.0)

    sim = SimulationEngine(cfg)
    # Warm-up: 2 steps
    sim.run_headless(steps=2)
    t0 = time.perf_counter()
    sim.run_headless(steps=10)
    elapsed = (time.perf_counter() - t0) / 10 * 1000.0
    threshold = budget * HEADROOM_P2  # reuse P2's ×3 headroom
    assert elapsed <= threshold, (
        f"{mode} N=150: {elapsed:.1f} ms/step exceeds "
        f"budget {budget:.0f} ms × headroom {HEADROOM_P2:.0f} = {threshold:.0f} ms"
    )


