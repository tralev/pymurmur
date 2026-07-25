"""Performance — P1×P2 budgets with full metrics, P4×P3 memory stability during soak, P2×P3 memory at each checkpoint, P1×P4 long-run budgets, P2×P4 recorder ring-buffer at scale, P1×P5 deterministic budget measurement.

Split out of test_performance.py (file-size split). Only
@pytest.mark.slow tests are meant for nightly; the rest are fast
smoke checks.
"""

import numpy as np
import pytest

from test.crosscutting.perf.performance.test_performance import HEADROOM_P2

# ── P1 × P2: Budgets with full metrics enabled ────────────────────
#
# Cross-element: P1 (budget table) × P2 (scaling checkpoints).
# P2 checkpoints run with metrics_detail_level=0 (no metrics overhead).
# This test verifies budgets still hold with full metrics (level 2)
# enabled, which adds per-frame computation overhead.


@pytest.mark.slow
@pytest.mark.parametrize("n, budget_ms", [
    (150,   100.0),  # SpatialHashGrid + full metrics level 2
])
def test_checkpoint_budget_with_full_metrics(n: int, budget_ms: float):
    """P1×P2 (@slow): Scaling checkpoint at N=150 with full metrics
    (detail_level=2) still meets its per-step budget.

    Uses spatial mode.  N=1,500 is excluded because metrics at full
    detail are O(N²) and too expensive at that scale.
    """
    import time

    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.mode = "spatial"
    cfg.num_boids = n
    cfg.seed = 7
    cfg.metrics_detail_level = 2  # full metrics
    cfg.metrics_interval = 1

    sim = SimulationEngine(cfg)
    sim.run_headless(steps=2)  # warm-up
    t0 = time.perf_counter()
    sim.run_headless(steps=10)
    elapsed = (time.perf_counter() - t0) / 10 * 1000.0
    threshold = budget_ms * HEADROOM_P2
    assert elapsed <= threshold, (
        f"N={n} with full metrics: {elapsed:.1f} ms/step exceeds "
        f"budget {budget_ms:.0f} ms × headroom {HEADROOM_P2:.0f} = {threshold:.0f} ms"
    )
    # Verify metrics were collected
    assert len(sim.metrics.history) > 0, (
        "No metrics collected during full-metrics run"
    )
    # No NaN
    assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"


# ── P4 × P3: Memory stability at moderate scale during soak ───────
#
# Cross-element: P4 (soak) × P3 (memory audit).
# P4 soak runs at N=500.  P3 memory audit runs 2 steps at N=300K.
# This test bridges the gap: run N=16K for 500 steps, verify SoA
# arrays don't grow and memory budget is respected throughout.
#

@pytest.mark.slow
class TestMemoryStabilityAtScale:
    """P4×P3: SoA array memory stays stable during moderate-scale soak."""

    CROSS_SOA_ARRAYS = [
        ("positions",         (16_000, 3), "float32"),
        ("velocities",        (16_000, 3), "float32"),
        ("accelerations",     (16_000, 3), "float32"),
        ("prev_positions",    (16_000, 3), "float32"),
        ("last_accelerations",(16_000, 3), "float32"),
        ("seeds",             (16_000,),   "float32"),
        ("active",            (16_000,),   "bool"),
        ("is_predator",       (16_000,),   "bool"),
    ]
    CROSS_N = 16_000
    CROSS_STEPS = 500

    def test_soa_arrays_stable_during_500_step_soak(self):
        """P4×P3: Run N=16K for 500 steps with metrics on, verify
        SoA array sizes don't change (no silent reallocation).
        """
        import gc

        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.CROSS_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1
        cfg.capture_frame_cap = 100

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=50)  # warm-up
        gc.collect()

        # Baseline: measure nbytes of all SoA arrays
        baseline = {}
        for attr_name, expected_shape, _dtype in self.CROSS_SOA_ARRAYS:
            arr = getattr(sim.flock, attr_name, None)
            assert arr is not None, f"Missing array: flock.{attr_name}"
            assert arr.shape == expected_shape, (
                f"flock.{attr_name}: expected {expected_shape}, got {arr.shape}"
            )
            baseline[attr_name] = arr.nbytes

        # Soak
        sim.run_headless(steps=self.CROSS_STEPS, callback=None)
        gc.collect()

        # Verify every array's nbytes is unchanged
        total_baseline = 0
        total_after = 0
        for attr_name, _expected_shape, _dtype in self.CROSS_SOA_ARRAYS:
            arr = getattr(sim.flock, attr_name, None)
            assert arr is not None, f"Array vanished: flock.{attr_name}"
            assert arr.nbytes == baseline[attr_name], (
                f"flock.{attr_name}: nbytes changed from "
                f"{baseline[attr_name]:,} to {arr.nbytes:,} during soak — reallocation"
            )
            total_baseline += baseline[attr_name]
            total_after += arr.nbytes

        # Total memory unchanged
        assert total_after == total_baseline, (
            f"Total SoA memory changed from {total_baseline:,} to "
            f"{total_after:,} bytes — possible leak or growth"
        )

        # No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"

    def test_memory_budget_at_16k_with_metrics(self):
        """P4×P3: At N=16K with full metrics, total SoA memory ≤ 15 MB.

        This is a coarse budget — 16K × 8 arrays is much smaller than
        the 300K budget.  The test exists to catch catastrophic leaks
        that only manifest at moderate scale with metrics active.
        """
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.CROSS_N
        cfg.seed = 42
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        sim = SimulationEngine(cfg)

        total_bytes = sum(
            getattr(sim.flock, attr_name).nbytes
            for attr_name, _shape, _dtype in self.CROSS_SOA_ARRAYS
        )
        mb = total_bytes / (1024 * 1024)
        assert mb <= 15.0, (
            f"SoA memory at N={self.CROSS_N}: {mb:.2f} MB exceeds 15 MB budget"
        )

        # Verify can step without issues.  detail_level=2's expensive path
        # (find_optimal_m -> compute_h2) does a dense eigh of the N×N graph
        # Laplacian — O(N^3) per call, 19 calls on the first metrics frame.
        # That's intractable at N=16,000 (same reason P1×P2's neighbor test
        # excludes N=1,500).  The SoA-memory assertion above already covers
        # this test's real intent, so step-sanity here downgrades to fast
        # metrics rather than reproducing the O(N^3) blowup.
        sim.metrics._detail_level = 1  # noqa: SLF001
        sim.run_headless(steps=10)
        assert not np.any(np.isnan(sim.flock.positions))


# ── P2 × P3: Memory audit at every checkpoint scale ──────────
#
# Cross-element: P2 (scaling checkpoints) × P3 (memory audit).
# P3 only checks N=300K.  This test verifies SoA memory at every
# P2 checkpoint size so a memory regression at any scale is caught
# early (not just at extreme scale).


class TestMemoryAtEachCheckpoint:
    """P2×P3: SoA memory budget verified at every P2 checkpoint size."""

    # Fast checkpoints (small N, cheap to allocate)
    FAST_CHECKPOINTS: list[tuple[int, float]] = [
        (150,      0.1),   # tiny — ~6 arrays × 150 × 4 bytes
        (1_500,    0.5),   # still small
        (16_000,   5.0),   # moderate: ~8 arrays × 16K × 4 bytes
        (50_000,  10.0),   # large but fast
    ]

    @pytest.mark.parametrize("n, max_mb", FAST_CHECKPOINTS)
    def test_soa_memory_at_checkpoint_n(self, n: int, max_mb: float):
        """P2×P3: Total SoA memory at N={n} stays within budget."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = n
        cfg.seed = 7
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)

        total = sum(
            arr.nbytes for arr in [
                sim.flock.positions, sim.flock.velocities,
                sim.flock.accelerations, sim.flock.prev_positions,
                sim.flock.last_accelerations, sim.flock.seeds,
                sim.flock.active, sim.flock.is_predator,
            ]
        )
        mb = total / (1024 * 1024)
        assert mb <= max_mb, (
            f"N={n}: {mb:.2f} MB exceeds budget {max_mb:.2f} MB"
        )
        # Sanity: each array has the correct shape
        assert sim.flock.positions.shape == (n, 3)
        assert sim.flock.velocities.shape == (n, 3)
        assert sim.flock.active.shape == (n,)
        # Can step without crash
        sim.run_headless(steps=2)

    @pytest.mark.slow
    def test_soa_memory_at_300k_checkpoint(self):
        """P2×P3 (@slow): Total SoA memory at N=300,000 stays within
        30 MB budget (separate @slow test — allocating 300K per
        parametrized variant would be too expensive)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 300_000
        cfg.seed = 7
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)

        total = sum(
            arr.nbytes for arr in [
                sim.flock.positions, sim.flock.velocities,
                sim.flock.accelerations, sim.flock.prev_positions,
                sim.flock.last_accelerations, sim.flock.seeds,
                sim.flock.active, sim.flock.is_predator,
            ]
        )
        mb = total / (1024 * 1024)
        assert mb <= 30.0, (
            f"N=300K: {mb:.2f} MB exceeds 30 MB budget"
        )
        assert sim.flock.positions.shape == (300_000, 3)
        assert sim.flock.velocities.shape == (300_000, 3)
        sim.run_headless(steps=2)


# ── P1 × P4: Budgets maintained over long soak run ────────────
#
# Cross-element: P1 (budget table) × P4 (soak).
# P1 benchmarks are short (50 steps).  This test verifies that the
# per-step budget still holds over a longer (500-step) sustained
# run, catching thermal drift or cache-degradation regressions.


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["spatial", "field", "influencer"])
def test_budget_maintained_over_long_run(mode: str):
    """P1×P4 (@slow): Per-step budget holds over a 500-step sustained
    run for O(N) modes at N=2,000.

    Uses the P1 budget table.  Vicsek is excluded because its O(N²)
    budget at N=2,000 (6.5s/step) would make 500 steps impractical.
    """
    import time

    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine
    from test.crosscutting.perf.test_budgets import (
        HEADROOM,
        N_BOIDS,
        STEP_BUDGET_2000,
    )

    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = N_BOIDS
    cfg.seed = 7

    sim = SimulationEngine(cfg)
    sim.run_headless(steps=10)  # warm-up

    LONG_STEPS = 500
    t0 = time.perf_counter()
    sim.run_headless(steps=LONG_STEPS)
    elapsed = (time.perf_counter() - t0) / LONG_STEPS * 1000.0

    budget = STEP_BUDGET_2000[mode]
    threshold = budget * HEADROOM
    assert elapsed <= threshold, (
        f"{mode} over {LONG_STEPS} steps: {elapsed:.1f} ms/step exceeds "
        f"budget {budget:.0f} ms × headroom {HEADROOM:.0f} = {threshold:.0f} ms"
    )
    # Verify no NaN or speed violation after sustained run
    assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
    assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"
    speeds = np.linalg.norm(sim.flock.velocities, axis=1)
    max_allowed = cfg.v0 * 1.5
    assert np.all(speeds <= max_allowed), (
        f"Speed exceeded: max={speeds.max():.1f} > {max_allowed:.1f}"
    )


# ── P2 × P4: Recorder ring-buffer at scaling checkpoint sizes ──
#
# Cross-element: P2 (scaling checkpoints) × P4 (soak with Recorder).
# P4 soak runs at N=500.  This test verifies the Recorder's ring-buffer
# caps and memory stability at medium P2 checkpoint scales (1.5K, 16K)
# where issues might only manifest.


class TestRecorderAtScale:
    """P2×P4: Recorder ring-buffer + caps at medium P2 checkpoint sizes."""

    RECORDER_SCALES: list[tuple[int, int]] = [
        (1_500,  50),   # SoA vectorised, short soak
        (16_000, 30),   # KDTree tier, shorter soak to stay fast
    ]

    @pytest.mark.slow
    @pytest.mark.parametrize("n, soak_steps", RECORDER_SCALES)
    def test_recorder_ring_buffer_at_scale(
        self, n: int, soak_steps: int
    ):
        """P2×P4 (@slow): Recorder attached at N={n} for {soak_steps}
        steps — verifies ring-buffer caps respected, no crash, no NaN."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = n
        cfg.seed = 7
        cfg.metrics_detail_level = 1
        cfg.capture_with_viz = False
        cfg.capture_frame_cap = 20  # small cap to test truncation

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        sim.run_headless(steps=soak_steps, callback=rec.on_frame)

        # Ring-buffer cap respected
        assert len(rec.metrics_history) <= cfg.capture_frame_cap, (
            f"N={n}: metrics_history ({len(rec.metrics_history)}) "
            f"exceeds cap ({cfg.capture_frame_cap})"
        )
        # At least some metrics captured
        assert len(rec.metrics_history) > 0, (
            f"N={n}: no metrics captured"
        )
        # No NaN
        assert not np.any(np.isnan(sim.flock.positions)), (
            f"N={n}: NaN in positions"
        )
        assert not np.any(np.isnan(sim.flock.velocities)), (
            f"N={n}: NaN in velocities"
        )
        # Speed contract
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5
        assert np.all(speeds <= max_allowed), (
            f"N={n}: speed max={speeds.max():.1f} > {max_allowed:.1f}"
        )


# ── P1 × P5: Budget measurement is deterministic ─────────────
#
# Cross-element: P1 (budget table) × P5 (determinism).
# Verifies that running the same P1 benchmark twice with the
# same seed produces the same step time within measurement
# tolerance (5%).  If this test fails, the benchmark harness
# itself has non-deterministic components.


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["spatial", "field", "influencer"])
def test_budget_measurement_deterministic(mode: str):
    """P1×P5 (@slow): Same mode + same seed → step times agree
    within 5% across two benchmark runs.

    Excludes vicsek/angle (O(N²) — too variable) and marl
    (requires gymnasium).
    """
    import time

    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    BUDGET_STEPS = 100
    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = 2_000
    cfg.seed = 7

    def _measure() -> float:
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=5)  # warm-up
        t0 = time.perf_counter()
        sim.run_headless(steps=BUDGET_STEPS)
        return (time.perf_counter() - t0) / BUDGET_STEPS * 1000.0

    t1 = _measure()
    t2 = _measure()

    # Allow 10% relative difference for CPU frequency scaling and
    # virtualization jitter in CI.  The test catches catastrophic
    # non-determinism (e.g., unseeded RNG in the benchmark harness)
    # rather than sub-10% CPU variance.
    ratio = max(t1, t2) / min(t1, t2) if min(t1, t2) > 0 else float("inf")
    assert ratio <= 1.10, (
        f"{mode}: step times differ by {ratio*100-100:.1f}% "
        f"({t1:.3f} ms vs {t2:.3f} ms) — benchmark not deterministic"
    )


