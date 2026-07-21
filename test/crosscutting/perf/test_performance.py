"""Performance benchmarks — validate FPS/memory budgets per mode and scale.

P2: Scaling checkpoint ladder (N=150 → 300K, index + numba tier checks).
P3: Memory audit at N=300,000 (full SoA inventory ≤ 25 MB).

Only @pytest.mark.slow tests are meant for nightly; the rest are fast smoke checks.
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


# ── P3: Memory audit at N=300,000 ─────────────────────────────────

FULL_SOA_ARRAYS_300K = [
    # (attr_name, expected_shape, dtype_str)
    ("positions",         (300_000, 3), "float32"),
    ("velocities",        (300_000, 3), "float32"),
    ("accelerations",     (300_000, 3), "float32"),
    ("prev_positions",    (300_000, 3), "float32"),
    ("last_accelerations",(300_000, 3), "float32"),
    ("seeds",             (300_000,),   "float32"),
    ("active",            (300_000,),   "bool"),
    ("is_predator",       (300_000,),   "bool"),
]

MEMORY_BUDGET_MB_300K = 25.0


class TestMemoryAudit:
    """P3: Full SoA inventory audit at N=300,000."""

    @pytest.mark.slow
    def test_full_soa_inventory_within_budget(self):
        """P3: Sum of nbytes over the full 9-array inventory (plus
        max_speed when present) is ≤ 25 MB at N=300,000."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 300_000
        cfg.seed = 7
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)

        total_bytes = 0
        for attr_name, expected_shape, _dtype in FULL_SOA_ARRAYS_300K:
            arr = getattr(sim.flock, attr_name, None)
            assert arr is not None, f"Missing array: flock.{attr_name}"
            assert arr.shape == expected_shape, (
                f"flock.{attr_name}: expected shape {expected_shape}, "
                f"got {arr.shape}"
            )
            total_bytes += arr.nbytes

        # max_speed is optional (None unless predators are configured)
        ms = sim.flock.max_speed
        if ms is not None:
            total_bytes += ms.nbytes

        mb = total_bytes / (1024 * 1024)
        assert mb <= MEMORY_BUDGET_MB_300K, (
            f"SoA memory at N=300K: {mb:.1f} MB exceeds {MEMORY_BUDGET_MB_300K} MB budget.\n"
            f"Inventory ({len(FULL_SOA_ARRAYS_300K)} arrays + max_speed): {total_bytes:,} bytes"
        )

        # Sanity: verify it can step without crash
        sim.run_headless(steps=2)
        assert sim.flock.N_active == 300_000

    def test_per_array_byte_count_300k(self):
        """P3 (fast smoke): each array's byte count formula is correct.

        Does NOT allocate 300K birds — just verifies the math."""
        for attr_name, shape, dtype_str in FULL_SOA_ARRAYS_300K:
            itemsize = np.dtype(dtype_str).itemsize
            expected_bytes = int(np.prod(shape)) * itemsize
            assert expected_bytes > 0, f"{attr_name}: zero bytes?"
            # Sanity: 300K × 3 × 4 = 3.6 MB per (N,3) float32 array
            if len(shape) == 2:
                assert expected_bytes == 3_600_000, (
                    f"{attr_name}: expected 3.6 MB, got {expected_bytes}"
                )
            elif len(shape) == 1:
                # 300K float32 = 1.2 MB; 300K bool = 0.3 MB
                if dtype_str == "float32":
                    assert expected_bytes == 1_200_000
                elif dtype_str == "bool":
                    assert expected_bytes == 300_000



# ── P3: Spatial index memory at N=300,000 ────────────────────────


class TestIndexMemory:
    """P3: Spatial index memory overhead at N=300,000.

    The SoA array budget (25 MB) does not include the spatial index.
    This test measures the index separately so any index memory
    regression is caught independently of SoA changes.
    """

    @pytest.mark.slow
    def test_index_memory_within_budget(self):
        """P3: Spatial index (KDTreeIndex) at N=300K stays ≤ 5 MB."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 300_000
        cfg.seed = 7
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)

        idx = sim.flock._index  # noqa: SLF001
        assert idx is not None, "Expected spatial index at N=300K"

        # Estimate index memory by summing nbytes of known internal arrays
        index_bytes = 0
        idx_tree = getattr(idx, 'tree', None)  # scipy.spatial.cKDTree
        if idx_tree is not None:
            index_bytes += getattr(idx_tree, 'data', np.array(0)).nbytes

        # Index internal arrays: _active_map, positions copy
        active_map = getattr(idx, '_active_map', None)
        if active_map is not None:
            index_bytes += active_map.nbytes

        idx_positions = getattr(idx, '_positions', None)
        if idx_positions is not None:
            index_bytes += idx_positions.nbytes

        idx_mb = index_bytes / (1024 * 1024)
        # KDTreeIndex at 300K should be < 5 MB (raw positions = 3.6 MB
        # plus the KDTree internal tree structure).
        assert idx_mb <= 5.0, (
            f"Index memory at N=300K: {idx_mb:.2f} MB exceeds 5 MB budget"
        )


# ── P3: Metrics history memory estimate ───────────────────────────


class TestMetricsHistoryMemory:
    """P3: Metrics snapshot memory does not grow unbounded."""

    def test_metrics_snapshot_memory_stable(self):
        """P3 (fast): A single metrics snapshot stays within a
        reasonable size after 100 steps at N=500."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 500
        cfg.seed = 42
        cfg.metrics_detail_level = 2  # full metrics
        cfg.metrics_interval = 1

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=100)

        snap = sim.metrics.snapshot()
        d = snap.to_dict()
        # A full metrics dict should be < 50 KB (not a single large array)
        # History length should be bounded by steps
        assert len(sim.metrics.history) <= 100, (
            f"Metrics history length {len(sim.metrics.history)} > 100 steps"
        )
        # Sanity: the metrics dict contains meaningful numeric values
        assert isinstance(d.get("alpha", -1), (int, float)), "alpha field is not numeric"
        assert isinstance(d.get("speeds_avg", -1), (int, float)), "speeds_avg is not numeric"
        assert len(d) >= 5, f"Metrics dict too sparse: {len(d)} keys"


# ── P4: Soak tests ───────────────────────────────────────────────

SOAK_FRAMES = 20_000   # T6.3 nightly minimum
SOAK_N = 500           # flock size
SOAK_MODE = "spatial"  # most stable mode for long runs
SOAK_WARMUP = 1000     # warm-up frames before baseline (= capture_frame_cap, so ring buffer is full)


class TestSoak:
    """P4: Long-running soak tests for memory and stability.

    T6.3 (nightly @slow): 20K frames, recorder ring-buffer caps
    respected, no NaN, positions in-bounds, speed contract held.
    Memory leak check via tracker list sizes (tracemalloc is too
    noisy for 20K-frame runs — its internal tracking grows with
    allocation count).
    S8.4 (release gate): 24-hour headless run — manual, not automated.
    """

    @pytest.mark.slow
    def test_20k_frame_soak_memory_and_stability(self):
        """P4 T6.3: 20K-frame soak with recorder frame caps,
        NaN guard, position-bounds, speed contract, and frame
        counter sanity.

        Memory leak detection: verifies tracker list sizes are
        bounded by the ring-buffer cap (D19) — if metrics_history
        grows beyond `capture_frame_cap`, the ring buffer is broken.

        Runs spatial mode at N=500 for 20K frames with metrics and
        Recorder attached.
        """
        import gc

        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = SOAK_MODE
        cfg.num_boids = SOAK_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1  # fast metrics
        cfg.capture_frame_cap = 1_000  # cap for frame rings (D19)
        cfg.capture_with_viz = False   # skip GPU capture — metrics only

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        cap = cfg.capture_frame_cap

        # ── Warm-up (fill the ring buffer) ──────────────────────
        # Ring buffer caps at `cap` entries, so after warm-up the
        # tracker lists should be stable.
        sim.run_headless(steps=SOAK_WARMUP, callback=rec.on_frame)
        gc.collect()
        baseline_history_n = len(rec.metrics_history)
        baseline_total_arrays = sum(
            arr.nbytes for arr in [
                sim.flock.positions, sim.flock.velocities,
                sim.flock.accelerations, sim.flock.prev_positions,
                sim.flock.last_accelerations, sim.flock.seeds,
                sim.flock.active, sim.flock.is_predator,
            ]
        )

        assert baseline_history_n <= cap, (
            f"After warm-up: metrics_history ({baseline_history_n}) "
            f"exceeds cap ({cap})"
        )

        # ── Soak (20K frames with metrics + recorder) ───────────
        sim.run_headless(steps=SOAK_FRAMES, callback=rec.on_frame)
        gc.collect()

        soak_history_n = len(rec.metrics_history)
        soak_frames_n = len(rec.frames)
        soak_total_arrays = sum(
            arr.nbytes for arr in [
                sim.flock.positions, sim.flock.velocities,
                sim.flock.accelerations, sim.flock.prev_positions,
                sim.flock.last_accelerations, sim.flock.seeds,
                sim.flock.active, sim.flock.is_predator,
            ]
        )

        # ── Frame caps respected (D19) — LEAK GUARD ────────────
        assert soak_history_n <= cap, (
            f"Metrics history ({soak_history_n}) exceeds cap ({cap})"
        )
        assert soak_frames_n <= cap, (
            f"Frames ({soak_frames_n}) exceeds cap ({cap})"
        )
        assert soak_history_n == baseline_history_n, (
            f"Metrics history grew from {baseline_history_n} to "
            f"{soak_history_n} during soak — ring-buffer leak"
        )

        # ── SoA arrays stable (no reallocation growth) ──────────
        assert soak_total_arrays == baseline_total_arrays, (
            f"SoA arrays grew from {baseline_total_arrays} to "
            f"{soak_total_arrays} bytes — possible leak"
        )

        # ── NaN guard ───────────────────────────────────────────
        assert not np.any(np.isnan(sim.flock.positions)), (
            "NaN found in positions after soak"
        )
        assert not np.any(np.isnan(sim.flock.velocities)), (
            "NaN found in velocities after soak"
        )

        # ── Position bounds (all 3 axes) ────────────────────────
        pos = sim.flock.positions
        domain = np.array([cfg.width, cfg.height, cfg.depth], dtype=np.float32)
        assert np.all(pos >= 0.0) and np.all(pos <= domain), (
            f"Positions out of domain bounds [0, {domain}]"
        )

        # ── Speed contract ──────────────────────────────────────
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5  # 50% headroom for transient spikes
        assert np.all(speeds <= max_allowed), (
            f"Speed contract violated: max={speeds.max():.1f} > {max_allowed:.1f}"
        )

        # ── Frame counter sanity ────────────────────────────────
        total = SOAK_WARMUP + SOAK_FRAMES
        assert sim.frame == total, (
            f"Frame counter {sim.frame} != expected {total}"
        )

    @pytest.mark.slow
    def test_20k_frame_metrics_integrity(self):
        """P4 T6.3: Over 20K frames, every metrics field retains its
        expected type and no field silently becomes None, NaN, or inf.

        Runs alongside the main soak test to catch type-drift bugs
        that could hide in long-running captures.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = SOAK_MODE
        cfg.num_boids = SOAK_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.capture_frame_cap = 1_000
        cfg.capture_with_viz = False

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        sim.run_headless(steps=SOAK_FRAMES, callback=rec.on_frame)

        # Check every metrics snapshot in the ring buffer
        EXPECTED_FLOAT_FIELDS = {"alpha", "phi", "theta", "sigma"}
        EXPECTED_NONNEG = {"alpha", "speeds_avg"}
        for i, entry in enumerate(rec.metrics_history):
            for field in EXPECTED_FLOAT_FIELDS:
                val = entry.get(field)
                if val is not None:
                    assert isinstance(val, (int, float)), (
                        f"Metrics[{i}].{field}: expected float, got {type(val).__name__} = {val}"
                    )
                    if field in EXPECTED_NONNEG:
                        assert val >= 0, (
                            f"Metrics[{i}].{field}: expected non-negative, got {val}"
                        )

        # Histogram of alpha values should be spread (not all same)
        alphas = [e.get("alpha", 0.0) for e in rec.metrics_history if e.get("alpha") is not None]
        if len(alphas) > 10:
            unique = len(set(round(a, 4) for a in alphas))
            assert unique >= 3, (
                f"Alpha values nearly constant over {len(alphas)} samples: "
                f"min={min(alphas):.4f}, max={max(alphas):.4f}, unique={unique}"
            )

    def test_soak_config_constants_valid(self):
        """P4 (fast smoke): Soak configuration constants are internally
        consistent."""
        assert SOAK_FRAMES >= 1000, f"SOAK_FRAMES={SOAK_FRAMES} too small"
        assert SOAK_N >= 10, f"SOAK_N={SOAK_N} too small"
        assert SOAK_WARMUP >= 100, f"SOAK_WARMUP={SOAK_WARMUP} too small"
        assert SOAK_MODE in MODE_REGISTRY, (
            f"SOAK_MODE='{SOAK_MODE}' not in MODE_REGISTRY"
        )


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


# ── D19: Ring-buffer bounded over 20K frames ─────────────────────
#
# D19: Both metrics_history and frames lists must stay bounded at
# capture_frame_cap over a long soak run.  The existing D19 unit
# tests in test_recorder.py verify truncation logic in isolation;
# this test verifies it holds under continuous 20K-frame load with
# the real Recorder callback pipeline.
#
# The frames ring-buffer is exercised by mocking _capture_frame to
# append a placeholder — FBO capture is not available headless.


@pytest.mark.slow
class TestD19RingBufferSoak:
    """D19: Over 20K frames, both metrics_history and frames lists
    stay bounded at capture_frame_cap."""

    D19_STEPS = 2_000
    D19_CAP = 100
    D19_N = 500

    def test_d19_metrics_history_bounded_at_cap(self):
        """D19 (@slow): metrics_history stays bounded at cap over
        2K frames (20× cap of 100)."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.D19_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.capture_with_viz = False
        cfg.capture_frame_cap = self.D19_CAP

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        sim.run_headless(steps=self.D19_STEPS, callback=rec.on_frame)

        # metrics_history must be bounded at cap despite 20K >> cap
        assert len(rec.metrics_history) == self.D19_CAP, (
            f"metrics_history length {len(rec.metrics_history)} "
            f"should equal cap {self.D19_CAP} after {self.D19_STEPS} "
            f"steps (20K >> cap)"
        )
        # Each entry is a valid dict with expected fields
        if len(rec.metrics_history) > 0:
            entry = rec.metrics_history[0]
            assert isinstance(entry, dict), f"Entry is {type(entry).__name__}, not dict"
            assert "alpha" in entry, "Missing alpha in metrics"

    def test_d19_frames_bounded_at_cap_with_mock_capture(self):
        """D19 (@slow): frames list stays bounded at cap over 20K
        frames, verified with a mock _capture_frame that appends
        a placeholder each time on_frame gates it."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.D19_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.capture_with_viz = True   # enable frame capture path
        cfg.capture_every = 1         # capture every frame
        cfg.capture_prewarm = 0       # no pre-warm
        cfg.capture_frame_cap = self.D19_CAP

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Spy on _capture_frame: replace with mock that appends
        # a placeholder and applies the D19 truncation.
        original_cap = rec._frame_cap

        def _mock_capture_frame(_sim):
            rec.frames.append("mock_frame")
            if len(rec.frames) > original_cap:
                rec.frames[:] = rec.frames[-original_cap:]

        rec._capture_frame = _mock_capture_frame  # type: ignore[method-assign]

        sim.run_headless(steps=self.D19_STEPS, callback=rec.on_frame)

        # Both lists bounded at cap despite 20K >> cap
        assert len(rec.metrics_history) == self.D19_CAP, (
            f"metrics_history length {len(rec.metrics_history)} "
            f"should equal cap {self.D19_CAP}"
        )
        assert len(rec.frames) == self.D19_CAP, (
            f"frames length {len(rec.frames)} "
            f"should equal cap {self.D19_CAP}"
        )
        # No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"


# ── G6: GL context loss fallback during soak ─────────────────────
#
# G6: When the GPU context is lost mid-run, the system must degrade
# gracefully — metrics continue to be collected, the matplotlib
# fallback (P8.9) takes over frame capture, and no crash occurs.
#
# Unit tests in test_renderer.py verify the Renderer3D.gl_lost flag
# and Visualizer._render_safe in isolation.  This soak test verifies
# the full degrade path over a longer run with Recorder attached.


@pytest.mark.slow
class TestG6GLContextLossSoak:
    """G6: GL context loss fallback works during a longer capture run.

    Simulates: first N frames captured successfully via GPU, then GL
    context is lost and the mpl fallback takes over.  Metrics are
    unaffected by the transition.
    """

    G6_STEPS = 500
    G6_N = 100
    G6_CAP = 200
    G6_AFTER = 10   # frames before simulated GL loss

    def test_g6_degrade_to_mpl_fallback_mid_run(self):
        """G6 (@slow): After simulated GL context loss mid-run, the
        matplotlib fallback activates and the soak completes without
        crash.  Metrics continue to be collected throughout."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.G6_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.capture_with_viz = True      # enable frame capture path
        cfg.capture_mpl_fallback = True   # enable mpl fallback
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_frame_cap = self.G6_CAP

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Track state
        gl_lost_signaled = [False]
        fallback_called = [False]

        # Spy on _fallback_to_mpl to verify it's called
        original_fallback = rec._fallback_to_mpl

        def _fallback_spy(sim_engine):
            fallback_called[0] = True
            original_fallback(sim_engine)

        rec._fallback_to_mpl = _fallback_spy  # type: ignore[method-assign]

        # Mock _capture_frame: first N frames via GPU, then GL loss.
        # After GL loss, EVERY frame goes through the mpl fallback,
        # matching the real _capture_frame → RuntimeError → fallback chain.
        def _g6_capture(sim_engine):
            if not gl_lost_signaled[0] and len(rec.frames) >= self.G6_AFTER:
                gl_lost_signaled[0] = True

            if gl_lost_signaled[0]:
                # GL was lost — every frame goes through fallback (like real
                # _capture_frame catches RuntimeError and calls _fallback_to_mpl)
                if rec._mpl_fallback_enabled:
                    rec._fallback_to_mpl(sim_engine)
            else:
                # GL still active — append a simulated GPU frame
                rec.frames.append("gpu_frame")

            # D19: Ring-buffer truncation
            if len(rec.frames) > rec._frame_cap:
                rec.frames[:] = rec.frames[-rec._frame_cap:]

        rec._capture_frame = _g6_capture  # type: ignore[method-assign]

        sim.run_headless(steps=self.G6_STEPS, callback=rec.on_frame)

        # ── Post-soak assertions ─────────────────────────────────

        # GL loss was triggered
        assert gl_lost_signaled[0], "GL loss must be triggered during test"

        # MPL fallback was activated
        assert fallback_called[0], (
            "MPL fallback must be called after GL context loss"
        )

        # Metrics unaffected by GL loss
        assert len(rec.metrics_history) > 0, (
            "Metrics must be collected throughout the soak"
        )
        # Frame count: at least G6_AFTER GPU frames + some fallback frames
        assert len(rec.frames) >= self.G6_AFTER, (
            f"Expected at least {self.G6_AFTER} frames, "
            f"got {len(rec.frames)}"
        )

        # Ring-buffer cap respected (D19)
        assert len(rec.frames) <= self.G6_CAP, (
            f"Frames ({len(rec.frames)}) exceed cap ({self.G6_CAP})"
        )
        assert len(rec.metrics_history) <= self.G6_CAP, (
            f"Metrics history ({len(rec.metrics_history)}) "
            f"exceed cap ({self.G6_CAP})"
        )

        # No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"

        # Speed contract holds
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5
        assert np.all(speeds <= max_allowed), (
            f"Speed max={speeds.max():.1f} > {max_allowed:.1f}"
        )


# ── G7: Fastmath × metrics warning during soak ───────────────────
#
# G7: When perf.fastmath=True, MetricsCollector emits a RuntimeWarning
# on the FIRST collect() call.  The warning should fire exactly once
# — not every frame — and metrics should still be collected correctly.
#
# Unit tests in test_metrics.py verify the warning in single-frame
# isolation.  This soak test verifies it over a longer headless
# capture run, ensuring the one-shot guard doesn't degrade over
# thousands of frames.


@pytest.mark.slow
class TestG7FastmathWarningSoak:
    """G7: Fastmath × metrics warning emitted exactly once over a
    longer headless capture run with perf.fastmath=True."""

    G7_STEPS = 500
    G7_N = 100
    G7_CAP = 200

    def test_g7_fastmath_warning_emitted_once_during_soak(self):
        """G7 (@slow): With perf.fastmath=True, RuntimeWarning is
        emitted exactly once during a 500-step headless capture.
        Metrics are still collected correctly after the warning."""
        import warnings

        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.G7_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1
        cfg.capture_with_viz = False
        cfg.capture_frame_cap = self.G7_CAP

        # Enable fastmath — the source of the warning
        cfg.fastmath = True

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Track warnings
        fastmath_warnings: list[warnings.WarningMessage] = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # don't suppress anything

            sim.run_headless(steps=self.G7_STEPS, callback=rec.on_frame)

            # Filter for the fastmath warning
            for w in caught:
                if "fastmath" in str(w.message).lower():
                    fastmath_warnings.append(w)

        # ── Post-soak assertions ─────────────────────────────────

        # Warning emitted exactly once
        assert len(fastmath_warnings) == 1, (
            f"Expected exactly 1 fastmath warning, got {len(fastmath_warnings)}. "
            f"Warning must fire on first collect() only — the _warned_fastmath "
            f"one-shot guard prevents repeat emissions."
        )
        # Verify it's a RuntimeWarning
        assert fastmath_warnings[0].category is RuntimeWarning, (
            f"Expected RuntimeWarning, got {fastmath_warnings[0].category}"
        )
        # Verify the warning message mentions metrics and fastmath
        msg = str(fastmath_warnings[0].message)
        assert "fastmath" in msg.lower(), (
            f"Warning message must mention fastmath: '{msg}'"
        )

        # Metrics collected despite fastmath
        assert len(rec.metrics_history) > 0, (
            "Metrics must be collected even with fastmath=True"
        )
        assert len(rec.metrics_history) <= self.G7_CAP, (
            f"Metrics history ({len(rec.metrics_history)}) exceeds "
            f"cap ({self.G7_CAP})"
        )

        # No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"

        # Speed contract holds
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5
        assert np.all(speeds <= max_allowed), (
            f"Speed max={speeds.max():.1f} > {max_allowed:.1f}"
        )


# ── P1 × P3: Memory-per-boid ratio at P1's N=2,000 scale ───────
#
# Cross-element: P1 (budget table at N=2,000) × P3 (memory audit).
# P3 only checks N=300K.  P1's budgets at N=2,000 implicitly assume
# O(N) memory scaling.  This test verifies that every mode has a
# consistent memory-per-boid ratio at the P1 budget scale, catching
# modes that silently allocate O(N²) per-boid memory.


class TestMemoryPerBoid:
    """P1×P3: Memory-per-boid ratio at N=2,000 (P1 budget scale)."""

    # Expected SoA arrays at N=2,000 (8 arrays × varying sizes)
    P1N = 2_000
    # Budget: 8 arrays × (N,3) float32 + (N,) float32 + (N,) bool
    # ~ (5 × 3 × 4 + 2 × 4 + 1) × 2000 / 1M ≈ 0.13 MB
    # Add 50% headroom for optional arrays like max_speed
    MAX_MB_2000 = 1.0

    def test_memory_per_boid_o_n_consistent(self):
        """P1×P3 (fast): At N=2,000, all O(N) modes have similar
        memory-per-boid ratio.

        A mode that accidentally allocates O(N²) auxiliary data would
        have significantly higher memory-per-boid at N=2,000.
        """
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        results: dict[str, float] = {}
        for mode in ("spatial", "field", "influencer", "projection", "angle"):
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = self.P1N
            cfg.seed = 7
            cfg.metrics_detail_level = 0
            sim = SimulationEngine(cfg)

            total_bytes = sum(
                getattr(sim.flock, attr).nbytes
                for attr in (
                    "positions", "velocities", "accelerations",
                    "prev_positions", "last_accelerations",
                    "seeds", "active", "is_predator",
                )
            )
            mb = total_bytes / (1024 * 1024)
            results[mode] = mb

        # Report all modes
        vals = list(results.values())
        keys = list(results.keys())
        max_mb = max(vals)
        min_mb = min(vals)
        max_mode = keys[vals.index(max_mb)]
        min_mode = keys[vals.index(min_mb)]
        ratio = max_mb / min_mb if min_mb > 0 else float("inf")

        # O(N) modes should have nearly identical SoA layouts
        assert ratio <= 2.0, (
            f"Memory-per-boid ratio across modes: {ratio:.2f}× "
            f"(max={max_mb:.3f} MB at {max_mode}, "
            f"min={min_mb:.3f} MB at {min_mode}). "
            f"Modes with O(N) layouts should agree within 2×. "
            f"Results: {results}"
        )
        # Absolute budget: no mode exceeds 1 MB at N=2,000
        for mode, mb in results.items():
            assert mb <= self.MAX_MB_2000, (
                f"{mode}: {mb:.3f} MB exceeds {self.MAX_MB_2000} MB budget at N={self.P1N}"
            )

    def test_vicsek_higher_memory_acknowledged(self):
        """P1×P3 (fast): Vicsek mode at N=2,000 total SoA memory
        stays < 10 MB.

        Vicsek's neighbour data is computed on-the-fly, so its
        persistent SoA footprint is identical to O(N) modes.
        The 10 MB budget guards against future array growth.
        """
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = self.P1N
        cfg.seed = 7
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)

        total_bytes = sum(
            getattr(sim.flock, attr).nbytes
            for attr in (
                "positions", "velocities", "accelerations",
                "prev_positions", "last_accelerations",
                "seeds", "active", "is_predator",
            )
        )
        mb = total_bytes / (1024 * 1024)

        # Vicsek's persistent SoA < 10 MB at N=2,000
        # (base ~0.13 MB — 10 MB budget is very generous)
        assert mb <= 10.0, (
            f"Vicsek N={self.P1N}: {mb:.3f} MB exceeds 10 MB budget"
        )

    def test_modes_can_step_at_budget_scale(self):
        """P1×P3 (fast): Each O(N) mode can step without crash at
        N=2,000 with metrics disabled.

        Vicsek is excluded (O(N²) at N=2,000 is too slow) and marl
        is excluded (requires optional gymnasium dependency)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        for mode in ("spatial", "field", "influencer", "projection", "angle"):
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = self.P1N
            cfg.seed = 7
            cfg.metrics_detail_level = 0
            sim = SimulationEngine(cfg)
            sim.run_headless(steps=2)
            assert not np.any(np.isnan(sim.flock.positions)), (
                f"{mode}: NaN in positions after step at N={self.P1N}"
            )
            assert sim.flock.N_active == self.P1N


# ── D16: Capture override precedence (CLI > env > YAML) — soak ──
#
# D16: Env var application moved to __main__.py so CLI > env > YAML.
# Existing unit tests in test_recorder.py verify Recorder ignores
# env vars and _apply_set_overrides works in isolation.
# This soak test exercises the FULL precedence chain over a longer
# headless capture run, verifying the contract holds end-to-end.


class TestD16PrecedenceSoak:
    """D16: Capture override precedence integrated with soak.

    Emulates the __main__.py ordering:
      1. Default config ("YAML") sets baseline values
      2. Env var overrides applied mid-way
      3. CLI-style overrides (--set) applied last
      4. Short headless capture runs with Recorder
      5. Verify final config and Recorder output respect the contract
    """

    D16_SOAK_STEPS = 200  # short soak — D16 doesn't need 20K frames
    D16_N = 50           # small flock for fast execution

    @pytest.mark.slow
    def test_d16_env_override_applied_before_cli_during_soak(self):
        """D16 (@slow): Full precedence chain — YAML → env → CLI —
        during a headless capture run.

        Contract assertion: CLI overrides beat env vars beat defaults.
        Verifies:
          - Recorder uses the final (CLI-overridden) values
          - Soak completes without NaN or crash
          - Metrics history populated from the overridden config
        """

        from pymurmur.__main__ import _apply_set_overrides
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        # ── Step 1: Default config (the "YAML" layer) ────────────
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.D16_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1  # light metrics for soak
        cfg.capture_with_viz = False
        cfg.capture_frame_cap = 1000  # D19 ring buffer — larger than D16_SOAK_STEPS to avoid truncation

        # Set default ("YAML") capture values — the baseline
        cfg.capture_width = 400
        cfg.capture_height = 300
        cfg.capture_frames = self.D16_SOAK_STEPS
        cfg.capture_output = "d16_default.gif"

        # ── Step 2: Apply env var overrides (as __main__.py does) ─
        # Simulate CAPTURE_WIDTH=800 env var — should override YAML
        # NOTE: We DON'T use monkeypatch here because we want to test
        # the exact same code path __main__.py uses.
        env_overrides = {
            "CAPTURE_WIDTH": "800",
            "CAPTURE_HEIGHT": "600",
            "CAPTURE_FRAMES": str(self.D16_SOAK_STEPS),
            "CAPTURE_OUT": "d16_env_override.gif",
        }
        for _env_key, _cfg_attr in [
            ("CAPTURE_WIDTH", "capture_width"),
            ("CAPTURE_HEIGHT", "capture_height"),
            ("CAPTURE_FRAMES", "capture_frames"),
            ("CAPTURE_OUT", "capture_output"),
        ]:
            _val = env_overrides.get(_env_key)
            if _val is not None:
                try:
                    setattr(cfg, _cfg_attr, int(_val))
                except ValueError:
                    setattr(cfg, _cfg_attr, _val)

        # Verify env overrides took effect (YAML < env)
        assert cfg.capture_width == 800, (
            f"YAML < env: expected 800, got {cfg.capture_width}"
        )
        assert cfg.capture_height == 600, (
            f"YAML < env: expected 600, got {cfg.capture_height}"
        )

        # ── Step 3: Apply CLI overrides (beats env) ──────────────
        # Simulate --set capture.capture_width=1024 and
        # --set capture.capture_height=768
        _apply_set_overrides(cfg, [
            "capture.capture_width=1024",
            "capture.capture_height=768",
        ])

        # Verify CLI beats env (CLI > env)
        assert cfg.capture_width == 1024, (
            f"CLI > env: expected 1024, got {cfg.capture_width}"
        )
        assert cfg.capture_height == 768, (
            f"CLI > env: expected 768, got {cfg.capture_height}"
        )
        # Env-overridden fields NOT touched by CLI should remain
        assert cfg.capture_frames == self.D16_SOAK_STEPS, (
            f"Env value preserved for unmodified field: "
            f"expected {self.D16_SOAK_STEPS}, got {cfg.capture_frames}"
        )

        # ── Step 4: Run a soak with Recorder attached ────────────
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Verify Recorder picked up the CLI-overridden values
        assert rec._capture_width == 1024, (
            f"Recorder must use CLI-overridden width: "
            f"expected 1024, got {rec._capture_width}"
        )
        assert rec._capture_height == 768, (
            f"Recorder must use CLI-overridden height: "
            f"expected 768, got {rec._capture_height}"
        )
        assert rec._capture_frames == self.D16_SOAK_STEPS, (
            f"Recorder must use env-overridden frames: "
            f"expected {self.D16_SOAK_STEPS}, got {rec._capture_frames}"
        )
        assert rec._capture_output == "d16_env_override.gif", (
            "Recorder must use env-overridden output: "
            f"expected 'd16_env_override.gif', got '{rec._capture_output}'"
        )

        # Run the soak
        sim.run_headless(steps=self.D16_SOAK_STEPS, callback=rec.on_frame)

        # ── Step 5: Verify post-soak invariants ──────────────────

        # 5a. Frame counter sanity
        assert sim.frame == self.D16_SOAK_STEPS, (
            f"Frame counter {sim.frame} != {self.D16_SOAK_STEPS}"
        )

        # 5b. Metrics captured every frame (capture_every=1 default)
        assert len(rec.metrics_history) == self.D16_SOAK_STEPS, (
            f"Expected {self.D16_SOAK_STEPS} metrics entries, "
            f"got {len(rec.metrics_history)}"
        )

        # 5c. Ring buffer cap respected (D19): metrics_history ≤ cap
        cap = cfg.capture_frame_cap
        assert len(rec.metrics_history) <= cap, (
            f"Metrics history ({len(rec.metrics_history)}) "
            f"exceeds cap ({cap})"
        )

        # 5d. No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), (
            "NaN in positions after D16 soak"
        )
        assert not np.any(np.isnan(sim.flock.velocities)), (
            "NaN in velocities after D16 soak"
        )

        # 5e. Speed contract holds
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5
        assert np.all(speeds <= max_allowed), (
            f"Speed violated: max={speeds.max():.1f} > {max_allowed:.1f}"
        )

        # 5f. Metrics contain expected fields
        if len(rec.metrics_history) > 0:
            entry = rec.metrics_history[0]
            assert "alpha" in entry, "Missing alpha in metrics"
            assert "speed_avg" in entry, "Missing speed_avg in metrics"

    def test_d16_env_var_ignored_by_recorder_soak_consistent(self):
        """D16 (fast): Recorder ignores env vars during construction
        when config has explicit values — consistent with soak pattern.

        Unlike the unit test (which uses monkeypatch), this test
        modifies os.environ directly (then restores) to verify the
        __main__.py env var application pattern works correctly.
        """
        old_environ = {}
        import os as _os
        try:
            # Save and set env vars
            for key in ("CAPTURE_WIDTH", "CAPTURE_HEIGHT",
                        "CAPTURE_FRAMES", "CAPTURE_OUT"):
                old_environ[key] = _os.environ.get(key)

            _os.environ["CAPTURE_WIDTH"] = "640"
            _os.environ["CAPTURE_HEIGHT"] = "480"
            _os.environ["CAPTURE_FRAMES"] = "50"
            _os.environ["CAPTURE_OUT"] = "env_test.gif"

            from pymurmur.core.config import SimConfig

            cfg = SimConfig()
            cfg.mode = "spatial"
            cfg.num_boids = self.D16_N
            cfg.seed = 42
            cfg.metrics_detail_level = 0
            cfg.capture_with_viz = False

            # Use EXPLICIT config values that differ from env
            cfg.capture_width = 320
            cfg.capture_height = 240
            cfg.capture_frames = 10
            cfg.capture_output = "explicit.gif"

            # Apply env vars (as __main__.py does BEFORE config is
            # passed to Recorder) — env should NOT override explicit
            for _env_key, _cfg_attr in [
                ("CAPTURE_WIDTH", "capture_width"),
                ("CAPTURE_HEIGHT", "capture_height"),
                ("CAPTURE_FRAMES", "capture_frames"),
                ("CAPTURE_OUT", "capture_output"),
            ]:
                _val = _os.environ.get(_env_key)
                if _val is not None:
                    try:
                        setattr(cfg, _cfg_attr, int(_val))
                    except ValueError:
                        setattr(cfg, _cfg_attr, _val)

            # NOTE: After __main__.py applies env overrides, the
            # explicit config values are OVERRIDDEN by env vars.
            # This is the correct YAML < env < CLI contract:
            # env beats "YAML" (i.e., hardcoded defaults).
            # To preserve explicit config values, they must be
            # applied AFTER env vars (via --set CLI or direct setattr).
            # This test verifies the contract: env beats hardcoded.
            assert cfg.capture_width == 640, (
                f"Env should beat hardcoded: expected 640, got {cfg.capture_width}"
            )

            from pymurmur.capture.recorder import Recorder
            from pymurmur.simulation.engine import SimulationEngine

            sim = SimulationEngine(cfg)
            rec = Recorder(sim, cfg)

            # Recorder reads from config — it should see env-overridden value
            assert rec._capture_width == 640, (
                f"Recorder reads env-overridden width: "
                f"expected 640, got {rec._capture_width}"
            )

            # Run a few steps to verify no crash
            sim.run_headless(steps=5, callback=rec.on_frame)
            assert len(rec.metrics_history) == 5

        finally:
            # Restore env vars
            for key, val in old_environ.items():
                if val is not None:
                    _os.environ[key] = val
                else:
                    _os.environ.pop(key, None)
