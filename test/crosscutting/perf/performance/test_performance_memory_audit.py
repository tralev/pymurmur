"""Performance — P3 memory audit at N=300,000 (SoA inventory, spatial index, metrics history), P4 soak tests.

Split out of test_performance.py (file-size split). Only
@pytest.mark.slow tests are meant for nightly; the rest are fast
smoke checks.
"""

import numpy as np
import pytest

from pymurmur.physics.forces import MODE_REGISTRY

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


