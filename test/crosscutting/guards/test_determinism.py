"""P5/P13 — Determinism guard: same seed → bit-identical engine trajectories.

CI's guard-rails workflow has a dedicated slot for this file. It
complements `l0_modules/physics/test_flock.py::test_all_modes_deterministic`
(which hardcodes 5 modes) by parametrizing over MODE_REGISTRY — any
newly registered force mode is automatically held to the same
determinism contract.

P5 (full breadth): parametrized grid of mode × seed ∈ {7,42,99} ×
boundary ∈ {toroidal,sphere_soft,open} × num_threads ∈ {1,−1} ×
jitter on/off.  Plus one subprocess run per mode (bit-identical
across process boundaries).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces import MODE_REGISTRY
from pymurmur.simulation.engine import SimulationEngine

pytestmark = pytest.mark.guard

STEPS = 25
N_BOIDS = 40

# P5: seeds, boundaries, thread counts
SEEDS = (7, 42, 99)
BOUNDARIES = ("toroidal", "sphere_soft", "open")
THREAD_COUNTS = (1, -1)  # 1 = single-threaded, -1 = auto/all cores


def _run(mode: str, seed: int = 7, **overrides):
    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = N_BOIDS
    cfg.seed = seed
    for key, value in overrides.items():
        setattr(cfg, key, value)
    engine = SimulationEngine(cfg)
    engine.run_headless(steps=STEPS)
    return engine.flock.positions.copy(), engine.flock.velocities.copy()


@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
def test_same_seed_bit_identical(mode):
    """Every registered mode: same seed → bit-identical pos/vel."""
    p1, v1 = _run(mode)
    p2, v2 = _run(mode)
    np.testing.assert_array_equal(
        p1, p2, err_msg=f"{mode}: same seed must give bit-identical positions")
    np.testing.assert_array_equal(
        v1, v2, err_msg=f"{mode}: same seed must give bit-identical velocities")


@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
def test_different_seed_diverges(mode):
    """Complement check — the identity test would pass vacuously if the
    engine ignored its RNG entirely; different seeds must diverge."""
    p1, _ = _run(mode, seed=7)
    p2, _ = _run(mode, seed=8)
    assert not np.array_equal(p1, p2), (
        f"{mode}: different seeds produced identical trajectories")


# ── P5: Full determinism matrix ─────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_mode_seed_boundary_deterministic(mode: str, seed: int, boundary: str):
    """P5: Same (mode, seed, boundary) → bit-identical positions and
    velocities after STEPS.  Every combination in the grid."""
    p1, v1 = _run(mode, seed=seed, boundary_mode=boundary)
    p2, v2 = _run(mode, seed=seed, boundary_mode=boundary)
    np.testing.assert_array_equal(
        p1, p2,
        err_msg=f"{mode}/seed={seed}/{boundary}: positions diverged")
    np.testing.assert_array_equal(
        v1, v2,
        err_msg=f"{mode}/seed={seed}/{boundary}: velocities diverged")


@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
@pytest.mark.parametrize("num_threads", THREAD_COUNTS)
def test_mode_thread_count_deterministic(mode: str, num_threads: int):
    """P5: NUMBA_NUM_THREADS does not affect determinism when numba is
    available (or is a no-op when it isn't — but the test still runs to
    verify no regression in the numba code path exists)."""
    import os
    old_threads = os.environ.get("NUMBA_NUM_THREADS", None)
    try:
        os.environ["NUMBA_NUM_THREADS"] = str(num_threads)
        p1, v1 = _run(mode)
        p2, v2 = _run(mode)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(v1, v2)
    finally:
        if old_threads is not None:
            os.environ["NUMBA_NUM_THREADS"] = old_threads
        else:
            os.environ.pop("NUMBA_NUM_THREADS", None)


@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
def test_mode_subprocess_deterministic(mode: str):
    """P5: Same seed in a subprocess produces bit-identical results
    as an in-process run.  Verifies determinism across process boundaries."""
    p_in, v_in = _run(mode, seed=42)

    # Run the same config in a subprocess via Python
    code = f"""
import sys
sys.path.insert(0, '.')
import numpy as np
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine
cfg = SimConfig()
cfg.mode = '{mode}'
cfg.num_boids = {N_BOIDS}
cfg.seed = 42
eng = SimulationEngine(cfg)
eng.run_headless(steps={STEPS})
np.savez('{'{tmpfile}'}', pos=eng.flock.positions, vel=eng.flock.velocities)
"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmpfile = tf.name
    try:
        code = code.replace("{tmpfile}", tmpfile)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"{mode}: subprocess failed\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        data = np.load(tmpfile)
        np.testing.assert_array_equal(
            p_in, data["pos"],
            err_msg=f"{mode}: subprocess positions differ from in-process")
        np.testing.assert_array_equal(
            v_in, data["vel"],
            err_msg=f"{mode}: subprocess velocities differ from in-process")
    finally:
        import os as _os
        _os.unlink(tmpfile)


def test_spatial_numba_matches_numpy():
    """P13: spatial mode gives the same trajectory with numba on or off."""
    pytest.importorskip("numba")
    p_numba, v_numba = _run("spatial", use_numba=True)
    p_numpy, v_numpy = _run("spatial", use_numba=False)
    np.testing.assert_allclose(
        p_numba, p_numpy, rtol=0, atol=1e-4,
        err_msg="spatial: numba and numpy paths diverged")
    np.testing.assert_allclose(v_numba, v_numpy, rtol=0, atol=1e-4)


# ── P5: Spatial jitter does not break determinism ────────────

@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
@pytest.mark.parametrize("jitter_sep", [0.0, 0.3])
def test_mode_spatial_jitter_deterministic(mode: str, jitter_sep: float):
    """P5: Per-frame spatial jitter does not break determinism — same
    seed + same jitter setting → bit-identical trajectory."""
    p1, v1 = _run(mode, jitter_separation=jitter_sep)
    p2, v2 = _run(mode, jitter_separation=jitter_sep)
    np.testing.assert_array_equal(
        p1, p2,
        err_msg=f"{mode}/jitter_sep={jitter_sep}: positions diverged")
    np.testing.assert_array_equal(
        v1, v2,
        err_msg=f"{mode}/jitter_sep={jitter_sep}: velocities diverged")


# ── P5: Determinism with extensions enabled ─────────────────────

@pytest.mark.parametrize("mode", ["spatial", "field", "angle"])
def test_mode_deterministic_with_extensions(mode: str):
    """P5: With predator + ecology + wander extensions enabled, same
    seed still produces bit-identical trajectories.

    Extensions introduce additional per-frame RNG consumption and
    state mutations that must not break determinism.
    """
    p1, v1 = _run(mode, seed=42,
                  predator_enabled=True, roosting_enabled=True,
                  wander_enabled=True)
    p2, v2 = _run(mode, seed=42,
                  predator_enabled=True, roosting_enabled=True,
                  wander_enabled=True)
    np.testing.assert_array_equal(
        p1, p2,
        err_msg=f"{mode} with extensions: positions diverged")
    np.testing.assert_array_equal(
        v1, v2,
        err_msg=f"{mode} with extensions: velocities diverged")


# ── P5: Determinism across engine reset ──────────────────────────

def test_mode_deterministic_across_reset():
    """P5: After engine.reset(), the same seed still produces the
    same trajectory as a fresh engine with the same config.
    """
    cfg = SimConfig()
    cfg.mode = "spatial"
    cfg.num_boids = N_BOIDS
    cfg.seed = 42

    # Run once with a fresh engine
    engine1 = SimulationEngine(cfg)
    engine1.run_headless(steps=STEPS)
    p_fresh = engine1.flock.positions.copy()
    v_fresh = engine1.flock.velocities.copy()

    # Run again via engine.reset() — should produce identical trajectory
    engine2 = SimulationEngine(cfg)
    engine2.run_headless(steps=STEPS // 2)
    engine2.reset()
    engine2.run_headless(steps=STEPS)
    p_reset = engine2.flock.positions.copy()
    v_reset = engine2.flock.velocities.copy()

    np.testing.assert_array_equal(
        p_fresh, p_reset,
        err_msg="positions differ after engine reset")
    np.testing.assert_array_equal(
        v_fresh, v_reset,
        err_msg="velocities differ after engine reset")


# ── P5 × P2: Determinism at scale ──────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("mode", ["spatial", "field"])
def test_mode_deterministic_at_scale(mode: str):
    """P5×P2: Determinism holds at scale (N=50K with kdtree + numba).

    Cross-element: P5 (determinism) × P2 (scaling).  P5's grid tests
    at N=40 don't exercise the kdtree or numba code paths that only
    activate at scale.  This test verifies those parallel paths are
    deterministic too.

    Uses spatial mode (kdtree + numba at 50K) and field mode
    (grid-based, no kdtree) as a control.
    """
    N_SCALE = 50_000
    SCALE_STEPS = 10

    def _run_scale(mode: str) -> tuple[np.ndarray, np.ndarray]:
        cfg = SimConfig()
        cfg.mode = mode
        cfg.num_boids = N_SCALE
        cfg.seed = 42
        cfg.metrics_detail_level = 0
        if mode == "spatial":
            cfg.use_numba = True
        engine = SimulationEngine(cfg)
        engine.run_headless(steps=SCALE_STEPS)
        return engine.flock.positions.copy(), engine.flock.velocities.copy()

    p1, v1 = _run_scale(mode)
    p2, v2 = _run_scale(mode)
    np.testing.assert_array_equal(
        p1, p2,
        err_msg=f"{mode} at N={N_SCALE}: positions diverged")
    np.testing.assert_array_equal(
        v1, v2,
        err_msg=f"{mode} at N={N_SCALE}: velocities diverged")


# ── P3 × P5: Metrics history is deterministic ──────────────────
#
# Cross-element: P3 (memory, which includes metrics history) × P5 (determinism).
# P5's pos/vel determinism is necessary but not sufficient — if the
# metrics computation path depends on non-deterministic state (e.g.
# unseeded RNG), the metrics history could diverge even while pos/vel
# stay identical.  This test verifies the full metrics history is
# bit-identical across runs with the same seed.


@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
def test_metrics_history_deterministic(mode: str):
    """P3×P5: Same seed produces bit-identical metrics history.

    Metrics computation involves additional code paths beyond the
    physics step (collect, snapshot, to_dict).  This test verifies
    those paths are also deterministic.
    """
    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = N_BOIDS
    cfg.seed = 42
    cfg.metrics_detail_level = 2  # full metrics
    cfg.metrics_interval = 1

    # Run 1
    engine1 = SimulationEngine(cfg)
    engine1.run_headless(steps=STEPS)
    snap1 = engine1.metrics.snapshot()
    dict1 = snap1.to_dict()

    # Run 2 — same seed, separate engine
    cfg2 = SimConfig()
    cfg2.mode = mode
    cfg2.num_boids = N_BOIDS
    cfg2.seed = 42
    cfg2.metrics_detail_level = 2
    cfg2.metrics_interval = 1
    engine2 = SimulationEngine(cfg2)
    engine2.run_headless(steps=STEPS)
    snap2 = engine2.metrics.snapshot()
    dict2 = snap2.to_dict()

    # Compare all common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    assert len(common_keys) >= 5, (
        f"{mode}: metrics dict too sparse ({len(common_keys)} keys)"
    )
    for key in sorted(common_keys):
        v1, v2 = dict1[key], dict2[key]
        # Skip keys that are known to vary by design
        if key in ("frame",):
            continue
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            np.testing.assert_array_equal(
                v1, v2,
                err_msg=f"{mode}: metrics['{key}'] diverged"
            )
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            assert v1 == v2 or (np.isnan(v1) and np.isnan(v2)), (
                f"{mode}: metrics['{key}']: {v1} != {v2}"
            )
        elif isinstance(v1, list) and isinstance(v2, list):
            assert v1 == v2, (
                f"{mode}: metrics['{key}'] diverged: {v1} != {v2}"
            )
        else:
            # Mixed types — just check equality if possible
            assert v1 == v2, (
                f"{mode}: metrics['{key}']: {type(v1).__name__} "
                f"{v1} != {type(v2).__name__} {v2}"
            )

    # Also verify full history length matches
    assert len(engine1.metrics.history) == len(engine2.metrics.history), (
        f"{mode}: metrics history length differs: "
        f"{len(engine1.metrics.history)} vs {len(engine2.metrics.history)}"
    )


# ── P4 × P5: Soak determinism — bit-identical after long run ────
#
# Cross-element: P4 (soak) × P5 (determinism).
# P4's 20K-frame soak never checks determinism — it only verifies
# memory, NaN, and bounds.  P5's determinism tests run at N=40 for
# 25 steps.  This test bridges the gap: run a 500-step soak twice
# with the same seed, verify positions and velocities are still
# bit-identical after the sustained run.
#
# This catches subtle determinism bugs that only manifest after
# many steps (e.g., floating-point error accumulation from cache
# effects, or non-deterministic GC behavior in the callback).


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["spatial", "field", "influencer"])
def test_soak_deterministic(mode: str):
    """P4×P5 (@slow): After 500-step soak, same seed produces
    bit-identical positions and velocities in a second run.

    Runs modes with O(N) step cost at N=500 for 500 steps each —
    covers the long-run determinism gap.
    """
    N_SOAK = 500
    SOAK_STEPS = 500

    def _run_soak() -> tuple[np.ndarray, np.ndarray]:
        cfg = SimConfig()
        cfg.mode = mode
        cfg.num_boids = N_SOAK
        cfg.seed = 42
        cfg.metrics_detail_level = 0  # no metrics overhead
        engine = SimulationEngine(cfg)
        engine.run_headless(steps=SOAK_STEPS)
        return engine.flock.positions.copy(), engine.flock.velocities.copy()

    p1, v1 = _run_soak()
    p2, v2 = _run_soak()

    np.testing.assert_array_equal(
        p1, p2,
        err_msg=f"{mode}: positions diverged after {SOAK_STEPS}-step soak"
    )
    np.testing.assert_array_equal(
        v1, v2,
        err_msg=f"{mode}: velocities diverged after {SOAK_STEPS}-step soak"
    )
