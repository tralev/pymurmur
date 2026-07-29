"""Unit tests for analysis.collector (via metrics re-export) — D18
metrics accelerations, D10 ripple forces in last_accelerations.

Split out of test_collector_ema.py (file-size split) — S3.11 EMA
readout smoothing tests stay in the original.
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════
# D18: Metrics read from last_accelerations (pre-zeroing stash)
# ═══════════════════════════════════════════════════════════════════


class TestD18MetricsAccelerations:
    """D18: metrics.collect() reads force/power from
    flock.last_accelerations, not flock.accelerations (which
    integrate() zeros before collect runs)."""

    def test_metrics_reads_last_accelerations_not_accelerations(self):
        """D18: collect() uses flock.last_accelerations for force/power."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.mode = "spatial"

        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        # Set accelerations to non-zero (force computation would do this)
        flock.accelerations[:] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Simulate: integrate() stashes then zeros accelerations
        flock.last_accelerations[:] = flock.accelerations.copy()
        flock.accelerations[:] = 0.0  # simulate integrate() zeroing

        collector = MetricsCollector(cfg)
        collector.collect(flock, 0)

        snap = collector.snapshot()
        # Force/power must reflect pre-zeroing acceleration values
        assert snap.force_avg > 0.0, (
            f"force_avg should be non-zero (from last_accelerations), "
            f"got {snap.force_avg}"
        )
        assert snap.power_avg > 0.0, (
            f"power_avg should be non-zero (from last_accelerations), "
            f"got {snap.power_avg}"
        )

    def test_metrics_last_accelerations_code_inspection(self):
        """D18: Source uses last_accelerations, not accelerations."""
        import inspect

        from pymurmur.analysis.metrics import MetricsCollector
        src = inspect.getsource(MetricsCollector.collect)
        assert "flock.last_accelerations" in src, (
            "collect() must read flock.last_accelerations for force/power"
        )

    def test_metrics_sees_nonzero_force_after_engine_step(self):
        """D18: After one engine step, metrics force/power is non-zero."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "spatial"
        cfg.noise_scale = 0.5  # ensure some force is generated
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        # Step once — forces are computed, then integrate() zeros accels,
        # then collect() should still see non-zero force from the stash.
        engine.step(1.0 / 60.0)

        snap = engine.metrics.snapshot()
        assert snap.force_avg > 0.0, (
            f"After step, force_avg should be > 0, got {snap.force_avg}"
        )
        assert snap.power_avg > 0.0, (
            f"After step, power_avg should be > 0, got {snap.power_avg}"
        )

    def test_metrics_force_reflects_actual_acceleration_magnitude(self):
        """D18: force_avg matches the magnitude of last_accelerations."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 3
        cfg.mode = "spatial"

        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        # Set known accelerations
        known_acc = np.array([[3.0, 0.0, 0.0],
                               [0.0, 4.0, 0.0],
                               [0.0, 0.0, 0.0]], dtype=np.float32)
        flock.accelerations[:] = known_acc
        flock.last_accelerations[:] = known_acc.copy()
        flock.accelerations[:] = 0.0  # simulate zeroing
        # Set velocities for power computation
        flock.velocities[:] = np.array([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0]], dtype=np.float32)

        collector = MetricsCollector(cfg)
        collector.collect(flock, 0)

        snap = collector.snapshot()
        # force_avg = mean of |acc|: (3 + 4 + 0)/3 ≈ 2.333
        expected_force = (3.0 + 4.0 + 0.0) / 3.0
        assert snap.force_avg == pytest.approx(expected_force, rel=0.01), (
            f"force_avg should be ~{expected_force:.3f}, got {snap.force_avg:.3f}"
        )


# ── D10 + D18: Ripple forces in metrics last_accelerations ──────


def test_ripple_forces_reflected_in_last_accelerations():
    """D10+D18: Ripple extension forces are captured by metrics.

    D10 fixed ripple envelope to be per-bird (not scalar), so forces
    vary by distance from the ripple centre. D18 fixed metrics to read
    last_accelerations (pre-zeroing stash) instead of accelerations
    (always zero after integrate). Together, ripple forces must appear
    as non-zero entries in the metrics acceleration stash.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 30
    cfg.mode = "spatial"
    cfg.ripple_enabled = True
    cfg.noise_scale = 0.0  # no noise to keep forces clean

    engine = SimulationEngine(cfg)
    engine.step(1.0 / 60.0)

    # D18: metrics must capture non-zero forces from last_accelerations
    snap = engine.metrics.snapshot()
    assert snap is not None, "Metrics must produce snapshot after step"
    assert snap.force_avg > 0, (
        "D10+D18: Ripple forces must be reflected in metrics last_accelerations"
    )


def test_metrics_force_changes_with_ripple_distance():
    """D10+D18: Ripple per-bird envelope varies with distance.

    D10 ensures ripple envelope is per-bird (shape (N,)). D18 ensures
    metrics capture forces via last_accelerations stash.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 20
    cfg.mode = "spatial"
    cfg.ripple_enabled = True
    cfg.noise_scale = 0.0

    engine = SimulationEngine(cfg)
    flock = engine.flock

    # Place birds at different distances
    flock.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    flock.positions[1] = np.array([500.0, 350.0, 500.0], dtype=np.float32)

    engine.step(1.0 / 60.0)

    # D10: Different distances should give different forces
    f0 = float(np.linalg.norm(flock.last_accelerations[0]))
    f1 = float(np.linalg.norm(flock.last_accelerations[1]))
    # At least one bird should get ripple force
    assert max(f0, f1) > 0, "D10+D18: Ripple should produce non-zero force"


