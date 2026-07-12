"""Unit tests for analysis.metrics — FlockMetrics, MetricsCollector."""

import numpy as np
import pytest

from pymurmur.analysis.metrics import FlockMetrics, MetricsCollector


def test_flock_metrics_defaults():
    """FlockMetrics has sensible defaults."""
    m = FlockMetrics()
    assert m.alpha == 0.0
    assert m.theta == 0.0
    assert m.dispersion == 0.0
    assert m.speed_avg == 0.0
    assert m.h2 is None  # expensive, not computed by default


def test_metrics_collector_snapshot(default_config):
    """snapshot() returns FlockMetrics after collect()."""
    from pymurmur.physics.flock import PhysicsFlock

    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    collector = MetricsCollector()
    collector.collect(flock, 0)

    snap = collector.snapshot()
    assert isinstance(snap, FlockMetrics)
    assert snap.alpha >= 0.0
    assert snap.speed_avg > 0.0


def test_metrics_order_parameter_perfect():
    """All identical velocities → alpha ≈ 1.0."""
    N = 50
    positions = np.zeros((N, 3), dtype=np.float32)
    velocities = np.ones((N, 3), dtype=np.float32)
    velocities[:, 1:] = 0.0  # all point in +x
    active = np.ones(N, dtype=bool)

    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.num_boids = N
    flock = PhysicsFlock(cfg)
    flock.positions = positions
    flock.velocities = velocities
    flock.active = active

    collector = MetricsCollector()
    collector.collect(flock, 0)
    assert collector.snapshot().alpha > 0.95


def test_metrics_presets():
    """All presets are importable."""
    from pymurmur.analysis.presets import PRESETS
    assert len(PRESETS) >= 7
    assert "ball" in PRESETS
    assert "acro" in PRESETS


def test_order_parameter_random():
    """Random velocities → alpha ≈ 0 for large N."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    N = 500
    cfg = SimConfig()
    cfg.num_boids = N
    flock = PhysicsFlock(cfg)

    # Override velocities with random directions
    rng = np.random.default_rng(123)
    rand_dirs = rng.normal(size=(N, 3)).astype(np.float32)
    norms = np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_dirs /= norms
    flock.velocities = rand_dirs * 4.0

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    # Random velocities → order parameter close to 0 for large N
    # Expected alpha ≈ 1/sqrt(N) ≈ 0.045 for N=500
    assert snap.alpha < 0.15
    assert snap.alpha >= 0.0


def test_order_parameter_opposite():
    """Half up, half down → alpha = 0."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    N = 100
    cfg = SimConfig()
    cfg.num_boids = N
    flock = PhysicsFlock(cfg)

    # Half +x, half -x
    half = N // 2
    flock.velocities[:half] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[half:] = np.array([-4.0, 0.0, 0.0], dtype=np.float32)

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    # Equal opposite → sum of normalized vectors = 0 → alpha = 0
    assert snap.alpha == pytest.approx(0.0, abs=1e-6)


def test_dispersion_spread():
    """Birds at corners of domain → high dispersion."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.num_boids = 4
    flock = PhysicsFlock(cfg)

    # Place birds at 4 corners of a cube
    flock.positions = np.array([
        [0, 0, 0],
        [1000, 0, 0],
        [0, 1000, 0],
        [1000, 1000, 0],
    ], dtype=np.float32)
    flock.active = np.ones(4, dtype=bool)

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    # Dispersion should be large — roughly half the domain width
    assert snap.dispersion > 400.0


def test_speed_avg():
    """speed_avg matches np.mean(np.linalg.norm(velocities, axis=1))."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    # Compute expected speed manually
    active_vel = flock.velocities[flock.active]
    expected_speed = float(np.mean(np.linalg.norm(active_vel, axis=1)))

    assert snap.speed_avg == pytest.approx(expected_speed, rel=1e-5)


def test_snapshot_empty_history():
    """snapshot() returns default FlockMetrics when no collect() called."""
    collector = MetricsCollector()
    snap = collector.snapshot()
    assert isinstance(snap, FlockMetrics)
    assert snap.alpha == 0.0
    assert snap.dispersion == 0.0
    # history property should return empty list
    assert collector.history == []


def test_angular_momentum_linear():
    """Straight-line motion produces finite angular momentum."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    N = 50
    cfg = SimConfig()
    cfg.num_boids = N
    flock = PhysicsFlock(cfg)

    # All birds moving in +x from varied positions
    flock.velocities[:] = np.array([4.0, 0.0, 0.0], dtype=np.float32)

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    assert np.isfinite(snap.angular_momentum).all()


def test_dispersion_concentrated():
    """All birds at same point → dispersion = 0."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # All at the same position
    flock.positions[:] = np.array([500.0, 350.0, 200.0], dtype=np.float32)

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    assert snap.dispersion == pytest.approx(0.0, abs=1e-5)


def test_force_avg():
    """force_avg matches manual computation from accelerations."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    # Manual computation
    acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
    expected = float(np.mean(acc_mags))

    assert snap.force_avg == pytest.approx(expected, rel=1e-5)


def test_angular_momentum_circular():
    """Circular motion in XY plane → angular momentum in +Z."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    N = 50
    cfg = SimConfig()
    cfg.num_boids = N
    flock = PhysicsFlock(cfg)

    # Place birds on a circle in XY plane, moving tangentially
    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
    radius = 200.0
    flock.positions[:, 0] = np.cos(angles) * radius + 500
    flock.positions[:, 1] = np.sin(angles) * radius + 350
    flock.positions[:, 2] = 200.0

    # Tangential velocity: perpendicular to position vector
    flock.velocities[:, 0] = -flock.positions[:, 1] + 350  # -py_centered
    flock.velocities[:, 1] = flock.positions[:, 0] - 500   # +px_centered
    flock.velocities[:, 2] = 0.0
    flock.active[:] = True

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    # r × v should point predominantly in +z for CCW motion in XY
    assert snap.angular_momentum[2] > 0
    assert np.isfinite(snap.angular_momentum).all()


def test_power_avg():
    """power_avg matches manual computation of |a·v|."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    collector = MetricsCollector()
    collector.collect(flock, 0)
    snap = collector.snapshot()

    # Manual computation
    accs = flock.accelerations[flock.active]
    vels = flock.velocities[flock.active]
    expected = float(np.mean(np.abs(np.sum(accs * vels, axis=1))))

    assert snap.power_avg == pytest.approx(expected, rel=1e-5)


def test_metrics_zero_active():
    """collect() with zero active birds returns early (no crash)."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.active[:] = False

    collector = MetricsCollector()
    collector.collect(flock, 0)
    # Should not crash; snapshot returns defaults
    snap = collector.snapshot()
    assert snap.alpha == 0.0
    assert snap.speed_avg == 0.0
