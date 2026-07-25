"""Unit tests for analysis.metrics — fast per-frame scalar metrics (alpha, dispersion, speed, force, power, angular momentum) + P4.4 physical-unit conversion.

Split out of test_metrics.py (file-size split).
"""

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

    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

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


def test_p44_physical_metrics_computed(default_config):
    """P4.4: Physical metrics — speed_real_ms, force_real_N, energy_J are populated."""
    from pymurmur.simulation.engine import SimulationEngine

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.v0 = 4.0
    cfg.max_force = 5.0
    cfg.metrics_detail_level = 1
    cfg.metrics_interval = 1  # every frame
    cfg.noise_scale = 0.0
    cfg.bird_mass_kg = 0.075
    cfg.cruise_speed_ms = 8.94
    cfg.acc_peak_ms2 = 40.0

    # Use SimulationEngine so metrics collector is wired correctly
    engine = SimulationEngine(cfg)
    for _ in range(10):
        engine.step(1.0 / 60.0)

    m = engine.metrics.snapshot()

    # Physical metrics should be converted from simulation→real units
    # Speed: sim_units × (cruise_speed_ms / v0)
    cfg.cruise_speed_ms / cfg.v0  # 8.94/4.0 = 2.235
    # After simulation settles, mean speed ~ v0 → real speed ~ cruise_speed_ms
    assert m.speed_real_ms >= 0.0, f"speed_real_ms={m.speed_real_ms} should be non-negative"

    # Force: acc_mags * (cruise_speed_ms / v0) * bird_mass_kg
    assert m.force_real_N >= 0.0, f"force_real_N={m.force_real_N} should be non-negative"

    # Energy: 0.5 * bird_mass_kg * speed_real_ms^2
    assert m.energy_J >= 0.0, f"energy_J={m.energy_J} should be non-negative"

    # If speeds are non-zero, forces and energy should also be non-zero
    if m.speed_real_ms > 0.01:
        assert m.energy_J > 0.0, (
            f"energy_J={m.energy_J} should be > 0 when speed_real_ms={m.speed_real_ms}"
        )


def test_p44_physical_metrics_conversion_factors(default_config):
    """P4.4: Physical metrics use correct unit conversion factors."""
    import numpy as np

    from pymurmur.analysis.metrics import FlockMetrics, _compute_physical_metrics

    # Known inputs — velocities/accs point along +x so the per-bird dot
    # product in the power formula reduces to a plain magnitude product.
    speeds = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # sim units
    acc_mags = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)  # sim units
    velocities = np.column_stack([speeds, np.zeros(4), np.zeros(4)]).astype(np.float32)
    accs = np.column_stack([acc_mags, np.zeros(4), np.zeros(4)]).astype(np.float32)
    bird_mass_kg = 0.075
    cruise_speed_ms = 9.0
    acc_peak_ms2 = 40.0
    v0 = 4.0
    max_force = 5.0
    dt = 1.0 / 60.0

    m = FlockMetrics()
    _compute_physical_metrics(m, speeds, acc_mags, velocities, accs, bird_mass_kg,
                              cruise_speed_ms, acc_peak_ms2, v0, max_force, dt)

    # speed_real_ms = mean(speeds) * (cruise_speed_ms / v0)
    expected_speed = 2.5 * (9.0 / 4.0)  # 5.625
    assert m.speed_real_ms == pytest.approx(expected_speed, rel=1e-4)

    # accel_real_ms2 = mean(acc_mags) * (acc_peak_ms2 / max_force)
    expected_accel = 1.25 * (40.0 / 5.0)  # 10.0
    assert m.accel_real_ms2 == pytest.approx(expected_accel, rel=1e-4)

    # force_real_N = accel_real_ms2 * bird_mass_kg
    expected_force = expected_accel * bird_mass_kg  # 0.75
    assert m.force_real_N == pytest.approx(expected_force, rel=1e-4)

    # S2.B4: power_real_W = mass * mean(|k_a*a_i * k_v*v_i|) (per-bird, aligned here)
    k_v = cruise_speed_ms / v0
    k_a = acc_peak_ms2 / max_force
    expected_power = bird_mass_kg * float(np.mean(acc_mags * k_a * speeds * k_v))
    assert m.power_real_W == pytest.approx(expected_power, rel=1e-4)

    # S2.B4: energy_J = power_real_W * dt (work done this frame)
    expected_energy = expected_power * dt
    assert m.energy_J == pytest.approx(expected_energy, rel=1e-4)


def test_p44_physical_metrics_zero_v0_guarded(default_config):
    """P4.4: Physical metrics return early when v0 <= 0 or max_force <= 0."""
    from pymurmur.analysis.metrics import FlockMetrics, _compute_physical_metrics

    speeds = np.array([1.0], dtype=np.float32)
    acc_mags = np.array([0.5], dtype=np.float32)
    velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    accs = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    dt = 1.0 / 60.0

    # v0=0 → should return without setting physical fields
    m = FlockMetrics()
    _compute_physical_metrics(m, speeds, acc_mags, velocities, accs, 0.075, 9.0, 40.0, 0.0, 5.0, dt)
    assert m.force_real_N == 0.0, "Should guard against v0=0"

    m2 = FlockMetrics()
    _compute_physical_metrics(m2, speeds, acc_mags, velocities, accs, 0.075, 9.0, 40.0, 4.0, 0.0, dt)
    assert m2.force_real_N == 0.0, "Should guard against max_force=0"


def test_p44_physical_metrics_power_is_mean_per_bird_dot_product(default_config):
    """S2.B4: power_real_W = m * mean(|k_a*a_i · k_v*v_i|) — a mean of
    per-bird dot products, NOT force_real_N × speed_real_ms (a product of
    means). The two formulas diverge whenever acceleration and velocity
    directions aren't perfectly correlated across birds — this test uses
    per-bird vectors at varying angles to prove the distinction.

    Verifies the complete physical-metrics chain:
      speed_real_ms  = mean(|v|) × (cruise / v0)
      accel_real_ms2 = mean(|a|) × (acc_peak / max_force)
      force_real_N   = accel_real_ms2 × bird_mass_kg
      power_real_W   = m × mean(|k_a·a_i · k_v·v_i|)
      energy_J       = power_real_W × dt
    """
    import numpy as np

    from pymurmur.analysis.metrics import FlockMetrics, _compute_physical_metrics

    # Two birds: one with velocity/acceleration aligned (full dot product),
    # one with them perpendicular (zero dot product) — same magnitudes as
    # the aligned case, so a product-of-means formula would give a
    # different (larger, nonzero) answer than the mean-of-dot-products.
    velocities = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    accs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    speeds = np.linalg.norm(velocities, axis=1)
    acc_mags = np.linalg.norm(accs, axis=1)
    bird_mass_kg = 0.08
    cruise_speed_ms = 10.0
    acc_peak_ms2 = 50.0
    v0 = 5.0
    max_force = 5.0
    dt = 1.0 / 60.0

    m = FlockMetrics()
    _compute_physical_metrics(m, speeds, acc_mags, velocities, accs, bird_mass_kg,
                              cruise_speed_ms, acc_peak_ms2, v0, max_force, dt)

    cruise_speed_ms / v0  # 2.0
    acc_peak_ms2 / max_force  # 10.0

    # speed_real = mean([2,2]) * 2.0 = 4.0
    assert m.speed_real_ms == pytest.approx(4.0, rel=1e-4)
    # accel_real = mean([1,1]) * 10.0 = 10.0
    assert m.accel_real_ms2 == pytest.approx(10.0, rel=1e-4)
    # force = 10.0 * 0.08 = 0.8
    assert m.force_real_N == pytest.approx(0.8, rel=1e-4)

    # Bird 0: (a·k_a)·(v·k_v) = (10,0,0)·(4,0,0) = 40. Bird 1: (10,0,0)·(0,4,0) = 0.
    # mean(|dot|) = 20.0 → power = 0.08 * 20.0 = 1.6 W
    expected_power = bird_mass_kg * 20.0
    assert m.power_real_W == pytest.approx(expected_power, rel=1e-4), (
        f"power_real_W={m.power_real_W:.4f} should be {expected_power} "
        "(mean of per-bird dot products)"
    )
    # This must NOT equal force_real_N * speed_real_ms (the old, wrong formula)
    wrong_power = m.force_real_N * m.speed_real_ms  # 0.8 * 4.0 = 3.2
    assert m.power_real_W != pytest.approx(wrong_power, rel=1e-4), (
        "power_real_W must diverge from force_real_N*speed_real_ms when "
        "per-bird a/v directions aren't correlated"
    )

    # energy_J = power_real_W * dt
    expected_energy = expected_power * dt
    assert m.energy_J == pytest.approx(expected_energy, rel=1e-4), (
        f"energy_J={m.energy_J:.6f} should be power*dt={expected_energy:.6f}"
    )
