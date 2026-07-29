"""P9.8 Motion metrics tests + P9.7 mean-vs-median gyration comparison.

Split out of test_metrics_motion.py (file-size split) — silhouette,
suggested_m*, eta(m), and robust gyration/density tests stay in the
original; this file covers P9.8's motion metrics (velocity deviation,
boundary overshoot, altitude deviation, normalized angular momentum)
plus the P9.7 mean-vs-median gyration comparison test.
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import FlockMetrics, MetricsCollector

# ── P9.8: Motion metrics ──────────────────────────────────────

def test_velocity_deviation_equal_headings():
    """P9.8: Equal headings + mixed speeds → deviation > 0 while α = 1."""
    # All same direction but different speeds: α=1 but speed deviation > 0
    N = 100
    velocities = np.zeros((N, 3), dtype=np.float32)
    velocities[:, 0] = np.linspace(1, 5, N).astype(np.float32)  # varying speeds

    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = velocities / norms

    alpha = float(np.linalg.norm(dirs.sum(axis=0)) / N)
    assert alpha == pytest.approx(1.0)  # All same direction

    # velocity_deviation = (1/N)Σ‖v̄ − v_i‖
    v_mean = velocities.mean(axis=0)
    dev = float(np.mean(np.linalg.norm(v_mean - velocities, axis=1)))
    assert dev > 0.0, "Different speeds should produce velocity_deviation > 0"


def test_boundary_overshoot_inside_zero():
    """P9.8: Points inside domain → overshoot = 0."""
    from pymurmur.analysis.metrics import _compute_boundary_overshoot

    positions = np.array([[250, 250, 250], [500, 500, 500]], dtype=np.float32)
    overshoot = _compute_boundary_overshoot(positions, 1000, 1000, 1000)
    assert overshoot == 0.0, f"Inside domain → overshoot=0, got {overshoot}"


def test_boundary_overshoot_outside_positive():
    """P9.8: Points outside domain → overshoot > 0."""
    from pymurmur.analysis.metrics import _compute_boundary_overshoot

    # Points far outside a small domain
    positions = np.array([[1000, 500, 500], [500, 1000, 500]], dtype=np.float32)
    overshoot = _compute_boundary_overshoot(positions, 200, 200, 200)
    assert overshoot > 0, f"Outside domain → overshoot > 0, got {overshoot}"


def test_altitude_deviation_from_target():
    """P9.8: Altitude deviation measures distance from z_target."""
    from pymurmur.analysis.metrics import _compute_altitude_deviation

    positions = np.array(
        [[0, 0, 100], [0, 0, 200], [0, 0, 500]], dtype=np.float32
    )
    # Explicit z_target=500
    dev = _compute_altitude_deviation(positions, z_target=500.0)
    # z values: 100, 200, 500 → deviations: 400, 300, 0 → mean = 233.3
    expected = (400 + 300 + 0) / 3.0
    assert dev == pytest.approx(expected, rel=0.01)


def test_altitude_target_defaults_to_domain_centre_z():
    """S3.8: MetricsCollector's z_target defaults to domain-centre z
    (depth/2) when roost.z_target hasn't been explicitly overridden
    away from its shared dataclass default (500.0)."""
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.width, cfg.height, cfg.depth = 1000.0, 700.0, 300.0
    collector = MetricsCollector(cfg)
    assert collector._roost_z_target == pytest.approx(150.0)


def test_altitude_target_respects_explicit_override():
    """S3.8: an explicitly-set roost.z_target is used as-is, not
    overridden by the domain-centre default."""
    from pymurmur.core.config import SimConfig

    cfg = SimConfig()
    cfg.depth = 300.0
    cfg.roost.z_target = 42.0
    collector = MetricsCollector(cfg)
    assert collector._roost_z_target == pytest.approx(42.0)


def test_normalized_angular_momentum_circular():
    """P9.8: Circular motion in XY → L_norm > 0."""
    from pymurmur.analysis.metrics import compute_gyration, compute_normalized_angular_momentum

    N = 50
    rng = np.random.RandomState(42)
    angles = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
    radius = 200.0
    positions = np.zeros((N, 3), dtype=np.float32)
    positions[:, 0] = np.cos(angles) * radius
    positions[:, 1] = np.sin(angles) * radius

    # Tangential velocities
    velocities = np.zeros((N, 3), dtype=np.float32)
    velocities[:, 0] = -np.sin(angles) * 4.0
    velocities[:, 1] = np.cos(angles) * 4.0

    Rg = compute_gyration(positions)
    L_norm = compute_normalized_angular_momentum(positions, velocities, 4.0, Rg)
    assert L_norm > 0, f"Circular motion L_norm={L_norm} should be > 0"


def test_normalized_angular_momentum_O1():
    """P9.8: L_norm is O(1) across ×10 domain scale."""
    from pymurmur.analysis.metrics import compute_gyration, compute_normalized_angular_momentum

    N = 50
    rng = np.random.RandomState(99)

    for scale in [100.0, 300.0, 1000.0]:
        positions = rng.randn(N, 3).astype(np.float32) * (scale / 6)
        velocities = rng.randn(N, 3).astype(np.float32) * 4.0

        Rg = compute_gyration(positions)
        L_norm = compute_normalized_angular_momentum(positions, velocities, 4.0, max(Rg, 0.01))

        # Should be in a reasonable range (not exploding)
        assert L_norm < 10.0, (
            f"Scale {scale}: L_norm={L_norm:.2f} should be O(1)"
        )


def test_motion_metrics_in_collected_metrics(default_config):
    """P9.8: Motion metrics are populated by MetricsCollector."""
    from pymurmur.physics.flock import PhysicsFlock

    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    collector = MetricsCollector(cfg)
    collector.collect(flock, 0)

    snap = collector.snapshot()
    assert snap.velocity_deviation >= 0.0
    assert snap.boundary_overshoot >= 0.0
    assert snap.altitude_deviation >= 0.0


def test_normalized_angular_momentum_field():
    """P9.8: normalized_angular_momentum field exists on FlockMetrics."""
    m = FlockMetrics()
    assert hasattr(m, "normalized_angular_momentum")


# ── P9.7: Mean vs median centroid ───────────────────────────────

def test_robust_gyration_vs_mean_gyration():
    """P9.7: Median-centroid gyration < mean-centroid when outliers present."""
    from pymurmur.analysis.metrics import compute_gyration

    rng = np.random.RandomState(42)
    N = 100
    # Tight cluster near origin
    positions = rng.randn(N, 3).astype(np.float32) * 10

    # Add one extreme outlier
    positions_out = np.vstack([positions, [[10000, 0, 0]]]).astype(np.float32)

    Rg_robust = compute_gyration(positions_out)

    # Compute mean-based gyration for comparison
    com_mean = positions_out.mean(axis=0)
    dists_mean = np.linalg.norm(positions_out - com_mean, axis=1)
    keep_mean = int(len(positions_out) * 0.85)
    kept_mean = np.sort(dists_mean)[:keep_mean]
    Rg_mean = float(np.sqrt(np.mean(kept_mean ** 2)))

    # Mean-centroid should be larger due to outlier pull on centroid
    assert Rg_robust < Rg_mean, (
        f"Robust R_g={Rg_robust:.1f} should be below mean-based {Rg_mean:.1f}"
    )
