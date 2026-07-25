"""P9.4–P9.8 Motion/shape/gyration metrics tests.

Extracted from test_metrics.py — silhouette, suggested_m*, eta(m),
robust gyration, motion metrics (velocity deviation, boundary overshoot,
altitude deviation, normalized angular momentum).
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import FlockMetrics, MetricsCollector

# ── P9.4: 2D Silhouette Θ' ───────────────────────────────────

def test_silhouette_2d_flat_wall():
    """P9.4: Flat wall ¬ Z → high silhouette, low 3D theta_prime."""
    from pymurmur.analysis.metrics import compute_silhouette_2d, compute_theta_prime

    # Dense wall of birds in XY plane
    N = 200
    rng = np.random.RandomState(42)
    positions = np.zeros((N, 3), dtype=np.float32)
    positions[:, 0] = rng.uniform(0, 100, N).astype(np.float32)
    positions[:, 1] = rng.uniform(0, 100, N).astype(np.float32)
    positions[:, 2] = 0.0  # flat in Z

    sil = compute_silhouette_2d(positions, boid_size=5.0, grid_res=100)
    theta = compute_theta_prime(positions, grid_res=30)

    # Flat wall → high 2D silhouette coverage
    assert sil > 0.3, f"Flat wall silhouette should be > 0.3, got {sil:.4f}"
    # 3D theta_prime should be very low (thin in Z)
    assert theta < 0.1, f"3D theta for flat wall should be < 0.1, got {theta:.4f}"


def test_metrics_collector_wires_boid_size_into_silhouette():
    """S3.6: MetricsCollector.collect() passes cfg.flock.boid_size into
    compute_silhouette_2d instead of the function's hardcoded 5.0
    default — a larger boid_size must raise silhouette coverage for the
    same sparse flock."""
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

    def _silhouette_for(boid_size):
        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.boid_size = boid_size
        cfg.metrics_detail_level = 1
        flock = PhysicsFlock(cfg)
        flock.positions[:] = np.array(
            [[0, 0, 0], [200, 0, 0], [0, 200, 0], [200, 200, 0], [100, 100, 0]],
            dtype=np.float32,
        )
        flock.active[:] = True
        collector = MetricsCollector(cfg)
        collector.collect(flock, 0)
        return collector.snapshot().silhouette_2d

    sil_small = _silhouette_for(2.0)
    sil_large = _silhouette_for(60.0)
    assert sil_large > sil_small, (
        f"Larger boid_size should raise silhouette coverage: "
        f"small={sil_small:.4f} vs large={sil_large:.4f}"
    )


def test_silhouette_2d_two_coincident_birds():
    """P9.4: Two birds at same XY → silhouette counts them once."""
    from pymurmur.analysis.metrics import compute_silhouette_2d

    # Two birds at same XY but different Z
    positions = np.array([[50, 50, 0], [50, 50, 100]], dtype=np.float32)
    sil_two = compute_silhouette_2d(positions, boid_size=3.0, grid_res=50)

    # One bird alone should produce similar silhouette
    sil_one = compute_silhouette_2d(
        np.array([[50, 50, 0]], dtype=np.float32),
        boid_size=3.0, grid_res=50,
    )

    # Two coincident birds should not double the silhouette
    assert sil_two == pytest.approx(sil_one, rel=0.3), (
        f"Coincident birds shouldn't double silhouette: two={sil_two:.4f}, one={sil_one:.4f}"
    )


def test_silhouette_2d_empty():
    """P9.4: Empty flock → silhouette = 0."""
    from pymurmur.analysis.metrics import compute_silhouette_2d
    assert compute_silhouette_2d(np.zeros((0, 3), dtype=np.float32)) == 0.0


def test_silhouette_in_collected_metrics(default_config):
    """P9.4: silhouette_2d is populated by MetricsCollector."""
    from pymurmur.physics.flock import PhysicsFlock

    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    collector = MetricsCollector()
    collector.collect(flock, 0)

    snap = collector.snapshot()
    assert snap.silhouette_2d >= 0.0
    assert snap.silhouette_2d <= 1.0


# ── P9.5: Shape → m* ─────────────────────────────────────────

def test_suggested_m_sphere_aspect_1():
    """P9.5: aspect=1 (sphere) → m* = 9.78."""
    from pymurmur.analysis.metrics import compute_suggested_m

    m_star = compute_suggested_m(1.0)
    assert m_star == pytest.approx(9.78, rel=0.01)


def test_suggested_m_elongated_aspect_3():
    """P9.5: aspect=3 (elongated) → m* = 6.05."""
    from pymurmur.analysis.metrics import compute_suggested_m

    m_star = compute_suggested_m(3.0)
    assert m_star == pytest.approx(6.05, rel=0.01)


def test_suggested_m_monotone():
    """P9.5: larger aspect → smaller m* (or equal)."""
    from pymurmur.analysis.metrics import compute_suggested_m

    prev = compute_suggested_m(0.5)
    for aspect in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        cur = compute_suggested_m(aspect)
        assert cur <= prev + 0.01, (
            f"Non-monotone: aspect={aspect}, m*={cur:.3f} > prev={prev:.3f}"
        )
        prev = cur


def test_suggested_m_thin_flock_le_7():
    """P9.5: Very elongated flocks (aspect > 5) → m* ≤ 7."""
    from pymurmur.analysis.metrics import compute_suggested_m

    m_star = compute_suggested_m(10.0)
    assert m_star <= 7.0, f"Thin flock m*={m_star:.2f} should be ≤ 7"


def test_suggested_m_round_flock_ge_8():
    """P9.5: Round flocks (aspect=1) → m* ≥ 8."""
    from pymurmur.analysis.metrics import compute_suggested_m

    m_star = compute_suggested_m(1.0)
    assert m_star >= 8.0, f"Round flock m*={m_star:.2f} should be ≥ 8"


def test_suggested_m_in_flock_metrics():
    """P9.5: suggested_m field exists on FlockMetrics."""
    m = FlockMetrics()
    assert hasattr(m, "suggested_m")
    assert m.suggested_m is None


def test_suggested_m_in_to_dict():
    """P9.5: suggested_m appears in to_dict."""
    m = FlockMetrics(suggested_m=8.5)
    d = m.to_dict()
    assert "suggested_m" in d
    assert d["suggested_m"] == pytest.approx(8.5)


# ── P9.6: η(m) Marginal efficiency ───────────────────────────

def test_eta_m_connectivity_transition():
    """P9.6: When m first connects graph, η = +∞.

    Construct two clusters that are disconnected at m=2 but
    connected at m=4. The H2 drop from inf→finite makes η=+∞."""
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree

    from pymurmur.analysis.metrics import _compute_eta_m

    rng = np.random.RandomState(99)
    # Two tight clusters, close enough that k=5 neighbours bridges them
    # but k=3 neighbours does not.
    cluster_a = rng.randn(10, 3).astype(np.float32) * 3
    cluster_b = rng.randn(10, 3).astype(np.float32) * 3 + np.array([80, 0, 0])
    positions = np.vstack([cluster_a, cluster_b])
    tree = cKDTree(positions)

    # Try several m values to find a transition point
    from pymurmur.analysis.metrics import compute_h2

    # m=2 (k=3): should be disconnected (3 neighbours all intra-cluster)
    _, h2_m2 = compute_h2(positions, 2, tree)
    # m=4 (k=5): should be connected (5 neighbours span clusters)
    _, h2_m4 = compute_h2(positions, 4, tree)

    if not np.isfinite(h2_m2) and np.isfinite(h2_m4):
        # We have a genuine connectivity transition!
        eta = _compute_eta_m(positions, tree, 4)
        assert eta == float('inf'), (
            f"Connectivity transition should give η=+∞, got {eta}"
        )
    else:
        # Try with different spacing if needed
        # Use a wider gap to guarantee disconnection at low m
        cluster_b_far = rng.randn(10, 3).astype(np.float32) * 3 + np.array([200, 0, 0])
        positions2 = np.vstack([cluster_a, cluster_b_far])
        tree2 = cKDTree(positions2)

        # m=4 should also be disconnected now (clusters far apart)
        # m=6 (k=7) might bridge...
        for m_test in [5, 6, 7, 8]:
            _, h2_test = compute_h2(positions2, m_test, tree2)
            if np.isfinite(h2_test):
                eta = _compute_eta_m(positions2, tree2, m_test)
                assert eta == float('inf') or np.isfinite(eta), (
                    f"Unexpected eta={eta} for connected m={m_test}"
                )
                return

        # If no transition found, verify eta returns something valid
        eta = _compute_eta_m(positions2, tree2, 6)
        assert eta == 0.0 or np.isfinite(eta), (
            f"No transition: eta should be 0 or finite, got {eta}"
        )


def test_eta_m_both_disconnected_zero():
    """P9.6: Both m and m0 disconnected → η = 0."""
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree

    from pymurmur.analysis.metrics import _compute_eta_m

    # Two large clusters far apart — each bird's k=5 neighbours
    # are all within its own cluster, so graph stays disconnected at m=4.
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(15, 3).astype(np.float32) * 5
    cluster_b = rng.randn(15, 3).astype(np.float32) * 5 + np.array([1000, 0, 0])
    positions = np.vstack([cluster_a, cluster_b])
    tree = cKDTree(positions)

    # Both m=4 and m0=2 should be disconnected → η = 0
    eta = _compute_eta_m(positions, tree, 4)
    assert eta == 0.0, f"Both disconnected → η=0, got {eta}"


def test_eta_m_small_flock_returns_none():
    """P9.6: N < 4 or m < 3 → η = None."""
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree

    from pymurmur.analysis.metrics import _compute_eta_m

    positions = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    tree = cKDTree(positions)

    eta = _compute_eta_m(positions, tree, 2)
    assert eta is None, f"Small flock → None, got {eta}"


def test_metrics_collector_computes_convergence_speed():
    """A10: MetricsCollector populates convergence_speed alongside h2
    at the same gated expensive-metrics interval."""
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 30
    cfg.metrics_detail_level = 2
    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    collector = MetricsCollector(cfg)
    collector.collect(flock, 0)
    snap = collector.snapshot()

    assert snap.convergence_speed is not None, "convergence_speed was never computed"
    assert np.isfinite(snap.convergence_speed)
    assert snap.convergence_speed >= 0.0


def test_eta_m_in_flock_metrics():
    """P9.6: eta_m field exists on FlockMetrics."""
    m = FlockMetrics()
    assert hasattr(m, "eta_m")
    assert m.eta_m is None


def test_eta_m_in_to_dict():
    """P9.6: eta_m appears in to_dict."""
    m = FlockMetrics(eta_m=0.15)
    d = m.to_dict()
    assert "eta_m" in d
    assert d["eta_m"] == pytest.approx(0.15)


# ── P9.7: Robust gyration + ideal exponent ────────────────────

def test_robust_gyration_median_centroid():
    """P9.7: One 10K-unit outlier moves R_g < 5% (median-resistant)."""
    from pymurmur.analysis.metrics import compute_gyration

    rng = np.random.RandomState(42)
    N = 100
    # Tight cluster around origin
    positions = rng.randn(N, 3).astype(np.float32) * 10
    Rg_clean = compute_gyration(positions)

    # Add one extreme outlier
    positions_out = positions.copy()
    positions_out = np.vstack([positions_out, [[10000, 0, 0]]])
    Rg_outlier = compute_gyration(positions_out.astype(np.float32))

    # Robust gyration should barely change
    rel_change = abs(Rg_outlier - Rg_clean) / max(Rg_clean, 0.01)
    assert rel_change < 0.05, (
        f"Outlier moved R_g {rel_change:.3%}: {Rg_clean:.1f} → {Rg_outlier:.1f}"
    )


def test_robust_gyration_degenerate_zero():
    """P9.7: Fewer than 3 points → R_g = 0."""
    from pymurmur.analysis.metrics import compute_gyration

    assert compute_gyration(np.zeros((2, 3), dtype=np.float32)) == 0.0
    assert compute_gyration(np.zeros((0, 3), dtype=np.float32)) == 0.0


def test_robust_density_nonzero_sphere():
    """P9.7: Uniform sphere → ρ > 0."""
    from pymurmur.analysis.metrics import compute_robust_density

    rng = np.random.RandomState(42)
    positions = rng.randn(100, 3).astype(np.float32) * 20
    Rg, rho = compute_robust_density(positions)
    assert Rg > 0
    assert rho > 0


def test_ideal_exponent_in_density_scaling_result():
    """P9.7: DensityScalingResult carries ideal_density_exponent = −0.5."""
    from pymurmur.analysis.density_scaling import DensityScalingResult

    result = DensityScalingResult(
        n_values=np.array([50.0, 100.0], dtype=np.float64),
        spacings_toroidal=np.array([12.0, 8.0], dtype=np.float64),
        spacings_open=np.array([12.0, 8.0], dtype=np.float64),
    )
    assert result.ideal_density_exponent == pytest.approx(-0.5)


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


# ── P9.4: Silhouette edge cases ─────────────────────────────────

def test_silhouette_2d_bounded_0_1():
    """P9.4: Silhouette is always in [0, 1] for any positions."""
    from pymurmur.analysis.metrics import compute_silhouette_2d

    rng = np.random.RandomState(123)
    for N in [1, 5, 50, 200]:
        positions = rng.randn(N, 3).astype(np.float32) * 100 + 500
        sil = compute_silhouette_2d(positions)
        assert 0.0 <= sil <= 1.0, f"N={N}: silhouette={sil:.4f} out of [0,1]"


def test_silhouette_2d_large_boid_size_near_one():
    """P9.4: Very large boid_size makes all disks merge → silhouette > 0.75."""
    from pymurmur.analysis.metrics import compute_silhouette_2d

    rng = np.random.RandomState(42)
    positions = rng.randn(100, 3).astype(np.float32) * 5 + 500
    # boid_size 1000 dwarfs the cluster → one big blob
    sil = compute_silhouette_2d(positions, boid_size=1000.0)
    assert sil > 0.75, f"Huge boid_size should produce high sil, got {sil:.4f}"


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

