"""Unit tests for physics.boid — P0.15 position-init variants, P3.10 blob velocity init, P4.9 velocity-init variants + dispatch.

Split out of test_boid.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.physics.boid import (
    init_velocities,
    init_velocities_blob,
    init_velocities_cube,
    init_velocities_fixed,
    init_velocities_speed_uniform,
    init_velocities_tangential,
    random_unit_sphere,
)

# ── P0.15 Position Init Variants ─────────────────────────────────


def test_init_positions_box():
    """Box mode: uniform random in domain."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(42)
    pos = init_positions(100, 1000.0, 700.0, 400.0, rng, mode="box")
    assert pos.shape == (100, 3)
    assert pos.dtype == np.float32
    assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= 1000.0).all()
    assert (pos[:, 1] >= 0).all() and (pos[:, 1] <= 700.0).all()
    assert (pos[:, 2] >= 0).all() and (pos[:, 2] <= 400.0).all()


def test_init_positions_sphere_shell():
    """Sphere shell: all points exactly on shell surface."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(42)
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 0.4 * min(W, H, D)  # 0.4 * 400 = 160

    pos = init_positions(200, W, H, D, rng, mode="sphere_shell")
    dists = np.linalg.norm(pos - C, axis=1)
    assert np.allclose(dists, R, atol=1e-4), (
        f"sphere_shell: all points must be at R={R}, got {dists.min():.3f}–{dists.max():.3f}"
    )


def test_init_positions_sphere_shell_shape():
    """Sphere shell returns correct shape and dtype."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(7)
    pos = init_positions(50, 1000.0, 700.0, 400.0, rng, mode="sphere_shell")
    assert pos.shape == (50, 3)
    assert pos.dtype == np.float32


def test_init_positions_gaussian():
    """Gaussian mode: positions cluster around centre."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(42)
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)

    pos = init_positions(500, W, H, D, rng, mode="gaussian", separation=9.0)
    assert pos.shape == (500, 3)

    # Mean should be near centre
    mean = pos.mean(axis=0)
    assert np.allclose(mean, C, atol=20.0), f"gaussian mean should be near C, got {mean}"

    # Std dev should be proportional to σ = n^(1/3) * separation
    expected_sigma = 500 ** (1.0 / 3.0) * 9.0  # ≈ 71.4
    std = pos.std(axis=0).mean()
    assert 30 < std < 150, f"gaussian std={std:.1f}, expected near {expected_sigma:.1f}"


def test_init_positions_grid():
    """Grid mode: deterministic, evenly spaced layout."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(42)

    pos = init_positions(125, 1000.0, 700.0, 400.0, rng, mode="grid")
    assert pos.shape == (125, 3)

    # Grid should produce unique positions with non-trivial spacing
    # 125 = 5³, so 5 points per axis
    assert len(np.unique(pos[:, 0])) >= 3
    assert len(np.unique(pos[:, 1])) >= 3
    assert len(np.unique(pos[:, 2])) >= 3

    # Deterministic: same seed → same grid
    pos2 = init_positions(125, 1000.0, 700.0, 400.0, rng, mode="grid")
    # Grid is deterministic regardless of rng
    np.testing.assert_array_equal(pos, pos2)


def test_init_positions_grid_no_overlaps():
    """Grid: no two birds at identical positions."""
    pytest.importorskip("scipy")
    from scipy.spatial.distance import cdist

    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(1)

    pos = init_positions(64, 500.0, 500.0, 500.0, rng, mode="grid")
    dists = cdist(pos, pos)
    # Set diagonal to large value so we only check inter-bird distances
    np.fill_diagonal(dists, np.inf)
    min_sep = dists.min()
    expected_spacing = (500.0 * 500.0 * 500.0 / 64) ** (1.0 / 3.0)
    assert min_sep > 0.8 * expected_spacing, (
        f"grid min sep={min_sep:.1f} < 0.8*spacing={0.8*expected_spacing:.1f}"
    )


def test_init_positions_blob():
    """Blob mode: 5-centre shell with jitter."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(42)
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)

    pos = init_positions(200, W, H, D, rng, mode="blob")
    assert pos.shape == (200, 3)
    assert pos.dtype == np.float32

    # Blob centre should be near domain centre (allowing for offsets)
    mean = pos.mean(axis=0)
    assert np.allclose(mean, C, atol=150.0), f"blob mean={mean} far from C={C}"

    # Points should have non-trivial spread (not all at same point)
    assert pos.std(axis=0).mean() > 5.0


def test_init_positions_seeded():
    """Same seed → same positions across all non-grid modes."""
    from pymurmur.physics.boid import init_positions
    for mode in ("box", "sphere_shell", "gaussian", "blob"):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = init_positions(50, 1000.0, 700.0, 400.0, rng1, mode=mode)
        p2 = init_positions(50, 1000.0, 700.0, 400.0, rng2, mode=mode)
        assert np.allclose(p1, p2), f"{mode}: same seed must produce same positions"


def test_init_positions_different_modes_different():
    """Different modes produce different position distributions."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(42)
    p1 = init_positions(50, 1000.0, 700.0, 400.0, rng, mode="box")
    p2 = init_positions(50, 1000.0, 700.0, 400.0, rng, mode="sphere_shell")
    # Different modes should not produce identical results
    assert not np.allclose(p1, p2)


def test_init_positions_grid_non_cubic():
    """Grid with non-cubic N (not a perfect cube): still produces correct count."""
    from pymurmur.physics.boid import init_positions
    rng = np.random.default_rng(42)

    # 50 is not a perfect cube — grid should still work
    pos = init_positions(50, 1000.0, 700.0, 400.0, rng, mode="grid")
    assert pos.shape == (50, 3)
    assert pos.dtype == np.float32
    # All positions should be within domain
    assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= 900.0).all()
    assert (pos[:, 1] >= 0).all() and (pos[:, 1] <= 630.0).all()
    assert (pos[:, 2] >= 0).all() and (pos[:, 2] <= 360.0).all()


# ── P3.10 Blob Velocity Init ──────────────────────────────────────


def test_blob_velocities_differ_from_random_sphere():
    """P3.10: Blob velocities are not isotropic — they differ from
    random_unit_sphere and exhibit a measurable forward drift bias."""
    N = 500
    v0 = 4.0
    rng = np.random.default_rng(42)

    # Blob velocities: drift-biased tangential per spec
    v_blob = init_velocities_blob(N, v0, rng)

    # Non-blob default: random_unit_sphere scaled by v0 * 0.8
    rng2 = np.random.default_rng(42)
    v_sphere = random_unit_sphere(N, rng2) * v0 * 0.8

    # ── The two distributions must differ ──
    assert not np.allclose(v_blob, v_sphere, atol=1e-6), (
        "blob velocities must differ from random sphere velocities"
    )

    # ── Shape and dtype ──
    assert v_blob.shape == (N, 3)
    assert v_blob.dtype == np.float32

    # ── Forward drift bias: mean x must be measurably positive ──
    # Expected: 0.34 * v0 * 0.5 = 0.68 at v0=4.0
    mean_x = v_blob[:, 0].mean()
    assert mean_x > 0.3, (
        f"blob velocities must have positive x-drift, got mean_x={mean_x:.4f}"
    )

    # ── x-component is always positive (range: [0.26*v0*0.5, 0.42*v0*0.5]) ──
    assert (v_blob[:, 0] > 0).all(), (
        "all blob x-velocities must be positive (forward drift)"
    )

    # ── y is centered at zero (range: [-0.16*v0*0.5, 0.16*v0*0.5]) ──
    mean_y = v_blob[:, 1].mean()
    assert abs(mean_y) < 0.15, (
        f"blob y-velocities must be centered near zero, got mean_y={mean_y:.4f}"
    )

    # ── z has slight upward bias: 0.08 * v0 * 0.5 = 0.16 ──
    mean_z = v_blob[:, 2].mean()
    assert mean_z > 0.0, (
        f"blob z-velocities must have slight upward bias, got mean_z={mean_z:.4f}"
    )

    # ── Contrast: random sphere has ~zero mean on all axes ──
    sphere_mean_x = v_sphere[:, 0].mean()
    sphere_mean_y = v_sphere[:, 1].mean()
    sphere_mean_z = v_sphere[:, 2].mean()
    assert abs(sphere_mean_x) < 0.15, (
        f"random sphere x-mean should be near zero, got {sphere_mean_x:.4f}"
    )
    assert abs(sphere_mean_y) < 0.15, (
        f"random sphere y-mean should be near zero, got {sphere_mean_y:.4f}"
    )
    assert abs(sphere_mean_z) < 0.15, (
        f"random sphere z-mean should be near zero, got {sphere_mean_z:.4f}"
    )


def test_blob_velocities_seeded():
    """P3.10: Same seed → same blob velocities (deterministic init)."""
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    v1 = init_velocities_blob(100, 4.0, rng1)
    v2 = init_velocities_blob(100, 4.0, rng2)
    np.testing.assert_array_equal(v1, v2)


# ── P4.9: Velocity-init variants ─────────────────────────────

def test_init_velocities_cube_shape():
    """P4.9: Cube mode returns (n, 3) float32 with values in [-v0, v0]."""
    rng = np.random.default_rng(42)
    v = init_velocities_cube(200, 4.0, rng)
    assert v.shape == (200, 3)
    assert v.dtype == np.float32
    # All components in [-v0, v0]
    assert (v >= -4.0).all() and (v <= 4.0).all()


def test_init_velocities_cube_distribution():
    """P4.9: Cube mode has wider speed distribution than sphere."""
    rng = np.random.default_rng(42)
    v = init_velocities_cube(1000, 4.0, rng)
    speeds = np.linalg.norm(v, axis=1)
    # Mean speed ≈ 0.96·v0 ≈ 3.84 (expected value of ‖U(−1,1)³‖)
    assert 3.5 < speeds.mean() < 4.2, f"mean speed={speeds.mean():.2f}"
    # Should have speeds above and below v0
    assert (speeds > 4.0).any(), "cube should produce speeds > v0"
    assert (speeds < 4.0).any(), "cube should produce speeds < v0"


def test_init_velocities_cube_seeded():
    """P4.9: Same seed → same cube velocities."""
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    v1 = init_velocities_cube(100, 4.0, rng1)
    v2 = init_velocities_cube(100, 4.0, rng2)
    np.testing.assert_array_equal(v1, v2)


def test_init_velocities_speed_uniform_range():
    """S2.B9: Speed_uniform mode produces speeds in [min(1, 0.3v0), v0]."""
    rng = np.random.default_rng(42)
    v = init_velocities_speed_uniform(1000, 4.0, rng)
    speeds = np.linalg.norm(v, axis=1)
    lo = min(1.0, 0.3 * 4.0)  # 1.0
    # All speeds in [lo, v0]
    assert (speeds >= lo - 1e-4).all(), f"min speed={speeds.min():.4f}"
    assert (speeds <= 4.05).all(), f"max speed={speeds.max():.4f}"
    # Uniform distribution over [1, 4] → mean ≈ 2.5
    assert 2.3 < speeds.mean() < 2.7, f"mean speed={speeds.mean():.2f}"


def test_init_velocities_speed_uniform_lower_bound_capped_at_one():
    """S2.B9: At high v0, the floor caps at 1.0 (min(1, 0.3v0) = 1.0
    once 0.3v0 exceeds 1, i.e. v0 > 10/3), not 0.3v0."""
    rng = np.random.default_rng(7)
    v0 = 10.0  # 0.3*v0 = 3.0 > 1.0 → floor caps at 1.0
    v = init_velocities_speed_uniform(2000, v0, rng)
    speeds = np.linalg.norm(v, axis=1)
    assert speeds.min() >= 1.0 - 1e-3, f"min speed={speeds.min():.4f}, floor should be 1.0"
    # A handful of samples should land near the floor (uniform draw over a
    # wide [1, 10] range), confirming the floor is 1.0 not 0.3*v0=3.0.
    assert (speeds < 2.5).any(), "expected some speeds below 0.3*v0=3.0"


def test_init_velocities_speed_uniform_directions():
    """P4.9: Speed_uniform has unit-vector directions (on sphere)."""
    rng = np.random.default_rng(42)
    v = init_velocities_speed_uniform(500, 4.0, rng)
    speeds = np.linalg.norm(v, axis=1, keepdims=True)
    nonzero = speeds.ravel() > 1e-6
    dirs = v[nonzero] / speeds[nonzero]
    dir_norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(dir_norms, 1.0, atol=1e-5)


def test_init_velocities_tangential_perpendicular():
    """P4.9: Tangential velocities are perpendicular to radial direction."""
    rng = np.random.default_rng(42)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    # Create positions at various distances from centre
    positions = center + rng.uniform(-100, 100, (50, 3)).astype(np.float32)
    # Ensure no position is exactly at centre
    positions += np.array([1.0, 0.0, 0.0], dtype=np.float32)

    v = init_velocities_tangential(50, 4.0, rng, center, positions)

    # Check each bird: velocity should be perpendicular to radial
    for i in range(50):
        radial = positions[i] - center
        radial /= np.linalg.norm(radial)
        dot = np.abs(np.dot(v[i], radial))
        assert dot < 0.1, (
            f"Bird {i}: velocity not tangential, dot(vel, radial)={dot:.4f}"
        )


def test_init_velocities_tangential_speed_in_range_not_constant():
    """S2.B9: Tangential speeds vary over U(1, v0), not fixed at v0."""
    rng = np.random.default_rng(3)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    positions = center + rng.uniform(-100, 100, (300, 3)).astype(np.float32)
    positions += np.array([1.0, 0.0, 0.0], dtype=np.float32)

    v0 = 4.0
    v = init_velocities_tangential(300, v0, rng, center, positions)
    speeds = np.linalg.norm(v, axis=1)
    assert (speeds >= 1.0 - 1e-3).all() and (speeds <= v0 + 1e-3).all(), (
        f"speeds out of [1, v0]: min={speeds.min():.3f} max={speeds.max():.3f}"
    )
    assert speeds.std() > 0.1, "tangential speeds must not be constant"


def test_init_velocities_tangential_at_centre():
    """P4.9: Tangential mode at centre falls back to random sphere."""
    rng = np.random.default_rng(42)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    positions = np.zeros((10, 3), dtype=np.float32)  # all at centre

    v = init_velocities_tangential(10, 4.0, rng, center, positions)
    assert v.shape == (10, 3)
    assert np.isfinite(v).all()
    # Should have non-trivial velocities (fallback to random)
    assert (np.linalg.norm(v, axis=1) > 0).all()


def test_init_velocities_fixed_all_same():
    """P4.9: Fixed mode gives all birds identical velocity."""
    v = init_velocities_fixed(100, 4.0, direction=(0.6, 0.0, 0.4))
    assert v.shape == (100, 3)
    # All rows should be identical
    assert np.allclose(v[0], v[1:]), "all birds must have same velocity"
    # Speed should be v0
    speeds = np.linalg.norm(v, axis=1)
    assert np.allclose(speeds, 4.0, atol=1e-4)


def test_init_velocities_fixed_zero_direction():
    """P4.9: Fixed mode with zero direction falls back to (1,0,0)."""
    v = init_velocities_fixed(10, 4.0, direction=(0.0, 0.0, 0.0))
    # Should fall back to (1, 0, 0) direction
    assert np.allclose(v[0, 1], 0.0)
    assert np.allclose(v[0, 2], 0.0)
    assert v[0, 0] > 3.99


def test_init_velocities_dispatch_sphere():
    """P4.9: dispatch mode='sphere' uses random_unit_sphere * v0 * 0.8."""
    rng = np.random.default_rng(42)
    v = init_velocities(100, 4.0, rng, mode="sphere")
    assert v.shape == (100, 3)
    speeds = np.linalg.norm(v, axis=1)
    # Sphere mode: fixed speed at 0.8 * v0 = 3.2
    assert np.allclose(speeds, 3.2, atol=1e-4)


def test_init_velocities_dispatch_blob():
    """P4.9: dispatch mode='blob' delegates to init_velocities_blob."""
    rng = np.random.default_rng(42)
    v_dispatch = init_velocities(100, 4.0, rng, mode="blob")
    rng2 = np.random.default_rng(42)
    v_direct = init_velocities_blob(100, 4.0, rng2)
    np.testing.assert_array_equal(v_dispatch, v_direct)


def test_init_velocities_dispatch_drift_aliases_blob():
    """C3: dispatch mode='drift' is a pure alias for mode='blob'."""
    rng = np.random.default_rng(42)
    v_drift = init_velocities(100, 4.0, rng, mode="drift")
    rng2 = np.random.default_rng(42)
    v_blob = init_velocities(100, 4.0, rng2, mode="blob")
    np.testing.assert_array_equal(v_drift, v_blob)


def test_init_velocities_dispatch_cube():
    """P4.9: dispatch mode='cube' delegates correctly."""
    rng = np.random.default_rng(99)
    v_dispatch = init_velocities(100, 4.0, rng, mode="cube")
    rng2 = np.random.default_rng(99)
    v_direct = init_velocities_cube(100, 4.0, rng2)
    np.testing.assert_array_equal(v_dispatch, v_direct)


def test_init_velocities_dispatch_speed_uniform():
    """P4.9: dispatch mode='speed_uniform' delegates correctly."""
    rng = np.random.default_rng(99)
    v_dispatch = init_velocities(100, 4.0, rng, mode="speed_uniform")
    rng2 = np.random.default_rng(99)
    v_direct = init_velocities_speed_uniform(100, 4.0, rng2)
    np.testing.assert_array_equal(v_dispatch, v_direct)


def test_init_velocities_dispatch_fixed():
    """P4.9: dispatch mode='fixed' delegates correctly."""
    v_dispatch = init_velocities(50, 4.0, mode="fixed")
    v_direct = init_velocities_fixed(50, 4.0)
    np.testing.assert_array_equal(v_dispatch, v_direct)


def test_init_velocities_seeded_deterministic():
    """P4.9: Same seed + same mode → identical velocities for all modes."""
    for mode in ("sphere", "blob", "cube", "speed_uniform"):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        v1 = init_velocities(50, 4.0, rng1, mode=mode)
        v2 = init_velocities(50, 4.0, rng2, mode=mode)
        np.testing.assert_array_equal(v1, v2, err_msg=f"mode={mode} not deterministic")


def test_velocity_init_config_field():
    """P4.9: SimConfig.velocity_init defaults to 'sphere'."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    assert cfg.velocity_init == "sphere"
    # Should validate correctly
    cfg.validate()  # no exception


def test_velocity_init_config_validation():
    """P4.9: Invalid velocity_init raises ValueError."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.velocity_init = "invalid_mode"
    with pytest.raises(ValueError, match="velocity_init"):
        cfg.validate()


def test_velocity_init_via_flock(default_config):
    """P4.9: PhysicsFlock uses velocity_init config field."""
    cfg = default_config
    cfg.velocity_init = "cube"
    cfg.num_boids = 20
    from pymurmur.physics.flock import PhysicsFlock
    flock = PhysicsFlock(cfg)
    assert flock.velocities.shape == (20, 3)
    assert flock.velocities.dtype == np.float32
    # Cube mode: velocities should be in [-v0, v0]³
    v0 = cfg.v0
    assert (flock.velocities >= -v0).all()
    assert (flock.velocities <= v0).all()


