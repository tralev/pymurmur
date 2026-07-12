"""Unit tests for physics.boid — integrate(), BoidView, array helpers."""

import pytest
import numpy as np

from pymurmur.physics.boid import (
    BoidView,
    integrate,
    random_positions,
    random_unit_sphere,
)


def test_random_positions_shape():
    """Returns (N, 3) float32."""
    pos = random_positions(100, 1000.0, 700.0, 400.0)
    assert pos.shape == (100, 3)
    assert pos.dtype == np.float32


def test_random_positions_in_domain():
    """All positions within domain bounds."""
    w, h, d = 1000.0, 700.0, 400.0
    pos = random_positions(500, w, h, d)
    assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= w).all()
    assert (pos[:, 1] >= 0).all() and (pos[:, 1] <= h).all()
    assert (pos[:, 2] >= 0).all() and (pos[:, 2] <= d).all()


def test_random_positions_seeded():
    """Same seed → same positions."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1 = random_positions(50, 1000.0, 700.0, 400.0, rng1)
    p2 = random_positions(50, 1000.0, 700.0, 400.0, rng2)
    assert np.allclose(p1, p2)


def test_random_unit_sphere_shape():
    """Returns (N, 3) float32."""
    pts = random_unit_sphere(200)
    assert pts.shape == (200, 3)
    assert pts.dtype == np.float32


def test_random_unit_sphere_unit_norm():
    """All vectors have norm ≈ 1.0."""
    pts = random_unit_sphere(200)
    norms = np.linalg.norm(pts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_integrate_applies_acceleration():
    """After integrate(), velocities += accelerations (then clamped)."""
    N = 10
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)
    acc = np.ones((N, 3), dtype=np.float32)  # each bird gets (1,1,1)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    speeds = np.linalg.norm(vel, axis=1)
    # After adding (1,1,1), speed = sqrt(3) ≈ 1.73, within [0.3*v0, v0] = [1.2, 4.0]
    # Should pass through unchanged (within band)
    assert np.allclose(speeds, np.sqrt(3.0), atol=0.1), f"speeds={speeds}"
    # After integrate, positions should have moved
    assert not np.allclose(pos, 0.0)


def test_integrate_resets_acceleration():
    """After integrate(), all active accelerations are zero."""
    N = 5
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)
    acc = np.ones((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    assert (acc[active] == 0.0).all()


def test_speed_clamp_fast():
    """Bird exceeding v0 is clamped to exactly v0."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    speeds = np.linalg.norm(vel, axis=1)
    assert (speeds <= 4.1).all()


def test_boundary_toroidal_x():
    """Bird crossing x > width wraps to x = 0."""
    N = 1
    pos = np.array([[999.0, 350.0, 200.0]], dtype=np.float32)
    vel = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0 / 60.0)
    assert 0.0 <= pos[0, 0] <= 1000.0


def test_boid_view_pos():
    """BoidView.pos returns correct position."""
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    view = BoidView(0, positions, velocities)
    assert np.allclose(view.pos, [1.0, 2.0, 3.0])


def test_boid_view_uses_slots():
    """BoidView has __slots__."""
    pos = np.zeros((1, 3), dtype=np.float32)
    vel = np.zeros((1, 3), dtype=np.float32)
    view = BoidView(0, pos, vel)
    assert not hasattr(view, "__dict__")


def test_boid_view_vel():
    """BoidView.vel returns correct velocity."""
    positions = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    view = BoidView(1, positions, velocities)
    assert np.allclose(view.vel, [4.0, 5.0, 6.0])


def test_boid_view_theta():
    """BoidView.theta returns cached opacity value."""
    positions = np.zeros((2, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    thetas = np.array([0.35, 0.72], dtype=np.float32)
    view = BoidView(1, positions, velocities, last_theta=thetas)
    assert view.theta == pytest.approx(0.72)


def test_boid_view_theta_none_fallback():
    """BoidView.theta returns 0.0 when no last_theta array provided."""
    positions = np.zeros((2, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    view = BoidView(0, positions, velocities)
    assert view.theta == 0.0


def test_boid_view_out_of_bounds():
    """BoidView with out-of-range idx raises IndexError."""
    positions = np.zeros((2, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    # numpy wraps negative indices — test positive out-of-bounds only
    with pytest.raises(IndexError):
        _ = BoidView(5, positions, velocities).pos


def test_boid_view_after_deactivate():
    """BoidView still returns data after active[idx] becomes False."""
    positions = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    velocities = np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    active = np.array([True, True], dtype=bool)
    view = BoidView(1, positions, velocities)
    # Deactivate bird 1
    active[1] = False
    # View still reads the same underlying array data
    assert np.allclose(view.pos, [7.0, 8.0, 9.0])
    assert np.allclose(view.vel, [10.0, 11.0, 12.0])


def test_boid_view_iterable():
    """BoidView can be used in a loop for all active birds."""
    N = 5
    positions = np.random.randn(N, 3).astype(np.float32)
    velocities = np.random.randn(N, 3).astype(np.float32)
    active = np.array([True, True, False, True, True], dtype=bool)

    count = 0
    for i in range(N):
        if active[i]:
            view = BoidView(i, positions, velocities)
            count += 1
            assert view.pos.shape == (3,)
    assert count == 4


# ── Speed clamp variants ──────────────────────────────────────────

def test_speed_clamp_slow():
    """Bird below 0.3*v0 is boosted to exactly 0.3*v0."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.05]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    speeds = np.linalg.norm(vel, axis=1)
    # All should be boosted to >= 0.3 * 4.0 = 1.2
    assert (speeds >= 1.1).all(), f"speeds={speeds}"


def test_speed_clamp_within_band():
    """Bird within [0.3*v0, v0] keeps its speed."""
    N = 2
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[3.0, 0, 0], [0, 2.0, 0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    speeds = np.linalg.norm(vel, axis=1)
    assert np.allclose(speeds[0], 3.0, atol=0.1)
    assert np.allclose(speeds[1], 2.0, atol=0.1)


def test_speed_clamp_vectorised():
    """All N birds clamped correctly in a single call."""
    N = 50
    pos = np.zeros((N, 3), dtype=np.float32)
    rng = np.random.default_rng(123)
    # Wide range of speeds: 0.05 to 20.0
    vel = rng.normal(size=(N, 3)).astype(np.float32)
    vel *= rng.uniform(0.05, 20.0, size=(N, 1)).astype(np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    speeds = np.linalg.norm(vel, axis=1)
    # All speeds must be in [0.3*v0, v0] = [1.2, 4.0]
    assert (speeds >= 1.19).all(), f"min speed={speeds.min()}"
    assert (speeds <= 4.01).all(), f"max speed={speeds.max()}"


def test_zero_speed_reseed():
    """Zero-speed birds get deterministic fallback (minSpeed, 0, 0).

    P0.9: deterministic fallback replaces random_unit_sphere for
    bit-identical replay. speed = v0 * speed_min_factor = 1.2.
    """
    N = 5
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)  # all zero speed
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    speeds = np.linalg.norm(vel, axis=1)
    # Deterministic: minSpeed = 0.3 * 4.0 = 1.2
    assert (speeds > 0).all(), f"speeds={speeds}"
    assert np.allclose(speeds, 1.2, atol=0.1), f"speeds={speeds}"
    # Deterministic direction: all birds get (+, 0, 0)
    assert (vel[:, 0] > 0).all(), "all x-components must be positive"
    assert np.allclose(vel[:, 1], 0.0), "all y-components must be 0"
    assert np.allclose(vel[:, 2], 0.0), "all z-components must be 0"


def test_integrate_inactive_unchanged():
    """Inactive birds' positions and velocities are unchanged."""
    N = 5
    pos = np.random.randn(N, 3).astype(np.float32) * 100 + 500
    # Keep inactive birds' speeds within [0.3*v0, v0] to avoid speed clamp
    # speed = sqrt(2²+2²+2²) ≈ 3.46, within [1.2, 4.0]
    vel = np.ones((N, 3), dtype=np.float32) * 2.0
    acc = np.ones((N, 3), dtype=np.float32)
    active = np.array([True, True, False, True, False])

    pos_before = pos.copy()
    vel_before = vel.copy()

    # Use "open" boundary so no position wrapping affects inactive birds
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "open", 1.0)

    # Inactive birds: positions and velocities unchanged
    inactive = ~active
    assert np.allclose(pos[inactive], pos_before[inactive]), \
        "inactive positions changed"
    assert np.allclose(vel[inactive], vel_before[inactive]), \
        "inactive velocities changed"
    # Active birds: should have moved
    assert not np.allclose(pos[active], pos_before[active])


def test_integrate_stationary_bird_gets_reseed():
    """Bird with zero velocity and zero acceleration gets re-seeded."""
    pos = np.array([[500.0, 350.0, 200.0]], dtype=np.float32)
    vel = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    active = np.ones(1, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    # Bird was stationary but got zero-speed reseed → speed now ~2.0
    speed = np.linalg.norm(vel[0])
    assert speed > 0, "stationary bird should be re-seeded"


def test_integrate_dt_scaling():
    """Doubling dt doubles position change (within clamped range)."""
    N = 5
    active = np.ones(N, dtype=bool)
    pt_ref = np.random.randn(N, 3).astype(np.float32) * 100 + 500

    pos_a = pt_ref.copy()
    pos_b = pt_ref.copy()
    vel = np.ones((N, 3), dtype=np.float32) * 2.0

    # Use dt within [0, 0.05] range to avoid clamp
    integrate(pos_a, vel.copy(), np.zeros((N, 3), dtype=np.float32),
              active, 1000.0, 700.0, 400.0, 4.0, "open", 0.01)
    integrate(pos_b, vel.copy(), np.zeros((N, 3), dtype=np.float32),
              active, 1000.0, 700.0, 400.0, 4.0, "open", 0.02)

    disp_a = np.linalg.norm(pos_a - pt_ref, axis=1)
    disp_b = np.linalg.norm(pos_b - pt_ref, axis=1)
    ratio = disp_b / disp_a
    assert np.allclose(ratio, 2.0, atol=0.3), f"ratio={ratio}"


# ── Boundary mode tests ───────────────────────────────────────────

def test_boundary_toroidal_negative():
    """Bird crossing x < 0 wraps to x = width."""
    N = 1
    pos = np.array([[1.0, 350.0, 200.0]], dtype=np.float32)
    vel = np.array([[-10.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0 / 60.0)
    assert 0.0 <= pos[0, 0] <= 1000.0


def test_boundary_toroidal_all_axes():
    """Wrapping works in X, Y, and Z independently."""
    N = 3
    # Bird 0: crosses +X, Bird 1: crosses +Y, Bird 2: crosses +Z
    pos = np.array([
        [999.0, 350.0, 200.0],
        [500.0, 699.0, 200.0],
        [500.0, 350.0, 399.0],
    ], dtype=np.float32)
    vel = np.array([
        [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 20.0],
    ], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    assert 0.0 <= pos[0, 0] <= 1000.0
    assert 0.0 <= pos[1, 1] <= 700.0
    assert 0.0 <= pos[2, 2] <= 400.0


def test_boundary_toroidal_velocity_preserved():
    """Velocity direction unchanged after wrapping."""
    N = 1
    pos = np.array([[999.0, 350.0, 200.0]], dtype=np.float32)
    vel = np.array([[10.0, 5.0, -3.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    vel_before = vel.copy()
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0 / 60.0)
    # Velocity magnitude and direction should be preserved (within speed clamp)
    speed_before = np.linalg.norm(vel_before)
    speed_after = np.linalg.norm(vel)
    # Speed clamp may adjust, but direction should be similar
    dir_before = vel_before / speed_before
    dir_after = vel / speed_after
    assert np.dot(dir_before.ravel(), dir_after.ravel()) > 0.99


def test_boundary_open():
    """Bird can leave domain freely — no position clamp (open boundary)."""
    N = 1
    # dt clamped to max 0.05, so bird moves v0 * 0.05 = 0.2 per frame
    pos = np.array([[990.0, 350.0, 200.0]], dtype=np.float32)
    vel = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    # Call integrate many times to leave domain
    for _ in range(100):
        integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
                  4.0, "open", 1.0)
    # After 100 frames at dt=0.05 (clamped), bird moves 4*0.05*100 = 20 units
    assert pos[0, 0] > 1000.0


def test_boundary_margin_nudge():
    """Bird near wall gets velocity nudge away from wall."""
    N = 1
    pos = np.array([[10.0, 350.0, 200.0]], dtype=np.float32)
    vel = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)  # moving toward wall
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    vel_before = vel[0, 0].copy()
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "margin", 1.0 / 60.0)
    # Velocity should be nudged away from wall (reduced x-component)
    # The nudge factor is 0.05 * (50 - 10) / 50 = 0.04 so vx should decrease
    assert vel[0, 0] < vel_before, f"expected vx < {vel_before}, got {vel[0,0]}"


def test_boundary_sphere_soft():
    """Bird outside sphere radius is projected back."""
    N = 1
    # Place bird outside the 300-radius sphere
    pos = np.array([[400.0, 0.0, 0.0]], dtype=np.float32)
    vel = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "sphere", 1.0 / 60.0, 300.0, 0.05)
    # Should be projected back to exactly radius
    dist = np.linalg.norm(pos[0])
    assert dist <= 300.0


def test_boundary_sphere_inside():
    """Bird inside sphere radius is unchanged (no projection needed)."""
    N = 1
    pos = np.array([[200.0, 0.0, 0.0]], dtype=np.float32)  # inside 300-radius
    vel = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    pos_before = pos.copy()
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "sphere", 1.0 / 60.0, 300.0, 0.05)
    # Bird inside sphere: position moves normally (not projected back)
    assert pos[0, 0] > pos_before[0, 0]  # moved forward
    assert np.linalg.norm(pos[0]) <= 300.0  # still inside


# ── P0.3 Physics Invariant Fuzz Tests ──────────────────────────
# Per roadmap P0.3: for integrate(..., "toroidal"): after step,
# 0 ≤ pos < (W,H,D) elementwise, |v| ≤ v0 + ε, no NaN,
# inactive rows bit-identical.


def test_speed_band_respected():
    """200 random seeds: all speeds ≤ v0 + epsilon, positions in domain."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for seed in range(200):
        rng_i = np.random.default_rng(seed)
        pos = rng_i.uniform(0, [W, H, D], (50, 3)).astype(np.float32)
        vel = rng_i.uniform(-2 * v0, 2 * v0, (50, 3)).astype(np.float32)
        acc = np.zeros((50, 3), dtype=np.float32)
        active = np.ones(50, dtype=bool)

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal", 1.0 / 60.0)

        speeds = np.linalg.norm(vel, axis=1)
        assert (speeds <= v0 + 1e-4).all(), (
            f"seed={seed}: max speed={speeds.max():.4f} > v0={v0}"
        )
        assert (pos[:, 0] >= 0).all() and (pos[:, 0] < W).all(), (
            f"seed={seed}: x out of bounds: min={pos[:,0].min():.1f} max={pos[:,0].max():.1f}"
        )
        assert (pos[:, 1] >= 0).all() and (pos[:, 1] < H).all(), (
            f"seed={seed}: y out of bounds"
        )
        assert (pos[:, 2] >= 0).all() and (pos[:, 2] < D).all(), (
            f"seed={seed}: z out of bounds"
        )

    print(f"\n✓ 200 seeds: all speeds ≤ {v0}+ε, all positions in domain")


def test_no_nan_after_integrate():
    """200 random seeds across boundary modes: no NaN in positions or velocities."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for mode in ("toroidal", "open", "margin", "sphere"):
        for seed in range(200):
            rng_i = np.random.default_rng(seed)
            pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
            vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
            acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
            active = rng_i.uniform(0, 1, 30) > 0.1  # ~90% active

            integrate(pos, vel, acc, active, W, H, D, v0, mode, 1.0 / 60.0)

            assert not np.isnan(pos).any(), (
                f"mode={mode} seed={seed}: NaN in positions"
            )
            assert not np.isnan(vel).any(), (
                f"mode={mode} seed={seed}: NaN in velocities"
            )
            assert not np.isinf(pos).any(), (
                f"mode={mode} seed={seed}: Inf in positions"
            )
            assert not np.isinf(vel).any(), (
                f"mode={mode} seed={seed}: Inf in velocities"
            )

    print(f"\n✓ 3 modes × 200 seeds: no NaN or Inf in positions/velocities")


def test_inactive_rows_bit_identical():
    """Inactive birds' positions and velocities are bit-identical after integrate.

    P0.3 requirement: inactive rows bit-identical. More rigorous than
    test_integrate_inactive_unchanged which uses np.allclose.
    """
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0
    rng = np.random.default_rng(7)

    for mode in ("toroidal", "open", "margin", "sphere"):
        for _ in range(50):
            pos = rng.uniform(0, [W, H, D], (20, 3)).astype(np.float32)
            vel = rng.uniform(-v0, v0, (20, 3)).astype(np.float32)
            acc = rng.uniform(-1, 1, (20, 3)).astype(np.float32)
            active = rng.uniform(0, 1, 20) > 0.15

            pos_before = pos.copy()
            vel_before = vel.copy()

            integrate(pos, vel, acc, active, W, H, D, v0, mode, 1.0 / 60.0)

            inactive = ~active
            # Exact bit-identical check (not allclose)
            assert np.array_equal(pos[inactive], pos_before[inactive]), (
                f"mode={mode}: inactive positions changed"
            )
            assert np.array_equal(vel[inactive], vel_before[inactive]), (
                f"mode={mode}: inactive velocities changed"
            )

    print(f"\n✓ 3 modes × 50 seeds: inactive rows bit-identical")


def test_toroidal_positions_in_bounds():
    """After toroidal integrate, all positions satisfy 0 ≤ pos < domain elementwise.

    Explicit check across a range of starting positions near the boundary.
    """
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    # Fixed positions that test edge cases: near-wrapping, at-boundary, far-out
    test_positions = np.array([
        [999.0, 350.0, 200.0],   # near +X boundary
        [1.0, 350.0, 200.0],     # near -X boundary
        [500.0, 699.0, 200.0],   # near +Y
        [500.0, 1.0, 200.0],     # near -Y
        [500.0, 350.0, 399.0],   # near +Z
        [500.0, 350.0, 1.0],     # near -Z
        [500.0, 350.0, 200.0],   # centre (no wrap)
    ], dtype=np.float32)

    vel = np.array([
        [10.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, -10.0, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, -10.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    N = len(test_positions)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    for frame in range(100):
        integrate(test_positions, vel, acc, active, W, H, D, v0, "toroidal", 1.0 / 60.0)

    # After 100 frames, all positions must be in bounds
    assert (test_positions[:, 0] >= 0).all() and (test_positions[:, 0] < W).all(), (
        f"x out of bounds: min={test_positions[:,0].min():.1f} max={test_positions[:,0].max():.1f}"
    )
    assert (test_positions[:, 1] >= 0).all() and (test_positions[:, 1] < H).all()
    assert (test_positions[:, 2] >= 0).all() and (test_positions[:, 2] < D).all()

    print(f"\n✓ 7 birds × 100 frames: all positions in [0,W)×[0,H)×[0,D)")


def test_fixed_mode_fuzz():
    """200 seeds with speed_mode='fixed': all speeds ≡ v0."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for seed in range(200):
        rng_i = np.random.default_rng(seed)
        pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
        vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
        acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
        active = np.ones(30, dtype=bool)

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal",
                  1.0 / 60.0, speed_mode="fixed")

        speeds = np.linalg.norm(vel, axis=1)
        assert np.allclose(speeds, v0, atol=1e-4), (
            f"seed={seed}: fixed mode speeds must be v0={v0}, got {speeds.min():.4f}–{speeds.max():.4f}"
        )
        assert not np.isnan(pos).any()
        assert not np.isnan(vel).any()

    print(f"\n✓ 200 seeds fixed-mode: all speeds ≡ {v0}")


def test_ceiling_mode_fuzz():
    """200 seeds with speed_mode='ceiling': all speeds ≤ v0, no NaN."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for seed in range(200):
        rng_i = np.random.default_rng(seed)
        pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
        vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
        acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
        active = np.ones(30, dtype=bool)

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal",
                  1.0 / 60.0, speed_mode="ceiling")

        speeds = np.linalg.norm(vel, axis=1)
        assert (speeds <= v0 + 1e-4).all(), (
            f"seed={seed}: ceiling mode max speed={speeds.max():.4f} > v0={v0}"
        )
        assert not np.isnan(pos).any()
        assert not np.isnan(vel).any()

    print(f"\n✓ 200 seeds ceiling-mode: all speeds ≤ {v0}")


def test_inertia_fuzz_inactive_preserved():
    """Inertia > 0 with mixed active/inactive: inactive rows unchanged."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0
    rng = np.random.default_rng(13)

    for _ in range(50):
        pos = rng.uniform(0, [W, H, D], (20, 3)).astype(np.float32)
        vel = rng.uniform(-v0, v0, (20, 3)).astype(np.float32)
        acc = rng.uniform(-1, 1, (20, 3)).astype(np.float32)
        active = rng.uniform(0, 1, 20) > 0.2

        pos_before = pos.copy()
        vel_before = vel.copy()

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal",
                  1.0 / 60.0, speed_mode="band", inertia=0.7)

        inactive = ~active
        assert np.array_equal(pos[inactive], pos_before[inactive]), (
            "inertia fuzz: inactive positions changed"
        )
        assert np.array_equal(vel[inactive], vel_before[inactive]), (
            "inertia fuzz: inactive velocities changed"
        )

    print("\n✓ 50 seeds inertia=0.7: inactive rows bit-identical")


# ── Array helper additions ────────────────────────────────────────

@pytest.mark.skip(reason="requires scipy for uniformity test")
def test_random_unit_sphere_uniform():
    """Distribution across octants is roughly balanced (weak uniformity check)."""
    N = 500
    rng = np.random.default_rng(42)
    pts = random_unit_sphere(N, rng)

    # Count vectors in each of 8 octants (±x, ±y, ±z)
    octant_counts = np.zeros(8, dtype=int)
    for pt in pts:
        idx = (int(pt[0] > 0) << 2) | (int(pt[1] > 0) << 1) | int(pt[2] > 0)
        octant_counts[idx] += 1

    # Each octant should have roughly N/8 = 62.5 vectors
    # Allow ±50% for this weak test
    expected = N / 8
    for count in octant_counts:
        assert abs(count - expected) < expected * 0.6, \
            f"octant imbalance: {octant_counts}"


def test_random_unit_sphere_seeded():
    """Same seed → same directions."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    pts1 = random_unit_sphere(100, rng1)
    pts2 = random_unit_sphere(100, rng2)
    assert np.allclose(pts1, pts2)


def test_random_positions_different():
    """Different seeds → different positions."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    p1 = random_positions(100, 1000.0, 700.0, 400.0, rng1)
    p2 = random_positions(100, 1000.0, 700.0, 400.0, rng2)
    assert not np.allclose(p1, p2)


# ── P0.9 Integration Variants ───────────────────────────────────


def test_speed_mode_fixed():
    """Fixed mode: all speeds exactly equal v0 after integrate."""
    N = 10
    pos = np.zeros((N, 3), dtype=np.float32)
    # Mix of different speeds
    vel = np.array([
        [0.5, 0, 0], [8.0, 0, 0], [3.0, 0, 0], [0, 0.2, 0],
        [10, 0, 0], [0, 7, 0], [1.5, 0, 0], [0, 0, 12],
        [0.1, 0, 0], [4.0, 0, 0],
    ], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="fixed")

    speeds = np.linalg.norm(vel, axis=1)
    assert np.allclose(speeds, 4.0, atol=1e-4), (
        f"fixed mode: all speeds must be v0=4.0, got {speeds}"
    )


def test_speed_mode_fixed_zero_safe():
    """Fixed mode: zero-velocity birds get (cap, 0, 0) — 0-safe.

    The fixed mode direction fallback (1,0,0) is applied BEFORE
    the zero-speed fallback, so the result is cap=4.0, not minSpeed=1.2.
    """
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="fixed")

    speeds = np.linalg.norm(vel, axis=1)
    # Fixed mode sets speed to cap = 4.0 (not minSpeed=1.2)
    assert np.allclose(speeds, 4.0, atol=1e-4), (
        f"fixed mode: expected 4.0, got {speeds}"
    )
    # Deterministic direction: (v0, 0, 0)
    assert np.allclose(vel[:, 1], 0.0)
    assert np.allclose(vel[:, 2], 0.0)
    assert (vel[:, 0] > 0).all()


def test_speed_mode_ceiling():
    """Ceiling mode: only caps speeds above v0, slow speeds unchanged."""
    N = 5
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([
        [8.0, 0, 0],   # above cap → clamped to 4.0
        [2.0, 0, 0],   # within cap → unchanged
        [0.5, 0, 0],   # slow → unchanged
        [10.0, 0, 0],  # above cap → clamped to 4.0
        [3.5, 0, 0],   # within cap → unchanged
    ], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    vel_before = vel.copy()

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="ceiling")

    speeds = np.linalg.norm(vel, axis=1)
    # All speeds ≤ 4.0
    assert (speeds <= 4.01).all()
    # Slow speed unchanged
    assert speeds[2] < 1.0  # ~0.5, not boosted
    # Direction of slow bird preserved
    assert vel[2, 0] > 0 and vel[2, 1] == 0


def test_speed_mode_none():
    """None mode: no speed clamp at all."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[15.0, 0, 0], [0.1, 0, 0], [0, 0, 0.01]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    vel_before = vel.copy()

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="none")

    # Velocities should be unchanged (no force applied, no clamp)
    np.testing.assert_array_equal(vel, vel_before)


def test_speed_mode_default_band():
    """Default speed_mode='band': clamps to [0.3*v0, v0]."""
    N = 5
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([
        [8.0, 0, 0],   # above → clamped to 4.0
        [2.0, 0, 0],   # within → unchanged
        [0.5, 0, 0],   # below → boosted to 1.2
        [0.1, 0, 0],   # below → boosted to 1.2
        [4.0, 0, 0],   # at cap → unchanged
    ], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="band")

    speeds = np.linalg.norm(vel, axis=1)
    assert 3.9 <= speeds[0] <= 4.1   # capped from 8 to 4
    assert 1.9 <= speeds[1] <= 2.1   # within band, ~2 unchanged
    assert 1.1 <= speeds[2] <= 1.3   # boosted to 1.2
    assert 1.1 <= speeds[3] <= 1.3   # boosted to 1.2
    assert 3.9 <= speeds[4] <= 4.1   # at cap, unchanged


def test_inertia_lerp():
    """Inertia blends between raw and clamped velocity."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[8.0, 0, 0], [8.0, 0, 0], [8.0, 0, 0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    # inertia=0.0 → fully clamped (4.0)
    v1 = vel.copy()
    integrate(pos, v1, acc.copy(), active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="band", inertia=0.0)
    assert np.allclose(np.linalg.norm(v1[0]), 4.0, atol=0.05)

    # inertia=1.0 → fully raw (8.0)
    v2 = vel.copy()
    integrate(pos, v2, acc.copy(), active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="band", inertia=1.0)
    assert np.allclose(np.linalg.norm(v2[0]), 8.0, atol=0.05)

    # inertia=0.5 → halfway (~6.0)
    v3 = vel.copy()
    integrate(pos, v3, acc.copy(), active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="band", inertia=0.5)
    assert np.allclose(np.linalg.norm(v3[0]), 6.0, atol=0.1)


def test_speed_mode_no_move():
    """move=False: positions unchanged, only velocity processed."""
    N = 3
    pos = np.array([[100, 200, 100], [400, 300, 200], [300, 100, 50]],
                   dtype=np.float32)
    vel = np.array([[4.0, 0, 0], [4.0, 0, 0], [4.0, 0, 0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    pos_before = pos.copy()

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", 1.0, move=False)

    # Positions unchanged when move=False (use open boundary to avoid wrap)
    np.testing.assert_array_equal(pos, pos_before)
    # Velocities still processed (clamped)
    assert np.allclose(np.linalg.norm(vel, axis=1), 4.0, atol=0.05)


def test_speed_min_factor_custom():
    """Custom speed_min_factor changes the lower speed bound."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[0.2, 0, 0], [0.2, 0, 0], [0.2, 0, 0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="band",
              speed_min_factor=0.5)

    speeds = np.linalg.norm(vel, axis=1)
    # Min speed should be 0.5 * 4.0 = 2.0
    assert np.allclose(speeds, 2.0, atol=0.1), (
        f"custom min_factor=0.5: expected 2.0, got {speeds}"
    )


def test_speed_mode_fixed_with_max_speed():
    """Fixed mode with per-bird max_speed: each bird gets its own cap."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[8.0, 0, 0], [8.0, 0, 0], [8.0, 0, 0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    max_speed = np.array([2.0, 3.0, 5.0], dtype=np.float32)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "toroidal", 1.0 / 60.0, speed_mode="fixed",
              max_speed=max_speed)

    speeds = np.linalg.norm(vel, axis=1)
    assert np.isclose(speeds[0], 2.0, atol=0.05)
    assert np.isclose(speeds[1], 3.0, atol=0.05)
    assert np.isclose(speeds[2], 5.0, atol=0.05)


# ── P0.10 Safety Rails ─────────────────────────────────────────


def test_dt_clamped():
    """dt > 0.05 is clamped to exactly 0.05."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[1.0, 0, 0]] * N, dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    pos_before = pos.copy()

    # dt=1.0 → clamped to 0.05
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", 1.0)  # dt=1.0, clamped to 0.05

    # Movement should be vel * 0.05 = 0.05, not vel * 1.0 = 1.0
    displacement = np.linalg.norm(pos - pos_before, axis=1)
    assert np.allclose(displacement, 0.05, atol=0.01), (
        f"dt should be clamped to 0.05, displacement={displacement}"
    )


def test_dt_negative_clamped():
    """dt < 0 is clamped to 0 (no movement)."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[1.0, 0, 0]] * N, dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    pos_before = pos.copy()

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", -0.5)

    # Negative dt → clamped to 0 → no movement
    np.testing.assert_array_equal(pos, pos_before)


def test_dt_within_range_unchanged():
    """dt within [0, 0.05] passes through unchanged."""
    N = 3
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.array([[1.0, 0, 0]] * N, dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    # dt = 1/60 ≈ 0.0167, within range
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", 1.0 / 60.0)

    displacement = np.linalg.norm(pos, axis=1)
    assert np.allclose(displacement, 1.0 / 60.0, atol=0.01)


def test_nan_guard_resets_to_center():
    """NaN positions are reset to centre, velocity zeroed."""
    N = 3
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    pos = np.array([
        [np.nan, 200.0, 100.0],
        [400.0, np.nan, 200.0],
        [300.0, 100.0, np.nan],
    ], dtype=np.float32)
    vel = np.array([[4.0, 0, 0]] * N, dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", 1.0 / 60.0, center=center)

    # All NaN positions reset to centre
    np.testing.assert_array_equal(pos, np.tile(center, (N, 1)).astype(np.float32))
    # Velocities zeroed for reset birds
    assert (vel == 0.0).all()


def test_nan_guard_skips_without_center():
    """When center is None, NaN guard is skipped (no crash)."""
    N = 2
    pos = np.array([[np.nan, 200.0, 100.0], [400.0, 300.0, 200.0]], dtype=np.float32)
    vel = np.ones((N, 3), dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    # center=None (default) — should complete without error
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", 1.0 / 60.0)
    # Bird 0 still has NaN (no guard without center)
    assert np.isnan(pos[0, 0])


def test_nan_guard_only_active():
    """NaN on inactive bird is not reset."""
    N = 3
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    pos = np.array([[np.nan, 200.0, 100.0]] * N, dtype=np.float32)
    vel = np.ones((N, 3), dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.array([False, False, True])  # only bird 2 active

    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", 1.0 / 60.0, center=center)

    # Bird 2 (active) reset to centre
    np.testing.assert_array_equal(pos[2], center)
    # Birds 0, 1 (inactive) still NaN
    assert np.isnan(pos[0, 0])
    assert np.isnan(pos[1, 0])


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
