"""Unit tests for physics.boid — random positions/sphere, integrate() basics, speed clamp variants, boundary toroidal/open/margin/sphere_soft.

Split out of test_boid.py (file-size split).
"""

import numpy as np

from pymurmur.physics.boid import (
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
    """Bird outside sphere radius from domain centre is projected back.

    D1: Sphere centred on C, not origin. The default center=None now
    computes C = (W/2, H/2, D/2) inside integrate().
    """
    N = 1
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    # Place bird outside the 300-radius sphere from domain centre
    pos = np.array([[C[0] + 400.0, C[1], C[2]]], dtype=np.float32)
    vel = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, 300.0, 0.05)
    # D1: distance from domain centre C, not origin
    dist_from_C = np.linalg.norm(pos[0] - C)
    assert dist_from_C <= 300.0


def test_boundary_sphere_inside():
    """Bird inside sphere radius from domain centre is unchanged (no projection needed).

    D1: Sphere centred on C, not origin.
    """
    N = 1
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    # Bird inside 300-radius sphere from domain centre
    pos = np.array([[C[0] + 200.0, C[1], C[2]]], dtype=np.float32)
    vel = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    pos_before = pos.copy()
    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, 300.0, 0.05)
    # Bird inside sphere: position moves normally (not projected back)
    assert pos[0, 0] > pos_before[0, 0]  # moved forward
    dist_from_C = np.linalg.norm(pos[0] - C)
    assert dist_from_C <= 300.0  # still inside

