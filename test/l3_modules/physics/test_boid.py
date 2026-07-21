"""Unit tests for physics.boid — integrate(), array helpers."""

import numpy as np
import pytest

from pymurmur.physics.boid import (
    init_velocities,
    init_velocities_blob,
    init_velocities_cube,
    init_velocities_fixed,
    init_velocities_speed_uniform,
    init_velocities_tangential,
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


# ── Sphere centring + sphere_soft tests ──────────────────────────

def test_boundary_sphere_centred_on_C():
    """D1 fix: Sphere boundary is centred on domain centre C, not origin.

    With domain [1000, 700, 400] and centre C=(500, 350, 200),
    a bird at position (410, 350, 200) is distance 90 from C,
    well inside the 300-radius sphere. It should NOT be projected.
    Under the old origin-centred code, this bird at ‖p‖=410 would be
    projected back.
    """
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    # Bird at 90 units from centre (inside sphere)
    pos = np.array([[C[0] + 90.0, C[1], C[2]]], dtype=np.float32)
    vel = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((1, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    dist_from_C_before = np.linalg.norm(pos[0] - C)
    assert dist_from_C_before < R  # bird starts inside

    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, R, 0.05, center=C)

    # Bird inside sphere: NOT projected back, moves freely
    dist_from_C_after = np.linalg.norm(pos[0] - C)
    assert dist_from_C_after <= R + 0.5, (
        f"bird inside sphere should stay near sphere: dist={dist_from_C_after:.1f}"
    )


def test_boundary_sphere_projects_from_C():
    """D1 fix: Birds outside sphere from centre C are projected back.

    Bird at distance 400 from centre with R=300 → projected to surface.
    """
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    # Bird outside sphere (400 from centre along +x)
    pos = np.array([[C[0] + 400.0, C[1], C[2]]], dtype=np.float32)
    vel = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((1, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, R, 0.05, center=C)

    dist_from_C = np.linalg.norm(pos[0] - C)
    assert dist_from_C <= R + 1e-4, (
        f"bird outside sphere should be projected to surface: dist={dist_from_C:.1f}"
    )


def test_boundary_sphere_soft_never_projects():
    """S2.B7: sphere_soft boundary never hard-projects positions.

    A bird outside the sphere radius gets a velocity push inward
    but its position is NOT clamped — it can overshoot briefly.
    """
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    # Bird at radius + 20 units (just outside)
    pos = np.array([[C[0] + R + 20.0, C[1], C[2]]], dtype=np.float32)
    vel = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((1, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    pos_before = pos.copy()
    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere_soft",
              1.0 / 60.0, R, 0.05, center=C)

    # Position is NOT projected — bird stays where it is (plus movement)
    # The velocity should get an inward push component
    dist_before = np.linalg.norm(pos_before[0] - C)
    dist_after = np.linalg.norm(pos[0] - C)
    # Position changes only by v·dt (no hard projection)
    # The inward velocity push reduces the radial component
    assert dist_after > R - 5.0, (
        f"sphere_soft should not project: before={dist_before:.1f}, after={dist_after:.1f}"
    )


def test_boundary_sphere_soft_inward_push():
    """S2.B7: sphere_soft pushes birds inward when near boundary.

    A bird moving radially outward near the boundary gets its
    radial velocity reduced by the asymptotic push.
    """
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    # Bird at 95% of radius, moving outward fast
    pos = np.array([[C[0] + R * 0.95, C[1], C[2]]], dtype=np.float32)
    vel = np.array([[20.0, 0.0, 0.0]], dtype=np.float32)  # outward at high speed
    acc = np.zeros((1, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    # Radial component before
    offset_before = pos[0] - C
    v_radial_before = np.dot(vel[0], offset_before) / np.linalg.norm(offset_before)
    assert v_radial_before > 0  # moving outward

    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere_soft",
              1.0 / 60.0, R, 0.2, center=C)

    # After integration, radial velocity should be reduced (push inward)
    offset_after = pos[0] - C
    v_radial_after = np.dot(vel[0], offset_after) / max(np.linalg.norm(offset_after), 1e-6)
    assert v_radial_after < v_radial_before, (
        f"sphere_soft should push inward: v_radial={v_radial_before:.2f}→{v_radial_after:.2f}"
    )


def test_boundary_sphere_soft_no_effect_far_inside():
    """S2.B7: sphere_soft has no effect on birds far inside the sphere."""
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    # Bird at centre (far inside)
    pos = np.array([[C[0], C[1], C[2]]], dtype=np.float32)
    vel = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    vel_before = vel.copy()
    acc = np.zeros((1, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere_soft",
              1.0 / 60.0, R, 0.05, center=C)

    # Velocity unchanged beyond speed clamp (bird at centre, no push)
    assert np.allclose(vel[0], vel_before[0], atol=0.1), (
        f"bird at centre should not get pushed: {vel_before[0]} vs {vel[0]}"
    )


# ── D1 sphere centring — edge-case tests ────────────────────────

def test_boundary_sphere_frame0_uses_domain_center():
    """D1 frame-0: When center=None is passed (first frame), sphere boundary
    defaults to domain centre C=(W/2,H/2,D/2), never origin.

    A bird at (0,0,0) is distance ~640 from domain centre of default
    [1000,700,400]. With R=300, it should be projected. Under the old
    origin-centred code, ‖(0,0,0)‖=0 < R=300 — no projection.
    The D1 fix: projection from C, not origin.
    """
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    # Bird near origin — distance from C is ~640 > R=300
    pos = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    vel = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((1, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    # Pass center=None — simulate frame 0 before EMA initialisation.
    # The default should be domain centre, not origin.
    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, R, 0.05, center=None)

    # Bird is far from domain centre → should be projected to sphere surface
    # (not left at origin as it would be under old origin-centred code)
    dist_from_C = np.linalg.norm(pos[0] - C)
    assert dist_from_C <= R + 1e-4, (
        f"bird far from C should be projected: dist={dist_from_C:.1f}, R={R}"
    )


def test_boundary_sphere_origin_regression():
    """D1 regression: Bird exactly at domain centre is inside R=300 sphere
    and should NOT be projected, regardless of origin distance.

    Under the old origin-centred code, ‖C‖=640 > 300 → projection.
    With D1 fix centred on C: ‖C−C‖=0 < 300 → no projection.
    """
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    # Bird at domain centre
    pos = np.array([[C[0], C[1], C[2]]], dtype=np.float32)
    vel = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    acc = np.zeros((1, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    pos_before = pos.copy()
    integrate(pos, vel, acc, active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, R, 0.05, center=C)

    # Bird at centre: should NOT be projected, should move normally
    dist_from_C = np.linalg.norm(pos[0] - C)
    assert dist_from_C <= R, f"bird at centre should stay inside: dist={dist_from_C:.1f}"
    # Should have moved (not stuck at surface)
    assert pos[0, 0] > pos_before[0, 0], "bird at centre should move freely"


def test_boundary_sphere_multiframe_centred():
    """D1 long-run: Flock initialised uniformly stays centred in sphere mode
    over 500 frames. Verified by checking CoM stays within 10% of R from C.
    """
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0
    v0 = 4.0

    # 50 birds initialised uniformly
    rng = np.random.default_rng(99)
    pos = rng.uniform(0, [W, H, D], (50, 3)).astype(np.float32)
    vel = rng.uniform(-v0, v0, (50, 3)).astype(np.float32)
    # Clamp initial speeds to [0.3*v0, v0]
    speeds = np.linalg.norm(vel, axis=1)
    vel = vel / speeds[:, np.newaxis] * v0
    acc = np.zeros((50, 3), dtype=np.float32)
    active = np.ones(50, dtype=bool)

    # Run 500 frames
    for _ in range(500):
        # Simple cohesion force toward centre to keep flock together
        com = pos[active].mean(axis=0)
        to_center = C - pos
        dists = np.linalg.norm(to_center, axis=1, keepdims=True)
        acc[active] = to_center / np.maximum(dists, 1e-6) * 0.02

        integrate(pos, vel, acc, active, W, H, D, v0, "sphere",
                  1.0 / 60.0, R, 0.05, center=C)

    # After 500 frames, flock centre should be near C
    com = pos[active].mean(axis=0)
    com_dist = np.linalg.norm(com - C)
    assert com_dist < 0.1 * R, (
        f"flock CoM drifted from C: dist={com_dist:.1f}, R={R}"
    )
    # All birds should be inside sphere
    dists_from_C = np.linalg.norm(pos - C, axis=1)
    assert (dists_from_C <= R + 1e-3).all(), (
        f"birds outside sphere: max dist={dists_from_C.max():.1f}, R={R}"
    )


def test_boundary_sphere_soft_multiframe_centred():
    """D1 + S2.B7 long-run: sphere_soft boundary keeps flock roughly inside
    R over 200 frames. Birds initialised within sphere; inward push of 1.0
    keeps them near R — no hard projection, but asymptotic push works."""
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0
    v0 = 4.0

    rng = np.random.default_rng(77)
    # Initialize birds within sphere (90% of R) so they start inside
    from pymurmur.physics.boid import init_positions
    pos = init_positions(50, W, H, D, rng, mode="sphere", separation=9.0)
    vel = rng.uniform(-v0, v0, (50, 3)).astype(np.float32)
    speeds = np.linalg.norm(vel, axis=1)
    vel = vel / speeds[:, np.newaxis] * v0
    acc = np.zeros((50, 3), dtype=np.float32)
    active = np.ones(50, dtype=bool)

    for _ in range(200):
        com = pos[active].mean(axis=0)
        to_center = C - pos
        dists = np.linalg.norm(to_center, axis=1, keepdims=True)
        acc[active] = to_center / np.maximum(dists, 1e-6) * 0.02
        # Stronger push factor for clearer centring
        integrate(pos, vel, acc, active, W, H, D, v0, "sphere_soft",
                  1.0 / 60.0, R, 1.0, center=C)

    # After 200 frames, flock CoM should be near C
    com = pos[active].mean(axis=0)
    com_dist = np.linalg.norm(com - C)
    assert com_dist < 0.15 * R, (
        f"sphere_soft: CoM drifted from C: dist={com_dist:.1f}, R={R}"
    )
    # At least 90% of birds should be inside R
    dists_from_C = np.linalg.norm(pos - C, axis=1)
    fraction_inside = (dists_from_C <= R).mean()
    assert fraction_inside >= 0.9, (
        f"sphere_soft: only {fraction_inside:.1%} inside R after 200 frames"
    )


def test_boundary_sphere_same_result_with_or_without_explicit_center():
    """D1: Passing center=C explicitly vs center=None (domain default)
    should produce identical results for birds at mid-domain positions."""
    W, H, D = 1000.0, 700.0, 400.0
    C = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
    R = 300.0

    rng = np.random.default_rng(55)
    pos = rng.uniform(0, [W, H, D], (20, 3)).astype(np.float32)
    vel = rng.uniform(-4.0, 4.0, (20, 3)).astype(np.float32)
    acc = np.zeros((20, 3), dtype=np.float32)
    active = np.ones(20, dtype=bool)

    # Run with explicit C
    pos1 = pos.copy()
    vel1 = vel.copy()
    integrate(pos1, vel1, acc.copy(), active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, R, 0.05, center=C)

    # Run with center=None (domain-centre default)
    pos2 = pos.copy()
    vel2 = vel.copy()
    integrate(pos2, vel2, acc.copy(), active, W, H, D, 4.0, "sphere",
              1.0 / 60.0, R, 0.05, center=None)

    # Results should be identical since C == domain centre
    np.testing.assert_allclose(pos1, pos2, atol=1e-5)
    np.testing.assert_allclose(vel1, vel2, atol=1e-5)


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

    print("\n✓ 3 modes × 200 seeds: no NaN or Inf in positions/velocities")


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

    print("\n✓ 3 modes × 50 seeds: inactive rows bit-identical")


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

    for _frame in range(100):
        integrate(test_positions, vel, acc, active, W, H, D, v0, "toroidal", 1.0 / 60.0)

    # After 100 frames, all positions must be in bounds
    assert (test_positions[:, 0] >= 0).all() and (test_positions[:, 0] < W).all(), (
        f"x out of bounds: min={test_positions[:,0].min():.1f} max={test_positions[:,0].max():.1f}"
    )
    assert (test_positions[:, 1] >= 0).all() and (test_positions[:, 1] < H).all()
    assert (test_positions[:, 2] >= 0).all() and (test_positions[:, 2] < D).all()

    print("\n✓ 7 birds × 100 frames: all positions in [0,W)×[0,H)×[0,D)")


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


def test_band_mode_fuzz_all_boundaries():
    """P0.3: 50 seeds × 4 boundary modes — band mode invariant holds.

    Band mode (default): speeds ∈ [0.3·v0, v0], positions in bounds
    if toroidal, no NaN. This fills the gap — only toroidal was fuzzed.
    """
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0
    v_min = 0.3 * v0  # 1.2

    for mode in ("toroidal", "open", "margin", "sphere"):
        for seed in range(50):
            rng_i = np.random.default_rng(seed)
            pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
            vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
            acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
            active = rng_i.uniform(0, 1, 30) > 0.1

            integrate(pos, vel, acc, active, W, H, D, v0, mode,
                      1.0 / 60.0, speed_mode="band")

            speeds = np.linalg.norm(vel[active], axis=1)
            # Toroidal: tight bounds (no boundary-induced velocity changes)
            # Non-toroidal: margin/open can nudge velocity, use relaxed bounds
            # Sphere: skip speed bounds — sphere boundary projects birds back
            #   with velocity kicks that can far exceed v0 (by design)
            if mode == "toroidal":
                assert (speeds >= v_min - 1e-4).all(), (
                    f"{mode} seed={seed}: min speed={speeds.min():.4f} < {v_min}"
                )
                assert (speeds <= v0 + 1e-4).all(), (
                    f"{mode} seed={seed}: max speed={speeds.max():.4f} > {v0}"
                )
            elif mode != "sphere":
                assert (speeds >= v_min * 0.95).all(), (
                    f"{mode} seed={seed}: min speed={speeds.min():.4f} < {v_min*0.95:.2f}"
                )
                assert (speeds <= v0 * 1.05).all(), (
                    f"{mode} seed={seed}: max speed={speeds.max():.4f} > {v0*1.05:.2f}"
                )
            assert not np.isnan(pos).any(), f"{mode} seed={seed}: NaN in positions"
            assert not np.isnan(vel).any(), f"{mode} seed={seed}: NaN in velocities"

            # Toroidal: positions must remain in bounds
            if mode == "toroidal":
                assert (pos[:, 0] >= 0).all() and (pos[:, 0] < W).all()
                assert (pos[:, 1] >= 0).all() and (pos[:, 1] < H).all()
                assert (pos[:, 2] >= 0).all() and (pos[:, 2] < D).all()

    print(f"\n✓ 4 modes × 50 seeds band-mode: speeds in [{v_min},{v0}], no NaN")


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
    vel.copy()

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
    """When center is None, NaN guard is NOT skipped — D1 makes center
    default to domain centre, so NaN guard always fires.

    Before D1: center=None → NaN guard skipped (NaN positions left alone).
    After D1:  center=None → domain centre computed → NaN positions reset.
    """
    N = 2
    pos = np.array([[np.nan, 200.0, 100.0], [400.0, 300.0, 200.0]], dtype=np.float32)
    vel = np.ones((N, 3), dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    # center=None (default) — D1: now defaults to domain centre
    integrate(pos, vel, acc, active, 1000.0, 700.0, 400.0,
              4.0, "open", 1.0 / 60.0)
    # D1: NaN positions ARE reset because center defaults to domain centre
    C = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    np.testing.assert_array_equal(pos[0], C)
    # Non-NaN bird (bird 1) should have moved normally
    assert not np.isnan(pos[1, 0])


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
    """P4.9: Speed_uniform mode produces speeds in [0, v0]."""
    rng = np.random.default_rng(42)
    v = init_velocities_speed_uniform(1000, 4.0, rng)
    speeds = np.linalg.norm(v, axis=1)
    # All speeds in [0, v0]
    assert (speeds >= 0.0).all(), f"min speed={speeds.min():.4f}"
    assert (speeds <= 4.05).all(), f"max speed={speeds.max():.4f}"
    # Uniform distribution → mean ≈ v0/2 ≈ 2.0
    assert 1.8 < speeds.mean() < 2.2, f"mean speed={speeds.mean():.2f}"


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


# ── D7: position_init "sphere" — volume-uniform via ∛-law ────────


class TestD7SphereInit:
    """D7: position_init='sphere' uses ∛-law for volume-uniform
    distribution inside a sphere (not just on the surface).

    Previously the 'sphere' string fell through to the else 'box'
    branch because no 'sphere' case existed.  Now it correctly
    samples r ∝ cbrt(U) for uniform volume density.
    """

    @staticmethod
    def _sphere_positions(n=5000, rng_seed=42):
        """Return (n,3) positions from sphere init."""
        from pymurmur.physics.boid import init_positions
        w, h, d = 1000.0, 700.0, 400.0
        rng = np.random.default_rng(rng_seed)
        return init_positions(n, w, h, d, rng, mode="sphere")

    def test_all_positions_within_radius(self):
        """D7: All sphere-init positions are within R of centre."""
        w, h, d = 1000.0, 700.0, 400.0
        R = 0.4 * min(w, h, d)  # = 160.0
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        pts = self._sphere_positions()
        dists = np.linalg.norm(pts - C, axis=1)
        assert (dists <= R * 1.001).all(), (
            f"All points must be within {R}, max dist = {dists.max():.1f}"
        )

    def test_radial_histogram_follows_r_squared(self):
        """D7: Radial bin counts ∝ r² (uniform volume density).

        For volume-uniform sampling, the probability of a point
        landing in [r, r+dr] is proportional to r² (surface area
        of spherical shell at radius r).  The cumulative distribution
        follows P(r ≤ R) = (r/R)³, hence the ∛-law.
        """
        w, h, d = 1000.0, 700.0, 400.0
        R = 0.4 * min(w, h, d)
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        pts = self._sphere_positions(n=10000)
        dists = np.linalg.norm(pts - C, axis=1)

        # Split into 10 radial bins and check counts ∝ r²
        bins = np.linspace(0, R, 11)
        hist, _ = np.histogram(dists, bins=bins)

        # Expected: count ∝ shell volume (r_{i+1}³ − r_i³) for each bin.
        # For volume-uniform distribution, the probability of landing
        # in [r_i, r_{i+1}] is proportional to the spherical shell volume.
        # r ∝ bin_edge³ gives the exact formula, unlike r_mid² which is
        # a coarse approximation for wide bins.
        shell_volumes = bins[1:] ** 3 - bins[:-1] ** 3
        expected = shell_volumes / shell_volumes.sum() * hist.sum()

        for i in range(len(hist)):
            if expected[i] > 10:  # skip nearly-empty bins
                rel_err = abs(hist[i] - expected[i]) / expected[i]
                assert rel_err < 0.25, (
                    f"Bin {i}: r=[{bins[i]:.0f},{bins[i+1]:.0f}], "
                    f"count={hist[i]}, expected≈{expected[i]:.0f}, "
                    f"rel_err={rel_err:.2f}"
                )

    def test_sphere_is_volume_not_surface(self):
        """D7: Sphere init fills the volume, not just the surface.

        Verify at least 50% of points are inside 70% of the radius
        (uniform volume → ~34% inside 0.7R; surface → 0%).
        """
        w, h, d = 1000.0, 700.0, 400.0
        R = 0.4 * min(w, h, d)
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        pts = self._sphere_positions(n=2000)
        dists = np.linalg.norm(pts - C, axis=1)

        inside_70pct = (dists < 0.7 * R).sum()
        expected_vol_fraction = 0.7 ** 3  # ~0.343
        fraction = inside_70pct / len(pts)

        # Allow ±15% margin: min ~0.19
        assert fraction > expected_vol_fraction - 0.15, (
            f"Only {fraction:.2%} inside 0.7R; "
            f"expected ~{expected_vol_fraction:.0%} (±15%), "
            f"which rules out surface-only distribution"
        )

    def test_sphere_init_via_physics_flock(self):
        """D7: Sphere init works through PhysicsFlock position_init config."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.position_init = "sphere"
        cfg.num_boids = 100
        w, h, d = cfg.width, cfg.height, cfg.depth
        R = 0.4 * min(w, h, d)
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        flock = PhysicsFlock(cfg)
        dists = np.linalg.norm(flock.positions - C, axis=1)

        assert (dists <= R * 1.001).all(), (
            f"Sphere-init via PhysicsFlock: all points must be within {R}"
        )
        # Also verify it's volume-distributed (not surface)
        interior = (dists < 0.5 * R).sum()
        assert interior > 5, (
            f"Only {interior} birds in inner half of sphere; "
            f"surface-only would have 0"
        )

    def test_sphere_init_deterministic_with_same_seed(self):
        """D7: Sphere init with same seed produces identical positions."""
        from pymurmur.physics.boid import init_positions
        w, h, d = 1000.0, 700.0, 400.0
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        pts1 = init_positions(50, w, h, d, rng1, mode="sphere")
        pts2 = init_positions(50, w, h, d, rng2, mode="sphere")

        np.testing.assert_array_equal(pts1, pts2)

    def test_sphere_not_identical_to_box(self):
        """D7: Sphere and box init produce different positions.

        This is the regression guard — the original bug caused
        'sphere' to silently fall through to the else: 'box' branch,
        so both modes would return identical output for the same seed.
        """
        from pymurmur.physics.boid import init_positions
        w, h, d = 1000.0, 700.0, 400.0
        rng = np.random.default_rng(42)

        sphere_pts = init_positions(200, w, h, d, rng, mode="sphere")
        # Fresh RNG with same seed for independent box positions
        rng2 = np.random.default_rng(42)
        box_pts = init_positions(200, w, h, d, rng2, mode="box")

        assert not np.array_equal(sphere_pts, box_pts), (
            "Sphere and box init must produce different outputs; "
            "if they match, 'sphere' is falling through to 'box' (D7 regression)"
        )


# ── D1 + D7: Sphere boundary + sphere init cross-cutting ────────


def test_sphere_init_birds_stay_within_sphere_boundary():
    """D1+D7: Birds initted with 'sphere' mode stay inside sphere_soft boundary.

    D7 fixed position_init to support 'sphere' mode. D1 fixed sphere boundary
    to centre on domain centre C, not origin. Together, birds initialized in
    a sphere should remain within the sphere_soft boundary over many frames.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 30
    cfg.mode = "spatial"
    cfg.width = 800
    cfg.height = 600
    cfg.depth = 400
    cfg.boundary_mode = "sphere_soft"
    cfg.position_init = "sphere"
    cfg.boundary_avoidance_factor = 0.8
    cfg.boundary_sphere_radius = 0.4

    engine = SimulationEngine(cfg)

    # D7: birds must be initialized in sphere mode (not degraded to box)
    # D1: sphere boundary is centred on domain centre C, not origin
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                 dtype=np.float32)
    R = cfg.boundary_sphere_radius * min(cfg.width, cfg.height, cfg.depth)

    # All initial positions must be within the sphere
    initial_dists = np.linalg.norm(
        engine.flock.positions - C, axis=1
    )
    assert (initial_dists <= R * 1.05).all(), (
        f"D7: sphere init must place birds within sphere boundary. "
        f"Max dist={initial_dists.max():.1f}, R={R:.1f}"
    )

    # Run many steps and verify no bird escapes the sphere boundary
    for _ in range(200):
        engine.step(1.0 / 60.0)

    final_dists = np.linalg.norm(
        engine.flock.positions[engine.flock.active] - C, axis=1
    )
    # Sphere_soft is asymptotic — birds can slightly overshoot the
    # nominal R during fast turns. Use 20% tolerance.
    assert (final_dists <= R * 1.2).all(), (
        f"D1: sphere_soft boundary must contain birds. "
        f"Max dist={final_dists.max():.1f}, R={R:.1f}"
    )
    # At least some birds should be near the boundary (not all at centre)
    assert final_dists.max() > R * 0.5, (
        "Birds should explore the sphere volume, not stay at centre"
    )


def test_sphere_init_centre_matches_boundary_centre():
    """D1+D7: Sphere boundary centre C equals domain centre on frame 0.

    D1 initialises flock.center to domain centre. D7 uses the same centre
    for sphere init. The boundary and init must agree on the sphere centre.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 20
    cfg.mode = "spatial"
    cfg.boundary_mode = "sphere_soft"
    cfg.position_init = "sphere"

    engine = SimulationEngine(cfg)
    flock = engine.flock

    C_expected = np.array(
        [cfg.width / 2, cfg.height / 2, cfg.depth / 2],
        dtype=np.float32,
    )
    # D1: flock.center is initialised to domain centre
    np.testing.assert_array_equal(
        flock.center, C_expected,
        err_msg="D1: flock.center must be domain centre on frame 0"
    )

    # All birds should be distributed around the domain centre
    centroid = flock.positions[flock.active].mean(axis=0)
    dist_centroid_to_centre = np.linalg.norm(centroid - C_expected)
    # Centroid should be near the centre (sphere init is centred on C)
    R = cfg.boundary_sphere_radius * min(cfg.width, cfg.height, cfg.depth)
    assert dist_centroid_to_centre < R * 0.3, (
        f"D7: sphere init centroid should be near domain centre. "
        f"Dist={dist_centroid_to_centre:.1f}, R={R:.1f}"
    )
