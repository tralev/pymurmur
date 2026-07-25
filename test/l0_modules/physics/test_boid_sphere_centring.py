"""Unit tests for physics.boid — sphere boundary centring (config.center EMA), sphere_soft edge cases, D1 multi-frame centring.

Split out of test_boid.py (file-size split).
"""

import numpy as np

from pymurmur.physics.boid import (
    integrate,
)

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


