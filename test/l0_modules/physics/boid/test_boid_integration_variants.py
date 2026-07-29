"""Unit tests for physics.boid — array helper additions, P0.9 integration variants (speed modes), P0.10 safety rails (dt clamping, NaN guard).

Split out of test_boid.py (file-size split).
"""

import numpy as np

from pymurmur.physics.boid import (
    integrate,
    random_positions,
    random_unit_sphere,
)

# ── Array helper additions ────────────────────────────────────────

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


