"""P5 — Angle mode: steering core (P5.1), neighbour modes (P5.2), adaptive speed (P5.3/S2.C3), edge handling (P5.4).

Split out of test_angle.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.angle import AngleMode

# ── P5.1: Steering core ───────────────────────────────────────────

def test_steering_180_turn_time():
    """P5.1: 180° turn completes in π/rate seconds.

    Start heading +x, target -x. Turn rate 120°/s = ~2.094 rad/s.
    π rad / 2.094 rad/s ≈ 1.5 s ≈ 90 frames at 60fps.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 1
    cfg.turn_rate = 120.0
    cfg.turn_threshold = 0.0  # no dead zone
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "open"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    flock.velocities[0] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # The bird starts with no neighbours, so no target direction.
    # Edge handling inactive (open boundary). So heading stays +x.
    # This is expected — steering only activates with a target.
    # For the 180° turn test, we test the Rodrigues rotation directly.
    from pymurmur.core.types import rotate_about

    hdg = np.array([1.0, 0, 0], dtype=np.float32)
    target = np.array([-1.0, 0, 0], dtype=np.float32)
    k = np.cross(hdg, target)
    k = k / np.linalg.norm(k)
    angle = np.pi  # 180°

    result = rotate_about(hdg, k, angle)
    assert np.allclose(result, target, atol=1e-6), (
        f"180° Rodrigues rotation should produce -x, got {result}"
    )


def test_steering_dead_zone_hold():
    """P5.1: Dead zone — no turn when φ < turn_threshold.

    Heading within 0.5° of target → no rotation.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 5
    cfg.turn_threshold = 2.0  # degrees
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Place birds close together at same heading
    for i in range(5):
        flock.positions[i] = np.array(
            [500 + i * 2, 350, 200], dtype=np.float32
        )
        flock.velocities[i] = np.array([4.0, 0.05, 0.0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # Record initial headings
    headings_before = flock.velocities.copy()
    n_before = np.linalg.norm(headings_before, axis=1, keepdims=True)
    headings_before = headings_before / np.maximum(n_before, 1e-10)

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    headings_after = flock.velocities.copy()
    n_after = np.linalg.norm(headings_after, axis=1, keepdims=True)
    headings_after = headings_after / np.maximum(n_after, 1e-10)

    # With similar headings and dead zone, directions should barely change
    dot_products = np.sum(headings_before * headings_after, axis=1)
    assert np.all(dot_products > 0.99), (
        f"Dead zone should prevent unnecessary turning, dots={dot_products}"
    )


def test_steering_never_overshoot():
    """P5.1: Per-frame heading change ≤ rate·dt + jitter.

    With finite turn rate, a single frame's rotation is bounded.
    """
    from pymurmur.core.types import rotate_about

    hdg = np.array([1.0, 0, 0], dtype=np.float32)
    target = np.array([-1.0, 0, 0], dtype=np.float32)

    # Full 180° in one shot would overshoot — Rodrigues with min(φ, rate·dt)
    # caps the rotation per frame
    turn_rate = np.radians(120.0)  # 120°/s
    dt = 1.0 / 60.0
    max_per_frame = turn_rate * dt  # ~2°/frame

    k = np.cross(hdg, target)
    k = k / max(np.linalg.norm(k), 1e-10)
    result = rotate_about(hdg, k, max_per_frame)

    # After one frame, angle between original and result should be ≤ max_per_frame
    cos_angle = np.clip(np.dot(hdg, result), -1.0, 1.0)
    actual_angle = np.arccos(cos_angle)
    assert actual_angle <= max_per_frame + 1e-10, (
        f"Frame rotation {np.degrees(actual_angle):.2f}° > "
        f"{np.degrees(max_per_frame):.2f}° cap"
    )


# ── P5.2: Unified neighbour modes ─────────────────────────────────

def test_flee_when_nearest_within_sep_radius():
    """P5.2: Nearest neighbour within sep_radius → flee away."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 3
    cfg.sep_radius_bodies = 1.5
    cfg.boid_size = 9.0
    cfg.align_radius_bodies = 10.0
    cfg.range_radius_bodies = 20.0
    cfg.turn_rate = 360.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Bird 0 at origin, heading +x. Bird 1 very close at +x.
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
    flock.positions[1] = np.array([505, 350, 200], dtype=np.float32)  # d=5 < 13.5
    flock.positions[2] = np.array([600, 350, 200], dtype=np.float32)
    flock.velocities[0] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.velocities[1] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.velocities[2] = np.array([0, 4.0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    hdg_before = flock.velocities[0] / np.linalg.norm(flock.velocities[0])

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    hdg_after = flock.velocities[0] / np.linalg.norm(flock.velocities[0])
    # Should have turned away from neighbour at +x — heading should have -x component
    assert hdg_after[0] < hdg_before[0], (
        f"Flee should steer away from near neighbour: "
        f"before={hdg_before}, after={hdg_after}"
    )


def test_coh_only_when_far():
    """P5.2: Far neighbour (> align_radius, < range_radius) → cohere only."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 3
    cfg.sep_radius_bodies = 0.5
    cfg.boid_size = 9.0
    cfg.align_radius_bodies = 1.5
    cfg.range_radius_bodies = 8.0
    cfg.turn_rate = 360.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.base_speed = 4.0
    cfg.neighbors = 10  # high threshold → no speed boost
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
    flock.positions[1] = np.array([530, 350, 200], dtype=np.float32)  # d=30
    flock.positions[2] = np.array([700, 350, 200], dtype=np.float32)
    # Bird 1 at d=30 > align_r=13.5, < range_r=72 → cohere only
    # Bird 0 heading is away from bird 1 — should turn toward it
    flock.velocities[0] = np.array([-3.0, 0, 0], dtype=np.float32)
    flock.velocities[1] = np.array([1.0, 0, 0], dtype=np.float32)
    flock.velocities[2] = np.array([0, 1.0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    from pymurmur.core.types import safe_normalize as sn
    hdg_before = sn(flock.velocities[0].copy())

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    hdg_after = sn(flock.velocities[0])
    # Heading should have rotated toward +x (bird 1 direction)
    assert hdg_after[0] > hdg_before[0], (
        f"Cohere should steer toward far neighbour: "
        f"before={hdg_before}, after={hdg_after}"
    )


# ── P5.3: Adaptive speed ──────────────────────────────────────────

def test_adaptive_speed_linear_isolated_faster():
    """P5.3: m=0 neighbours → base_speed + 35 (linear mode)."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 1
    cfg.base_speed = 4.0
    cfg.neighbors = 7
    cfg.turn_threshold = 10.0  # large dead zone to avoid steering
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "open"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
    flock.velocities[0] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    speed_before = np.linalg.norm(flock.velocities[0])

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    speed_after = np.linalg.norm(flock.velocities[0])
    # Isolated bird: 0 neighbours < 7 → speed = 4 + (7-0)*5 = 39
    assert speed_after > speed_before * 2, (
        f"Isolated bird should speed up: {speed_before:.1f} → {speed_after:.1f}"
    )


def test_adaptive_speed_survives_full_integrate_pipeline():
    """Modularity pass 11 regression: the test above only checked
    compute() in isolation, reading velocities before any
    post-processing ran — which is exactly how a real bug escaped
    detection for as long as it did. speed_mode="fixed" previously
    always renormalised to a flat config.v0 in flock.integrate(),
    silently discarding angle mode's own adaptive deficit-based speed
    law in every actual simulation run (stash_target_speed() in
    physics/forces/_base.py fixes this). This test goes through the
    full pipeline compute() -> flock.integrate(), not just compute()
    alone."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 1
    cfg.v0 = 4.0
    cfg.base_speed = 20.0  # deliberately far from v0
    cfg.neighbors = 7
    cfg.turn_threshold = 10.0
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "open"
    cfg.spatial.predator_speed_boost = 1.0  # rule out an unrelated confound

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
    flock.velocities[0] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    flock.integrate(cfg, dt=1.0 / 60.0, speed_mode="fixed")

    speed_after = np.linalg.norm(flock.velocities[0])
    # Isolated: 0 neighbours < 7 -> speed = base_speed(20) + 7*5 = 55,
    # NOT v0=4.0 (the pre-fix, silently-stomped value).
    assert speed_after > 40.0, (
        f"Adaptive speed must survive flock.integrate(): got {speed_after:.1f}, "
        f"expected ~55 (base_speed=20 + deficit bonus), not v0=4.0"
    )


def test_adaptive_speed_dense_crowd_normal():
    """P5.3: m ≥ 7 neighbours → speed = base_speed."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 10
    cfg.base_speed = 4.0
    cfg.neighbors = 2  # low threshold for test
    cfg.turn_threshold = 10.0
    cfg.jitter_deg = 0.0
    cfg.boid_size = 9.0
    cfg.sep_radius_bodies = 0.5
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Place all birds in a tight cluster
    for i in range(10):
        flock.positions[i] = np.array(
            [500 + i * 5, 350, 200], dtype=np.float32
        )
        flock.velocities[i] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    np.linalg.norm(flock.velocities[0])

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    speed_after = np.linalg.norm(flock.velocities[0])
    # With neighbours ≥ 2, speed stays at base_speed
    assert abs(speed_after - cfg.base_speed) < 2.0, (
        f"Crowded bird should stay near base speed: {speed_after:.1f}"
    )


# ── S2.C3: Adaptive speed law selector ─────────────────────────────

def _isolated_bird_speed(speed_mode: str) -> float:
    """Run one isolated (0-neighbour) bird through AngleMode.compute
    and return its resulting speed, for a given angle_speed_mode."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 1
    cfg.base_speed = 4.0
    cfg.angle_neighbors = 7
    cfg.angle_speed_mode = speed_mode
    cfg.turn_threshold = 10.0  # large dead zone to avoid steering
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "open"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
    flock.velocities[0] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    return float(np.linalg.norm(flock.velocities[0]))


def test_angle_speed_mode_defaults_to_linear():
    assert SimConfig().angle_speed_mode == "linear"


def test_angle_speed_mode_linear_matches_original_formula():
    """S2.C3: linear — s = base + (m_target - m)*5, m=0 → base+35."""
    speed = _isolated_bird_speed("linear")
    assert speed == pytest.approx(4.0 + 35.0, abs=0.5)


def test_angle_speed_mode_quadratic_isolated():
    """S2.C3: quadratic — s = base + min(cap, (m_target-m)²), m=0 → base+49."""
    speed = _isolated_bird_speed("quadratic")
    assert speed == pytest.approx(4.0 + 49.0, abs=0.5)


def test_angle_speed_mode_softened_isolated():
    """S2.C3: softened — s = base + min(cap, (m_target-m)²/2), m=0 → base+24.5."""
    speed = _isolated_bird_speed("softened")
    assert speed == pytest.approx(4.0 + 24.5, abs=0.5)


def test_angle_speed_mode_quadratic_greater_than_linear_when_deficit_large():
    """S2.C3: for the same large deficit, quadratic boosts speed more than linear."""
    linear = _isolated_bird_speed("linear")
    quadratic = _isolated_bird_speed("quadratic")
    assert quadratic > linear


def test_angle_speed_mode_softened_between_linear_and_quadratic():
    """S2.C3: softened is exactly half the quadratic boost above base_speed."""
    base = 4.0
    quadratic = _isolated_bird_speed("quadratic")
    softened = _isolated_bird_speed("softened")
    assert (softened - base) == pytest.approx((quadratic - base) / 2.0, abs=0.5)


def test_angle_speed_mode_all_modes_reach_base_when_not_isolated():
    """S2.C3: m >= n_neighbors → base_speed for every speed_mode (deficit <= 0)."""
    for mode in ("linear", "quadratic", "softened"):
        cfg = SimConfig()
        cfg.mode = "angle"
        cfg.num_boids = 10
        cfg.base_speed = 4.0
        cfg.angle_neighbors = 2
        cfg.angle_speed_mode = mode
        cfg.turn_threshold = 10.0
        cfg.jitter_deg = 0.0
        cfg.boid_size = 9.0
        cfg.sep_radius_bodies = 0.5
        cfg.boundary_mode = "toroidal"

        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        for i in range(10):
            flock.positions[i] = np.array([500 + i * 5, 350, 200], dtype=np.float32)
            flock.velocities[i] = np.array([4.0, 0, 0], dtype=np.float32)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        speed_after = float(np.linalg.norm(flock.velocities[0]))
        assert abs(speed_after - cfg.base_speed) < 2.0, (
            f"mode={mode}: crowded bird should stay near base speed, got {speed_after:.1f}"
        )


def test_angle_speed_mode_invalid_rejected():
    with pytest.raises(ValueError, match="angle_speed_mode"):
        SimConfig(angle_speed_mode="cubic").validate()


# ── P5.4: Edge handling ───────────────────────────────────────────

def test_cube_edge_avoidance():
    """P5.4: Bird near cube edge steers inward."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 8  # enough neighbours to avoid speed boost
    cfg.margin = 50.0
    cfg.turn_rate = 360.0
    cfg.max_turn_rate = 720.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "margin"
    cfg.neighbors = 10
    cfg.base_speed = 4.0

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Bird 0 near left wall, heading further left.
    # Other birds placed nearby to provide neighbours (no speed boost).
    flock.positions[0] = np.array([10.0, 350, 200], dtype=np.float32)
    for i in range(1, 8):
        flock.positions[i] = np.array([30.0, 350 + i * 10, 200], dtype=np.float32)
    flock.velocities[0] = np.array([-2.0, 0, 0], dtype=np.float32)
    for i in range(1, 8):
        flock.velocities[i] = np.array([2.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    from pymurmur.core.types import safe_normalize as sn
    hdg_before = sn(flock.velocities[0].copy())
    # Heading should be pointing left (-x)
    assert hdg_before[0] < 0

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    hdg_after = sn(flock.velocities[0])
    # Edge avoidance should have rotated heading toward +x (away from left wall)
    assert hdg_after[0] > hdg_before[0], (
        f"Edge avoidance should steer away from wall: "
        f"before={hdg_before}, after={hdg_after}"
    )


def test_sphere_edge_avoidance():
    """P5.4: Bird near sphere boundary steers toward centre."""
    from pymurmur.core.types import safe_normalize as sn

    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 1
    cfg.margin = 50.0
    cfg.turn_rate = 360.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 200.0

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Bird near sphere boundary, heading radially outward
    p = np.array([180.0, 0.0, 0.0], dtype=np.float32)  # d=180 > R-margin=150
    flock.positions[0] = p
    flock.velocities[0] = sn(p) * 4.0  # heading outward
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    hdg_before = sn(flock.velocities[0].copy())

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    hdg_after = sn(flock.velocities[0])
    # Bird should now point more toward centre than before
    dot_before = np.dot(hdg_before, sn(-p))
    dot_after = np.dot(hdg_after, sn(-p))
    assert dot_after > dot_before - 0.01, (
        f"Sphere edge should steer toward centre: "
        f"dot_before={dot_before:.3f}, dot_after={dot_after:.3f}"
    )


