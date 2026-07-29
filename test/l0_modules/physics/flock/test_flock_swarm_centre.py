"""P0.5 — Smoothed swarm centre tests for physics.flock.

Split out of test_flock.py (file-size split) — core init/state tests and
P0.4 determinism tests stay elsewhere; this file covers flock.center's
EMA-smoothed centroid tracking.
"""

import numpy as np

from pymurmur.physics.flock import PhysicsFlock
from test.helpers import _step_flock  # noqa: E402 — shared test helper


def test_center_initialised_none(default_config):
    """D1: flock.center is initialised to domain centre (not None).

    Before D1: center was None before any step (snap-to-centroid on frame 0).
    After D1:  center starts at (W/2, H/2, D/2) so sphere boundary is
               always centred on domain centre from frame 0.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    assert flock.center is not None, "D1: center initialised to domain centre"
    assert flock.center.shape == (3,), "center must be (3,) float32"
    assert flock.center.dtype == np.float32
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    np.testing.assert_array_equal(flock.center, C)


def test_center_set_after_first_step(default_config):
    """D1: flock.center is initialised to domain centre before first step,
    and EMA-drifts toward the centroid after step().

    Before D1: center snapped to centroid on frame 0 (None → centroid).
    After D1:  center starts at domain centre, then EMA blends toward
               centroid: center ← center + 0.5·(centroid − center).
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    C_initial = flock.center.copy()
    _step_flock(flock, cfg, 1.0 / 60.0)
    assert flock.center is not None, "center must be set after first step"
    assert flock.center.shape == (3,), "center must be (3,) float32"
    assert flock.center.dtype == np.float32
    # D1: After EMA, center moves toward centroid (not identity snap)
    # It should differ from initial domain centre after the first step
    assert not np.allclose(flock.center, C_initial), (
        "center should EMA-drift from domain centre toward centroid"
    )


def test_center_close_to_centroid(default_config):
    """D1: After first step, centre is halfway between domain centre and centroid.

    Before D1: center snapped to centroid (EMA init snap, exact match).
    After D1:  center starts at domain centre, EMA: center += 0.5·(centroid − center).
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    C_initial = flock.center.copy()
    _step_flock(flock, cfg, 1.0 / 60.0)

    centroid = flock.positions[flock.active].mean(axis=0)
    # D1: After EMA, center = (C_initial + centroid) / 2
    expected = (C_initial + centroid) / 2.0
    np.testing.assert_allclose(
        flock.center, expected, rtol=0.05, atol=5.0,
        err_msg="center should be halfway between domain centre and centroid"
    )


def test_center_ema_smoothing(default_config):
    """Teleport flock — centre moves exactly 50% toward centroid (EMA α=0.5).

    Uses update_center() directly to isolate EMA behaviour from physics.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    # Initialise centre
    flock.update_center()
    old_center = flock.center.copy()

    # Teleport all birds far away (no step, no physics)
    flock.positions += np.array([500.0, 0.0, 0.0], dtype=np.float32)

    # Update centre directly — pure EMA, no force/integrate
    flock.update_center()

    new_centroid = flock.positions[flock.active].mean(axis=0)
    distance_center_moved = np.linalg.norm(flock.center - old_center)
    distance_to_centroid = np.linalg.norm(new_centroid - old_center)

    # Centre should have moved toward the centroid
    assert distance_center_moved > 0, "center should move after teleport"

    # Centre should NOT have reached the centroid in one step (EMA lag)
    assert distance_center_moved < distance_to_centroid, (
        f"EMA lag: center moved {distance_center_moved:.1f}, "
        f"but centroid moved {distance_to_centroid:.1f}"
    )

    # EMA α=0.5 → centre moves exactly 50% of the way
    expected_move = 0.5 * distance_to_centroid
    assert np.isclose(distance_center_moved, expected_move, atol=1e-4), (
        f"EMA α=0.5: expected move ≈ {expected_move:.1f}, "
        f"got {distance_center_moved:.1f}"
    )


def test_center_converges(default_config):
    """Pure EMA: after teleport, centre converges to <1% of centroid in ~7 frames.

    Uses update_center() directly (no physics) so convergence follows
    error = D · (0.5)^n exactly.  With D=500, error < 5.0 after 7 frames.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    # Initialise centre
    flock.update_center()

    # Teleport — zero velocities so no physics drift
    flock.positions += np.array([500.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[:] = 0.0
    flock.accelerations[:] = 0.0

    centroid = flock.positions[flock.active].mean(axis=0)
    for i in range(20):  # noqa: B007
        flock.update_center()
        error = np.linalg.norm(flock.center - centroid)
        if error < 0.01 * np.linalg.norm(centroid):
            break

    assert i < 10, (
        f"Pure EMA should converge within 10 frames, took {i + 1}"
    )


def test_center_no_active_birds(default_config):
    """update_center() is a no-op when all birds are inactive."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    _step_flock(flock, cfg, 1.0 / 60.0)

    # Deactivate all birds
    flock.active[:] = False
    center_before = flock.center.copy()

    flock.update_center()

    np.testing.assert_array_equal(
        flock.center, center_before,
        err_msg="center must not change when no active birds"
    )
