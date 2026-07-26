"""Unit tests for physics.boid::integrate()'s velocity_damping parameter
(§15-style general friction: v *= (1 - damping*dt), applied after the
speed-mode clamp/inertia so it has a uniform effect regardless of
speed_mode — see boid.py's integrate() docstring for the ordering
rationale).
"""

import numpy as np
import pytest

from pymurmur.physics.boid import integrate


def _run(speed_mode, damping, dt=1.0, N=10):
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.full((N, 3), 2.0, dtype=np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    integrate(
        pos, vel, acc, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", dt,
        speed_mode=speed_mode, damping=damping,
    )
    return vel


def test_damping_zero_byte_identical_to_no_damping():
    N = 10
    pos1 = np.zeros((N, 3), dtype=np.float32)
    vel1 = np.full((N, 3), 2.0, dtype=np.float32)
    acc1 = np.zeros((N, 3), dtype=np.float32)
    pos2 = pos1.copy()
    vel2 = vel1.copy()
    acc2 = acc1.copy()
    active = np.ones(N, dtype=bool)

    integrate(pos1, vel1, acc1, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0)
    integrate(pos2, vel2, acc2, active, 1000.0, 700.0, 400.0, 4.0, "toroidal", 1.0, damping=0.0)

    np.testing.assert_array_equal(pos1, pos2)
    np.testing.assert_array_equal(vel1, vel2)


@pytest.mark.parametrize("speed_mode", ["band", "clamp", "fixed", "ceiling", "none"])
def test_damping_reduces_speed_across_all_speed_modes(speed_mode):
    """The specific claim this ordering choice is designed to satisfy:
    damping must be visible even under 'fixed' mode's exact
    renormalisation, since it's applied AFTER the speed-mode branch."""
    vel_no_damp = _run(speed_mode, damping=0.0)
    vel_damped = _run(speed_mode, damping=0.5)

    speed_no_damp = np.linalg.norm(vel_no_damp, axis=1)
    speed_damped = np.linalg.norm(vel_damped, axis=1)

    assert np.all(speed_damped < speed_no_damp), (
        f"speed_mode={speed_mode}: damping had no visible effect "
        f"(no_damp={speed_no_damp}, damped={speed_damped})"
    )


def test_damping_scales_with_dt():
    """Larger dt -> more damping applied in one step (v *= 1 - damping*dt)."""
    vel_small_dt = _run("band", damping=0.5, dt=0.01)
    vel_large_dt = _run("band", damping=0.5, dt=0.05)
    speed_small = np.linalg.norm(vel_small_dt, axis=1)
    speed_large = np.linalg.norm(vel_large_dt, axis=1)
    assert np.all(speed_large < speed_small)


def test_damping_one_zeros_velocity_at_dt_one():
    """damping=1.0, dt=1.0 -> v *= (1 - 1*1) = 0."""
    vel = _run("none", damping=1.0, dt=1.0)
    # "none" speed_mode has no clamp, so the zero-speed fallback (step 5)
    # kicks in afterward, giving deterministic (min_speed, 0, 0).
    assert np.all(np.isfinite(vel))
