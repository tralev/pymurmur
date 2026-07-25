"""P4.8 — Ecology unit tests: day advancement rate, roost pull direction, time-window boundary edges, is_roosting_time/is_murmuration_season/roost force.

Split out of test_ecology.py (file-size split).
"""


from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions.ecology import Ecology

# ── S2.B8: Day advancement rate ────────────────────────────────────

def test_ecology_day_advances_at_correct_rate():
    """S2.B8: ecology._day advances at _day_dt * 60.0 * dt per frame.
    Default _day_dt = 1/600 → rate = 0.1 days/second at 60fps."""
    import pytest

    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 5
    cfg.seed = 42

    flock = PhysicsFlock(cfg)
    eco = Ecology(cfg)
    eco._day = 100.0  # known starting day

    dt = 1.0 / 60.0
    ctx = StepContext(
        frame=0, dt=dt, rng=flock.rng,
        center=flock.center, config=cfg,
    )
    day_before = eco._day
    eco.apply(flock, ctx)
    day_after = eco._day

    # Rate: _day_dt * 60.0 * dt = (1/600) * 60 * (1/60) = 1/600 ≈ 0.001667
    expected_delta = eco._day_dt * 60.0 * dt
    actual_delta = day_after - day_before
    assert actual_delta == pytest.approx(expected_delta, rel=1e-6), (
        f"Day advance: expected {expected_delta:.10f}, got {actual_delta:.10f}"
    )

    # After 60 frames at 60fps, should advance ~0.1 days
    for _ in range(59):
        eco.apply(flock, ctx)
    total_delta = eco._day - day_before
    assert total_delta == pytest.approx(0.1, abs=0.001), (
        f"After 60 frames, day should advance ~0.1, got {total_delta:.6f}"
    )


# ── S2.B8: Roost pull direction toward roost ───────────────────────

def test_ecology_roost_pull_direction_toward_roost():
    """S2.B8: Roost pull is toward the ecology_roost position, not just
    downward. Birds placed on all sides of the roost should be pulled
    in the correct direction."""
    import numpy as np

    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 5
    cfg.ecology_roost = (500.0, 350.0, 200.0)
    cfg.ecology_critical_mass = 1  # any flock passes
    cfg.seed = 42

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    # Place birds on different sides of the roost
    roost = np.array(cfg.ecology_roost, dtype=np.float32)
    flock.positions[0] = roost + np.array([50.0, 0.0, 0.0], dtype=np.float32)
    flock.positions[1] = roost + np.array([-50.0, 0.0, 0.0], dtype=np.float32)
    flock.positions[2] = roost + np.array([0.0, 50.0, 0.0], dtype=np.float32)
    flock.positions[3] = roost + np.array([0.0, 0.0, 50.0], dtype=np.float32)
    flock.positions[4] = roost + np.array([0.0, 0.0, -50.0], dtype=np.float32)

    eco = Ecology(cfg)
    # 40 min before dusk — strong enough dusk_factor for measurable force
    # At day 172, dusk=20.25, hour=20.25-40/60=19.583
    eco._day = 172.0 + 19.583 / 24.0
    eco._day_dt = 0

    ctx = StepContext(
        frame=0, dt=1.0 / 60.0, rng=flock.rng,
        center=flock.center, config=cfg,
    )
    eco.apply(flock, ctx)

    forces = flock.accelerations[flock.active]
    # Each bird should be pulled toward roost
    for i in range(5):
        to_roost = roost - flock.positions[i]
        dot = np.dot(forces[i], to_roost)
        assert dot > 0, (
            f"Bird {i}: force should pull toward roost. "
            f"dot={dot:.6f}, to_roost={to_roost}, force={forces[i]}"
        )


# ── S2.B8: Time-window boundary edges ──────────────────────────────

def test_ecology_roost_pull_just_inside_window():
    """S2.B8: At 40 minutes before dusk (firmly inside window), roost
    pull is active (minutes_before_dusk = 40 > 0, dusk_factor > 0)."""
    import numpy as np

    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 600
    cfg.ecology_critical_mass = 500
    cfg.seed = 42

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology(cfg)
    # Day 172: dusk = 20.25. 40 min before = hour 19.583.
    eco._day = 172.0 + 19.583 / 24.0
    eco._day_dt = 0

    ctx = StepContext(
        frame=0, dt=1.0 / 60.0, rng=flock.rng,
        center=flock.center, config=cfg,
    )
    eco.apply(flock, ctx)

    # Inside window → dusk_factor > 0 (sigmoid at z≈3.33 gives ~0.035)
    # With 600 birds → coherence=1.0 → ramp > 0 → roost pull fires
    assert eco.coherence_factor == 1.0, (
        f"600 birds at crit_mass=500: coherence_factor should be 1.0, "
        f"got {eco.coherence_factor}"
    )
    assert np.isfinite(flock.accelerations).all()
    assert not np.allclose(flock.accelerations, 0.0), (
        "Roost pull should fire at 40 min before dusk"
    )


def test_ecology_roost_pull_just_outside_window():
    """S2.B8: At 121 minutes before dusk (just outside window), roost
    pull is NOT active (minutes_before_dusk = 121 > 120)."""
    import numpy as np

    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 600
    cfg.ecology_critical_mass = 500
    cfg.seed = 42

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology(cfg)
    # 121 min before dusk = hour 18.233. _day = 172.76
    eco._day = 172.76
    eco._day_dt = 0

    ctx = StepContext(
        frame=0, dt=1.0 / 60.0, rng=flock.rng,
        center=flock.center, config=cfg,
    )
    eco.apply(flock, ctx)

    # Outside window → coherence_factor reset to 1.0
    assert eco.coherence_factor == 1.0, (
        f"Outside dusk window: coherence_factor should be 1.0, "
        f"got {eco.coherence_factor}"
    )
    # No roost pull applied → accelerations unchanged
    assert np.allclose(flock.accelerations, 0.0), (
        "No roost pull outside the dusk time window"
    )


def test_ecology_roost_pull_after_dusk_30_min():
    """S2.B8: At 29 minutes after dusk (just inside window), roost pull
    is active. At 31 minutes after dusk (just outside), it's not."""
    import numpy as np

    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 600
    cfg.ecology_critical_mass = 500
    cfg.seed = 42

    # Just inside (29 min after dusk)
    flock_inside = PhysicsFlock(cfg)
    flock_inside.accelerations[:] = 0.0
    eco_inside = Ecology(cfg)
    # dusk at day 172 = 20.25. 29 min after = hour 20.733
    eco_inside._day = 172.0 + 20.733 / 24.0
    eco_inside._day_dt = 0
    ctx = StepContext(
        frame=0, dt=1.0 / 60.0, rng=flock_inside.rng,
        center=flock_inside.center, config=cfg,
    )
    eco_inside.apply(flock_inside, ctx)
    # Inside window → coherence_factor is 1.0 (600 birds), roost pull fires
    assert eco_inside.coherence_factor == 1.0
    assert not np.allclose(flock_inside.accelerations, 0.0)

    # Just outside (31 min after dusk)
    flock_outside = PhysicsFlock(cfg)
    flock_outside.accelerations[:] = 0.0
    eco_outside = Ecology(cfg)
    # 31 min after dusk = hour 20.767
    eco_outside._day = 172.0 + 20.767 / 24.0
    eco_outside._day_dt = 0
    ctx2 = StepContext(
        frame=0, dt=1.0 / 60.0, rng=flock_outside.rng,
        center=flock_outside.center, config=cfg,
    )
    eco_outside.apply(flock_outside, ctx2)
    # Outside window → coherence=1.0, no roost pull
    assert eco_outside.coherence_factor == 1.0
    assert np.allclose(flock_outside.accelerations, 0.0)


# ── S2.B8: is_roosting_time / is_murmuration_season / roost force ──

def test_is_roosting_time_before_and_after_dusk():
    """S2.B8: is_roosting_time flips from False to True around dusk."""
    day = 172.0
    dusk = Ecology.dusk_hour(day)
    assert not Ecology.is_roosting_time(dusk - 2.0, day)  # 2h before dusk
    assert Ecology.is_roosting_time(dusk + 1.0, day)       # 1h after dusk


def test_is_murmuration_season_boundaries():
    """S2.B8: Oct 1 (day 274) through Mar 31 (day 90) is murmuration
    season; mid-summer (day 172) is not."""
    assert Ecology.is_murmuration_season(274.0)  # Oct 1
    assert Ecology.is_murmuration_season(1.0)    # Jan 1
    assert Ecology.is_murmuration_season(90.0)   # Mar 31
    assert not Ecology.is_murmuration_season(172.0)  # summer solstice
    assert not Ecology.is_murmuration_season(200.0)


def test_predator_present_deterministic_mode_reproducible():
    """S2.B8: predator_present(day) with no rng is same-day-same-result."""
    for day in (1, 50, 100, 365, 1000):
        r1 = Ecology.predator_present(day)
        r2 = Ecology.predator_present(day)
        assert r1 == r2


def test_predator_present_stochastic_mode_uses_rate():
    """S2.B8: predator_present(day, rng=...) draws at PREDATOR_RATE
    (0.296), not the deterministic hash — frequency should land near
    the rate over many draws."""
    import numpy as np
    rng = np.random.default_rng(123)
    hits = sum(Ecology.predator_present(0, rng=rng) for _ in range(20000))
    freq = hits / 20000
    assert abs(freq - Ecology.PREDATOR_RATE) < 0.02, f"freq={freq:.4f}"


def test_ecology_config_predator_presence_selector():
    """S2.B8: cfg.ecology_predator_presence selects deterministic vs
    stochastic draws for Ecology.apply()'s day-boundary check."""
    cfg = SimConfig()
    assert cfg.ecology_predator_presence == "deterministic"
    cfg.ecology_predator_presence = "stochastic"
    eco = Ecology(cfg)
    assert eco._predator_presence_mode == "stochastic"


def test_ecology_roost_force_is_distance_independent_direction_scaled():
    """S2.B8: roost_force = unit(roost-p)*roost_strength — magnitude is
    the same for a near bird and a far bird (was linear-in-distance)."""
    import numpy as np

    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 2
    cfg.ecology_roost = (500.0, 350.0, 200.0)
    cfg.ecology_critical_mass = 1
    cfg.seed = 42

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    roost = np.array(cfg.ecology_roost, dtype=np.float32)
    # One bird close, one bird 10x farther, same direction
    flock.positions[0] = roost + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    flock.positions[1] = roost + np.array([100.0, 0.0, 0.0], dtype=np.float32)

    eco = Ecology(cfg)
    eco._day = 172.0 + 19.583 / 24.0  # 40 min before dusk
    eco._day_dt = 0
    ctx = StepContext(
        frame=0, dt=1.0 / 60.0, rng=flock.rng,
        center=flock.center, config=cfg,
    )
    eco.apply(flock, ctx)

    mag_near = np.linalg.norm(flock.accelerations[0])
    mag_far = np.linalg.norm(flock.accelerations[1])
    assert mag_near > 0
    assert np.isclose(mag_near, mag_far, rtol=1e-4), (
        f"roost pull should be distance-independent: near={mag_near:.6f} "
        f"far={mag_far:.6f}"
    )
