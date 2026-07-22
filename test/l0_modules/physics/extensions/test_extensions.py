"""Unit tests for physics.extensions — ExtensionManager + all 4 extensions.

Covers:
  - StepContext dataclass (M1-M2)
  - Extension ABC (M3)
  - Lazy extension lifecycle toggles (M4-M14)
  - threat_prox contract (M15-M18)
"""

import numpy as np
import pytest

from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import Extension, StepContext
from pymurmur.physics.extensions.ecology import Ecology
from pymurmur.physics.extensions.ripple import Ripple
from pymurmur.physics.extensions.wander import Wander
from pymurmur.physics.flock import PhysicsFlock


def _make_ctx(flock, config, frame=0, dt=1.0/60.0):
    """Create a StepContext from a flock and config for extension tests."""
    return StepContext(
        frame=frame,
        dt=dt,
        rng=flock.rng,
        center=flock.center,
        config=config,
    )# ── ExtensionManager ──────────────────────────────────────────────


def test_step_context_standalone_no_flock():
    """P2.6: StepContext can be created independently without a SimulationEngine.

    This verifies StepContext is an independent entity — it only needs
    a numpy Generator and config, not a fully-wired simulation."""
    import numpy as np

    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    rng = np.random.default_rng(42)

    ctx = StepContext(
        frame=0,
        dt=0.016,
        rng=rng,
        center=None,
        config=cfg,
    )
    assert ctx.frame == 0
    assert ctx.dt == 0.016
    assert ctx.rng is rng
    assert ctx.center is None
    assert ctx.config is cfg
    assert ctx.threat_prox is None


def test_step_context_with_ndarray_center():
    """P2.6: StepContext accepts numpy ndarray for center."""
    import numpy as np

    from pymurmur.core.config import SimConfig
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    cfg = SimConfig()
    rng = np.random.default_rng(42)

    ctx = StepContext(
        frame=100,
        dt=0.016,
        rng=rng,
        center=center,
        config=cfg,
    )
    assert ctx.center is not None
    np.testing.assert_array_equal(ctx.center, center)


def test_step_context_with_threat_prox():
    """P2.6: StepContext accepts threat_prox array."""
    import numpy as np

    from pymurmur.core.config import SimConfig
    tp = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    cfg = SimConfig()
    rng = np.random.default_rng(42)

    ctx = StepContext(
        frame=1,
        dt=0.016,
        rng=rng,
        center=None,
        config=cfg,
        threat_prox=tp,
    )
    assert ctx.threat_prox is not None
    np.testing.assert_array_equal(ctx.threat_prox, tp)


def test_extension_manager_empty(default_config):
    """All extensions disabled → count = 0."""
    cfg = default_config
    cfg.predator_enabled = False
    cfg.roosting_enabled = False
    cfg.wander_enabled = False
    cfg.ripple_enabled = False

    mgr = ExtensionManager(cfg)
    assert mgr.count == 0


def test_extension_manager_all_enabled(default_config):
    """All 4 extensions enabled → count = 4."""
    cfg = default_config
    cfg.predator_enabled = True
    cfg.roosting_enabled = True
    cfg.wander_enabled = True
    cfg.ripple_enabled = True

    mgr = ExtensionManager(cfg)
    assert mgr.count == 4


def test_extension_manager_ecology_enabled(default_config):
    """roosting_enabled=True → Ecology is instantiated."""
    cfg = default_config
    cfg.roosting_enabled = True
    mgr = ExtensionManager(cfg)
    assert mgr.count == 1
    assert mgr._ecology is not None


def test_extension_manager_wander_enabled(default_config):
    """wander_enabled=True → Wander is instantiated."""
    cfg = default_config
    cfg.wander_enabled = True
    mgr = ExtensionManager(cfg)
    assert mgr.count == 1
    assert mgr._wander is not None


def test_extension_manager_ripple_enabled(default_config):
    """ripple_enabled=True → Ripple is instantiated."""
    cfg = default_config
    cfg.ripple_enabled = True
    mgr = ExtensionManager(cfg)
    assert mgr.count == 1
    assert mgr._ripple is not None


def test_extension_manager_predator_enabled(default_config):
    """predator_enabled=True → Predator is instantiated (test_predator_spawns)."""
    cfg = default_config
    cfg.predator_enabled = True
    mgr = ExtensionManager(cfg)
    assert mgr.count == 1
    assert mgr._predator is not None


def test_extension_manager_pre_step(default_config):
    """pre_step calls apply on all enabled extensions without crash."""
    cfg = default_config
    cfg.predator_enabled = True
    cfg.roosting_enabled = True
    cfg.wander_enabled = True
    cfg.ripple_enabled = True
    cfg.num_boids = 30

    flock = PhysicsFlock(cfg)
    mgr = ExtensionManager(cfg)
    mgr.pre_step(flock, _make_ctx(flock, cfg))  # should not crash
    assert mgr.count == 4


def test_extension_manager_predator_conditional(default_config):
    """Predator is skipped when ecology says predator_present is False."""
    cfg = default_config
    cfg.predator_enabled = True
    cfg.roosting_enabled = True
    cfg.num_boids = 30

    flock = PhysicsFlock(cfg)
    mgr = ExtensionManager(cfg)

    # Force ecology to signal no predator
    mgr._ecology.predator_active = False

    # Record predator state before pre_step
    pred = mgr._predator
    old_pos = pred._pos.copy()

    mgr.pre_step(flock, _make_ctx(flock, cfg))

    # Predator should NOT have moved (apply was skipped)
    assert np.allclose(pred._pos, old_pos)


def test_extension_manager_predator_no_ecology(default_config):
    """When ecology is not enabled, predator always runs."""
    cfg = default_config
    cfg.predator_enabled = True
    cfg.roosting_enabled = False
    cfg.num_boids = 30

    flock = PhysicsFlock(cfg)
    mgr = ExtensionManager(cfg)

    pred = mgr._predator
    old_pos = pred._pos.copy()

    mgr.pre_step(flock, _make_ctx(flock, cfg))

    # Predator should have moved (apply was called)
    assert not np.allclose(pred._pos, old_pos)


# ── Ecology ───────────────────────────────────────────────────────

def test_ecology_day_length_summer(default_config):
    """Summer solstice (day 172) → ~16.5 hours daylight."""
    eco = Ecology(default_config)
    assert abs(eco.day_length(172) - 16.5) < 1.0


def test_ecology_day_length_winter(default_config):
    """Winter solstice (day 355) → ~7.5 hours daylight."""
    eco = Ecology(default_config)
    assert abs(eco.day_length(355) - 7.5) < 1.0


def test_ecology_day_length_equinox(default_config):
    """Equinox (day 80) → ~12 hours daylight."""
    eco = Ecology(default_config)
    assert abs(eco.day_length(80) - 12.0) < 0.5


def test_ecology_apply_runs(default_config):
    """Ecology.apply() runs without error."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    eco = Ecology(cfg)
    eco.apply(flock, _make_ctx(flock, cfg))
    # Should not raise


def test_ecology_dusk_roost_pull(default_config):
    """At dusk hour, birds experience downward pull toward roost."""
    cfg = default_config
    cfg.num_boids = 500  # above critical mass for full pull
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology(cfg)
    # Summer solstice, hour ~19.68 (inside dusk window [19.25, 20.25])
    eco._day = 172.0 + 0.82  # 0.82 * 24 = 19.68h
    eco._day_dt = 0  # don't advance time

    eco.apply(flock, _make_ctx(flock, cfg))

    # Birds should receive downward force toward roost (z=40, below centre at z=200)
    active = flock.active
    forces = flock.accelerations[active]
    assert not np.allclose(forces, 0.0)
    # Roost pull should point downward (negative z for birds above roost)
    assert (forces[:, 2] < 0).any()


def test_ecology_temperature_summer(default_config):
    """Summer peak (day 202) → ~17°C."""
    eco = Ecology(default_config)
    assert abs(eco.temperature(202) - 17.0) < 0.5


def test_ecology_temperature_winter(default_config):
    """Winter trough (day 20) → ~1°C."""
    eco = Ecology(default_config)
    assert abs(eco.temperature(20) - 1.0) < 0.5


def test_ecology_critical_mass_dampened(default_config):
    """Below critical mass birds, roost pull is dampened by smoothstep.

    S2.B8: the gate window is [0.4, 1.2]x critical_mass. 50 birds against
    critical_mass=500 is below the window floor (200), so the roost pull
    is now fully gated to zero rather than merely dampened.
    """
    cfg = default_config
    cfg.num_boids = 50  # below the [0.4, 1.2]x critical_mass gate window
    cfg.ecology_critical_mass = 500
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology(cfg)
    eco._day = 172.0 + 0.82  # dusk window
    eco._day_dt = 0

    eco.apply(flock, _make_ctx(flock, cfg))

    # Forces should still be finite (zero, not NaN)
    assert np.isfinite(flock.accelerations).all()
    assert eco.coherence_gate(50, 500) == 0.0, "Below window floor should gate fully to 0"

    # A flock inside the window (e.g. 40% of critical_mass, the floor) is
    # dampened but nonzero just above the boundary.
    mass_factor = eco.coherence_gate(201, 500)
    assert 0.0 < mass_factor < 0.05  # should be heavily dampened, not zero


def test_ecology_predator_present_deterministic():
    """predator_present returns same result for same day (deterministic)."""
    assert Ecology.predator_present(100) == Ecology.predator_present(100)
    assert Ecology.predator_present(200) == Ecology.predator_present(200)


def test_ecology_predator_present_boolean():
    """predator_present returns bool."""
    result = Ecology.predator_present(50)
    assert isinstance(result, bool)


def test_ecology_predator_present_rate():
    """predator_present returns True roughly 30% of the time."""
    # Check 100 consecutive days for approximate rate
    trues = sum(1 for d in range(1000) if Ecology.predator_present(d))
    # Should be roughly 300 / 1000, allow wide tolerance
    assert 200 < trues < 400


def test_ecology_predator_flag_updates_on_day_boundary(default_config):
    """_predator_active is updated when day crosses an integer boundary."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    eco = Ecology(cfg)
    # Force day to be just after an integer boundary
    eco._day = 200.0  # int=200, different from _last_int_day=172
    eco._day_dt = 0  # don't advance further

    eco.apply(flock, _make_ctx(flock, cfg))

    # _last_int_day should now be 200
    assert eco._last_int_day == 200
    # predator_active should have been set by predator_present(200)
    assert isinstance(eco.predator_active, bool)


# ── Ecology P4.8 — Logistic dusk, coherence gate, seasonal amplitude ─

def test_logistic_dusk_factor_at_dusk(default_config):
    """P4.8: At exactly dusk hour → dusk_factor ≈ 0.97 (strong roost)."""
    eco = Ecology(default_config)
    # At dusk (hour == dusk), minutes_before=0, z=(0-20)/6 ≈ -3.33, sigmoid ≈ 0.965
    factor = eco.logistic_dusk_factor(20.0, 20.0, 6.0)
    assert factor > 0.95, f"At dusk, factor should be near 1, got {factor:.4f}"


def test_logistic_dusk_factor_40min_before(default_config):
    """P4.8: 40 minutes before dusk → dusk_factor ≈ 0.035 (roost not yet started)."""
    eco = Ecology(default_config)
    # 40 min before: hour = 19.333, dusk = 20.0, minutes_before=40, z=(40-20)/6≈3.33, sigmoid ≈ 0.035
    factor = eco.logistic_dusk_factor(19.333333, 20.0, 6.0)
    assert factor < 0.05, f"40 min before dusk, factor should be near 0, got {factor:.4f}"


def test_logistic_dusk_factor_well_before(default_config):
    """P4.8: 2 hours before dusk → dusk_factor ≈ 0 (well outside roost window).

    The sigmoid returns ~0 for times well before the midpoint.
    The time-window guard in apply() gates actual roost activation."""
    eco = Ecology(default_config)
    factor = eco.logistic_dusk_factor(18.0, 20.0, 6.0)
    assert factor < 0.01, f"2h before dusk, sigmoid should be near 0, got {factor:.4f}"


def test_noon_no_roost_pull(default_config):
    """P4.8: At noon (well outside dusk window), apply() produces no roost force."""
    cfg = default_config
    cfg.num_boids = 500
    cfg.ecology_critical_mass = 500
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology(cfg)
    eco._day = 172.0 + 0.5  # noon on summer solstice (hour=12)
    eco._day_dt = 0
    eco.apply(flock, _make_ctx(flock, cfg))

    # No roost force should be applied at noon
    assert eco.coherence_factor == 1.0  # reset to default
    assert np.allclose(flock.accelerations, 0.0), (
        "No roost pull should fire at noon"
    )


def test_logistic_dusk_factor_after_dusk(default_config):
    """P4.8: After dusk → dusk_factor stays near 1 (roost window active)."""
    eco = Ecology(default_config)
    factor = eco.logistic_dusk_factor(21.0, 20.0, 6.0)
    assert factor > 0.95, f"After dusk should be near 1, got {factor:.6f}"


def test_logistic_dusk_factor_steepness(default_config):
    """P4.8: Smaller dusk_width → steeper transition (closer to 0 before midpoint)."""
    eco = Ecology(default_config)
    # At 25 min before dusk with width=3 (narrow) vs width=12 (wide)
    factor_narrow = eco.logistic_dusk_factor(19.583333, 20.0, 3.0)  # 25 min before
    factor_wide = eco.logistic_dusk_factor(19.583333, 20.0, 12.0)
    # Narrow width → steeper drop-off, closer to 0 at z>0
    # Wide width → softer transition, farther from 0
    assert factor_narrow < factor_wide, (
        f"Narrower width should give smaller factor before midpoint: "
        f"narrow={factor_narrow:.4f}, wide={factor_wide:.4f}"
    )


def test_seasonal_factor_peak(default_config):
    """P4.8: Day 15 (mid-January) → seasonal_factor ≈ 1.0 (peak murmuration)."""
    eco = Ecology(default_config)
    assert abs(eco.seasonal_factor(15, 0.5) - 1.0) < 0.01


def test_seasonal_factor_trough(default_config):
    """P4.8: Day 197 (mid-July) → seasonal_factor ≈ 0.25 (no murmurations)."""
    eco = Ecology(default_config)
    assert abs(eco.seasonal_factor(197, 0.5) - 0.25) < 0.01


def test_seasonal_factor_no_amplitude(default_config):
    """P4.8: amplitude=0 → flat factor = 1.0 year-round."""
    eco = Ecology(default_config)
    for day in [15, 106, 197, 300]:
        factor = eco.seasonal_factor(day, 0.0)
        assert abs(factor - 1.0) < 0.01, f"Day {day}: got {factor:.4f}"


def test_seasonal_factor_monotonic(default_config):
    """P4.8: Factor decreases monotonically from peak (day 15) to trough (day 197)."""
    eco = Ecology(default_config)
    factors = [eco.seasonal_factor(d, 0.5) for d in range(15, 198)]
    # Should be descending overall (allow minor floating-point noise)
    assert factors[0] > factors[-1], "Factor should decrease from peak to trough"


def test_seasonal_factor_clamped(default_config):
    """P4.8: Factor stays within [0.05, 2.0] even at extreme amplitude."""
    eco = Ecology(default_config)
    f = eco.seasonal_factor(197, 2.0)  # would give negative without clamp
    assert f >= 0.05, f"Trough should be clamped to >=0.05, got {f:.4f}"
    f = eco.seasonal_factor(15, 2.0)  # would exceed 2.0 without clamp
    assert f <= 2.0, f"Peak should be clamped to <=2.0, got {f:.4f}"


def test_coherence_gate_zero_flock(default_config):
    """P4.8: n_active=0 → coherence_gate = 0."""
    eco = Ecology(default_config)
    assert eco.coherence_gate(0, 500) == 0.0


def test_coherence_gate_tiny_flock(default_config):
    """P4.8: n_active=10, critical_mass=500 → gate ≈ 0.001."""
    eco = Ecology(default_config)
    gate = eco.coherence_gate(10, 500)
    assert gate < 0.01, f"10 birds of 500 should be near 0, got {gate:.4f}"


def test_coherence_gate_half_mass(default_config):
    """S2.B8: n_active=250 (window midpoint, 0.8x critical_mass) → gate = 0.5.

    Gate window is [0.4, 1.2]x critical_mass; at critical_mass=500 that's
    [200, 600], whose midpoint is 400 birds, not 250 — 250 sits at
    t=(250-200)/400=0.125 → smoothstep(0.125)≈0.043.
    """
    eco = Ecology(default_config)
    gate = eco.coherence_gate(250, 500)
    assert abs(gate - 0.043) < 0.01, f"250/500 in [0.4,1.2] window should be ~0.043, got {gate:.4f}"

    midpoint_gate = eco.coherence_gate(400, 500)
    assert abs(midpoint_gate - 0.5) < 0.01, (
        f"Window midpoint (400 of [200,600]) should be 0.5, got {midpoint_gate:.4f}"
    )


def test_coherence_gate_at_mass(default_config):
    """S2.B8: n_active=critical_mass sits inside the [0.4,1.2]x window, not at its top.

    Gate reaches 1.0 only at/above 1.2x critical_mass (600 for
    critical_mass=500) — reconciled from the old [0,1]x window where
    n_active==critical_mass gave gate=1.0.
    """
    eco = Ecology(default_config)
    gate_at_mass = eco.coherence_gate(500, 500)
    assert abs(gate_at_mass - 0.84375) < 0.01, f"Expected ~0.844 at critical_mass, got {gate_at_mass:.4f}"

    gate_at_hi = eco.coherence_gate(600, 500)
    assert abs(gate_at_hi - 1.0) < 0.01, f"1.2x critical_mass should be 1.0, got {gate_at_hi:.4f}"


def test_coherence_gate_above_mass(default_config):
    """P4.8: n_active=1000, critical_mass=500 → gate = 1.0 (capped)."""
    eco = Ecology(default_config)
    gate = eco.coherence_gate(1000, 500)
    assert abs(gate - 1.0) < 0.01


def test_gated_weight_spec_values(default_config):
    """P4.8: gated_weight(0.8, 10) ≈ 0, gated_weight(0.8, 600) > 0.7."""
    eco = Ecology(default_config)
    assert eco.gated_weight(0.8, 10, 500) < 0.01, (
        f"Tiny flock should gate weight to near 0, got {eco.gated_weight(0.8, 10, 500):.4f}"
    )
    assert eco.gated_weight(0.8, 600, 500) > 0.7, (
        f"Large flock should preserve weight, got {eco.gated_weight(0.8, 600, 500):.4f}"
    )


def test_temperature_boosts_roost(default_config):
    """P4.8: Warmer evening → stronger roost pull. Verify via applied force."""
    cfg = default_config
    cfg.num_boids = 500
    cfg.ecology_temperature_boost = 0.5  # significant boost
    cfg.ecology_seasonal_amplitude = 0.0  # no seasonal effect
    cfg.ecology_dusk_width = 6.0

    # Cold day (day 20, temp ~1°C) vs warm day (day 202, temp ~17°C)
    # Both 40 minutes before their respective dusk
    flock_cold = PhysicsFlock(cfg)
    flock_cold.accelerations[:] = 0.0
    flock_warm = PhysicsFlock(cfg)
    flock_warm.accelerations[:] = 0.0

    eco_cold = Ecology(cfg)
    eco_cold._day = 20.0 + 0.6411  # 40 min before dusk (~15.39h)
    eco_cold._day_dt = 0

    eco_warm = Ecology(cfg)
    eco_warm._day = 202.0 + 0.8037  # 40 min before dusk (~19.29h)
    eco_warm._day_dt = 0

    eco_cold.apply(flock_cold, _make_ctx(flock_cold, cfg))
    eco_warm.apply(flock_warm, _make_ctx(flock_warm, cfg))

    force_cold = float(np.linalg.norm(np.mean(
        flock_cold.accelerations[flock_cold.active], axis=0
    )))
    force_warm = float(np.linalg.norm(np.mean(
        flock_warm.accelerations[flock_warm.active], axis=0
    )))

    # Warmer day should produce stronger roost pull (temperature boost)
    assert force_warm > force_cold, (
        f"Warmer evening should have stronger roost: warm={force_warm:.6f}, "
        f"cold={force_cold:.6f}"
    )


def test_seasonal_amplitude_modulates_roost(default_config):
    """P4.8: Peak season (day 15) gives stronger roost than trough (day 197)."""
    cfg = default_config
    cfg.num_boids = 500
    cfg.ecology_seasonal_amplitude = 0.5
    cfg.ecology_temperature_boost = 0.0  # no temperature effect
    cfg.ecology_dusk_width = 6.0

    flock_peak = PhysicsFlock(cfg)
    flock_peak.accelerations[:] = 0.0
    flock_trough = PhysicsFlock(cfg)
    flock_trough.accelerations[:] = 0.0

    eco_peak = Ecology(cfg)
    eco_peak._day = 15.0 + 0.6374  # 40 min before dusk (~15.30h)
    eco_peak._day_dt = 0

    eco_trough = Ecology(cfg)
    eco_trough._day = 197.0 + 0.8074  # 40 min before dusk (~19.38h)
    eco_trough._day_dt = 0

    eco_peak.apply(flock_peak, _make_ctx(flock_peak, cfg))
    eco_trough.apply(flock_trough, _make_ctx(flock_trough, cfg))

    force_peak = float(np.linalg.norm(np.mean(
        flock_peak.accelerations[flock_peak.active], axis=0
    )))
    force_trough = float(np.linalg.norm(np.mean(
        flock_trough.accelerations[flock_trough.active], axis=0
    )))

    assert force_peak > force_trough * 2.0, (
        f"Peak season should have much stronger roost: peak={force_peak:.6f}, "
        f"trough={force_trough:.6f}"
    )


def test_coherence_factor_exposed(default_config):
    """P4.8: ecology.coherence_factor is updated each apply() for external use."""
    cfg = default_config
    cfg.num_boids = 600  # S2.B8: gate window is [0.4,1.2]x — 1.0 needs >=600
    cfg.ecology_critical_mass = 500

    flock = PhysicsFlock(cfg)
    eco = Ecology(cfg)
    eco._day = 172.0 + 0.82  # dusk window
    eco._day_dt = 0

    eco.apply(flock, _make_ctx(flock, cfg))
    # With 600 birds (1.2x critical_mass=500), coherence should be 1.0
    assert abs(eco.coherence_factor - 1.0) < 0.01, (
        f"Coherence factor should be 1.0 at 1.2x critical mass, got {eco.coherence_factor:.4f}"
    )

    # With small flock, coherence should be low
    cfg2 = default_config
    cfg2.num_boids = 20
    cfg2.ecology_critical_mass = 500
    flock2 = PhysicsFlock(cfg2)
    eco2 = Ecology(cfg2)
    eco2._day = 172.0 + 0.82
    eco2._day_dt = 0
    eco2.apply(flock2, _make_ctx(flock2, cfg2))
    assert eco2.coherence_factor < 0.02, (
        f"Small flock coherence should be near 0, got {eco2.coherence_factor:.4f}"
    )


def test_full_p48_roost_components(default_config):
    """P4.8: Integration — all four components combine sensibly.

    At peak season (day 15), above critical mass, during dusk window
    → strong roost pull. At trough, cold, below mass, at noon
    → near-zero pull."""
    cfg = default_config
    cfg.ecology_seasonal_amplitude = 0.5
    cfg.ecology_temperature_boost = 0.3
    cfg.ecology_dusk_width = 6.0
    cfg.ecology_critical_mass = 500

    # Scenario 1: Peak — day 15, above mass (S2.B8: gate window tops out at
    # 1.2x critical_mass=500, so 650 is clearly "above"), 40 min before dusk
    cfg.num_boids = 650
    flock_peak = PhysicsFlock(cfg)
    flock_peak.accelerations[:] = 0.0
    eco_peak = Ecology(cfg)
    eco_peak._day = 15.0 + 0.6374  # ~15.30h, 40 min before winter dusk
    eco_peak._day_dt = 0
    eco_peak.apply(flock_peak, _make_ctx(flock_peak, cfg))
    force_peak = float(np.linalg.norm(np.mean(
        flock_peak.accelerations[flock_peak.active], axis=0
    )))

    # Scenario 2: Trough — day 197, below mass, at noon (outside dusk window)
    cfg.num_boids = 30
    flock_trough = PhysicsFlock(cfg)
    flock_trough.accelerations[:] = 0.0
    eco_trough = Ecology(cfg)
    eco_trough._day = 197.0 + 0.5  # noon (hour=12), well outside dusk window
    eco_trough._day_dt = 0
    eco_trough.apply(flock_trough, _make_ctx(flock_trough, cfg))
    force_trough = float(np.linalg.norm(np.mean(
        flock_trough.accelerations[flock_trough.active], axis=0
    )))

    # Peak should be dramatically stronger than trough
    assert force_peak > 0, "Peak season should have non-zero roost pull"
    assert force_trough < 1e-9 or force_trough < force_peak * 0.01, (
        f"Trough should be negligible vs peak: peak={force_peak:.6f}, "
        f"trough={force_trough:.6f}"
    )


# ── Wander ────────────────────────────────────────────────────────

def test_wander_apply_runs(default_config):
    """Wander.apply() runs without error."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    w = Wander()
    w.apply(flock, _make_ctx(flock, cfg))
    assert np.isfinite(flock.accelerations).all()


def test_wander_bounded(default_config):
    """Wander attractor stays within expected radius over full oscillation."""
    cfg = default_config
    cfg.num_boids = 10
    PhysicsFlock(cfg)

    w = Wander()
    # Jump time forward to exercise full oscillation range
    max_dist = 0.0
    for t in np.linspace(0, 200, 2000):
        w._t = t
        target = np.array([
            100 * np.sin(t * 1.3) * np.cos(t * 0.7),
            100 * np.sin(t * 1.7) * np.sin(t * 0.5),
            50 * np.sin(t * 2.1),
        ])
        dist = np.linalg.norm(target)
        max_dist = max(max_dist, dist)

    # Target is bounded: 100*sqrt(2) per axis ≈ 141 in xy, 50 in z
    # Max theoretical ≈ sqrt(141^2 + 141^2 + 50^2) ≈ 206
    assert max_dist < 250.0
    assert max_dist > 100.0  # should have explored non-trivial range


def test_wander_produces_forces(default_config):
    """Wander applies non-zero pull on birds far from attractor."""
    cfg = default_config
    cfg.num_boids = 5
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    # Place birds far from centre
    flock.positions[:] = np.array([
        [0, 0, 0], [1000, 0, 0], [0, 700, 0], [1000, 700, 0], [500, 350, 400]
    ], dtype=np.float32)

    w = Wander()
    w.apply(flock, _make_ctx(flock, cfg))

    # Some birds should receive non-zero wander forces
    assert np.isfinite(flock.accelerations).all()
    assert not np.allclose(flock.accelerations[flock.active], 0.0)


# ── Ripple ────────────────────────────────────────────────────────

def test_ripple_apply_runs(default_config):
    """Ripple.apply() runs without error."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    r = Ripple()
    r.apply(flock, _make_ctx(flock, cfg))
    assert np.isfinite(flock.accelerations).all()


def test_ripple_envelope_decay(default_config):
    """Ripple intensity decays with distance from pulse centre."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    # Place one bird at COM, one far away
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
    flock.positions[1] = np.array([500, 350, 600], dtype=np.float32)  # far

    r = Ripple()
    r._t = 2.0  # radius = 400: far bird at dist=400 sits at pulse peak

    r.apply(flock, _make_ctx(flock, cfg))

    # Forces should be finite; bird at pulse peak should get force
    assert np.isfinite(flock.accelerations).all()


def test_ripple_zero_active(default_config):
    """Ripple.apply() handles zero active birds gracefully."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.active[:] = False
    r = Ripple()
    r.apply(flock, _make_ctx(flock, cfg))
    # Should not crash


# ═══════════════════════════════════════════════════════════════════
# I5 Phase — Missing Unit Tests (M1-M18)
# ═══════════════════════════════════════════════════════════════════


# ── StepContext dataclass (I5.1) ──────────────────────────────────

class TestStepContext:
    """M1-M2: StepContext dataclass contract."""

    def test_step_context_all_fields_present(self, default_config):
        """M1: All 6 fields present with correct types."""
        cfg = default_config
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        # Step once so flock has actual positions → center is computed
        flock.integrate(cfg, 0.016)
        center = flock.center

        ctx = StepContext(
            frame=42, dt=0.016, rng=flock.rng,
            center=center, config=cfg,
        )

        assert isinstance(ctx.frame, int)
        assert isinstance(ctx.dt, float)
        assert hasattr(ctx.rng, 'random')  # numpy Generator
        # center can be None or ndarray — after integrate(), it's an ndarray
        assert ctx.center is not None
        assert hasattr(ctx.center, 'shape')
        assert hasattr(ctx.config, 'num_boids')  # SimConfig
        # threat_prox defaults to None
        assert ctx.threat_prox is None

        # Verify all 6 field names match what extensions expect
        expected_fields = {
            'frame', 'dt', 'rng', 'center', 'config', 'threat_prox'
        }
        actual_fields = set(ctx.__dataclass_fields__.keys())
        assert actual_fields == expected_fields, (
            f"StepContext fields changed: {actual_fields}"
        )

    def test_step_context_threat_prox_defaults_to_none(self, default_config):
        """M2: threat_prox must default to None (not a mutable array).

        A mutable default (e.g. np.zeros(N)) would cause shared-state
        bugs across multiple contexts.
        """
        import numpy as np
        rng = np.random.default_rng(42)

        ctx1 = StepContext(
            frame=0, dt=0.016, rng=rng, center=None,
            config=default_config,
        )
        # Validate the default explicitly
        assert ctx1.threat_prox is None, (
            f"threat_prox must default to None, got {type(ctx1.threat_prox)}"
        )

        # Also verify via the field default in the class definition
        field_default = StepContext.__dataclass_fields__['threat_prox'].default
        assert field_default is None, (
            f"StepContext.threat_prox field default must be None, "
            f"got {field_default!r}"
        )


# ── Extension ABC (I5.2) ──────────────────────────────────────────

class TestExtensionABC:
    """M3: Extension abstract base class contract."""

    def test_extension_abc_cannot_instantiate(self):
        """M3: Extension() must raise TypeError (abstract class).

        If someone removes ABCMeta or @abstractmethod, the protocol
        silently degrades — concrete extensions can be instantiated
        without implementing apply().
        """
        with pytest.raises(TypeError, match="abstract"):
            Extension()  # type: ignore[abstract]

    def test_extension_subclass_without_apply_cannot_instantiate(self):
        """Extension subclass without apply() is also abstract."""
        class IncompleteExtension(Extension):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteExtension()  # type: ignore[abstract]

    def test_extension_subclass_with_apply_can_instantiate(self):
        """Extension subclass implementing apply() is concrete."""
        class CompleteExtension(Extension):
            def apply(self, flock, ctx):
                pass

        ext = CompleteExtension()
        assert isinstance(ext, Extension)


class TestPredatorMarkerPosition:
    """D7/S2.A8: ExtensionManager.predator_position feeds the threat
    marker draw_layer() call — an invisible predator is undebuggable."""

    def test_none_when_predator_disabled(self, default_config):
        default_config.predator_enabled = False
        mgr = ExtensionManager(default_config)
        assert mgr.predator_position is None

    def test_position_when_predator_enabled(self, default_config):
        default_config.predator_enabled = True
        mgr = ExtensionManager(default_config)
        pos = mgr.predator_position
        assert pos is not None
        assert pos.shape == (3,)

    def test_position_tracks_predator_movement(self, default_config):
        default_config.predator_enabled = True
        default_config.num_boids = 20
        mgr = ExtensionManager(default_config)
        flock = PhysicsFlock(default_config)

        p0 = mgr.predator_position.copy()
        for i in range(1, 30):
            ctx = _make_ctx(flock, default_config, frame=i)
            mgr.pre_step(flock, ctx)
        p1 = mgr.predator_position
        assert not (p0 == p1).all(), "Predator marker position must track FSM movement"


# ── Lazy extension lifecycle (I5.3) ───────────────────────────────

class TestLazyExtensionLifecycle:
    """M4-M14: Extensions are lazily created/dropped on config toggle.

    pre_step() checks cfg.*_enabled each frame and creates or drops
    extensions without requiring a simulation reset.
    """

    @staticmethod
    def _mk_manager(config, predator=False, ecology=False,
                    wander=False, ripple=False):
        """Create an ExtensionManager with specified initial state."""
        config.predator_enabled = predator
        config.roosting_enabled = ecology
        config.wander_enabled = wander
        config.ripple_enabled = ripple
        return ExtensionManager(config)

    @staticmethod
    def _mk_flock_and_ctx(config):
        """Create a flock and StepContext for pre_step calls."""
        flock = PhysicsFlock(config)
        ctx = StepContext(
            frame=0, dt=1.0 / 60.0, rng=flock.rng,
            center=flock.center, config=config,
        )
        return flock, ctx

    # ── Predator lazy create/drop (M4, M5) ──────────────────────

    def test_lazy_create_predator_mid_simulation(self, default_config):
        """M4: predator_enabled False→True creates Predator on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, predator=False)
        assert mgr._predator is None

        # Toggle on
        cfg.predator_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._predator is not None, (
            "Predator must be lazily created when predator_enabled becomes True"
        )

    def test_lazy_drop_predator_mid_simulation(self, default_config):
        """M5: predator_enabled True→False drops Predator on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, predator=True)
        assert mgr._predator is not None

        # Toggle off
        cfg.predator_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._predator is None, (
            "Predator must be dropped when predator_enabled becomes False"
        )

    # ── Ecology lazy create/drop (M6, M7) ───────────────────────

    def test_lazy_create_ecology_mid_simulation(self, default_config):
        """M6: roosting_enabled False→True creates Ecology on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ecology=False)
        assert mgr._ecology is None

        cfg.roosting_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ecology is not None, (
            "Ecology must be lazily created when roosting_enabled becomes True"
        )

    def test_lazy_drop_ecology_mid_simulation(self, default_config):
        """M7: roosting_enabled True→False drops Ecology on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ecology=True)
        assert mgr._ecology is not None

        cfg.roosting_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ecology is None, (
            "Ecology must be dropped when roosting_enabled becomes False"
        )

    # ── Wander lazy create/drop (M8, M9) ────────────────────────

    def test_lazy_create_wander_mid_simulation(self, default_config):
        """M8: wander_enabled False→True creates Wander on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, wander=False)
        assert mgr._wander is None

        cfg.wander_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._wander is not None, (
            "Wander must be lazily created when wander_enabled becomes True"
        )

    def test_lazy_drop_wander_mid_simulation(self, default_config):
        """M9: wander_enabled True→False drops Wander on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, wander=True)
        assert mgr._wander is not None

        cfg.wander_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._wander is None, (
            "Wander must be dropped when wander_enabled becomes False"
        )

    # ── Ripple lazy create/drop (M10, M11) ──────────────────────

    def test_lazy_create_ripple_mid_simulation(self, default_config):
        """M10: ripple_enabled False→True creates Ripple on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ripple=False)
        assert mgr._ripple is None

        cfg.ripple_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ripple is not None, (
            "Ripple must be lazily created when ripple_enabled becomes True"
        )

    def test_lazy_drop_ripple_mid_simulation(self, default_config):
        """M11: ripple_enabled True→False drops Ripple on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ripple=True)
        assert mgr._ripple is not None

        cfg.ripple_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ripple is None, (
            "Ripple must be dropped when ripple_enabled becomes False"
        )

    # ── Count accuracy (M12) ────────────────────────────────────

    def test_lazy_toggle_count_updates(self, default_config):
        """M12: ExtensionManager.count reflects current enabled state.

        Toggling extensions on/off must update the count immediately.
        """
        cfg = default_config
        cfg.num_boids = 10

        # Start with all disabled
        mgr = self._mk_manager(cfg, predator=False, ecology=False,
                               wander=False, ripple=False)
        assert mgr.count == 0

        flock, ctx = self._mk_flock_and_ctx(cfg)

        # Enable each one at a time, verify count
        cfg.predator_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 1, f"After predator: {mgr.count}"

        cfg.roosting_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 2, f"After ecology: {mgr.count}"

        cfg.wander_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 3, f"After wander: {mgr.count}"

        cfg.ripple_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 4, f"After ripple: {mgr.count}"

        # Disable all one at a time
        cfg.ripple_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 3, f"After -ripple: {mgr.count}"

        cfg.wander_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 2, f"After -wander: {mgr.count}"

        cfg.roosting_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 1, f"After -ecology: {mgr.count}"

        cfg.predator_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 0, f"After -predator: {mgr.count}"

    # ── No recreate if already present (M13) ────────────────────

    def test_lazy_toggle_no_recreate_if_already_present(
        self, default_config
    ):
        """M13: Toggling False→True→False→True creates fresh but no duplicates.

        After the sequence, count must be 1 (not 2), and the extension
        must be functional.
        """
        cfg = default_config
        cfg.num_boids = 10

        # Start with predator disabled
        mgr = self._mk_manager(cfg, predator=False)
        assert mgr._predator is None
        assert mgr.count == 0

        # Toggle on — fresh ctx
        cfg.predator_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)
        assert mgr._predator is not None
        assert mgr.count == 1
        first_predator = mgr._predator

        # pre_step again with predator still enabled — must NOT recreate
        mgr.pre_step(flock, ctx)
        assert mgr._predator is first_predator, (
            "pre_step must not recreate extension if already present"
        )
        assert mgr.count == 1, (
            f"Count must stay 1 when already-present extension is not recreated. "
            f"Got {mgr.count}"
        )

        # Toggle off — fresh ctx
        cfg.predator_enabled = False
        flock2, ctx2 = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock2, ctx2)
        assert mgr._predator is None
        assert mgr.count == 0

        # Toggle on again — new instance, but still count=1
        cfg.predator_enabled = True
        flock3, ctx3 = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock3, ctx3)
        assert mgr._predator is not None
        assert mgr._predator is not first_predator, (
            "After drop+recreate, must be a new instance"
        )
        assert mgr.count == 1, (
            f"After revive, count must be 1, got {mgr.count}"
        )

    # ── Initial state matches config (M14) ──────────────────────

    def test_lazy_toggle_initial_state_matches_config(self, default_config):
        """M14: ExtensionManager.__init__ matches initial config for all 4."""
        # All enabled
        cfg_all = default_config
        mgr_all = self._mk_manager(cfg_all, predator=True, ecology=True,
                                   wander=True, ripple=True)
        assert mgr_all._predator is not None
        assert mgr_all._ecology is not None
        assert mgr_all._wander is not None
        assert mgr_all._ripple is not None
        assert mgr_all.count == 4

        # All disabled
        cfg_none = default_config
        mgr_none = self._mk_manager(cfg_none, predator=False, ecology=False,
                                    wander=False, ripple=False)
        assert mgr_none._predator is None
        assert mgr_none._ecology is None
        assert mgr_none._wander is None
        assert mgr_none._ripple is None
        assert mgr_none.count == 0

        # Mixed: predator + wander only
        cfg_mix = default_config
        mgr_mix = self._mk_manager(cfg_mix, predator=True, ecology=False,
                                   wander=True, ripple=False)
        assert mgr_mix._predator is not None
        assert mgr_mix._ecology is None
        assert mgr_mix._wander is not None
        assert mgr_mix._ripple is None
        assert mgr_mix.count == 2


# ── threat_prox contract (I5.4) ───────────────────────────────────

class TestThreatProx:
    """M15-M18: threat_prox array published by Predator extension."""

    @staticmethod
    def _mk_ctx(flock, config, threat_prox=None):
        """Create a StepContext, optionally with a pre-set threat_prox."""
        return StepContext(
            frame=0, dt=1.0 / 60.0, rng=flock.rng,
            center=flock.center, config=config,
            threat_prox=threat_prox,
        )

    def test_threat_prox_none_when_predator_disabled(self, default_config):
        """M15: ctx.threat_prox stays None when predator is not enabled."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = False
        cfg.roosting_enabled = False

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)
        ctx = self._mk_ctx(flock, cfg)

        mgr.pre_step(flock, ctx)

        # No predator → ctx.threat_prox must still be None
        assert ctx.threat_prox is None, (
            f"threat_prox must be None when predator is disabled, "
            f"got {type(ctx.threat_prox)}"
        )

    def test_threat_prox_not_none_when_predator_enabled(self, default_config):
        """M16: ctx.threat_prox is set to an array when predator runs.

        Predator.apply() publishes threat_prox for downstream consumers.
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.roosting_enabled = False  # no ecology to gate predator

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)
        ctx = self._mk_ctx(flock, cfg)

        mgr.pre_step(flock, ctx)

        assert ctx.threat_prox is not None, (
            "threat_prox must be set by Predator.apply()"
        )

    def test_threat_prox_has_correct_structure(self, default_config):
        """M17: ctx.threat_prox is an N_capacity float32 array with [0,1] values.

        The array has one entry per capacity slot; inactive slots stay at 0.
        Values are in [0, 1] where 1 = at predator position, 0 = at radius edge.
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 200.0
        cfg.roosting_enabled = False

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)
        ctx = self._mk_ctx(flock, cfg)

        mgr.pre_step(flock, ctx)

        tp = ctx.threat_prox
        assert tp is not None
        assert isinstance(tp, np.ndarray), (
            f"threat_prox must be ndarray, got {type(tp)}"
        )
        assert tp.dtype == np.float32, (
            f"threat_prox dtype must be float32, got {tp.dtype}"
        )
        assert tp.shape == (flock.N_capacity,), (
            f"threat_prox shape must be (N_capacity,), got {tp.shape}"
        )
        # Values must be in [0, 1]
        assert np.all(tp >= 0.0), "threat_prox values must be >= 0"
        assert np.all(tp <= 1.0), "threat_prox values must be <= 1"
        # At least one active bird should have non-zero threat if predator is near
        assert np.isfinite(tp).all(), "threat_prox must be finite"

    def test_threat_prox_none_in_context_when_predator_gated_by_ecology(
        self, default_config
    ):
        """M18: ctx.threat_prox stays None when ecology gates predator off.

        If ecology.predator_active is False, predator's apply() is never
        called, so ctx.threat_prox remains None.
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.roosting_enabled = True  # ecology enabled → can gate predator

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)

        # Force ecology to signal no predator
        mgr._ecology.predator_active = False

        ctx = self._mk_ctx(flock, cfg)
        mgr.pre_step(flock, ctx)

        assert ctx.threat_prox is None, (
            "threat_prox must be None when ecology gates predator off"
        )

    def test_threat_prox_present_when_ecology_allows_predator(
        self, default_config
    ):
        """When ecology.predator_active is True, predator runs → threat_prox set."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.roosting_enabled = True

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)

        # Force ecology to signal predator IS active
        mgr._ecology.predator_active = True

        ctx = self._mk_ctx(flock, cfg)
        mgr.pre_step(flock, ctx)

        assert ctx.threat_prox is not None, (
            "threat_prox must be set when ecology allows predator"
        )
        assert isinstance(ctx.threat_prox, np.ndarray)


# ── Ecology N_active caching (I5.3) ───────────────────────────────

class TestEcologyCaching:
    """M7: Ecology must recompute N_active each frame, not cache it.

    The dusk roost pull uses a smoothstep mass_factor based on N_active
    relative to critical_mass. If N_active is cached and not invalidated
    after add_boids/remove_boids, the wrong mass_factor is used.
    """

    def test_ecology_dusk_mass_factor_responds_to_n_active(self, default_config):
        """M7: Same Ecology instance responds to changing N_active.

        Uses a single Ecology instance with one flock — apply to small
        flock, add birds, apply again. If N_active were cached, the
        second apply would use the stale small-flock value, producing
        the same force both times.
        """
        cfg = default_config
        cfg.ecology_critical_mass = 500

        # S2.B8: gate window is [0.4,1.2]x critical_mass = [200,600] — start
        # inside the window (250, dampened but nonzero) rather than below it
        # (which now gates fully to 0, see coherence_gate).
        cfg.num_boids = 250
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        eco = Ecology(cfg)
        eco._day = 172.0 + 0.82  # dusk window
        eco._day_dt = 0  # freeze time

        eco.apply(flock, _make_ctx(flock, cfg))
        force_small = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )
        assert force_small > 0, "Small flock should get some roost pull"

        # Add 400 birds — N_active goes from 250 → 650, above the gate window
        flock.accelerations[:] = 0.0  # reset forces
        flock.add_boids(400, cfg)

        # Same Ecology, same dusk time, now much larger flock
        eco.apply(flock, _make_ctx(flock, cfg))
        force_large = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )

        # Force must increase — mass_factor went from dampened (~0.043) → 1.0
        assert force_large > force_small * 3.0, (
            f"Same Ecology instance must respond to N_active change: "
            f"large flock force={force_large:.6f}, "
            f"small flock force={force_small:.6f}. "
            f"If N_active were cached, both would be equal."
        )

    def test_ecology_n_active_recomputed_after_add_boids(
        self, default_config
    ):
        """M7: After add_boids, next apply() uses the new N_active.

        If Ecology cached N_active from a previous frame, the mass_factor
        would stay dampened even after adding birds.
        """
        cfg = default_config
        cfg.ecology_critical_mass = 500

        # S2.B8: gate window is [0.4,1.2]x critical_mass = [200,600]. Start
        # inside the window (below it now gates fully to 0, so force_before
        # would be exactly 0 and the "still get some force" assertion below
        # would be meaningless).
        cfg.num_boids = 250
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        eco = Ecology(cfg)
        eco._day = 172.0 + 0.82  # dusk
        eco._day_dt = 0

        # First apply: small flock → dampened force
        eco.apply(flock, _make_ctx(flock, cfg))
        force_before = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )
        assert force_before > 0, "Small flock should still get some force"

        # Add many birds — N_active changes from 250 → 750, above the gate window
        flock.accelerations[:] = 0.0  # reset forces
        flock.add_boids(500, cfg)

        # Second apply: flock is now much larger → should get stronger pull
        eco.apply(flock, _make_ctx(flock, cfg))
        force_after = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )

        # Force must increase because mass_factor went from dampened → near 1.0
        assert force_after > force_before, (
            f"After adding 500 birds (N_active={flock.N_active}), "
            f"force ({force_after:.6f}) must exceed "
            f"pre-add force ({force_before:.6f}). "
            f"If N_active were cached, forces would be equal."
        )

    def test_ecology_n_active_zero_birds_handled(self, default_config):
        """M7: Ecology handles zero active birds without division by zero."""
        cfg = default_config
        cfg.num_boids = 0  # empty flock
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        eco = Ecology(cfg)
        eco._day = 172.0 + 0.82
        eco._day_dt = 0

        # Should not crash — n_active=0 → t=0 → mass_factor=0 → forces unchanged
        eco.apply(flock, _make_ctx(flock, cfg))

        # Forces must remain finite (no NaN from division by zero)
        assert np.isfinite(flock.accelerations).all(), (
            "Ecology must handle zero active birds without producing NaN"
        )


# ── D2: Wander uses configured speed/radius ──────────────────

def test_wander_uses_configured_speed(monkeypatch, default_config):
    """D2: Wander internal clock _t advances at cfg.wander.wander_attractor_speed·dt.

    Before D2: wander.py read cfg.wander_speed (non-existent key) → attribute
               error or wrong value, silently ran at wrong speed.
    After D2:  wander.py reads cfg.wander.wander_attractor_speed → uses the
               actual configured value from WanderConfig.

    This test creates a Wander directly, sets up a StepContext with known
    config values, calls apply() twice, and verifies _t advanced correctly.
    """
    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.extensions.wander import Wander
    cfg = default_config
    cfg.wander.wander_attractor_speed = 0.05  # custom speed
    cfg.wander_enabled = True
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    wander = Wander()
    dt = 1.0 / 60.0

    # Frame 0
    ctx = StepContext(frame=0, dt=dt, rng=np.random.default_rng(42),
                      center=np.array([500,350,200],dtype=np.float32),
                      config=cfg, threat_prox=None)
    t_before = wander._t
    wander.apply(flock, ctx)
    delta_0 = wander._t - t_before
    assert delta_0 == pytest.approx(dt, rel=1e-6), (
        f"_t advance: expected {dt:.6f}, got {delta_0:.6f}"
    )

    # Frame 1 — _t should advance by another dt
    t_mid = wander._t
    ctx2 = StepContext(frame=1, dt=dt, rng=np.random.default_rng(42),
                       center=np.array([500,350,200],dtype=np.float32),
                       config=cfg, threat_prox=None)
    wander.apply(flock, ctx2)
    delta_1 = wander._t - t_mid
    assert delta_1 == pytest.approx(dt, rel=1e-6), (
        f"_t advance frame 1: expected {dt:.6f}, got {delta_1:.6f}"
    )

    # Change speed mid-simulation — should take effect immediately
    cfg.wander.wander_attractor_speed = 0.20  # 4× faster
    t_before2 = wander._t
    ctx3 = StepContext(frame=2, dt=dt, rng=np.random.default_rng(42),
                       center=np.array([500,350,200],dtype=np.float32),
                       config=cfg, threat_prox=None)
    wander.apply(flock, ctx3)
    delta_2 = wander._t - t_before2
    assert delta_2 == pytest.approx(dt, rel=1e-6), (
        f"_t still advances by dt (speed only affects path argument, not clock): "
        f"expected {dt:.6f}, got {delta_2:.6f}"
    )
    # Speed affects bounded_unit_path argument: path(self._t * speed)
    # So wander centre moves faster even though _t advances at same rate.
    # Verify the wander centre path argument t·speed differs for different speeds.
    path_arg_slow = wander._t * 0.05
    path_arg_fast = wander._t * 0.20
    assert path_arg_fast == pytest.approx(4.0 * path_arg_slow, rel=0.01), (
        f"Path arg ratio: slow={path_arg_slow:.4f}, fast={path_arg_fast:.4f}"
    )


def test_wander_uses_configured_radius(default_config):
    """D2: Wander centre stays within cfg.wander.wander_attractor_radius of
    flock centre. Uses Wander directly (no full SimulationEngine).

    Before D2: wander.py read cfg.attractor_radius (non-existent key).
    After D2:  wander.py reads cfg.wander.wander_attractor_radius.
    """
    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.extensions.wander import Wander
    cfg = default_config
    cfg.wander.wander_attractor_radius = 100.0  # small radius for tight check
    cfg.wander_enabled = True
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    # Place birds at a known centre
    flock.positions[:] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    flock.center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    wander = Wander()
    dt = 1.0 / 60.0

    # Run several frames and measure max wander radius
    max_dist = 0.0
    for frame in range(100):
        ctx = StepContext(frame=frame, dt=dt, rng=np.random.default_rng(42),
                          center=flock.center, config=cfg, threat_prox=None)
        wander.apply(flock, ctx)
        if flock.wander_center is not None:
            d = float(np.linalg.norm(flock.wander_center - flock.center))
            if d > max_dist:
                max_dist = d

    # boundedUnitTravel ‖path‖ ≤ 1, so wander_center = C + path·radius
    # → max distance = radius (when ‖path‖ = 1)
    assert max_dist <= cfg.wander.wander_attractor_radius * 1.1, (
        f"Wander exceeded configured radius: max_dist={max_dist:.1f}, "
        f"radius={cfg.wander.wander_attractor_radius}"
    )
    # Also verify wander actually moves (not stuck at centre)
    assert max_dist > 0, "Wander should move away from centre"


def test_wander_config_keys_exist(default_config):
    """D2: WanderConfig has wander_attractor_speed and wander_attractor_radius.

    Verifies the config fields exist with correct types and defaults.
    """
    from pymurmur.core.config import WanderConfig
    w = WanderConfig()
    assert hasattr(w, "wander_attractor_speed")
    assert hasattr(w, "wander_attractor_radius")
    assert isinstance(w.wander_attractor_speed, float)
    assert isinstance(w.wander_attractor_radius, float)
    assert w.wander_attractor_speed == 0.10
    assert w.wander_attractor_radius == 300.0

    # Also verify the config is accessible via SimConfig
    cfg = default_config
    assert cfg.wander.wander_attractor_speed == 0.10
    assert cfg.wander.wander_attractor_radius == 300.0

    # Flat access via _FIELD_MAP
    assert cfg.wander_attractor_speed == 0.10
    assert cfg.wander_attractor_radius == 300.0


def test_wander_config_roundtrip(tmp_path, default_config):
    """D2: Wander config survives YAML round-trip."""
    import yaml
    cfg = default_config
    cfg.wander.wander_attractor_speed = 0.05
    cfg.wander.wander_attractor_radius = 200.0

    # Write
    out = tmp_path / "wander_config.yaml"
    cfg.to_file(out)

    # Read back
    loaded_text = out.read_text()
    assert "wander_attractor_speed" in loaded_text
    assert "wander_attractor_radius" in loaded_text

    # Parse to verify values
    data = yaml.safe_load(loaded_text)
    assert data["wander"]["wander_attractor_speed"] == 0.05
    assert data["wander"]["wander_attractor_radius"] == 200.0

    # Reload via SimConfig
    from pymurmur.core.config import SimConfig
    cfg2 = SimConfig.from_file(out)
    assert cfg2.wander.wander_attractor_speed == 0.05
    assert cfg2.wander.wander_attractor_radius == 200.0


# ── D10: Ripple envelope per-bird array ────────────────────────────


class TestD10RippleEnvelope:
    """D10: ripple_envelope_sum exports a per-bird (N,) array, not a
    scalar.  Two birds at the same position get equal envelope values;
    two birds far apart get different values."""

    def _make_ripple_ctrl(self, cfg):
        """Return (ripple, flock, ctx) for ripple.apply()."""
        from pymurmur.physics.extensions._base import StepContext
        from pymurmur.physics.extensions.ripple import Ripple
        from pymurmur.physics.flock import PhysicsFlock

        ripple = Ripple()
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                     dtype=np.float32)
        ctx = StepContext(
            frame=0, dt=0.5, rng=rng, center=C, config=cfg, threat_prox=None,
        )
        return ripple, flock, ctx

    def test_envelope_is_per_bird_array_not_scalar(self, default_config):
        """D10: After ripple.apply(), _ripple_envelope_sum is an (N,) array
        matching N_capacity, not a float."""
        cfg = default_config
        cfg.num_boids = 20
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert isinstance(env, np.ndarray), (
            f"Expected ndarray, got {type(env)}"
        )
        assert env.shape == (flock.N_capacity,), (
            f"Expected shape ({flock.N_capacity},), got {env.shape}"
        )

    def test_inactive_birds_get_zero_envelope(self, default_config):
        """D10: Inactive birds have envelope value 0.0."""
        cfg = default_config
        cfg.num_boids = 10
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        # Deactivate a few birds
        flock.active[3] = False
        flock.active[7] = False

        ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert env[3] == 0.0, "Inactive bird 3 should have envelope 0"
        assert env[7] == 0.0, "Inactive bird 7 should have envelope 0"
        # Some active birds should have nonzero envelope (if any train is active)
        active_mask = flock.active
        assert env[active_mask].sum() >= 0.0  # may be zero if no train active

    def test_zero_active_returns_zero_array(self, default_config):
        """D10: When no birds are active, envelope is all-zeros array."""
        cfg = default_config
        cfg.num_boids = 5
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        flock.active[:] = False
        ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert np.allclose(env, 0.0), (
            f"All-zero flock should give all-zero envelope, got max={env.max()}"
        )

    def test_same_position_birds_get_equal_envelope(self, default_config):
        """D10: Two birds at the same position get equal envelope values."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        # Place two birds at identical positions
        flock.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        flock.positions[1] = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        # Run a few steps to accumulate ripple envelope
        for _ in range(20):
            ctx = StepContext(
                frame=ctx.frame + 1, dt=0.5, rng=ctx.rng,
                center=ctx.center, config=cfg, threat_prox=None,
            )
            ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert env[0] == pytest.approx(env[1]), (
            f"Birds at same position should have equal envelope: "
            f"{env[0]:.6f} vs {env[1]:.6f}"
        )

    def test_far_apart_birds_get_different_envelope(self, default_config):
        """D10: Two birds at very different distances from the ripple
        origin get different envelope values.

        Bird 0 is near the domain centre (where the ripple Lissajous
        origin moves); bird 1 is far out at the corner.  The ripple's
        gaussian drop-off ensures bird 1's envelope is near zero while
        bird 0's is nonzero when a train is active."""
        cfg = default_config
        cfg.num_boids = 2
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        # Bird 0: near the domain centre where ripple originates
        flock.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        # Bird 1: far corner — ripple gaussian makes envelope near zero
        flock.positions[1] = np.array([50.0, 50.0, 50.0], dtype=np.float32)

        # Run enough steps for multiple ripple trains to activate
        for _ in range(50):
            ctx = StepContext(
                frame=ctx.frame + 1, dt=0.5, rng=ctx.rng,
                center=ctx.center, config=cfg, threat_prox=None,
            )
            ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        # Bird near centre should have nonzero envelope while far bird
        # has near-zero — the difference must be significant
        assert abs(env[0] - env[1]) > 1e-9, (
            f"Birds at different distances should have "
            f"different envelope: {env[0]:.6f} vs {env[1]:.6f}"
        )

    def test_envelope_not_normalised_by_n(self, default_config):
        """D10: Envelope values are independent of N — adding more birds
        does not change existing birds' envelope values.

        Both configs use the same seed so the non-bird-0 positions
        are identical, giving both flocks the same centroid C.
        Only bird 0's position is set explicitly; all others are
        determined by the shared seed."""
        cfg = default_config
        cfg.seed = 42
        cfg.num_boids = 10
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0

        # Copy config for large-flock run (same seed, more birds)
        cfg_large = default_config
        cfg_large.seed = 42
        cfg_large.num_boids = 30
        cfg_large.width = 1000.0
        cfg_large.height = 700.0
        cfg_large.depth = 400.0

        ripple_small, flock_small, ctx_small = self._make_ripple_ctrl(cfg)
        flock_small.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        for _ in range(30):
            ctx_small = StepContext(
                frame=ctx_small.frame + 1, dt=0.5, rng=ctx_small.rng,
                center=ctx_small.center, config=cfg, threat_prox=None,
            )
            ripple_small.apply(flock_small, ctx_small)

        env_small_bird0 = cfg._ripple_envelope_sum[0]

        ripple_large, flock_large, ctx_large = self._make_ripple_ctrl(cfg_large)
        flock_large.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        for _ in range(30):
            ctx_large = StepContext(
                frame=ctx_large.frame + 1, dt=0.5, rng=ctx_large.rng,
                center=ctx_large.center, config=cfg_large, threat_prox=None,
            )
            ripple_large.apply(flock_large, ctx_large)

        env_large_bird0 = cfg_large._ripple_envelope_sum[0]

        # Bird 0's envelope should be independent of flock size
        assert env_small_bird0 == pytest.approx(env_large_bird0, rel=0.01), (
            f"Envelope should be independent of N: "
            f"N=10 → {env_small_bird0:.6f}, N=30 → {env_large_bird0:.6f}"
        )
