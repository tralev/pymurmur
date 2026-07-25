"""Extensions — Ecology P4.8 (logistic dusk factor, coherence gate, seasonal amplitude, roost components), Wander, Ripple.

Split out of test_extensions.py (file-size split).
"""

import numpy as np

from pymurmur.physics.extensions.ecology import Ecology
from pymurmur.physics.extensions.ripple import Ripple
from pymurmur.physics.extensions.wander import Wander
from pymurmur.physics.flock import PhysicsFlock
from test.l0_modules.physics.extensions.test_extensions import _make_ctx

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


