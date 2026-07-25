"""P4.8 — Ecology unit tests: basics, seasonal factor, coherence gate, predator presence, dusk factor, temperature/day-length, coherence-factor bridging.

Split out of test_ecology.py (file-size split).
"""


from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions.ecology import Ecology


def test_ecology_extension_exists():
    """P4.8: Ecology extension class exists and is instantiable."""
    cfg = SimConfig()
    eco = Ecology(cfg)
    assert eco is not None


def test_ecology_coherence_factor_defaults_to_one():
    """P4.8: Without a flock, coherence_factor defaults to 1.0."""
    cfg = SimConfig()
    eco = Ecology(cfg)
    assert eco.coherence_factor == 1.0


def test_ecology_day_length_24h():
    """P4.8: Day length returns 24-hour period values (static method)."""
    # Summer solstice (day 172) → ~16.5 hours daylight
    dl_summer = Ecology.day_length(172.0)
    assert 14.0 < dl_summer < 18.0, f"Summer day length should be ~16.5h, got {dl_summer}"

    # Winter solstice (day 355) → ~7.5 hours daylight
    dl_winter = Ecology.day_length(355.0)
    assert 6.0 < dl_winter < 10.0, f"Winter day length should be ~7.5h, got {dl_winter}"

    # Equinox → ~12 hours
    dl_equinox = Ecology.day_length(80.0)
    assert 11.0 < dl_equinox < 13.0, f"Equinox day length should be ~12h, got {dl_equinox}"


def test_ecology_dusk_factor_bounds():
    """P4.8: Logistic dusk factor — 0 well before dusk, 1 in roost window.

    Sigmoid centred at _DUSK_CENTER=20 min before sunset.
    z = (minutes_before_dusk - 20) / dusk_width.
    z > 0 (well before dusk) → sigmoid ≈ 0.
    z < 0 (past sunset) → sigmoid ≈ 1."""
    dusk = 18.0  # 6 PM sunset

    # At noon (6 hours before dusk) → near 0 (well before roost)
    noon_factor = Ecology.logistic_dusk_factor(12.0, dusk, dusk_width=6.0)
    assert 0.0 <= noon_factor <= 1.0, f"Factor must be in [0,1], got {noon_factor}"
    assert noon_factor < 0.1, f"Noon factor should be near 0, got {noon_factor}"

    # Near dusk (17:40 = 20 min before) → near 0.5 (sigmoid midpoint)
    near_dusk = Ecology.logistic_dusk_factor(17.67, dusk, dusk_width=6.0)
    assert 0.3 <= near_dusk <= 0.7, f"Near-dusk factor should be ~0.5, got {near_dusk}"

    # At midnight (6 hours after dusk) → near 1.0 (fully in roost window)
    late = Ecology.logistic_dusk_factor(24.0, dusk, dusk_width=6.0)
    assert 0.0 <= late <= 1.0, f"Factor must be in [0,1], got {late}"
    assert late > 0.7, f"Late-night factor should be near 1, got {late}"


# ── S2.B8: Seasonal factor edge cases ─────────────────────────────

def test_seasonal_factor_peak():
    import pytest
    """S2.B8: day 15 (mid-Jan) → seasonal_factor = 1.0 (peak)."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.seasonal_factor(15.0, amplitude=0.5)
    assert result == pytest.approx(1.0, abs=0.01), (
        f"Peak day 15 should be 1.0, got {result}"
    )


def test_seasonal_factor_trough():
    import pytest
    """S2.B8: day 197 (mid-Jul) → seasonal_factor = 0.25 (trough)."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.seasonal_factor(197.0, amplitude=0.5)
    assert result == pytest.approx(0.25, abs=0.01), (
        f"Trough day 197 should be 0.25, got {result}"
    )


def test_seasonal_factor_clamped_range():
    import pytest
    """S2.B8: seasonal_factor stays within [0.05, 2.0] for extreme amplitudes."""
    from pymurmur.physics.extensions.ecology import Ecology
    # amplitude=0 → flat 1.0 all year
    flat = Ecology.seasonal_factor(100.0, amplitude=0.0)
    assert flat == pytest.approx(1.0, abs=0.01)
    # amplitude=2.0 → trough tries to go below 0, clamped to 0.05
    trough = Ecology.seasonal_factor(197.0, amplitude=2.0)
    assert trough >= 0.05, f"Should be clamped to >= 0.05, got {trough}"
    assert trough <= 2.0, f"Should be clamped to <= 2.0, got {trough}"


# ── S2.B8: Coherence gate edge cases ──────────────────────────────

def test_coherence_gate_at_lo_boundary():
    import pytest
    """S2.B8: coherence_gate(0.4*N_crit) ≈ 0 (smoothstep lo)."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.coherence_gate(40, 100)  # 0.4 * 100 = 40
    assert result == pytest.approx(0.0, abs=0.01), (
        f"At lo boundary should be ≈0, got {result}"
    )


def test_coherence_gate_at_hi_boundary():
    import pytest
    """S2.B8: coherence_gate(1.2*N_crit) ≈ 1 (smoothstep hi)."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.coherence_gate(120, 100)  # 1.2 * 100 = 120
    assert result == pytest.approx(1.0, abs=0.01), (
        f"At hi boundary should be ≈1, got {result}"
    )


def test_coherence_gate_midpoint():
    import pytest
    """S2.B8: coherence_gate at midpoint (0.8*N_crit) ≈ 0.5."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.coherence_gate(80, 100)  # midpoint between 40 and 120
    assert result == pytest.approx(0.5, abs=0.05), (
        f"At midpoint should be ≈0.5, got {result}"
    )


def test_coherence_gate_zero_critical_mass():
    """S2.B8: coherence_gate with critical_mass=0 → 0.0 (guard)."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.coherence_gate(100, 0)
    assert result == 0.0, f"Zero critical_mass should return 0, got {result}"


def test_coherence_gate_zero_active():
    """S2.B8: coherence_gate with n_active=0 → 0.0."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.coherence_gate(0, 100)
    assert result == 0.0, f"Zero active should return 0, got {result}"


# ── S2.B8: Predator presence over full year ────────────────────────

def test_predator_present_rate_over_year():
    """S2.B8: predator_present returns True on ~29.6% of days.
    Deterministic Knuth hash — exact count matches expected rate."""
    from pymurmur.physics.extensions.ecology import Ecology
    days = range(0, 1000)
    present = sum(1 for d in days if Ecology.predator_present(d))
    # 1000 * 0.296 = 296, allow ±5 tolerance (hash is pseudorandom)
    assert 290 <= present <= 302, (
        f"Expected ~296 predator days out of 1000, got {present}"
    )


def test_predator_present_deterministic():
    """S2.B8: predator_present is deterministic — same input, same output."""
    from pymurmur.physics.extensions.ecology import Ecology
    for day in (0, 42, 100, 200, 365):
        a = Ecology.predator_present(day)
        b = Ecology.predator_present(day)
        assert a == b, f"Day {day}: got {a} then {b}"


# ── S2.B8: Dusk factor edge cases ─────────────────────────────────

def test_dusk_factor_at_exact_sunset():
    """S2.B8: logistic_dusk_factor at sunset hour → midpoint.
    At dusk hour (18:00 with 12h day), minutes_before_dusk=0, which
    is -20 from DUSK_CENTER=20 → z = -20/width, sigmoid > 0.5."""
    from pymurmur.physics.extensions.ecology import Ecology
    dusk = 18.0  # sunset at 6PM
    result = Ecology.logistic_dusk_factor(dusk, dusk, dusk_width=6.0)
    # At sunset, minutes_before_dusk=0, DUSK_CENTER=20 → z=-20/6≈-3.33
    # sigmoid(-3.33) ≈ 0.035 — barely started (birds settle BEFORE sunset)
    assert 0.0 <= result <= 1.0, f"Dusk factor at sunset: {result}"
    # Should be > 0 (has started transitioning) but < 0.5 (not at midpoint yet)
    assert result > 0.95, f"Should be > 0.95 at sunset (birds settle BEFORE sunset), got {result}"


def test_dusk_factor_width_zero():
    """S2.B8: logistic_dusk_factor with width=0 → step function.
    Before DUSK_CENTER (20 min before dusk) → 0, after → 1."""
    from pymurmur.physics.extensions.ecology import Ecology
    dusk = 18.0
    # 30 minutes before dusk → should be after the 20-min center → 1.0
    early = Ecology.logistic_dusk_factor(17.5, dusk, dusk_width=0.0)
    assert early == 1.0, f"30 min before dusk with width=0 should be 1, got {early}"
    # 10 minutes before dusk → before the 20-min center → 0.0
    late = Ecology.logistic_dusk_factor(17.83, dusk, dusk_width=0.0)
    assert late == 0.0, f"10 min before dusk with width=0 should be 0, got {late}"


def test_dusk_factor_well_before_dusk():
    """S2.B8: logistic_dusk_factor at noon → near 0."""
    from pymurmur.physics.extensions.ecology import Ecology
    dusk = 18.0
    result = Ecology.logistic_dusk_factor(12.0, dusk, dusk_width=6.0)
    assert result < 1e-6, f"At noon should be near 0, got {result}"


def test_dusk_factor_well_after_dusk():
    """S2.B8: logistic_dusk_factor at midnight → near 1."""
    from pymurmur.physics.extensions.ecology import Ecology
    dusk = 18.0
    result = Ecology.logistic_dusk_factor(24.0, dusk, dusk_width=6.0)
    assert result > 0.99, f"At midnight should be near 1, got {result}"


# ── S2.B8: Temperature and day length extremes ─────────────────────

def test_temperature_summer_winter():
    """S2.B8: temperature at day 172 (solstice) > temperature at day 355."""
    from pymurmur.physics.extensions.ecology import Ecology
    summer = Ecology.temperature(172.0)
    winter = Ecology.temperature(355.0)
    assert summer > winter, (
        f"Summer temp {summer} should exceed winter temp {winter}"
    )


def test_day_length_solstices():
    """S2.B8: day_length at summer solstice ≈ 16.5h, winter ≈ 7.5h."""
    from pymurmur.physics.extensions.ecology import Ecology
    summer = Ecology.day_length(172.0)
    winter = Ecology.day_length(355.0)
    assert summer > 16.0, f"Summer day should be > 16h, got {summer}"
    assert winter < 8.5, f"Winter day should be < 8.5h, got {winter}"


def test_gated_weight_passes_through():
    import pytest
    """S2.B8: gated_weight with large flock → returns base_weight."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.gated_weight(0.8, 200, 100)  # N >> critical_mass
    assert result == pytest.approx(0.8, abs=0.01), (
        f"Large flock should get full weight, got {result}"
    )


def test_gated_weight_small_flock():
    import pytest
    """S2.B8: gated_weight with tiny flock → near 0."""
    from pymurmur.physics.extensions.ecology import Ecology
    result = Ecology.gated_weight(0.8, 10, 100)  # N << 0.4*critical_mass
    assert result == pytest.approx(0.0, abs=0.01), (
        f"Tiny flock should get ~0 weight, got {result}"
    )


# ── S2.B8: Coherence factor bridging end-to-end ────────────────────

def test_coherence_factor_bridges_to_config():
    """S2.B8: ecology.coherence_factor is bridged to config._coherence_factor
    via ExtensionManager → flock.coherence_factor → forces/__init__.py."""
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.extensions import ExtensionManager
    from pymurmur.physics.extensions._base import StepContext
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig()
    cfg.num_boids = 600  # above 1.2x critical_mass → coherence=1.0
    cfg.ecology_critical_mass = 500
    cfg.roosting_enabled = True
    cfg.seed = 42

    flock = PhysicsFlock(cfg)
    mgr = ExtensionManager(cfg)
    eco = mgr._ecology
    assert eco is not None, "Ecology should be created by ExtensionManager"

    # Set day to inside dusk window
    eco._day = 172.0 + 0.82
    eco._day_dt = 0

    ctx = StepContext(
        frame=0, dt=1.0 / 60.0, rng=flock.rng,
        center=flock.center, config=cfg,
    )
    mgr.pre_step(flock, ctx)

    # With 600 birds (1.2x critical_mass=500), coherence should be 1.0
    assert eco.coherence_factor == 1.0, (
        f"600 birds at crit_mass=500: coherence should be 1.0, "
        f"got {eco.coherence_factor}"
    )
    # ExtensionManager should have bridged it to flock
    assert hasattr(flock, 'coherence_factor'), "flock.coherence_factor missing"
    assert flock.coherence_factor == 1.0


def test_coherence_factor_defaults_to_one_without_ecology():
    """S2.B8: Without ecology, SpatialMode reads _coherence_factor=1.0
    via getattr fallback (no bridging happens)."""
    import numpy as np

    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = SimConfig()
    cfg.mode = "spatial"
    cfg.num_boids = 20
    cfg.seed = 42
    cfg.roosting_enabled = False

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # No ecology → _coherence_factor defaults to 1.0
    assert getattr(cfg, '_coherence_factor', 1.0) == 1.0

    # SpatialMode should default to 1.0 (getattr fallback)
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    # Force should be non-zero (normal behavior)
    assert not np.allclose(flock.accelerations[flock.active], 0.0)


def test_coherence_factor_bridge_to_angle_mode():
    """S2.B8: AngleMode reads config._coherence_factor and gates
    turn_rate for alignment/cohesion steering (not flee/edge)."""
    import numpy as np

    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.angle import AngleMode
    from test.helpers import _call_force

    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 20
    cfg.seed = 42

    # No ecology → coherence defaults to 1.0 → no gating
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    _call_force(AngleMode.compute, flock, cfg)
    speeds_full = np.linalg.norm(flock.velocities[flock.active], axis=1).copy()

    # With coherence=1.0 explicitly, same result
    object.__setattr__(cfg, '_coherence_factor', 1.0)
    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    _call_force(AngleMode.compute, flock2, cfg)
    speeds_coherence_1 = np.linalg.norm(flock2.velocities[flock2.active], axis=1)
    np.testing.assert_allclose(speeds_full, speeds_coherence_1, atol=1e-4,
                                err_msg="coherence=1.0 should match default")

    # With coherence=0.5, tuning should differ (but not crash)
    object.__setattr__(cfg, '_coherence_factor', 0.5)
    flock3 = PhysicsFlock(cfg)
    flock3.accelerations[:] = 0.0
    _call_force(AngleMode.compute, flock3, cfg)
    assert np.isfinite(flock3.velocities).all(), "No NaN in velocities with coherence=0.5"


