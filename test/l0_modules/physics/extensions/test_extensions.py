"""Extensions — StepContext standalone construction, ExtensionManager enable/disable per extension, pre_step, predator-conditional dispatch, Ecology basics (day length, apply, dusk roost pull, temperature, critical mass, predator flag).

Split out of test_extensions.py (file-size split).
"""

import numpy as np

from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.ecology import Ecology
from pymurmur.physics.flock import PhysicsFlock


def _make_ctx(flock, config, frame=0, dt=1.0/60.0):
    """Create a StepContext from a flock and config for extension tests."""
    return StepContext(
        frame=frame,
        dt=dt,
        rng=flock.rng,
        center=flock.center,
        config=config,
    )

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


