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
from pymurmur.physics.extensions.predator import Predator
from pymurmur.physics.extensions.ecology import Ecology
from pymurmur.physics.extensions.wander import Wander
from pymurmur.physics.extensions.ripple import Ripple
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


# ── Predator ──────────────────────────────────────────────────────

def test_predator_apply_runs(default_config):
    """Predator.apply() runs without error."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)
    predator = Predator(cfg)
    predator.apply(flock, _make_ctx(flock, cfg))
    # Should not raise


def test_predator_approach_phase(default_config):
    """Predator moves toward flock centre in approach phase."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    predator = Predator(cfg)
    predator._phase = "approach"
    predator._pos = np.array([0, 0, 0], dtype=np.float32)
    predator._vel = np.array([0, 0, 0], dtype=np.float32)

    predator.apply(flock, _make_ctx(flock, cfg))

    # After apply, predator should have non-zero velocity toward COM
    assert np.linalg.norm(predator._vel) > 0
    # Position should have changed
    assert not np.allclose(predator._pos, [0, 0, 0])


def test_predator_pass_through(default_config):
    """P3.9: Predator in egress phase moves away from flock centre."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    predator = Predator(cfg)
    # Force egress phase (P3.9 renames pass_through → egress)
    com = np.mean(flock.positions[flock.active], axis=0)
    predator._phase = "egress"
    predator._pos = com + np.array([100, 0, 0], dtype=np.float32)
    predator._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dist_before = np.linalg.norm(predator._pos - com)

    predator.apply(flock, _make_ctx(flock, cfg))

    # During egress, predator moves further away from centre
    dist_after = np.linalg.norm(predator._pos - com)
    assert dist_after > dist_before, "Egress predator must move away from centre"


def test_predator_threat_force(default_config):
    """Birds very close to predator receive non-zero threat force."""
    cfg = default_config
    cfg.num_boids = 10
    cfg.predator_threat_radius = 200.0  # large radius for test
    cfg.predator_strength = 0.5
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    predator = Predator(cfg)
    # Place predator at centre
    predator._pos = np.array([500, 350, 200], dtype=np.float32)

    # Place a bird very close to the predator
    flock.positions[0] = np.array([510, 350, 200], dtype=np.float32)

    predator.apply(flock, _make_ctx(flock, cfg))

    # Bird 0 should have received a threat force
    assert not np.allclose(flock.accelerations[0], 0.0)
    # Direction should be away from predator (+x for bird at 510 vs pred at 500)
    assert flock.accelerations[0, 0] > 0


def test_predator_threat_force_decays_with_distance(default_config):
    """Threat force is stronger for closer birds."""
    cfg = default_config
    cfg.num_boids = 10
    cfg.predator_threat_radius = 200.0
    cfg.predator_strength = 0.5
    flock_near = PhysicsFlock(cfg)
    flock_near.accelerations[:] = 0.0
    flock_near.positions[0] = np.array([520, 350, 200], dtype=np.float32)  # d=20

    flock_far = PhysicsFlock(cfg)
    flock_far.accelerations[:] = 0.0
    flock_far.positions[0] = np.array([680, 350, 200], dtype=np.float32)  # d=180

    predator = Predator(cfg)
    predator._pos = np.array([500, 350, 200], dtype=np.float32)

    predator.apply(flock_near, _make_ctx(flock_near, cfg))
    predator.apply(flock_far, _make_ctx(flock_far, cfg))

    force_near = np.linalg.norm(flock_near.accelerations[0])
    force_far = np.linalg.norm(flock_far.accelerations[0])
    # Closer bird should experience more force
    assert force_near > force_far


def test_predator_approach_to_pass_through(default_config):
    """P3.9: Predator transitions from approach to egress when close to COM."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    predator = Predator(cfg)
    predator._phase = "approach"
    # Place predator at COM → dist=0 < capture_dist → egress
    com = np.mean(flock.positions[flock.active], axis=0)
    predator._pos = com.copy()

    predator.apply(flock, _make_ctx(flock, cfg))

    # Should have transitioned to egress
    assert predator._phase == "egress"


def test_predator_panic_speed_boost(default_config):
    """P3.8: Birds close to predator get max_speed CEILING raised, not velocity multiplied.

    panic = clamp(prox, 0,1) · threat_strength
    boost = panic · (0.72 + wave_gain·0.18 + vacuole·0.12)
    max_speed = v0 · (1 + min(1.35, boost))  [ceiling raise, NOT compound multiply]
    """
    cfg = default_config
    cfg.num_boids = 30
    cfg.predator_threat_radius = 200.0
    cfg.predator_strength = 0.5
    flock = PhysicsFlock(cfg)

    predator = Predator(cfg)
    com = np.mean(flock.positions[flock.active], axis=0)
    predator._pos = com

    # Place one bird very close to predator
    bird_idx = np.where(flock.active)[0][0]
    flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
    old_speed = np.linalg.norm(flock.velocities[bird_idx])

    predator.apply(flock, _make_ctx(flock, cfg))

    # P3.8: velocity is NOT multiplied — only max_speed ceiling changes
    new_speed = np.linalg.norm(flock.velocities[bird_idx])
    assert abs(new_speed - old_speed) < 1e-4, (
        f"Panic must NOT compound-multiply velocity: {old_speed:.2f}→{new_speed:.2f}"
    )
    # But max_speed ceiling should have been raised
    assert flock.max_speed is not None
    assert flock.max_speed[bird_idx] > cfg.v0, (
        "Panic must raise max_speed ceiling"
    )


def test_predator_panic_blackening(default_config):
    """Panicked birds get cohesion pull toward panic group centre."""
    cfg = default_config
    cfg.num_boids = 30
    cfg.predator_threat_radius = 200.0  # half=100 for panic threshold
    cfg.predator_strength = 0.5
    cfg.predator_split_gain = 0.3
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    predator = Predator(cfg)
    com = np.mean(flock.positions[flock.active], axis=0)
    predator._pos = com

    # Place two birds near predator (both within panic radius 100)
    active_idx = np.where(flock.active)[0]
    flock.positions[active_idx[0]] = com + np.array([30, 0, 0], dtype=np.float32)
    flock.positions[active_idx[1]] = com + np.array([-30, 0, 0], dtype=np.float32)

    predator.apply(flock, _make_ctx(flock, cfg))

    # Both birds should have received non-zero additional forces
    # (threat force + cohesion pull from panic blackening)
    assert not np.allclose(flock.accelerations[active_idx[0]], 0.0)
    assert not np.allclose(flock.accelerations[active_idx[1]], 0.0)


def test_predator_zero_active(default_config):
    """Predator.apply() handles zero active birds gracefully (early return)."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.active[:] = False

    predator = Predator(cfg)
    predator.apply(flock, _make_ctx(flock, cfg))
    # Should not crash — exercises the `if active.sum() == 0: return` branch


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
    """Below critical mass birds, roost pull is dampened by smoothstep."""
    cfg = default_config
    cfg.num_boids = 50  # well below critical mass
    cfg.ecology_critical_mass = 500
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology(cfg)
    eco._day = 172.0 + 0.82  # dusk window
    eco._day_dt = 0

    eco.apply(flock, _make_ctx(flock, cfg))

    # Forces should still be finite (dampened, not zero)
    assert np.isfinite(flock.accelerations).all()

    # With 50 birds (10% of critical mass), forces should be much smaller
    # than with 500 birds. Test by comparing force magnitudes.
    mass_factor = (50 / 500) ** 2 * (3 - 2 * 50 / 500)  # ~0.028
    assert mass_factor < 0.05  # should be heavily dampened


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
    flock = PhysicsFlock(cfg)

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

        # Start with small flock: 50 birds → heavily dampened
        cfg.num_boids = 50
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

        # Add 450 birds — N_active goes from 50 → 500, reaching critical mass
        flock.accelerations[:] = 0.0  # reset forces
        flock.add_boids(450, cfg)

        # Same Ecology, same dusk time, now much larger flock
        eco.apply(flock, _make_ctx(flock, cfg))
        force_large = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )

        # Force must increase — mass_factor went from dampened (~0.028) → 1.0
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

        # Start with a very small flock
        cfg.num_boids = 30
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

        # Add many birds — N_active changes from 30 → 530
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
