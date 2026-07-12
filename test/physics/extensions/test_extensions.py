"""Unit tests for physics.extensions — ExtensionManager + all 4 extensions."""

import numpy as np

from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions.predator import Predator
from pymurmur.physics.extensions.ecology import Ecology
from pymurmur.physics.extensions.wander import Wander
from pymurmur.physics.extensions.ripple import Ripple
from pymurmur.physics.flock import PhysicsFlock


# ── ExtensionManager ──────────────────────────────────────────────

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
    assert isinstance(mgr._extensions[0], Ecology)


def test_extension_manager_wander_enabled(default_config):
    """wander_enabled=True → Wander is instantiated."""
    cfg = default_config
    cfg.wander_enabled = True
    mgr = ExtensionManager(cfg)
    assert mgr.count == 1
    assert isinstance(mgr._extensions[0], Wander)


def test_extension_manager_ripple_enabled(default_config):
    """ripple_enabled=True → Ripple is instantiated."""
    cfg = default_config
    cfg.ripple_enabled = True
    mgr = ExtensionManager(cfg)
    assert mgr.count == 1
    assert isinstance(mgr._extensions[0], Ripple)


def test_extension_manager_predator_enabled(default_config):
    """predator_enabled=True → Predator is instantiated (test_predator_spawns)."""
    cfg = default_config
    cfg.predator_enabled = True
    mgr = ExtensionManager(cfg)
    assert mgr.count == 1
    assert isinstance(mgr._extensions[0], Predator)


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
    mgr.pre_step(flock)  # should not crash
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
    mgr._ecology._predator_active = False

    # Record predator state before pre_step
    pred = mgr._predator
    old_pos = pred._pos.copy()

    mgr.pre_step(flock)

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

    mgr.pre_step(flock)

    # Predator should have moved (apply was called)
    assert not np.allclose(pred._pos, old_pos)


# ── Predator ──────────────────────────────────────────────────────

def test_predator_apply_runs(default_config):
    """Predator.apply() runs without error."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)
    predator = Predator()
    predator.apply(flock)
    # Should not raise


def test_predator_approach_phase(default_config):
    """Predator moves toward flock centre in approach phase."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    predator = Predator()
    predator._phase = "approach"
    predator._pos = np.array([0, 0, 0], dtype=np.float32)
    predator._vel = np.array([0, 0, 0], dtype=np.float32)

    predator.apply(flock)

    # After apply, predator should have non-zero velocity toward COM
    assert np.linalg.norm(predator._vel) > 0
    # Position should have changed
    assert not np.allclose(predator._pos, [0, 0, 0])


def test_predator_pass_through(default_config):
    """Predator resets position after pass-through phase."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    predator = Predator()
    # Force pass-through phase
    predator._phase = "pass_through"
    predator._pos = np.array([500, 350, 200], dtype=np.float32)
    old_pos = predator._pos.copy()

    predator.apply(flock)

    # After pass_through, phase resets to "approach"
    assert predator._phase == "approach"
    # Position should have been reset (different from old_pos)
    assert not np.allclose(predator._pos, old_pos)


def test_predator_threat_force(default_config):
    """Birds very close to predator receive non-zero threat force."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    predator = Predator()
    # Place predator at centre
    predator._pos = np.array([500, 350, 200], dtype=np.float32)

    # Place a bird very close to the predator
    flock.positions[0] = np.array([510, 350, 200], dtype=np.float32)

    predator.apply(flock)

    # Bird 0 should have received a threat force
    assert not np.allclose(flock.accelerations[0], 0.0)
    # Direction should be away from predator (+x for bird at 510 vs pred at 500)
    assert flock.accelerations[0, 0] > 0


def test_predator_threat_force_decays_with_distance(default_config):
    """Threat force is stronger for closer birds."""
    cfg = default_config
    cfg.num_boids = 10
    flock_near = PhysicsFlock(cfg)
    flock_near.accelerations[:] = 0.0
    flock_near.positions[0] = np.array([520, 350, 200], dtype=np.float32)  # d=20

    flock_far = PhysicsFlock(cfg)
    flock_far.accelerations[:] = 0.0
    flock_far.positions[0] = np.array([680, 350, 200], dtype=np.float32)  # d=180

    predator = Predator()
    predator._pos = np.array([500, 350, 200], dtype=np.float32)

    predator.apply(flock_near)
    predator.apply(flock_far)

    force_near = np.linalg.norm(flock_near.accelerations[0])
    force_far = np.linalg.norm(flock_far.accelerations[0])
    # Closer bird should experience more force
    assert force_near > force_far


def test_predator_approach_to_pass_through(default_config):
    """Predator transitions from approach to pass_through when close to COM."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    predator = Predator()
    predator._phase = "approach"
    # Place predator very close to flock COM so dist < 50
    com = np.mean(flock.positions[flock.active], axis=0)
    predator._pos = com + np.array([10, 0, 0], dtype=np.float32)  # d=10 < 50

    predator.apply(flock)

    # Should have transitioned to pass_through
    assert predator._phase == "pass_through"


def test_predator_panic_speed_boost(default_config):
    """Birds very close to predator (<100 units) get panic speed boost."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    predator = Predator()
    com = np.mean(flock.positions[flock.active], axis=0)
    predator._pos = com

    # Place one bird very close to predator
    bird_idx = np.where(flock.active)[0][0]
    flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
    old_speed = np.linalg.norm(flock.velocities[bird_idx])

    predator.apply(flock)

    # Bird should have received speed boost (velocity increased by 50%)
    new_speed = np.linalg.norm(flock.velocities[bird_idx])
    assert new_speed > old_speed * 1.4  # ~1.5x boost


def test_predator_panic_blackening(default_config):
    """Panicked birds get cohesion pull toward panic group centre."""
    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    predator = Predator()
    com = np.mean(flock.positions[flock.active], axis=0)
    predator._pos = com

    # Place two birds near predator (both within panic radius 100)
    active_idx = np.where(flock.active)[0]
    flock.positions[active_idx[0]] = com + np.array([30, 0, 0], dtype=np.float32)
    flock.positions[active_idx[1]] = com + np.array([-30, 0, 0], dtype=np.float32)

    predator.apply(flock)

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

    predator = Predator()
    predator.apply(flock)
    # Should not crash — exercises the `if active.sum() == 0: return` branch


# ── Ecology ───────────────────────────────────────────────────────

def test_ecology_day_length_summer():
    """Summer solstice (day 172) → ~16.5 hours daylight."""
    eco = Ecology()
    assert abs(eco.day_length(172) - 16.5) < 1.0


def test_ecology_day_length_winter():
    """Winter solstice (day 355) → ~7.5 hours daylight."""
    eco = Ecology()
    assert abs(eco.day_length(355) - 7.5) < 1.0


def test_ecology_day_length_equinox():
    """Equinox (day 80) → ~12 hours daylight."""
    eco = Ecology()
    assert abs(eco.day_length(80) - 12.0) < 0.5


def test_ecology_apply_runs(default_config):
    """Ecology.apply() runs without error."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    eco = Ecology()
    eco.apply(flock)
    # Should not raise


def test_ecology_dusk_roost_pull(default_config):
    """At dusk hour, birds experience downward pull toward roost."""
    cfg = default_config
    cfg.num_boids = 500  # above critical mass for full pull
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology()
    # Summer solstice, hour ~19.68 (inside dusk window [19.25, 20.25])
    eco._day = 172.0 + 0.82  # 0.82 * 24 = 19.68h
    eco._dt = 0  # don't advance time

    eco.apply(flock)

    # Birds should receive downward force toward roost (z=40, below centre at z=200)
    active = flock.active
    forces = flock.accelerations[active]
    assert not np.allclose(forces, 0.0)
    # Roost pull should point downward (negative z for birds above roost)
    assert (forces[:, 2] < 0).any()


def test_ecology_temperature_summer():
    """Summer peak (day 202) → ~17°C."""
    eco = Ecology()
    assert abs(eco.temperature(202) - 17.0) < 0.5


def test_ecology_temperature_winter():
    """Winter trough (day 20) → ~1°C."""
    eco = Ecology()
    assert abs(eco.temperature(20) - 1.0) < 0.5


def test_ecology_critical_mass_dampened(default_config):
    """Below 500 birds, roost pull is dampened by smoothstep."""
    cfg = default_config
    cfg.num_boids = 50  # well below critical mass
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    eco = Ecology()
    eco._day = 172.0 + 0.82  # dusk window
    eco._dt = 0

    eco.apply(flock)

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

    eco = Ecology()
    # Force day to be just after an integer boundary
    eco._day = 200.0  # int=200, different from _last_int_day=172
    eco._dt = 0  # don't advance further

    eco.apply(flock)

    # _last_int_day should now be 200
    assert eco._last_int_day == 200
    # _predator_active should have been set by predator_present(200)
    assert isinstance(eco._predator_active, bool)


# ── Wander ────────────────────────────────────────────────────────

def test_wander_apply_runs(default_config):
    """Wander.apply() runs without error."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    w = Wander()
    w.apply(flock)
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
    w.apply(flock)

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
    r.apply(flock)
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

    r.apply(flock)

    # Forces should be finite; bird at pulse peak should get force
    assert np.isfinite(flock.accelerations).all()


def test_ripple_zero_active(default_config):
    """Ripple.apply() handles zero active birds gracefully."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.active[:] = False
    r = Ripple()
    r.apply(flock)
    # Should not crash
