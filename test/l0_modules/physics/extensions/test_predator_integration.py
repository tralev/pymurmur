"""P3.8-P3.9 Threat FSM + panic/blackening — integration tests.

Split out of test_predator.py (file-size split) — unit tests (FSM,
force bundle, panic, blackening as isolated classes) stay in the
original; this file covers full Predator lifecycle through apply()
(extracted from test_extensions.py originally).
"""

import numpy as np
import pytest

from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.predator import Predator
from pymurmur.physics.flock import PhysicsFlock


def _make_ctx(flock, config, frame=0, dt=1.0 / 60.0):
    return StepContext(
        frame=frame, dt=dt, rng=flock.rng,
        center=flock.center, config=config,
    )


# ── Integration-level predator tests (extracted from test_extensions.py) ──

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
    assert getattr(cfg, '_threat_present', None) is False


def test_predator_mode_validation_rejects_unknown(default_config):
    """C1: Invalid predator_mode raises ValueError at validation time."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.num_boids = 10
    cfg.predator_enabled = True  # must be True for validation guard to fire
    cfg.predator_mode = "hyperspace"
    with pytest.raises(ValueError, match="predator_mode"):
        cfg.validate()
