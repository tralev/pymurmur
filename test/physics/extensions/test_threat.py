"""P3.8–P3.9 Threat baseline tests — updated for full FSM rewrite.

After P3.8–P3.9 rewrote the Predator with approach/egress FSM,
panic ceiling (not compound multiply), and U-scaled distances,
this file tests the NEW behaviour.
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.predator import Predator


def _make_ctx(flock, config, frame=0, dt=1.0 / 60.0):
    """Create a StepContext for predator tests."""
    return StepContext(
        frame=frame, dt=dt, rng=flock.rng,
        center=flock.center, config=config,
    )


# ── Predator FSM: approach → egress cycle (P3.9) ──────────────────

def test_predator_starts_in_approach_phase():
    """Predator initialises in 'approach' phase."""
    cfg = SimConfig()
    p = Predator(cfg)
    assert p._phase == "approach"
    # P3.9: predator starts at edge position, not [0,0,0]
    assert np.allclose(p._pos, [cfg.width * 0.2, cfg.height * 0.5, cfg.depth * 0.5])


def test_predator_approaches_flock_centre():
    """In approach phase, predator moves toward flock centre."""
    cfg = SimConfig()
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)
    p = Predator(cfg)

    # Place predator far from centre
    p._pos = np.array([0, 0, 0], dtype=np.float32)

    com_before = np.mean(flock.positions[flock.active], axis=0)
    dist_before = np.linalg.norm(p._pos - com_before)

    p.apply(flock, _make_ctx(flock, cfg))

    # After apply, predator should be closer to COM
    dist_after = np.linalg.norm(p._pos - com_before)
    assert dist_after < dist_before, "Predator must move toward flock"


def test_predator_transitions_to_egress():
    """Predator transitions from approach → egress when within capture_dist of COM."""
    cfg = SimConfig()
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)
    p = Predator(cfg)
    p._phase = "approach"

    com = np.mean(flock.positions[flock.active], axis=0)
    # Place predator at COM → dist=0 < capture_dist → should transition
    p._pos = com.copy()

    p.apply(flock, _make_ctx(flock, cfg))
    assert p._phase == "egress", (
        f"Predator at COM should transition to egress, got {p._phase}"
    )


def test_predator_resets_to_approach_after_egress():
    """After egress, predator resets to approach after clearing clear_dist."""
    cfg = SimConfig()
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)
    p = Predator(cfg)

    com = np.mean(flock.positions[flock.active], axis=0)
    U = 0.4 * min(cfg.width, cfg.height, cfg.depth)  # = 160
    threat_radius = getattr(cfg, 'predator_threat_radius', 12.0)
    momentum = getattr(cfg, 'predator_momentum', 0.5)
    pass_dist = (0.92 + threat_radius * 2.6 + momentum * 1.32) * U  # large
    clear_dist = pass_dist * (0.72 + momentum * 0.16)  # even larger

    # Place predator far beyond clear_dist with dir pointing away
    p._phase = "egress"
    p._pos = com + np.array([clear_dist * 1.5, 0, 0], dtype=np.float32)
    p._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # away from centre
    old_pos = p._pos.copy()

    p.apply(flock, _make_ctx(flock, cfg))

    assert p._phase == "approach", (
        f"Predator beyond clear_dist should reset to approach, got {p._phase}"
    )
    # Position should have changed (moved away from centre)
    assert not np.allclose(p._pos, old_pos), (
        "Egress must move predator position"
    )


# ── Threat force on nearby birds (P3.9) ────────────────────────────

def test_predator_applies_threat_force():
    """Birds within threat_radius receive radial push away from predator."""
    cfg = SimConfig()
    cfg.num_boids = 10
    cfg.predator_threat_radius = 200.0
    cfg.predator_strength = 0.5
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    p = Predator(cfg)
    p._phase = "approach"
    p._pos = np.array([500, 350, 200], dtype=np.float32)

    # Place one bird very close to predator
    flock.positions[0] = np.array([510, 350, 200], dtype=np.float32)

    p.apply(flock, _make_ctx(flock, cfg))

    # Bird 0 should feel a force pushing it AWAY from predator
    acc = flock.accelerations[0]
    assert not np.allclose(acc, 0.0), "Bird near predator must feel threat force"
    # Force should push away from predator (predator at 500, bird at 510 → +x)
    assert acc[0] > 0, f"Threat force should push away (+x), got {acc}"


def test_threat_force_decays_with_distance():
    """Threat force is stronger for birds closer to predator."""
    cfg = SimConfig()
    cfg.num_boids = 10
    cfg.predator_threat_radius = 200.0
    cfg.predator_strength = 0.5

    # Near bird
    flock_near = PhysicsFlock(cfg)
    flock_near.accelerations[:] = 0.0
    flock_near.positions[0] = np.array([520, 350, 200], dtype=np.float32)  # d=20

    # Far bird
    flock_far = PhysicsFlock(cfg)
    flock_far.accelerations[:] = 0.0
    flock_far.positions[0] = np.array([680, 350, 200], dtype=np.float32)  # d=180

    p = Predator(cfg)
    p._pos = np.array([500, 350, 200], dtype=np.float32)
    p._phase = "approach"

    p.apply(flock_near, _make_ctx(flock_near, cfg))
    p.apply(flock_far, _make_ctx(flock_far, cfg))

    force_near = np.linalg.norm(flock_near.accelerations[0])
    force_far = np.linalg.norm(flock_far.accelerations[0])
    assert force_near > force_far, (
        f"Near force ({force_near:.4f}) must exceed far force ({force_far:.4f})"
    )


def test_threat_force_zero_beyond_radius():
    """Birds beyond threat_radius receive zero threat force.

    threat_dist = predator_threat_radius * U * 2.0.
    With default config U=160, radius=12 → threat_dist=3840.
    Place birds >5000 away.
    """
    cfg = SimConfig()
    cfg.num_boids = 5
    cfg.predator_threat_radius = 12.0
    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0

    p = Predator(cfg)
    p._pos = np.array([500, 350, 200], dtype=np.float32)

    # Place all birds far from predator
    for i in range(5):
        flock.positions[i] = np.array([5000 + i * 10, 350, 200], dtype=np.float32)

    p.apply(flock, _make_ctx(flock, cfg))

    assert np.allclose(flock.accelerations, 0.0), (
        "Birds beyond threat_radius must receive zero force"
    )


# ── threat_prox publishing ────────────────────────────────────────

def test_predator_publishes_threat_prox():
    """Predator sets ctx.threat_prox after apply()."""
    cfg = SimConfig()
    cfg.num_boids = 10
    cfg.predator_threat_radius = 200.0
    flock = PhysicsFlock(cfg)
    p = Predator(cfg)
    p._pos = np.array([500, 350, 200], dtype=np.float32)

    ctx = _make_ctx(flock, cfg)
    assert ctx.threat_prox is None, "threat_prox must start as None"

    p.apply(flock, ctx)

    assert ctx.threat_prox is not None, "Predator must publish threat_prox array"
    assert isinstance(ctx.threat_prox, np.ndarray)
    assert ctx.threat_prox.dtype == np.float32


def test_threat_prox_range_zero_to_one():
    """threat_prox values are in [0, 1] — 1 at predator pos, 0 at radius edge."""
    cfg = SimConfig()
    cfg.num_boids = 10
    cfg.predator_threat_radius = 200.0
    flock = PhysicsFlock(cfg)

    p = Predator(cfg)
    p._pos = np.array([500, 350, 200], dtype=np.float32)

    # Place a bird very close to predator (offset slightly — the
    # predator code guards d > 0, so d=0 is excluded)
    flock.positions[0] = np.array([500.01, 350, 200], dtype=np.float32)

    ctx = _make_ctx(flock, cfg)
    p.apply(flock, ctx)

    tp = ctx.threat_prox
    assert tp is not None
    assert np.all(tp >= 0.0), f"threat_prox values must be >= 0, got min={tp.min()}"
    assert np.all(tp <= 1.0), f"threat_prox values must be <= 1, got max={tp.max()}"
    # Bird at predator pos should have ~1.0 threat proximity
    assert np.isclose(tp[0], 1.0, atol=0.01), (
        f"Bird at predator pos should have threat_prox≈1.0, got {tp[0]}"
    )


def test_threat_prox_zero_beyond_radius():
    """threat_prox is 0 for birds beyond threat_dist.

    threat_dist = predator_threat_radius * U * 2.0.
    With default config U=160, radius=12 → threat_dist=3840.
    Place birds >5000 away.
    """
    cfg = SimConfig()
    cfg.num_boids = 5
    cfg.predator_threat_radius = 12.0
    flock = PhysicsFlock(cfg)

    p = Predator(cfg)
    p._pos = np.array([500, 350, 200], dtype=np.float32)
    for i in range(5):
        flock.positions[i] = np.array([5000 + i * 10, 600, 350], dtype=np.float32)

    ctx = _make_ctx(flock, cfg)
    p.apply(flock, ctx)

    assert ctx.threat_prox is not None
    assert np.allclose(ctx.threat_prox, 0.0), "Birds beyond radius must have threat_prox=0"


# ── Panic speed boost (P3.8: ceiling raise, NOT compound multiply) ─

def test_predator_panic_speed_boost():
    """Birds close to predator get max_speed CEILING raised, NOT velocity multiplied.

    P3.8: panic = clamp(prox_i, 0, 1) · threat_strength
    boost = panic · (0.72 + wave_gain·0.18 + vacuole_strength·0.12)
    max_speed_i = v0 · (1 + min(1.35, boost))   [ceiling raise]

    Velocity itself is NOT multiplied — only the cap changes.
    """
    cfg = SimConfig()
    cfg.num_boids = 30
    cfg.predator_threat_radius = 200.0
    cfg.predator_strength = 0.5
    flock = PhysicsFlock(cfg)

    p = Predator(cfg)
    com = np.mean(flock.positions[flock.active], axis=0)
    p._pos = com

    # Place one bird near predator
    bird_idx = np.where(flock.active)[0][0]
    flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
    old_speed = np.linalg.norm(flock.velocities[bird_idx])

    p.apply(flock, _make_ctx(flock, cfg))

    new_speed = np.linalg.norm(flock.velocities[bird_idx])
    # Velocity should NOT have been multiplied (panic is ceiling, not compound)
    assert abs(new_speed - old_speed) < 1e-4, (
        f"Panic must NOT compound-multiply velocity: {old_speed:.2f}→{new_speed:.2f}"
    )
    # But max_speed should have been raised
    assert flock.max_speed is not None
    assert flock.max_speed[bird_idx] > cfg.v0, (
        "Panic must raise max_speed ceiling"
    )


def test_panic_only_near_predator():
    """Birds far from predator do NOT get panic boost."""
    cfg = SimConfig()
    cfg.num_boids = 10
    cfg.predator_threat_radius = 200.0
    flock = PhysicsFlock(cfg)

    p = Predator(cfg)
    p._pos = np.array([500, 350, 200], dtype=np.float32)

    # Place a bird at d=5000 (far outside threat_dist ≈ 200*160*2 = 64000... hmm)
    # Actually threat_dist = 200 * 160 * 2 = 64000. So d=5000 IS within.
    # Let's use a smaller threat_radius.
    cfg2 = SimConfig()
    cfg2.num_boids = 10
    cfg2.predator_threat_radius = 1.0  # threat_dist = 1*160*2 = 320
    flock2 = PhysicsFlock(cfg2)
    p2 = Predator(cfg2)
    p2._pos = np.array([500, 350, 200], dtype=np.float32)
    flock2.positions[0] = np.array([5000, 3500, 2000], dtype=np.float32)  # far away
    old_speed = np.linalg.norm(flock2.velocities[0])

    p2.apply(flock2, _make_ctx(flock2, cfg2))

    new_speed = np.linalg.norm(flock2.velocities[0])
    # Speed should NOT have changed
    assert np.isclose(new_speed, old_speed, atol=0.1), (
        f"Bird far from predator should NOT get panic: {old_speed:.2f}→{new_speed:.2f}"
    )


# ── Zero-active handling ──────────────────────────────────────────

def test_predator_zero_active():
    """Predator.apply() handles zero active birds (early return).

    P3.9: early return publishes zero threat_prox, does not crash.
    """
    cfg = SimConfig()
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.active[:] = False

    p = Predator(cfg)
    old_pos = p._pos.copy()

    ctx = _make_ctx(flock, cfg)
    p.apply(flock, ctx)

    # Predator position and phase must be unchanged
    assert np.allclose(p._pos, old_pos)
    # P3.9: threat_prox is set to zeros (not None) for downstream consumers
    assert ctx.threat_prox is not None, "threat_prox should be set (zeros) even with no active birds"
    assert np.allclose(ctx.threat_prox, 0.0)
