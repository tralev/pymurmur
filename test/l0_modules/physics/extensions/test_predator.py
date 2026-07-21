"""P3.8-P3.9 Threat FSM + panic/blackening — unit and integration tests.

Unit tests (L0): _rotate_toward, approach/egress FSM transitions,
force bundle components, panic ceiling, blackening values.

Integration tests (extracted from test_extensions.py): full Predator
lifecycle through apply(), threat force decay, panic/blackening
end-to-end, and zero-active edge case.
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.predator import Predator, _rotate_toward
from pymurmur.physics.flock import PhysicsFlock


def _make_ctx(flock, config, frame=0, dt=1.0 / 60.0):
    return StepContext(
        frame=frame, dt=dt, rng=flock.rng,
        center=flock.center, config=config,
    )


class TestRotateToward:
    """P3.9: Rodrigues _rotate_toward helper."""

    def test_exact_alignment_returns_target(self):
        cur = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tgt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _rotate_toward(cur, tgt, 0.1)
        np.testing.assert_allclose(result, tgt, atol=1e-6)

    def test_rotation_capped_at_max_angle(self):
        """Large rotation (>max_angle) is capped, not jumped."""
        cur = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tgt = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 90°
        max_angle = np.deg2rad(10.0)  # only 10° per step
        result = _rotate_toward(cur, tgt, max_angle)
        angle = np.arccos(np.clip(np.dot(cur / np.linalg.norm(cur), result), -1, 1))
        assert angle <= max_angle + 1e-6, f"Rotation {np.rad2deg(angle):.1f}° exceeds cap {np.rad2deg(max_angle):.1f}°"

    def test_anti_parallel_handled(self):
        """Rotation from +x to -x works despite anti-parallel edge case."""
        cur = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tgt = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        result = _rotate_toward(cur, tgt, 0.5)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_result_is_unit_vector(self):
        """Output is always unit length."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            cur = rng.normal(size=3).astype(np.float32)
            tgt = rng.normal(size=3).astype(np.float32)
            result = _rotate_toward(cur, tgt, rng.uniform(0.01, 3.0))
            assert abs(np.linalg.norm(result) - 1.0) < 1e-6


class TestPredatorFSM:
    """P3.9: Predator FSM phase transitions."""

    def test_predator_starts_in_approach(self):
        cfg = SimConfig()
        p = Predator(cfg)
        assert p._phase == "approach"

    def test_approach_to_egress_when_close_to_center(self):
        """Approach→egress when within capture distance of centre."""
        cfg = SimConfig()
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        p._phase = "approach"
        # Place predator at the flock centre → dist=0, should trigger egress
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com.copy()
        p.apply(flock, _make_ctx(flock, cfg))
        assert p._phase == "egress", f"Expected egress, got {p._phase}"

    def test_egress_target_is_beyond_center(self):
        """During egress, target is center + dir * pass_dist."""
        cfg = SimConfig()
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        p._phase = "egress"
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com + np.array([100, 0, 0], dtype=np.float32)
        p._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Predator should move further away from centre during egress
        dist_before = np.linalg.norm(p._pos - com)
        p.apply(flock, _make_ctx(flock, cfg))
        dist_after = np.linalg.norm(p._pos - com)
        assert dist_after > dist_before, (
            f"Egress should increase distance: {dist_before:.0f}→{dist_after:.0f}"
        )


class TestPredatorMode:
    """C1: predator_mode selector — off | cursor | orbit | autonomous."""

    def test_default_mode_is_autonomous(self):
        assert SimConfig().predator_mode == "autonomous"

    def test_off_mode_freezes_position_and_publishes_zero_prox(self):
        """off: no movement, no force, but the extension instance survives."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_mode = "off"
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        pos_before = p._pos.copy()

        ctx = _make_ctx(flock, cfg)
        p.apply(flock, ctx)

        np.testing.assert_array_equal(p._pos, pos_before)
        assert np.allclose(ctx.threat_prox, 0.0)
        assert np.allclose(flock.accelerations, 0.0)

    def test_orbit_mode_never_enters_approach(self):
        """orbit: phase is forced to 'egress' even starting at capture distance."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_mode = "orbit"
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        p._phase = "approach"
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com.copy()  # would trigger approach→egress anyway under autonomous

        p.apply(flock, _make_ctx(flock, cfg))
        assert p._phase == "egress"

        # Run several more frames — orbit must stay in "egress" throughout,
        # never falling back to a capture-driven "approach".
        for _ in range(20):
            p.apply(flock, _make_ctx(flock, cfg))
            assert p._phase == "egress"

    def test_cursor_mode_targets_bridge_position(self):
        """cursor: with a live _cursor_world_pos bridge, the threat steers there."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_mode = "cursor"
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        p._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        cursor_pos = p._pos + np.array([0.0, 500.0, 0.0], dtype=np.float32)
        object.__setattr__(cfg, '_cursor_world_pos', tuple(cursor_pos))

        heading_before = p._dir.copy()
        p.apply(flock, _make_ctx(flock, cfg))
        # Steering should rotate toward the cursor (which is along +Y),
        # so the heading's y-component should grow from 0.
        assert p._dir[1] > heading_before[1]

    def test_cursor_mode_without_bridge_falls_back_to_autonomous(self):
        """cursor: with no _cursor_world_pos set, behaves like autonomous."""
        cfg_cursor = SimConfig()
        cfg_cursor.num_boids = 30
        cfg_cursor.seed = 7
        cfg_cursor.predator_mode = "cursor"

        cfg_auto = SimConfig()
        cfg_auto.num_boids = 30
        cfg_auto.seed = 7
        cfg_auto.predator_mode = "autonomous"

        flock_cursor = PhysicsFlock(cfg_cursor)
        flock_auto = PhysicsFlock(cfg_auto)
        p_cursor = Predator(cfg_cursor)
        p_auto = Predator(cfg_auto)
        # Same starting state
        p_auto._pos = p_cursor._pos.copy()
        p_auto._dir = p_cursor._dir.copy()

        p_cursor.apply(flock_cursor, _make_ctx(flock_cursor, cfg_cursor))
        p_auto.apply(flock_auto, _make_ctx(flock_auto, cfg_auto))

        np.testing.assert_allclose(p_cursor._pos, p_auto._pos)
        np.testing.assert_allclose(p_cursor._dir, p_auto._dir)


class TestPredatorForceBundle:
    """P3.9: Threat force bundle on nearby birds."""

    def test_threat_force_pushes_away(self):
        """Birds near predator are pushed radially outward."""
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        p = Predator(cfg)
        p._pos = np.array([500, 350, 200], dtype=np.float32)
        # Place bird slightly to the right of predator
        flock.positions[0] = np.array([510, 350, 200], dtype=np.float32)

        p.apply(flock, _make_ctx(flock, cfg))
        # Force on bird 0 should push right (+x, away from predator)
        assert flock.accelerations[0, 0] > 0, (
            f"Threat force should push away: acc={flock.accelerations[0]}"
        )

    def test_threat_force_zero_beyond_radius(self):
        """Birds beyond threat_dist receive no force."""
        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.predator_threat_radius = 12.0
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        p = Predator(cfg)
        p._pos = np.array([500, 350, 200], dtype=np.float32)
        # Place ALL birds far away so no bird is within threat_dist
        # threat_dist = predator_threat_radius * U * 2 = 12 * 160 * 2 = 3840
        # Place all birds > 5000 units away
        for i in range(5):
            flock.positions[i] = np.array([5000 + i * 10, 350, 200], dtype=np.float32)

        p.apply(flock, _make_ctx(flock, cfg))
        np.testing.assert_allclose(flock.accelerations, 0.0, atol=1e-6)

    def test_threat_prox_published_in_range(self):
        """ctx.threat_prox has values in [0,1] for birds within radius."""
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.predator_threat_radius = 200.0
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        p._pos = np.array([500, 350, 200], dtype=np.float32)

        ctx = _make_ctx(flock, cfg)
        p.apply(flock, ctx)

        assert ctx.threat_prox is not None
        tp = ctx.threat_prox
        assert tp.dtype == np.float32
        assert (tp >= 0.0).all() and (tp <= 1.0).all()


class TestPredatorPanic:
    """P3.8: Panic ceiling raise (NOT compound multiply)."""

    def test_panic_raises_max_speed_not_velocity(self):
        """Panic sets max_speed ceiling, does NOT multiply velocity."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        flock = PhysicsFlock(cfg)

        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com

        # Place bird near predator
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        old_speed = np.linalg.norm(flock.velocities[bird_idx])

        p.apply(flock, _make_ctx(flock, cfg))

        new_speed = np.linalg.norm(flock.velocities[bird_idx])
        # Speed should NOT have been multiplied (panic is ceiling, not compound)
        assert abs(new_speed - old_speed) < 1e-4, (
            f"Panic must NOT compound-multiply velocity: {old_speed:.2f}→{new_speed:.2f}"
        )
        # But max_speed should have been raised
        assert flock.max_speed is not None
        assert flock.max_speed[bird_idx] > cfg.v0, (
            "Panic must raise max_speed ceiling"
        )


class TestPredatorBlackening:
    """P3.8: Blackening values published to config."""

    def test_blackening_published_for_affected_birds(self):
        """cfg._threat_blackening is set with values > 1 for affected birds."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        flock = PhysicsFlock(cfg)

        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com

        # Place bird near predator
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)

        p.apply(flock, _make_ctx(flock, cfg))

        assert hasattr(cfg, '_threat_blackening'), "Blackening must be published"
        assert hasattr(cfg, '_threat_present'), "threat_present flag must be set"
        assert cfg._threat_present is True


class TestPredatorThreatPresent:
    """C1: _threat_present reset — three stale-state paths."""

    def test_off_mode_resets_threat_present(self):
        """predator_mode="off" ⇒ _threat_present=False, no stale leak."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_mode = "off"
        cfg._threat_present = True  # simulate stale state from prior frame

        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        p.apply(flock, _make_ctx(flock, cfg))
        assert cfg._threat_present is False, (
            "predator_mode='off' must reset _threat_present to False"
        )

    def test_zero_active_resets_threat_present(self):
        """n_active==0 ⇒ _threat_present=False, even if it was True before."""
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg._threat_present = True  # simulate stale state from a prior frame
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        p = Predator(cfg)
        p.apply(flock, _make_ctx(flock, cfg))
        assert cfg._threat_present is False, (
            "Zero active birds must reset _threat_present to False"
        )

    def test_no_birds_in_range_resets_threat_present(self):
        """Predator far from all birds ⇒ _threat_present=False."""
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.predator_threat_radius = 0.001  # near-zero radius
        cfg._threat_present = True  # simulate stale state

        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        # Place predator extremely far from all birds
        p._pos = np.array([-1e6, -1e6, -1e6], dtype=np.float32)

        p.apply(flock, _make_ctx(flock, cfg))
        assert cfg._threat_present is False, (
            "No birds within threat range must reset _threat_present to False"
        )


class TestPredatorZeroActive:
    """Edge case: zero active birds."""

    def test_predator_handles_zero_active(self):
        cfg = SimConfig()
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        p = Predator(cfg)
        old_pos = p._pos.copy()
        ctx = _make_ctx(flock, cfg)
        p.apply(flock, ctx)
        # Position unchanged, threat_prox not set
        assert np.allclose(p._pos, old_pos)
        assert ctx.threat_prox is not None  # predator sets it to zeros even with no active


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


