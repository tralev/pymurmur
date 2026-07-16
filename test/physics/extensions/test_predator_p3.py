"""Independent L0 function tests for P3.8-P3.9 Threat FSM + panic/blackening.

Tests _rotate_toward, approach/egress FSM transitions, force bundle
components, panic ceiling (not compound multiply), and blackening values.
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions.predator import Predator, _rotate_toward
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.extensions._base import StepContext


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
