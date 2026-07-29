"""Phase 3 missing coverage tests — cross-extension integration.

Gaps filled:
  Integration: ripple->fold noise coupling, wander->drift alignment,
  predator->field cross-item integration.

Split out of test_field_integration.py (file-size split).
Predator->other-force-modes tests live in
test_field_integration_crossitem_predator_other_modes.py (file-size
split of this file).
"""

from __future__ import annotations

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions.ripple import Ripple
from pymurmur.physics.extensions.wander import wander_heading
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.field import (
    FieldMode,
    _compute_drift_alignment,
    _compute_fold_noise,
)
from test.helpers import _call_force
from test.l0_modules.physics.field_integration.test_field_integration import _make_ctx


class TestIntegrationRippleToFoldNoise:
    """P3.7→P3.6: Ripple envelope sum feeds into fold noise scaling."""

    def test_fold_noise_scales_linearly_with_envelope(self):
        """Doubling ripple_envelope_sum doubles fold noise force."""
        pos = np.random.randn(20, 3).astype(np.float32) * 10
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(20, dtype=np.float32)
        F1 = _compute_fold_noise(pos, C, seeds, 1.0, 100.0, 1.0, 1.0, 1.0)
        F2 = _compute_fold_noise(pos, C, seeds, 1.0, 100.0, 1.0, 1.0, 2.0)
        np.testing.assert_allclose(F2, F1 * 2.0, atol=1e-4)

    def test_fold_noise_zero_when_envelope_zero(self):
        """ripple_envelope_sum=0 → no fold noise."""
        pos = np.random.randn(10, 3).astype(np.float32)
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(10, dtype=np.float32)
        F = _compute_fold_noise(pos, C, seeds, 1.0, 100.0, 1.0, 1.0, 0.0)
        np.testing.assert_allclose(F, 0.0)

    def test_ripple_sets_envelope_for_field_consumption(self):
        """Ripple.apply() sets config._ripple_envelope_sum that fold noise can read."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        cfg.field_flow = 1.0
        flock = PhysicsFlock(cfg)

        r = Ripple()
        r._t = 5.0  # well within train 1 envelope
        r.apply(flock, _make_ctx(flock, cfg))

        assert hasattr(cfg, '_ripple_envelope_sum'), "Ripple must export envelope"
        env = cfg._ripple_envelope_sum
        # D10: per-bird (N_capacity,) array that fold noise broadcasts per bird
        assert isinstance(env, np.ndarray)
        assert env.shape == (len(flock.positions),)
        # At t=5, train 1 is active → envelope should be non-negative
        assert np.all(env >= 0.0)
        # D10 whole-pipeline: fold noise must accept the per-bird envelope
        pos = flock.positions[flock.active]
        C = flock.center
        seeds = np.zeros(len(pos), dtype=np.float32)
        F = _compute_fold_noise(pos, C, seeds, 5.0, 100.0, 1.0, 1.0,
                                env[flock.active])
        assert F.shape == (len(pos), 3)
        assert np.isfinite(F).all()

    def test_ripple_fold_noise_full_pipeline_through_field_mode(self):
        """C3×C3: Ripple.apply() → FieldMode.compute() fold_noise reads envelope.

        Full cross-item pipeline: the ripple extension sets
        _ripple_envelope_sum on the config, then field mode's
        _compute_fold_noise reads it — all within one simulation step.
        """
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_flow = 5.0
        cfg.field_flow_pull = 2.0
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_separation = 0.0
        cfg.field_chase_strength = 0.0
        cfg.field_tangent_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0
        cfg.field_noise = 0.0
        # Isolate fold_noise
        cfg.disabled_terms = [
            "shell", "slot_repulsion", "tangential", "buoyancy",
            "curl_flow", "viscous_drag", "drift_alignment", "floating_boundary",
        ]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        # Step 1: Ripple sets _ripple_envelope_sum
        r = Ripple()
        r._t = 5.0  # active train
        r.apply(flock, _make_ctx(flock, cfg))
        assert np.any(cfg._ripple_envelope_sum > 0), (
            "Ripple must produce non-zero envelope at t=5.0"
        )

        # Step 2: Field mode's _compute_fold_noise reads the envelope
        _call_force(FieldMode.compute, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6), (
            "Fold noise must produce forces when ripple envelope is non-zero"
        )

        # Verify per-bird variance: envelope is per-bird, so fold noise
        # should vary per bird (not all birds get the same force)
        acc_active = flock.accelerations[flock.active]
        unique_axes = np.unique(np.round(acc_active[:, 0], 4))
        assert len(unique_axes) >= 2, (
            "Fold noise with per-bird envelope must produce per-bird variation"
        )

    def test_disabled_terms_noise_zeroes_jitter_in_engine_pipeline(self):
        """C3×C3: disabled_terms=["noise"] through field mode → no jitter.

        When "noise" is in disabled_terms, _compute_field_noise must be
        skipped. All other forces should still apply.
        """
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0
        cfg.field_chase_strength = 0.0
        cfg.field_tangent_pull = 0.0
        cfg.field_flow_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0
        cfg.field_noise = 0.5  # would normally produce jitter
        # Disable only noise; all other terms already zero-weighted
        cfg.disabled_terms = ["noise"]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(FieldMode.compute, flock, cfg)

        # With noise disabled and all other terms zero, only buoyancy
        # (z-axis only) should remain. All x/y forces should be zero.
        acc_active = flock.accelerations[flock.active]
        assert np.allclose(acc_active[:, :2], 0.0), (
            'disabled_terms=["noise"] must zero jitter: '
            f"x max={np.abs(acc_active[:,0]).max():.6f}, "
            f"y max={np.abs(acc_active[:,1]).max():.6f}"
        )

        # Now re-enable noise and verify jitter appears
        cfg.disabled_terms = []
        flock2 = PhysicsFlock(cfg)
        flock2.accelerations[:] = 0.0
        _call_force(FieldMode.compute, flock2, cfg)
        acc_with_noise = flock2.accelerations[flock2.active]
        # x/y forces should now be non-zero due to noise jitter
        assert np.max(np.abs(acc_with_noise[:, :2])) > 1e-6, (
            "Without disabled_terms, noise must produce x/y jitter"
        )


class TestIntegrationWanderToDrift:
    """P3.1→P3.6: Wander heading feeds into drift alignment force."""

    def test_drift_alignment_steers_toward_wander_heading(self):
        """Drift alignment accelerates birds toward wander_heading * v0."""
        # Get a wander heading at a known time
        heading = wander_heading(10.0)
        v0 = 4.0

        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        F = _compute_drift_alignment(v, heading, v0, 1.0, 1.0)

        # Force should point in same direction as heading (since v=0)
        dot = np.dot(F[0], heading)
        assert dot > 0, f"Drift should steer toward wander heading: dot={dot:.4f}"

    def test_drift_alignment_symmetry(self):
        """If velocity already matches wander_heading*v0, force is zero."""
        heading = wander_heading(5.0)
        v0 = 4.0
        target_vel = heading * v0

        F = _compute_drift_alignment(
            target_vel.reshape(1, 3).astype(np.float32),
            heading, v0, 1.0, 1.0,
        )
        np.testing.assert_allclose(F, 0.0, atol=1e-4)

    def test_wander_publishes_heading_for_drift(self):
        """Wander extension publishes flock.wander_heading that drift alignment can use."""
        from pymurmur.physics.extensions.wander import Wander
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "field"
        cfg.wander_enabled = True
        flock = PhysicsFlock(cfg)

        w = Wander()
        w.apply(flock, _make_ctx(flock, cfg))

        assert flock.wander_heading is not None
        assert np.abs(np.linalg.norm(flock.wander_heading) - 1.0) < 1e-6


class TestIntegrationPredatorToField:
    """P3.8→P3.6: Predator blackening published for field mode consumption."""

