"""Phase 3 missing coverage tests — cross-extension integration.

Gaps filled:
  Integration: ripple->fold noise coupling, wander->drift alignment,
  predator->field and predator->other-modes cross-item integration.

Split out of test_field_integration.py (file-size split).
"""

from __future__ import annotations

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions.predator import Predator
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


class TestIntegrationPredatorToOtherModes:
    """C1×C3: Predator + non-field force modes engine pipeline.

    Verifies that the engine step order (predator extension → force
    compute → integrate) works correctly for vicsek and influencer
    modes, not just field mode. The predator publishes threat data
    to config; each force mode must handle (or ignore) it gracefully.
    """

    def test_predator_with_vicsek_engine_pipeline(self):
        """C1×C3: Predator + vicsek mode through engine pipeline.

        Vicsek mode reads _is_predator for species-aware alignment
        and collision resolution. The predator extension sets
        _is_predator on the flock. Verify full pipeline: predator
        applies forces → vicsek computes species-aware forces →
        integrate updates positions.
        """
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        # Step 1: Predator publishes threat data + _is_predator
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com.copy()
        # Place one bird near predator for blackening
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        p.apply(flock, _make_ctx(flock, cfg))

        assert cfg._threat_present is True, (
            "Predator must set _threat_present=True when birds are near threat"
        )

        # Step 2: Vicsek mode computes species-aware forces
        from pymurmur.physics.forces.vicsek import vicsek_forces
        _call_force(vicsek_forces, flock, cfg)

        # Should not crash, forces computed
        assert np.isfinite(flock.accelerations).all()
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6), (
            "Vicsek mode must apply non-zero forces after predator"
        )

    def test_predator_with_influencer_engine_pipeline(self):
        """C1×C3: Predator + influencer mode through engine pipeline.

        Influencer mode doesn't consume predator data, but the
        extension still runs in the engine step pipeline. Verify
        the full pipeline: predator applies forces → influencer
        computes → integrate updates positions.
        """
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        cfg.influencer_scale = 1.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        # Step 1: Predator publishes threat data
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com.copy()
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        p.apply(flock, _make_ctx(flock, cfg))

        assert cfg._threat_present is True, (
            "Predator must set _threat_present=True when birds are near threat"
        )

        # Step 2: Influencer mode computes — must not crash even though
        # it doesn't consume predator data. Forces may be zero depending
        # on bird-influencer distances; the key cross-item assertion is
        # that the pipeline doesn't crash and output is valid.
        from pymurmur.physics.forces.influencer import InfluencerMode
        _call_force(InfluencerMode.compute, flock, cfg)

        assert np.isfinite(flock.accelerations).all(), (
            "Influencer mode must not crash after predator step"
        )

    def test_predator_with_projection_engine_pipeline(self):
        """C1×C3: Predator + projection mode through engine pipeline.

        Projection mode doesn't consume predator data, but the
        engine step order (predator → force compute → integrate)
        must still work without error.
        """
        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com.copy()
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        p.apply(flock, _make_ctx(flock, cfg))
        assert cfg._threat_present is True

        from pymurmur.physics.forces.projection import projection_forces
        _call_force(projection_forces, flock, cfg)
        assert np.isfinite(flock.accelerations).all(), (
            "Projection mode must not crash after predator step"
        )

    def test_predator_with_spatial_engine_pipeline(self):
        """C1×C3: Predator + spatial mode through engine pipeline.

        Spatial mode reads is_predator for fear-weighted separation
        (predator birds get different avoidance radius). Verify the
        full pipeline: predator applies forces → spatial computes
        species-aware forces → output is valid.
        """
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com.copy()
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        p.apply(flock, _make_ctx(flock, cfg))
        assert cfg._threat_present is True

        from pymurmur.physics.forces.spatial import spatial_forces
        _call_force(spatial_forces, flock, cfg)
        assert np.isfinite(flock.accelerations).all(), (
            "Spatial mode must not crash after predator step"
        )
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6), (
            "Spatial mode must apply non-zero forces after predator"
        )

    def test_predator_with_angle_engine_pipeline(self):
        """C1×C3: Predator + angle mode through engine pipeline.

        Angle mode doesn't consume predator data, but the engine
        step order must still work without error.
        """
        cfg = SimConfig()
        cfg.mode = "angle"
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com.copy()
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        p.apply(flock, _make_ctx(flock, cfg))
        assert cfg._threat_present is True

        from pymurmur.physics.forces.angle import angle_forces
        _call_force(angle_forces, flock, cfg)
        assert np.isfinite(flock.accelerations).all(), (
            "Angle mode must not crash after predator step"
        )

    def test_predator_off_mode_all_force_modes(self):
        """C1×C3: predator_mode="off" should not crash any force mode.

        When predator is in "off" mode, it returns early without
        setting threat data. Every force mode must handle this
        gracefully. Test all modes (field already covered in
        TestIntegrationPredatorToField).
        """
        for mode in ["projection", "spatial", "vicsek", "influencer", "angle"]:
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = 20
            cfg.predator_mode = "off"
            cfg.seed = 42
            if mode == "influencer":
                cfg.influencer_scale = 1.0

            flock = PhysicsFlock(cfg)
            flock.accelerations[:] = 0.0

            p = Predator(cfg)
            p.apply(flock, _make_ctx(flock, cfg))
            assert cfg._threat_present is False, (
                f"predator_mode='off' must set _threat_present=False in {mode}"
            )

            # Force mode must run without crash
            if mode == "projection":
                from pymurmur.physics.forces.projection import projection_forces
                _call_force(projection_forces, flock, cfg)
            elif mode == "spatial":
                from pymurmur.physics.forces.spatial import spatial_forces
                _call_force(spatial_forces, flock, cfg)
            elif mode == "vicsek":
                from pymurmur.physics.forces.vicsek import vicsek_forces
                _call_force(vicsek_forces, flock, cfg)
            elif mode == "influencer":
                from pymurmur.physics.forces.influencer import InfluencerMode
                _call_force(InfluencerMode.compute, flock, cfg)
            elif mode == "angle":
                from pymurmur.physics.forces.angle import angle_forces
                _call_force(angle_forces, flock, cfg)

            assert np.isfinite(flock.accelerations).all(), (
                f"{mode} mode must not crash after predator 'off'"
            )

    def test_predator_publishes_data_for_field_consumption(self):
        """After predator.apply(), cfg has _threat_blackening and _threat_present."""
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

        # Verify all integration points
        assert hasattr(cfg, '_threat_present'), "Predator must set _threat_present"
        assert cfg._threat_present is True
        assert hasattr(cfg, '_threat_blackening'), "Predator must set _threat_blackening"
        assert hasattr(cfg, '_threat_active'), "Predator must set _threat_active"
        assert cfg._threat_blackening.dtype == np.float32

    def test_field_mode_runs_after_predator_without_crash(self):
        """Simulate the engine step order: predator → field mode."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        cfg.field_cohesion = 1.0
        cfg.field_flow = 1.0
        flock = PhysicsFlock(cfg)

        # Step 1: Predator
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        p.apply(flock, _make_ctx(flock, cfg))

        # Step 2: Field mode reads predator data
        flock.accelerations[:] = 0.0
        _call_force(FieldMode.compute, flock, cfg)

        # Should not crash, forces computed
        assert np.isfinite(flock.accelerations).all()
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6)

    def test_predator_off_mode_prevents_blackening_in_field(self):
        """C1×C3: predator_mode="off" → _threat_present=False → field mode
        uses default sep/coh, NOT blackening-modulated values."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.predator_mode = "off"
        cfg.seed = 42  # deterministic positions for reproducible comparison
        cfg.field_cohesion = 2.0
        cfg.field_separation = 3.0
        cfg.field_flow = 0.0
        cfg.field_alignment = 0.0
        cfg.field_noise = 0.0
        # Silence all but shell + slot_repulsion so coh/sep effect is visible
        cfg.disabled_terms = [
            "tangential", "buoyancy", "curl_flow", "fold_noise",
            "viscous_drag", "drift_alignment", "floating_boundary",
        ]

        # Create one flock, snapshot its positions for reuse
        flock = PhysicsFlock(cfg)
        ref_positions = flock.positions.copy()
        ref_velocities = flock.velocities.copy()

        # Step 1: Predator in "off" mode — must NOT set _threat_present
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com
        p.apply(flock, _make_ctx(flock, cfg))
        assert cfg._threat_present is False, (
            "predator_mode='off' must not set _threat_present"
        )

        # Step 2: Field mode — forces with default sep/coh (not blackened)
        flock.accelerations[:] = 0.0
        _call_force(FieldMode.compute, flock, cfg)
        acc_off = flock.accelerations[flock.active].copy()

        # Step 3: Reuse same flock state but with autonomous mode and blackening
        cfg.predator_mode = "autonomous"
        # Clone flock to same initial state
        flock2 = PhysicsFlock(cfg)
        # Place same bird near predator for blackening
        flock2.positions[:] = ref_positions
        flock2.velocities[:] = ref_velocities
        flock2.accelerations[:] = 0.0
        com2 = np.mean(flock2.positions[flock2.active], axis=0)
        p2 = Predator(cfg)
        p2._pos = com2.copy()
        bird_idx = np.where(flock2.active)[0][0]
        flock2.positions[bird_idx] = com2 + np.array([30, 0, 0], dtype=np.float32)
        p2.apply(flock2, _make_ctx(flock2, cfg))
        assert cfg._threat_present is True, (
            "autonomous mode with birds nearby must set _threat_present"
        )

        _call_force(FieldMode.compute, flock2, cfg)
        acc_auto = flock2.accelerations[flock2.active]

        # Forces should differ — blackening changes sep_eff/coh_eff
        assert not np.allclose(acc_off, acc_auto, atol=1e-4), (
            "predator_mode='off' vs 'autonomous' must produce different field forces"
            " (blackening modulates sep/coh)"
        )
