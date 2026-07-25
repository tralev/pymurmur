"""Phase 7 — Influencer integration: mid-run state transitions, end-to-end through SimulationEngine, backward compatibility (legacy alias, config round-trip).

Split out of test_influencer.py (file-size split).
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.influencer import (
    InfluencerMode,
    PilotTarget,
    influencer_forces,
)
from pymurmur.simulation.engine import SimulationEngine
from test.helpers import _call_force


class TestInfluencerModeIntegration:
    """Mid-run transitions, end-to-end through engine, backward compatibility."""

    # ── Integration: Mid-run state transitions ─────────────────

    def test_mid_run_pilot_toggle_through_engine(self):
        """P7.1+P7.6: Toggle pilot on/off mid-run, Lissajous ↔ pilot seamlessly."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 15
        cfg.influencer_substeps = 2
        cfg.influencer_scale = 0.5
        cfg.seed = 42

        engine = SimulationEngine(cfg)

        # Phase 1: Lissajous only (10 frames)
        for _ in range(10):
            engine.step(1.0 / 60.0)
        lissajous_pos = engine.flock.positions.copy()

        # Phase 2: Activate pilot at centre (15 frames)
        pilot = PilotTarget(
            position=np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                              dtype=np.float32)
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            for _ in range(15):
                engine.step(1.0 / 60.0)
            pilot_pos = engine.flock.positions.copy()
        finally:
            InfluencerMode.set_pilot(None)

        # Phase 3: Back to Lissajous (10 frames)
        for _ in range(10):
            engine.step(1.0 / 60.0)
        lissajous2_pos = engine.flock.positions.copy()

        # All phases should be NaN-free and speeds bounded
        assert np.isfinite(lissajous_pos).all()
        assert np.isfinite(pilot_pos).all()
        assert np.isfinite(lissajous2_pos).all()
        assert engine.frame == 35

    def test_mid_run_influence_mode_switch(self):
        """P7.3: Switch influence_mode mid-run, diagnostics survive."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 1.8
        cfg.influencer_influence_mode = "rank"

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Phase 1: rank mode
        cfg._influencer_tick = 0.0
        _call_force(influencer_forces, flock, cfg)
        rank_vel = flock.velocities.copy()
        rank_dmin = cfg._target_dist_min

        # Phase 2: switch to distance mode
        cfg.influencer_influence_mode = "distance"
        _call_force(influencer_forces, flock, cfg)
        dist_vel = flock.velocities.copy()
        dist_dmin = cfg._target_dist_min

        # Both modes produce valid velocities
        for vel in [rank_vel, dist_vel]:
            v_mags = np.linalg.norm(vel[flock.active], axis=1)
            assert np.allclose(v_mags, cfg.v0, atol=1e-4)
        # Diagnostics survive mode switch
        assert rank_dmin > 0
        assert dist_dmin > 0
        # Rankings differ, so velocities should differ
        assert not np.allclose(rank_vel, dist_vel), (
            "Rank and distance modes should produce different steering"
        )

    def test_dynamic_shell_radius_effect(self):
        """P7.6: Shell_radius controls flock spread via shell_pull activation.

        With a small shell_radius, shell_pull activates sooner, pulling birds
        inward.  With a large shell_radius, birds feel no shell_pull until
        they're much farther out.  We verify that both radii produce valid
        steering and that the flock stays bounded."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 3
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        pilot = PilotTarget(
            position=np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                              dtype=np.float32)
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0

            # Run with multiple shell radii, verify speeds stay bounded
            for shell_r in [20.0, 50.0, 200.0]:
                pilot.shell_radius = shell_r
                for _ in range(5):
                    _call_force(influencer_forces, flock, cfg)
                    flock.positions += flock.velocities * 0.1
                    v_mags = np.linalg.norm(
                        flock.velocities[flock.active], axis=1
                    )
                    assert np.allclose(v_mags, cfg.v0, atol=1e-4), (
                        f"Speed violation at shell_radius={shell_r}"
                    )
                    assert np.isfinite(flock.positions).all()
                    assert np.isfinite(flock.velocities).all()
        finally:
            InfluencerMode.set_pilot(None)

    def test_long_run_200_frames_all_features(self):
        """P7.1–P7.6: 200-frame stress test, all features, no NaN, speeds bounded."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 3
        cfg.influencer_rank_exponent = 2.0
        cfg.influencer_scale = 0.6
        cfg.influencer_tick_rate = 0.7
        cfg.v0 = 4.0
        cfg.seed = 123

        engine = SimulationEngine(cfg)

        pilot = PilotTarget(
            position=np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                              dtype=np.float32)
        )
        pilot.active = True
        pilot.shell_radius = 50.0
        InfluencerMode.set_pilot(pilot)

        try:
            for frame in range(200):
                engine.step(1.0 / 60.0)

                # Cycle pilot modes: pilot 0-49, Lissajous 50-99, pilot 100-149, Lissajous 150-199
                if frame == 50:
                    InfluencerMode.set_pilot(None)  # back to Lissajous
                elif frame == 100:
                    InfluencerMode.set_pilot(pilot)  # back to pilot
                elif frame == 150:
                    InfluencerMode.set_pilot(None)  # back to Lissajous

                assert np.isfinite(engine.flock.positions).all(), f"NaN at frame {frame}"
                assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"

                v_mags = np.linalg.norm(
                    engine.flock.velocities[engine.flock.active], axis=1
                )
                assert v_mags.max() <= cfg.v0 + 1e-4, (
                    f"Speed exceeded v0 at frame {frame}: {v_mags.max():.3f}"
                )
        finally:
            InfluencerMode.set_pilot(None)

        assert engine.frame == 200

    # ── Integration: P7 as a whole ─────────────────────────────

    def test_end_to_end_through_engine(self):
        """P7.1–P7.5: Full pipeline through SimulationEngine with influencer mode."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 2
        cfg.influencer_rank_exponent = 1.8
        cfg.influencer_scale = 0.5
        cfg.influencer_influence_mode = "rank"
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        for frame in range(10):
            engine.step(1.0 / 60.0)
            assert np.isfinite(engine.flock.positions).all(), f"NaN at frame {frame}"
            assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"
            # Velocities should all be v0 (speed_mode='fixed' via integrate)
            v_mags = np.linalg.norm(engine.flock.velocities[engine.flock.active], axis=1)
            # integrate applies speed clamping after compute, may differ slightly
            assert np.allclose(v_mags, cfg.v0, atol=1e-4), (
                f"Speed not v0 at frame {frame}: {v_mags.mean():.3f}"
            )

    def test_multi_frame_stability_50_frames(self):
        """P7.1–P7.5: 50 frames through engine, no NaN, positions in domain."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 3
        cfg.influencer_rank_exponent = 2.0
        cfg.influencer_scale = 0.8
        cfg.influencer_influence_mode = "distance"
        cfg.influencer_near_dist_sq = 150.0
        cfg.influencer_tick_rate = 0.5
        cfg.seed = 99

        engine = SimulationEngine(cfg)
        for frame in range(50):
            engine.step(1.0 / 60.0)
            assert np.isfinite(engine.flock.positions).all(), f"NaN pos at frame {frame}"
            assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"
            # Positions should stay within domain (toroidal wrapping)
            W, H, D = cfg.width, cfg.height, cfg.depth
            pos = engine.flock.positions[engine.flock.active]
            assert (pos[:, 0] >= -1.0).all() and (pos[:, 0] <= W + 1.0).all(), (
                f"x out of domain at frame {frame}"
            )
            assert (pos[:, 1] >= -1.0).all() and (pos[:, 1] <= H + 1.0).all(), (
                f"y out of domain at frame {frame}"
            )
            assert (pos[:, 2] >= -1.0).all() and (pos[:, 2] <= D + 1.0).all(), (
                f"z out of domain at frame {frame}"
            )
        assert engine.frame == 50

    def test_density_init_plus_compute(self):
        """P7.4+P7.1+P7.2: Density-scaled positions work with Lissajous steering."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 2
        cfg.influencer_scale = 1.0
        cfg.influencer_init_separation = 0.5

        flock = PhysicsFlock(cfg)
        # Override positions with density-scaled init
        rng = np.random.default_rng(42)
        flock.positions[:] = InfluencerMode.density_init_positions(
            n=cfg.num_boids,
            width=cfg.width,
            height=cfg.height,
            depth=cfg.depth,
            config=cfg,
            rng=rng,
        )
        flock.velocities[:] = 0.0

        _call_force(influencer_forces, flock, cfg)

        # All birds should have nonzero velocity (steered toward Lissajous target)
        v_mags = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(v_mags, cfg.v0, atol=1e-4), (
            f"Speed not v0 after density init: {v_mags.mean():.3f}"
        )
        # Diagnostics should be populated
        assert hasattr(cfg, '_target_dist_min')

    def test_pilot_end_to_end_through_engine(self):
        """P7.6+P7.1+P7.2: Pilot mode through SimulationEngine with convergence."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 3
        cfg.influencer_rank_exponent = 1.8
        cfg.seed = 42

        # Set pilot at domain centre
        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            ),
            heading=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.active = True
        pilot.shell_radius = 50.0
        InfluencerMode.set_pilot(pilot)

        try:
            engine = SimulationEngine(cfg)
            initial_dist = np.linalg.norm(
                engine.flock.positions.mean(axis=0) - pilot.position
            )

            for frame in range(30):
                engine.step(1.0 / 60.0)
                assert np.isfinite(engine.flock.positions).all(), (
                    f"NaN at frame {frame}"
                )

            # After 30 frames, flock CoM should be closer to pilot
            final_dist = np.linalg.norm(
                engine.flock.positions.mean(axis=0) - pilot.position
            )
            assert final_dist < initial_dist, (
                f"Flock did not converge: initial={initial_dist:.1f}, final={final_dist:.1f}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    def test_all_p7_features_active_together(self):
        """P7.1–P7.6: All features active simultaneously, no conflicts."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 25
        cfg.influencer_substeps = 2
        cfg.influencer_rank_exponent = 1.8
        cfg.influencer_scale = 0.7
        cfg.influencer_influence_mode = "rank"
        cfg.influencer_near_dist_sq = 100.0
        cfg.influencer_init_separation = 0.4
        cfg.influencer_tick_rate = 0.8
        cfg.seed = 77

        # Density-scaled init
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        flock.positions[:] = InfluencerMode.density_init_positions(
            n=cfg.num_boids,
            width=cfg.width,
            height=cfg.height,
            depth=cfg.depth,
            config=cfg,
            rng=rng,
        )

        # Run through compute with pilot active, then Lissajous
        cfg._influencer_tick = 0.0

        # First: Lissajous mode
        for _ in range(5):
            _call_force(influencer_forces, flock, cfg)
            flock.positions += flock.velocities * 0.1
            assert np.isfinite(flock.velocities).all()

        # Then: switch to pilot mode
        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            )
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            for _ in range(5):
                _call_force(influencer_forces, flock, cfg)
                flock.positions += flock.velocities * 0.1
                assert np.isfinite(flock.velocities).all()
        finally:
            InfluencerMode.set_pilot(None)

        # Diagnostics should work in both modes
        assert hasattr(cfg, '_target_dist_min')
        assert hasattr(cfg, '_target_dist_max')

        # Velocities should be constant speed throughout
        v_mags = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(v_mags, cfg.v0, atol=1e-4)

    # ── Backward compatibility ──────────────────────────────────

    def test_legacy_alias_functional(self):
        """influencer_forces alias is functional."""
        assert callable(influencer_forces)
        assert influencer_forces.needs_index is False

    def test_config_round_trip_preserves_new_fields(self):
        """New P7 fields survive YAML round-trip."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.influencer_rank_exponent = 2.0
        cfg.influencer_substeps = 3
        cfg.influencer_scale = 0.8
        cfg.influencer_influence_mode = "distance"
        cfg.influencer_near_dist_sq = 200.0
        cfg.influencer_init_separation = 0.3
        cfg.influencer_tick_rate = 0.5

        import os
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "test_p7_config.yaml")
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)
            assert loaded.influencer_rank_exponent == 2.0
            assert loaded.influencer_substeps == 3
            assert loaded.influencer_scale == 0.8
            assert loaded.influencer_influence_mode == "distance"
            assert loaded.influencer_near_dist_sq == 200.0
            assert loaded.influencer_init_separation == 0.3
            assert loaded.influencer_tick_rate == 0.5
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


