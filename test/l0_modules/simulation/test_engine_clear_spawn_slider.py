"""Unit tests for simulation.engine — P10.4 enqueue_clear/drain_commands, slider engine integration.

Split out of test_engine.py (file-size split). The cursor-ray spawn
pipeline (TestSpawnPipeline) was further split out to
test_engine_spawn_pipeline.py (file-size split).
"""


import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine


class TestClearBirds:
    """P10.4: enqueue_clear() + drain_commands() — clear all active boids."""

    def test_enqueue_clear_sets_pending_flag(self, default_config):
        """P10.4: enqueue_clear() sets pending_clear = True."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        assert not engine.commands.pending_clear
        engine.enqueue_clear()
        assert engine.commands.pending_clear

    def test_drain_commands_clear_deactivates_all_birds(self, default_config):
        """P10.4: drain_commands() with pending_clear sets flock.active[:] = False."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        assert engine.flock.N_active == 10

        engine.enqueue_clear()
        engine.drain_commands()

        assert engine.flock.N_active == 0
        assert not engine.flock.active.any(), (
            "All birds must be inactive after clear"
        )

    def test_drain_commands_clear_sets_num_boids_zero(self, default_config):
        """P10.4: After clear drain, config.num_boids is 0."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        engine.drain_commands()

        assert cfg.num_boids == 0, (
            f"config.num_boids should be 0 after clear, got {cfg.num_boids}"
        )

    def test_drain_commands_clear_resets_pending_flag(self, default_config):
        """P10.4: After drain, pending_clear is reset to False."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        assert engine.commands.pending_clear
        engine.drain_commands()
        assert not engine.commands.pending_clear, (
            "pending_clear must be reset after drain"
        )

    def test_clear_preserves_positions_and_velocities(self, default_config):
        """P10.4: Clear only flips active flag; positions/velocities are untouched."""
        cfg = default_config
        cfg.num_boids = 5
        engine = SimulationEngine(cfg)

        pos_before = engine.flock.positions.copy()
        vel_before = engine.flock.velocities.copy()

        engine.enqueue_clear()
        engine.drain_commands()

        # Positions and velocities should be unchanged (just inactive)
        np.testing.assert_array_equal(
            engine.flock.positions, pos_before,
            err_msg="Clear must preserve positions — only active flag changes"
        )
        np.testing.assert_array_equal(
            engine.flock.velocities, vel_before,
            err_msg="Clear must preserve velocities — only active flag changes"
        )

    def test_clear_with_no_active_birds_is_noop(self, default_config):
        """P10.4: Clearing an already-empty flock does nothing."""
        cfg = default_config
        cfg.num_boids = 0  # no initial birds — but capacity is allocated
        engine = SimulationEngine(cfg)
        # Deactivate all
        engine.flock.active[:] = False
        assert engine.flock.N_active == 0

        engine.enqueue_clear()
        engine.drain_commands()

        assert engine.flock.N_active == 0
        assert not engine.commands.pending_clear

    def test_clear_does_not_corrupt_other_config_fields(self, default_config):
        """P10.4: Clear only changes num_boids; other config fields unchanged."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.v0 = 3.5
        cfg.mode = "spatial"
        old_v0 = cfg.v0
        old_mode = cfg.mode
        old_sep = cfg.spatial.separation_weight

        engine = SimulationEngine(cfg)
        engine.enqueue_clear()
        engine.drain_commands()

        assert cfg.v0 == pytest.approx(old_v0)
        assert cfg.mode == old_mode
        assert cfg.spatial.separation_weight == pytest.approx(old_sep)

    def test_clear_resets_n_active_correctly(self, default_config):
        """P10.4: N_active is 0 after clear; previously active count tracked."""
        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)
        assert engine.flock.N_active == 20

        engine.enqueue_clear()
        engine.drain_commands()

        assert engine.flock.N_active == 0
        # active array length (capacity) should be unchanged
        assert len(engine.flock.active) >= 20

    def test_clear_after_add_resets_everything(self, default_config):
        """P10.4: Adding birds then clearing → all inactive, N_active=0."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        # Add some birds
        engine.enqueue_add(30)
        engine.drain_commands()
        assert engine.flock.N_active > 10

        # Then clear
        engine.enqueue_clear()
        engine.drain_commands()
        assert engine.flock.N_active == 0

    def test_clear_command_queue_flag_independent(self, default_config):
        """P10.4: pending_clear is independent of pending_add/pending_remove."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        # Queue multiple commands simultaneously
        engine.enqueue_add(5)
        engine.enqueue_remove(3)
        engine.enqueue_clear()

        assert engine.commands.pending_add == 5
        assert engine.commands.pending_remove == 3
        assert engine.commands.pending_clear

        # Drain — add/remove fire first, then clear
        engine.drain_commands()

        # After drain: add was processed, remove was processed, clear happened last
        assert engine.flock.N_active == 0
        assert not engine.commands.pending_clear
        assert engine.commands.pending_add == 0, (
            "pending_add should be consumed during drain"
        )
        assert engine.commands.pending_remove == 0, (
            "pending_remove should be consumed during drain"
        )

    def test_double_clear_is_noop(self, default_config):
        """P10.4: Enqueuing clear twice (just sets bool to True twice)."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        engine.enqueue_clear()  # second call: pending_clear already True, no-op
        engine.drain_commands()

        assert engine.flock.N_active == 0
        # pending_clear was reset to False by drain
        assert not engine.commands.pending_clear
        # Second drain is a no-op since there's no pending_clear
        engine.drain_commands()
        assert engine.flock.N_active == 0  # still empty
        assert not engine.commands.pending_clear

    def test_clear_in_step_cycle(self, default_config):
        """P10.4: Clear survives a full step() cycle (drain + physics)."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        # step() calls drain_commands() first, then runs physics
        engine.step()

        assert engine.flock.N_active == 0, (
            f"After step with clear, N_active should be 0, got {engine.flock.N_active}"
        )
        assert cfg.num_boids == 0

# Cross-cutting: P10.3 + P10.4 engine — HUD slider change affects physics step

class TestSliderEngineIntegration:
    """P10.3 + P10.4: Changing config via slider affects the next engine step."""

    def test_slider_noise_change_affects_step(self, default_config):
        """P10.3->P10.4 engine: Changing noise_scale via config changes
        the position deltas produced by the next step()."""
        from pymurmur.viz.hud import SliderHUD
        cfg = default_config
        cfg.num_boids = 20
        cfg.mode = "spatial"
        cfg.separation_weight = 0.0
        cfg.alignment_weight = 0.0
        cfg.cohesion_weight = 0.0
        cfg.noise_scale = 0.0
        cfg.v0 = 1.0  # slow speed so noise is measurable
        engine = SimulationEngine(cfg)

        # Step with zero noise
        engine.step()
        vel_no_noise = engine.flock.velocities.copy()

        # Use HUD slider to set noise to max
        hud = SliderHUD(cfg)
        hud._set_value(4, hud.TRACK_X0 + hud.TRACK_W)  # noise slider to max=0.5
        assert cfg.noise_scale == pytest.approx(0.5)

        engine.step()
        vel_with_noise = engine.flock.velocities.copy()

        # With noise, velocities change — speeds differ
        speeds_before = np.linalg.norm(vel_no_noise, axis=1).mean()
        speeds_after = np.linalg.norm(vel_with_noise, axis=1).mean()
        assert speeds_after != speeds_before, (
            "Noise should alter velocity magnitudes"
        )

    def test_slider_separation_change_affects_step(self, default_config):
        """P10.3->P10.4 engine: Increasing separation_weight via config
        produces different results than zero separation."""
        from pymurmur.viz.hud import SliderHUD
        cfg = default_config
        cfg.num_boids = 30
        cfg.mode = "spatial"
        cfg.separation_weight = 0.0
        cfg.alignment_weight = 0.0
        cfg.cohesion_weight = 0.0
        cfg.noise_scale = 0.0
        cfg.v0 = 1.0
        engine = SimulationEngine(cfg)

        # Step with zero separation — record velocities
        engine.step()
        vel_zero = engine.flock.velocities.copy()

        # Use HUD slider to set separation to max
        hud = SliderHUD(cfg)
        hud._set_value(0, hud.TRACK_X0 + hud.TRACK_W)  # sep to max=5.0
        assert cfg.spatial.separation_weight == pytest.approx(5.0)

        engine.step()
        vel_with_sep = engine.flock.velocities.copy()

        # With separation forces, velocities should differ
        assert not np.allclose(vel_zero, vel_with_sep, atol=1e-6), (
            "Separation should alter velocity patterns"
        )

    def test_slider_then_clear_then_slider(self, default_config):
        """P10.3->P10.4->P10.3: Change slider, clear birds, change slider
        again — engine doesn't crash and config is correct."""
        from pymurmur.viz.hud import SliderHUD
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        hud = SliderHUD(cfg)

        # Set sep to max
        hud._set_value(0, hud.TRACK_X0 + hud.TRACK_W)
        assert cfg.spatial.separation_weight == pytest.approx(5.0)

        # Clear
        engine.enqueue_clear()
        engine.drain_commands()
        assert engine.flock.N_active == 0

        # Change slider again — shouldn't crash
        hud._set_value(1, hud.TRACK_X0)  # coh to min
        assert cfg.spatial.cohesion_weight == pytest.approx(0.0)

        # Config integrity: sep should still be at max
        assert cfg.spatial.separation_weight == pytest.approx(5.0)
