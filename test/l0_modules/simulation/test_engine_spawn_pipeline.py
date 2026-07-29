"""Unit tests for simulation.engine — P10.4 cursor-ray spawn pipeline.

Split out of test_engine_clear_spawn_slider.py (file-size split) —
ClearBirds/SliderEngineIntegration stay in the original; this file
covers enqueue_spawn -> drain_commands -> flock.spawn_at integration.
"""


import numpy as np

from pymurmur.simulation.engine import SimulationEngine


# ── P10.4: Cursor-ray spawn pipeline — enqueue_spawn + drain_commands ─

class TestSpawnPipeline:
    """P10.4: enqueue_spawn → drain_commands → flock.spawn_at integration."""

    def test_spawn_bird_at_exact_position(self, default_config):
        """P10.4: enqueue_spawn(pos) + drain → bird at exact world position."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)
        n_before = engine.flock.N_active

        target = (123.0, 456.0, 789.0)
        engine.enqueue_spawn(target)
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 1
        # Find the newly spawned bird (last active index)
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[new_bird],
            np.array(target, dtype=np.float32),
        )

    def test_spawn_bird_velocity_bounded_by_v0(self, default_config):
        """P10.4: Spawned bird velocity obeys config.v0 (cube-velocity law)."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.v0 = 3.0
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        speed = float(np.linalg.norm(engine.flock.velocities[new_bird]))
        assert speed <= cfg.v0 + 1e-6, (
            f"Spawn velocity {speed:.4f} exceeds config.v0={cfg.v0}"
        )
        assert speed >= 0.0

    def test_spawn_predator_flag(self, default_config):
        """P10.4: enqueue_spawn(pos, is_predator=True) → predator flag set."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200), is_predator=True)
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert bool(engine.flock.is_predator[new_bird]) is True

    def test_spawn_prey_by_default(self, default_config):
        """P10.4: enqueue_spawn(pos) without is_predator → prey (False)."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert bool(engine.flock.is_predator[new_bird]) is False

    def test_spawn_updates_num_boids(self, default_config):
        """P10.4: After drain, config.num_boids reflects new N_active."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        assert cfg.num_boids == engine.flock.N_active

    def test_multiple_spawns_in_one_drain(self, default_config):
        """P10.4: Multiple enqueued spawns all appear in one drain."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)
        n_before = engine.flock.N_active

        engine.enqueue_spawn((100, 200, 300))
        engine.enqueue_spawn((400, 500, 600))
        engine.enqueue_spawn((700, 800, 900))
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 3
        active_idx = np.where(engine.flock.active)[0]
        b1, b2, b3 = active_idx[-3], active_idx[-2], active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[b1], np.array([100, 200, 300], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[b2], np.array([400, 500, 600], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[b3], np.array([700, 800, 900], dtype=np.float32),
        )

    def test_mixed_bird_and_predator_spawns(self, default_config):
        """P10.4: Mix of bird and predator spawns in same drain."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((100, 200, 300))         # prey
        engine.enqueue_spawn((400, 500, 600), is_predator=True)  # predator
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        bird, predator = active_idx[-2], active_idx[-1]
        assert bool(engine.flock.is_predator[bird]) is False
        assert bool(engine.flock.is_predator[predator]) is True
        # Positions should match enqueue order
        np.testing.assert_array_equal(
            engine.flock.positions[bird], np.array([100, 200, 300], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[predator], np.array([400, 500, 600], dtype=np.float32),
        )

    def test_spawn_different_v0_values(self, default_config):
        """P10.4: Changing config.v0 between spawns affects velocity."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.v0 = 2.0
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_2 = active_idx[-1]
        speed_2 = float(np.linalg.norm(engine.flock.velocities[bird_v0_2]))
        assert speed_2 <= 2.01

        # Change v0 and spawn again
        cfg.v0 = 0.5
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_05 = active_idx[-1]
        speed_05 = float(np.linalg.norm(engine.flock.velocities[bird_v0_05]))
        assert speed_05 <= 0.51

    def test_spawn_rng_advances(self, default_config):
        """P10.4: Two consecutive spawns at same position → different velocities."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        b1, b2 = active_idx[-2], active_idx[-1]
        assert not np.array_equal(
            engine.flock.velocities[b1], engine.flock.velocities[b2],
        ), "Two spawns at same position should have different velocities"

    def test_spawn_then_step_keeps_bird(self, default_config):
        """P10.4: After spawn + step(), the bird remains active and moves."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        pos_after_spawn = engine.flock.positions[new_bird].copy()

        # Step the simulation — bird should move
        engine.step()
        assert engine.flock.active[new_bird], "Bird should still be active after step"
        assert not np.array_equal(
            engine.flock.positions[new_bird], pos_after_spawn,
        ), "Bird should move after integration step"

    def test_spawn_then_clear_then_spawn(self, default_config):
        """P10.4: Spawn → clear → spawn again — second spawn works fine."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((100, 200, 300))
        engine.drain_commands()
        assert engine.flock.N_active == 11

        engine.enqueue_clear()
        engine.drain_commands()
        assert engine.flock.N_active == 0

        # Spawn again — should reuse an inactive slot
        engine.enqueue_spawn((500, 500, 500))
        engine.drain_commands()
        assert engine.flock.N_active == 1
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[new_bird],
            np.array([500, 500, 500], dtype=np.float32),
        )

    def test_spawn_acceleration_zero(self, default_config):
        """P10.4: Spawned bird starts with zero acceleration."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.accelerations[new_bird],
            np.zeros(3, dtype=np.float32),
        )

    def test_spawn_seed_assigned(self, default_config):
        """P10.4: Spawned bird gets a seed in [0, 1)."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert 0.0 <= engine.flock.seeds[new_bird] < 1.0

    def test_spawn_persists_across_multiple_steps(self, default_config):
        """P10.4: Spawned bird survives 5 simulation steps."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]

        for _ in range(5):
            engine.step()
            assert engine.flock.active[new_bird], (
                "Spawned bird should survive simulation steps"
            )
