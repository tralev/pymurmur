"""Unit tests for physics.flock — P10.4 spawn_at engine plumbing (v0/rng flow).

Split out of test_flock.py (file-size split).
"""

import numpy as np
import pytest


class TestSpawnAtEnginePlumbing:
    """P10.4: engine.enqueue_spawn + drain_commands passes v0 and rng correctly."""

    @pytest.fixture
    def _engine(self) -> tuple:
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.v0 = 3.0
        engine = SimulationEngine(cfg)
        return engine, cfg

    def test_engine_spawn_uses_config_v0(self, _engine):
        """Engine passes config.v0 to spawn_at — velocity bounded by v0."""
        engine, cfg = _engine
        n_before = engine.flock.N_active

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 1
        # Find the newly spawned bird (highest active index)
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        speed = float(np.linalg.norm(engine.flock.velocities[new_bird]))
        assert speed <= cfg.v0 + 1e-6, (
            f"Spawned velocity {speed:.4f} exceeds config.v0={cfg.v0}"
        )

    def test_engine_spawn_position_is_exact(self, _engine):
        """Engine spawn places bird at exact enqueued position."""
        engine, cfg = _engine
        target = (123.0, 456.0, 789.0)
        engine.enqueue_spawn(target)
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[new_bird],
            np.array(target, dtype=np.float32),
        )

    def test_engine_spawn_predator_flag(self, _engine):
        """Engine spawn with is_predator=True sets the flag."""
        engine, cfg = _engine
        engine.enqueue_spawn((500, 350, 200), is_predator=True)
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert bool(engine.flock.is_predator[new_bird]) is True

    def test_config_v0_change_affects_spawn(self, _engine):
        """Changing config.v0 before drain_commands affects spawn velocity."""
        engine, cfg = _engine

        # Spawn with v0=3.0
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_3 = active_idx[-1]
        speed_3 = float(np.linalg.norm(engine.flock.velocities[bird_v0_3]))

        # Change config.v0 to a much lower value
        cfg.v0 = 0.5
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_05 = active_idx[-1]
        speed_05 = float(np.linalg.norm(engine.flock.velocities[bird_v0_05]))

        # Second spawn should have lower velocity (bounded by new v0=0.5)
        assert speed_05 <= 0.51, (
            f"After v0→0.5, spawned speed {speed_05:.4f} should be ≤ 0.5"
        )
        # First spawn should have higher velocity (bounded by old v0=3.0)
        assert speed_3 <= 3.01

    def test_engine_spawn_uses_flock_rng(self, _engine):
        """Engine drain_commands passes flock.rng to spawn_at."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        # Two engines with same seed spawn same velocity
        cfg1 = SimConfig()
        cfg1.num_boids = 10
        cfg1.seed = 42
        e1 = SimulationEngine(cfg1)
        e1.enqueue_spawn((500, 350, 200))
        e1.drain_commands()

        cfg2 = SimConfig()
        cfg2.num_boids = 10
        cfg2.seed = 42
        e2 = SimulationEngine(cfg2)
        e2.enqueue_spawn((500, 350, 200))
        e2.drain_commands()

        # Both should produce the same velocity (same seed → same flock.rng state)
        a1 = np.where(e1.flock.active)[0][-1]
        a2 = np.where(e2.flock.active)[0][-1]
        np.testing.assert_array_equal(
            e1.flock.velocities[a1], e2.flock.velocities[a2],
            err_msg="Same seed must produce identical spawn velocity via engine"
        )

    def test_engine_spawn_different_seeds_diverge(self, _engine):
        """Different seeds → different spawn velocities through engine."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg1 = SimConfig()
        cfg1.num_boids = 10
        cfg1.seed = 42
        e1 = SimulationEngine(cfg1)
        e1.enqueue_spawn((500, 350, 200))
        e1.drain_commands()

        cfg2 = SimConfig()
        cfg2.num_boids = 10
        cfg2.seed = 99
        e2 = SimulationEngine(cfg2)
        e2.enqueue_spawn((500, 350, 200))
        e2.drain_commands()

        a1 = np.where(e1.flock.active)[0][-1]
        a2 = np.where(e2.flock.active)[0][-1]
        assert not np.array_equal(
            e1.flock.velocities[a1], e2.flock.velocities[a2],
        ), "Different seeds must produce different spawn velocities"

    def test_engine_multiple_spawns_in_one_drain(self, _engine):
        """Multiple enqueued spawns are all processed in one drain_commands."""
        engine, cfg = _engine
        n_before = engine.flock.N_active

        engine.enqueue_spawn((100, 200, 300))
        engine.enqueue_spawn((400, 500, 600))
        engine.enqueue_spawn((700, 800, 900))
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 3
        # Last 3 active birds should have the enqueued positions
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

    def test_engine_spawn_rng_advances_per_spawn(self, _engine):
        """Each spawn via engine advances flock.rng — two spawns differ."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.enqueue_spawn((500, 350, 200))  # same position
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        b1, b2 = active_idx[-2], active_idx[-1]

        assert not np.array_equal(
            engine.flock.velocities[b1], engine.flock.velocities[b2],
        ), "Two consecutive spawns through engine must differ (RNG advances)"

    def test_engine_spawn_updates_num_boids(self, _engine):
        """After drain_commands, config.num_boids reflects new N_active."""
        engine, cfg = _engine
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        assert cfg.num_boids == engine.flock.N_active


