"""Obstacle engine integration tests.

Split out of test_engine_pipeline.py (file-size split).
"""


import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine

# ═══════════════════════════════════════════════════════════════════════
# S6.4 Obstacle Engine Integration — "As a Whole"
# ═══════════════════════════════════════════════════════════════════════
# Exercises ObstacleScene collision detection, kinematic correction,
# avoidance steering, and per-step collision counter through the full
# engine pipeline.
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.s6_4
class TestObstacleEngineIntegration:
    """S6.4: ObstacleScene wired into SimulationEngine._step_physics().

    Verifies collision detection, kinematic correction, avoidance
    steering, and per-step collision counter published to metrics.
    """

    @staticmethod
    def _make_sphere_scene(center=(500.0, 500.0, 500.0), radius=100.0):
        """Build a single-sphere ObstacleScene for testing."""
        from pymurmur.physics.obstacles import ObstacleScene
        return ObstacleScene().add_sphere(center, radius)

    def test_obstacle_scene_resolves_collisions_in_pipeline(self, default_config):
        """S6.4: Birds that enter a sphere are corrected to the surface
        and collision count is published to metrics."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to reach surface quickly
        cfg.dt_phys = 1.0 / 60.0
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)

        # Place a small sphere obstacle at the centre
        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds just outside the sphere (~51 units from centre),
        # moving inward at high speed so they collide within 1 step
        rng = np.random.default_rng(42)
        for i in range(cfg.num_boids):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            engine.flock.positions[i] = np.array([500.0, 500.0, 500.0], dtype=np.float32) + direction.astype(np.float32) * 51.0
            engine.flock.velocities[i] = -direction.astype(np.float32) * cfg.v0

        # Step a few times — birds should collide with the sphere
        total_collisions = 0
        for _ in range(10):
            engine.step()
            total_collisions += engine.metrics.snapshot().collisions_this_step

        # At least some birds should have collided with the sphere
        assert total_collisions > 0, (
            f"Expected some collisions with sphere obstacle, got {total_collisions}"
        )
        assert scene.collision_count == total_collisions

        # After correction, no bird should be deep inside the sphere
        # (allow small penetration due to discrete timesteps, but not more
        # than half the sphere radius)
        positions = engine.flock.positions[engine.flock.active]
        dists = np.linalg.norm(positions - np.array([500.0, 500.0, 500.0]), axis=1)
        assert np.all(dists >= 50.0 * 0.5), (
            f"Birds penetrated too deep: min dist={dists.min():.1f}"
        )

    def test_obstacle_collision_counter_in_metrics_schema(self, default_config):
        """S6.4: collisions_this_step appears in FlockMetrics.to_dict()
        and is JSON-serializable."""
        import json

        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide in 1 step
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)
        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds heading toward centre, very close to surface
        for i in range(cfg.num_boids):
            engine.flock.positions[i] = np.array([448.0, 500.0, 500.0], dtype=np.float32)
            engine.flock.velocities[i] = np.array([cfg.v0, 0.0, 0.0], dtype=np.float32)

        engine.step()
        snap = engine.metrics.snapshot()

        # collisions_this_step is an int
        assert isinstance(snap.collisions_this_step, int), (
            f"Expected int, got {type(snap.collisions_this_step)}"
        )
        assert snap.collisions_this_step >= 0

        # JSON round-trip
        d = snap.to_dict()
        assert "collisions_this_step" in d
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert isinstance(restored["collisions_this_step"], int)
        assert restored["collisions_this_step"] == snap.collisions_this_step

    def test_obstacle_avoidance_reduces_collisions_over_time(self, default_config):
        """S6.4: With avoidance weights active, collision rate drops
        over successive steps as birds learn to steer away."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide quickly
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        # Enable avoidance
        cfg.spatial.static_avoid_weight = 2.0
        cfg.spatial.predictive_avoid_weight = 1.0
        cfg.spatial.fly_away_max_dist = 50.0
        cfg.spatial.min_time_to_collide = 2.0

        engine = SimulationEngine(cfg)

        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 60.0)
        engine.obstacle_scene = scene

        # Place birds in a ring heading toward centre, close to surface
        rng = np.random.default_rng(42)
        for i in range(cfg.num_boids):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            engine.flock.positions[i] = np.array([500.0, 500.0, 500.0], dtype=np.float32) + direction.astype(np.float32) * 62.0
            engine.flock.velocities[i] = -direction.astype(np.float32) * cfg.v0

        collisions_first_5 = 0
        collisions_last_5 = 0

        for step in range(30):
            engine.step()
            c = engine.metrics.snapshot().collisions_this_step
            if step < 5:
                collisions_first_5 += c
            elif step >= 25:
                collisions_last_5 += c

        # Avoidance should reduce collisions over time
        # (birds learn to steer away after initial collisions)
        # This is probabilistic but with strong weights should hold
        assert collisions_last_5 <= collisions_first_5, (
            f"Avoidance did not reduce collisions: "
            f"first 5 steps={collisions_first_5}, last 5 steps={collisions_last_5}"
        )

    def test_obstacle_scene_none_is_noop(self, default_config):
        """S6.4: When obstacle_scene is None, the pipeline runs
        normally with zero collisions."""
        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.boundary_mode = "sphere"
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)
        # No obstacle_scene set → should be None
        assert engine.obstacle_scene is None

        for _ in range(10):
            engine.step()

        snap = engine.metrics.snapshot()
        assert snap.collisions_this_step == 0, (
            f"No obstacle scene but collisions={snap.collisions_this_step}"
        )

    def test_obstacle_scene_empty_is_noop(self, default_config):
        """S6.4: An ObstacleScene with no shapes is a no-op."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.boundary_mode = "sphere"
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)
        engine.obstacle_scene = ObstacleScene()  # empty, no shapes

        for _ in range(5):
            engine.step()

        snap = engine.metrics.snapshot()
        assert snap.collisions_this_step == 0, (
            f"Empty scene should have zero collisions, got {snap.collisions_this_step}"
        )

    @pytest.mark.parametrize("mode", ["spatial", "angle", "projection"])
    def test_obstacle_scene_works_across_all_modes(self, default_config, mode):
        """S6.4: Obstacle scene works with spatial, angle, and projection modes."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide in 1 step
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        cfg.spatial.static_avoid_weight = 1.0
        cfg.spatial.fly_away_max_dist = 40.0

        engine = SimulationEngine(cfg)

        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds heading toward centre, very close to surface
        for i in range(cfg.num_boids):
            engine.flock.positions[i] = np.array([448.0, 500.0, 500.0], dtype=np.float32)
            engine.flock.velocities[i] = np.array([cfg.v0, 0.0, 0.0], dtype=np.float32)

        total_collisions = 0
        for _ in range(10):
            engine.step()
            total_collisions += engine.metrics.snapshot().collisions_this_step

        # All modes should detect collisions with the sphere
        assert total_collisions > 0, (
            f"{mode} mode: expected collisions, got {total_collisions}"
        )
        assert scene.collision_count == total_collisions

    def test_obstacle_avoidance_zero_weights_still_corrects(self, default_config):
        """S6.4: With zero avoidance weights, collisions are still
        detected and positions corrected (kinematic correction)."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide quickly
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        # Zero avoidance weights — only kinematic correction
        cfg.spatial.static_avoid_weight = 0.0
        cfg.spatial.predictive_avoid_weight = 0.0

        engine = SimulationEngine(cfg)

        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds heading straight toward centre, very close to surface
        for i in range(cfg.num_boids):
            engine.flock.positions[i] = np.array([448.0, 500.0, 500.0], dtype=np.float32)
            engine.flock.velocities[i] = np.array([cfg.v0, 0.0, 0.0], dtype=np.float32)

        collisions_found = 0
        for _ in range(15):
            engine.step()
            collisions_found += engine.metrics.snapshot().collisions_this_step

        # Collisions must be detected even without avoidance
        assert collisions_found > 0, (
            f"Zero-avoidance: expected collisions, got {collisions_found}"
        )
        assert scene.collision_count == collisions_found

        # After correction, positions should not be deep inside the sphere
        positions = engine.flock.positions[engine.flock.active]
        dists = np.linalg.norm(positions - np.array([500.0, 500.0, 500.0]), axis=1)
        assert np.all(dists >= 10.0), (
            f"Birds too deep: min dist={dists.min():.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Part V Cross-Item Integration — "As a Whole"
# ═══════════════════════════════════════════════════════════════════════
# Exercises combinations of S2.A5, S4.4a, S2.E6, S5.6, S6.1–S6.6, S4.10
# through the full engine pipeline.
# ═══════════════════════════════════════════════════════════════════════


