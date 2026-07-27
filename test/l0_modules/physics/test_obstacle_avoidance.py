"""Unit tests for physics.obstacle_avoidance — ObstacleAvoidanceStrategy
ABC, OBSTACLE_AVOIDANCE_REGISTRY, and dispatch-vs-direct-call equality
against ObstacleScene.avoidance_accel().

Modularity pass 3: formalises pymurmur's one obstacle-avoidance strategy
(SDF-gradient static fly-away + linear TTC predictive steering) behind
a registry mirroring ForceMode's proven pattern. These tests verify the
registry itself plus that the registry dispatch reproduces
avoidance_accel()'s direct-call behavior exactly (pure extraction, no
behavior change).
"""

from __future__ import annotations

import numpy as np

from pymurmur.physics.obstacle_avoidance import (
    OBSTACLE_AVOIDANCE_REGISTRY,
    ObstacleAvoidanceStrategy,
    SDFTTCStrategy,
)
from pymurmur.physics.obstacles import ObstacleScene


class TestObstacleAvoidanceRegistry:
    def test_sdf_ttc_registered(self):
        assert set(OBSTACLE_AVOIDANCE_REGISTRY.keys()) == {"sdf_ttc"}

    def test_registered_class_is_obstacle_avoidance_strategy_subclass(self):
        assert issubclass(OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"], ObstacleAvoidanceStrategy)

    def test_registry_maps_to_expected_class(self):
        assert OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"] is SDFTTCStrategy


class TestSDFTTCStrategyDispatchMatchesDirectCall:
    def test_static_and_predictive_weights_match_direct_call(self):
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[7.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)

        direct = scene.avoidance_accel(
            pos, vel, static_weight=1.0, predictive_weight=1.0,
            fly_away_max_dist=5.0, min_time_to_collide=5.0,
        )
        via_registry = OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"].compute_accel(
            scene, pos, vel, static_weight=1.0, predictive_weight=1.0,
            fly_away_max_dist=5.0, min_time_to_collide=5.0,
        )
        np.testing.assert_array_equal(direct, via_registry)

    def test_static_only_matches_direct_call(self):
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[6.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)

        direct = scene.avoidance_accel(
            pos, vel, static_weight=1.0, predictive_weight=0.0,
            fly_away_max_dist=5.0,
        )
        via_registry = OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"].compute_accel(
            scene, pos, vel, static_weight=1.0, predictive_weight=0.0,
            fly_away_max_dist=5.0,
        )
        np.testing.assert_array_equal(direct, via_registry)

    def test_predictive_only_matches_direct_call(self):
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[7.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)

        direct = scene.avoidance_accel(
            pos, vel, static_weight=0.0, predictive_weight=1.0,
            fly_away_max_dist=5.0, min_time_to_collide=5.0,
        )
        via_registry = OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"].compute_accel(
            scene, pos, vel, static_weight=0.0, predictive_weight=1.0,
            fly_away_max_dist=5.0, min_time_to_collide=5.0,
        )
        np.testing.assert_array_equal(direct, via_registry)

    def test_zero_weights_noop_matches_direct_call(self):
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[6.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)

        direct = scene.avoidance_accel(pos, vel)
        via_registry = OBSTACLE_AVOIDANCE_REGISTRY["sdf_ttc"].compute_accel(scene, pos, vel)
        np.testing.assert_array_equal(direct, via_registry)
        assert np.all(via_registry == 0.0)
