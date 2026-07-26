"""Engine integration tests for priority_stack_enabled — the opt-in
obstacle > predator-threat > flocking priority-budget feature.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.physics.obstacles import ObstacleScene
from pymurmur.simulation.engine import SimulationEngine

ALL_MODES = ["spatial", "field", "projection", "vicsek", "angle", "influencer", "marl"]


class TestPriorityStackDisabledIsIdentical:
    """Disabled (default, and explicit False) must be byte-identical to
    never touching the flag at all — this is the feature's core safety
    guarantee, since the whole implementation hinges on it."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_disabled_matches_baseline(self, default_config, mode):
        cfg_baseline = default_config
        cfg_baseline.mode = mode
        cfg_baseline.num_boids = 30
        cfg_baseline.seed = 7

        cfg_disabled = default_config.__class__()
        cfg_disabled.mode = mode
        cfg_disabled.num_boids = 30
        cfg_disabled.seed = 7
        cfg_disabled.priority_stack_enabled = False

        eng_baseline = SimulationEngine(cfg_baseline)
        eng_disabled = SimulationEngine(cfg_disabled)

        for _ in range(10):
            eng_baseline.step()
            eng_disabled.step()

        np.testing.assert_array_equal(
            eng_baseline.flock.positions, eng_disabled.flock.positions,
        )
        np.testing.assert_array_equal(
            eng_baseline.flock.velocities, eng_disabled.flock.velocities,
        )


class TestPriorityStackCutoff:
    """When the obstacle force alone saturates the resolved budget,
    predator-threat and flocking must be entirely absent from that
    tick's result — not blended in at reduced strength."""

    @staticmethod
    def _isolated_bird_engine(default_config, mode, with_predator: bool):
        """Single bird, right next to a strong-avoidance obstacle,
        optionally with a predator very close by. num_boids=1 means
        there is no flocking-neighbour contribution to worry about, so
        tier3 is effectively zero already — isolating the predator's
        (tier2) contribution as the only variable between the two runs.
        """
        cfg = default_config.__class__()
        cfg.mode = mode
        cfg.num_boids = 1
        cfg.seed = 3
        cfg.priority_stack_enabled = True
        cfg.spatial.static_avoid_weight = 500.0
        cfg.spatial.predictive_avoid_weight = 0.0
        cfg.spatial.fly_away_max_dist = 200.0

        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine = SimulationEngine(cfg)
        engine.obstacle_scene = scene
        engine.flock.positions[0] = np.array([560.0, 500.0, 500.0], dtype=np.float32)  # SDF=10, outside surface
        engine.flock.velocities[0] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        if with_predator:
            cfg.predator_enabled = True
            cfg.predator.predator_mode = "cursor"
            cfg.predator.predator_threat_radius = 400.0
            cfg.predator.predator_strength = 50.0
            cfg._cursor_world_pos = np.array([491.0, 500.0, 500.0], dtype=np.float32)

        return engine

    @pytest.mark.parametrize("mode", ["spatial", "vicsek"])
    def test_obstacle_saturation_cuts_off_predator(self, default_config, mode):
        eng_no_pred = self._isolated_bird_engine(default_config, mode, with_predator=False)
        eng_with_pred = self._isolated_bird_engine(default_config, mode, with_predator=True)

        eng_no_pred._step_physics(1.0 / 60.0)
        eng_with_pred._step_physics(1.0 / 60.0)

        # Compare STEERING DIRECTION, not raw velocity magnitude: the
        # Predator extension's panic-ceiling speed boost (flock.max_speed)
        # is an intentional, separate side-channel NOT covered by tier-2
        # isolation (documented in the plan) — it legitimately raises the
        # bird's speed cap regardless of whether the steering force itself
        # got cut off. If the obstacle saturates the budget, the STEERING
        # (direction) must agree even though final speed may not.
        dir_no_pred = eng_no_pred.flock.velocities[0] / np.linalg.norm(eng_no_pred.flock.velocities[0])
        dir_with_pred = eng_with_pred.flock.velocities[0] / np.linalg.norm(eng_with_pred.flock.velocities[0])
        np.testing.assert_allclose(
            dir_no_pred, dir_with_pred,
            atol=1e-3,
            err_msg=(
                f"[{mode}] predator-threat steering leaked through despite "
                "obstacle saturating the budget — cutoff did not zero tier2"
            ),
        )


class TestPriorityStackMarlBudgetScale:
    """marl's native velocity deltas live on the v_cap scale (tens of
    units) — using max_force (0.15) as the budget would crush them
    ~50x. Confirms the per-mode budget resolution (marl_velocity_cap * U)
    keeps marl's dynamics on its native scale."""

    def test_marl_does_not_collapse_under_priority_stack(self, default_config):
        cfg_off = default_config
        cfg_off.mode = "marl"
        cfg_off.num_boids = 20
        cfg_off.seed = 11

        cfg_on = default_config.__class__()
        cfg_on.mode = "marl"
        cfg_on.num_boids = 20
        cfg_on.seed = 11
        cfg_on.priority_stack_enabled = True

        eng_off = SimulationEngine(cfg_off)
        eng_on = SimulationEngine(cfg_on)

        pos_off_start = eng_off.flock.positions.copy()
        pos_on_start = eng_on.flock.positions.copy()

        for _ in range(20):
            eng_off.step()
            eng_on.step()

        disp_off = np.linalg.norm(
            eng_off.flock.positions - pos_off_start, axis=1,
        ).mean()
        disp_on = np.linalg.norm(
            eng_on.flock.positions - pos_on_start, axis=1,
        ).mean()

        assert disp_on > disp_off * 0.1, (
            f"marl displacement collapsed under priority_stack: "
            f"disabled={disp_off:.3f}, enabled={disp_on:.3f} "
            "(budget scale mismatch would crush this ~50x)"
        )


class TestPriorityStackInfluencerDocumentedGap:
    """influencer (owns_positions=True) commits its position move
    inside compute_all_forces(), before the priority stack runs — so
    this tick's movement is NOT protected by tier1, only the exit
    velocity carried into next tick is. This test asserts the known
    gap exists (so a future fix is a deliberate change, not a silent
    regression discovery)."""

    def test_influencer_position_move_unprotected_this_tick(self, default_config):
        cfg_with_obstacle = default_config
        cfg_with_obstacle.mode = "influencer"
        cfg_with_obstacle.num_boids = 1
        cfg_with_obstacle.seed = 5
        cfg_with_obstacle.priority_stack_enabled = True
        cfg_with_obstacle.spatial.static_avoid_weight = 500.0
        cfg_with_obstacle.spatial.fly_away_max_dist = 200.0

        cfg_no_obstacle = default_config.__class__()
        cfg_no_obstacle.mode = "influencer"
        cfg_no_obstacle.num_boids = 1
        cfg_no_obstacle.seed = 5
        cfg_no_obstacle.priority_stack_enabled = True

        eng_obstacle = SimulationEngine(cfg_with_obstacle)
        eng_obstacle.obstacle_scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        eng_plain = SimulationEngine(cfg_no_obstacle)

        start_pos = np.array([560.0, 500.0, 500.0], dtype=np.float32)  # SDF=10, outside surface
        for eng in (eng_obstacle, eng_plain):
            eng.flock.positions[0] = start_pos.copy()
            eng.flock.velocities[0] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # heading toward obstacle

        eng_obstacle._step_physics(1.0 / 60.0)
        eng_plain._step_physics(1.0 / 60.0)

        # Documented gap: this tick's position ends up identical whether
        # or not an obstacle is present, because influencer's compute()
        # already committed the move before tier1 was computed.
        np.testing.assert_array_equal(
            eng_obstacle.flock.positions[0], eng_plain.flock.positions[0],
        )


class TestPriorityStackStabilitySweep:
    """Cross-mode stability sanity check under a combined obstacle +
    predator + flocking stress scenario — no NaN/inf, no runaway speed
    explosion, for every mode."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_no_nan_or_explosion(self, default_config, mode):
        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 40
        cfg.seed = 13
        cfg.priority_stack_enabled = True
        cfg.predator_enabled = True
        cfg.spatial.static_avoid_weight = 5.0
        cfg.spatial.fly_away_max_dist = 80.0

        engine = SimulationEngine(cfg)
        engine.obstacle_scene = ObstacleScene().add_sphere((500.0, 350.0, 200.0), 60.0)

        for _ in range(15):
            engine.step()

        assert np.all(np.isfinite(engine.flock.positions)), f"{mode}: non-finite positions"
        assert np.all(np.isfinite(engine.flock.velocities)), f"{mode}: non-finite velocities"
        speeds = np.linalg.norm(engine.flock.velocities[engine.flock.active], axis=1)
        # generous sanity bound — not a tight budget check (see
        # test_priority_stack.py for the exact function-level bound),
        # just guarding against unbounded blow-up
        assert speeds.max() < 10_000, f"{mode}: speed exploded to {speeds.max()}"
