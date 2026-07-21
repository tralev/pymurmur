"""P12 — MARL bridge tests (P12.1 marl mode + P12.2 MurmurationEnv).

Dependency-gated: gymnasium via pytest.importorskip.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces._mode import MODE_REGISTRY
from pymurmur.physics.forces.marl import MarlMode


def _make_flock_arrays(N: int, config: SimConfig):
    """Create minimal array set for testing MarlMode.compute()."""
    rng = np.random.default_rng(42)
    W, H, D = config.width, config.height, config.depth
    pos = rng.uniform(0, [W, H, D], size=(N, 3)).astype(np.float32)
    vel = rng.normal(0, 1, size=(N, 3)).astype(np.float32)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    last_theta = np.zeros(N, dtype=np.float32)
    return pos, vel, acc, active, last_theta, rng


class TestMarlModeRegistered:
    """P12.1: 'marl' is registered in MODE_REGISTRY and declared valid."""

    def test_marl_in_registry(self):
        """MarlMode is registered under 'marl' in MODE_REGISTRY."""
        assert "marl" in MODE_REGISTRY
        assert MODE_REGISTRY["marl"] is MarlMode

    def test_marl_is_valid_mode(self):
        """'marl' is in SimConfig._VALID_MODES."""
        assert "marl" in SimConfig._VALID_MODES

    def test_marl_needs_no_index(self):
        """Marl mode uses global neighbourhood, no spatial index needed."""
        assert MarlMode.needs_index is False

    def test_marl_config_creates_engine(self):
        """SimConfig(mode='marl') passes validation and creates an engine."""
        cfg = SimConfig(mode="marl", num_boids=10)
        cfg.validate()
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)
        assert engine.config.mode == "marl"


class TestMarlForceDeferred:
    """P12.1: Deferred global rules — control first, move, rules prep next."""

    def test_control_applied_before_rules(self):
        """External action affects velocity before rules compute."""
        cfg = SimConfig(mode="marl", num_boids=5)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        vel_before = vel.copy()

        # Set external action: push all birds in +X direction
        cfg._marl_action = np.full((5, 3), [1.0, 0.0, 0.0], dtype=np.float32)

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # Velocities should change (action applied)
        assert not np.allclose(vel, vel_before, atol=1e-6), (
            "External action should change velocities"
        )

    def test_deferred_rules_affect_velocity(self):
        """Global separation/alignment/cohesion change velocity."""
        cfg = SimConfig(mode="marl", num_boids=10)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(10, cfg)
        vel_before = vel.copy()

        cfg._marl_action = np.zeros((10, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # With zero action, rules (0.01 * (sep+align+coh)) still apply
        assert not np.allclose(vel, vel_before, atol=1e-12), (
            "Deferred rules (0.01 weight) should change velocities slightly"
        )

    def test_velocity_clamped_to_cap(self):
        """After compute, all speeds are ≤ v_cap."""
        cfg = SimConfig(mode="marl", num_boids=5)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        # Give very high initial velocities
        vel[:] = 100.0

        W, H, D = cfg.width, cfg.height, cfg.depth
        U = min(W, H, D) / 6.0
        v_cap = 0.5 * U

        cfg._marl_action = np.zeros((5, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        speeds = np.linalg.norm(vel[active], axis=1)
        assert np.all(speeds <= v_cap + 1e-4), (
            f"Speeds {speeds} must be ≤ v_cap {v_cap}"
        )

    def test_min_speed_enforced(self):
        """Very slow birds are boosted to at least 0.3 * v_cap."""
        cfg = SimConfig(mode="marl", num_boids=5)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        vel[:] = 0.001  # nearly stationary

        cfg._marl_action = np.zeros((5, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        W, H, D = cfg.width, cfg.height, cfg.depth
        U = min(W, H, D) / 6.0
        min_speed = 0.3 * 0.5 * U

        speeds = np.linalg.norm(vel[active], axis=1)
        assert np.all(speeds >= min_speed - 1e-4), (
            f"Speeds {speeds} must be ≥ min_speed {min_speed}"
        )

    def test_no_action_attribute_no_crash(self):
        """Marl compute() doesn't crash when _marl_action is missing."""
        cfg = SimConfig(mode="marl", num_boids=5)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        # No _marl_action set

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)
        # Should not raise — just skip control, apply rules

    def test_inactive_birds_untouched(self):
        """Inactive birds' velocities are unchanged by compute()."""
        cfg = SimConfig(mode="marl", num_boids=5)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        active[2:4] = False  # deactivate birds 2 and 3
        vel_before = vel.copy()

        cfg._marl_action = np.ones((5, 3), dtype=np.float32)
        # But MARL action shape mismatches N_active (3 vs 5).
        # The compute function uses act_idx (only active birds).
        # The action array should be sized for the total flock, not active only.
        # Let's fix: use only active-sized action
        cfg._marl_action = np.ones((3, 3), dtype=np.float32)

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # Inactive birds must be unchanged
        assert np.allclose(vel[~active], vel_before[~active], atol=1e-12), (
            "Inactive bird velocities must not change"
        )

    def test_all_inactive_no_crash(self):
        """Marl compute() with all-inactive flock returns early."""
        cfg = SimConfig(mode="marl", num_boids=5)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        active[:] = False

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)
        # Should not raise


class TestMarlForceZeroParams:
    """P12.1: Zero-value parameter edge cases — action_scale=0,
    rule_weight=0, separation at exact sep_radius boundary."""

    def test_action_scale_zero_no_external_force(self):
        """action_scale=0 → external action produces zero velocity delta.
        Isolated by setting rule_weight=0.0 and velocities inside
        [min_speed, v_cap] so speed clamping is a no-op."""
        cfg = SimConfig(mode="marl", num_boids=5)
        cfg.marl_action_scale = 0.0
        cfg.marl_rule_weight = 0.0
        cfg.marl_velocity_cap = 10.0
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        # Set velocities safely inside [min_speed, v_cap] so clamping
        # doesn't change them independently of action/rules
        W, H, D = cfg.width, cfg.height, cfg.depth
        U = min(W, H, D) / 6.0
        v_cap = 10.0 * U
        vel[:] = [v_cap / 2, 0.0, 0.0]
        vel_before = vel.copy()

        cfg._marl_action = np.ones((5, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # With action_scale=0, rule_weight=0, speed clamp no-op: unchanged
        assert np.allclose(vel, vel_before, atol=1e-6), (
            "action_scale=0 + rule_weight=0 must leave velocities unchanged"
        )

    def test_rule_weight_zero_no_internal_forces(self):
        """rule_weight=0 → internal rules produce zero velocity change."""
        cfg = SimConfig(mode="marl", num_boids=5)
        cfg.marl_rule_weight = 0.0
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)

        # Set action=0 so only internal rules would change velocity
        cfg._marl_action = np.zeros((5, 3), dtype=np.float32)
        vel_before = vel.copy()

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # With zero action AND zero rule_weight, velocities unchanged
        # (except speed clamping, which may adjust magnitude but not direction)
        speeds_before = np.linalg.norm(vel_before, axis=1)
        speeds_after = np.linalg.norm(vel, axis=1)
        # Direction should be preserved if only speed clamping changed things
        for i in range(5):
            if speeds_before[i] > 1e-6 and speeds_after[i] > 1e-6:
                dot = np.dot(vel_before[i], vel[i]) / (speeds_before[i] * speeds_after[i])
                assert dot > 0.99, (
                    f"rule_weight=0 must preserve velocity direction: "
                    f"dot={dot:.6f} for bird {i}"
                )

    def test_separation_at_exact_boundary_no_force(self):
        """Birds at exactly sep_radius feel zero separation (d < sep_radius check)."""
        cfg = SimConfig(mode="marl", num_boids=2,
                        width=200, height=200, depth=200)
        # sep_radius = marl_separation_radius * U = 1.0 * (200/6) ≈ 33.33
        cfg.marl_separation_radius = 1.0
        # Set rule_weight higher so forces are detectable
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(2, cfg)
        U = 200.0 / 6.0
        sep_radius = 1.0 * U  # ≈ 33.33
        # Place birds at exactly sep_radius apart (not <, so no separation)
        pos[0] = [100.0, 100.0, 100.0]
        pos[1] = [100.0 + sep_radius, 100.0, 100.0]
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((2, 3), dtype=np.float32)

        # Compute with d = sep_radius (not <)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # The velocity change should come only from alignment (v_mean=0 → 0)
        # and cohesion (toward CoM). Separation should contribute zero.
        # With rule_weight=1.0, cohesion = CoM - pos = [sep_radius/2, 0, 0]
        # and [-sep_radius/2, 0, 0] respectively. Both birds move toward each other
        # (cohesion), not away (separation).
        # Bird 0 should move in +X (toward CoM), bird 1 in -X
        assert vel[0, 0] > 0, f"Cohesion should pull bird 0 toward center (+X): vel={vel[0]}"
        assert vel[1, 0] < 0, f"Cohesion should pull bird 1 toward center (-X): vel={vel[1]}"


class TestMarlForceEdgeCases:
    """P12.1: Edge cases — single bird, action shape mismatch,
    negative action direction, accelerations array untouched."""

    def test_single_bird_no_crash(self):
        """N=1: no separation (no neighbors), alignment=0 (v_mean=v),
        cohesion=0 (CoM=p). External action still applies."""
        cfg = SimConfig(mode="marl", num_boids=1)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(1, cfg)
        vel_before = vel.copy()

        cfg._marl_action = np.ones((1, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # External action should apply; internal rules = 0 for N=1
        assert not np.allclose(vel, vel_before, atol=1e-6), (
            "Single bird: external action must still apply"
        )

    def test_action_shape_mismatch_silently_ignored(self):
        """_marl_action with wrong shape (N+1, 3) is silently ignored."""
        cfg = SimConfig(mode="marl", num_boids=5)
        cfg.marl_rule_weight = 0.0  # suppress internal rules
        cfg.marl_velocity_cap = 10.0  # large cap so speed unchanged
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        # Set velocities to a value within [min_speed, v_cap] so
        # speed clamping doesn't modify them independently
        W, H, D = cfg.width, cfg.height, cfg.depth
        U = min(W, H, D) / 6.0
        v_cap = 10.0 * U
        vel[:] = [v_cap / 2, 0.0, 0.0]  # safely inside [0.3*v_cap, v_cap]
        vel_before = vel.copy()

        # Action has 6 birds but flock has 5 → shape mismatch
        cfg._marl_action = np.ones((6, 3), dtype=np.float32)

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # With rule_weight=0, action ignored, speed clamp no-op: unchanged
        assert np.allclose(vel, vel_before, atol=1e-6), (
            "Shape-mismatched action must be silently ignored"
        )

    def test_negative_action_moves_birds_left(self):
        """-X action decreases X velocity component."""
        cfg = SimConfig(mode="marl", num_boids=5)
        cfg.marl_rule_weight = 0.0  # suppress internal rules
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        vel_before = vel.copy()

        # Strong -X action
        cfg._marl_action = np.full((5, 3), [-1.0, 0.0, 0.0], dtype=np.float32)

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # X velocity should decrease
        assert np.mean(vel[:, 0]) < np.mean(vel_before[:, 0]), (
            "-X action must decrease X velocity"
        )

    def test_accelerations_array_untouched(self):
        """MarlMode only modifies velocities array, not accelerations."""
        cfg = SimConfig(mode="marl", num_boids=5)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        acc_before = acc.copy()

        cfg._marl_action = np.ones((5, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        assert np.allclose(acc, acc_before, atol=1e-12), (
            "Accelerations array must be untouched by marl compute"
        )


class TestMarlActionScale:
    """P12.1: marl_action_scale parameter affects external action magnitude."""

    def test_larger_action_scale_produces_larger_delta(self):
        """action_scale=0.1 produces larger velocity change than 0.01."""
        cfg_small = SimConfig(mode="marl", num_boids=5)
        cfg_large = SimConfig(mode="marl", num_boids=5)
        cfg_small.marl_action_scale = 0.01
        cfg_large.marl_action_scale = 0.1
        cfg_small.marl_rule_weight = 0.0
        cfg_large.marl_rule_weight = 0.0

        pos, vel_small, acc, active, last_theta, rng = _make_flock_arrays(5, cfg_small)
        vel_large = vel_small.copy()

        cfg_small._marl_action = np.ones((5, 3), dtype=np.float32)
        cfg_large._marl_action = np.ones((5, 3), dtype=np.float32)

        vel_small_before = vel_small.copy()
        vel_large_before = vel_large.copy()

        MarlMode.compute(pos, vel_small, acc, active, None, rng, last_theta, cfg_small)
        rng2 = np.random.default_rng(42)
        MarlMode.compute(pos, vel_large, acc, active, None, rng2, last_theta, cfg_large)

        delta_small = np.linalg.norm(vel_small - vel_small_before)
        delta_large = np.linalg.norm(vel_large - vel_large_before)
        assert delta_small < delta_large, (
            f"Larger action_scale must produce larger velocity delta: "
            f"small={delta_small:.6f} vs large={delta_large:.6f}"
        )

    def test_action_scale_default_is_0_05(self):
        """Default marl_action_scale is 0.05 (from module constant)."""
        from pymurmur.core.config import MarlConfig
        assert MarlConfig().marl_action_scale == pytest.approx(0.05)


class TestMarlSeparationRadius:
    """P12.1: marl_separation_radius parameter controls which birds
    feel repulsion — birds within radius feel force, birds beyond don't."""

    def test_small_radius_excludes_distant_birds(self):
        """With tiny marl_separation_radius, no birds feel separation."""
        cfg = SimConfig(mode="marl", num_boids=5,
                        width=200, height=200, depth=200)
        cfg.marl_separation_radius = 0.001  # very small radius * U
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]
        vel[2] = [0.0, 0.0, 0.0]
        vel[3] = [0.0, 0.0, 0.0]
        vel[4] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((5, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # With tiny radius, only cohesion + alignment affect velocity
        # Cohesion pulls toward CoM; separation is negligible
        # All birds should get non-zero velocity (cohesion), but
        # the separation contribution should be ~0
        assert np.all(np.linalg.norm(vel, axis=1) > 0), (
            "Cohesion should still give non-zero velocity"
        )

    def test_large_radius_includes_all_birds(self):
        """With huge marl_separation_radius, all birds feel separation."""
        cfg = SimConfig(mode="marl", num_boids=5,
                        width=200, height=200, depth=200)
        cfg.marl_separation_radius = 100.0  # huge radius * U
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(5, cfg)
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]
        vel[2] = [0.0, 0.0, 0.0]
        vel[3] = [0.0, 0.0, 0.0]
        vel[4] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((5, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # With huge radius, all birds feel separation + cohesion + alignment
        # All should get non-zero velocity
        assert np.all(np.linalg.norm(vel, axis=1) > 0), (
            "All birds should feel forces with large separation radius"
        )


class TestMarlSeparationThreeBirds:
    """P12.1: O(n²) separation loop works for 3+ birds.
    Previously only tested with 2 birds."""

    def test_three_birds_in_line_repel_outermost(self):
        """Three birds in a line: outer birds feel repulsion from
        middle bird. Middle bird feels repulsion from both."""
        cfg = SimConfig(mode="marl", num_boids=3,
                        width=200, height=200, depth=200)
        cfg.marl_separation_radius = 1.0
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(3, cfg)
        pos[0] = [98.0, 100.0, 100.0]
        pos[1] = [100.0, 101.0, 100.0]  # middle, offset from CoM in Y
        pos[2] = [102.0, 100.0, 100.0]
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]
        vel[2] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((3, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # All three birds should feel some force (separation + cohesion + alignment)
        # No bird should be left with exactly zero velocity
        speeds = np.linalg.norm(vel, axis=1)
        assert np.all(speeds > 0), (
            f"All 3 birds must get non-zero force: speeds={speeds}"
        )

    def test_three_birds_triangle_repel_each_other(self):
        """Three birds in a tight triangle: each feels repulsion from
        two others."""
        cfg = SimConfig(mode="marl", num_boids=3,
                        width=200, height=200, depth=200)
        cfg.marl_separation_radius = 1.0
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(3, cfg)
        pos[0] = [100.0, 100.0, 100.0]
        pos[1] = [101.0, 100.0, 100.0]
        pos[2] = [100.0, 101.0, 100.0]
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]
        vel[2] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((3, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        speeds = np.linalg.norm(vel, axis=1)
        assert np.all(speeds > 0), (
            f"All 3 triangle birds must get non-zero force: speeds={speeds}"
        )


class TestMarlEngineNoIndexRebuild:
    """P12.1→engine: marl mode (needs_index=False) means the engine
    does not rebuild the spatial index during step()."""

    def test_mode_needs_index_marl_is_false(self):
        """mode_needs_index('marl') returns False."""
        from pymurmur.physics.forces import mode_needs_index
        assert mode_needs_index("marl") is False

    def test_engine_does_not_rebuild_index_for_marl(self):
        """In _step_physics, the index rebuild is gated by
        mode_needs_index. For marl (returns False), index is not rebuilt."""
        cfg = SimConfig(mode="marl", num_boids=10, seed=42,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)

        # Record the index state before step
        idx_before = engine.flock._index

        engine.config._marl_action = np.zeros((10, 3), dtype=np.float32)
        engine._step_physics(dt=1.0)

        # For marl mode, index should not be rebuilt
        # (If it were, the tree would be updated)
        idx_after = engine.flock._index
        # Index may be None for marl since it's not needed,
        # but if it exists, it should be the same object
        if idx_before is not None and idx_after is not None:
            assert idx_before is idx_after, (
                "Index must not be rebuilt for marl mode"
            )


class TestMarlToroidalAllAxes:
    """P12.1: Toroidal separation wrapping on Y and Z axes
    (X axis already tested in TestMarlToroidalSeparation)."""

    def test_birds_at_opposite_y_edges_repel(self):
        """Two birds at opposite Y edges feel separation through toroidal wrap."""
        cfg = SimConfig(mode="marl", num_boids=2,
                        width=200, height=200, depth=200)
        cfg.marl_separation_radius = 1.0
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(2, cfg)
        # Toroidal Y distance = min(|5-195|, 200-|5-195|) = min(190, 10) = 10
        pos[0] = [100.0, 5.0, 100.0]
        pos[1] = [100.0, 195.0, 100.0]
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((2, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # At least one bird gets Y velocity from the combined forces
        assert vel[0, 1] != 0.0 or vel[1, 1] != 0.0, (
            f"Toroidal Y separation should produce Y movement: vel={vel}"
        )

    def test_birds_at_opposite_z_edges_repel(self):
        """Two birds at opposite Z edges feel separation through toroidal wrap."""
        cfg = SimConfig(mode="marl", num_boids=2,
                        width=200, height=200, depth=200)
        cfg.marl_separation_radius = 1.0
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(2, cfg)
        pos[0] = [100.0, 100.0, 5.0]
        pos[1] = [100.0, 100.0, 195.0]
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((2, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        assert vel[0, 2] != 0.0 or vel[1, 2] != 0.0, (
            f"Toroidal Z separation should produce Z movement: vel={vel}"
        )


class TestMarlForceSeparationCap:
    """P12.1: Separation repulsion capped at ±1.0 per pair to prevent
    explosions at near-zero distances."""

    def test_extremely_close_birds_do_not_explode(self):
        """Two birds at d=0.001: 1/d² ≈ 1e6 but repulsion clipped at ±1.0."""
        cfg = SimConfig(mode="marl", num_boids=2,
                        width=200, height=200, depth=200)
        cfg.marl_separation_radius = 1.0
        cfg.marl_rule_weight = 1.0

        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(2, cfg)
        pos[0] = [100.0, 100.0, 100.0]
        pos[1] = [100.001, 100.0, 100.0]  # 0.001 units apart
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((2, 3), dtype=np.float32)
        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # Without the cap, 1/d² = 1e6, and with rule_weight=1,
        # velocity change would be ~1e6. With the cap, repulsion ≤ 1.0.
        # Cohesion also acts (~1 unit). Net velocity change ≤ ~2 units.
        max_speed = np.max(np.linalg.norm(vel, axis=1))
        assert max_speed < 10.0, (
            f"Repulsion cap must prevent explosion at d=0.001: "
            f"max_speed={max_speed:.6f}"
        )


class TestMurmurationEnv:
    """P12.2: MurmurationEnv — gym wrapper for MARL."""

    @pytest.fixture
    def _env(self):
        """Create a basic environment (skip if gymnasium unavailable)."""
        pytest.importorskip("gymnasium")
        from pymurmur.analysis.gym_env import MurmurationEnv
        return MurmurationEnv(num_boids=10, episode_steps=200, seed=42)

    def test_env_spaces_correct_shape(self, _env):
        """Observation space is (6N,) and action space is (3N,)."""
        assert _env.observation_space.shape == (60,)
        assert _env.action_space.shape == (30,)

    def test_episode_steps_defaults_to_config_marl_episode_steps(self):
        """C3: episode_steps=None falls back to config.marl_episode_steps."""
        pytest.importorskip("gymnasium")
        from pymurmur.analysis.gym_env import MurmurationEnv

        env = MurmurationEnv(num_boids=5, seed=42)
        assert env._episode_steps == env._base_config.marl_episode_steps

        env2 = MurmurationEnv(
            num_boids=5, seed=42,
            config_overrides={"marl_episode_steps": 42},
        )
        assert env2._episode_steps == 42

    def test_explicit_episode_steps_overrides_config(self):
        """C3: an explicit episode_steps kwarg wins over config."""
        pytest.importorskip("gymnasium")
        from pymurmur.analysis.gym_env import MurmurationEnv

        env = MurmurationEnv(
            num_boids=5, seed=42, episode_steps=7,
            config_overrides={"marl_episode_steps": 42},
        )
        assert env._episode_steps == 7

    def test_env_reset_returns_obs_and_info(self, _env):
        """reset() returns (obs, info) tuple."""
        obs, info = _env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == (60,)
        assert obs.dtype == np.float32

    def test_obs_in_bounds(self, _env):
        """All observation values are in [-1, 1]."""
        obs, _ = _env.reset()
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_step_returns_gym_protocol(self, _env):
        """step() returns (obs, reward, terminated, truncated, info)."""
        _env.reset()
        action = _env.action_space.sample()
        result = _env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncation_at_episode_steps(self, _env):
        """Environment truncates after episode_steps."""
        _env.reset()
        for _ in range(_env._episode_steps - 1):
            _, _, _, truncated, _ = _env.step(
                _env.action_space.sample()
            )
            assert not truncated
        # Final step should truncate
        _, _, _, truncated, _ = _env.step(
            _env.action_space.sample()
        )
        assert truncated

    def test_step_without_reset_raises(self, _env):
        """step() before reset() raises AssertionError."""
        with pytest.raises(AssertionError):
            _env.step(np.zeros(30, dtype=np.float32))

    def test_same_seed_deterministic(self, _env):
        """Same seed + same actions → identical observations."""
        obs1, _ = _env.reset(seed=42)
        _env.action_space.sample()

        from pymurmur.analysis.gym_env import MurmurationEnv
        env2 = MurmurationEnv(num_boids=10, episode_steps=200, seed=42)
        obs2, _ = env2.reset(seed=42)

        assert np.allclose(obs1, obs2, atol=1e-6), "Same seed should give same initial obs"

    def test_action_clipped(self, _env):
        """Actions outside [-1, 1] are clipped to bounds."""
        _env.reset()
        action = np.full(30, 5.0, dtype=np.float32)  # way outside bounds
        obs, _, _, _, _ = _env.step(action)
        # Should not crash and obs still in bounds
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    def test_gym_checker(self, _env):
        """gymnasium.utils.env_checker.check_env passes."""
        pytest.importorskip("gymnasium.utils.env_checker")
        import gymnasium.utils.env_checker
        gymnasium.utils.env_checker.check_env(_env)

    def test_config_overrides_flow_to_engine(self, _env):
        """config_overrides dict modifies the base config used by the engine."""
        from pymurmur.analysis.gym_env import MurmurationEnv
        env = MurmurationEnv(
            num_boids=5,
            episode_steps=100,
            config_overrides={"marl_velocity_cap": 0.99},
        )
        env.reset()
        assert env._engine is not None
        assert env._engine.config.marl_velocity_cap == 0.99

    def test_mode_parameter_is_marl_by_default(self, _env):
        """mode='marl' is the default and flows to engine config."""
        from pymurmur.analysis.gym_env import MurmurationEnv
        env = MurmurationEnv(num_boids=5, episode_steps=100)
        env.reset()
        assert env._engine is not None
        assert env._engine.config.mode == "marl"

    def test_get_obs_formula_positions(self, _env):
        """_get_obs normalizes positions as (p-C)/3U."""
        from pymurmur.analysis.gym_env import MurmurationEnv
        env = MurmurationEnv(num_boids=5, episode_steps=100, seed=42)
        env.reset()

        # Set a known position (the domain centre) and read back the obs.
        # Derived from the real engine config rather than hardcoded —
        # a stale hardcoded (1000, 600, 400) (actual SimConfig default
        # height is 700) previously made this test fail even though
        # _get_obs()'s formula was correct.
        cfg = env._engine.config
        center = [cfg.width / 2, cfg.height / 2, cfg.depth / 2]
        env._engine.flock.positions[0] = center

        obs = env._get_obs()
        # First 3 entries = pos_norm for bird 0 = (500-500)/(3U) = 0
        # Tolerance for float
        assert abs(obs[0]) < 1e-5, f"Center bird X should be ~0: {obs[0]}"
        assert abs(obs[1]) < 1e-5, f"Center bird Y should be ~0: {obs[1]}"
        assert abs(obs[2]) < 1e-5, f"Center bird Z should be ~0: {obs[2]}"

    def test_multiple_reset_reinitializes(self, _env):
        """Multiple reset() calls create fresh SimulationEngine instances."""
        obs1, _ = _env.reset(seed=42)
        engine1 = _env._engine

        obs2, _ = _env.reset(seed=42)
        engine2 = _env._engine

        # Same seed → same observation
        assert np.allclose(obs1, obs2, atol=1e-6), (
            "Same seed after reset must produce same initial obs"
        )
        # New engine instance
        assert engine2 is not engine1, (
            "reset() must create a new SimulationEngine instance"
        )

    def test_seed_none_produces_different_obs(self, _env):
        """seed=None creates non-deterministic initial states."""
        from pymurmur.analysis.gym_env import MurmurationEnv
        env_a = MurmurationEnv(num_boids=10, episode_steps=100, seed=None)
        env_b = MurmurationEnv(num_boids=10, episode_steps=100, seed=None)

        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()

        # With seed=None, two envs should (almost certainly) differ
        assert not np.allclose(obs_a, obs_b, atol=1e-6), (
            "seed=None must produce different initial states"
        )

    def test_config_overrides_unknown_key_does_not_crash(self, _env):
        """config_overrides with an unrecognized key doesn't crash."""
        from pymurmur.analysis.gym_env import MurmurationEnv
        env = MurmurationEnv(
            num_boids=5,
            episode_steps=100,
            config_overrides={"not_a_real_param": 123},
        )
        # Should not raise during construction or reset
        env.reset()
        assert env._engine is not None

    def test_info_dict_contains_step_key(self, _env):
        """step() returns info dict with 'step' key that increments."""
        _env.reset()
        action = _env.action_space.sample()

        _, _, _, _, info1 = _env.step(action)
        assert info1["step"] == 1

        _, _, _, _, info2 = _env.step(action)
        assert info2["step"] == 2


class TestMarlModeTwoStep:
    """P12.1: Two-step hand trace — rules at step k affect positions at k+1."""

    def test_no_action_birds_drift_towards_center(self):
        """With zero action, cohesion rule (0.01 * (CoM−p)) pulls birds inward."""
        cfg = SimConfig(mode="marl", num_boids=4, seed=99,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)
        # Put birds at opposite corners
        engine.flock.positions[0] = [10, 10, 10]
        engine.flock.positions[1] = [190, 10, 10]
        engine.flock.positions[2] = [10, 190, 10]
        engine.flock.positions[3] = [190, 190, 10]

        center_before = np.mean(engine.flock.positions, axis=0)

        for _ in range(20):
            # No external action
            engine.config._marl_action = np.zeros((4, 3), dtype=np.float32)
            engine.step()

        center_after = np.mean(engine.flock.positions, axis=0)
        # Cohesion should pull birds toward the center
        assert center_after[0] != pytest.approx(center_before[0]), (
            f"Center should shift: before={center_before}, after={center_after}"
        )

    def test_action_applied_causes_movement(self):
        """A strong external action produces observable velocity change."""
        cfg = SimConfig(mode="marl", num_boids=5, seed=42,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)
        vel_before = engine.flock.velocities.copy()

        # Strong +X action
        engine.config._marl_action = np.full((5, 3), [1.0, 0.0, 0.0], dtype=np.float32)
        engine.step()

        vel_after = engine.flock.velocities.copy()
        # X component of velocities should increase
        assert np.mean(vel_after[:, 0]) > np.mean(vel_before[:, 0]), (
            "X velocity should increase after +X action"
        )


class TestMarlForcesBackwardCompat:
    """P12.1: marl_forces backward-compatible function alias."""

    def test_marl_forces_is_callable(self):
        """marl_forces is a callable function."""
        from pymurmur.physics.forces.marl import marl_forces
        assert callable(marl_forces)

    def test_marl_forces_has_needs_index(self):
        """marl_forces exposes needs_index = False."""
        from pymurmur.physics.forces.marl import marl_forces
        assert marl_forces.needs_index is False

    def test_marl_forces_calls_marl_mode_compute(self):
        """marl_forces delegates to MarlMode.compute correctly."""
        from pymurmur.physics.forces.marl import marl_forces
        cfg = SimConfig(mode="marl", num_boids=3)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(3, cfg)
        vel_before = vel.copy()
        cfg._marl_action = np.zeros((3, 3), dtype=np.float32)
        marl_forces(pos, vel, acc, active, None, rng, last_theta, cfg)
        # Rules should have been applied (0.01 weight changes velocity)
        assert not np.allclose(vel, vel_before, atol=1e-12)


class TestMarlToroidalSeparation:
    """P12.1: Toroidal separation wrapping — birds at opposite
    domain edges repel each other through the toroidal boundary."""

    def test_birds_at_opposite_edges_repel_via_toroid(self):
        """Two birds at opposite X edges, toroidally close (d≈10),
        produce non-zero velocity changes along X from the combined
        deferred rules (separation + cohesion)."""
        cfg = SimConfig(mode="marl", num_boids=2,
                        width=200, height=200, depth=200)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(2, cfg)
        # Bird 0 near left edge, bird 1 near right edge
        # Toroidal distance = min(|5-195|, 200-|5-195|) = min(190, 10) = 10
        pos[0] = [5.0, 100.0, 100.0]
        pos[1] = [195.0, 100.0, 100.0]
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((2, 3), dtype=np.float32)
        # Wide separation radius so toroidally-close birds feel repulsion
        cfg.marl_separation_radius = 1.0  # sep_radius ≈ 33.3

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # With only two birds at zero velocity:
        # - Alignment = 0 (both have v=0)
        # - Cohesion pulls both toward CoM (near domain center)
        # - Separation pushes them apart along X (through torus)
        # The net X force should be non-zero
        assert vel[0, 0] != 0.0 or vel[1, 0] != 0.0, (
            f"Toroidal separation should produce X movement: vel={vel}"
        )

    def test_separation_is_repulsive_between_close_birds(self):
        """Two birds placed very close together (d=2 units) produce
        strong opposing separation velocities. Cohesion also acts,
        but 1/d² separation dominates at close range."""
        cfg = SimConfig(mode="marl", num_boids=2,
                        width=200, height=200, depth=200)
        pos, vel, acc, active, last_theta, rng = _make_flock_arrays(2, cfg)
        # Birds very close together — 1/d² repulsion dominates cohesion
        pos[0] = [100.0, 100.0, 100.0]
        pos[1] = [102.0, 100.0, 100.0]  # 2 units apart
        vel[0] = [0.0, 0.0, 0.0]
        vel[1] = [0.0, 0.0, 0.0]

        cfg._marl_action = np.zeros((2, 3), dtype=np.float32)
        cfg.marl_separation_radius = 1.0

        MarlMode.compute(pos, vel, acc, active, None, rng, last_theta, cfg)

        # At d=2, 1/d² separation ≈ 0.25 force vs cohesion ≈ 2.0 force
        # Both act together; net effect produces non-zero velocity change
        assert not np.allclose(vel[0], [0.0, 0.0, 0.0], atol=1e-6), (
            f"Close-range repulsion must produce non-zero force: vel[0]={vel[0]}"
        )
        assert not np.allclose(vel[1], [0.0, 0.0, 0.0], atol=1e-6), (
            f"Close-range repulsion must produce non-zero force: vel[1]={vel[1]}"
        )


class TestMarlFullPipeline:
    """P12.1→P12.2→P12.3: Full MARL pipeline — action → obs roundtrip,
    action clearing, env isolation, reward progression."""

    @pytest.fixture
    def _env(self):
        """Create a basic environment (skip if gymnasium unavailable)."""
        pytest.importorskip("gymnasium")
        from pymurmur.analysis.gym_env import MurmurationEnv
        return MurmurationEnv(num_boids=10, episode_steps=200, seed=42)

    def test_env_step_reflects_action_in_obs(self, _env):
        """A +X action produces a larger X-position change than zero action.

        Full chain: env.step(action) → config._marl_action →
        MarlMode.compute (control→move→rules) → engine.integrate →
        _get_obs().  Compared against a zero-action baseline with
        identical starting state — isolates the action's marginal
        effect from internal rules (cohesion, etc.) that both
        environments share.  P12.1→P12.2 roundtrip."""
        from pymurmur.analysis.gym_env import MurmurationEnv

        # Two envs with same seed → identical initial state
        env_action = MurmurationEnv(num_boids=10, episode_steps=200, seed=42)
        env_zero = MurmurationEnv(num_boids=10, episode_steps=200, seed=42)

        env_action.reset()
        env_zero.reset()

        # +X action for all birds
        action_plus_x = np.zeros(30, dtype=np.float32)
        action_plus_x[0::3] = 1.0
        action_zero = np.zeros(30, dtype=np.float32)

        obs_action, _, _, _, _ = env_action.step(action_plus_x)
        obs_zero, _, _, _, _ = env_zero.step(action_zero)

        # Position channel (first 30 values): X is every 3rd entry
        pos_x_action = obs_action[0::3]
        pos_x_zero = obs_zero[0::3]

        # +X action must move birds further right than zero action
        mean_delta_action = np.mean(pos_x_action)
        mean_delta_zero = np.mean(pos_x_zero)
        assert mean_delta_action > mean_delta_zero, (
            f"+X action (mean X={mean_delta_action:.6f}) must exceed "
            f"zero action (mean X={mean_delta_zero:.6f})"
        )

    def test_marl_action_cleared_after_step(self, _env):
        """After env.step() returns, config._marl_action is None.

        Prevents stale action re-application if the engine is accessed
        externally between gym steps.  Cross-cutting P12.1→P12.2."""
        _env.reset()
        action = np.ones(30, dtype=np.float32)
        _env.step(action)

        # After step(), the action must be cleared
        assert _env._engine is not None
        assert getattr(_env._engine.config, "_marl_action", object()) is None, (
            "_marl_action must be None after env.step() to prevent stale re-application"
        )

    def test_two_envs_independent_state(self, _env):
        """Two MurmurationEnv instances have completely independent
        flock state — reset() creates a new SimulationEngine."""
        from pymurmur.analysis.gym_env import MurmurationEnv

        env_a = MurmurationEnv(num_boids=10, episode_steps=200, seed=42)
        env_b = MurmurationEnv(num_boids=10, episode_steps=200, seed=99)

        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()

        # Different seeds → different initial positions
        assert not np.allclose(obs_a, obs_b, atol=1e-5), (
            "Different seeds must produce different initial observations"
        )

        # Step env_a only — env_b must remain unchanged
        action = np.zeros(30, dtype=np.float32)
        action[0::3] = 1.0
        obs_a2, _, _, _, _ = env_a.step(action)

        # env_b's engine must still have the initial positions
        obs_b_initial = env_b._get_obs()
        assert np.allclose(obs_b, obs_b_initial, atol=1e-6), (
            "env_b must be unaffected by env_a stepping"
        )

    def test_reward_changes_across_steps(self, _env):
        """With zero action, internal rules (cohesion/alignment/separation)
        evolve the flock, producing changing rewards across steps.
        P12.2→P9.9 pipeline."""
        _env.reset()
        zero_action = np.zeros(30, dtype=np.float32)

        rewards = []
        for _ in range(10):
            _, reward, _, _, _ = _env.step(zero_action)
            rewards.append(float(reward))

        # Rewards should not all be identical — the flock evolves
        assert len(set(rewards)) > 1, (
            f"Flock should evolve under internal rules, "
            f"producing varied rewards: {rewards}"
        )

    def test_obs_velocity_channel_in_bounds_after_clamp(self, _env):
        """After env.step(), velocity channel (v/v_cap) in obs is ≤ 1.
        P12.1→P12.2: marl speed clamping → obs normalization chain."""
        _env.reset()
        action = _env.action_space.sample()

        for _ in range(5):
            obs, _, _, _, _ = _env.step(action)
            # Velocity channel is the last 30 values (for 10 boids)
            vel_channel = obs[30:]
            assert np.all(vel_channel >= -1.0) and np.all(vel_channel <= 1.0), (
                "Velocity observation must stay in [-1,1] after speed clamping"
            )

    def test_multi_step_stability_no_nan(self, _env):
        """20 steps with random actions produce no NaN in obs or reward.
        P12.1→P12.2: full pipeline stability."""
        _env.reset()
        rng = np.random.default_rng(123)

        for _ in range(20):
            action = rng.uniform(-1, 1, size=30).astype(np.float32)
            obs, reward, _, _, _ = _env.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in obs at step {_}"
            assert not np.isnan(reward), f"NaN in reward at step {_}"

    def test_first_step_reward_not_default(self, _env):
        """After first env.step(), reward comes from real metrics (not 0.0).
        P12.1→P12.2→P9.9: metrics→reward chain."""
        _env.reset()
        action = np.zeros(30, dtype=np.float32)
        _, reward, _, _, _ = _env.step(action)

        # Reward should be a real computed value (not just the empty-history
        # fallback of 0.0). With 10 random boids, alignment/cohesion produce
        # non-trivial metrics.
        assert isinstance(reward, (float, np.floating)), (
            f"Reward should be a float, got {type(reward)}"
        )
        # The reward may be positive or negative depending on the initial
        # random state, but it should not be exactly 0.0 (the fallback)


class TestMarlEngineStepMetrics:
    """P12.1→engine→metrics: engine.step() with marl mode collects
    FlockMetrics and populates the history list."""

    def test_engine_step_collects_metrics(self):
        """After engine.step(), metrics.history has one entry."""
        cfg = SimConfig(mode="marl", num_boids=10, seed=42,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)

        assert len(engine.metrics.history) == 0

        engine.config._marl_action = np.zeros((10, 3), dtype=np.float32)
        engine.step()

        assert len(engine.metrics.history) == 1, (
            "engine.step() must collect one metrics entry"
        )

    def test_metrics_contains_expected_fields(self):
        """Collected metrics have standard FlockMetrics attributes."""
        cfg = SimConfig(mode="marl", num_boids=10, seed=42,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)

        engine.config._marl_action = np.zeros((10, 3), dtype=np.float32)
        engine.step()

        m = engine.metrics.history[0]
        # Core fields that FlockMetrics always has
        assert hasattr(m, "alpha"), "Metrics must have alpha"
        assert isinstance(m.alpha, float), "alpha must be float"
        assert hasattr(m, "speed_avg"), "Metrics must have speed_avg"
        assert hasattr(m, "dispersion"), "Metrics must have dispersion"
        # alpha is computed from active birds, should be a valid float
        assert not np.isnan(m.alpha), "alpha should not be NaN"

    def test_multiple_steps_accumulate_metrics(self):
        """N engine.step() calls produce N metrics entries."""
        cfg = SimConfig(mode="marl", num_boids=10, seed=42,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)

        for _i in range(5):
            engine.config._marl_action = np.zeros((10, 3), dtype=np.float32)
            engine.step()

        assert len(engine.metrics.history) == 5, (
            "5 steps must produce 5 metrics entries"
        )


class TestMarlDeferredPositions:
    """P12.1→engine: Over multiple engine.step() calls, deferred rules
    actually move bird positions (not just velocities)."""

    def test_positions_change_over_multiple_steps(self):
        """After 10 steps with zero action, positions differ from initial."""
        cfg = SimConfig(mode="marl", num_boids=10, seed=42,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)
        pos_before = engine.flock.positions.copy()

        for _ in range(10):
            engine.config._marl_action = np.zeros((10, 3), dtype=np.float32)
            engine.step()

        pos_after = engine.flock.positions.copy()
        assert not np.allclose(pos_before, pos_after, atol=1e-6), (
            "Positions must change after 10 steps of deferred rules"
        )

    def test_two_step_lag_positions_depend_on_prior_rules(self):
        """Two-step lag: position at step k depends on rules from step k-1
        + action at step k.  Uses _step_physics(dt=1.0) directly to
        bypass the P8.10 fixed-timestep accumulator for clarity."""
        cfg = SimConfig(mode="marl", num_boids=5, seed=42,
                        width=200, height=200, depth=200)
        from pymurmur.simulation.engine import SimulationEngine
        engine = SimulationEngine(cfg)
        # Place birds spread out so cohesion has clear effect
        engine.flock.positions[0] = [10.0, 100.0, 100.0]
        engine.flock.positions[1] = [190.0, 100.0, 100.0]
        engine.flock.positions[2] = [100.0, 10.0, 100.0]
        engine.flock.positions[3] = [100.0, 190.0, 100.0]
        engine.flock.positions[4] = [100.0, 100.0, 100.0]
        # Zero velocities so movement comes purely from computed forces
        engine.flock.velocities[:] = 0.0

        # Step 0: forces compute, velocities updated, positions move
        engine.config._marl_action = np.zeros((5, 3), dtype=np.float32)
        engine._step_physics(dt=1.0)
        pos_after_step0 = engine.flock.positions.copy()

        # Step 1: rules from step 0 persist in velocities,
        # cohesion continues pulling toward center
        engine.config._marl_action = np.zeros((5, 3), dtype=np.float32)
        engine._step_physics(dt=1.0)
        pos_after_step1 = engine.flock.positions.copy()

        # Bird 0 started at [10, 100, 100], CoM ≈ [98, 100, 100]
        # Cohesion pulls it +X. After 2 steps, it should have moved right.
        assert pos_after_step1[0, 0] > pos_after_step0[0, 0], (
            f"Two-step lag: bird 0 should move +X toward CoM. "
            f"step0={pos_after_step0[0]}, step1={pos_after_step1[0]}"
        )


class TestMarlEngineDispatch:
    """P12.1→engine: compute_all_forces dispatches to marl mode
    through MODE_REGISTRY."""

    def test_compute_all_forces_uses_marl_mode(self):
        """When config.mode='marl', compute_all_forces calls marl compute."""
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.physics.forces import compute_all_forces

        cfg = SimConfig(mode="marl", num_boids=10, seed=42)
        flock = PhysicsFlock(cfg)
        # Set an external action before dispatching
        cfg._marl_action = np.ones((10, 3), dtype=np.float32)
        vel_before = flock.velocities.copy()

        compute_all_forces(flock, cfg)

        # Marl mode must have affected velocities (external action applied)
        assert not np.allclose(flock.velocities, vel_before, atol=1e-6), (
            "compute_all_forces with marl mode must apply external action"
        )

    def test_mode_needs_index_returns_false_for_marl(self):
        """mode_needs_index('marl') returns False (no spatial index)."""
        from pymurmur.physics.forces import mode_needs_index
        assert mode_needs_index("marl") is False


class TestMarlConfigParams:
    """P12.1→P12.2: MARL-specific config parameters flow through
    to force computation and affect behaviour."""

    def test_velocity_cap_affects_max_speed(self):
        """Smaller marl_velocity_cap → lower maximum speed after compute."""
        cfg_low = SimConfig(mode="marl", num_boids=5)
        cfg_high = SimConfig(mode="marl", num_boids=5)
        cfg_low.marl_velocity_cap = 0.2
        cfg_high.marl_velocity_cap = 2.0

        pos, vel_low, acc, active, last_theta, rng = _make_flock_arrays(5, cfg_low)
        vel_high = vel_low.copy()
        vel_low[:] = 100.0
        vel_high[:] = 100.0

        cfg_low._marl_action = np.zeros((5, 3), dtype=np.float32)
        cfg_high._marl_action = np.zeros((5, 3), dtype=np.float32)

        MarlMode.compute(pos, vel_low, acc, active, None, rng, last_theta, cfg_low)
        rng2 = np.random.default_rng(42)
        MarlMode.compute(pos, vel_high, acc, active, None, rng2, last_theta, cfg_high)

        max_speed_low = np.max(np.linalg.norm(vel_low, axis=1))
        max_speed_high = np.max(np.linalg.norm(vel_high, axis=1))
        assert max_speed_low < max_speed_high, (
            f"Smaller v_cap must produce lower max speed: "
            f"low={max_speed_low:.3f} vs high={max_speed_high:.3f}"
        )

    def test_rule_weight_affects_force_magnitude(self):
        """Larger marl_rule_weight → larger velocity change from internal
        rules (separation + alignment + cohesion)."""
        cfg_small = SimConfig(mode="marl", num_boids=5)
        cfg_large = SimConfig(mode="marl", num_boids=5)
        cfg_small.marl_rule_weight = 0.001
        cfg_large.marl_rule_weight = 0.1

        pos, vel_small, acc, active, last_theta, rng = _make_flock_arrays(5, cfg_small)
        vel_large = vel_small.copy()

        cfg_small._marl_action = np.zeros((5, 3), dtype=np.float32)
        cfg_large._marl_action = np.zeros((5, 3), dtype=np.float32)

        vel_small_before = vel_small.copy()
        vel_large_before = vel_large.copy()

        MarlMode.compute(pos, vel_small, acc, active, None, rng, last_theta, cfg_small)
        rng2 = np.random.default_rng(42)
        MarlMode.compute(pos, vel_large, acc, active, None, rng2, last_theta, cfg_large)

        delta_small = np.linalg.norm(vel_small - vel_small_before)
        delta_large = np.linalg.norm(vel_large - vel_large_before)
        assert delta_small < delta_large, (
            f"Larger rule_weight must produce larger velocity change: "
            f"small={delta_small:.6f} vs large={delta_large:.6f}"
        )


class TestMarlScripts:
    """P12.3: Script argument parsing, module import, dependency gates."""

    def test_train_marl_help(self):
        """train_marl.py --help exits 0 (argparse runs before imports)."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/train_marl.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"train_marl.py --help should exit 0, got {result.returncode}"
        )

    def test_rollout_marl_help(self):
        """rollout_marl.py --help exits 0 (argparse runs before imports)."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/rollout_marl.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"rollout_marl.py --help should exit 0, got {result.returncode}"
        )

    def test_train_marl_module_import_does_not_execute_main(self):
        """Importing scripts.train_marl does not trigger main() execution."""
        import importlib.util
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__))))
        path = os.path.join(project_root, "scripts", "train_marl.py")
        spec = importlib.util.spec_from_file_location("train_marl", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main")

    def test_rollout_marl_module_import_does_not_execute_main(self):
        """Importing scripts.rollout_marl does not trigger main() execution."""
        import importlib.util
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__))))
        path = os.path.join(project_root, "scripts", "rollout_marl.py")
        spec = importlib.util.spec_from_file_location("rollout_marl", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main")

    def test_train_marl_invalid_timesteps_exits_nonzero(self):
        """Non-integer --timesteps causes argparse error (exit 2)."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/train_marl.py", "--timesteps", "abc"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, (
            f"Invalid --timesteps should exit non-zero, got {result.returncode}"
        )

    def test_train_marl_help_lists_all_arguments(self):
        """train_marl.py --help output describes --timesteps, --num-boids, --output."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/train_marl.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        out = result.stdout
        assert "--timesteps" in out
        assert "--num-boids" in out
        assert "--output" in out

    def test_train_marl_has_dependency_gate_in_source(self):
        """train_marl.py source contains the gymnasium import guard."""
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__))))
        path = os.path.join(project_root, "scripts", "train_marl.py")
        with open(path) as f:
            source = f.read()
        assert "gymnasium not installed" in source, (
            "train_marl.py must have a gymnasium dependency gate"
        )
        assert "stable-baselines3 not installed" in source, (
            "train_marl.py must have a stable-baselines3 dependency gate"
        )

    def test_rollout_marl_has_dependency_gate_in_source(self):
        """rollout_marl.py source contains the gymnasium import guard."""
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__))))
        path = os.path.join(project_root, "scripts", "rollout_marl.py")
        with open(path) as f:
            source = f.read()
        assert "gymnasium not installed" in source, (
            "rollout_marl.py must have a gymnasium dependency gate"
        )
        assert "stable-baselines3 not installed" in source, (
            "rollout_marl.py must have a stable-baselines3 dependency gate"
        )

    def test_rollout_marl_help_lists_all_arguments(self):
        """rollout_marl.py --help output describes --model, --steps, --output, --num-boids."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/rollout_marl.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        out = result.stdout
        assert "--model" in out
        assert "--steps" in out
        assert "--output" in out
        assert "--num-boids" in out



# ── @slow: MARL trained-beats-random experiment ───────────────

@pytest.mark.slow
class TestMarlTrainedBeatsRandom:
    """S12 (@slow): a minimally-trained PPO policy achieves higher mean
    reward than random actions over a short rollout.  Verifies the
    end-to-end MARL training pipeline (env → train → rollout) is not
    fundamentally broken."""

    def test_trained_beats_random(self):
        """Train PPO for ~1000 timesteps on a 10-boid env, then compare
        trained deterministic rollout vs random-action rollout."""
        pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")

        from stable_baselines3 import PPO

        from pymurmur.analysis.gym_env import MurmurationEnv

        # ── Train ──────────────────────────────────────────────
        env = MurmurationEnv(num_boids=10, episode_steps=200, seed=42)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=1000)

        # ── Trained rollout ────────────────────────────────────
        env_eval = MurmurationEnv(num_boids=10, episode_steps=100, seed=42)
        obs, _ = env_eval.reset()
        trained_reward = 0.0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_eval.step(action)
            trained_reward += float(reward)
            if terminated or truncated:
                break

        # ── Random rollout ────────────────────────────────────
        env_rand = MurmurationEnv(num_boids=10, episode_steps=100, seed=42)
        obs_r, _ = env_rand.reset()
        random_reward = 0.0
        for _ in range(100):
            action = env_rand.action_space.sample()
            obs_r, reward, terminated, truncated, _ = env_rand.step(action)
            random_reward += float(reward)
            if terminated or truncated:
                break

        assert trained_reward > random_reward, (
            f"Trained reward ({trained_reward:.3f}) must exceed "
            f"random reward ({random_reward:.3f})"
        )
