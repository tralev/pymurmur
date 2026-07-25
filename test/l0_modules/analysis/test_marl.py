"""P12 — MARL mode registration, MarlMode.compute() force basics (deferred, zero-params, edge cases), action scale.

Dependency-gated: gymnasium via pytest.importorskip.

Split out of test_marl.py (file-size split).
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


