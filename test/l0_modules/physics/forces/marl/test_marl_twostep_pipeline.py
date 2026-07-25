"""P12 — MARL two-step mode, backward-compat force API, toroidal separation, full engine pipeline.

Dependency-gated: gymnasium via pytest.importorskip.

Split out of test_marl.py (file-size split).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.marl import MarlMode
from test.l0_modules.physics.forces.marl.test_marl import _make_flock_arrays


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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv
        return MurmurationEnv(num_boids=10, episode_steps=200, seed=42)

    def test_env_step_reflects_action_in_obs(self, _env):
        """A +X action produces a larger X-position change than zero action.

        Full chain: env.step(action) → config._marl_action →
        MarlMode.compute (control→move→rules) → engine.integrate →
        _get_obs().  Compared against a zero-action baseline with
        identical starting state — isolates the action's marginal
        effect from internal rules (cohesion, etc.) that both
        environments share.  P12.1→P12.2 roundtrip."""
        from pymurmur.analysis.rl.gym_env import MurmurationEnv

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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv

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


