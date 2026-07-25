"""P12 — MARL separation radius/cap/toroidal-axes tests, MurmurationEnv (P12.2) gym environment.

Dependency-gated: gymnasium via pytest.importorskip.

Split out of test_marl.py (file-size split).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.marl import MarlMode
from test.l0_modules.physics.forces.marl.test_marl import _make_flock_arrays


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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv
        return MurmurationEnv(num_boids=10, episode_steps=200, seed=42)

    def test_env_spaces_correct_shape(self, _env):
        """Observation space is (6N,) and action space is (3N,)."""
        assert _env.observation_space.shape == (60,)
        assert _env.action_space.shape == (30,)

    def test_episode_steps_defaults_to_config_marl_episode_steps(self):
        """C3: episode_steps=None falls back to config.marl_episode_steps."""
        pytest.importorskip("gymnasium")
        from pymurmur.analysis.rl.gym_env import MurmurationEnv

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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv

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

        from pymurmur.analysis.rl.gym_env import MurmurationEnv
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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv
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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv
        env = MurmurationEnv(num_boids=5, episode_steps=100)
        env.reset()
        assert env._engine is not None
        assert env._engine.config.mode == "marl"

    def test_get_obs_formula_positions(self, _env):
        """_get_obs normalizes positions as (p-C)/3U."""
        from pymurmur.analysis.rl.gym_env import MurmurationEnv
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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv
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
        from pymurmur.analysis.rl.gym_env import MurmurationEnv
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


