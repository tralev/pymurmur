"""P12 — MARL engine step metrics, deferred positions, engine dispatch, config params, helper scripts, trained-vs-random experiment (@slow).

Dependency-gated: gymnasium via pytest.importorskip.

Split out of test_marl.py (file-size split).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.marl import MarlMode
from test.l0_modules.physics.forces.marl.test_marl import _make_flock_arrays


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
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
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
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
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
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
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
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
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

        from pymurmur.analysis.rl.gym_env import MurmurationEnv

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
