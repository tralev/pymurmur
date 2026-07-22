"""P12.2 — MurmurationEnv: Gymnasium wrapper for multi-agent flock control.

Lazy import — gymnasium is optional. On import failure, a helpful
error message directs the user to `pip install gymnasium`.

Observation: Box(−1, 1, (6N,)) — concat((p−C)/3U, v/v_cap)
Action: Box(−1, 1, (3N,)) — per-bird velocity adjustment
Reward: from pymurmur.analysis.rewards (P9.9 weighted composite)
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:

    from ..simulation.engine import SimulationEngine


# ── Lazy gymnasium import ──────────────────────────────────────────

def _require_gym():
    """Import gymnasium or raise a helpful error."""
    try:
        import gymnasium  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "MurmurationEnv requires gymnasium. Install with: pip install gymnasium"
        ) from err


# Conditional base class: when gymnasium is absent, fall back to object
# so the module can be imported without it.  _require_gym() in __init__
# still catches actual usage.  This is needed because class definitions
# are evaluated at module-import time, not at __init__ time, and
# stable-baselines3 v2.x requires isinstance(env, gymnasium.Env).
_BaseEnv: Any = object
try:
    import gymnasium as _gymnasium
    _BaseEnv = _gymnasium.Env
except ImportError:
    pass


# ── MurmurationEnv ─────────────────────────────────────────────────

class MurmurationEnv(_BaseEnv):
    """Gymnasium environment for MARL flock control.

    Subclasses ``gymnasium.Env`` so stable-baselines3 v2.x
    (which checks ``isinstance(env, gymnasium.Env)``) can
    consume it directly.  All required gymnasium methods
    (reset, step, observation_space, action_space) are
    implemented.

    Usage::

        import gymnasium
        from pymurmur.analysis.gym_env import MurmurationEnv

        env = MurmurationEnv(num_boids=20, episode_steps=500)
        obs, info = env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        num_boids: int = 20,
        episode_steps: int | None = None,
        mode: str = "marl",
        seed: int | None = None,
        config_overrides: dict | None = None,
    ) -> None:
        _require_gym()
        super().__init__()
        import gymnasium

        self._num_boids = num_boids
        self._seed = seed
        self._step_count: int = 0

        from ..core.config import SimConfig
        self._base_config = SimConfig(
            num_boids=num_boids,
            mode=mode,
            seed=seed,
            **(config_overrides or {}),
        )
        # C3: marl_episode_steps — explicit episode_steps kwarg wins,
        # otherwise fall back to config (default or config_overrides).
        self._episode_steps = (
            episode_steps if episode_steps is not None
            else self._base_config.marl_episode_steps
        )
        # Ensure the config has marl mode registered
        self._engine: SimulationEngine | None = None

        # Gym spaces
        U = min(self._base_config.width, self._base_config.height,
                self._base_config.depth) / 6.0
        v_cap = getattr(self._base_config, "marl_velocity_cap", 0.5) * U
        self._obs_scale = 3.0 * U
        self._act_scale = v_cap

        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(6 * num_boids,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(3 * num_boids,), dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the simulation and return the initial observation."""
        # gymnasium.Env contract: subclasses must call super().reset(seed=...)
        # so self.np_random is (re)seeded — gymnasium.utils.env_checker
        # verifies this explicitly. The flock's own determinism runs
        # through cfg.seed/flock.rng below, independent of self.np_random;
        # this call exists to satisfy the base-class contract, not to
        # drive the simulation's RNG.
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        cfg = copy.copy(self._base_config)
        if self._seed is not None:
            cfg.seed = self._seed
        cfg.num_boids = self._num_boids

        from ..simulation.engine import SimulationEngine
        self._engine = SimulationEngine(cfg)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Step the simulation with an external action.

        Args:
            action: (3N,) float32 array in [-1, 1]. Clipped to bounds.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self._engine is not None, "Must call reset() before step()"

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # D8: control hook — step() takes and clears the action itself
        # (previously this reached into engine.config._marl_action
        # directly, which also meant remembering to clear it after so a
        # stale action wasn't silently re-applied if the engine got used
        # outside the env wrapper between steps).
        self._engine.step(control=action.reshape(self._num_boids, 3))
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self._step_count >= self._episode_steps
        info = {"step": self._step_count}

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build observation: concat((p-C)/3U, v/v_cap) → (6N,)."""
        assert self._engine is not None
        flock = self._engine.flock
        pos = flock.positions.astype(np.float32)
        vel = flock.velocities.astype(np.float32)

        W, H, D = self._engine.config.width, self._engine.config.height, self._engine.config.depth
        U = min(W, H, D) / 6.0
        center = np.array([W / 2, H / 2, D / 2], dtype=np.float32)
        obs_scale = 3.0 * U
        v_cap = self._act_scale

        pos_norm = (pos - center) / obs_scale
        vel_norm = vel / max(v_cap, 1e-8)
        obs = np.concatenate([pos_norm.flatten(), vel_norm.flatten()]).astype(np.float32)
        # observation_space declares Box(-1, 1) — neither term is a hard
        # physical limit (birds can be near a domain edge; speed can
        # transiently exceed the soft v_cap), so clip to actually honor
        # the declared contract. Without this, gymnasium's check_env and
        # any code trusting the declared bounds sees out-of-space values.
        return np.clip(obs, -1.0, 1.0)

    def _compute_reward(self) -> float:
        """Compute reward from the current flock metrics."""
        assert self._engine is not None
        from ..analysis.rewards import RewardConfig, compute_reward

        history = self._engine.metrics.history
        if not history:
            return 0.0
        m = history[-1]
        return compute_reward(m, RewardConfig(faithful_signs=False))
