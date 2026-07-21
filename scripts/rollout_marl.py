#!/usr/bin/env python3
"""P12.3 — Rollout a trained MARL policy and export a dual-view GIF.

Dependency-gated: requires gymnasium, stable-baselines3, and PIL.

Usage:
    python scripts/rollout_marl.py [--model output/marl_ppo] [--steps 500] [--output output/marl_rollout.gif]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout trained MARL policy")
    parser.add_argument("--model", type=str, default="output/marl_ppo",
                        help="Path to the trained PPO model")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of deterministic-predict steps")
    parser.add_argument("--output", type=str, default="output/marl_rollout.gif",
                        help="Output GIF path")
    parser.add_argument("--num-boids", type=int, default=20,
                        help="Number of boids")
    args = parser.parse_args()

    # Dependency gates
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("Error: gymnasium not installed. Run: pip install gymnasium")
        sys.exit(1)
    try:
        from stable_baselines3 import PPO  # noqa: F401
    except ImportError:
        print(
            "Error: stable-baselines3 not installed. "
            "Run: pip install stable-baselines3"
        )
        sys.exit(1)

    from stable_baselines3 import PPO

    from pymurmur.analysis.gym_env import MurmurationEnv

    print(f"Loading model from {args.model}")
    model = PPO.load(args.model)

    env = MurmurationEnv(
        num_boids=args.num_boids,
        episode_steps=args.steps,
        seed=123,
    )
    obs, _ = env.reset()

    total_reward = 0.0

    print(f"Running {args.steps} deterministic-predict steps...")
    for step in range(args.steps):  # noqa: B007 — used after the loop below
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    print(f"Rollout complete: {step + 1} steps, total reward = {total_reward:.3f}")
    print(
        "Full GIF export requires matplotlib/PIL — see "
        "pymurmur/capture for the capture pipeline."
    )


if __name__ == "__main__":
    main()
