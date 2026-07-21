#!/usr/bin/env python3
"""P12.3 — Train a PPO policy on MurmurationEnv.

Dependency-gated: requires gymnasium and stable-baselines3.
Docstring notes centralized-MLP quadratic scaling and points to IPPO
for large N.

Usage:
    python scripts/train_marl.py [--timesteps 5000] [--num-boids 20] [--output output/marl_ppo]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on MurmurationEnv")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="Total timesteps (default: 5000)")
    parser.add_argument("--num-boids", type=int, default=20,
                        help="Number of boids (default: 20)")
    parser.add_argument("--output", type=str, default="output/marl_ppo",
                        help="Save path for the trained model")
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

    print(
        f"Training PPO on MurmurationEnv (N={args.num_boids}, "
        f"timesteps={args.timesteps})"
    )
    print(
        "Note: centralized MLP scales quadratically with N. "
        "For N > 50, consider IPPO."
    )

    env = MurmurationEnv(
        num_boids=args.num_boids,
        episode_steps=500,
        seed=42,
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
