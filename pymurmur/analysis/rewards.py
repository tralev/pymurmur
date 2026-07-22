"""S3.9: Five-term penalty-composite reward module.

Computes scalar rewards from FlockMetrics, shared by MARL (S7,
`analysis/gym_env.py`) and EvoFlock scalarization pipelines. Every
term is an existing S3.8/FlockMetrics observable — no new physics.

    R = ±w_a·velocity_deviation − w_c·dispersion
        − w_L·‖Σᵢ(pᵢ−CoM)×vᵢ‖/N
        − w_b·boundary_overshoot − w_z·altitude_deviation

`velocity_deviation`'s sign is the source's quirk: under
`faithful_signs=True` it is `+w_a·velocity_deviation` (the agent trades
deviation against compactness — deliberately not "corrected"); every
other term stays negative in both modes. Under `faithful_signs=False`
("corrected") the alignment term also flips negative, so the reward
is a pure penalty composite with maximum 0 at perfect order (α=1
alignment ⇒ velocity_deviation=0; no dispersion, no rotation, no
boundary/altitude excursion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .metrics import FlockMetrics


@dataclass
class RewardConfig:
    """Per-term weights for the five-term penalty composite.

    Defaults match the spec: w_a = w_c = 1 (the two core terms live),
    the three extension terms (angular momentum, boundary, altitude)
    default to 0 — opt-in per training/evaluation run.
    """

    w_a: float = 1.0   # velocity_deviation weight
    w_c: float = 1.0   # dispersion weight
    w_L: float = 0.0   # angular-momentum penalty weight
    w_b: float = 0.0   # boundary_overshoot weight
    w_z: float = 0.0   # altitude_deviation weight
    faithful_signs: bool = True


def compute_reward(
    metrics: FlockMetrics,
    config: RewardConfig | None = None,
) -> float:
    """S3.9: Compute the five-term penalty-composite reward.

    Args:
        metrics: FlockMetrics snapshot (raw values — never smoothed;
            see S3.11's display/analysis separation).
        config: RewardConfig with per-term weights. Uses defaults if None.

    Returns:
        Scalar reward. Under faithful_signs=False every term is a
        penalty (<= 0), maximum 0 at perfect order. Under
        faithful_signs=True (default) the velocity_deviation term
        flips positive.
    """
    if config is None:
        config = RewardConfig()

    angular_momentum_mag = float(np.linalg.norm(metrics.angular_momentum))
    align_sign = 1.0 if config.faithful_signs else -1.0

    return (
        align_sign * config.w_a * metrics.velocity_deviation
        - config.w_c * metrics.dispersion
        - config.w_L * angular_momentum_mag
        - config.w_b * metrics.boundary_overshoot
        - config.w_z * metrics.altitude_deviation
    )


def reward_linearity_check(
    metrics_a: FlockMetrics,
    metrics_b: FlockMetrics,
    config: RewardConfig | None = None,
) -> bool:
    """S3.9: Verify per-term weight linearity.

    Doubling every weight should exactly double the reward for an
    identical metrics snapshot (the composite is linear in the
    weights by construction; this guards against a future
    non-linear term breaking that property).

    Args:
        metrics_a, metrics_b: same-snapshot pair (should be identical).
        config: base config.

    Returns:
        True if compute_reward(metrics_b, 2*weights) ==
        2*compute_reward(metrics_a, weights) within numerical tolerance.
    """
    if config is None:
        config = RewardConfig()

    r1 = compute_reward(metrics_a, config)
    doubled = RewardConfig(
        w_a=2.0 * config.w_a,
        w_c=2.0 * config.w_c,
        w_L=2.0 * config.w_L,
        w_b=2.0 * config.w_b,
        w_z=2.0 * config.w_z,
        faithful_signs=config.faithful_signs,
    )
    r2 = compute_reward(metrics_b, doubled)
    return math.isclose(r2, 2.0 * r1, rel_tol=1e-8, abs_tol=1e-9)
