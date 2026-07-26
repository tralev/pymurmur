"""Priority-ordered force budget allocation.

Level 0-equivalent — numpy only, zero pymurmur imports. Used by the
engine's priority_stack_enabled branch to combine obstacle-avoidance
(tier 1), predator-threat (tier 2), and flocking (tier 3) forces with
a binary-cutoff cascade rather than a plain weighted sum.
"""

from __future__ import annotations

import numpy as np


def allocate_priority_budget(
    tier1: np.ndarray,
    tier2: np.ndarray,
    tier3: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Binary-cutoff priority allocation over a shared per-boid force budget.

    tier1 (obstacle avoidance) claims the budget first and is clamped to
    it. tier2 (predator-threat) only gets what's left, and is ZEROED
    ENTIRELY — not scaled down — when tier1 alone already saturates the
    budget. tier3 (flocking) cascades the same way against tier1+tier2.
    This matches the reference semantic ("if the obstacle force
    saturates the budget, no other forces are applied") — a hard
    cutoff, not a proportional blend.

    Args:
        tier1, tier2, tier3: (N, 3) float arrays, requested acceleration
            per tier
        budget: scalar per-tick force-budget ceiling

    Returns:
        (N, 3) float32 array, combined result with magnitude <= budget
    """
    tier1 = np.asarray(tier1, dtype=np.float64)
    tier2 = np.asarray(tier2, dtype=np.float64)
    tier3 = np.asarray(tier3, dtype=np.float64)

    mag1 = np.linalg.norm(tier1, axis=1)
    over1 = mag1 > budget
    scale1 = np.where(over1, np.divide(budget, mag1, out=np.ones_like(mag1), where=mag1 > 0), 1.0)
    tier1_clamped = tier1 * scale1[:, np.newaxis]
    remaining1 = np.maximum(budget - np.minimum(mag1, budget), 0.0)
    saturated1 = mag1 >= budget

    tier2_allowed = np.where(saturated1[:, np.newaxis], 0.0, tier2)
    mag2 = np.linalg.norm(tier2_allowed, axis=1)
    over2 = mag2 > remaining1
    scale2 = np.where(over2, np.divide(remaining1, mag2, out=np.ones_like(mag2), where=mag2 > 0), 1.0)
    tier2_clamped = tier2_allowed * scale2[:, np.newaxis]
    mag2_clamped = np.linalg.norm(tier2_clamped, axis=1)
    remaining2 = np.maximum(remaining1 - mag2_clamped, 0.0)
    saturated2 = saturated1 | (mag2 >= remaining1)

    tier3_allowed = np.where(saturated2[:, np.newaxis], 0.0, tier3)
    mag3 = np.linalg.norm(tier3_allowed, axis=1)
    over3 = mag3 > remaining2
    scale3 = np.where(over3, np.divide(remaining2, mag3, out=np.ones_like(mag3), where=mag3 > 0), 1.0)
    tier3_clamped = tier3_allowed * scale3[:, np.newaxis]

    return (tier1_clamped + tier2_clamped + tier3_clamped).astype(np.float32)
