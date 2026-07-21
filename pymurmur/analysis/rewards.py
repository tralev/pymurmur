"""P9.9: Weighted composite reward module.

Computes scalar rewards from FlockMetrics, shared by MARL (P12)
and EvoFlock (P11) scalarization pipelines. Supports per-term
weights, a faithful_signs flag (flips sign so direction punishes
disorder), and optional baseline correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .metrics import FlockMetrics


@dataclass
class RewardTerms:
    """Individual reward contributions from named metric terms.

    All terms are normalised to an approximate [0, 1] or [−1, 1]
    range so that weights have consistent scale.
    """

    alignment: float = 0.0       # α (polar order), range [0, 1]
    cohesion: float = 0.0        # 1 / (1 + dispersion/100), range [0, 1]
    spacing: float = 0.0         # 1 / (1 + |local_spacing − target|/target)
    nematic: float = 0.0         # nematic S, range [0, 1]
    silhouette: float = 0.0      # 2D silhouette coverage, range [0, 1]
    speed_match: float = 0.0     # 1 / (1 + velocity_deviation)
    boundary: float = 0.0        # 1 / (1 + boundary_overshoot/100)
    altitude: float = 0.0        # 1 / (1 + altitude_deviation/100)
    energy: float = 0.0          # −min(energy_J / emax, 1), lower energy = better

    @property
    def vector(self) -> np.ndarray:
        """All terms as a 1D numpy array (fixed order)."""
        return np.array(
            [
                self.alignment,
                self.cohesion,
                self.spacing,
                self.nematic,
                self.silhouette,
                self.speed_match,
                self.boundary,
                self.altitude,
                self.energy,
            ],
            dtype=np.float64,
        )


@dataclass
class RewardConfig:
    """Configuration for reward computation.

    Attributes:
        weights: Per-term weight dict. Keys match RewardTerms fields.
            Default gives equal weight to alignment and cohesion.
        faithful_signs: If True, all terms are positive (higher is better).
            If False, some terms flip sign so that the overall reward
            behaves as a \"cost\" (lower is better). Default True.
        baseline: Optional baseline scalar subtracted from reward.
            Default 0 (no correction).
        max_reward: Clips reward to this maximum. Default +∞ (no clip).
    """

    weights: dict[str, float] = field(default_factory=lambda: {
        "alignment": 1.0,
        "cohesion": 1.0,
        "spacing": 0.5,
        "nematic": 0.5,
        "silhouette": 0.0,
        "speed_match": 0.5,
        "boundary": 0.3,
        "altitude": 0.0,
        "energy": 0.0,
    })
    faithful_signs: bool = True
    baseline: float = 0.0
    max_reward: float | None = None


def compute_reward(
    metrics: FlockMetrics,
    config: RewardConfig | None = None,
    target_spacing: float = 15.0,
    emax: float = 1.0,
) -> float:
    """P9.9: Compute weighted composite reward from flock metrics.

    Each term is normalised to [0, 1] (faithful_signs=True) so a
    perfect flock scores maximally. With faithful_signs=False the
    reward represents a cost (lower is better).

    Args:
        metrics: FlockMetrics snapshot.
        config: RewardConfig with per-term weights. Uses defaults if None.
        target_spacing: desired local_spacing for spacing term.
        emax: max energy cap for energy penalty term.

    Returns:
        Scalar reward (higher = better if faithful_signs=True).
    """
    if config is None:
        config = RewardConfig()
    weights = config.weights
    faithful = config.faithful_signs

    terms = RewardTerms()

    # Alignment: α ∈ [0, 1]
    terms.alignment = float(metrics.alpha)

    # Cohesion: 1/(1 + dispersion/100) — large dispersion → low reward
    terms.cohesion = 1.0 / (1.0 + metrics.dispersion / 100.0)

    # Spacing: 1 when local_spacing == target
    if target_spacing > 0:
        rel_err = abs(metrics.local_spacing - target_spacing) / target_spacing
        terms.spacing = 1.0 / (1.0 + rel_err)
    else:
        terms.spacing = 0.0

    # Nematic: S ∈ [0, 1]
    terms.nematic = float(metrics.nematic_S)

    # Silhouette: 2D coverage ∈ [0, 1]
    terms.silhouette = float(metrics.silhouette_2d)

    # Speed match: 1/(1 + velocity_deviation)
    if metrics.velocity_deviation >= 0:
        terms.speed_match = 1.0 / (1.0 + metrics.velocity_deviation)
    else:
        terms.speed_match = 0.0

    # Boundary: 1/(1 + overshoot/100)
    terms.boundary = 1.0 / (1.0 + metrics.boundary_overshoot / 100.0)

    # Altitude: 1/(1 + altitude_deviation/100)
    terms.altitude = 1.0 / (1.0 + metrics.altitude_deviation / 100.0)

    # Energy: lower energy → higher reward
    energy_clamped = min(metrics.energy_J, emax)
    terms.energy = 1.0 - energy_clamped / emax if emax > 0 else 0.0

    # Weighted sum
    reward = 0.0
    for name, weight in weights.items():
        term_val = getattr(terms, name, 0.0)
        reward += weight * term_val

    # Sign flip for cost-like rewards
    if not faithful:
        # In cost mode: higher raw = worse, so negate
        reward = -reward

    # Baseline correction
    reward -= config.baseline

    # Clip
    if config.max_reward is not None:
        reward = min(reward, config.max_reward)

    return reward


def reward_linearity_check(
    metrics_a: FlockMetrics,
    metrics_b: FlockMetrics,
    config: RewardConfig | None = None,
) -> bool:
    """P9.9: Verify per-term weight linearity.

    Doubling every weight should exactly double the reward.

    Args:
        metrics_a, metrics_b: same-snapshot pair (should be identical).
        config: base config.

    Returns:
        True if (sum 2w·t) == 2·(sum w·t) within numerical tolerance.
    """
    if config is None:
        config = RewardConfig()

    r1 = compute_reward(metrics_a, config)
    # Double all weights
    doubled = RewardConfig(
        weights={k: 2.0 * v for k, v in config.weights.items()},
        faithful_signs=config.faithful_signs,
        baseline=config.baseline,
        max_reward=config.max_reward,
    )
    r2 = compute_reward(metrics_b, doubled)
    return math.isclose(r2, 2.0 * r1, rel_tol=1e-8)
