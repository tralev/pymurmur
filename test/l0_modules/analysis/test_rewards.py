"""P9.9 — Rewards module tests.

Tests weighted composite reward computation, faithful_signs flag,
and per-term weight linearity.
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import FlockMetrics
from pymurmur.analysis.rewards import (
    RewardConfig,
    RewardTerms,
    compute_reward,
    reward_linearity_check,
)


class TestRewardComputation:
    """Weighted composite reward from FlockMetrics."""

    def test_perfect_flock_maximum_reward(self):
        """Perfect flock (α=1, no dispersion, no deviation) → high reward."""
        m = FlockMetrics()
        m.alpha = 1.0
        m.nematic_S = 1.0
        m.dispersion = 0.0
        m.velocity_deviation = 0.0
        m.boundary_overshoot = 0.0
        m.altitude_deviation = 0.0
        m.silhouette_2d = 1.0
        m.local_spacing = 15.0  # exactly target
        m.energy_J = 0.0

        config = RewardConfig(
            weights={
                "alignment": 1.0,
                "cohesion": 1.0,
                "spacing": 1.0,
                "nematic": 1.0,
                "silhouette": 1.0,
                "speed_match": 1.0,
                "boundary": 1.0,
                "altitude": 1.0,
                "energy": 1.0,
            },
            faithful_signs=True,
        )
        reward = compute_reward(m, config)
        # Perfect flock: all terms ~1.0, sum of weights = 9.0
        expected_max = sum(config.weights.values())
        assert reward == pytest.approx(expected_max, rel=0.01), (
            f"Perfect flock reward {reward:.3f} ≈ {expected_max}"
        )

    def test_random_flock_lower_reward(self):
        """Random flock → reward < perfect flock reward."""
        m = FlockMetrics()
        m.alpha = 0.3
        m.dispersion = 300.0
        m.velocity_deviation = 2.0
        m.boundary_overshoot = 500.0
        m.altitude_deviation = 200.0
        m.local_spacing = 5.0

        config = RewardConfig(
            weights={
                "alignment": 1.0,
                "cohesion": 1.0,
                "spacing": 1.0,
                "nematic": 0.0,
                "silhouette": 0.0,
                "speed_match": 1.0,
                "boundary": 1.0,
                "altitude": 0.0,
                "energy": 0.0,
            },
            faithful_signs=True,
        )
        reward = compute_reward(m, config)
        expected_max = sum(config.weights.values())
        # Random should be well below max
        assert reward < expected_max * 0.8, (
            f"Random flock reward {reward:.3f} should be < {expected_max*0.8:.1f}"
        )

    def test_faithful_signs_flips_sign(self):
        """faithful_signs=False produces negative of faithful=True reward."""
        m = FlockMetrics()
        m.alpha = 0.5
        m.dispersion = 100.0
        m.velocity_deviation = 1.0
        m.boundary_overshoot = 50.0
        m.altitude_deviation = 50.0

        config_faithful = RewardConfig(
            weights={"alignment": 1.0, "cohesion": 1.0},
            faithful_signs=True,
        )
        config_cost = RewardConfig(
            weights={"alignment": 1.0, "cohesion": 1.0},
            faithful_signs=False,
        )

        r_faithful = compute_reward(m, config_faithful)
        r_cost = compute_reward(m, config_cost)

        assert r_cost == pytest.approx(-r_faithful), (
            f"Cost mode {r_cost} should equal -faithful {r_faithful}"
        )

    def test_per_term_weight_linearity(self):
        """Doubling all weights exactly doubles the reward."""
        m = FlockMetrics()
        m.alpha = 0.7
        m.dispersion = 50.0
        m.local_spacing = 12.0
        m.silhouette_2d = 0.3
        m.velocity_deviation = 0.5
        m.boundary_overshoot = 20.0
        m.altitude_deviation = 30.0
        m.energy_J = 0.1

        # Use two identical instances for the linearity check
        m2 = FlockMetrics()
        m2.alpha = 0.7
        m2.dispersion = 50.0
        m2.local_spacing = 12.0
        m2.silhouette_2d = 0.3
        m2.velocity_deviation = 0.5
        m2.boundary_overshoot = 20.0
        m2.altitude_deviation = 30.0
        m2.energy_J = 0.1

        assert reward_linearity_check(m, m2), "Weight linearity should hold"

    def test_zero_weight_terms_ignored(self):
        """Terms with weight=0 contribute nothing to the reward."""
        m = FlockMetrics()
        m.alpha = 1.0
        m.dispersion = 0.0
        m.velocity_deviation = 0.0

        # alignment=0, cohesion=1
        config_cohesion_only = RewardConfig(
            weights={"alignment": 0.0, "cohesion": 1.0},
            faithful_signs=True,
        )
        r_cohesion = compute_reward(m, config_cohesion_only)

        # alignment=0.5, cohesion=1 (different alignment should not matter if weight=0)
        m_alt = FlockMetrics()
        m_alt.alpha = 0.1  # low alignment
        m_alt.dispersion = 0.0
        m_alt.velocity_deviation = 0.0

        config_alignment_zero = RewardConfig(
            weights={"alignment": 0.0, "cohesion": 1.0},
            faithful_signs=True,
        )
        r_alt = compute_reward(m_alt, config_alignment_zero)

        # Both should be identical since alignment weight=0
        assert r_cohesion == pytest.approx(r_alt), (
            "Zero-weighted terms should not affect reward"
        )

    def test_baseline_subtracted(self):
        """Baseline is subtracted from the final reward."""
        m = FlockMetrics()
        m.alpha = 0.5
        m.dispersion = 100.0
        m.velocity_deviation = 1.0
        m.boundary_overshoot = 50.0
        m.altitude_deviation = 50.0

        config_no_baseline = RewardConfig(
            weights={"alignment": 1.0, "cohesion": 1.0},
            baseline=0.0,
        )
        config_with_baseline = RewardConfig(
            weights={"alignment": 1.0, "cohesion": 1.0},
            baseline=0.5,
        )

        r_no = compute_reward(m, config_no_baseline)
        r_with = compute_reward(m, config_with_baseline)

        assert r_with == pytest.approx(r_no - 0.5), (
            f"Baseline should subtract 0.5: {r_no} → {r_with}"
        )

    def test_reward_terms_vector(self):
        """RewardTerms.vector returns correctly ordered numpy array."""
        terms = RewardTerms(
            alignment=0.9,
            cohesion=0.8,
            spacing=0.5,
            nematic=0.7,
            silhouette=0.6,
            speed_match=0.4,
            boundary=0.3,
            altitude=0.2,
            energy=0.1,
        )
        v = terms.vector
        assert isinstance(v, np.ndarray)
        assert len(v) == 9
        # Check first and last match
        assert v[0] == pytest.approx(0.9)
        assert v[-1] == pytest.approx(0.1)

    def test_default_config_reward_positive(self):
        """Default RewardConfig produces a non-negative reward for a typical flock."""
        m = FlockMetrics()
        m.alpha = 0.6
        m.dispersion = 80.0
        m.velocity_deviation = 1.5
        m.boundary_overshoot = 100.0
        m.altitude_deviation = 80.0
        m.local_spacing = 12.0
        m.silhouette_2d = 0.2

        reward = compute_reward(m)
        assert reward >= 0.0, f"Default reward should be >= 0, got {reward}"
        assert np.isfinite(reward), f"Reward should be finite, got {reward}"


# ── P9.9: Edge cases ───────────────────────────────────────────

def test_max_reward_clipping():
    """P9.9: max_reward clips the output."""
    m = FlockMetrics()
    m.alpha = 1.0
    m.nematic_S = 1.0
    m.dispersion = 0.0
    m.velocity_deviation = 0.0
    m.boundary_overshoot = 0.0
    m.altitude_deviation = 0.0
    m.silhouette_2d = 1.0
    m.local_spacing = 15.0
    m.energy_J = 0.0

    config = RewardConfig(
        weights={"alignment": 1.0, "cohesion": 1.0},
        max_reward=1.0,
    )
    reward = compute_reward(m, config)
    # Perfect flock should give 2.0, but clipped to 1.0
    assert reward == pytest.approx(1.0), (
        f"Clip should limit to 1.0, got {reward:.4f}"
    )


def test_reward_linearity_fails_for_different_metrics():
    """P9.9: reward_linearity_check returns False when metrics differ."""
    m1 = FlockMetrics()
    m1.alpha = 0.8
    m1.dispersion = 50.0

    m2 = FlockMetrics()
    m2.alpha = 0.3  # different!
    m2.dispersion = 50.0

    # Same weights but different metrics → linearity should fail
    assert not reward_linearity_check(m1, m2), (
        "Different metrics should fail linearity check"
    )


def test_empty_weights_yields_zero():
    """P9.9: All-zero weights → reward = 0."""
    m = FlockMetrics()
    m.alpha = 0.9
    m.dispersion = 10.0

    config = RewardConfig(
        weights={"alignment": 0.0, "cohesion": 0.0, "spacing": 0.0},
    )
    reward = compute_reward(m, config)
    assert reward == pytest.approx(0.0), f"Zero weights → zero reward, got {reward}"
