"""S3.9 — Rewards module tests.

Five-term penalty-composite reward:
    R = ±w_a*velocity_deviation - w_c*dispersion - w_L*|L|/N
        - w_b*boundary_overshoot - w_z*altitude_deviation
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import FlockMetrics
from pymurmur.analysis.rewards import (
    RewardConfig,
    compute_reward,
    reward_linearity_check,
)


def _perfect_metrics() -> FlockMetrics:
    m = FlockMetrics()
    m.velocity_deviation = 0.0
    m.dispersion = 0.0
    m.angular_momentum = np.zeros(3, dtype=np.float32)
    m.boundary_overshoot = 0.0
    m.altitude_deviation = 0.0
    return m


class TestRewardComputation:
    """Five-term penalty-composite reward from FlockMetrics."""

    def test_perfect_flock_corrected_reward_is_zero(self):
        """Perfect flock (no deviation/dispersion/rotation/overshoot) →
        reward is exactly 0 under faithful_signs=False (the maximum)."""
        m = _perfect_metrics()
        config = RewardConfig(
            w_a=1.0, w_c=1.0, w_L=1.0, w_b=1.0, w_z=1.0,
            faithful_signs=False,
        )
        reward = compute_reward(m, config)
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_imperfect_flock_corrected_reward_is_negative(self):
        """Every term is a penalty under faithful_signs=False — any
        deviation from perfect order makes the reward strictly negative."""
        m = FlockMetrics()
        m.velocity_deviation = 2.0
        m.dispersion = 10.0
        m.angular_momentum = np.zeros(3, dtype=np.float32)
        m.boundary_overshoot = 0.0
        m.altitude_deviation = 0.0

        config = RewardConfig(w_a=1.0, w_c=1.0, faithful_signs=False)
        reward = compute_reward(m, config)
        assert reward == pytest.approx(-12.0)
        assert reward < 0.0

    def test_faithful_signs_flips_only_alignment_term(self):
        """faithful_signs=True flips only the velocity_deviation term's
        sign — every other term stays a penalty in both modes (this is
        the source's documented quirk, not a full sign flip)."""
        m = FlockMetrics()
        m.velocity_deviation = 2.0
        m.dispersion = 10.0
        m.angular_momentum = np.zeros(3, dtype=np.float32)
        m.boundary_overshoot = 0.0
        m.altitude_deviation = 0.0

        r_faithful = compute_reward(m, RewardConfig(w_a=1.0, w_c=1.0, faithful_signs=True))
        r_corrected = compute_reward(m, RewardConfig(w_a=1.0, w_c=1.0, faithful_signs=False))

        assert r_faithful == pytest.approx(2.0 - 10.0)   # +w_a*2 - w_c*10
        assert r_corrected == pytest.approx(-2.0 - 10.0)  # -w_a*2 - w_c*10
        # Not a full negation — only the alignment term differs
        assert r_faithful != pytest.approx(-r_corrected)

    def test_angular_momentum_term_uses_magnitude(self):
        """The w_L term penalizes ‖angular_momentum‖, regardless of axis."""
        m = _perfect_metrics()
        m.angular_momentum = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # |L| = 5

        config = RewardConfig(w_a=0, w_c=0, w_L=1.0, w_b=0, w_z=0, faithful_signs=False)
        reward = compute_reward(m, config)
        assert reward == pytest.approx(-5.0)

    def test_boundary_and_altitude_terms(self):
        m = _perfect_metrics()
        m.boundary_overshoot = 7.0
        m.altitude_deviation = 3.0

        config = RewardConfig(w_a=0, w_c=0, w_L=0, w_b=2.0, w_z=5.0, faithful_signs=False)
        reward = compute_reward(m, config)
        assert reward == pytest.approx(-2.0 * 7.0 - 5.0 * 3.0)

    def test_zero_weight_terms_ignored(self):
        """Terms with weight=0 contribute nothing to the reward."""
        m = _perfect_metrics()
        m.dispersion = 999.0  # would dominate if w_c weren't 0

        config = RewardConfig(w_a=0.0, w_c=0.0, w_L=0.0, w_b=0.0, w_z=0.0)
        reward = compute_reward(m, config)
        assert reward == pytest.approx(0.0)

    def test_default_config_matches_spec_defaults(self):
        """Default RewardConfig: w_a=w_c=1, extension terms (w_L/w_b/w_z) = 0."""
        config = RewardConfig()
        assert config.w_a == 1.0
        assert config.w_c == 1.0
        assert config.w_L == 0.0
        assert config.w_b == 0.0
        assert config.w_z == 0.0
        assert config.faithful_signs is True


class TestRewardLinearity:
    def test_per_term_weight_linearity(self):
        """Doubling all weights exactly doubles the reward for an
        identical metrics snapshot."""
        m = FlockMetrics()
        m.velocity_deviation = 0.5
        m.dispersion = 50.0
        m.angular_momentum = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m.boundary_overshoot = 20.0
        m.altitude_deviation = 30.0

        m2 = FlockMetrics()
        m2.velocity_deviation = 0.5
        m2.dispersion = 50.0
        m2.angular_momentum = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m2.boundary_overshoot = 20.0
        m2.altitude_deviation = 30.0

        config = RewardConfig(w_a=1.0, w_c=1.0, w_L=0.5, w_b=0.3, w_z=0.2)
        assert reward_linearity_check(m, m2, config)

    def test_reward_linearity_fails_for_different_metrics(self):
        m1 = FlockMetrics()
        m1.velocity_deviation = 0.8
        m1.dispersion = 50.0

        m2 = FlockMetrics()
        m2.velocity_deviation = 0.1  # different!
        m2.dispersion = 50.0

        assert not reward_linearity_check(m1, m2)


class TestAngularMomentumCoMCentering:
    """S3.9: MetricsCollector.angular_momentum must be centered on the
    flock's centre of mass, not the domain origin — otherwise its
    magnitude wouldn't match the reward spec's
    ‖Σᵢ(pᵢ−CoM)×vᵢ‖/N term."""

    def test_angular_momentum_invariant_to_domain_translation(self):
        """Pure translation of the whole flock (same relative motion)
        must not change the CoM-centered angular momentum — an
        origin-centered computation would fail this."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        rng = np.random.default_rng(3)
        N = 30
        radius = 50.0
        angles = rng.uniform(0, 2 * np.pi, N).astype(np.float32)

        def make_flock(offset):
            cfg = SimConfig()
            cfg.num_boids = N
            flock = PhysicsFlock(cfg)
            flock.positions[:, 0] = np.cos(angles) * radius + offset[0]
            flock.positions[:, 1] = np.sin(angles) * radius + offset[1]
            flock.positions[:, 2] = offset[2]
            flock.velocities[:, 0] = -np.sin(angles) * radius
            flock.velocities[:, 1] = np.cos(angles) * radius
            flock.velocities[:, 2] = 0.0
            flock.active[:] = True
            return flock

        flock_a = make_flock((500.0, 350.0, 200.0))
        flock_b = make_flock((5000.0, 8000.0, 200.0))  # far translated

        col_a = MetricsCollector()
        col_a.collect(flock_a, 0)
        col_b = MetricsCollector()
        col_b.collect(flock_b, 0)

        L_a = col_a.snapshot().angular_momentum
        L_b = col_b.snapshot().angular_momentum
        assert np.allclose(L_a, L_b, atol=1e-2), (
            f"CoM-centered angular momentum must be translation-invariant: "
            f"{L_a} vs {L_b}"
        )
