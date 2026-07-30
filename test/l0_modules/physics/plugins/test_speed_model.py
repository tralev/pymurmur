"""Unit tests for physics.plugins.speed_model — SPEED_MODEL_REGISTRY, mirroring
test_boundary_registry.py's shape (registry-membership assertions plus
per-strategy apply() behavior for a modularity-pass registry).

Modularity pass 4: extracts the 4-tier if/elif speed-enforcement chain
from boid.py::integrate() (band/clamp, fixed, ceiling, none) behind a
registry of SpeedModel ABC subclasses. These tests verify registry
contents and each strategy's clamp/renorm/passthrough math; boid.py's
actual dispatch (SPEED_MODEL_REGISTRY.get(speed_mode, ...["band"])) is
covered by test_boid*.py.
"""

from __future__ import annotations

import numpy as np

from pymurmur.physics.plugins.speed_model import SPEED_MODEL_REGISTRY, SpeedModel


class TestSpeedModelRegistry:
    def test_all_seven_keys_registered(self):
        assert set(SPEED_MODEL_REGISTRY.keys()) == {
            "band", "clamp", "fixed", "ceiling", "none",
            "noise_modulated", "velocity_adaptive",
        }

    def test_clamp_aliases_band(self):
        """"clamp" is config vocabulary compat for "band" — same class."""
        assert SPEED_MODEL_REGISTRY["clamp"] is SPEED_MODEL_REGISTRY["band"]

    def test_entries_are_speedmodel_subclasses(self):
        for strategy in SPEED_MODEL_REGISTRY.values():
            assert issubclass(strategy, SpeedModel)

    def test_apply_is_callable_on_every_entry(self):
        for strategy in SPEED_MODEL_REGISTRY.values():
            assert callable(strategy.apply)


class TestBandSpeedModel:
    """band/clamp: clamp to [min_speed, caps]."""

    def test_too_fast_scaled_down_to_cap(self):
        velocities = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["band"].apply(velocities, active, caps, min_speed, speeds)

        assert np.isclose(np.linalg.norm(velocities[0]), 4.0)

    def test_too_slow_boosted_to_floor(self):
        velocities = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["band"].apply(velocities, active, caps, min_speed, speeds)

        assert np.isclose(np.linalg.norm(velocities[0]), 1.0)

    def test_within_band_left_unchanged(self):
        velocities = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["band"].apply(velocities, active, caps, min_speed, speeds)

        assert np.allclose(velocities[0], [2.0, 0.0, 0.0])

    def test_inactive_bird_ignored_even_out_of_band(self):
        velocities = np.array([[99.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([False])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["band"].apply(velocities, active, caps, min_speed, speeds)

        assert np.allclose(velocities[0], [99.0, 0.0, 0.0])

    def test_returns_none(self):
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.zeros((1, 1), dtype=np.float32)

        result = SPEED_MODEL_REGISTRY["band"].apply(velocities, active, caps, min_speed, speeds)
        assert result is None


class TestFixedSpeedModel:
    """fixed: exact renormalisation to caps, deterministic zero-vel fallback."""

    def test_renormalizes_to_cap_exactly(self):
        velocities = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([0.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["fixed"].apply(velocities, active, caps, min_speed, speeds)

        assert np.isclose(np.linalg.norm(velocities[0]), 4.0)

    def test_slow_velocity_also_renormalized_up_to_cap(self):
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([0.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["fixed"].apply(velocities, active, caps, min_speed, speeds)

        assert np.isclose(np.linalg.norm(velocities[0]), 4.0)

    def test_zero_velocity_gets_deterministic_direction(self):
        """Zero-velocity birds get direction (1, 0, 0) to avoid NaN."""
        velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([0.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["fixed"].apply(velocities, active, caps, min_speed, speeds)

        assert np.allclose(velocities[0], [4.0, 0.0, 0.0])
        assert np.isfinite(velocities).all()

    def test_returns_none(self):
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([0.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        result = SPEED_MODEL_REGISTRY["fixed"].apply(velocities, active, caps, min_speed, speeds)
        assert result is None


class TestCeilingSpeedModel:
    """ceiling: cap only, no floor — slow speeds pass through."""

    def test_too_fast_scaled_down_to_cap(self):
        velocities = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["ceiling"].apply(velocities, active, caps, min_speed, speeds)

        assert np.isclose(np.linalg.norm(velocities[0]), 4.0)

    def test_slow_velocity_passes_through_unclamped(self):
        """No lower bound — boids can drift arbitrarily slow, unlike band."""
        velocities = np.array([[0.01, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["ceiling"].apply(velocities, active, caps, min_speed, speeds)

        assert np.allclose(velocities[0], [0.01, 0.0, 0.0])

    def test_returns_none(self):
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.zeros((1, 1), dtype=np.float32)

        result = SPEED_MODEL_REGISTRY["ceiling"].apply(velocities, active, caps, min_speed, speeds)
        assert result is None


class TestNoneSpeedModel:
    """none: no speed enforcement — velocities pass through unchanged."""

    def test_velocities_unchanged(self):
        velocities = np.array([[123.0, -45.0, 6.0]], dtype=np.float32)
        original = velocities.copy()
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["none"].apply(velocities, active, caps, min_speed, speeds)

        assert np.array_equal(velocities, original)

    def test_returns_none(self):
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.zeros((1, 1), dtype=np.float32)

        result = SPEED_MODEL_REGISTRY["none"].apply(velocities, active, caps, min_speed, speeds)
        assert result is None


class TestNoiseModulatedSpeedModel:
    """noise_modulated: speed cap varies with a deterministic 3D noise
    field sampled at each bird's position."""

    def test_without_positions_degrades_to_band(self):
        """A caller that forgets to pass positions still gets a real
        speed policy (band), not a silent no-op."""
        velocities = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["noise_modulated"].apply(
            velocities, active, caps, min_speed, speeds
        )
        assert np.isclose(np.linalg.norm(velocities[0]), 4.0)

    def test_finite_and_within_extended_band(self):
        """With positions supplied, speed stays within [min_speed,
        2*caps] (the taxonomy's lerp(0.5, 2.0, ...) multiplier range)."""
        rng = np.random.default_rng(1)
        n = 200
        velocities = rng.normal(size=(n, 3)).astype(np.float32) * 3.0
        positions = rng.uniform(-500, 500, size=(n, 3)).astype(np.float32)
        active = np.ones(n, dtype=bool)
        caps = np.full(n, 4.0, dtype=np.float32)
        min_speed = np.full(n, 1.0, dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["noise_modulated"].apply(
            velocities, active, caps, min_speed, speeds, positions=positions,
        )
        result_speeds = np.linalg.norm(velocities, axis=1)
        assert np.isfinite(velocities).all()
        assert (result_speeds <= 2.0 * caps + 1e-4).all()
        assert (result_speeds >= min_speed - 1e-4).all()

    def test_different_positions_produce_different_caps(self):
        """Two birds with identical velocity but different positions
        must be able to end up at different speeds — proves the noise
        field is actually position-dependent, not a uniform multiplier."""
        velocities = np.array(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32
        )
        positions = np.array(
            [[0.0, 0.0, 0.0], [400.0, 250.0, 130.0]], dtype=np.float32
        )
        active = np.array([True, True])
        caps = np.array([4.0, 4.0], dtype=np.float32)
        min_speed = np.array([1.0, 1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["noise_modulated"].apply(
            velocities, active, caps, min_speed, speeds, positions=positions,
        )
        result_speeds = np.linalg.norm(velocities, axis=1)
        assert not np.isclose(result_speeds[0], result_speeds[1]), (
            "identical velocity + different positions should generally "
            "clamp to different speeds under a position-dependent field"
        )

    def test_inactive_bird_ignored(self):
        velocities = np.array([[99.0, 0.0, 0.0]], dtype=np.float32)
        positions = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        active = np.array([False])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["noise_modulated"].apply(
            velocities, active, caps, min_speed, speeds, positions=positions,
        )
        assert np.allclose(velocities[0], [99.0, 0.0, 0.0])

    def test_returns_none(self):
        velocities = np.zeros((1, 3), dtype=np.float32)
        positions = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.zeros((1, 1), dtype=np.float32)

        result = SPEED_MODEL_REGISTRY["noise_modulated"].apply(
            velocities, active, caps, min_speed, speeds, positions=positions,
        )
        assert result is None


class TestVelocityAdaptiveSpeedModel:
    """velocity_adaptive: speed smoothly approaches a randomized-bonus
    target via exponential lerp."""

    def test_without_rng_uses_bonus_one(self):
        """No rng -> bonus defaults to 1.0, goal = caps exactly."""
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["velocity_adaptive"].apply(
            velocities, active, caps, min_speed, speeds, dt=1.0,
        )
        # LERP_RATE=3.0, dt=1.0 -> alpha=clip(3.0,0,1)=1.0 -> fully at goal
        assert np.isclose(np.linalg.norm(velocities[0]), 4.0, atol=1e-4)

    def test_zero_dt_is_noop(self):
        """dt=0 -> alpha=0 -> velocity unchanged this frame."""
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        original = velocities.copy()
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["velocity_adaptive"].apply(
            velocities, active, caps, min_speed, speeds, dt=0.0,
        )
        assert np.allclose(velocities[0], original[0])

    def test_partial_lerp_moves_toward_goal_not_past_it(self):
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["velocity_adaptive"].apply(
            velocities, active, caps, min_speed, speeds, dt=1.0 / 60.0,
        )
        new_speed = np.linalg.norm(velocities[0])
        assert 1.0 < new_speed < 4.0

    def test_bonus_varies_with_rng(self):
        """With an rng supplied, repeated calls at full alpha (dt large)
        must be able to land at different final speeds across birds —
        proves the randomized bonus is actually applied, not a no-op."""
        rng = np.random.default_rng(0)
        n = 50
        velocities = np.tile([1.0, 0.0, 0.0], (n, 1)).astype(np.float32)
        active = np.ones(n, dtype=bool)
        caps = np.full(n, 4.0, dtype=np.float32)
        min_speed = np.full(n, 1.0, dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["velocity_adaptive"].apply(
            velocities, active, caps, min_speed, speeds, rng=rng, dt=1.0,
        )
        result_speeds = np.linalg.norm(velocities, axis=1)
        assert result_speeds.std() > 0, "bonus should vary per bird with an rng"
        assert (result_speeds >= caps * 0.85 - 1e-3).all()
        assert (result_speeds <= caps * 1.15 + 1e-3).all()

    def test_inactive_bird_ignored(self):
        velocities = np.array([[99.0, 0.0, 0.0]], dtype=np.float32)
        active = np.array([False])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

        SPEED_MODEL_REGISTRY["velocity_adaptive"].apply(
            velocities, active, caps, min_speed, speeds, dt=1.0,
        )
        assert np.allclose(velocities[0], [99.0, 0.0, 0.0])

    def test_returns_none(self):
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        caps = np.array([4.0], dtype=np.float32)
        min_speed = np.array([1.0], dtype=np.float32)
        speeds = np.zeros((1, 1), dtype=np.float32)

        result = SPEED_MODEL_REGISTRY["velocity_adaptive"].apply(
            velocities, active, caps, min_speed, speeds, dt=1.0,
        )
        assert result is None
