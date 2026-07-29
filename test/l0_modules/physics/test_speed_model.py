"""Unit tests for physics.speed_model — SPEED_MODEL_REGISTRY, mirroring
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

from pymurmur.physics.speed_model import SPEED_MODEL_REGISTRY, SpeedModel


class TestSpeedModelRegistry:
    def test_all_five_keys_registered(self):
        assert set(SPEED_MODEL_REGISTRY.keys()) == {
            "band", "clamp", "fixed", "ceiling", "none",
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
