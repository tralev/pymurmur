"""Unit tests for physics.boundary — BoundaryMode ABC, BOUNDARY_REGISTRY,
and boid.py's registry-based _apply_boundary dispatch.

Modularity pass: formalises the 5 boundary strategies (previously a
plain if/elif in boid.py) behind a registry mirroring ForceMode's
proven pattern. These tests verify the registry itself plus that
_apply_boundary's dispatch reproduces each strategy's direct-call
behavior exactly (pure extraction, no behavior change).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.physics.boid import _apply_boundary
from pymurmur.physics.boundary import (
    BOUNDARY_REGISTRY,
    BoundaryMode,
    MarginBoundary,
    OpenBoundary,
    SphereBoundary,
    SphereSoftBoundary,
    ToroidalBoundary,
)


class TestBoundaryRegistry:
    def test_all_five_strategies_registered(self):
        assert set(BOUNDARY_REGISTRY.keys()) == {
            "toroidal", "open", "margin", "sphere", "sphere_soft",
        }

    def test_registered_classes_are_boundary_mode_subclasses(self):
        for cls in BOUNDARY_REGISTRY.values():
            assert issubclass(cls, BoundaryMode)

    def test_name_classvar_set_by_register(self):
        assert BOUNDARY_REGISTRY["toroidal"].name == "toroidal"
        assert BOUNDARY_REGISTRY["sphere_soft"].name == "sphere_soft"

    def test_registry_maps_to_expected_classes(self):
        assert BOUNDARY_REGISTRY["toroidal"] is ToroidalBoundary
        assert BOUNDARY_REGISTRY["open"] is OpenBoundary
        assert BOUNDARY_REGISTRY["margin"] is MarginBoundary
        assert BOUNDARY_REGISTRY["sphere"] is SphereBoundary
        assert BOUNDARY_REGISTRY["sphere_soft"] is SphereSoftBoundary


def _scenario(mode_extra_offset=0.0):
    """Shared fixture: birds positioned to exercise each boundary's
    active branch (near/outside walls or sphere radius)."""
    positions = np.array([
        [5.0, 5.0, 5.0],       # near low-corner wall
        [995.0, 695.0, 395.0],  # near high-corner wall
        [500.0, 350.0, 200.0],  # centre, unaffected
    ], dtype=np.float32) + mode_extra_offset
    velocities = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
    ], dtype=np.float32)
    active = np.ones(3, dtype=bool)
    return positions, velocities, active


class TestApplyBoundaryDispatchMatchesDirectCall:
    """_apply_boundary(mode=...) must reproduce calling the registered
    strategy's .apply() directly, byte-for-byte, for every mode."""

    WIDTH, HEIGHT, DEPTH = 1000.0, 700.0, 400.0
    SPHERE_RADIUS = 300.0
    AVOIDANCE_FACTOR = 0.05
    CENTER = np.array([500.0, 350.0, 200.0], dtype=np.float32)

    @pytest.mark.parametrize("mode", ["toroidal", "open", "margin", "sphere", "sphere_soft"])
    def test_dispatch_matches_direct_strategy_call(self, mode):
        pos_a, vel_a, active = _scenario()
        pos_b, vel_b, _ = _scenario()

        _apply_boundary(
            pos_a, vel_a, active, self.WIDTH, self.HEIGHT, self.DEPTH,
            mode, self.SPHERE_RADIUS, self.AVOIDANCE_FACTOR, center=self.CENTER,
        )
        BOUNDARY_REGISTRY[mode].apply(
            pos_b, vel_b, active, self.WIDTH, self.HEIGHT, self.DEPTH,
            self.SPHERE_RADIUS, self.AVOIDANCE_FACTOR, center=self.CENTER,
        )

        np.testing.assert_array_equal(pos_a, pos_b)
        np.testing.assert_array_equal(vel_a, vel_b)

    def test_unrecognized_mode_falls_back_to_toroidal(self):
        pos_a, vel_a, active = _scenario()
        pos_b, vel_b, _ = _scenario()

        _apply_boundary(
            pos_a, vel_a, active, self.WIDTH, self.HEIGHT, self.DEPTH,
            "bogus_mode", self.SPHERE_RADIUS, self.AVOIDANCE_FACTOR, center=self.CENTER,
        )
        BOUNDARY_REGISTRY["toroidal"].apply(
            pos_b, vel_b, active, self.WIDTH, self.HEIGHT, self.DEPTH,
            self.SPHERE_RADIUS, self.AVOIDANCE_FACTOR, center=self.CENTER,
        )

        np.testing.assert_array_equal(pos_a, pos_b)
        np.testing.assert_array_equal(vel_a, vel_b)


class TestBoundaryStrategyBehavior:
    """Spot-check each strategy's actual math, independent of dispatch —
    guards against a future strategy swap silently changing behavior."""

    def test_toroidal_wraps_positions(self):
        positions = np.array([[1050.0, -20.0, 410.0]], dtype=np.float32)
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        ToroidalBoundary.apply(positions, velocities, active, 1000.0, 700.0, 400.0, 300.0, 0.05)
        assert 0.0 <= positions[0, 0] < 1000.0
        assert 0.0 <= positions[0, 1] < 700.0
        assert 0.0 <= positions[0, 2] < 400.0

    def test_open_never_modifies_arrays(self):
        positions = np.array([[-500.0, 2000.0, -100.0]], dtype=np.float32)
        pos_before = positions.copy()
        velocities = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        vel_before = velocities.copy()
        active = np.array([True])
        OpenBoundary.apply(positions, velocities, active, 1000.0, 700.0, 400.0, 300.0, 0.05)
        np.testing.assert_array_equal(positions, pos_before)
        np.testing.assert_array_equal(velocities, vel_before)

    def test_margin_pushes_velocity_away_from_wall(self):
        positions = np.array([[5.0, 350.0, 200.0]], dtype=np.float32)
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        MarginBoundary.apply(positions, velocities, active, 1000.0, 700.0, 400.0, 300.0, 0.05)
        assert velocities[0, 0] > 0  # pushed toward +x, away from low wall

    def test_sphere_hard_projects_outside_birds_onto_surface(self):
        center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        positions = np.array([[500.0, 350.0, 700.0]], dtype=np.float32)  # far outside radius=300
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        SphereBoundary.apply(positions, velocities, active, 1000.0, 700.0, 400.0, 300.0, 0.05, center=center)
        dist = np.linalg.norm(positions[0] - center)
        assert dist == pytest.approx(300.0, abs=1e-3)

    def test_sphere_soft_never_hard_clamps_position(self):
        center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        positions = np.array([[500.0, 350.0, 650.0]], dtype=np.float32)  # outside radius=300
        pos_before = positions.copy()
        velocities = np.zeros((1, 3), dtype=np.float32)
        active = np.array([True])
        SphereSoftBoundary.apply(positions, velocities, active, 1000.0, 700.0, 400.0, 300.0, 0.05, center=center)
        np.testing.assert_array_equal(positions, pos_before)  # no position clamp
        assert velocities[0, 2] < 0  # pushed inward (-z, toward centre)
