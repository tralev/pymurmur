"""Phase 1 acceptance-criterion tests (P1.8-P1.10): vicsek memory term,
flock-shape thickness ratio, Θ NaN-in-non-projection-mode.

Split out of test_phase1_p6_p10.py (file-size split) — P1.6/P1.7
(steric/cohesion) tests stay in the original.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pymurmur.analysis.metrics import MetricsCollector, compute_shape
from test.helpers import _call_force  # noqa: E402

# ── Vicsek memory-term autocorrelation (P1.8) ────────────

@pytest.mark.phase1
class TestVicsekMemory:
    """P1.8: Vicsek memory term preserves direction exactly when D=0."""

    @staticmethod
    def _make_flock(num_boids: int = 1, diffusion: float = 0.0):
        """Create a PhysicsFlock + SimConfig pair in vicsek mode."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = num_boids
        cfg.seed = 42
        cfg.vicsek_diffusion = diffusion
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.5
        cfg.vicsek_time_step = 0.1
        return PhysicsFlock(cfg), cfg

    def test_d_zero_single_bird_direction_preserved(self) -> None:
        """D=0, single bird: direction unchanged after one vicsek step."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=1, diffusion=0.0)

        # Set known direction
        known_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        flock.velocities[0] = known_dir * cfg.vicsek_velocity

        _call_force(vicsek_forces, flock, cfg)

        new_vel = flock.velocities[0]
        new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
        dot = float(np.dot(new_dir, known_dir))

        assert dot > 0.999999, (
            f"Direction drifted: dot(new, old) = {dot:.12f}, expected > 0.999999"
        )
        # Speed should remain constant at v0
        speed = float(np.linalg.norm(new_vel))
        assert abs(speed - cfg.vicsek_velocity) < 1e-6, (
            f"Speed changed: {speed:.6f}, expected {cfg.vicsek_velocity}"
        )

    def test_d_zero_repeated_frames_no_drift(self) -> None:
        """D=0 over 100 frames: direction never drifts."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=1, diffusion=0.0)

        known_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        flock.velocities[0] = known_dir.astype(np.float32) * cfg.vicsek_velocity

        for _ in range(100):
            _call_force(vicsek_forces, flock, cfg)
            new_vel = flock.velocities[0]
            new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
            dot = float(np.dot(new_dir, known_dir))
            assert dot > 0.999999, (
                f"Drift at frame: dot(new, old) = {dot:.12f}"
            )
            # Speed should remain constant
            speed = float(np.linalg.norm(new_vel))
            assert abs(speed - cfg.vicsek_velocity) < 1e-6

    def test_d_zero_two_birds_opposite_directions_preserved(self) -> None:
        """D=0, 2 birds heading opposite ways: multi-bird path,
        neighbour average cancels → falls back to pure memory,
        both directions preserved independently."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=2, diffusion=0.0)

        # Place birds close together so they see each other
        flock.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        flock.positions[1] = np.array([2.0, 0.0, 0.0], dtype=np.float32)

        # Opposite directions: +x and -x
        dir0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dir1 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        flock.velocities[0] = dir0 * cfg.vicsek_velocity
        flock.velocities[1] = dir1 * cfg.vicsek_velocity

        _call_force(vicsek_forces, flock, cfg)

        # Both directions should be preserved (opposite avg → zero → no blend)
        for i, expected_dir in enumerate([dir0, dir1]):
            new_vel = flock.velocities[i]
            new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
            dot = float(np.dot(new_dir, expected_dir))
            assert dot > 0.999999, (
                f"Bird {i}: dot(new, old) = {dot:.12f}, expected > 0.999999"
            )
            speed = float(np.linalg.norm(new_vel))
            assert abs(speed - cfg.vicsek_velocity) < 1e-6, (
                f"Bird {i}: speed = {speed:.6f}, expected {cfg.vicsek_velocity}"
            )

    def test_d_zero_two_birds_same_direction_preserved(self) -> None:
        """D=0, 2 birds heading same direction: neighbour blend should
        not change the direction since û_target = û_old (aligned)."""
        from pymurmur.physics.forces.vicsek import vicsek_forces

        flock, cfg = self._make_flock(num_boids=2, diffusion=0.0)

        flock.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        flock.positions[1] = np.array([2.0, 0.0, 0.0], dtype=np.float32)

        # Same direction
        dir0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        flock.velocities[0] = dir0 * cfg.vicsek_velocity
        flock.velocities[1] = dir0 * cfg.vicsek_velocity

        _call_force(vicsek_forces, flock, cfg)

        # Since neighbour average equals each bird's own direction,
        # the blend eta*nd + (1-eta)*noisy = eta*dir + (1-eta)*dir = dir.
        # With D=0, noisy_dirs = old_dirs = dir. Direction preserved.
        for i in range(2):
            new_vel = flock.velocities[i]
            new_dir = new_vel / (np.linalg.norm(new_vel) + 1e-10)
            dot = float(np.dot(new_dir, dir0))
            assert dot > 0.999999, (
                f"Bird {i}: dot(new, old) = {dot:.12f}, expected > 0.999999"
            )


# ── Thickness ratio √(λ₃/λ₁) ∈ (0,1] (P1.9) ──────────────

@pytest.mark.phase1
class TestThicknessRatio:
    """P1.9: thickness = sqrt(λ₃/λ₁) ∈ (0,1]; → 1 for spheres, → 0 for lines."""

    def test_sphere_thickness_near_one(self) -> None:
        """100 random points on a 3D sphere → thickness ≈ 1."""
        rng = np.random.default_rng(42)
        # Generate points uniformly on unit sphere, scale to radius 50
        pts = rng.normal(size=(100, 3)).astype(np.float32)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 50.0

        _, thickness = compute_shape(pts)

        assert 0.8 < thickness <= 1.0, (
            f"Sphere thickness = {thickness:.3f}, expected > 0.8 (nearly isotropic)"
        )

    def test_line_thickness_near_zero(self) -> None:
        """50 points along x-axis with tiny noise → thickness ≈ 0."""
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            np.linspace(0, 100, 50),
            rng.normal(0, 1e-3, 50),
            rng.normal(0, 1e-3, 50),
        ]).astype(np.float32)

        _, thickness = compute_shape(positions)

        # P1.9: λ₃ ≪ λ₁ for a line → thickness ≈ 0
        assert 0.0 < thickness < 0.2, (
            f"Line thickness = {thickness:.3f}, expected < 0.2"
        )

    def test_pancake_thickness_near_zero(self) -> None:
        """Wide pancake with thin z → large aspect, thickness ≈ 0."""
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            rng.uniform(-200, 200, 100),
            rng.uniform(-200, 200, 100),
            rng.normal(0, 1e-3, 100),
        ]).astype(np.float32)

        _, thickness = compute_shape(positions)

        # P1.9: thin z → λ₃ ≪ λ₁ → thickness ≈ 0
        assert 0.0 < thickness < 0.1, (
            f"Pancake thickness = {thickness:.3f}, expected < 0.1"
        )

    def test_less_than_three_points_returns_one(self) -> None:
        """N < 3 → returns (1.0, 1.0) per the degenerate guard."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        aspect, thickness = compute_shape(positions)

        assert aspect == 1.0
        assert thickness == 1.0


# ── Θ NaN in non-projection modes (P1.10) ─────────────────

@pytest.mark.phase1
class TestThetaNaN:
    """P1.10: Θ returns NaN when mode is not 'projection'."""

    @staticmethod
    def _make_dummy_flock(mode: str):
        """Create a PhysicsFlock + MetricsCollector pair for mode."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = mode
        cfg.num_boids = 5
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)
        return flock, collector

    def test_theta_is_nan_in_non_projection_mode(self) -> None:
        """Vicsek mode → m.theta should be NaN."""
        flock, collector = self._make_dummy_flock("vicsek")

        # Set some last_theta values (should be ignored in vicsek mode)
        flock.last_theta[:] = 0.5

        collector.collect(flock, 0)
        m = collector.snapshot()

        assert math.isnan(m.theta), (
            f"Θ in vicsek mode should be NaN, got {m.theta}"
        )

    def test_theta_is_finite_in_projection_mode(self) -> None:
        """Projection mode → m.theta should be a finite number."""
        flock, collector = self._make_dummy_flock("projection")

        # Set last_theta so the mean is meaningful
        flock.last_theta[:] = 0.3

        collector.collect(flock, 0)
        m = collector.snapshot()

        assert not math.isnan(m.theta), (
            "Θ in projection mode should be finite, got NaN"
        )
        assert math.isfinite(m.theta), (
            f"Θ in projection mode should be finite, got {m.theta}"
        )
        # Mean of [0.3, 0.3, 0.3, 0.3, 0.3] = 0.3
        assert abs(m.theta - 0.3) < 1e-6, (
            f"Θ should be mean of last_theta = 0.3, got {m.theta:.6f}"
        )
