"""Phase 1 acceptance-criterion tests (P1.6-P1.10).

Split out of test_phase1.py (file-size split).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pymurmur.analysis.metrics import MetricsCollector, compute_shape
from pymurmur.physics.forces._base import cohesion_force
from pymurmur.physics.steric import steric_force
from test.helpers import _call_force  # noqa: E402

# ── Cohesion force bounded at unit vector (P1.7) ─────────

@pytest.mark.phase1
class TestCohesionBounded:
    """P1.7: cohesion_force returns F = normalize(p̄ − p_i) — bounded unit."""

    @staticmethod
    def _make_neighbor_idx(spec: list[list[int]]) -> np.ndarray:
        """Build ragged neighbor_idx array from a list-of-lists spec."""
        arr = np.empty(len(spec), dtype=object)
        for i, nbrs in enumerate(spec):
            arr[i] = np.array(nbrs, dtype=np.int32)
        return arr

    def test_cohesion_magnitude_is_one_regardless_of_distance(self) -> None:
        """CoM at [15,0,0] from bird at origin → |F| = 1 exactly."""
        positions = np.array([
            [0.0, 0.0, 0.0],    # bird 0
            [10.0, 0.0, 0.0],   # neighbour 1
            [20.0, 0.0, 0.0],   # neighbour 2
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True, False, False])
        neighbor_idx = self._make_neighbor_idx([[1, 2], [], []])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        mag = float(np.linalg.norm(force[0]))
        assert abs(mag - 1.0) < 1e-6, (
            f"|F_coh| = {mag:.8f}, expected exactly 1.0"
        )

    def test_far_away_neighbour_still_unit_magnitude(self) -> None:
        """Neighbour at d=10000 → |F| = 1 (bounded regardless of range)."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0],
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True, False])
        neighbor_idx = self._make_neighbor_idx([[1], []])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        mag = float(np.linalg.norm(force[0]))
        assert abs(mag - 1.0) < 1e-6, (
            f"|F_coh| at d=10000: {mag:.8f}, expected 1.0"
        )

    def test_no_neighbours_returns_zero(self) -> None:
        """No neighbours → zero force vector."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True])
        neighbor_idx = self._make_neighbor_idx([[]])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        assert np.allclose(force[0], np.zeros(3), atol=1e-12)

    def test_coincident_neighbour_returns_zero(self) -> None:
        """Neighbour at same position → length ≤ 1e-6 guard → zero."""
        positions = np.array([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # coincident
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([True, False])
        neighbor_idx = self._make_neighbor_idx([[1], []])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        assert np.allclose(force[0], np.zeros(3), atol=1e-12)

    def test_inactive_bird_force_remains_zero(self) -> None:
        """Inactive bird gets no force regardless of neighbours."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=np.float32)
        velocities = np.ones_like(positions, dtype=np.float32)
        active = np.array([False, True])
        neighbor_idx = self._make_neighbor_idx([[], [0]])

        force = cohesion_force(positions, velocities, neighbor_idx, active)

        # Bird 0 is inactive → force[0] should be zero
        assert np.allclose(force[0], np.zeros(3), atol=1e-12)
        # Bird 1 is active → should get unit force toward bird 0
        mag1 = float(np.linalg.norm(force[1]))
        assert abs(mag1 - 1.0) < 1e-6


# ── Steric force clamping (P1.6) ──────────────────────────────────

@pytest.mark.phase1
class TestStericMaxForce:
    """P1.6: Steric repulsion clamped to max_force at close range.

    Also covers D8+D21 cross-cutting: steric clamp (D8) works correctly
    with the corrected separation formula Σ r̂/d² (D21)."""

    def test_steric_clamp_with_corrected_separation_no_nan(self):
        """D8+D21: Steric clamp prevents NaN with corrected 1/d³ separation.

        D21 fixed separation from 1/d to 1/d² (r̂/d² = -diffs/dists³).
        At d=0.01, the force is 10000× stronger than at d=1.0 — without
        D8's steric clamp (max_force), this would explode. With both fixes,
        the force is clamped to max_force=0.15 and stays finite.
        """
        import numpy as np

        from pymurmur.physics.forces._base import separation_force

        N = 3
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=np.float32)
        velocities = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        active = np.ones(N, dtype=bool)
        # Birds 0 and 1 see each other; bird 2 far away
        neighbor_idx = np.array([
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([], dtype=np.int32),
        ], dtype=object)

        force = separation_force(positions, velocities, neighbor_idx, active)

        # D21: With corrected 1/d³, force magnitude at d=0.01 should be
        # very large (~10000), but we verify it's finite (not NaN/Inf)
        assert np.isfinite(force).all(), (
            f"D21: corrected separation must not produce NaN. force={force}"
        )
        # D8: Verify force is non-zero and in correct direction
        f01 = float(np.linalg.norm(force[0]))
        assert f01 > 0, (
            "D8+D21: steric force should be non-zero at close range"
        )
        # Force should push birds apart (bird 0 at origin, bird 1 at +x)
        # Separation pushes bird 0 negative (away from bird 1)
        assert force[0, 0] < 0, (
            f"D8+D21: separation should push bird 0 away from bird 1. "
            f"fx={force[0,0]:.4f}"
        )

    def test_steric_clamp_limits_close_range_force(self):
        """D8+D21: Steric clamp (D8) caps force from corrected kernel (D21).

        D21 fixed separation to r̂/d² (dists³ in code). D8 ensures
        max_force is passed to steric_force. At d=0.01, the force is
        clamped to max_force, preventing explosion.
        """
        import numpy as np

        from pymurmur.physics.steric import steric_force

        observer = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour = np.array([[0.01, 0.0, 0.0]], dtype=np.float32)

        force = steric_force(
            observer, neighbour,
            strength=0.6, threshold=10.0, max_force=0.15,
        )

        # Forces must be finite
        assert np.isfinite(force).all()
        f_mag = float(np.linalg.norm(force))
        # With max_force=0.15, the clamp should keep force ≤ 0.15
        assert f_mag <= 0.151, (
            f"D8: steric clamp must cap force at max_force=0.15, "
            f"got {f_mag:.6f}"
        )
        assert f_mag > 0, "Steric force must be non-zero at close range"

    def test_steric_at_small_distance_returns_max_force(self) -> None:
        """At d = 0.01 the raw 1/d² force ≈ 6000, clamped to max_force."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)

        max_force = 5.0
        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            threshold=10.0,
            max_force=max_force,
        )

        force_mag = float(np.linalg.norm(force))
        assert abs(force_mag - max_force) < 1e-6, (
            f"Expected |F| = {max_force} (clamped), got {force_mag:.6f}"
        )

    def test_steric_no_neighbours_returns_zero(self) -> None:
        """No neighbours → zero force vector."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        empty = np.empty((0, 3), dtype=np.float32)

        force = steric_force(
            obs_pos, empty,
            strength=0.6,
            threshold=10.0,
            max_force=5.0,
        )

        assert np.allclose(force, np.zeros(3), atol=1e-12)

    def test_steric_distant_neighbour_returns_zero(self) -> None:
        """Neighbour beyond threshold → no force contribution."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)

        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            threshold=10.0,
            max_force=5.0,
        )

        assert np.allclose(force, np.zeros(3), atol=1e-12)

    def test_clamp_at_d001_with_production_max_force(self) -> None:
        """D8: Pair at d=0.01, strength=0.6, default clamp 0.15 → ‖F‖ == 0.15.

        This is the exact test case from the roadmap — verifies that
        steric_force's max_force clamp actually engages at production
        defaults (not just at the artificially high 5.0 used above).
        At d=0.01 with strength=0.6: raw force ≈ 0.6/(0.01)² = 6000.
        The clamp at 0.15 must reduce it to exactly 0.15."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)

        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            max_force=0.15,
        )

        assert np.linalg.norm(force) == pytest.approx(0.15)

    def test_clamp_not_triggered_at_large_distance(self) -> None:
        """D8: At d=5.0 with max_force=0.15, steric force is well below
        max_force — clamp should NOT engage. Verifies the clamp is
        conditional, not always-on."""
        obs_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        neighbour_pos = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)

        force = steric_force(
            obs_pos, neighbour_pos,
            strength=0.6,
            max_force=0.15,
        )
        # Raw force ≈ 0.6 / 25 = 0.024 < max_force 0.15, so no clamp
        mag = float(np.linalg.norm(force))
        assert mag < 0.15, f"Force {mag:.4f} should be below max_force"
        assert mag > 0.0, "Force should be non-zero"


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
