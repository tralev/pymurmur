"""Targeted tests for uncovered branches in P3 phase modules.

Closes coverage gaps identified in the coverage report:
- predator.py L61:      _rotate_toward anti-parallel fallback
- predator.py L169:     drift normalisation zero-drift case
- wander.py L121:       flock.center is None → positions.mean fallback
- wander.py L156:       _apply_pull with all-zero distances
- field.py L255:        inner cavity (when some birds are inside)
- field.py L398:        buoyancy early return (n_active==0 guarding path)
- field.py L572:        grid separation normalisation
- field.py L677–678:    force clamp when accelerations exceed max_force
"""

import numpy as np
import pytest

from pymurmur.physics.extensions.predator import _rotate_toward
from pymurmur.physics.extensions.wander import Wander
from pymurmur.physics.forces.field import (
    _compute_shell_force,
    _compute_buoyancy,
    _compute_floating_boundary,
    _compute_grid_sep_normalized,
    FieldMode,
)
from pymurmur.physics.extensions._base import StepContext
from pymurmur.core.config import SimConfig


# ══════════════════════════════════════════════════════════════════════
# predator.py — _rotate_toward anti-parallel fallback (L61)
# ══════════════════════════════════════════════════════════════════════

class TestPredatorRotateTowardAntiparallel:
    """Test that _rotate_toward handles anti-parallel inputs correctly."""

    def test_antiparallel_x_axis(self):
        """current = (1,0,0), target = (−1,0,0) → picks perpendicular axis."""
        current = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        result = _rotate_toward(current, target, 0.1)
        # Must be unit vector
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)
        # Must have rotated by max_angle (0.1 rad) from current
        dot_cur = np.dot(result / np.linalg.norm(result), current / np.linalg.norm(current))
        assert np.isclose(dot_cur, np.cos(0.1), atol=1e-4), f"expected cos(0.1)≈{np.cos(0.1):.4f}, got dot={dot_cur:.4f}"

    def test_antiparallel_y_axis(self):
        """current = (0,1,0), target = (0,−1,0) — different primary axis."""
        current = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        target = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        result = _rotate_toward(current, target, 0.05)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_antiparallel_non_cardinal(self):
        """current = (0.99, 0.01, 0), target = (−0.99, −0.01, 0) — anti-parallel off axes."""
        current = np.array([0.99, 0.01, 0.0], dtype=np.float32)
        target = np.array([-0.99, -0.01, 0.0], dtype=np.float32)
        result = _rotate_toward(current, target, 0.1)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_parallel_already_aligned(self):
        """current = target → no rotation needed, returns target."""
        current = np.array([0.707, 0.707, 0.0], dtype=np.float32)
        current = current / np.linalg.norm(current)
        result = _rotate_toward(current, current, 0.1)
        np.testing.assert_allclose(result, current, atol=1e-6)

    def test_angle_less_than_max_returns_target(self):
        """Dot product > cos(max_angle) → returns target directly."""
        current = np.array([0.98, 0.0, 0.199], dtype=np.float32)  # ~0.2 rad away from x-axis
        current = current / np.linalg.norm(current)
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _rotate_toward(current, target, 0.3)  # max_angle=0.3 > angle
        np.testing.assert_allclose(result, target, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
# predator.py — drift normalisation zero-drift (L169)
# ══════════════════════════════════════════════════════════════════════

class TestPredatorEgressDriftEdge:
    """Verify the egress drift normalisation branch in Predator.apply()
    when the drift computed from cross(_turn_axis, _dir) has near-zero norm."""

    def test_predator_with_aligned_turn_axis(self):
        """Run a full predator step with _turn_axis aligned to _dir so that
        cross(_turn_axis, _dir) ≈ 0, triggering drift = zeros(3) branch (L169)."""
        from pymurmur.physics.extensions.predator import Predator
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.mode = "field"
        cfg.v0 = 4.0
        cfg.predator_enabled = True

        flock = PhysicsFlock(cfg)
        ctx = StepContext(frame=0, dt=1 / 60, rng=np.random.default_rng(42),
                          center=np.array([500, 350, 200], dtype=np.float32), config=cfg)

        p = Predator(cfg)
        # Force predator to egress phase and align _turn_axis with _dir
        p._phase = "egress"
        p._pos = np.array([800.0, 350.0, 200.0], dtype=np.float32)
        p._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p._turn_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # aligned → cross=0

        # Should not crash; drift normalisation handles zero-norm gracefully
        p.apply(flock, ctx)
        # Predator should have moved or stayed — either is fine, just no crash
        assert p._pos is not None


# ══════════════════════════════════════════════════════════════════════
# wander.py — flock.center fallback (L121)
# ══════════════════════════════════════════════════════════════════════

class TestWanderCenterFallback:
    """Test wander when flock.center is None (EMA not yet computed)."""

    def test_wander_uses_positions_mean(self):
        """When flock.center is None, wander falls back to np.mean(positions[active])."""
        # Create a minimal mock flock
        class MockFlock:
            def __init__(self):
                self.positions = np.array(
                    [[100, 200, 300], [400, 500, 600], [700, 800, 900]],
                    dtype=np.float32,
                )
                self.velocities = np.zeros((3, 3), dtype=np.float32)
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.ones(3, dtype=bool)
                self.N_capacity = 3
                self.center = None  # <-- not yet set
                self.wander_center = None
                self.wander_heading = None

        flock = MockFlock()
        cfg = SimConfig()
        ctx = StepContext(frame=0, dt=1 / 60, rng=np.random.default_rng(42),
                          center=np.array([500, 350, 200], dtype=np.float32), config=cfg)

        wander = Wander()
        wander.apply(flock, ctx)

        # Wander centre should have been computed (via positions.mean fallback)
        assert flock.wander_center is not None
        assert flock.wander_heading is not None
        assert flock.wander_center.shape == (3,)
        assert flock.wander_heading.shape == (3,)

    def test_wander_no_active_fallback(self):
        """When no birds are active and center is None, uses domain centre."""
        class MockFlock:
            def __init__(self):
                self.positions = np.zeros((3, 3), dtype=np.float32)
                self.velocities = np.zeros((3, 3), dtype=np.float32)
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.zeros(3, dtype=bool)  # all inactive
                self.N_capacity = 3
                self.center = None
                self.wander_center = None
                self.wander_heading = None

        flock = MockFlock()
        cfg = SimConfig()
        ctx = StepContext(frame=0, dt=1 / 60, rng=np.random.default_rng(42),
                          center=None, config=cfg)

        wander = Wander()
        wander.apply(flock, ctx)

        # Should not crash; wander_center computed from domain centre fallback
        assert flock.wander_center is not None


# ══════════════════════════════════════════════════════════════════════
# wander.py — _apply_pull zero-distance early return (L156)
# ══════════════════════════════════════════════════════════════════════

class TestWanderApplyPullEdge:
    """Test _apply_pull when all birds are exactly at the wander centre."""

    def test_pull_all_at_center_returns_early(self):
        """When all birds are at the target, dists=0 → mask.all()=False → return."""
        class MockFlock:
            def __init__(self):
                self.positions = np.array(
                    [[500, 350, 200], [500, 350, 200], [500, 350, 200]],
                    dtype=np.float32,
                )
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.ones(3, dtype=bool)
                self.N_capacity = 3

        flock = MockFlock()
        target = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        acc_before = flock.accelerations.copy()

        Wander._apply_pull(flock, target, flock.active)

        # Accelerations should be unchanged (early return triggered)
        np.testing.assert_array_equal(flock.accelerations, acc_before)

    def test_pull_partial_masks_correctly(self):
        """Birds at different distances — only distant ones get pull."""
        class MockFlock:
            def __init__(self):
                self.positions = np.array(
                    [[500, 350, 200],    # at target
                     [550, 350, 200],    # 50 units away
                     [500, 350, 200]],   # at target
                    dtype=np.float32,
                )
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.ones(3, dtype=bool)
                self.N_capacity = 3

        flock = MockFlock()
        target = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        Wander._apply_pull(flock, target, flock.active)

        # Bird 1 should have acceleration applied (pulled toward target)
        assert np.abs(flock.accelerations[1]).sum() > 0
        # Birds 0 and 2 should be unchanged (at target)
        assert np.abs(flock.accelerations[0]).sum() == 0
        assert np.abs(flock.accelerations[2]).sum() == 0


# ══════════════════════════════════════════════════════════════════════
# field.py — inner cavity force (L255)
# ══════════════════════════════════════════════════════════════════════

class TestFieldShellInnerCavity:
    """Verify the inner cavity push-out activates when birds are inside R_blob."""

    def test_bird_inside_inner_gets_expelled(self):
        """Bird very close to target (d ≈ 0.5, well inside inner cavity)
        gets push-out force from the inner cavity expansion."""
        pos = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        targets = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        seeds = np.array([0.0], dtype=np.float32)
        U = 100.0

        F = _compute_shell_force(pos, targets, seeds, 0.0, U,
                                  cohesion=1.0, chase_strength=0.0, sep=1.0,
                                  shell_influence=1.0)

        # d=0.5 is inside inner cavity (~15.1) → should get outward push
        assert np.linalg.norm(F) > 0, "bird inside inner cavity should get push-out"

    def test_shell_force_zero_birds(self):
        """n=0 → returns (0,3) zeros (L255 early return)."""
        F = _compute_shell_force(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            0.0, 100.0, 1.0, 0.0, 1.0, 1.0,
        )
        assert F.shape == (0, 3)
        assert F.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════
# field.py — floating boundary R_boundary ≤ 0 (L572)
# ══════════════════════════════════════════════════════════════════════

class TestFieldFloatingBoundaryEdge:
    """Test floating boundary edge case where R_blobs are all zero."""

    def test_boundary_zero_blobs(self):
        """R_blobs=0 → R_boundary=0 → early return (L572)."""
        pos = np.array([[500.0, 350.0, 200.0]], dtype=np.float32)
        C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        R_blobs = np.zeros(1, dtype=np.float32)
        F = _compute_floating_boundary(pos, C, R_blobs, U=100.0)
        assert F.shape == (1, 3)
        assert np.all(F == 0.0), "zero R_boundary should produce zero force"

    def test_boundary_bird_outside_gets_contained(self):
        """Bird far outside R_boundary gets containment force."""
        pos = np.array([[1000.0, 0.0, 0.0]], dtype=np.float32)
        C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        R_blobs = np.array([32.0], dtype=np.float32)  # R_boundary ≈ 46.4
        F = _compute_floating_boundary(pos, C, R_blobs, U=100.0)
        # Bird at 1000 > 46.4 → should get containment force
        assert np.linalg.norm(F) > 0, "bird outside boundary should get force"


# ══════════════════════════════════════════════════════════════════════
# field.py — force clamp with threat_present but empty arrays (L677-678)
# ══════════════════════════════════════════════════════════════════════

class TestFieldForceClampThreatEdge:
    """Test FieldMode.compute when _threat_present=True but blackening/active
    arrays are None or empty — exercises the else branch at L677-678."""

    def test_compute_with_threat_present_empty_arrays(self):
        """_threat_present=True but _threat_blackening=None → falls to else branch."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 3
        cfg._field_time = 0.0
        cfg._threat_present = True
        cfg._threat_blackening = None
        cfg._threat_active = None

        positions = np.array([[500, 350, 200], [510, 360, 210], [490, 340, 190]],
                             dtype=np.float32)
        velocities = np.zeros((3, 3), dtype=np.float32)
        accelerations = np.zeros((3, 3), dtype=np.float32)
        active = np.ones(3, dtype=bool)

        FieldMode.compute(positions, velocities, accelerations, active,
                          index=None, rng=np.random.default_rng(42),
                          last_theta=np.zeros(3, dtype=np.float32),
                          config=cfg)

        # Should complete without crash — uses scalar coh/sep (else branch)
        assert np.isfinite(accelerations).all()


# ══════════════════════════════════════════════════════════════════════
# wander.py — wander_heading zero-diff fallback (L74)
# ══════════════════════════════════════════════════════════════════════

class TestWanderHeadingZeroDiff:
    """Test wander_heading when diff norm ≤ 1e-10 (L74 fallback)."""

    def test_heading_zero_diff_returns_default(self, monkeypatch):
        """When bounded_unit_path returns same value for t and t+0.75,
        wander_heading falls back to (1,0,0)."""
        from pymurmur.physics.extensions import wander as wander_mod

        same_path = np.array([0.5, 0.3, 0.1], dtype=np.float32)

        def mock_path(t):
            return same_path.copy()

        monkeypatch.setattr(wander_mod, "bounded_unit_path", mock_path)
        # Need to re-import wander_heading to pick up patched bounded_unit_path
        heading = wander_mod.wander_heading(0.0)
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(heading, expected, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
# flock.py — add_boids / remove_boids paths (L42, L79, L119, L160, L401)
# ══════════════════════════════════════════════════════════════════════

class TestFlockAddRemoveBoids:
    """Test PhysicsFlock.add_boids and remove_boids methods."""

    def test_add_boids_non_blob_mode(self):
        """add_boids with default position_init uses random_unit_sphere fallback."""
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.mode = "field"
        cfg.position_init = "box"

        flock = PhysicsFlock(cfg)
        initial_active = flock.active.sum()

        added = flock.add_boids(3, cfg)
        assert added == 3
        assert flock.active.sum() == initial_active + 3
        # New positions should be finite
        assert np.isfinite(flock.positions).all()
        assert np.isfinite(flock.velocities).all()

    def test_add_boids_blob_mode(self):
        """add_boids with position_init='blob' uses blob velocity init."""
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.mode = "field"
        cfg.position_init = "blob"

        flock = PhysicsFlock(cfg)
        added = flock.add_boids(3, cfg)
        assert added == 3
        assert np.isfinite(flock.positions).all()
        assert np.isfinite(flock.velocities).all()

    def test_remove_boids(self):
        """remove_boids deactivates birds and returns correct count."""
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "field"

        flock = PhysicsFlock(cfg)
        initial_active = flock.active.sum()

        removed = flock.remove_boids(3)
        assert removed == 3
        assert flock.active.sum() == initial_active - 3

    def test_add_boids_extends_capacity(self):
        """add_boids extends arrays when N_capacity is full."""
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.mode = "field"

        flock = PhysicsFlock(cfg)
        # All 5 slots are active (initial state)
        assert flock.active.sum() == 5
        old_capacity = flock.N_capacity

        added = flock.add_boids(3, cfg)
        assert added == 3  # extends and adds
        assert flock.N_capacity > old_capacity
        assert np.isfinite(flock.positions).all()
        assert np.isfinite(flock.velocities).all()


# ══════════════════════════════════════════════════════════════════════
# field.py — buoyancy n==0 guard (L398)
# ══════════════════════════════════════════════════════════════════════

class TestFieldBuoyancyZeroBirds:
    """Test that buoyancy returns zero array when n_active==0."""

    def test_buoyancy_zero_birds(self):
        """Call _compute_buoyancy with n=0 → returns (0,3) zeros."""
        pos = np.zeros((0, 3), dtype=np.float32)
        targets = np.zeros((0, 3), dtype=np.float32)
        seeds = np.zeros(0, dtype=np.float32)
        U = 100.0

        F = _compute_buoyancy(pos, targets, seeds, 0.0, U, flow=0.3)
        assert F.shape == (0, 3)
        assert F.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════
# field.py — grid separation normalisation (L572)
# ══════════════════════════════════════════════════════════════════════

class TestFieldGridSepNormalized:
    """Test grid separation normalisation per P3.11."""

    def test_single_neighbour(self):
        """1 neighbour → sep / 1 = sep."""
        result = _compute_grid_sep_normalized(
            np.zeros((10, 3), dtype=np.float32), 1.0, neighbour_count=1
        )
        assert result == 1.0

    def test_many_neighbours(self):
        """100 neighbours → sep / 100."""
        result = _compute_grid_sep_normalized(
            np.zeros((10, 3), dtype=np.float32), 2.5, neighbour_count=100
        )
        assert result == 0.025

    def test_zero_neighbours_uses_one(self):
        """0 neighbours → denominator clamped to 1."""
        result = _compute_grid_sep_normalized(
            np.zeros((10, 3), dtype=np.float32), 3.0, neighbour_count=0
        )
        assert result == 3.0


# ══════════════════════════════════════════════════════════════════════
# field.py — force clamp (L677–678)
# ══════════════════════════════════════════════════════════════════════

class TestFieldForceClamp:
    """Verify that excessive forces are clamped to max_force."""

    def test_force_clamped_when_too_strong(self):
        """Apply a huge force and verify it's clamped at max_force."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 3
        cfg.max_force = 0.15  # low clamp

        positions = np.array([[500, 350, 200], [100, 100, 100], [500, 350, 200]],
                             dtype=np.float32)
        velocities = np.zeros((3, 3), dtype=np.float32)
        # Pre-set huge accelerations to trigger clamp
        accelerations = np.array([[10, 0, 0], [0, 20, 0], [0, 0, 30]],
                                  dtype=np.float32)
        active = np.ones(3, dtype=bool)

        # Must set _field_time for anchors
        cfg._field_time = 0.0

        FieldMode.compute(positions, velocities, accelerations, active,
                          index=None, rng=np.random.default_rng(42),
                          last_theta=np.zeros(3, dtype=np.float32),
                          config=cfg)

        # After compute, all active accelerations must have magnitude ≤ max_force
        acc_mags = np.linalg.norm(accelerations, axis=1)
        assert (acc_mags[active] <= cfg.max_force + 1e-4).all(), (
            f"all active forces must be ≤ max_force={cfg.max_force}, got {acc_mags}"
        )
