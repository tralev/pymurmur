"""Targeted tests for uncovered branches in P3 phase modules —
predator.py rotate/drift edges, flock.py add/remove-boids paths.

Closes coverage gaps identified in the coverage report:
- predator.py L61:      _rotate_toward anti-parallel fallback
- predator.py L169:     drift normalisation zero-drift case
- flock.py L42, L79, L119, L160, L401: add_boids / remove_boids paths

field.py and wander.py edge-case tests moved to
test_edge_cases_field.py / test_edge_cases_wander.py (file-size split
of this file).

Split out of test_edge_cases.py (file-size split).
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.predator import _rotate_toward

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
