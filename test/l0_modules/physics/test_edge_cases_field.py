"""Targeted tests for uncovered branches in P3 phase modules —
field.py edge cases.

Closes coverage gaps identified in the coverage report:
- field.py L255:     inner cavity (when some birds are inside)
- field.py L398:     buoyancy early return (n_active==0 guarding path)
- field.py L572:     grid separation normalisation
- field.py L677-678: force clamp when accelerations exceed max_force

Split out of test_edge_cases.py (file-size split) — predator.py and
flock.py tests stay in the original; wander.py tests moved to
test_edge_cases_wander.py.
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.field import (
    FieldMode,
    _compute_buoyancy,
    _compute_floating_boundary,
    _compute_grid_sep_normalized,
    _compute_shell_force,
)

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
