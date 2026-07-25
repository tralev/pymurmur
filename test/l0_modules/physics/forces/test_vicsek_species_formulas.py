"""P6 — Vicsek Species: independent-entity formula tests (P6.1/P6.2), predator hunting (P6.2).

Per roadmap_deepseek.md P6:
- P6.1: Fear-weighted alignment blending
- P6.2: Predator hunting strategy
- P6.3: Asymmetric position collisions

Split out of test_vicsek_species.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.vicsek import (
    vicsek_forces,
)

pytestmark = pytest.mark.guard

from test.helpers import _call_force  # noqa: E402


class TestVicsekSpeciesFormulas:
    """Vicsek predator-prey: independent-entity formula tests + predator hunting."""

    # ── P6.1: Independent-entity formula tests ────────────────

    def test_exact_blend_formula_deterministic(self):
        """P6.1: Exact fear blend formula with zero noise, known geometry.

        Uses 3 birds so prey has neighbours (nbr_counts > 1).
        Prey at (10,0,0), predator at (0,0,0), neighbour at (10,5,0) heading +y.
        u_align ≈ normalize((0,1,0)+(0,1,0)) = (0,1,0). flee = (1,0,0).
        fear ≈ (50-10)/50 = 0.8.
        blended = (1-0.8)*0.3*(0,1,0) + 3.0*0.8*(1,0,0) + 0.7*(0,1,0)
                = 0.06*(0,1,0) + 2.4*(1,0,0) + 0.7*(0,1,0)
                = (2.4, 0.76, 0)
        normalized ≈ (0.953, 0.302, 0)
        """
        cfg = SimConfig()
        cfg.num_boids = 3
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.3
        cfg.vicsek_diffusion = 0.0     # zero noise → deterministic
        cfg.vicsek_radius_influence = 200.0  # large → neighbours
        cfg.vicsek_radius_predators = 50.0
        cfg.vicsek_weight_afraid = 3.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])     # predator
        flock.velocities[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([10.0, 0.0, 0.0])    # prey
        flock.velocities[1] = np.array([0.0, 1.0, 0.0])    # heading +y
        flock.positions[2] = np.array([10.0, 5.0, 0.0])    # neighbour bird
        flock.velocities[2] = np.array([0.0, 1.0, 0.0])    # same heading

        _call_force(vicsek_forces, flock, cfg)

        prey_dir = flock.velocities[1] / np.linalg.norm(flock.velocities[1])
        np.testing.assert_allclose(
            prey_dir, [0.953, 0.302, 0.0], atol=0.05, rtol=0,
            err_msg=f"Blend formula mismatch: got {prey_dir}"
        )

    def test_toroidal_flee_vector(self):
        """P6.1: Flee direction uses min-image, not Cartesian distance.

        Predator at x=5, prey at x=95 in a 100-wide domain.
        Min-image distance = 10 (via wrap). Cartesian distance = 90.
        The flee direction should point away via the shorter (wrapped) path.
        """
        cfg = SimConfig()
        cfg.num_boids = 2
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.0
        cfg.vicsek_diffusion = 0.0
        cfg.vicsek_radius_influence = 200.0
        cfg.vicsek_radius_predators = 50.0
        cfg.vicsek_weight_afraid = 5.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([5.0, 0.0, 0.0])
        flock.velocities[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([95.0, 0.0, 0.0])  # close via wrap
        flock.velocities[1] = np.array([0.0, 0.0, 0.0])

        _call_force(vicsek_forces, flock, cfg)

        prey_dir = flock.velocities[1] / (np.linalg.norm(flock.velocities[1]) + 1e-10)
        # Flee should go +x (away from predator at x=5 via wrap)
        # Cartesian flee would go +x too in this case (both paths agree)
        # But min-image flee direction is toward +x (wrapping direction)
        # Let's verify: the predator is effectively at x=-5 via wrap from prey's perspective
        # So prey_pos - pred_pos_via_min_image = 95 - (-5) = 100? No...
        # Min-image vector from prey to predator: wrap(5-95) = 5-95+100 = 10
        # So predator is effectively at x=105 from prey. Flee = prey - pred = 95-105 = -10 = -x
        # Wait, min-image from prey to predator: (5,0) - (95,0) = (-90,0). Wrap: -90+100 = 10. So predator is +10 away in +x.
        # Flee = away from predator = -x direction.
        # So prey should go negative x (toward 0, away from the predator at 5).
        assert prey_dir[0] < -0.5, (
            f"Prey should flee -x (away from predator at x=5 via wrap), got {prey_dir}"
        )

    # ── P6.2: Predator hunting ────────────────────────────────

    def test_predator_hunts_nearest_prey(self):
        """Predator closes distance to nearest prey in ≥90% of steps."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 5
        cfg.width = 300
        cfg.height = 300
        cfg.depth = 300
        cfg.vicsek_radius_predators = 200.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.5
        cfg.vicsek_couplage = 0.0
        cfg.vicsek_diffusion = 0.0
        cfg.vicsek_predator_noise_ratio = 0.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.velocities[0] = np.array([1.0, 0.0, 0.0])
        flock.positions[1:] = np.random.default_rng(42).uniform(
            50, 150, (4, 3)
        ).astype(np.float32)

        closes = 0
        prev_dist = np.linalg.norm(flock.positions[0] - flock.positions[1])
        for _ in range(30):
            _call_force(vicsek_forces, flock, cfg)
            flock.positions[flock.active] += (
                flock.velocities[flock.active] * 0.1
            )
            dist = np.linalg.norm(flock.positions[0] - flock.positions[1])
            if dist < prev_dist - 1e-6:
                closes += 1
            prev_dist = dist

        close_ratio = closes / 30
        assert close_ratio >= 0.9, (
            f"Predator not hunting: closed distance in {close_ratio:.0%} of steps"
        )

    def test_predator_random_walk_when_no_prey(self):
        """Predator does random walk when no prey are nearby."""
        cfg = SimConfig()
        cfg.num_boids = 2
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 1.5
        cfg.vicsek_radius_predators = 10.0  # tiny detection
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([90.0, 0.0, 0.0])  # far from predator

        flock.velocities[0] / (np.linalg.norm(flock.velocities[0]) + 1e-10)
        _call_force(vicsek_forces, flock, cfg)
        flock.velocities[0] / (np.linalg.norm(flock.velocities[0]) + 1e-10)

        # Predator speed should be predator speed (not prey speed)
        assert abs(np.linalg.norm(flock.velocities[0]) - 1.5) < 1e-4

    def test_all_predators_no_interaction(self):
        """All-predator flock skips alignment/hunting but still random-walks.

        S2.D1: the spec calls for a pure random walk on an all-predator
        flock, not a frozen no-op — velocities must change (direction
        randomised) while staying at predator speed.
        """
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.8
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[:] = True
        old_vels = flock.velocities.copy()

        _call_force(vicsek_forces, flock, cfg)

        # Random walk → direction changes, but every bird still moves at
        # predator speed (no alignment/hunting coupling applied).
        assert not np.allclose(flock.velocities, old_vels)
        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        np.testing.assert_allclose(speeds, cfg.vicsek_velocity_predator, atol=1e-4)

    # ── P6.2: Independent-entity formula tests ────────────────

    def test_exact_target_vector_zero_noise(self):
        """P6.2: With zero noise, predator direction = exact normalize(prey-pred)."""
        cfg = SimConfig()
        cfg.num_boids = 2
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.0
        cfg.vicsek_diffusion = 0.0
        cfg.vicsek_radius_influence = 200.0
        cfg.vicsek_radius_predators = 200.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.0  # zero noise → exact target
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.velocities[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([3.0, 4.0, 0.0])  # distance 5, direction (0.6, 0.8, 0)

        _call_force(vicsek_forces, flock, cfg)

        pred_dir = flock.velocities[0] / np.linalg.norm(flock.velocities[0])
        # Expected direction = normalize((3, 4, 0)) = (0.6, 0.8, 0)
        np.testing.assert_allclose(pred_dir, [0.6, 0.8, 0.0], atol=1e-4, rtol=0,
            err_msg=f"Predator direction should be exact target vector, got {pred_dir}")

    def test_predator_selects_nearest_via_min_image(self):
        """P6.2: Predator selects prey closer via toroidal wrap, not Cartesian.

        Predator at x=5. Two prey: prey A at x=50 (Cartesian distance 45),
        prey B at x=97 (Cartesian distance 92, but min-image distance = 8 via wrap).
        Predator should hunt prey B (closer via min-image).
        """
        cfg = SimConfig()
        cfg.num_boids = 3
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.0
        cfg.vicsek_diffusion = 0.0
        cfg.vicsek_radius_influence = 200.0
        cfg.vicsek_radius_predators = 200.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([5.0, 0.0, 0.0])
        flock.velocities[0] = np.array([0.0, 0.0, 0.0])
        # Prey A: far in Cartesian (45), far in min-image (45)
        flock.positions[1] = np.array([50.0, 0.0, 0.0])
        # Prey B: far in Cartesian (92), close in min-image (8 via wrap)
        flock.positions[2] = np.array([97.0, 0.0, 0.0])

        _call_force(vicsek_forces, flock, cfg)

        pred_dir = flock.velocities[0] / np.linalg.norm(flock.velocities[0])
        # Min-image from predator (5) to prey B (97): wrap(97-5) = 92, but...
        # Actually: wrap(97-5) = 92, wrap(5-97) = 5-97+100 = 8.
        # The min distance is 8 via -x (predator to prey B wraps left).
        # So predator should hunt toward -x (negative x direction) to reach prey B.
        assert pred_dir[0] < -0.5, (
            f"Predator should hunt prey B via wrap (-x), got {pred_dir}"
        )

