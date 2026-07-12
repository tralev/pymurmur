"""P6 — Vicsek Species tests (predator-prey, asymmetric collisions).

Tests the Vicsek mode with predator/prey species interaction.
P6 features are not yet implemented — tests requiring predator logic
are marked xfail. Current survival tests verify the code path doesn't
crash with is_predator flags present.

Per roadmap_deepseek.md P6:
- P6.1: Fear-weighted alignment blending
- P6.2: Predator hunting strategy
- P6.3: Asymmetric position collisions
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.vicsek import vicsek_forces
from pymurmur.physics.flock import PhysicsFlock

pytestmark = pytest.mark.guard


class TestVicsekSpecies:
    """Vicsek predator-prey interaction tests."""

    # ── Current behaviour (must pass now) ────────────────────────

    def test_is_predator_present_no_crash(self):
        """Vicsek runs without crash when flock has is_predator flags set.

        The is_predator column is a P0.6 roadmap item — if it doesn't exist
        yet, skip with a clear message pointing to P0.6.
        """
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 30
        cfg.vicsek_velocity = 1.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)

        if not hasattr(flock, 'is_predator'):
            pytest.skip("P0.6: PhysicsFlock.is_predator column not yet implemented")

        # Mark first 5 birds as predators
        flock.is_predator[:5] = True
        assert flock.is_predator.sum() == 5

        # Run multiple steps — should not crash
        for _ in range(10):
            vicsek_forces(flock, cfg)

        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(speeds, 1.0, atol=1e-4)

    def test_all_predators_no_crash(self):
        """Vicsek with all birds marked as predators doesn't crash."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 20
        cfg.vicsek_velocity = 1.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)

        if not hasattr(flock, 'is_predator'):
            pytest.skip("P0.6: PhysicsFlock.is_predator column not yet implemented")

        flock.is_predator[:] = True

        vicsek_forces(flock, cfg)
        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(speeds, 1.0, atol=1e-4)

    # ── P6.1: Fear-weighted alignment (xfail) ────────────────────

    @pytest.mark.xfail(
        reason="P6.1: Vicsek doesn't use is_predator — all birds treated equally. "
               "Needs fear = clamp((R_pred − d̄_pred)/R_pred, 0, 1) and "
               "û_combined = normalize((1−fear)·û_align + fear·û_flee)."
    )
    def test_fear_weighted_alignment(self):
        """Stationary predator at centre → prey ⟨û·r̂⟩ > 0.8 within 5 steps."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 50
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.3
        cfg.vicsek_diffusion = 0.1
        cfg.vicsek_radius_influence = 20.0
        cfg.vicsek_radius_predators = 80.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        # Place one predator at centre of domain
        centre = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2])
        flock.is_predator[0] = True
        flock.positions[0] = centre
        flock.velocities[0] = np.array([0.0, 0.0, 0.0])

        for _ in range(5):
            vicsek_forces(flock, cfg)

        # Prey birds (idx ≥ 1) should flee the centre
        prey_idx = np.where(flock.active & ~flock.is_predator)[0]
        prey_vels = flock.velocities[prey_idx]
        prey_norms = np.linalg.norm(prey_vels, axis=1, keepdims=True) + 1e-10
        prey_dirs = prey_vels / prey_norms
        # Directions away from centre
        from_centre = flock.positions[prey_idx] - centre
        from_centre_norms = np.linalg.norm(from_centre, axis=1, keepdims=True) + 1e-10
        radial_dirs = from_centre / from_centre_norms
        dot_products = np.sum(prey_dirs * radial_dirs, axis=1)
        mean_dot = float(np.mean(dot_products))
        assert mean_dot > 0.8, (
            f"Prey not fleeing predator: mean dot={mean_dot:.3f}"
        )

    # ── P6.2: Predator hunting (xfail) ───────────────────────────

    @pytest.mark.xfail(
        reason="P6.2: No predator agent — Vicsek treats all birds identically. "
               "Needs hunting strategy (nearest prey within detect_ratio·R_pred)."
    )
    def test_predator_hunts_nearest_prey(self):
        """Predator closes distance to nearest prey in ≥90% of steps."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 5
        cfg.vicsek_radius_predators = 200.0
        cfg.vicsek_velocity = 1.5
        cfg.vicsek_velocity_predator = 2.5
        cfg.vicsek_detect_ratio = 1.5
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        # Predator at (0, 0, 0), prey at (100, 0, 0)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.velocities[0] = np.array([1.0, 0.0, 0.0])
        # Prey scattered away from origin
        flock.positions[1:] = np.random.default_rng(42).uniform(
            50, 150, (4, 3)
        ).astype(np.float32)

        closes = 0
        prev_dist = np.linalg.norm(flock.positions[0] - flock.positions[1])
        for _ in range(30):
            vicsek_forces(flock, cfg)
            # Advance positions manually (Vicsek only sets velocity)
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

    # ── P6.3: Asymmetric collisions (xfail) ──────────────────────

    @pytest.mark.xfail(
        reason="P6.3: Collision resolution not implemented. "
               "Needs same-type symmetric + prey-predator asymmetric push."
    )
    def test_asymmetric_collision_prey_pushed_back(self):
        """Prey-predator at d < R_pred: prey takes full push, predator unmoved."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 2
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 20.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        # Place predator and prey very close
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([2.0, 0.0, 0.0])  # within R_pred

        pos_pred_before = flock.positions[0].copy()
        vicsek_forces(flock, cfg)

        # Predator should NOT move from collision push
        assert np.allclose(flock.positions[0], pos_pred_before), (
            "Predator moved — should be unmoved in asymmetric collision"
        )
        # Prey should be pushed away (distance increased)
        new_dist = np.linalg.norm(flock.positions[1] - flock.positions[0])
        assert new_dist > 2.0, (
            f"Prey not pushed away: distance {new_dist:.1f} (was 2.0)"
        )
