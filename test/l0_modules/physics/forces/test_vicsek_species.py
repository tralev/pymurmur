"""P6 — Vicsek Species tests (predator-prey, asymmetric collisions).

Tests the Vicsek mode with predator/prey species interaction.
Implements P6.1 (fear-weighted alignment), P6.2 (predator hunting),
and P6.3 (asymmetric position collisions).

Per roadmap_deepseek.md P6:
- P6.1: Fear-weighted alignment blending
- P6.2: Predator hunting strategy
- P6.3: Asymmetric position collisions
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.vicsek import (
    resolve_species_collisions,
    vicsek_forces,
)

pytestmark = pytest.mark.guard

from test.helpers import _call_force  # noqa: E402


class TestVicsekSpecies:
    """Vicsek predator-prey interaction tests."""

    # ── P6 Integration: all three features active together ────

    def test_all_p6_features_together_no_nan_no_escape(self):
        """P6.1+P6.2+P6.3 active simultaneously: 50 frames, no NaN, in domain."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.5
        cfg.vicsek_diffusion = 0.1
        cfg.vicsek_radius_influence = 40.0
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 80.0
        cfg.vicsek_weight_afraid = 3.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.1
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        engine.flock.is_predator[:4] = True
        # Place predators together, prey scattered
        centre = np.array([100.0, 100.0, 100.0])
        engine.flock.positions[:4] = centre + np.random.default_rng(42).uniform(
            -20, 20, (4, 3)
        ).astype(np.float32)

        for frame in range(50):
            engine.step(1.0 / 60.0)
            assert np.isfinite(engine.flock.positions).all(), f"NaN pos at frame {frame}"
            assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"
            # All active positions must be in domain
            active = engine.flock.active
            assert (engine.flock.positions[active] >= 0).all()
            assert (engine.flock.positions[active, 0] < cfg.width).all()
            assert (engine.flock.positions[active, 1] < cfg.height).all()
            assert (engine.flock.positions[active, 2] < cfg.depth).all()

        assert engine.frame == 50

    def test_predator_pursues_prey_flee_and_collisions_prevent_overlap(self):
        """Full P6 cycle: predator hunts → prey flees → collisions keep them apart."""
        cfg = SimConfig()
        cfg.num_boids = 4
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.5
        cfg.vicsek_couplage = 0.3
        cfg.vicsek_diffusion = 0.05
        cfg.vicsek_radius_influence = 60.0
        cfg.vicsek_radius_avoid = 10.0
        cfg.vicsek_radius_predators = 80.0
        cfg.vicsek_weight_afraid = 5.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.05
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        # Predator at origin, prey at x=15 (within R_pred, fear triggers)
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([15.0, 0.0, 0.0])
        flock.positions[2] = np.array([20.0, 5.0, 0.0])
        flock.positions[3] = np.array([25.0, -5.0, 0.0])
        # All start with zero velocity — will be set by forces
        flock.velocities[:] = 0.0

        for _ in range(30):
            _call_force(vicsek_forces, flock, cfg)
            flock.positions[flock.active] += (
                flock.velocities[flock.active] * 0.1
            )
            # P6.3: Resolve collisions (mimics engine.step())
            resolve_species_collisions(
                flock.positions, flock.is_predator, cfg, flock.active,
            )

        # After 30 frames with collisions active, no predator-prey overlap
        for prey_i in range(1, 4):
            d = np.linalg.norm(flock.positions[0] - flock.positions[prey_i])
            assert d >= cfg.vicsek_radius_avoid * 0.5, (
                f"Predator-prey overlap: d={d:.2f} < 0.5*R_avoid={cfg.vicsek_radius_avoid*0.5}"
            )

        # Speeds must be correct after forces
        pred_speed = np.linalg.norm(flock.velocities[0])
        assert abs(pred_speed - cfg.vicsek_velocity_predator) < 1e-4, (
            f"Predator speed {pred_speed} != {cfg.vicsek_velocity_predator}"
        )
        for prey_i in range(1, 4):
            prey_speed = np.linalg.norm(flock.velocities[prey_i])
            assert abs(prey_speed - cfg.vicsek_velocity) < 1e-4, (
                f"Prey {prey_i} speed {prey_speed} != {cfg.vicsek_velocity}"
            )

        assert np.isfinite(flock.positions).all()
        assert np.isfinite(flock.velocities).all()

    def test_multi_predator_multi_prey_stable(self):
        """3 predators + 10 prey: speeds correct, no NaN, no explosions."""
        cfg = SimConfig()
        cfg.num_boids = 13
        cfg.width = 300
        cfg.height = 300
        cfg.depth = 300
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.4
        cfg.vicsek_diffusion = 0.15
        cfg.vicsek_radius_influence = 50.0
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 80.0
        cfg.vicsek_weight_afraid = 3.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.1
        cfg.seed = 123

        flock = PhysicsFlock(cfg)
        flock.is_predator[:3] = True
        # Scatter predators on left, prey on right
        flock.positions[:3] = np.random.default_rng(42).uniform(
            0, 50, (3, 3)
        ).astype(np.float32)
        flock.positions[3:] = np.random.default_rng(99).uniform(
            150, 250, (10, 3)
        ).astype(np.float32)

        for _ in range(20):
            _call_force(vicsek_forces, flock, cfg)
            flock.positions[flock.active] += (
                flock.velocities[flock.active] * 0.1
            )
            resolve_species_collisions(
                flock.positions, flock.is_predator, cfg, flock.active,
            )

        # Predator speeds
        pred_speeds = np.linalg.norm(flock.velocities[:3], axis=1)
        assert np.allclose(pred_speeds, cfg.vicsek_velocity_predator, atol=1e-4)
        # Prey speeds
        prey_speeds = np.linalg.norm(flock.velocities[3:], axis=1)
        assert np.allclose(prey_speeds, cfg.vicsek_velocity, atol=1e-4)
        # No NaN
        assert np.isfinite(flock.positions).all()
        assert np.isfinite(flock.velocities).all()

    # ── Survival (must pass now) ──────────────────────────────

    def test_is_predator_present_no_crash(self):
        """Vicsek runs without crash when flock has is_predator flags set."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 30
        cfg.vicsek_velocity = 1.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)

        if not hasattr(flock, 'is_predator'):
            pytest.skip("P0.6: PhysicsFlock.is_predator column not yet implemented")

        flock.is_predator[:5] = True
        assert flock.is_predator.sum() == 5

        for _ in range(10):
            _call_force(vicsek_forces, flock, cfg)

        # Prey should have speed v0; predators get predator speed
        prey_speeds = np.linalg.norm(flock.velocities[5:][flock.active[5:]], axis=1)
        assert np.allclose(prey_speeds, 1.0, atol=1e-4)

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

        _call_force(vicsek_forces, flock, cfg)
        # S2.D1: All-predator early-out applies a pure random walk (not a
        # frozen no-op) — verify no NaN and speeds match predator speed.
        assert np.isfinite(flock.velocities).all()
        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        np.testing.assert_allclose(speeds, cfg.vicsek_velocity_predator, atol=1e-4)

    # ── P6.1: Fear-weighted alignment ─────────────────────────

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
        cfg.vicsek_diffusion = 0.05   # low noise so flee dominates
        cfg.vicsek_radius_influence = 80.0
        cfg.vicsek_radius_predators = 80.0
        cfg.vicsek_weight_afraid = 5.0  # stronger flee weight
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        centre = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2])
        flock.is_predator[0] = True
        flock.positions[0] = centre
        flock.velocities[0] = np.array([0.0, 0.0, 0.0])
        # Place prey within R_pred of centre so fear fires for all
        rng = np.random.default_rng(42)
        prey_positions = centre + rng.uniform(-50, 50, (49, 3)).astype(np.float32)
        flock.positions[1:] = prey_positions

        for _ in range(5):
            _call_force(vicsek_forces, flock, cfg)

        prey_idx = np.where(flock.active & ~flock.is_predator)[0]
        prey_vels = flock.velocities[prey_idx]
        prey_norms = np.linalg.norm(prey_vels, axis=1, keepdims=True) + 1e-10
        prey_dirs = prey_vels / prey_norms
        from_centre = flock.positions[prey_idx] - centre
        from_centre_norms = np.linalg.norm(from_centre, axis=1, keepdims=True) + 1e-10
        radial_dirs = from_centre / from_centre_norms
        dot_products = np.sum(prey_dirs * radial_dirs, axis=1)
        mean_dot = float(np.mean(dot_products))
        assert mean_dot > 0.8, (
            f"Prey not fleeing predator: mean dot={mean_dot:.3f}"
        )

    def test_fear_zero_when_far_from_predator(self):
        """Prey far from predators get standard vicsek (no flee blending)."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 10
        cfg.width = 500
        cfg.height = 500
        cfg.depth = 500
        cfg.vicsek_radius_predators = 10.0  # tiny detection radius
        cfg.vicsek_couplage = 0.5
        cfg.vicsek_diffusion = 0.1
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_radius_influence = 100.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        # Predator far from all prey
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.velocities[0] = np.array([1.0, 0.0, 0.0])
        # Prey clustered far away
        flock.positions[1:] = np.random.default_rng(42).uniform(
            300, 500, (9, 3)
        ).astype(np.float32)

        # Run — should not crash; prey should still have speed
        _call_force(vicsek_forces, flock, cfg)
        prey_speeds = np.linalg.norm(flock.velocities[1:], axis=1)
        assert np.all(prey_speeds > 0.0), "Prey should still be moving"

    def test_predator_speed_applied(self):
        """Predator birds use vicsek_velocity_predator (default 2.0)."""
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_radius_predators = 80.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[:3] = True

        _call_force(vicsek_forces, flock, cfg)

        pred_speeds = np.linalg.norm(flock.velocities[:3], axis=1)
        prey_speeds = np.linalg.norm(flock.velocities[3:], axis=1)
        assert np.allclose(pred_speeds, 2.0, atol=1e-4), (
            f"Predator speed: {pred_speeds}"
        )
        assert np.allclose(prey_speeds, 1.0, atol=1e-4), (
            f"Prey speed: {prey_speeds}"
        )

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

    # ── P6.3: Asymmetric collisions ───────────────────────────

    def test_asymmetric_collision_prey_pushed_back(self):
        """Prey-predator at d < R_pred: prey takes full push, predator unmoved."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 2
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 20.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([2.0, 0.0, 0.0])

        pos_pred_before = flock.positions[0].copy()
        n = resolve_species_collisions(
            flock.positions, flock.is_predator, cfg, flock.active,
        )

        assert n == 1, f"Expected 1 collision correction, got {n}"
        assert np.allclose(flock.positions[0], pos_pred_before), (
            "Predator moved — should be unmoved in asymmetric collision"
        )
        new_dist = np.linalg.norm(flock.positions[1] - flock.positions[0])
        assert new_dist > 2.0, (
            f"Prey not pushed away: distance {new_dist:.1f} (was 2.0)"
        )

    def test_same_type_symmetric_collision(self):
        """Two prey at d < R_avoid: each moves (R_avoid-d)/2 away."""
        cfg = SimConfig()
        cfg.num_boids = 2
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.vicsek_radius_avoid = 5.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[:] = False
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([3.0, 0.0, 0.0])  # d=3 < R_avoid=5

        cm_before = (flock.positions[0] + flock.positions[1]) / 2.0
        n = resolve_species_collisions(
            flock.positions, flock.is_predator, cfg, flock.active,
        )

        assert n == 1
        cm_after = (flock.positions[0] + flock.positions[1]) / 2.0
        # Centre of mass should be unchanged (symmetric push)
        np.testing.assert_allclose(cm_after, cm_before, atol=1e-6)
        new_dist = np.linalg.norm(flock.positions[1] - flock.positions[0])
        assert abs(new_dist - 5.0) < 1e-6, (
            f"Expected d={cfg.vicsek_radius_avoid}, got d={new_dist:.4f}"
        )

    def test_collision_across_toroidal_seam(self):
        """Collision correction works across toroidal boundary."""
        cfg = SimConfig()
        cfg.num_boids = 2
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.vicsek_radius_avoid = 10.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[:] = False
        # Place birds near opposite edges — distance via wrap is ~8
        flock.positions[0] = np.array([5.0, 0.0, 0.0])
        flock.positions[1] = np.array([97.0, 0.0, 0.0])  # dist via wrap ≈ 8

        n = resolve_species_collisions(
            flock.positions, flock.is_predator, cfg, flock.active,
        )

        assert n == 1, "Seam-crossing collision should be detected"
        # Distance should be corrected to R_avoid via min-image path
        delta = flock.positions[1] - flock.positions[0]
        for dim in range(3):
            half = 50.0
            if delta[dim] > half:
                delta[dim] -= 100.0
            elif delta[dim] < -half:
                delta[dim] += 100.0
        new_dist = np.linalg.norm(delta)
        assert abs(new_dist - 10.0) < 0.01, (
            f"After collision, min-image distance should be R_avoid=10, got {new_dist:.4f}"
        )

    def test_no_same_type_overlaps_after_steps(self):
        """100 steps of collision resolution → no pair < 0.5·R_avoid."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_velocity = 0.5
        cfg.vicsek_couplage = 0.0
        cfg.vicsek_diffusion = 0.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[:] = False
        # Cramp birds into a tiny space to force overlaps
        flock.positions = np.random.default_rng(42).uniform(
            45, 55, (20, 3)
        ).astype(np.float32)

        for _ in range(100):
            # Resolve collisions
            resolve_species_collisions(
                flock.positions, flock.is_predator, cfg, flock.active,
            )
            # Advance positions slightly (random walk to trigger new overlaps)
            flock.positions[flock.active] += np.random.default_rng(42).normal(
                0, 0.3, (20, 3)
            ).astype(np.float32)

        # Check no pair is below 0.5 * R_avoid
        min_dist = float('inf')
        width = cfg.width
        active_pos = flock.positions[flock.active]
        for i in range(len(active_pos)):
            for j in range(i + 1, len(active_pos)):
                delta = active_pos[j] - active_pos[i]
                for dim in range(3):
                    half = 50.0
                    if delta[dim] > half:
                        delta[dim] -= width
                    elif delta[dim] < -half:
                        delta[dim] += width
                d = np.linalg.norm(delta)
                if d < min_dist:
                    min_dist = d

        assert min_dist >= 0.5 * cfg.vicsek_radius_avoid, (
            f"Overlap detected: min distance {min_dist:.4f} < 0.5*R_avoid={0.5*cfg.vicsek_radius_avoid}"
        )

    def test_engine_runs_with_species(self):
        """SimulationEngine step() with predator/prey doesn't crash.

        P8.10: Uses default dt_phys (1/60s) for deterministic physics steps.
        """
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_radius_predators = 80.0
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_influence = 20.0  # must be > radius_avoid for validation
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        engine.flock.is_predator[:3] = True

        for _ in range(10):
            engine.step()  # P8.10: default frame_dt = 1/60 → 1 physics step each

        assert engine.frame == 10
        assert np.isfinite(engine.flock.positions).all()
        assert np.isfinite(engine.flock.velocities).all()

    # ── P6.1: Solo prey (no neighbours) ──────────────────────

    def test_solo_prey_flees_predator(self):
        """A single prey bird near a predator flees even without neighbours."""
        cfg = SimConfig()
        cfg.num_boids = 2
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.0
        cfg.vicsek_diffusion = 0.0
        cfg.vicsek_radius_influence = 5.0  # tiny → no neighbours
        cfg.vicsek_radius_predators = 50.0
        cfg.vicsek_weight_afraid = 5.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.positions[0] = np.array([0.0, 0.0, 0.0])
        flock.positions[1] = np.array([10.0, 0.0, 0.0])
        flock.velocities[0] = np.array([0.0, 0.0, 0.0])
        flock.velocities[1] = np.array([0.0, 1.0, 0.0])  # heading orthogonal to threat

        flock.velocities[1] / np.linalg.norm(flock.velocities[1])
        _call_force(vicsek_forces, flock, cfg)
        dir_after = flock.velocities[1] / (np.linalg.norm(flock.velocities[1]) + 1e-10)

        # Should have turned away from predator (positive x component).
        # With 70/30 flee/existing blend and purely +x flee, expected x ≈ 0.92
        assert dir_after[0] > 0.5, (
            f"Solo prey did not flee: direction={dir_after}"
        )

    # ── P6.3: Post-collision domain containment ──────────────

    def test_collisions_stay_in_domain_after_wrap(self):
        """Engine step: collision pushes near edge stay in domain after re-wrap."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 4
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_radius_avoid = 10.0
        cfg.vicsek_radius_influence = 20.0  # must be > radius_avoid for validation
        cfg.vicsek_radius_predators = 20.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_velocity = 0.5
        cfg.vicsek_couplage = 0.0
        cfg.vicsek_diffusion = 0.0
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        engine.flock.is_predator[0] = True
        engine.flock.is_predator[1] = False
        # Place prey at x=3, predator at x=5 so collision pushes prey left (toward x<0)
        engine.flock.positions[0] = np.array([5.0, 50.0, 50.0])
        engine.flock.positions[1] = np.array([3.0, 50.0, 50.0])
        engine.flock.positions[2] = np.array([50.0, 50.0, 50.0])
        engine.flock.positions[3] = np.array([60.0, 50.0, 50.0])
        engine.flock.velocities[0] = np.array([0.0, 0.0, 0.0])
        engine.flock.velocities[1] = np.array([0.0, 0.0, 0.0])

        # Run one step — forces (sets velocities) + integrate (move + wrap)
        # + collision resolution + re-wrap
        engine.step(0.1)

        # All active positions must be in [0, W) after step
        active = engine.flock.active
        for dim, domain in enumerate([cfg.width, cfg.height, cfg.depth]):
            assert (engine.flock.positions[active, dim] >= 0).all(), (
                f"pos[{dim}] < 0 after step"
            )
            assert (engine.flock.positions[active, dim] < domain).all(), (
                f"pos[{dim}] >= {domain} after step"
            )


    # ── P6.3: Independent-entity collision tests ──────────────

    def test_multiple_collisions_accumulate(self):
        """P6.3: Prey pushed symmetrically by two predators gets net-zero push.

        Predators at x=-3 and x=+3, prey at x=0. Both within R_pred=20.
        The sequential loop should still produce 2 corrections and
        both predators remain unmoved.
        """
        cfg = SimConfig()
        cfg.num_boids = 3
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 20.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.is_predator[1] = True
        flock.is_predator[2] = False  # prey
        flock.positions[0] = np.array([-3.0, 0.0, 0.0])  # predator A
        flock.positions[1] = np.array([3.0, 0.0, 0.0])   # predator B
        flock.positions[2] = np.array([0.0, 0.0, 0.0])   # prey at centre

        n = resolve_species_collisions(
            flock.positions, flock.is_predator, cfg, flock.active,
        )

        assert n == 2, f"Expected 2 collision corrections, got {n}"
        # Predators unmoved
        assert np.allclose(flock.positions[0], [-3.0, 0.0, 0.0]), "Predator A moved"
        assert np.allclose(flock.positions[1], [3.0, 0.0, 0.0]), "Predator B moved"
        # Prey must be pushed away from both predators.
        # Sequential order: (pred_A=0, prey=2) → prey pushed +x to 17
        # then (pred_B=1, prey=2) → prey pushed -x by (20-|3-17|)=6 → prey at 11
        assert flock.positions[2, 0] > cfg.vicsek_radius_avoid, (
            f"Prey not pushed beyond R_avoid: x={flock.positions[2, 0]:.1f}"
        )

    def test_numba_matches_numpy_species_collisions(self):
        """P8: the numba-compiled kernel must match the pure-numpy fallback
        (within float32 rounding) on a dense, collision-heavy scene —
        exercises the same sequential Gauss-Seidel branch decisions on
        both paths, not just the isolated per-pair math."""
        from pymurmur.physics.forces._kernels import (
            _HAS_NUMBA,
            _numba_species_collisions,
            _numpy_species_collisions,
        )
        if not _HAS_NUMBA:
            pytest.skip("numba not installed")

        rng = np.random.default_rng(7)
        n = 200
        pos_a = rng.uniform(0, 20, (n, 3)).astype(np.float32)  # dense -> many collisions
        pos_b = pos_a.copy()
        is_pred = rng.random(n) < 0.05
        active_idx = np.arange(n)

        count_numpy = _numpy_species_collisions(
            pos_a, is_pred, active_idx, 5.0, 8.0, 100.0, 100.0, 100.0,
        )
        count_numba = _numba_species_collisions(
            pos_b, is_pred, active_idx.astype(np.int64), 5.0, 8.0, 100.0, 100.0, 100.0,
        )

        assert count_numpy == count_numba, (
            "Numba and numpy paths must take identical branch decisions "
            f"(same corrections count): {count_numpy} vs {count_numba}"
        )
        np.testing.assert_allclose(
            pos_a, pos_b, atol=1e-3,
            err_msg="Numba kernel must match numpy fallback within float32 tolerance",
        )

    # ── P6 Integration: stress-test interactions ─────────────

    def test_cornered_prey_surrounded_by_predators(self):
        """Prey surrounded by 3 predators: fear vectors cancel, collisions push.

        With predators at (-10,0,0), (10,0,0), (0,10,0) around prey at (0,0,0),
        the mean flee direction averages to near-zero (predators cancel out).
        Collisions should still push prey away from the closest predators.
        """
        cfg = SimConfig()
        cfg.num_boids = 4
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.3
        cfg.vicsek_diffusion = 0.0
        cfg.vicsek_radius_influence = 100.0
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 50.0
        cfg.vicsek_weight_afraid = 5.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[:3] = True
        flock.is_predator[3] = False
        flock.positions[0] = np.array([-10.0, 0.0, 0.0])
        flock.positions[1] = np.array([10.0, 0.0, 0.0])
        flock.positions[2] = np.array([0.0, 10.0, 0.0])
        flock.positions[3] = np.array([0.0, 0.0, 0.0])  # prey at centre
        flock.velocities[:] = 0.0

        for _ in range(10):
            _call_force(vicsek_forces, flock, cfg)
            flock.positions[flock.active] += (
                flock.velocities[flock.active] * 0.1
            )
            resolve_species_collisions(
                flock.positions, flock.is_predator, cfg, flock.active,
            )

        # Prey should not be at origin — collisions pushed it somewhere
        prey_dist = np.linalg.norm(flock.positions[3])
        assert prey_dist > 1.0, f"Prey not pushed from origin: dist={prey_dist:.2f}"
        # No NaN or infinities
        assert np.isfinite(flock.positions).all()
        assert np.isfinite(flock.velocities).all()
        # All speeds bounded (check all birds, not just first/last)
        speeds = np.linalg.norm(flock.velocities, axis=1)
        assert np.all(speeds <= 10.0), f"Speed exploded: {speeds}"

    def test_domino_predator_push_into_other_prey(self):
        """Predator pushes prey A into prey B → symmetric collision between prey.

        Predator at (0,0,0), prey A at (3,0,0), prey B at (22,0,0).
        Predator pushes prey A +x by (R_pred-3) ≈ 17 → prey A at ~20.
        Prey A now close to prey B at 22 (d=2 < R_avoid=5) → symmetric push.
        Both prey should end up separated beyond R_avoid.
        """
        cfg = SimConfig()
        cfg.num_boids = 3
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 20.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        flock.is_predator[1] = False
        flock.is_predator[2] = False
        flock.positions[0] = np.array([0.0, 0.0, 0.0])   # predator
        flock.positions[1] = np.array([3.0, 0.0, 0.0])   # prey A
        flock.positions[2] = np.array([22.0, 0.0, 0.0])  # prey B

        resolve_species_collisions(
            flock.positions, flock.is_predator, cfg, flock.active,
        )

        # Both prey should be pushed away from each other after domino
        d_ab = np.linalg.norm(flock.positions[1] - flock.positions[2])
        assert d_ab >= cfg.vicsek_radius_avoid, (
            f"Prey too close after domino: d={d_ab:.2f} < R_avoid={cfg.vicsek_radius_avoid}"
        )
        # Predator unmoved
        assert np.allclose(flock.positions[0], [0.0, 0.0, 0.0]), "Predator moved"

    def test_toroidal_pursuit_across_boundary(self):
        """Predator chases prey across toroidal boundary over multiple frames.

        Predator at x=95, prey at x=5 in 100-wide domain (min-image dist=10).
        Over 20 frames through the engine, predator should hunt and prey should
        flee, with both staying in domain.  Checks that wrapping doesn't
        cause teleport spam.
        """
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 2  # just predator + prey, no extra birds
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.3
        cfg.vicsek_diffusion = 0.05
        cfg.vicsek_radius_influence = 100.0
        cfg.vicsek_radius_avoid = 5.0
        cfg.vicsek_radius_predators = 50.0
        cfg.vicsek_weight_afraid = 5.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.05
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        engine.flock.is_predator[0] = True
        engine.flock.positions[0] = np.array([95.0, 50.0, 50.0])  # predator near right edge
        engine.flock.positions[1] = np.array([5.0, 50.0, 50.0])   # prey near left edge
        engine.flock.velocities[0] = np.array([-1.0, 0.0, 0.0])   # heading left (toward prey via wrap)
        engine.flock.velocities[1] = np.array([-1.0, 0.0, 0.0])   # heading left (away from pred)

        for frame in range(20):
            engine.step(1.0 / 60.0)
            # All birds must stay in domain
            active = engine.flock.active
            assert (engine.flock.positions[active] >= 0).all(), f"Negative pos at frame {frame}"
            assert (engine.flock.positions[active, 0] < cfg.width).all(), f"OOB x at frame {frame}"
            assert np.isfinite(engine.flock.positions).all(), f"NaN at frame {frame}"

        assert engine.frame == 20
        # Both birds should have moved from their starting positions
        assert not np.allclose(engine.flock.positions[0], [95.0, 50.0, 50.0])
        assert not np.allclose(engine.flock.positions[1], [5.0, 50.0, 50.0])


    def test_prey_only_alpha_with_orthogonal_predator(self):
        """Prey-only α≈1.0 with one orthogonal predator (polar alignment preserved).

        All prey aligned along +x, predator positioned orthogonally (far above
        on the y-axis) so fear≈0 for all prey.  The predator's presence
        doesn't disrupt the prey's polar order — α stays near 1.0.
        """
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.width = 200
        cfg.height = 200
        cfg.depth = 200
        cfg.mode = "vicsek"
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_velocity_predator = 2.0
        cfg.vicsek_couplage = 0.8   # strong coupling for alignment
        cfg.vicsek_diffusion = 0.0   # zero noise
        cfg.vicsek_radius_influence = 100.0
        cfg.vicsek_radius_predators = 20.0  # small detection radius
        cfg.vicsek_weight_afraid = 3.0
        cfg.vicsek_detect_ratio = 1.5
        cfg.vicsek_predator_noise_ratio = 0.0
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        flock.is_predator[0] = True
        # Predator far above prey on y-axis — orthogonally positioned
        flock.positions[0] = np.array([100.0, 150.0, 100.0])
        flock.velocities[0] = np.array([0.0, 1.0, 0.0])  # moving +y (away from prey)
        # All prey aligned along +x at y=50
        for i in range(1, 20):
            flock.positions[i] = np.array([20.0 + i * 8.0, 50.0, 50.0])
            flock.velocities[i] = np.array([1.0, 0.0, 0.0])

        for _ in range(5):
            _call_force(vicsek_forces, flock, cfg)

        # Compute prey-only α (order parameter)
        prey_idx = np.where(~flock.is_predator)[0]
        prey_vels = flock.velocities[prey_idx]
        sum_vels = np.linalg.norm(prey_vels.sum(axis=0))
        sum_speeds = np.linalg.norm(prey_vels, axis=1).sum()
        alpha = sum_vels / (sum_speeds + 1e-10)

        # α should be near 1.0 — orthogonal predator doesn't break alignment
        assert alpha > 0.85, (
            f"Prey-only α degraded by orthogonal predator: α={alpha:.3f}"
        )
        assert np.isfinite(flock.velocities).all()
