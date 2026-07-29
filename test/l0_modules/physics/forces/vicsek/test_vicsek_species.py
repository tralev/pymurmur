"""P6 — Vicsek Species: all-features integration, survival smoke, fear-weighted alignment (P6.1).

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
from pymurmur.physics.forces.vicsek import vicsek_forces
from pymurmur.physics.forces.vicsek_predator import resolve_species_collisions

pytestmark = pytest.mark.guard

from test.helpers import _call_force  # noqa: E402


class TestVicsekSpeciesIntegration:
    """Vicsek predator-prey: all-features integration + fear-weighted alignment."""

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

