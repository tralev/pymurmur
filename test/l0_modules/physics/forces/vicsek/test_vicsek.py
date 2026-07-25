"""Vicsek mode tests — Phase 8.1

Phase transition, constant speed, disorder, edge cases.
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.vicsek import vicsek_forces
from test.helpers import _call_force  # noqa: E402


class TestVicsekMode:
    """Vicsek 1995 constant-speed angle-coupling model."""

    def test_constant_speed(self):
        """All birds maintain exactly config.vicsek_velocity speed."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 100
        cfg.vicsek_velocity = 2.5
        cfg.vicsek_couplage = 0.8
        cfg.vicsek_diffusion = 0.1

        flock = PhysicsFlock(cfg)
        _call_force(vicsek_forces, flock, cfg)

        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(speeds, 2.5, atol=1e-4)

    def test_order_transition_high_couplage(self):
        """High couplage + low noise → high order parameter (> 0.7).

        Uses a small domain so birds have neighbours within the radius.
        """
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 200
        cfg.width = 100
        cfg.height = 100
        cfg.depth = 100
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.9
        cfg.vicsek_diffusion = 0.01
        cfg.vicsek_radius_influence = 100.0  # covers entire small domain

        flock = PhysicsFlock(cfg)
        for _ in range(20):
            _call_force(vicsek_forces, flock, cfg)

        active = flock.velocities[flock.active]
        norms = np.linalg.norm(active, axis=1)
        dirs = active / norms[:, np.newaxis]
        alpha = np.linalg.norm(np.mean(dirs, axis=0))
        assert alpha > 0.7, f"Expected alpha > 0.7, got {alpha:.3f}"

    def test_disorder_low_couplage(self):
        """Low couplage + high noise → low order parameter (< 0.3)."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 200
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.1
        cfg.vicsek_diffusion = 3.0
        cfg.vicsek_radius_influence = 10.0

        flock = PhysicsFlock(cfg)
        for _ in range(10):
            _call_force(vicsek_forces, flock, cfg)

        active = flock.velocities[flock.active]
        norms = np.linalg.norm(active, axis=1)
        dirs = active / norms[:, np.newaxis]
        alpha = np.linalg.norm(np.mean(dirs, axis=0))
        assert alpha < 0.3, f"Expected alpha < 0.3, got {alpha:.3f}"

    def test_zero_active(self):
        """Empty flock produces no change."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        old_vel = flock.velocities.copy()
        _call_force(vicsek_forces, flock, cfg)
        assert np.allclose(flock.velocities, old_vel)

    def test_single_bird(self):
        """Single bird gets pure noise direction (no neighbours)."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 1
        cfg.vicsek_velocity = 1.0

        flock = PhysicsFlock(cfg)
        _call_force(vicsek_forces, flock, cfg)

        speed = np.linalg.norm(flock.velocities[0])
        assert speed == pytest.approx(1.0, abs=1e-4)

    def test_phase_transition(self):
        """Phase transition: increasing couplage increases order monotonically."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 150
        cfg.width = 50
        cfg.height = 50
        cfg.depth = 50
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_diffusion = 0.5
        cfg.vicsek_radius_influence = 10.0  # local spatial coupling (not all-to-all)
        cfg.vicsek_time_step = 0.1         # explicit for clarity
        cfg.seed = 42

        alphas = []
        for eta in [0.1, 0.3, 0.5, 0.8]:
            cfg.vicsek_couplage = eta
            flock = PhysicsFlock(cfg)
            for _ in range(50):
                _call_force(vicsek_forces, flock, cfg)
                # Advance positions so birds can encounter new neighbours
                flock.positions[flock.active] += (
                    flock.velocities[flock.active] * cfg.vicsek_time_step
                )
            active = flock.velocities[flock.active]
            norms = np.linalg.norm(active, axis=1)
            dirs = active / norms[:, np.newaxis]
            a = np.linalg.norm(np.mean(dirs, axis=0))
            alphas.append(a)

        # Order should increase with couplage: high eta clearly above low eta
        assert alphas[-1] > alphas[0] + 0.15, (
            f"Expected phase transition, got {alphas[0]:.3f} → {alphas[-1]:.3f}"
        )

    def test_vicsek_does_not_modify_acceleration(self):
        """Vicsek sets velocity directly — verify it doesn't touch accelerations."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 50
        cfg.max_force = 1.0

        flock = PhysicsFlock(cfg)
        old_acc = flock.accelerations.copy()
        _call_force(vicsek_forces, flock, cfg)

        # Vicsek sets velocity, doesn't modify acceleration
        assert np.allclose(flock.accelerations[flock.active], old_acc[flock.active])

    def test_no_neighbours_within_radius(self):
        """Radius=0 → no neighbours → pure noise, no crash."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 50
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_radius_influence = 0.0  # no neighbours possible

        flock = PhysicsFlock(cfg)
        _call_force(vicsek_forces, flock, cfg)

        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(speeds, 1.0, atol=1e-4)

    def test_two_bird_alignment(self):
        """N=2: both birds see each other → directions converge."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 2
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.9
        cfg.vicsek_diffusion = 0.01
        cfg.vicsek_radius_influence = 1000.0

        flock = PhysicsFlock(cfg)
        for _ in range(20):
            _call_force(vicsek_forces, flock, cfg)

        vels = flock.velocities[flock.active]
        norms = np.linalg.norm(vels, axis=1, keepdims=True)
        dirs = vels / norms
        alpha = np.linalg.norm(np.mean(dirs, axis=0))
        assert alpha > 0.9, f"Two birds should align: alpha={alpha:.3f}"

    def test_does_not_modify_positions(self):
        """Vicsek only changes velocity, never position."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 30

        flock = PhysicsFlock(cfg)
        old_pos = flock.positions.copy()
        _call_force(vicsek_forces, flock, cfg)
        assert np.allclose(flock.positions, old_pos)
