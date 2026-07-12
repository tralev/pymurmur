"""Field/blob mode tests — Phase 8.2

O(N) scaling, shell force, alignment, noise, slot repulsion, edge cases.
"""

import time
import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.field import field_forces
from pymurmur.physics.flock import PhysicsFlock


class TestFieldMode:
    """crs48 field/blob anchor mode — O(N) per-bird, no neighbour queries."""

    # ── Core behaviour ──────────────────────────────────────────

    def test_produces_nonzero_forces(self):
        """Field mode produces non-zero accelerations with default settings."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 100
        cfg.field_cohesion = 1.0
        cfg.field_alignment = 1.0
        cfg.field_flow = 1.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        # At least half the flock should feel forces
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.mean(acc_mags > 1e-6) > 0.5, (
            f"Only {np.mean(acc_mags > 1e-6)*100:.0f}% of birds felt forces"
        )

    def test_shell_force_pulls_toward_com(self):
        """Shell force: all birds pulled toward the center of mass."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 50
        cfg.field_cohesion = 100.0  # strong shell pull
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        com = np.mean(flock.positions[flock.active], axis=0)
        for i in np.where(flock.active)[0]:
            to_com = com - flock.positions[i]
            if np.linalg.norm(to_com) < 1e-6:
                continue  # bird at CoM — skip
            acc = flock.accelerations[i]
            # Acceleration should point toward CoM
            dot = np.dot(acc, to_com)
            assert dot > 0, (
                f"Bird {i} acceleration points away from CoM: dot={dot:.4f}"
            )

    def test_alignment_pulls_toward_avg_velocity(self):
        """Alignment force: birds steered toward the average flock velocity."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 20
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 100.0  # strong alignment
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        avg_vel = np.mean(flock.velocities[flock.active], axis=0)
        for i in np.where(flock.active)[0]:
            v = flock.velocities[i]
            to_avg = avg_vel - v
            if np.linalg.norm(to_avg) < 1e-6:
                continue  # already at avg — skip
            # Acceleration should point toward average velocity
            dot = np.dot(flock.accelerations[i], to_avg)
            assert dot > 0, (
                f"Bird {i} acc opposes avg vel: dot={dot:.4f}"
            )

    def test_flow_noise_produces_nonzero_force(self):
        """Flow noise alone produces non-zero acceleration in each bird."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 50
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 10.0
        cfg.field_separation = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.all(acc_mags > 1e-6)

    def test_slot_repulsion_separates(self):
        """Slot repulsion pushes birds with offset neighbours apart."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 100
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 10.0  # strong repulsion
        cfg.max_force = 1000.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        # At least a few birds should feel repulsion (offset-based, ~3 per offset)
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.sum(acc_mags > 1e-6) >= 2, (
            f"Only {np.sum(acc_mags > 1e-6)} birds felt repulsion"
        )

    # ── Edge cases ──────────────────────────────────────────────

    def test_zero_active(self):
        """Empty flock produces no change."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        old_acc = flock.accelerations.copy()
        field_forces(flock, cfg)
        assert np.allclose(flock.accelerations, old_acc)

    def test_single_bird(self):
        """Single bird: no crash, forces computed normally."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 1
        cfg.field_flow = 1.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        # Noise force should still be applied
        acc_mag = np.linalg.norm(flock.accelerations[0])
        assert acc_mag > 0

    def test_force_clamped_to_max(self):
        """No acceleration exceeds config.max_force."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 100
        cfg.field_cohesion = 100.0
        cfg.field_alignment = 100.0
        cfg.field_flow = 100.0
        cfg.field_separation = 100.0
        cfg.max_force = 5.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.all(acc_mags <= cfg.max_force + 1e-6), (
            f"Max force exceeded: {acc_mags.max():.3f} > {cfg.max_force}"
        )

    # ── Scaling ─────────────────────────────────────────────────

    def test_o_n_scaling(self):
        """Field mode scales roughly O(N) — time ratio << N ratio."""
        cfg = SimConfig()
        cfg.mode = "field"

        times = []
        sizes = [500, 1000, 2000, 5000]
        for n in sizes:
            cfg.num_boids = n
            flock = PhysicsFlock(cfg)
            flock.accelerations[:] = 0.0
            t0 = time.perf_counter()
            field_forces(flock, cfg)
            times.append(time.perf_counter() - t0)

        ratio_t = times[-1] / times[0]
        ratio_n = sizes[-1] / sizes[0]  # 10x
        assert ratio_t < ratio_n * 2, (
            f"Field mode not O(N): t ratio {ratio_t:.1f}x for n ratio {ratio_n:.1f}x"
        )

    def test_10k_boids(self):
        """Field mode handles 10K+ boids without error."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10000

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        # Should complete without NaN or inf
        assert np.isfinite(flock.accelerations).all()
        assert flock.accelerations.shape == (10000, 3)

    def test_separation_zero_skips_repulsion(self):
        """field_separation=0 → offset loop skipped, but other forces still apply."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 100
        cfg.field_cohesion = 1.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 1.0
        cfg.field_separation = 0.0  # guard: if > 0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        # Other forces still apply even with separation=0
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.mean(acc_mags > 1e-6) > 0.5, (
            "Forces should still apply when separation is disabled"
        )

    def test_all_weights_zero(self):
        """All force weights zero → zero acceleration."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 50
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.allclose(acc_mags, 0.0, atol=1e-6)

    def test_inactive_birds_unchanged(self):
        """Inactive birds get zero force while active ones move."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_cohesion = 1.0
        cfg.field_flow = 1.0

        flock = PhysicsFlock(cfg)
        flock.active[10:20] = False  # deactivate middle third
        flock.accelerations[:] = 0.0
        old_acc_inactive = flock.accelerations[~flock.active].copy()

        field_forces(flock, cfg)

        # Inactive birds: acceleration unchanged
        assert np.allclose(flock.accelerations[~flock.active], old_acc_inactive)
        # Active birds: some force applied
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6)

    def test_shell_force_directions_consistent(self):
        """Every bird's shell force points toward the CoM (not away or sideways)."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_cohesion = 100.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        field_forces(flock, cfg)

        com = np.mean(flock.positions[flock.active], axis=0)
        active_idx = np.where(flock.active)[0]
        for i in active_idx:
            to_com = com - flock.positions[i]
            if np.linalg.norm(to_com) < 1e-6:
                continue
            # Acceleration must point toward CoM (not away)
            dot = np.dot(flock.accelerations[i], to_com)
            assert dot > 0, (
                f"Bird {i} shell force points away from CoM: dot={dot:.4f}"
            )
