"""P1.8 — Vicsek Core tests (memory term, tangent-plane noise, D-live).

Tests the current Vicsek implementation against its documented contract.
P1.8 features (memory term, tangent-plane noise, D parameter live) are
marked xfail until implemented per roadmap_deepseek.md P1.8.
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.vicsek import vicsek_forces
from pymurmur.physics.flock import PhysicsFlock

pytestmark = pytest.mark.guard


class TestVicsekCore:
    """Tests for the Vicsek update kernel — constant speed, noise, alignment."""

    # ── Current behaviour (must pass now) ────────────────────────

    def test_constant_speed_preserved(self):
        """All birds maintain exactly config.vicsek_velocity after one step."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 100
        cfg.vicsek_velocity = 2.5

        flock = PhysicsFlock(cfg)
        vicsek_forces(flock, cfg)

        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(speeds, 2.5, atol=1e-4)

    def test_noise_applied_to_lone_bird(self):
        """Single bird with no neighbours gets pure noise direction."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 1
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_diffusion = 0.5

        flock = PhysicsFlock(cfg)
        np.random.seed(99)
        vicsek_forces(flock, cfg)

        speed = np.linalg.norm(flock.velocities[0])
        assert speed == pytest.approx(1.0, abs=1e-4)
        # Direction is unit-length (normalised noise)
        direction = flock.velocities[0] / speed
        assert pytest.approx(np.linalg.norm(direction), abs=1e-6) == 1.0

    def test_two_birds_align(self):
        """N=2 within radius: directions converge (high couplage, low noise)."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 2
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.95
        cfg.vicsek_diffusion = 0.01
        cfg.vicsek_radius_influence = 1000.0

        flock = PhysicsFlock(cfg)
        for _ in range(30):
            vicsek_forces(flock, cfg)

        vels = flock.velocities[flock.active]
        norms = np.linalg.norm(vels, axis=1, keepdims=True)
        dirs = vels / norms
        alpha = np.linalg.norm(np.mean(dirs, axis=0))
        assert alpha > 0.95, f"Two birds should align: alpha={alpha:.3f}"

    def test_does_not_modify_positions(self):
        """Vicsek only changes velocity — positions must be unchanged."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 30

        flock = PhysicsFlock(cfg)
        old_pos = flock.positions.copy()
        vicsek_forces(flock, cfg)
        assert np.allclose(flock.positions, old_pos)

    def test_does_not_modify_acceleration(self):
        """Vicsek sets velocity directly — acceleration array untouched."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 50

        flock = PhysicsFlock(cfg)
        old_acc = flock.accelerations.copy()
        vicsek_forces(flock, cfg)
        assert np.allclose(flock.accelerations[flock.active], old_acc[flock.active])

    def test_directions_are_unit_length(self):
        """Every active bird's velocity direction is unit length."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 100
        cfg.vicsek_velocity = 3.0
        cfg.vicsek_diffusion = 0.8

        flock = PhysicsFlock(cfg)
        vicsek_forces(flock, cfg)

        vels = flock.velocities[flock.active]
        norms = np.linalg.norm(vels, axis=1)
        dirs = vels / norms[:, np.newaxis]
        dir_norms = np.linalg.norm(dirs, axis=1)
        assert np.allclose(dir_norms, 1.0, atol=1e-4)

    def test_zero_active_noop(self):
        """Empty flock: no change to velocities."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        old_vel = flock.velocities.copy()
        vicsek_forces(flock, cfg)
        assert np.allclose(flock.velocities, old_vel)

    def test_high_diffusion_reduces_order(self):
        """High D compared to low D: order drops (noise overwhelms alignment)."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 100
        cfg.width = 50
        cfg.height = 50
        cfg.depth = 50
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.5
        cfg.vicsek_radius_influence = 50.0
        cfg.seed = 42

        # Low noise
        cfg.vicsek_diffusion = 0.1
        flock_lo = PhysicsFlock(cfg)
        for _ in range(15):
            vicsek_forces(flock_lo, cfg)
        alpha_lo = _compute_alpha(flock_lo)

        # High noise
        cfg.vicsek_diffusion = 2.0
        flock_hi = PhysicsFlock(cfg)
        for _ in range(15):
            vicsek_forces(flock_hi, cfg)
        alpha_hi = _compute_alpha(flock_hi)

        assert alpha_lo > alpha_hi, (
            f"Low D should order more: alpha_lo={alpha_lo:.3f}, alpha_hi={alpha_hi:.3f}"
        )

    # ── P1.8 features (xfail until implemented) ──────────────────

    @pytest.mark.xfail(
        reason="P1.8: no memory term — direction autocorrelation decays immediately "
               "instead of persisting (needs û_noisy = normalize(û_old + √(2·D·Δt)·n_⊥))"
    )
    def test_memory_term_autocorrelation(self):
        """Lone bird, D=0, no neighbours → direction autocorr > 0.999 at lag 1."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 1
        cfg.vicsek_velocity = 1.0
        cfg.vicsek_couplage = 0.5
        cfg.vicsek_diffusion = 0.0  # zero noise
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        directions = []
        for _ in range(20):
            vicsek_forces(flock, cfg)
            v = flock.velocities[0]
            directions.append(v / np.linalg.norm(v))

        dirs = np.array(directions)
        # Autocorrelation at lag 1
        ac = np.dot(dirs[:-1], dirs[1:]) / (len(dirs) - 1)
        assert ac > 0.999, f"Memory term missing: ac(1)={ac:.6f}"

    @pytest.mark.xfail(
        reason="P1.8: D parameter currently normalised away — D=0.01 vs D=4 "
               "should produce different autocorrelation, but both are identical"
    )
    def test_d_parameter_live(self):
        """D=0.01 → high autocorr; D=4 → autocorr < 0.5 at lag 1."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 1
        cfg.vicsek_couplage = 0.0  # no neighbour coupling
        cfg.seed = 42

        def autocorr_at_D(D_val):
            cfg.vicsek_diffusion = D_val
            flock = PhysicsFlock(cfg)
            directions = []
            for _ in range(20):
                vicsek_forces(flock, cfg)
                v = flock.velocities[0]
                directions.append(v / np.linalg.norm(v))
            dirs = np.array(directions)
            return np.dot(dirs[:-1], dirs[1:]) / (len(dirs) - 1)

        ac_low = autocorr_at_D(0.01)
        ac_high = autocorr_at_D(4.0)
        assert ac_low > 0.8, f"Expected high autocorr at D=0.01, got {ac_low:.3f}"
        assert ac_high < 0.5, f"Expected low autocorr at D=4, got {ac_high:.3f}"

    @pytest.mark.xfail(
        reason="P1.8: noise is applied in R³ then normalised, not projected onto "
               "tangent plane (n_⊥ = g − (g·û)·û). |n_⊥·û| should be < 1e-6."
    )
    def test_noise_in_tangent_plane(self):
        """Noise component parallel to current direction is zero."""
        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = 1
        cfg.vicsek_diffusion = 0.3
        cfg.seed = 42

        flock = PhysicsFlock(cfg)
        # Run once to get an initial direction, then check delta direction
        vicsek_forces(flock, cfg)
        dir_before = flock.velocities[0] / np.linalg.norm(flock.velocities[0])

        # Run again and check the change vector
        vicsek_forces(flock, cfg)
        dir_after = flock.velocities[0] / np.linalg.norm(flock.velocities[0])
        delta = dir_after - dir_before

        # The change should be orthogonal to the original direction
        parallel_comp = np.abs(np.dot(delta, dir_before))
        assert parallel_comp < 1e-6, (
            f"Noise has parallel component: |Δ·û| = {parallel_comp:.2e}"
        )


def _compute_alpha(flock: PhysicsFlock) -> float:
    """Helper: compute polar order parameter."""
    vels = flock.velocities[flock.active]
    norms = np.linalg.norm(vels, axis=1, keepdims=True)
    dirs = vels / (norms + 1e-10)
    return float(np.linalg.norm(np.mean(dirs, axis=0)))
