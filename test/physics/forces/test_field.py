"""Field/blob mode tests — P3.2–P3.12.

O(N) scaling, shell force, alignment, noise, slot repulsion,
tangential orbital, buoyancy, curl flow, fold noise, drag, drift,
floating boundary, edge cases.
"""

import time
import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.field import (
    field_forces,
    _compute_anchors,
    _compute_targets,
    _compute_phases,
    _compute_leader_chaser,
    _compute_shell_force,
    _compute_slot_repulsion,
    _compute_tangential,
    _compute_buoyancy,
    _compute_curl_flow,
    _compute_fold_noise,
    _compute_viscous_drag,
    _compute_drift_alignment,
    _compute_floating_boundary,
)
from pymurmur.physics.flock import PhysicsFlock
from test.helpers import _call_force


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
        _call_force(field_forces, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.mean(acc_mags > 1e-6) > 0.5, (
            f"Only {np.mean(acc_mags > 1e-6)*100:.0f}% of birds felt forces"
        )

    def test_shell_force_pulls_toward_target(self):
        """P3.4: Shell force pulls birds toward their per-bird targets."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_cohesion = 1.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0
        cfg.field_chase_strength = 0.0
        # Silence non-shell terms
        cfg.field_tangent_pull = 0.0
        cfg.field_flow_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)

        # Compute per-bird targets the same way compute() does
        active = flock.active
        t = getattr(cfg, '_field_time', 0.0)
        C = np.mean(flock.positions[active], axis=0)
        U = 0.4 * min(cfg.width, cfg.height, cfg.depth)
        seeds = np.arange(active.sum(), dtype=np.float32)
        anchors = _compute_anchors(t, C, U)
        T_legacy = _compute_targets(seeds, t, anchors)
        targets = _compute_leader_chaser(seeds, t, T_legacy, anchors, U, 0.0, 0.0)

        for idx, i in enumerate(np.where(active)[0]):
            to_target = targets[idx] - flock.positions[i]
            d = np.linalg.norm(to_target)
            if d < 1e-6:
                continue
            acc = flock.accelerations[i]
            if np.linalg.norm(acc) < 1e-6:
                continue
            dot = np.dot(acc, to_target)
            assert dot > 0, (
                f"Bird {i} acceleration points away from target: dot={dot:.4f}"
            )

    def test_flow_noise_produces_nonzero_force(self):
        """P3.6: Curl flow + fold noise produce non-zero acceleration."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 50
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 10.0
        cfg.field_separation = 0.0
        cfg.field_flow_pull = 10.0
        cfg.field_chase_strength = 0.0
        cfg.field_tangent_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.mean(acc_mags > 1e-6) > 0.5

    def test_slot_repulsion_separates(self):
        """P3.5: Slot repulsion pushes birds with offset neighbours apart."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 100
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 10.0
        cfg.max_force = 1000.0
        cfg.field_chase_strength = 0.0
        cfg.field_tangent_pull = 0.0
        cfg.field_flow_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.sum(acc_mags > 1e-6) >= 2, (
            f"Only {np.sum(acc_mags > 1e-6)} birds felt repulsion"
        )

    # ── P3.2: Blob anchors ─────────────────────────────────────

    def test_anchors_match_hand_values(self):
        """P3.2 anchors at t=0 produce known values."""
        C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        anchors = _compute_anchors(0.0, C, 1.0)
        assert anchors.shape == (5, 3)
        # B₀ at t=0: (0, sin(0.8)*0.48, cos(0)*0.62) = (0, ~0.344, 0.62)
        np.testing.assert_allclose(anchors[0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(anchors[0, 2], 0.62, atol=1e-5)

    # ── P3.3: Leader/chaser ────────────────────────────────────

    def test_chase_zero_returns_t_legacy(self):
        """P3.3: chase_strength=0 returns T_legacy unchanged."""
        seeds = np.arange(10, dtype=np.float32)
        T_in = np.random.randn(10, 3).astype(np.float32)
        anchors = _compute_anchors(0.0, np.zeros(3), 1.0)
        T_out = _compute_leader_chaser(seeds, 0.0, T_in, anchors, 1.0, 0.0, 0.0)
        np.testing.assert_allclose(T_out, T_in, atol=1e-6)

    def test_leader_fraction_roughly_16_percent(self):
        """P3.3: ~16% of birds are leaders."""
        from pymurmur.physics.forces.field import _hash01
        seeds = np.arange(1000, dtype=np.float32)
        is_leader = _hash01(seeds + 5.91) >= 0.84
        fraction = is_leader.mean()
        assert 0.14 <= fraction <= 0.18, (
            f"Leader fraction {fraction:.3f} not in [0.14, 0.18]"
        )

    # ── P3.4: Shell + cavity ───────────────────────────────────

    def test_shell_force_pulls_outside_birds_inward(self):
        """P3.4: Birds outside R_blob are pulled toward target."""
        n = 5
        positions = np.array([[10.0, 0.0, 0.0],
                               [5.0, 0.0, 0.0],
                               [-5.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.5, 0.0, 0.0]], dtype=np.float32)
        targets = np.zeros_like(positions)
        seeds = np.arange(n, dtype=np.float32)
        F = _compute_shell_force(positions, targets, seeds, 0.0, 10.0, 1.0, 0.0, 0.0, 1.0)
        # Bird at x=10 is far outside R_blob (~3.2) → should be pulled toward target (x=0)
        assert F[0, 0] < 0, f"Far bird not pulled inward: F[0,0]={F[0,0]:.4f}"

    def test_shell_force_zero_at_equilibrium(self):
        """P3.4: Bird at exactly R_blob feels zero radial force."""
        n = 1
        U = 1.0
        R = 0.32 * U  # base R_blob at t=0, seed=0, phase=0
        pos = np.array([[R, 0.0, 0.0]], dtype=np.float32)
        targ = np.zeros_like(pos)
        seeds = np.zeros(1, dtype=np.float32)
        F = _compute_shell_force(pos, targ, seeds, 0.0, U, 1.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(F, 0.0, atol=1e-4)

    # ── P3.5: Slot repulsion ───────────────────────────────────

    def test_slot_kernel_zero_at_r_slot(self):
        """P3.5: Quadratic kernel is zero at r_slot boundary."""
        n = 2
        U = 1.0
        sep = 1.0
        r_slot = (0.07 + sep * 0.02) * U  # = 0.09
        # Place two birds exactly at r_slot distance
        positions = np.array([[0.0, 0.0, 0.0],
                               [r_slot, 0.0, 0.0]], dtype=np.float32)
        active = np.ones(n, dtype=bool)
        F = _compute_slot_repulsion(positions, active, n, U, sep, 0.0)
        # At exactly r_slot, kernel should be ~0
        assert np.allclose(F, 0.0, atol=1e-6), (
            f"Non-zero force at r_slot boundary: {F}"
        )

    def test_slot_kernel_positive_inside_r_slot(self):
        """P3.5: Birds closer than r_slot feel repulsion."""
        n = 2
        U = 1.0
        sep = 1.0
        r_slot = (0.07 + sep * 0.02) * U
        positions = np.array([[0.0, 0.0, 0.0],
                               [r_slot * 0.5, 0.0, 0.0]], dtype=np.float32)
        active = np.ones(n, dtype=bool)
        F = _compute_slot_repulsion(positions, active, n, U, sep, 0.0)
        # Bird 0 pushed away from bird 1 → F[0, 0] < 0 (left)
        # Bird 1 pushed away from bird 0 → F[1, 0] > 0 (right)
        assert F[0, 0] < 0, f"Bird 0 not pushed left: F[0,0]={F[0,0]}"
        assert F[1, 0] > 0, f"Bird 1 not pushed right: F[1,0]={F[1,0]}"

    # ── P3.6: Individual term tests ────────────────────────────

    def test_tangential_produces_orbital_force(self):
        """P3.6: Tangential force is perpendicular to (p-T)."""
        n = 3
        pos = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
        targ = np.zeros_like(pos)
        seeds = np.arange(n, dtype=np.float32)
        F = _compute_tangential(pos, targ, seeds, 0.0, 1.0, 0.0, 1.0)
        # Tangential force should be perpendicular to (pos - targ)
        to_target = pos - targ
        dots = np.sum(F * to_target, axis=1)
        np.testing.assert_allclose(dots, 0.0, atol=1e-4)

    def test_buoyancy_is_z_only(self):
        """P3.6: Buoyancy force only has z-component."""
        n = 3
        pos = np.random.randn(n, 3).astype(np.float32)
        targ = np.random.randn(n, 3).astype(np.float32)
        seeds = np.arange(n, dtype=np.float32)
        F = _compute_buoyancy(pos, targ, seeds, 0.0, 100.0, 1.0)
        # x and y components should be zero
        np.testing.assert_allclose(F[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(F[:, 1], 0.0, atol=1e-6)
        # z component should be non-zero on average
        assert np.mean(np.abs(F[:, 2])) > 1e-6

    def test_curl_flow_is_unit_length_pre_gain(self):
        """P3.6: Curl flow vectors are normalized before gain."""
        n = 100
        pos = np.random.randn(n, 3).astype(np.float32) * 10.0
        C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        seeds = np.arange(n, dtype=np.float32)
        F = _compute_curl_flow(pos, C, seeds, 0.0, 1.0, 1.0, 1.0)
        # With gain=1, max force should be flow * 0.08 = 0.08
        mags = np.linalg.norm(F, axis=1)
        assert np.max(mags) <= 0.08 + 1e-4

    def test_viscous_drag_opposes_velocity(self):
        """P3.6: Drag force is anti-parallel to velocity."""
        v = np.array([[1.0, 2.0, 0.0],
                       [-3.0, 1.0, 0.0]], dtype=np.float32)
        F = _compute_viscous_drag(v, 1.0, 1.0)
        # Drag should point opposite to velocity
        for i in range(2):
            dot = np.dot(F[i], v[i])
            assert dot < 0, f"Drag not anti-parallel: dot={dot}"

    def test_drift_alignment_zeros_with_none_heading(self):
        """P3.6: Drift alignment returns zero when wander_heading is None."""
        v = np.random.randn(10, 3).astype(np.float32)
        F = _compute_drift_alignment(v, None, 4.0, 1.0, 1.0)
        np.testing.assert_allclose(F, 0.0)

    # ── P3.12: Floating boundary ───────────────────────────────

    def test_floating_boundary_contains_outside_birds(self):
        """P3.12: Birds beyond R_boundary are pushed inward."""
        n = 2
        C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos = np.array([[10.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0]], dtype=np.float32)
        R_blobs = np.array([1.0, 1.0], dtype=np.float32)  # R_boundary = 1.45
        F = _compute_floating_boundary(pos, C, R_blobs, 1.0, mu=0.1)
        # Bird at x=10 is far outside → pushed toward origin (F_x < 0)
        assert F[0, 0] < 0
        # Bird at x=0.5 is inside → zero force
        np.testing.assert_allclose(F[1], 0.0, atol=1e-6)

    # ── Edge cases ──────────────────────────────────────────────

    def test_zero_active(self):
        """Empty flock produces no change."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        old_acc = flock.accelerations.copy()
        _call_force(field_forces, flock, cfg)
        assert np.allclose(flock.accelerations, old_acc)

    def test_single_bird(self):
        """Single bird: no crash, forces computed normally."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 1
        cfg.field_flow = 1.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)

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
        cfg.field_chase_strength = 1.0
        cfg.field_tangent_pull = 100.0
        cfg.field_flow_pull = 100.0
        cfg.field_drift_pull = 100.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)

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
            _call_force(field_forces, flock, cfg)
            times.append(time.perf_counter() - t0)

        ratio_t = times[-1] / times[0]
        ratio_n = sizes[-1] / sizes[0]
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
        _call_force(field_forces, flock, cfg)

        assert np.isfinite(flock.accelerations).all()
        assert flock.accelerations.shape == (10000, 3)

    def test_separation_zero_skips_repulsion(self):
        """field_separation=0 → slot repulsion skipped, but other forces still apply."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 100
        cfg.field_cohesion = 1.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 1.0
        cfg.field_separation = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.mean(acc_mags > 1e-6) > 0.5, (
            "Forces should still apply when separation is disabled"
        )

    def test_all_weights_zero_produces_only_buoyancy(self):
        """With all config weights zero, only buoyancy produces forces.

        Buoyancy always runs (it's a fundamental term, not weight-gated)
        and produces z-axis forces from (T_z−p_z)/U differences.
        All other terms should be silenced.
        """
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 50
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0
        cfg.field_chase_strength = 0.0
        cfg.field_tangent_pull = 0.0
        cfg.field_flow_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)

        acc_active = flock.accelerations[flock.active]
        # All forces should be purely z-axis (buoyancy only)
        x_mags = np.abs(acc_active[:, 0])
        y_mags = np.abs(acc_active[:, 1])
        assert np.max(x_mags) < 1e-6, f"x-axis forces: max={np.max(x_mags):.6f}"
        assert np.max(y_mags) < 1e-6, f"y-axis forces: max={np.max(y_mags):.6f}"
        # z-axis may have small buoyancy forces
        assert np.any(np.abs(acc_active[:, 2]) > 0)

    def test_inactive_birds_unchanged(self):
        """Inactive birds get zero force while active ones move."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_cohesion = 1.0
        cfg.field_flow = 1.0

        flock = PhysicsFlock(cfg)
        flock.active[10:20] = False
        flock.accelerations[:] = 0.0
        old_acc_inactive = flock.accelerations[~flock.active].copy()

        _call_force(field_forces, flock, cfg)

        assert np.allclose(flock.accelerations[~flock.active], old_acc_inactive)
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6)
