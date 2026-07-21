"""Field/blob mode tests — P3.2–P3.12.

O(N) scaling, shell force, alignment, noise, slot repulsion,
tangential orbital, buoyancy, curl flow, fold noise, drag, drift,
floating boundary, edge cases.
"""

import time
import warnings

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.field import (
    _compute_anchors,
    _compute_buoyancy,
    _compute_curl_flow,
    _compute_drift_alignment,
    _compute_floating_boundary,
    _compute_leader_chaser,
    _compute_shell_force,
    _compute_slot_repulsion,
    _compute_tangential,
    _compute_targets,
    _compute_viscous_drag,
    field_forces,
)
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
        cfg.seed = 42  # D6: default seed is None — pin so geometry is deterministic
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
        cfg.field_noise = 0.0  # C3: field_noise now produces a real per-bird jitter term

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
        cfg.field_noise = 0.0  # C3: field_noise now produces a real per-bird jitter term

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
        cfg.field_noise = 0.0  # C3: field_noise now produces a real per-bird jitter term

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


class TestFieldConfigWiring:
    """C3: previously-orphan FieldConfig leaves now drive real terms."""

    def test_field_noise_scales_jitter(self):
        """field_noise=0 → no jitter; larger field_noise → larger jitter."""
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
        cfg.field_noise = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)
        assert np.allclose(flock.accelerations[flock.active][:, :2], 0.0)

        cfg.field_noise = 0.5
        flock2 = PhysicsFlock(cfg)
        flock2.accelerations[:] = 0.0
        _call_force(field_forces, flock2, cfg)
        acc_active = flock2.accelerations[flock2.active]
        assert np.max(np.abs(acc_active[:, :2])) > 0.05

    def test_field_num_groups_changes_group_count(self):
        """field_num_groups controls how many seed groups leader/chaser uses.

        `group_seed = floor(seeds*ng)/ng` only produces distinct fractional
        buckets for non-integer seeds (production seeds birds by plain bird
        index, which makes this term degenerate to identity regardless of
        ng — a pre-existing quirk out of scope here). Use fractional seeds
        to exercise the num_groups parameter itself in isolation.
        """
        seeds = np.arange(20, dtype=np.float32) * 0.37
        anchors = _compute_anchors(0.0, np.zeros(3, dtype=np.float32), 100.0)
        T_legacy = _compute_targets(seeds, 0.0, anchors)

        t3 = _compute_leader_chaser(
            seeds, 0.0, T_legacy, anchors, 100.0, chase_strength=0.5, sep=0.5,
            num_groups=3, leader_fraction=0.16,
        )
        t7 = _compute_leader_chaser(
            seeds, 0.0, T_legacy, anchors, 100.0, chase_strength=0.5, sep=0.5,
            num_groups=7, leader_fraction=0.16,
        )
        assert not np.allclose(t3, t7)

    def test_field_num_groups_default_matches_hardcoded_seven(self):
        """Default field_num_groups=7 reproduces the original hardcoded behaviour."""
        seeds = np.arange(20, dtype=np.float32)
        anchors = _compute_anchors(0.0, np.zeros(3, dtype=np.float32), 100.0)
        T_legacy = _compute_targets(seeds, 0.0, anchors)

        t_default = _compute_leader_chaser(
            seeds, 0.0, T_legacy, anchors, 100.0, chase_strength=0.5, sep=0.5,
        )
        t_explicit_7 = _compute_leader_chaser(
            seeds, 0.0, T_legacy, anchors, 100.0, chase_strength=0.5, sep=0.5,
            num_groups=7, leader_fraction=0.16,
        )
        np.testing.assert_allclose(t_default, t_explicit_7)

    def test_field_leader_fraction_changes_leader_count(self):
        """Higher field_leader_fraction classifies more birds as leaders."""
        seeds = np.arange(200, dtype=np.float32)
        anchors = _compute_anchors(0.0, np.zeros(3, dtype=np.float32), 100.0)
        T_legacy = _compute_targets(seeds, 0.0, anchors)

        t_low = _compute_leader_chaser(
            seeds, 0.0, T_legacy, anchors, 100.0, chase_strength=0.5, sep=0.5,
            num_groups=7, leader_fraction=0.05,
        )
        t_high = _compute_leader_chaser(
            seeds, 0.0, T_legacy, anchors, 100.0, chase_strength=0.5, sep=0.5,
            num_groups=7, leader_fraction=0.5,
        )
        assert not np.allclose(t_low, t_high)

    def test_field_shell_radius_base_changes_shell_size(self):
        """Larger field_shell_radius_base pushes the equilibrium shell outward."""
        seeds = np.arange(20, dtype=np.float32)
        positions = np.zeros((20, 3), dtype=np.float32)
        positions[:, 0] = 5.0  # offset from target so d_hat is well-defined
        targets = np.zeros((20, 3), dtype=np.float32)

        f_small = _compute_shell_force(
            positions, targets, seeds, 0.0, 100.0, cohesion=1.0,
            chase_strength=0.0, sep=0.0, shell_influence=1.0,
            shell_radius_base=0.1, inner_radius_factor=0.28,
        )
        f_large = _compute_shell_force(
            positions, targets, seeds, 0.0, 100.0, cohesion=1.0,
            chase_strength=0.0, sep=0.0, shell_influence=1.0,
            shell_radius_base=0.5, inner_radius_factor=0.28,
        )
        assert not np.allclose(f_small, f_large)

    def test_field_drift_direction_fallback_when_wander_disabled(self):
        """A nonzero field_drift_direction drives drift alignment without Wander."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_alignment = 1.0
        cfg.field_drift_pull = 0.5
        cfg.field_noise = 0.0
        # Isolate drift_alignment — buoyancy/shell/etc. run unconditionally
        # and would otherwise mask the effect under test.
        cfg.disabled_terms = [
            "shell", "target_pull", "slot_repulsion", "tangential", "buoyancy",
            "curl_flow", "fold_noise", "viscous_drag", "floating_boundary",
        ]
        cfg.field_drift_direction = (0.0, 0.0, 0.0)

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = 0.0
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)
        force_zero_dir = flock.accelerations[flock.active].copy()

        cfg.field_drift_direction = (1.0, 0.0, 0.0)
        flock2 = PhysicsFlock(cfg)
        flock2.velocities[:] = 0.0
        flock2.accelerations[:] = 0.0
        _call_force(field_forces, flock2, cfg)
        force_with_dir = flock2.accelerations[flock2.active]

        # Zero drift_direction (default) is a no-op fallback (still None → no force).
        assert np.allclose(force_zero_dir, 0.0)
        # A configured static direction produces real drift-alignment force.
        assert np.max(np.abs(force_with_dir)) > 1e-6

    def test_field_flow_pull_scales_curl_fold(self):
        """C3: field_flow_pull amplifies curl_flow and fold_noise forces."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 50
        cfg.field_cohesion = 0.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 1.0
        cfg.field_separation = 0.0
        cfg.field_chase_strength = 0.0
        cfg.field_tangent_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0
        cfg.field_noise = 0.0
        cfg.max_force = 500.0  # disable force clamping so flow_pull difference is visible
        # Isolate curl_flow + fold_noise only
        cfg.disabled_terms = [
            "shell", "slot_repulsion", "tangential", "buoyancy",
            "viscous_drag", "drift_alignment", "floating_boundary",
        ]
        cfg.field_flow_pull = 0.1

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)
        mag_low = float(np.mean(np.linalg.norm(flock.accelerations[flock.active], axis=1)))

        cfg.field_flow_pull = 2.0
        flock2 = PhysicsFlock(cfg)
        flock2.accelerations[:] = 0.0
        _call_force(field_forces, flock2, cfg)
        mag_high = float(np.mean(np.linalg.norm(flock2.accelerations[flock2.active], axis=1)))

        assert mag_high > mag_low * 1.5, (
            f"flow_pull=2.0 ({mag_high:.4f}) should be much larger than "
            f"flow_pull=0.1 ({mag_low:.4f})"
        )

    def test_field_inner_radius_factor_changes_cavity(self):
        """C3: Larger field_inner_radius_factor expands the inner cavity floor."""
        seeds = np.arange(20, dtype=np.float32)
        positions = np.zeros((20, 3), dtype=np.float32)
        positions[:, 0] = 1.0  # all just inside shell, so cavity matters
        targets = np.zeros((20, 3), dtype=np.float32)

        f_small = _compute_shell_force(
            positions, targets, seeds, 0.0, 100.0, cohesion=1.0,
            chase_strength=0.0, sep=1.0, shell_influence=1.0,
            shell_radius_base=0.32, inner_radius_factor=0.1,
        )
        f_large = _compute_shell_force(
            positions, targets, seeds, 0.0, 100.0, cohesion=1.0,
            chase_strength=0.0, sep=1.0, shell_influence=1.0,
            shell_radius_base=0.32, inner_radius_factor=0.6,
        )
        # Different inner_radius_factor → different forces (cavity push-out changes)
        assert not np.allclose(f_small, f_large), (
            "inner_radius_factor must change the inner cavity force"
        )

    def test_disabled_terms_skips_shell_force(self):
        """C3: disabled_terms=["shell"] zeroes the shell force contribution."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.field_cohesion = 1.0
        cfg.field_alignment = 0.0
        cfg.field_flow = 0.0
        cfg.field_separation = 0.0
        cfg.field_chase_strength = 0.0
        cfg.field_tangent_pull = 0.0
        cfg.field_flow_pull = 0.0
        cfg.field_drift_pull = 0.0
        cfg.boundary_avoidance_factor = 0.0
        cfg.field_noise = 0.0
        cfg.field_target_pull = 0.0

        # First run with shell enabled
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)
        acc_with_shell = flock.accelerations[flock.active].copy()

        # Now disable shell
        cfg.disabled_terms = ["shell"]
        flock2 = PhysicsFlock(cfg)
        flock2.accelerations[:] = 0.0
        _call_force(field_forces, flock2, cfg)
        acc_no_shell = flock2.accelerations[flock2.active]

        # With shell disabled, the ONLY remaining non-zero term is buoyancy
        # (which only produces z-axis forces).  All x/y forces should vanish.
        assert np.allclose(acc_no_shell[:, :2], 0.0), (
            'disabled_terms=["shell"] should zero all x/y forces '
            "(only buoyancy remains on z-axis)"
        )
        # And the forces should be different from when shell was active
        assert not np.allclose(acc_with_shell[:, :2], acc_no_shell[:, :2])


class TestFieldTermComposition:
    """S2.A5: FIELD_TERMS/composeForces composition contract."""

    def test_target_pull_formula(self):
        """S2.A5: F_target_pull = (T-p)/U * coh * target_pull."""
        from pymurmur.physics.forces.field import _compute_target_pull

        positions_active = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        targets = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        U = 10.0
        cohesion = 2.0
        target_pull = 0.5

        result = _compute_target_pull(positions_active, targets, U, cohesion, target_pull)
        expected = (targets - positions_active) / U * cohesion * target_pull
        np.testing.assert_allclose(result, expected)

    def test_target_pull_zero_gain_returns_zero(self):
        from pymurmur.physics.forces.field import _compute_target_pull

        positions_active = np.zeros((5, 3), dtype=np.float32)
        targets = np.ones((5, 3), dtype=np.float32)
        result = _compute_target_pull(positions_active, targets, 10.0, 1.0, 0.0)
        assert np.allclose(result, 0.0)

    def test_target_pull_no_longer_dead_config_field(self):
        """S2.A5: field_target_pull (Part III C3 deferral) now drives real force."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 20
        cfg.field_cohesion = 1.0
        cfg.field_target_pull = 0.5
        cfg.field_noise = 0.0
        cfg.disabled_terms = [
            "shell", "slot_repulsion", "tangential", "buoyancy",
            "curl_flow", "fold_noise", "viscous_drag", "drift_alignment",
            "floating_boundary",
        ]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)
        assert not np.allclose(flock.accelerations[flock.active], 0.0)

    def test_slot_repulsion_mod_wraps_first_and_last(self):
        """S2.A5: birds at opposite ends of the active-index ordering now
        interact via offset=1's mod-wrap (previously they never paired,
        an artefact of index ordering, not physical distance)."""
        from pymurmur.physics.forces.field import _compute_slot_repulsion

        n = 10
        positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            positions[i] = [i * 1000.0, 0.0, 0.0]
        # Bird 9 (last) placed right next to bird 0 (first) — only the
        # offset=1 wrap pairing (9 -> 0) can explain a force between them.
        positions[9] = positions[0] + np.array([1.0, 0.0, 0.0], dtype=np.float32)
        active = np.ones(n, dtype=bool)

        F = _compute_slot_repulsion(positions, active, n, U=100.0, separation=1.0, chase_strength=0.0)

        assert not np.allclose(F[0], 0.0), "Mod-wrapped pairing (9->0) should produce force on bird 0"
        assert not np.allclose(F[9], 0.0), "Mod-wrapped pairing (9->0) should produce force on bird 9"
        np.testing.assert_allclose(F[0], -F[9], atol=1e-4)  # action-reaction

    def test_disabled_terms_unknown_name_warns(self):
        """S2.A5: an unrecognized disabled_terms entry warns instead of silently no-op'ing."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10
        cfg.disabled_terms = ["not_a_real_term"]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        with pytest.warns(UserWarning, match="unknown term name"):
            _call_force(field_forces, flock, cfg)

    def test_disabled_terms_known_names_do_not_warn(self):
        """Sanity: real term names never trigger the unknown-name warning."""
        from pymurmur.physics.forces.field import FIELD_TERMS

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10
        cfg.disabled_terms = [term.name for term in FIELD_TERMS]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _call_force(field_forces, flock, cfg)

    def test_full_step_equals_sum_of_individually_isolated_terms(self):
        """S2.A5: Σ (each term run alone via disabled_terms) == full step
        with every term enabled — proves FIELD_TERMS/composeForces sums
        linearly, as the contract requires."""
        from pymurmur.physics.forces.field import FIELD_TERMS

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 25
        cfg.seed = 3
        cfg.max_force = 1e6  # disable clamping so the sum stays exact
        cfg.field_target_pull = 0.3
        all_names = [term.name for term in FIELD_TERMS]

        def run(disabled: list[str]) -> np.ndarray:
            cfg.disabled_terms = disabled
            flock = PhysicsFlock(cfg)
            flock.accelerations[:] = 0.0
            _call_force(field_forces, flock, cfg)
            return flock.accelerations[flock.active].copy()

        full = run([])
        total = np.zeros_like(full)
        for name in all_names:
            isolate_disabled = [n for n in all_names if n != name]
            total += run(isolate_disabled)

        np.testing.assert_allclose(full, total, atol=1e-3)

    def test_disabling_one_term_changes_sum_by_exactly_its_contribution(self):
        """S2.A5: full − (full with X disabled) == X run in isolation."""
        from pymurmur.physics.forces.field import FIELD_TERMS

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 25
        cfg.seed = 3
        cfg.max_force = 1e6
        cfg.field_target_pull = 0.3
        all_names = [term.name for term in FIELD_TERMS]

        def run(disabled: list[str]) -> np.ndarray:
            cfg.disabled_terms = disabled
            flock = PhysicsFlock(cfg)
            flock.accelerations[:] = 0.0
            _call_force(field_forces, flock, cfg)
            return flock.accelerations[flock.active].copy()

        full = run([])
        for name in all_names:
            without = run([name])
            alone = run([n for n in all_names if n != name])
            np.testing.assert_allclose(
                full - without, alone, atol=1e-3,
                err_msg=f"term {name!r}: (full - without) != alone",
            )
