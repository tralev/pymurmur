"""Field/blob mode tests — config wiring (C3: previously-orphan
FieldConfig leaves now drive real terms).

Split out of test_field_config_composition_presets.py (file-size split)
— FieldTermComposition and FieldPresets stay in the original; this file
covers FieldConfigWiring.
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.field import (
    _compute_anchors,
    _compute_leader_chaser,
    _compute_shell_force,
    _compute_targets,
    field_forces,
)
from test.helpers import _call_force


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
