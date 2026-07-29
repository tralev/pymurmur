"""Field/blob mode tests — edge cases and O(N) scaling.

Split out of test_field.py (file-size split) — core force-term tests
(shell, blob anchors, leader/chaser, slot repulsion, individual terms,
floating boundary) stay in the original; this file covers the "Edge
cases" and "Scaling" sections.
"""

import time

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.field import field_forces
from test.helpers import _call_force


class TestFieldModeScalingEdgeCases:
    """crs48 field/blob anchor mode — edge cases and O(N) scaling."""

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
