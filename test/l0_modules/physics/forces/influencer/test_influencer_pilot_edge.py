"""Phase 7 — Influencer pilot mode (P7.6, WASD attractor) and edge cases (zero active, single bird, substeps).

Split out of test_influencer.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.influencer import (
    InfluencerMode,
    PilotTarget,
    influencer_forces,
)
from pymurmur.simulation.engine import SimulationEngine
from test.helpers import _call_force


class TestInfluencerModePilotEdge:
    """P7.6 pilot mode + edge cases."""

    # ── P7.6: Pilot mode ────────────────────────────────────────

    def test_pilot_target_set_and_clear(self):
        """P7.6: PilotTarget set/clear via set_pilot()."""
        pilot = PilotTarget(
            position=np.array([100.0, 200.0, 300.0], dtype=np.float32),
            heading=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        assert InfluencerMode._pilot is None
        InfluencerMode.set_pilot(pilot)
        assert InfluencerMode._pilot is pilot
        InfluencerMode.set_pilot(None)
        assert InfluencerMode._pilot is None

    def test_pilot_mode_produces_velocity_changes(self):
        """P7.6: With active pilot, birds steer toward pilot position."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 2

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_vels = flock.velocities.copy()

        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            )
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            vel_diffs = np.linalg.norm(
                flock.velocities[flock.active] - old_vels[flock.active], axis=1
            )
            assert np.all(vel_diffs > 1e-6), "No velocity change with active pilot"
        finally:
            InfluencerMode.set_pilot(None)

    def test_shell_radius_expands_on_scatter(self):
        """P7.6: Scatter increases shell_radius monotonically up to 2.2 cap."""
        pilot = PilotTarget()
        pilot.shell_radius = 1.0
        dt = 1.0 / 60.0

        radii = [pilot.shell_radius]
        for _ in range(100):
            pilot.update_shell(dt, scatter=True, gather=False)
            radii.append(pilot.shell_radius)

        # Monotonically non-decreasing
        for i in range(1, len(radii)):
            assert radii[i] >= radii[i - 1] - 1e-10, (
                f"Shell radius decreased at step {i}: {radii[i]:.4f} < {radii[i-1]:.4f}"
            )

        # Should reach or approach the 2.2 cap
        assert radii[-1] >= 2.19, f"Shell radius {radii[-1]:.4f} didn't reach cap 2.2"

    def test_shell_radius_contracts_on_gather(self):
        """P7.6: Gather decreases shell_radius monotonically down to 0.42 floor."""
        pilot = PilotTarget()
        pilot.shell_radius = 2.0
        dt = 1.0 / 60.0

        radii = [pilot.shell_radius]
        for _ in range(100):
            pilot.update_shell(dt, scatter=False, gather=True)
            radii.append(pilot.shell_radius)

        # Monotonically non-increasing
        for i in range(1, len(radii)):
            assert radii[i] <= radii[i - 1] + 1e-10, (
                f"Shell radius increased at step {i}: {radii[i]:.4f} > {radii[i-1]:.4f}"
            )

        # Should reach or approach the 0.42 floor
        assert radii[-1] <= 0.43, f"Shell radius {radii[-1]:.4f} didn't reach floor 0.42"

    def test_pilot_heading_force_isolated(self):
        """P7.6: F_heading = pilot_heading * 0.12 when bird at pilot position."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 1
        cfg.influencer_substeps = 5  # more substeps → heading force accumulates
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        # Place bird exactly at pilot position (negates core_follow and shell_pull)
        pilot_pos = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        flock.positions[0] = pilot_pos.copy()
        flock.velocities[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # heading +z

        pilot = PilotTarget(
            position=pilot_pos.copy(),
            heading=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # heading +x
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            # Heading is (1,0,0), initial vel is (0,0,1).
            # heading_force is very small (0.12), so the +x component is tiny
            # after normalization.  Just verify it's nonzero (heading works).
            vel = flock.velocities[0]
            assert vel[0] > 0, f"Heading force should add +x component: vel={vel}"
        finally:
            InfluencerMode.set_pilot(None)

    def test_pilot_core_follow_isolated(self):
        """P7.6: F_core = (pilot_pos - p_i) * 0.22, inside shell_radius."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 1
        cfg.influencer_substeps = 5
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        # Bird at (510,350,200), pilot at (500,350,200).  d=10 < shell_radius=50.
        # Use orthogonal initial velocity (0,0,1) so core_follow clearly pulls -x.
        pilot_pos = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        flock.positions[0] = np.array([510.0, 350.0, 200.0], dtype=np.float32)
        flock.velocities[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        pilot = PilotTarget(
            position=pilot_pos.copy(),
            heading=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.shell_radius = 50.0
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            # With zero heading + inside shell: only core_follow fires
            # F_core = (-10, 0, 0) * 0.22 = (-2.2, 0, 0)
            # After 5 substeps accumulated, renormalized to v0
            vel = flock.velocities[0]
            assert vel[0] < 0, f"Core follow should pull -x, got vel={vel}"
            assert abs(vel[1]) < 0.5, f"No lateral component, got vel={vel}"
        finally:
            InfluencerMode.set_pilot(None)

    def test_pilot_shell_pull_activation(self):
        """P7.6: shell_pull fires only when d > shell_radius."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 2
        cfg.influencer_substeps = 5
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        pilot_pos = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        # Bird A: inside shell (d=40 < radius=50) — no shell_pull
        flock.positions[0] = np.array([540.0, 350.0, 200.0], dtype=np.float32)
        # Bird B: outside shell (d=70 > radius=50) — shell_pull fires
        flock.positions[1] = np.array([570.0, 350.0, 200.0], dtype=np.float32)
        # Orthogonal initial velocity so force direction clearly dominates
        flock.velocities[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        pilot = PilotTarget(
            position=pilot_pos.copy(),
            heading=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.shell_radius = 50.0
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            vel_a = flock.velocities[0]  # inside shell (d=40)
            vel_b = flock.velocities[1]  # outside shell (d=70)

            # Both should be pulled toward pilot (-x direction)
            assert vel_a[0] < 0, f"Bird A should move -x, got {vel_a}"
            assert vel_b[0] < 0, f"Bird B should move -x, got {vel_b}"

            # Bird B (farther + shell_pull) should have at least as strong -x
            assert abs(vel_b[0]) >= abs(vel_a[0]) - 1e-4, (
                f"Bird B shell-pull should match or exceed bird A: a={vel_a} b={vel_b}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    def test_flock_follows_within_2_shell_radius(self):
        """P7.6 acceptance: Flock CoM tracks pilot within 2·shell_radius."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 5
        cfg.influencer_rank_exponent = 2.0
        cfg.seed = 42

        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                dtype=np.float32,
            ),
            heading=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.shell_radius = 60.0
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            engine = SimulationEngine(cfg)
            for _ in range(60):
                engine.step(1.0 / 60.0)

            com = engine.flock.positions[engine.flock.active].mean(axis=0)
            dist = np.linalg.norm(com - pilot.position)
            assert dist < 2.0 * pilot.shell_radius, (
                f"Flock CoM {dist:.1f} should be within "
                f"2·shell_radius={2*pilot.shell_radius:.1f}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    def test_frozen_target_convergence(self):
        """P7.6: Static pilot → birds converge toward it over multiple calls."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 5
        cfg.influencer_rank_exponent = 2.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            )
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0

            # Run several calls, simulating integrate() movement
            for _ in range(10):
                _call_force(influencer_forces, flock, cfg)
                flock.positions += flock.velocities * 0.1

            dists = np.linalg.norm(
                flock.positions[flock.active] - pilot.position, axis=1
            )
            assert dists.mean() < 500.0, (
                f"Birds too far from pilot: {dists.mean():.1f}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    # ── Edge cases ──────────────────────────────────────────────

    def test_zero_active(self):
        """Empty flock produces no change."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        old_acc = flock.accelerations.copy()
        old_vel = flock.velocities.copy()
        _call_force(influencer_forces, flock, cfg)
        assert np.allclose(flock.accelerations, old_acc)
        assert np.allclose(flock.velocities, old_vel)

    def test_single_bird(self):
        """Single bird: rank=0 → influence=1.0, velocity steered."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 1
        cfg.influencer_substeps = 1

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        old_vel = flock.velocities.copy()

        np.random.seed(42)
        _call_force(influencer_forces, flock, cfg)

        assert not np.allclose(flock.velocities[0], old_vel[0])
        assert np.linalg.norm(flock.velocities[0]) == pytest.approx(cfg.v0)

    def test_no_neighbour_queries(self):
        """Influencer mode never queries the spatial index."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.random.randn(*flock.velocities.shape).astype(np.float32)
        old_vels = flock.velocities.copy()

        class SpyIndex:
            def __init__(self):
                self.ready = True
            def query_knn(self, *a, **kw):
                raise RuntimeError("Should not be called")
            def query_radius(self, *a, **kw):
                raise RuntimeError("Should not be called")
            def rebuild(self, *a, **kw):
                pass

        flock._index = SpyIndex()
        _call_force(influencer_forces, flock, cfg)

        vel_diffs = np.linalg.norm(
            flock.velocities[flock.active] - old_vels[flock.active], axis=1
        )
        assert np.all(vel_diffs > 1e-6)

    def test_substeps_multiply_turn(self):
        """More substeps → proportionally larger turn angle."""
        cfg1 = SimConfig()
        cfg1.seed = 42  # D6: pin seed so both flocks share initial geometry
        cfg1.mode = "influencer"
        cfg1.num_boids = 30
        cfg1.influencer_substeps = 1

        cfg2 = SimConfig()
        cfg2.seed = 42  # D6: identical geometry to flock1
        cfg2.mode = "influencer"
        cfg2.num_boids = 30
        cfg2.influencer_substeps = 3

        flock1 = PhysicsFlock(cfg1)
        flock1.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_dirs1 = (
            flock1.velocities[flock1.active]
            / np.linalg.norm(flock1.velocities[flock1.active], axis=1, keepdims=True)
        )
        _call_force(influencer_forces, flock1, cfg1)
        new_dirs1 = (
            flock1.velocities[flock1.active]
            / (np.linalg.norm(flock1.velocities[flock1.active], axis=1, keepdims=True) + 1e-10)
        )
        ang1 = np.arccos(np.clip(np.sum(old_dirs1 * new_dirs1, axis=1), -1.0, 1.0)).mean()

        flock2 = PhysicsFlock(cfg2)
        flock2.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_dirs2 = (
            flock2.velocities[flock2.active]
            / np.linalg.norm(flock2.velocities[flock2.active], axis=1, keepdims=True)
        )
        _call_force(influencer_forces, flock2, cfg2)
        new_dirs2 = (
            flock2.velocities[flock2.active]
            / (np.linalg.norm(flock2.velocities[flock2.active], axis=1, keepdims=True) + 1e-10)
        )
        ang2 = np.arccos(np.clip(np.sum(old_dirs2 * new_dirs2, axis=1), -1.0, 1.0)).mean()

        ratio = ang2 / max(ang1, 1e-10)
        # Not 3.0 despite 3 substeps: the direction blend saturates toward
        # t̂ per substep, and D11 move-then-steer shifts positions between
        # substeps. The point is super-unity accumulation, not linearity.
        assert 1.3 < ratio < 6.0, f"Substep scaling: ang2/ang1 = {ratio:.2f}"

    def test_inactive_birds_unchanged(self):
        """Inactive birds unchanged while active ones are steered."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 2

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        flock.active[10:20] = False
        old_vel_inactive = flock.velocities[~flock.active].copy()
        old_vel_active = flock.velocities[flock.active].copy()

        _call_force(influencer_forces, flock, cfg)

        assert np.allclose(flock.velocities[~flock.active], old_vel_inactive)
        vel_diff_active = np.linalg.norm(
            flock.velocities[flock.active] - old_vel_active, axis=1
        )
        assert np.any(vel_diff_active > 1e-6)

    def test_substeps_zero(self):
        """substeps=0 → no velocity change, no crash, diagnostics still run."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_vel = flock.velocities.copy()
        _call_force(influencer_forces, flock, cfg)

        assert np.allclose(flock.velocities, old_vel)
        assert hasattr(cfg, '_target_dist_min')

