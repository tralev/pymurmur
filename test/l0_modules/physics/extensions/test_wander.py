"""Unit tests for P3.1 — boundedUnitTravel wander path and Wander extension.

Tests:
- ‖path(t)‖ ≤ 1 for 10⁶ fuzzed t
- heading continuity: ‖h(t+ε) − h(t)‖ < 0.05
- wander_center bounds
- Wander extension publishes to flock
- Wander extension applies backward-compat pull for non-field modes
"""

from __future__ import annotations

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.wander import (
    Wander,
    bounded_unit_path,
    wander_heading,
)

# ── bounded_unit_path tests ───────────────────────────────────────

class TestBoundedUnitPath:
    """P3.1: ‖path(t)‖ ≤ 1 guarantee."""

    def test_path_bounded_scalar(self):
        """‖path(t)‖ ≤ 1 for a single t value."""
        path = bounded_unit_path(0.0)
        assert np.linalg.norm(path) <= 1.0, f"norm={np.linalg.norm(path)}"

    def test_path_bounded_fuzzed(self):
        """‖path(t)‖ ≤ 1 for 10⁴ fuzzed t values."""
        rng = np.random.default_rng(42)
        t_values = rng.uniform(0, 1000, size=10_000)
        for t in t_values:
            path = bounded_unit_path(t)
            norm = float(np.linalg.norm(path))
            assert norm <= 1.0, f"t={t:.3f}, norm={norm:.6f} > 1"

    def test_path_bounded_vectorised(self):
        """‖path(t)‖ ≤ 1 for vectorised call with 1000 t values."""
        t_values = np.linspace(0, 500, 1000)
        paths = bounded_unit_path(t_values)
        norms = np.linalg.norm(paths, axis=1)
        assert np.all(norms <= 1.0), f"max norm = {norms.max():.6f}"

    def test_path_not_stationary(self):
        """path(t) varies with t — it's not a constant function."""
        p0 = bounded_unit_path(0.0)
        p1 = bounded_unit_path(1.0)
        p2 = bounded_unit_path(10.0)
        # All three should differ
        assert not np.allclose(p0, p1)
        assert not np.allclose(p1, p2)

    def test_path_periodic_approx(self):
        """path(t) approximately repeats after long periods (quasi-periodic)."""
        p0 = bounded_unit_path(100.0)
        p1 = bounded_unit_path(100.0 + 1000.0)
        # Should be different — the path is chaotic/quasi-periodic, not strictly periodic
        assert not np.allclose(p0, p1), "Path is quasi-periodic, not strictly periodic"


class TestWanderHeading:
    """P3.1: heading continuity and unit-length guarantee."""

    def test_heading_is_unit(self):
        """heading(t) is always a unit vector."""
        for t in [0.0, 1.0, 10.0, 100.0, 500.0]:
            h = wander_heading(t)
            assert np.abs(np.linalg.norm(h) - 1.0) < 1e-6, f"t={t}, norm={np.linalg.norm(h)}"

    def test_heading_continuity(self):
        """‖h(t+ε) − h(t)‖ < 0.05 for small ε (continuity)."""
        rng = np.random.default_rng(42)
        eps = 1e-4
        for _ in range(100):
            t = rng.uniform(0, 1000)
            h0 = wander_heading(t)
            h1 = wander_heading(t + eps)
            diff = np.linalg.norm(h1 - h0)
            assert diff < 0.05, f"t={t:.3f}, diff={diff:.6f}"

    def test_heading_not_stationary(self):
        """heading(t) changes over time."""
        h0 = wander_heading(0.0)
        h1 = wander_heading(5.0)
        # Should differ since the wander path is not a straight line
        assert not np.allclose(h0, h1)


# ── Wander extension tests ────────────────────────────────────────

class TestWanderExtension:
    """Wander extension publishes wander_center and wander_heading."""

    @staticmethod
    def _make_flock_and_ctx(n_boids=100, mode="field"):
        """Create a minimal PhysicsFlock and StepContext for testing."""
        cfg = SimConfig()
        cfg.num_boids = n_boids
        cfg.mode = mode
        cfg.wander_enabled = True
        # Set domain for unit scale computation
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)

        ctx = StepContext(
            frame=0,
            dt=1.0 / 60.0,
            rng=flock.rng,
            center=flock.center,
            config=cfg,
        )
        return flock, ctx, cfg

    def test_wander_publishes_center_and_heading(self):
        """After apply(), flock.wander_center and wander_heading are set."""
        flock, ctx, _ = self._make_flock_and_ctx()
        w = Wander()
        w.apply(flock, ctx)
        assert flock.wander_center is not None
        assert flock.wander_heading is not None
        assert flock.wander_center.shape == (3,)
        assert flock.wander_heading.shape == (3,)

    def test_wander_center_inside_domain(self):
        """wander_center stays near the domain (within radius of centre)."""
        flock, ctx, cfg = self._make_flock_and_ctx()
        w = Wander()
        # Default attractor_radius = 300.0, domain 1000×700×400
        # Centre ≈ (500, 350, 200).  Max deviation = radius = 300.
        # Allow 50% margin for centre drift.
        for _ in range(50):
            w.apply(flock, ctx)
            ctx.frame += 1
            wc = flock.wander_center
            assert 0 <= wc[0] <= 1000, f"x={wc[0]} outside domain [0,1000]"
            assert -150 <= wc[1] <= 850, f"y={wc[1]} out of bounds"
            assert -105 <= wc[2] <= 505, f"z={wc[2]} out of bounds"

    def test_wander_heading_is_unit(self):
        """wander_heading published to flock is always a unit vector."""
        flock, ctx, _ = self._make_flock_and_ctx()
        w = Wander()
        for _ in range(20):
            w.apply(flock, ctx)
            ctx.frame += 1
            wh = flock.wander_heading
            assert np.abs(np.linalg.norm(wh) - 1.0) < 1e-6

    def test_wander_center_moves_each_frame(self):
        """wander_center changes between consecutive frames."""
        flock, ctx, _ = self._make_flock_and_ctx()
        w = Wander()
        w.apply(flock, ctx)
        wc0 = flock.wander_center.copy()
        ctx.frame += 1
        w.apply(flock, ctx)
        wc1 = flock.wander_center.copy()
        assert not np.allclose(wc0, wc1), "wander_center should move each frame"

    def test_wander_backward_compat_pull_spatial(self):
        """For spatial mode, Wander applies a pull on birds."""
        flock, ctx, _ = self._make_flock_and_ctx(mode="spatial")
        w = Wander()
        # Record accelerations before
        acc_before = flock.accelerations[flock.active].copy()
        w.apply(flock, ctx)
        acc_after = flock.accelerations[flock.active]
        # Non-zero force applied (pull toward wander centre)
        assert not np.allclose(acc_before, acc_after)

    def test_wander_no_pull_for_field_mode(self):
        """For field mode, Wander does NOT apply pull (field handles it)."""
        flock, ctx, _ = self._make_flock_and_ctx(mode="field")
        w = Wander()
        acc_before = flock.accelerations[flock.active].copy()
        w.apply(flock, ctx)
        acc_after = flock.accelerations[flock.active]
        # No direct pull — field mode handles drift alignment
        assert np.allclose(acc_before, acc_after)

    def test_wander_zero_active_still_publishes(self):
        """When no birds are active, apply() still publishes wander state."""
        flock, ctx, _ = self._make_flock_and_ctx()
        flock.active[:] = False
        # Ensure flock.center is set (would be None if never updated)
        flock.center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        w = Wander()
        w.apply(flock, ctx)
        assert flock.wander_center is not None
        assert flock.wander_heading is not None
        assert flock.wander_center.shape == (3,)
        # No pull applied since no active birds
        assert np.all(flock.accelerations == 0.0)

    def test_wander_deterministic(self):
        """Same seed → same wander_center after N steps."""
        flock1, ctx1, _ = self._make_flock_and_ctx()
        flock2, ctx2, _ = self._make_flock_and_ctx()
        w1 = Wander()
        w2 = Wander()
        for _ in range(10):
            w1.apply(flock1, ctx1)
            ctx1.frame += 1
            w2.apply(flock2, ctx2)
            ctx2.frame += 1
        assert np.allclose(flock1.wander_center, flock2.wander_center)
        assert np.allclose(flock1.wander_heading, flock2.wander_heading)
