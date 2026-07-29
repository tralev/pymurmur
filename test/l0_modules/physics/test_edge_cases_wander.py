"""Targeted tests for uncovered branches in P3 phase modules —
wander.py edge cases.

Closes coverage gaps identified in the coverage report:
- wander.py L121: flock.center is None → positions.mean fallback
- wander.py L156: _apply_pull with all-zero distances
- wander.py L74:  wander_heading zero-diff fallback

Split out of test_edge_cases.py (file-size split) — predator.py and
flock.py tests stay in the original; field.py tests moved to
test_edge_cases_field.py.
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.wander import Wander

# ══════════════════════════════════════════════════════════════════════
# wander.py — flock.center fallback (L121)
# ══════════════════════════════════════════════════════════════════════

class TestWanderCenterFallback:
    """Test wander when flock.center is None (EMA not yet computed)."""

    def test_wander_uses_positions_mean(self):
        """When flock.center is None, wander falls back to np.mean(positions[active])."""
        # Create a minimal mock flock
        class MockFlock:
            def __init__(self):
                self.positions = np.array(
                    [[100, 200, 300], [400, 500, 600], [700, 800, 900]],
                    dtype=np.float32,
                )
                self.velocities = np.zeros((3, 3), dtype=np.float32)
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.ones(3, dtype=bool)
                self.N_capacity = 3
                self.center = None  # <-- not yet set
                self.wander_center = None
                self.wander_heading = None

        flock = MockFlock()
        cfg = SimConfig()
        ctx = StepContext(frame=0, dt=1 / 60, rng=np.random.default_rng(42),
                          center=np.array([500, 350, 200], dtype=np.float32), config=cfg)

        wander = Wander()
        wander.apply(flock, ctx)

        # Wander centre should have been computed (via positions.mean fallback)
        assert flock.wander_center is not None
        assert flock.wander_heading is not None
        assert flock.wander_center.shape == (3,)
        assert flock.wander_heading.shape == (3,)

    def test_wander_no_active_fallback(self):
        """When no birds are active and center is None, uses domain centre."""
        class MockFlock:
            def __init__(self):
                self.positions = np.zeros((3, 3), dtype=np.float32)
                self.velocities = np.zeros((3, 3), dtype=np.float32)
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.zeros(3, dtype=bool)  # all inactive
                self.N_capacity = 3
                self.center = None
                self.wander_center = None
                self.wander_heading = None

        flock = MockFlock()
        cfg = SimConfig()
        ctx = StepContext(frame=0, dt=1 / 60, rng=np.random.default_rng(42),
                          center=None, config=cfg)

        wander = Wander()
        wander.apply(flock, ctx)

        # Should not crash; wander_center computed from domain centre fallback
        assert flock.wander_center is not None


# ══════════════════════════════════════════════════════════════════════
# wander.py — _apply_pull zero-distance early return (L156)
# ══════════════════════════════════════════════════════════════════════

class TestWanderApplyPullEdge:
    """Test _apply_pull when all birds are exactly at the wander centre."""

    def test_pull_all_at_center_returns_early(self):
        """When all birds are at the target, dists=0 → mask.all()=False → return."""
        class MockFlock:
            def __init__(self):
                self.positions = np.array(
                    [[500, 350, 200], [500, 350, 200], [500, 350, 200]],
                    dtype=np.float32,
                )
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.ones(3, dtype=bool)
                self.N_capacity = 3

        flock = MockFlock()
        target = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        acc_before = flock.accelerations.copy()

        Wander._apply_pull(flock, target, flock.active)

        # Accelerations should be unchanged (early return triggered)
        np.testing.assert_array_equal(flock.accelerations, acc_before)

    def test_pull_partial_masks_correctly(self):
        """Birds at different distances — only distant ones get pull."""
        class MockFlock:
            def __init__(self):
                self.positions = np.array(
                    [[500, 350, 200],    # at target
                     [550, 350, 200],    # 50 units away
                     [500, 350, 200]],   # at target
                    dtype=np.float32,
                )
                self.accelerations = np.zeros((3, 3), dtype=np.float32)
                self.active = np.ones(3, dtype=bool)
                self.N_capacity = 3

        flock = MockFlock()
        target = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        Wander._apply_pull(flock, target, flock.active)

        # Bird 1 should have acceleration applied (pulled toward target)
        assert np.abs(flock.accelerations[1]).sum() > 0
        # Birds 0 and 2 should be unchanged (at target)
        assert np.abs(flock.accelerations[0]).sum() == 0
        assert np.abs(flock.accelerations[2]).sum() == 0


# ══════════════════════════════════════════════════════════════════════
# wander.py — wander_heading zero-diff fallback (L74)
# ══════════════════════════════════════════════════════════════════════

class TestWanderHeadingZeroDiff:
    """Test wander_heading when diff norm ≤ 1e-10 (L74 fallback)."""

    def test_heading_zero_diff_returns_default(self, monkeypatch):
        """When bounded_unit_path returns same value for t and t+0.75,
        wander_heading falls back to (1,0,0)."""
        from pymurmur.physics.extensions import wander as wander_mod

        same_path = np.array([0.5, 0.3, 0.1], dtype=np.float32)

        def mock_path(t):
            return same_path.copy()

        monkeypatch.setattr(wander_mod, "bounded_unit_path", mock_path)
        # Need to re-import wander_heading to pick up patched bounded_unit_path
        heading = wander_mod.wander_heading(0.0)
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(heading, expected, atol=1e-6)
