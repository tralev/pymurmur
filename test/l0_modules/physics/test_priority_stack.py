"""Unit tests for physics.priority_stack.allocate_priority_budget —
the binary-cutoff priority-budget allocator backing the
priority_stack_enabled engine feature.
"""

from __future__ import annotations

import numpy as np

from pymurmur.physics.priority_stack import allocate_priority_budget


class TestAllocatePriorityBudget:
    def test_tier1_saturation_zeroes_tier2_and_tier3(self):
        """When tier1 alone consumes the full budget, tier2/tier3 are
        dropped entirely (binary cutoff), not proportionally reduced."""
        budget = 5.0
        tier1 = np.array([[10.0, 0.0, 0.0]])
        tier2 = np.array([[3.0, 0.0, 0.0]])
        tier3 = np.array([[3.0, 0.0, 0.0]])
        out = allocate_priority_budget(tier1, tier2, tier3, budget)
        np.testing.assert_allclose(out, [[5.0, 0.0, 0.0]], atol=1e-4)

    def test_tier1_and_tier2_saturation_zeroes_tier3(self):
        budget = 5.0
        tier1 = np.array([[2.0, 0.0, 0.0]])
        tier2 = np.array([[10.0, 0.0, 0.0]])
        tier3 = np.array([[3.0, 0.0, 0.0]])
        out = allocate_priority_budget(tier1, tier2, tier3, budget)
        np.testing.assert_allclose(out, [[5.0, 0.0, 0.0]], atol=1e-4)

    def test_no_saturation_is_plain_sum(self):
        budget = 5.0
        tier1 = np.array([[1.0, 0.0, 0.0]])
        tier2 = np.array([[1.0, 0.0, 0.0]])
        tier3 = np.array([[1.0, 0.0, 0.0]])
        out = allocate_priority_budget(tier1, tier2, tier3, budget)
        np.testing.assert_allclose(out, [[3.0, 0.0, 0.0]], atol=1e-4)

    def test_all_zero_tiers_is_noop(self):
        out = allocate_priority_budget(
            np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), 5.0,
        )
        np.testing.assert_allclose(out, 0.0)

    def test_magnitude_never_exceeds_budget(self):
        """Adversarial random inputs at the max_force-ish scale."""
        rng = np.random.default_rng(0)
        n = 500
        budget = 5.0
        t1 = rng.uniform(-50, 50, size=(n, 3))
        t2 = rng.uniform(-50, 50, size=(n, 3))
        t3 = rng.uniform(-50, 50, size=(n, 3))
        out = allocate_priority_budget(t1, t2, t3, budget)
        mags = np.linalg.norm(out, axis=1)
        assert np.all(mags <= budget + 1e-4)

    def test_scale_agnostic_large_budget(self):
        """Confirms the function itself is scale-agnostic — proves
        marl's v_cap-scale (tens of units) budgets work correctly too;
        any scale mismatch bug belongs to the caller's budget
        resolution, not this function."""
        rng = np.random.default_rng(1)
        n = 500
        budget = 40.0
        t1 = rng.uniform(-100, 100, size=(n, 3))
        t2 = rng.uniform(-100, 100, size=(n, 3))
        t3 = rng.uniform(-100, 100, size=(n, 3))
        out = allocate_priority_budget(t1, t2, t3, budget)
        mags = np.linalg.norm(out, axis=1)
        assert np.all(mags <= budget + 1e-4)

    def test_per_boid_independence(self):
        """Saturation for one boid must not affect another boid's row."""
        budget = 5.0
        tier1 = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        tier2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        tier3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        out = allocate_priority_budget(tier1, tier2, tier3, budget)
        np.testing.assert_allclose(out[0], [5.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(out[1], [2.0, 0.0, 0.0], atol=1e-4)

    def test_output_dtype_float32(self):
        out = allocate_priority_budget(
            np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), 1.0,
        )
        assert out.dtype == np.float32
