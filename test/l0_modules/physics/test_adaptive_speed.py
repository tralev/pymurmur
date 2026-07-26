"""Unit tests for physics.adaptive_speed.adaptive_speed_bonus — the
pure neighbor-deficit speed law shared by NeighborAdaptiveSpeed,
verified for exact numeric parity with angle.py's inline formula.
"""

from __future__ import annotations

import numpy as np

from pymurmur.physics.adaptive_speed import adaptive_speed_bonus


def _angle_py_reference(deficit, mode, deficit_cap, linear_scale=5.0):
    """Direct re-implementation of angle.py's scalar formula, used only
    to prove numeric parity — not imported from angle.py itself."""
    if deficit > 0:
        if mode == "quadratic":
            return min(deficit_cap, deficit * deficit)
        if mode == "softened":
            return min(deficit_cap, deficit * deficit / 2.0)
        return deficit * linear_scale
    return 0.0


class TestAdaptiveSpeedBonus:
    def test_matches_angle_py_formula(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            deficit = rng.uniform(-5, 10)
            mode = rng.choice(["linear", "quadratic", "softened"])
            deficit_cap = rng.uniform(1, 50)
            expected = _angle_py_reference(deficit, mode, deficit_cap)
            actual = adaptive_speed_bonus(
                np.array([deficit]), mode, deficit_cap,
            )[0]
            assert actual == expected or abs(actual - expected) < 1e-5

    def test_negative_deficit_is_zero_bonus(self):
        for mode in ("linear", "quadratic", "softened"):
            bonus = adaptive_speed_bonus(np.array([-3.0]), mode, 100.0)
            assert bonus[0] == 0.0

    def test_linear_uncapped(self):
        bonus = adaptive_speed_bonus(np.array([100.0]), "linear", deficit_cap=1.0)
        assert bonus[0] == 500.0  # 100 * 5.0, cap does not apply to linear

    def test_quadratic_capped(self):
        bonus = adaptive_speed_bonus(np.array([100.0]), "quadratic", deficit_cap=50.0)
        assert bonus[0] == 50.0

    def test_softened_is_half_of_quadratic(self):
        quad = adaptive_speed_bonus(np.array([3.0]), "quadratic", deficit_cap=1000.0)
        soft = adaptive_speed_bonus(np.array([3.0]), "softened", deficit_cap=1000.0)
        assert soft[0] == quad[0] / 2.0

    def test_vectorized_over_array(self):
        deficits = np.array([-1.0, 0.0, 2.0, 5.0])
        bonus = adaptive_speed_bonus(deficits, "linear", deficit_cap=1000.0)
        np.testing.assert_allclose(bonus, [0.0, 0.0, 10.0, 25.0])
