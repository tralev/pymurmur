"""Unit tests for physics.forces.kernels — the separation/cohesion
kernel registry (pure functions operating on precomputed diffs/dists).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.physics.forces import force_kernels as kernels


def _single_bird_scenario(dist, offset_axis=1):
    """One neighbour at distance `dist` along +x, offset slightly on
    another axis to avoid symmetry issues. Returns (diffs, dists, close)
    each shaped (1, 3)/(1,)/(1,) — a single "batch" of one neighbour."""
    diffs = np.zeros((1, 3), dtype=np.float32)
    diffs[0, 0] = dist
    dists = np.linalg.norm(diffs, axis=1)
    close = dists > 1e-6
    return diffs, dists, close


class TestKernelSum:
    def test_magnitude_is_inverse_square(self):
        diffs_near, dists_near, close_near = _single_bird_scenario(2.0)
        diffs_far, dists_far, close_far = _single_bird_scenario(4.0)
        near = kernels.kernel_sum(diffs_near, dists_near, close_near)
        far = kernels.kernel_sum(diffs_far, dists_far, close_far)
        # magnitude ~ 1/d^2 -> doubling distance quarters the magnitude
        ratio = np.linalg.norm(near) / np.linalg.norm(far)
        assert 3.5 < ratio < 4.5

    def test_points_away_from_neighbor(self):
        diffs, dists, close = _single_bird_scenario(5.0)
        out = kernels.kernel_sum(diffs, dists, close)
        assert out[0] < 0  # neighbour at +x -> push toward -x


class TestKernelMean:
    def test_density_invariant(self):
        diffs2 = np.array([[5.0, 0, 0], [5.0, 0.3, 0]], dtype=np.float32)
        dists2 = np.linalg.norm(diffs2, axis=1)
        close2 = dists2 > 1e-6
        diffs8 = np.tile(diffs2, (4, 1))
        dists8 = np.linalg.norm(diffs8, axis=1)
        close8 = dists8 > 1e-6
        mag2 = np.linalg.norm(kernels.kernel_mean(diffs2, dists2, close2))
        mag8 = np.linalg.norm(kernels.kernel_mean(diffs8, dists8, close8))
        assert mag2 == pytest.approx(mag8, rel=0.05)


class TestKernelUnit:
    def test_distance_independent_magnitude(self):
        diffs_near, dists_near, close_near = _single_bird_scenario(2.0)
        diffs_far, dists_far, close_far = _single_bird_scenario(50.0)
        near = kernels.kernel_unit(diffs_near, dists_near, close_near)
        far = kernels.kernel_unit(diffs_far, dists_far, close_far)
        assert np.linalg.norm(near) == pytest.approx(np.linalg.norm(far), rel=0.05)


class TestKernelExpLinearRampAsymptotic:
    def test_exp_decreases_with_distance(self):
        near = kernels.kernel_exp(*_single_bird_scenario(5.0), radius=20.0)
        far = kernels.kernel_exp(*_single_bird_scenario(15.0), radius=20.0)
        assert np.linalg.norm(near) > np.linalg.norm(far)

    def test_linear_ramp_decreases_with_distance(self):
        near = kernels.kernel_linear_ramp(*_single_bird_scenario(5.0), radius=20.0)
        far = kernels.kernel_linear_ramp(*_single_bird_scenario(15.0), radius=20.0)
        assert np.linalg.norm(near) > np.linalg.norm(far)

    def test_linear_ramp_zero_beyond_radius(self):
        beyond = kernels.kernel_linear_ramp(*_single_bird_scenario(25.0), radius=20.0)
        np.testing.assert_allclose(beyond, 0.0)

    def test_asymptotic_decreases_with_distance(self):
        near = kernels.kernel_asymptotic(*_single_bird_scenario(5.0), radius=20.0)
        far = kernels.kernel_asymptotic(*_single_bird_scenario(15.0), radius=20.0)
        assert np.linalg.norm(near) > np.linalg.norm(far)

    def test_asymptotic_zero_beyond_radius(self):
        beyond = kernels.kernel_asymptotic(*_single_bird_scenario(25.0), radius=20.0)
        np.testing.assert_allclose(beyond, 0.0)


class TestKernelVelocityWeighted:
    def test_approaching_neighbor_contributes(self):
        diffs, dists, close = _single_bird_scenario(10.0)
        closing_speed = np.array([3.0], dtype=np.float32)
        out = kernels.kernel_velocity_weighted(diffs, dists, close, closing_speed)
        assert np.linalg.norm(out) > 0

    def test_receding_neighbor_zeroed(self):
        diffs, dists, close = _single_bird_scenario(10.0)
        closing_speed = np.array([-3.0], dtype=np.float32)
        out = kernels.kernel_velocity_weighted(diffs, dists, close, closing_speed)
        np.testing.assert_allclose(out, 0.0)

    def test_faster_closing_speed_stronger_push(self):
        diffs, dists, close = _single_bird_scenario(10.0)
        slow = kernels.kernel_velocity_weighted(diffs, dists, close, np.array([1.0], dtype=np.float32))
        fast = kernels.kernel_velocity_weighted(diffs, dists, close, np.array([5.0], dtype=np.float32))
        assert np.linalg.norm(fast) > np.linalg.norm(slow)


class TestKernelCosineZone:
    def test_neighbor_ahead_weighted_more_than_behind(self):
        heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ahead_diffs, ahead_dists, ahead_close = _single_bird_scenario(10.0)  # +x = ahead
        behind_diffs = -ahead_diffs
        behind_dists = ahead_dists
        behind_close = ahead_close

        ahead = kernels.kernel_cosine_zone(ahead_diffs, ahead_dists, ahead_close, heading)
        behind = kernels.kernel_cosine_zone(behind_diffs, behind_dists, behind_close, heading)
        assert np.linalg.norm(ahead) > np.linalg.norm(behind)

    def test_directly_behind_is_near_zero(self):
        heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        diffs, dists, close = _single_bird_scenario(-10.0)  # neighbour behind
        out = kernels.kernel_cosine_zone(diffs, dists, close, heading)
        assert np.linalg.norm(out) < 1e-5


class TestKernelUnweighted:
    def test_matches_plain_mean(self):
        diffs = np.array([[4.0, 0, 0], [0, 4.0, 0]], dtype=np.float32)
        out = kernels.kernel_unweighted(diffs)
        np.testing.assert_allclose(out, [2.0, 2.0, 0.0])


class TestKernelInverseDistance:
    def test_nearer_neighbor_dominates(self):
        # one very close, one far, both on opposite sides
        diffs = np.array([[1.0, 0, 0], [-100.0, 0, 0]], dtype=np.float32)
        dists = np.linalg.norm(diffs, axis=1)
        close = dists > 1e-6
        out = kernels.kernel_inverse_distance(diffs, dists, close)
        # near neighbour (weight 1/1=1) dominates far one (weight 1/100=0.01)
        assert out[0] > 0

    def test_symmetric_neighbors_cancel(self):
        diffs = np.array([[5.0, 0, 0], [-5.0, 0, 0]], dtype=np.float32)
        dists = np.linalg.norm(diffs, axis=1)
        close = dists > 1e-6
        out = kernels.kernel_inverse_distance(diffs, dists, close)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)


class TestValidKernelSets:
    def test_separation_kernels_needing_radius(self):
        assert kernels.SEPARATION_KERNELS_NEEDING_RADIUS == {
            "exp", "linear_ramp", "asymptotic",
        }

    def test_valid_separation_kernels_matches_registry(self):
        expected = {
            "sum", "mean", "unit", "exp", "linear_ramp",
            "asymptotic", "velocity_weighted", "cosine_zone",
        }
        assert kernels.VALID_SEPARATION_KERNELS == expected

    def test_valid_cohesion_kernels(self):
        assert kernels.VALID_COHESION_KERNELS == {"unweighted", "inverse_distance"}
