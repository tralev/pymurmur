"""Density scaling tests — Phase 9.6 N sweep, power-law fit, boundary comparison.
"""

import numpy as np
import pytest

from pymurmur.analysis.density_scaling import (
    DensityScalingResult,
    sweep_density_scaling,
    _fit_power_laws,
    save_results,
    load_results,
)


class TestSweep:
    """Population sweep measuring density (local_spacing)."""

    @pytest.mark.slow
    def test_sweep_returns_correct_shapes(self):
        """Sweep produces results of expected dimensions."""
        result = sweep_density_scaling(
            n_values=[50, 100, 200],
            steps=80,
            seed=42,
        )
        assert len(result.n_values) == 3
        assert len(result.spacings_toroidal) == 3
        assert len(result.spacings_open) == 3
        assert not np.all(np.isnan(result.spacings_toroidal))

    @pytest.mark.slow
    def test_larger_n_produces_finite_spacing(self):
        """Both N values produce finite spacing (no NaN)."""
        result = sweep_density_scaling(
            n_values=[50, 200],
            steps=100,
            seed=99,
        )
        idx_small = 0  # N=50
        idx_large = 1  # N=200
        assert result.spacings_toroidal[idx_large] > 0
        assert result.spacings_toroidal[idx_small] > 0
        assert not np.isnan(result.spacings_toroidal[idx_small])
        assert not np.isnan(result.spacings_toroidal[idx_large])


class TestPowerLawFit:
    """Log-log regression for power-law exponent."""

    def test_power_law_fit_on_synthetic_data(self):
        """Known power-law -> beta close to true value."""
        result = DensityScalingResult(
            n_values=np.array([50, 100, 200, 400, 800], dtype=np.float64),
            spacings_toroidal=np.array([10.0, 7.07, 5.0, 3.54, 2.5], dtype=np.float64),
            # spacing = 70.7 / sqrt(N)  ->  log(spacing) = -0.5*log(N) + C
            spacings_open=np.full(5, np.nan),
        )
        _fit_power_laws(result)
        assert result.beta_toroidal == pytest.approx(-0.5, abs=0.01)
        assert result.r_sq_toroidal > 0.99

    def test_power_law_requires_min_points(self):
        """Less than 3 valid points -> beta stays NaN."""
        result = DensityScalingResult(
            n_values=np.array([50, 100], dtype=np.float64),
            spacings_toroidal=np.array([10.0, 7.0], dtype=np.float64),
            spacings_open=np.full(2, np.nan),
        )
        _fit_power_laws(result)
        assert np.isnan(result.beta_toroidal)
        assert np.isnan(result.beta_open)

    def test_nan_points_excluded_from_fit(self):
        """NaN values are excluded; remaining 3 points used for fit."""
        result = DensityScalingResult(
            n_values=np.array([50, 100, 200, 400], dtype=np.float64),
            spacings_toroidal=np.array([10.0, np.nan, 5.0, 2.5], dtype=np.float64),
            spacings_open=np.full(4, np.nan),
        )
        _fit_power_laws(result)
        assert not np.isnan(result.beta_toroidal)
        assert -2.0 < result.beta_toroidal < 0.0

    @pytest.mark.slow
    def test_real_sweep_produces_negative_beta(self):
        """Real sweep: more birds in scaled domain -> negative beta (density increases)."""
        result = sweep_density_scaling(
            n_values=[50, 100, 200],
            steps=100,
            seed=77,
        )
        assert result.beta_toroidal < 0, (
            f"Expected negative beta (denser with more birds), got {result.beta_toroidal:.3f}"
        )
        assert result.r_sq_toroidal > 0.5, (
            f"Poor fit: R^2={result.r_sq_toroidal:.3f}"
        )


class TestSaveLoad:
    """Round-trip save/load."""

    def test_round_trip(self, tmp_path):
        """Save then load produces identical data."""
        result = DensityScalingResult(
            n_values=np.array([50, 100, 200], dtype=np.float64),
            spacings_toroidal=np.array([10.0, 7.07, 5.0], dtype=np.float64),
            spacings_open=np.array([12.0, 9.0, 6.5], dtype=np.float64),
            beta_toroidal=-0.5,
            beta_open=-0.4,
            r_sq_toroidal=0.99,
            r_sq_open=0.95,
        )
        path = str(tmp_path / "density.npz")
        save_results(result, path)
        loaded = load_results(path)

        np.testing.assert_array_equal(loaded.n_values, result.n_values)
        np.testing.assert_array_almost_equal(loaded.spacings_toroidal, result.spacings_toroidal)
        np.testing.assert_array_almost_equal(loaded.spacings_open, result.spacings_open)
        assert loaded.beta_toroidal == pytest.approx(result.beta_toroidal)
        assert loaded.beta_open == pytest.approx(result.beta_open)


class TestBoundaryComparison:
    """Toroidal vs open boundary effects."""

    @pytest.mark.slow
    def test_both_boundaries_produce_measurements(self):
        """Both toroidal and open boundaries produce valid spacing data."""
        result = sweep_density_scaling(
            n_values=[50, 100],
            steps=100,
            seed=42,
        )
        assert not np.all(np.isnan(result.spacings_toroidal))
        assert not np.all(np.isnan(result.spacings_open))
        # Both should have finite spacing
        for s in result.spacings_toroidal:
            if not np.isnan(s):
                assert s > 0
        for s in result.spacings_open:
            if not np.isnan(s):
                assert s > 0
