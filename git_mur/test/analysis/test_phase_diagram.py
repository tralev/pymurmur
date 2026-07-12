"""Phase diagram tests — Phase 9.4 Vicsek eta x D sweep.
"""

import numpy as np
import pytest

from pymurmur.analysis.phase_diagram import (
    PhaseDiagramResult,
    sweep_vicsek_phase,
    _find_phase_boundary,
    save_results,
    load_results,
)


class TestSweep:
    """Grid sweep over (eta, D) space — these run real Vicsek simulations."""

    @pytest.mark.slow
    def test_sweep_returns_correct_shapes(self):
        """Sweep produces grids of expected dimensions."""
        result = sweep_vicsek_phase(
            eta_range=(0.0, 1.0),
            d_range=(0.0, 2.0),
            n_eta=5,
            n_d=3,
            n_boids=30,
            steps=80,
            seed=42,
        )
        assert result.eta_grid.shape == (5,)
        assert result.d_grid.shape == (3,)
        assert result.alpha_grid.shape == (3, 5)
        assert not np.all(np.isnan(result.alpha_grid))

    @pytest.mark.slow
    def test_high_eta_low_d_gives_high_alpha(self):
        """High coupling + low noise -> ordered (alpha close to 1)."""
        result = sweep_vicsek_phase(
            eta_range=(0.7, 1.0),
            d_range=(0.0, 0.5),
            n_eta=4,
            n_d=2,
            n_boids=40,
            steps=120,
            seed=99,
        )
        # With high eta and low D, alpha should generally be > 0.5
        high_alpha = result.alpha_grid > 0.5
        assert high_alpha.any(), "Expected some ordered points"

    @pytest.mark.slow
    def test_low_eta_high_d_gives_low_alpha(self):
        """Low coupling + high noise -> disordered (alpha close to 0)."""
        result = sweep_vicsek_phase(
            eta_range=(0.0, 0.3),
            d_range=(2.0, 4.0),
            n_eta=3,
            n_d=2,
            n_boids=30,
            steps=100,
            seed=123,
        )
        # With low eta and high D, alpha should generally be < 0.5
        low_alpha = result.alpha_grid < 0.5
        assert low_alpha.any(), "Expected some disordered points"

    @pytest.mark.slow
    def test_alpha_monotonic_in_eta(self):
        """For fixed D, alpha should be roughly increasing with eta."""
        result = sweep_vicsek_phase(
            eta_range=(0.0, 1.0),
            d_range=(0.5, 0.5),
            n_eta=6,
            n_d=1,
            n_boids=40,
            steps=120,
            seed=77,
        )
        alphas = result.alpha_grid[0]
        # Allow minor non-monotonicity due to noise
        diffs = np.diff(alphas)
        violations = np.sum(diffs < -0.15)  # count significant drops
        assert violations <= 1, f"Too many non-monotonic drops in alpha: {alphas}"


class TestPhaseBoundary:
    """Phase boundary detection."""

    def test_boundary_in_expected_range(self):
        """Phase boundary eta values are in [0, 1] or NaN."""
        result = PhaseDiagramResult(
            eta_grid=np.linspace(0, 1, 10),
            d_grid=np.array([0.5, 2.0]),
            # Simulate: alpha rises from 0 to 1 as eta increases
            alpha_grid=np.array([
                [0.05, 0.10, 0.20, 0.35, 0.55, 0.70, 0.85, 0.92, 0.96, 0.98],
                [0.02, 0.03, 0.05, 0.08, 0.15, 0.30, 0.48, 0.62, 0.78, 0.88],
            ]),
        )
        boundary = _find_phase_boundary(result)
        assert len(boundary) == 2
        # First row: crossing between eta=0.3 (alpha=0.35) and eta=0.4 (alpha=0.55)
        assert 0.3 < boundary[0] < 0.5
        # Second row: crossing between eta=0.6 (alpha=0.48) and eta=0.7 (alpha=0.62)
        assert 0.55 < boundary[1] < 0.75

    def test_boundary_nan_when_no_transition(self):
        """No alpha >= 0.5 -> NaN boundary."""
        result = PhaseDiagramResult(
            eta_grid=np.linspace(0, 1, 5),
            d_grid=np.array([3.0]),
            alpha_grid=np.array([[0.01, 0.02, 0.03, 0.04, 0.05]]),
        )
        boundary = _find_phase_boundary(result)
        assert np.isnan(boundary[0])

    def test_boundary_at_lowest_eta(self):
        """Already ordered at eta=0 -> boundary at first grid point."""
        result = PhaseDiagramResult(
            eta_grid=np.linspace(0, 1, 5),
            d_grid=np.array([0.1]),
            alpha_grid=np.array([[0.6, 0.7, 0.8, 0.85, 0.9]]),
        )
        boundary = _find_phase_boundary(result)
        assert boundary[0] == 0.0

    def test_boundary_with_nan_row(self):
        """Row of all NaN -> NaN boundary."""
        result = PhaseDiagramResult(
            eta_grid=np.linspace(0, 1, 5),
            d_grid=np.array([0.5]),
            alpha_grid=np.full((1, 5), np.nan),
        )
        boundary = _find_phase_boundary(result)
        assert np.isnan(boundary[0])


class TestSaveLoad:
    """Round-trip save/load."""

    def test_round_trip(self, tmp_path):
        """Save then load produces identical data."""
        result = PhaseDiagramResult(
            eta_grid=np.linspace(0, 1, 5),
            d_grid=np.array([0.5, 1.0, 2.0]),
            alpha_grid=np.random.uniform(0, 1, (3, 5)),
            boundary_eta=np.array([0.3, 0.5, np.nan]),
        )
        path = str(tmp_path / "diagram.npz")
        save_results(result, path)
        loaded = load_results(path)

        np.testing.assert_array_equal(loaded.eta_grid, result.eta_grid)
        np.testing.assert_array_equal(loaded.d_grid, result.d_grid)
        np.testing.assert_array_almost_equal(loaded.alpha_grid, result.alpha_grid)

    @pytest.mark.slow
    def test_boundary_shape_matches_d_grid(self):
        """boundary_eta has same length as d_grid after sweep."""
        result = sweep_vicsek_phase(
            n_eta=4, n_d=3, n_boids=30, steps=80, seed=42,
        )
        assert len(result.boundary_eta) == len(result.d_grid)
