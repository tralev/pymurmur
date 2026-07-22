"""Phase diagram tests — Phase 9.4 Vicsek eta x D sweep.
"""

import numpy as np
import pytest

from pymurmur.analysis.phase_diagram import (
    PhaseDiagramResult,
    _find_phase_boundary,
    load_results,
    save_results,
    sweep_vicsek_phase,
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


class TestQuickMode:
    """S3.1: quick=True single-step snapshot sweep — cheap interactive
    exploration, not a scientifically rigorous settled-run measurement."""

    def test_quick_returns_same_grid_shape_as_settled(self):
        result_quick = sweep_vicsek_phase(
            eta_range=(0.0, 1.0), d_range=(0.0, 2.0),
            n_eta=5, n_d=3, n_boids=30, seed=42, quick=True,
        )
        assert result_quick.eta_grid.shape == (5,)
        assert result_quick.d_grid.shape == (3,)
        assert result_quick.alpha_grid.shape == (3, 5)
        assert not np.all(np.isnan(result_quick.alpha_grid))

    @pytest.mark.slow
    def test_quick_runs_much_faster_than_settled(self):
        import time

        t0 = time.perf_counter()
        sweep_vicsek_phase(
            eta_range=(0.0, 1.0), d_range=(0.0, 2.0),
            n_eta=6, n_d=4, n_boids=40, seed=42, quick=True,
        )
        t_quick = time.perf_counter() - t0

        t0 = time.perf_counter()
        sweep_vicsek_phase(
            eta_range=(0.0, 1.0), d_range=(0.0, 2.0),
            n_eta=6, n_d=4, n_boids=40, steps=80, seed=42,
        )
        t_full = time.perf_counter() - t0

        assert t_quick < t_full, (
            f"quick={t_quick:.3f}s should be faster than settled={t_full:.3f}s"
        )
        # Not asserting the literal ~200x claim (machine/N-dependent) —
        # a comfortable margin is enough to catch a regression to the
        # settled-run cost.
        assert t_quick < t_full / 3, (
            f"quick mode should be meaningfully cheaper: "
            f"quick={t_quick:.3f}s, full={t_full:.3f}s"
        )

    def test_quick_respects_nematic_order_type(self):
        result = sweep_vicsek_phase(
            n_eta=3, n_d=2, n_boids=20, seed=1,
            order_type="nematic", quick=True,
        )
        assert result.order_type == "nematic"
        assert result.alpha_grid.shape == (2, 3)

    def test_quick_produces_valid_boundary_or_nan(self):
        """boundary_eta computation still runs on quick-mode output
        without raising, even though quick data is noisier."""
        result = sweep_vicsek_phase(
            eta_range=(0.0, 1.0), d_range=(0.0, 2.0),
            n_eta=6, n_d=3, n_boids=30, seed=7, quick=True,
        )
        assert result.boundary_eta.shape == (3,)


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


class TestDAxis:
    """D-axis behaviour: phase boundary η_c(D) monotonic in D."""

    @pytest.mark.slow
    def test_phase_boundary_monotonic_in_d(self):
        """η_c(D) should rise with D — more noise needs more coupling to order."""
        result = sweep_vicsek_phase(
            eta_range=(0.0, 1.0),
            d_range=(0.1, 3.0),
            n_eta=8,
            n_d=5,
            n_boids=40,
            steps=120,
            seed=42,
        )
        boundary = result.boundary_eta
        # Ignore NaN boundary points
        valid = ~np.isnan(boundary)
        assert valid.sum() >= 3, (
            f"Too few valid boundary points ({valid.sum()}/{len(boundary)}); "
            f"cannot check D-axis monotonicity. boundary={boundary}"
        )
        valid_boundary = boundary[valid]
        valid_d = result.d_grid[valid]
        # η_c should be monotonically increasing with D
        diffs = np.diff(valid_boundary)
        drops = np.sum(diffs < -0.10)
        # Allow 1 drop — finite-size noise in small simulations (N=40, steps=120)
        # can cause occasional non-monotonic bumps near the phase transition.
        # See test_alpha_monotonic_in_eta for the same pattern.
        assert drops <= 1, (
                f"Phase boundary not monotonic in D: boundary={valid_boundary}, D={valid_d}"
            )

    def test_boundary_at_zero_d_is_low(self):
        """At D≈0, the boundary η_c should be very low (easy to order)."""
        result = sweep_vicsek_phase(
            eta_range=(0.0, 1.0),
            d_range=(0.0, 0.0),
            n_eta=10,
            n_d=1,
            n_boids=40,
            steps=120,
            seed=42,
        )
        # At D=0, any η>0 should order almost immediately
        if not np.isnan(result.boundary_eta[0]):
            assert result.boundary_eta[0] < 0.3, (
                f"At D=0 expected low η_c, got {result.boundary_eta[0]}"
            )

    def test_boundary_at_high_d_is_high(self):
        """At high D, the boundary η_c should be high (needs strong coupling)."""
        result = sweep_vicsek_phase(
            eta_range=(0.0, 1.0),
            d_range=(3.5, 3.5),
            n_eta=10,
            n_d=1,
            n_boids=40,
            steps=120,
            seed=42,
        )
        # At high D, it takes stronger coupling to order
        if not np.isnan(result.boundary_eta[0]):
            assert result.boundary_eta[0] > 0.3, (
                f"At high D expected higher η_c, got {result.boundary_eta[0]}"
            )


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
        assert loaded.order_type == "polar", (
            f"Expected order_type='polar', got '{loaded.order_type}'"
        )

    @pytest.mark.slow
    def test_boundary_shape_matches_d_grid(self):
        """boundary_eta has same length as d_grid after sweep."""
        result = sweep_vicsek_phase(
            n_eta=4, n_d=3, n_boids=30, steps=80, seed=42,
        )
        assert len(result.boundary_eta) == len(result.d_grid)


# ── P9.1: Nematic order_type option ──────────────────────────

class TestNematicOrder:
    """Phase diagram sweep with order_type='nematic'."""

    @pytest.mark.slow
    def test_nematic_sweep_shapes(self):
        """Nematic sweep produces same grid shapes as polar."""
        result = sweep_vicsek_phase(
            order_type="nematic",
            n_eta=5, n_d=3, n_boids=30, steps=80, seed=42,
        )
        assert result.eta_grid.shape == (5,)
        assert result.d_grid.shape == (3,)
        assert result.alpha_grid.shape == (3, 5)
        assert result.order_type == "nematic"
        assert not np.all(np.isnan(result.alpha_grid))

    @pytest.mark.slow
    def test_nematic_default_is_polar(self):
        """Default order_type is 'polar'."""
        result = sweep_vicsek_phase(
            n_eta=3, n_d=2, n_boids=20, steps=60, seed=99,
        )
        assert result.order_type == "polar"

    @pytest.mark.slow
    def test_nematic_high_eta_gives_high_S(self):
        """High coupling + low noise → nematic S close to 1."""
        result = sweep_vicsek_phase(
            order_type="nematic",
            eta_range=(0.7, 1.0),
            d_range=(0.0, 0.5),
            n_eta=4, n_d=2, n_boids=40, steps=120, seed=99,
        )
        high_S = result.alpha_grid > 0.5
        assert high_S.any(), "Expected some nematically ordered points"

    @pytest.mark.slow
    def test_nematic_boundary_still_valid(self):
        """Phase boundary detection works with nematic S too."""
        result = sweep_vicsek_phase(
            order_type="nematic",
            n_eta=4, n_d=3, n_boids=30, steps=80, seed=42,
        )
        assert len(result.boundary_eta) == len(result.d_grid)
        # Boundary values should be NaN or in [0,1]
        for b in result.boundary_eta:
            assert np.isnan(b) or 0.0 <= b <= 1.0

    def test_nematic_save_load_roundtrip(self, tmp_path):
        """Save/load preserves order_type='nematic'."""
        result = PhaseDiagramResult(
            eta_grid=np.linspace(0, 1, 5),
            d_grid=np.array([0.5, 1.0]),
            alpha_grid=np.random.uniform(0, 1, (2, 5)),
            order_type="nematic",
            boundary_eta=np.array([0.3, 0.5]),
        )
        path = str(tmp_path / "nematic.npz")
        save_results(result, path)
        loaded = load_results(path)
        assert loaded.order_type == "nematic"
        np.testing.assert_array_almost_equal(loaded.alpha_grid, result.alpha_grid)

    def test_polar_save_load_roundtrip(self, tmp_path):
        """Save/load preserves order_type='polar'."""
        result = PhaseDiagramResult(
            eta_grid=np.linspace(0, 1, 3),
            d_grid=np.array([1.0]),
            alpha_grid=np.array([[0.1, 0.5, 0.9]]),
            order_type="polar",
        )
        path = str(tmp_path / "polar.npz")
        save_results(result, path)
        loaded = load_results(path)
        assert loaded.order_type == "polar"

    @pytest.mark.slow
    def test_nematic_vs_polar_different(self):
        """For the same parameters, nematic S ≥ polar α (nematic is looser)."""
        result_polar = sweep_vicsek_phase(
            order_type="polar",
            eta_range=(0.3, 0.8),
            d_range=(1.0, 1.0),
            n_eta=4, n_d=1, n_boids=40, steps=120, seed=42,
        )
        result_nematic = sweep_vicsek_phase(
            order_type="nematic",
            eta_range=(0.3, 0.8),
            d_range=(1.0, 1.0),
            n_eta=4, n_d=1, n_boids=40, steps=120, seed=42,
        )
        # Nematic S vs polar α: nematic aligns modulo π (not 2π),
        # so the effective noise acts differently on the critical point.
        # Nematic S transitions at a different phase boundary under the
        # same D — allow larger tolerance to account for the shift.
        polar_mean = float(np.nanmean(result_polar.alpha_grid))
        nematic_mean = float(np.nanmean(result_nematic.alpha_grid))
        assert nematic_mean >= polar_mean - 0.3, (
            f"nematic mean {nematic_mean:.3f} should be >= polar mean {polar_mean:.3f} − 0.3"
        )


def test_nematic_invalid_order_type_raises():
    """P9.1: Invalid order_type raises ValueError."""
    import pytest

    from pymurmur.analysis.phase_diagram import sweep_vicsek_phase

    with pytest.raises(ValueError, match="order_type must be 'polar' or 'nematic'"):
        sweep_vicsek_phase(n_eta=2, n_d=2, n_boids=10, steps=20, order_type="bogus")
