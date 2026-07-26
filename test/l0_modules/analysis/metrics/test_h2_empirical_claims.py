"""H₂ robustness empirical-claim tests — A9 (connectivity threshold),
A11 (m* independence from N).

Split out of test_h2.py (file-size split, pure extraction).
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    compute_h2,
    find_connectivity_threshold,
    find_m_star_by_sensing_cost,
)
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine


class TestA11MStarIndependentOfN:
    """A11 (Young et al. 2013): m* shows negligible dependence on
    flock size N (paper: R^2=0.0178). Uses find_m_star_by_sensing_cost
    (A7's paper-correct m* definition, argmax R_per_m), not
    find_optimal_m's J(m) cost function -- a different quantity."""

    @staticmethod
    def _make_flock(N, thickness_scale, rng):
        """Fixed-thickness squashed-Gaussian cloud (same construction
        as TestMStarThickness in test_metrics_expensive.py)."""
        xy = rng.normal(0, 100, (N, 2))
        z = rng.normal(0, 100 * thickness_scale, (N, 1))
        return np.hstack([xy, z]).astype(np.float32)

    @pytest.mark.slow
    def test_a11_m_star_independent_of_n(self):
        """Real, passing test: measured directly before writing this
        assertion -- m* stayed exactly 3 across N=100..1200 at fixed
        thickness (R^2=0.0), reproducing the paper's 'no dependence'
        finding."""
        from scipy.stats import linregress

        rng = np.random.default_rng(9)
        Ns = [100, 300, 600, 900, 1200]
        m_stars = []
        for N in Ns:
            positions = self._make_flock(N, 0.2, rng)
            m_star, _ = find_m_star_by_sensing_cost(positions)
            m_stars.append(m_star)

        if len(set(m_stars)) == 1:
            r_squared = 0.0  # constant series -- linregress correlation is undefined
        else:
            result = linregress(Ns, m_stars)
            r_squared = result.rvalue ** 2

        assert r_squared < 0.1, (
            f"m* vs N fit R^2={r_squared:.3f}, expected <0.1 "
            f"(negligible dependence); m_stars={m_stars}"
        )

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason=(
            "A11's second sub-claim (Young et al. 2013): peak "
            "robustness-per-neighbour is ALSO independent of N "
            "(paper: R^2=0.023). Measured directly: N=100..1200 at "
            "fixed thickness, the peak R_per_m VALUE (not m*) shows "
            "R^2~=0.97 against N -- strongly N-dependent, not "
            "independent. This is because R_nodal=sqrt(N)/H2 (A6) "
            "bakes in a sqrt(N) factor that A7's R_per_m=R_nodal/m "
            "doesn't remove, unlike whatever normalization the paper's "
            "own H2 pipeline uses on real starling data. The m*-vs-N "
            "sub-claim (tested above, passing) and this "
            "peak-value-vs-N sub-claim are different quantities with "
            "different outcomes in this codebase."
        ),
        strict=False,
    )
    def test_a11_peak_robustness_independent_of_n(self):
        from scipy.stats import linregress

        rng = np.random.default_rng(9)
        Ns = [100, 300, 600, 900, 1200]
        peak_vals = []
        for N in Ns:
            positions = self._make_flock(N, 0.2, rng)
            _, r_per_m = find_m_star_by_sensing_cost(positions)
            peak_vals.append(r_per_m)

        result = linregress(Ns, peak_vals)
        r_squared = result.rvalue ** 2
        assert r_squared < 0.1, (
            f"peak R_per_m vs N fit R^2={r_squared:.3f}, expected <0.1 "
            f"(negligible dependence); peak_vals={peak_vals}"
        )


class TestA9ConnectivityThreshold:
    """A9 (Young et al. 2013): m>=5 connects the sensing graph across
    394 field-observed starling-flock snapshots (N=440-2600); m=1,2
    are almost always disconnected. Measured directly in this codebase
    before writing these assertions (random-uniform and simulated
    projection-mode flocks, N=50-300, multiple seeds): m=1 is always
    disconnected, m=2 disconnects in ~9/10 cases, and the graph is
    fully connected by m=3 in every case tested -- earlier than the
    paper's own m=5, which doesn't contradict "m>=5 connects" (a
    sufficiency claim); this codebase's own graphs just reach it sooner."""

    def test_m1_always_disconnected(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            positions = rng.uniform(-100, 100, (300, 3)).astype(np.float32)
            _, h2 = compute_h2(positions, 1)
            assert not np.isfinite(h2), (
                f"seed={seed}: m=1 unexpectedly connected"
            )

    def test_connected_by_m5_random_flocks(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            positions = rng.uniform(-100, 100, (300, 3)).astype(np.float32)
            threshold = find_connectivity_threshold(positions)
            assert threshold is not None and threshold <= 5, (
                f"seed={seed}: connectivity threshold m={threshold}, "
                f"expected <=5 (A9's m>=5 sufficiency claim)"
            )

    def test_connected_by_m5_simulated_flock(self):
        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.num_boids = 200
        cfg.seed = 7
        engine = SimulationEngine(cfg)
        for _ in range(200):
            engine.step()
        threshold = find_connectivity_threshold(engine.flock.positions)
        assert threshold is not None and threshold <= 5, (
            f"simulated projection flock: connectivity threshold "
            f"m={threshold}, expected <=5"
        )

    def test_returns_none_when_nothing_connects(self):
        """Two far-apart clusters -- mirrors TestRPerM's disconnected-
        everywhere fixture -- stay disconnected across the whole tested
        m range, so find_connectivity_threshold returns None."""
        rng = np.random.default_rng(0)
        cluster_a = rng.uniform(0, 1, (25, 3)).astype(np.float32)
        cluster_b = rng.uniform(10000, 10001, (25, 3)).astype(np.float32)
        pts = np.vstack([cluster_a, cluster_b])
        assert find_connectivity_threshold(pts, m_max=20) is None
