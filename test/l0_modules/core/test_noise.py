"""Unit tests for core.noise — value_noise3, the position-sampled
3D value-noise field backing the speed_noise extension.
"""

import numpy as np

from pymurmur.core.noise import value_noise3


class TestValueNoise3:
    def test_output_range(self):
        rng = np.random.default_rng(0)
        pos = rng.uniform(-500.0, 1500.0, size=(5000, 3)).astype(np.float32)
        result = value_noise3(pos, frequency=0.003, seed=42)
        assert result.shape == (5000,)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_determinism(self):
        rng = np.random.default_rng(1)
        pos = rng.uniform(-100.0, 100.0, size=(200, 3)).astype(np.float32)
        a = value_noise3(pos, frequency=0.01, seed=7)
        b = value_noise3(pos.copy(), frequency=0.01, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_spatial_coherence(self):
        """A tiny position perturbation must yield a nearly-equal
        noise value — catches a discontinuous/broken hash."""
        rng = np.random.default_rng(2)
        pos = rng.uniform(-200.0, 200.0, size=(500, 3)).astype(np.float32)
        base = value_noise3(pos, frequency=0.005, seed=3)
        perturbed = value_noise3(pos + 1e-3, frequency=0.005, seed=3)
        assert np.max(np.abs(base - perturbed)) < 1e-3

    def test_non_degenerate_over_domain(self):
        """Positions spread over the default 1000x700x400 domain at the
        default speed_noise_frequency must not collapse to one value."""
        rng = np.random.default_rng(3)
        pos = rng.uniform([0, 0, 0], [1000, 700, 400], size=(500, 3)).astype(np.float32)
        result = value_noise3(pos, frequency=0.003, seed=1)
        assert result.std() > 0.05

    def test_seed_perturbs_field(self):
        rng = np.random.default_rng(4)
        pos = rng.uniform(-300.0, 300.0, size=(500, 3)).astype(np.float32)
        a = value_noise3(pos, frequency=0.01, seed=1)
        b = value_noise3(pos, frequency=0.01, seed=2)
        assert not np.allclose(a, b)
        assert np.mean(np.abs(a - b)) > 0.05

    def test_negative_coordinates(self):
        """Lattice indexing must handle negative world positions cleanly
        (open/sphere boundary modes allow boids to drift negative)."""
        pos = np.array(
            [[-1000.0, -700.0, -400.0], [1000.0, 700.0, 400.0]],
            dtype=np.float32,
        )
        result = value_noise3(pos, frequency=0.003, seed=5)
        assert np.all(np.isfinite(result))
        assert result.shape == (2,)
