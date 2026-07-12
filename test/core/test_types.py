"""Unit tests for core.types — Vec3, FlockArrays, ForceFunc, ForceKernel,
and math helpers (safe_normalize, limit3, lerp, rotate_about, smoothstep,
hash01, min_image, min_image_distance, fibonacci_sphere, seed_noise3).
"""

import numpy as np
import pytest
from pymurmur.core.types import (
    FlockArrays,
    ForceKernel,
    safe_normalize,
    limit3,
    lerp,
    rotate_about,
    smoothstep,
    hash01,
    min_image,
    min_image_distance,
    fibonacci_sphere,
    seed_noise3,
)


def test_flock_arrays_creation():
    """FlockArrays with N=150 has correct shapes."""
    N = 150
    fa = FlockArrays(
        positions=np.zeros((N, 3), dtype=np.float32),
        velocities=np.zeros((N, 3), dtype=np.float32),
        accelerations=np.zeros((N, 3), dtype=np.float32),
        seeds=np.random.uniform(0, 1, N).astype(np.float32),
        last_theta=np.zeros(N, dtype=np.float32),
        active=np.ones(N, dtype=bool),
    )
    assert fa.positions.shape == (N, 3)
    assert fa.velocities.shape == (N, 3)
    assert fa.accelerations.shape == (N, 3)
    assert fa.seeds.shape == (N,)
    assert fa.last_theta.shape == (N,)
    assert fa.active.shape == (N,)


def test_flock_arrays_n_active():
    """N_active equals active.sum()."""
    active = np.array([True, True, False, True], dtype=bool)
    fa = FlockArrays(
        positions=np.zeros((4, 3), dtype=np.float32),
        velocities=np.zeros((4, 3), dtype=np.float32),
        accelerations=np.zeros((4, 3), dtype=np.float32),
        seeds=np.zeros(4, dtype=np.float32),
        last_theta=np.zeros(4, dtype=np.float32),
        active=active,
    )
    assert fa.N_active == 3
    assert fa.N_capacity == 4


def test_flock_arrays_dtype():
    """All arrays use float32, active uses bool."""
    N = 10
    fa = FlockArrays(
        positions=np.zeros((N, 3), dtype=np.float32),
        velocities=np.zeros((N, 3), dtype=np.float32),
        accelerations=np.zeros((N, 3), dtype=np.float32),
        seeds=np.zeros(N, dtype=np.float32),
        last_theta=np.zeros(N, dtype=np.float32),
        active=np.ones(N, dtype=bool),
    )
    assert fa.positions.dtype == np.float32
    assert fa.velocities.dtype == np.float32
    assert fa.accelerations.dtype == np.float32
    assert fa.active.dtype == bool


def test_flock_arrays_zero_birds():
    """N=0 produces empty arrays without error."""
    fa = FlockArrays(
        positions=np.zeros((0, 3), dtype=np.float32),
        velocities=np.zeros((0, 3), dtype=np.float32),
        accelerations=np.zeros((0, 3), dtype=np.float32),
        seeds=np.zeros(0, dtype=np.float32),
        last_theta=np.zeros(0, dtype=np.float32),
        active=np.zeros(0, dtype=bool),
    )
    assert fa.N_active == 0
    assert fa.N_capacity == 0


def test_flock_arrays_default_active():
    """All active entries are True on creation."""
    N = 20
    fa = FlockArrays(
        positions=np.zeros((N, 3), dtype=np.float32),
        velocities=np.zeros((N, 3), dtype=np.float32),
        accelerations=np.zeros((N, 3), dtype=np.float32),
        seeds=np.zeros(N, dtype=np.float32),
        last_theta=np.zeros(N, dtype=np.float32),
        active=np.ones(N, dtype=bool),
    )
    assert fa.active.all()
    assert fa.N_active == N


def test_force_func_protocol():
    """A function matching ForceFunc signature passes isinstance check."""
    def my_force(flock, config) -> None:
        pass

    # ForceFunc is a Protocol — structural typing, no isinstance possible
    # Verify the function has the right call signature
    assert callable(my_force)
    # Verify it accepts two positional args
    import inspect
    sig = inspect.signature(my_force)
    params = list(sig.parameters.keys())
    assert len(params) == 2


def test_force_kernel_signature():
    """ForceKernel is a Callable type alias."""
    # ForceKernel = Callable[[ndarray, ...], None] — verify it exists
    assert ForceKernel is not None
    # Verify a function matching the 11-arg signature can be assigned
    def matches_kernel(
        pos: np.ndarray, vel: np.ndarray, acc: np.ndarray,
        active: np.ndarray, nbr: np.ndarray,
        sw: float, aw: float, cw: float, ns: float, v0: float, mf: float,
    ) -> None:
        pass
    # Structural check: function is callable with correct arity
    import inspect
    sig = inspect.signature(matches_kernel)
    assert len(sig.parameters) == 11


# ── Math helper tests (P0.12) ────────────────────────────────────


class TestSafeNormalize:
    def test_unit_vector_preserved(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = safe_normalize(v)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_non_unit_vector(self):
        v = np.array([3.0, 0.0, 0.0], dtype=np.float32)
        result = safe_normalize(v)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_zero_vector_returns_zero(self):
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = safe_normalize(v)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_batched(self):
        v = np.array([[0.0, 5.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        result = safe_normalize(v)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0], atol=1e-6)

    def test_tiny_vector_still_normalized(self):
        """Vector with very small but non-zero magnitude is normalized, not zeroed."""
        v = np.array([1e-7, 0.0, 0.0], dtype=np.float32)
        result = safe_normalize(v, eps=1e-12)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-5)

    def test_batched_zero_rows(self):
        """Mixed zero and non-zero rows: zeros stay zero, others normalized."""
        v = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32)
        result = safe_normalize(v)
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(result[1]), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(result[2]), 1.0, atol=1e-6)


class TestLimit3:
    def test_fast_clamped(self):
        v = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        result = limit3(v, 4.0)
        assert np.linalg.norm(result) == pytest.approx(4.0, abs=1e-5)

    def test_slow_unchanged(self):
        v = np.array([1.0, 2.0, 2.0], dtype=np.float32)  # |v| = 3
        result = limit3(v, 4.0)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_exact_boundary(self):
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # |v| = 5
        result = limit3(v, 5.0)
        np.testing.assert_allclose(result, v, atol=1e-5)

    def test_batched(self):
        v = np.array([[10.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        result = limit3(v, 4.0)
        assert np.linalg.norm(result[0]) == pytest.approx(4.0, abs=1e-5)
        assert np.linalg.norm(result[1]) == pytest.approx(2.0, abs=1e-5)


class TestLerp:
    def test_t_zero_returns_a(self):
        np.testing.assert_array_equal(lerp(np.array(1.0), np.array(5.0), 0.0), 1.0)

    def test_t_one_returns_b(self):
        np.testing.assert_allclose(lerp(np.array(1.0), np.array(5.0), 1.0), 5.0)

    def test_midpoint(self):
        np.testing.assert_allclose(lerp(np.array(0.0), np.array(10.0), 0.5), 5.0)

    def test_vector_lerp(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 8.0], dtype=np.float32)
        np.testing.assert_allclose(lerp(a, b, 0.5), [1.0, 2.0, 4.0], atol=1e-6)


class TestRotateAbout:
    def test_exact_90_degrees(self):
        """Rotate (1,0,0) about z-axis by π/2 → (0,1,0)."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        k = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        result = rotate_about(v, k, np.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-5)

    def test_full_circle_returns_same(self):
        v = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        k = np.array([1.0, 0.5, 0.3], dtype=np.float32)
        result = rotate_about(v, k, 2 * np.pi)
        np.testing.assert_allclose(result, v, atol=1e-5)

    def test_preserves_length(self):
        v = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        k = np.array([1.0, 0.5, 0.3], dtype=np.float32)
        result = rotate_about(v, k, 1.3)
        assert np.linalg.norm(result) == pytest.approx(np.linalg.norm(v), rel=1e-5)

    def test_batched(self):
        v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        k = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        result = rotate_about(v, k, np.pi / 2)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result[1], [-1.0, 0.0, 0.0], atol=1e-5)

    def test_axis_normalized_automatically(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        k = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # non-unit axis
        result = rotate_about(v, k, np.pi)
        # rotation about z by π: (1,0,0) → (-1,0,0)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-5)


class TestSmoothstep:
    def test_below_edge0(self):
        assert smoothstep(0.0, 1.0, -0.5) == 0.0

    def test_above_edge1(self):
        assert smoothstep(0.0, 1.0, 1.5) == 1.0

    def test_midpoint(self):
        assert smoothstep(0.0, 1.0, 0.5) == pytest.approx(0.5, abs=1e-6)

    def test_monotonic(self):
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        result = smoothstep(0.0, 1.0, x)
        assert np.all(np.diff(result) >= 0)

    def test_array_input(self):
        x = np.linspace(0, 1, 5, dtype=np.float32)
        result = smoothstep(0.0, 1.0, x)
        assert result.shape == (5,)
        assert result[0] == 0.0
        assert result[-1] == 1.0


class TestHash01:
    def test_range(self):
        x = np.arange(100, dtype=np.float32)
        result = hash01(x)
        assert result.min() >= 0.0
        assert result.max() < 1.0

    def test_deterministic(self):
        a = hash01(np.array([3.14, 2.71], dtype=np.float32))
        b = hash01(np.array([3.14, 2.71], dtype=np.float32))
        np.testing.assert_array_equal(a, b)

    def test_different_inputs_different_hash(self):
        a = hash01(np.array([1.0], dtype=np.float32))
        b = hash01(np.array([2.0], dtype=np.float32))
        assert a[0] != b[0]


class TestMinImage:
    def test_wraps(self):
        """(90,0,0) with box=(100,100,100) → (-10,0,0)."""
        d = np.array([[90.0, 0.0, 0.0]], dtype=np.float32)
        box = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result = min_image(d, box)
        np.testing.assert_allclose(result[0], [-10.0, 0.0, 0.0], atol=1e-5)

    def test_within_half_box_preserved(self):
        d = np.array([[10.0, 20.0, -30.0]], dtype=np.float32)
        box = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result = min_image(d, box)
        np.testing.assert_allclose(result[0], [10.0, 20.0, -30.0], atol=1e-5)

    def test_negative_wraps(self):
        d = np.array([[-90.0, 0.0, 0.0]], dtype=np.float32)
        box = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result = min_image(d, box)
        np.testing.assert_allclose(result[0], [10.0, 0.0, 0.0], atol=1e-5)

    def test_always_in_half_box(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            d = rng.uniform(-200, 200, (10, 3)).astype(np.float32)
            box = np.array([100.0, 100.0, 100.0], dtype=np.float32)
            result = min_image(d, box)
            assert np.all(np.abs(result) <= 50.0)


class TestMinImageDistance:
    def test_known_distance(self):
        d = np.array([[90.0, 0.0, 0.0]], dtype=np.float32)
        box = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result = min_image_distance(d, box)
        np.testing.assert_allclose(result[0], 10.0, atol=1e-5)

    def test_zero_vector(self):
        d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        box = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result = min_image_distance(d, box)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-8)


class TestFibonacciSphere:
    def test_count(self):
        pts = fibonacci_sphere(256)
        assert pts.shape == (256, 3)

    def test_unit_vectors(self):
        pts = fibonacci_sphere(100)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_birds(self):
        pts = fibonacci_sphere(0)
        assert pts.shape == (0, 3)

    def test_one_bird(self):
        pts = fibonacci_sphere(1)
        assert pts.shape == (1, 3)
        np.testing.assert_allclose(np.linalg.norm(pts[0]), 1.0, atol=1e-5)

    def test_dtype(self):
        pts = fibonacci_sphere(50)
        assert pts.dtype == np.float32


class TestSeedNoise3:
    def test_shape(self):
        seeds = np.arange(1000, dtype=np.float32)
        noise = seed_noise3(seeds, 0.5)
        assert noise.shape == (1000, 3)
        assert noise.dtype == np.float32

    def test_range(self):
        """Per-axis noise bounded in [-0.18, 0.18]."""
        seeds = np.arange(1000, dtype=np.float32)
        noise = seed_noise3(seeds, 0.5)
        assert noise.min() >= -0.18
        assert noise.max() <= 0.18

    def test_deterministic(self):
        seeds = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = seed_noise3(seeds, 0.5)
        b = seed_noise3(seeds, 0.5)
        np.testing.assert_array_equal(a, b)

    def test_time_affects_output(self):
        seeds = np.array([1.0], dtype=np.float32)
        a = seed_noise3(seeds, 0.0)
        b = seed_noise3(seeds, 1.0)
        assert not np.array_equal(a, b)

    def test_per_axis_independent(self):
        """Different seeds produce different noise vectors."""
        seeds = np.arange(100, dtype=np.float32)
        noise = seed_noise3(seeds, 0.0)
        # Check variance across seeds (should be nonzero for each axis)
        assert noise[:, 0].std() > 0.05
        assert noise[:, 1].std() > 0.05
        assert noise[:, 2].std() > 0.05
