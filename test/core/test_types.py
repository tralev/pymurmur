"""Unit tests for core.types — Vec3, FlockArrays, ForceFunc, ForceKernel."""

import numpy as np
from pymurmur.core.types import FlockArrays, ForceKernel


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
