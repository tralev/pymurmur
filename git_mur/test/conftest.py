"""Shared pytest fixtures for all test files."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure pymurmur is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def default_config():
    """SimConfig with default projection mode parameters."""
    from pymurmur.core.config import SimConfig
    return SimConfig()


@pytest.fixture
def spatial_config():
    """SimConfig with spatial mode, N=200."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.mode = "spatial"
    cfg.num_boids = 200
    return cfg


@pytest.fixture
def small_flock(default_config):
    """PhysicsFlock with N=50 for fast unit tests."""
    from pymurmur.physics.flock import PhysicsFlock
    cfg = default_config
    cfg.num_boids = 50
    return PhysicsFlock(cfg)


@pytest.fixture
def two_bird_flock(default_config):
    """PhysicsFlock with exactly 2 birds for neighbour tests."""
    from pymurmur.physics.flock import PhysicsFlock
    cfg = default_config
    cfg.num_boids = 2
    return PhysicsFlock(cfg)


@pytest.fixture
def known_positions():
    """Return (N, 3) array with known positions."""
    return np.array([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [-10, 0, 0]
    ], dtype=np.float32)


@pytest.fixture
def known_velocities():
    """Return (N, 3) array with known velocities."""
    return np.array([
        [1, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]
    ], dtype=np.float32)


@pytest.fixture
def neighbor_idx():
    """Pre-computed neighbour indices: bird 0 sees [1,2,3], etc."""
    N = 4
    idx = np.empty((N, N - 1), dtype=np.int32)
    for i in range(N):
        idx[i] = [j for j in range(N) if j != i]
    return idx


@pytest.fixture(scope="session")
def gpu_available():
    """Check if ModernGL can create a standalone context."""
    try:
        import moderngl
        moderngl.create_context(standalone=True, require=330)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def numba_available():
    """Check if numba is importable."""
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False
