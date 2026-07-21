"""Shared pytest fixtures for all test files."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure pymurmur is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: the GL-context-cycle `gc.collect()` fixture (needed by tests that
# construct real moderngl contexts) deliberately does NOT live here.
# Applying it suite-wide cost 2.5x total runtime (120s -> 300s) for a
# concern that only affects ~830 of ~2900 tests. It's scoped instead to
# conftest.py files in test/l2_integration/, test/l3_modules/viz/,
# test/l3_modules/simulation/, and test/l3_modules/capture/ — the
# directories that actually construct Renderer3D/Visualizer instances.


@pytest.fixture
def default_config():
    """SimConfig with default projection mode parameters.

    D6: SimConfig.seed defaults to None (fresh entropy per flock), which
    makes geometry-dependent assertions flaky. The shared fixture pins a
    seed; tests exercising seed=None semantics build their own SimConfig.
    """
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.seed = 42
    return cfg


@pytest.fixture
def spatial_config():
    """SimConfig with spatial mode, N=200."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig()
    cfg.seed = 42  # D6: pin — see default_config
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
    """Check if ModernGL can create a standalone context.

    Releases the probe context immediately — an unreleased context
    here was found to become moderngl's process-global "default
    context" for the rest of the session (this fixture is session-scoped
    and typically the first context created), which then interacted
    badly with later tests' own context releases.
    """
    try:
        import moderngl
        ctx = moderngl.create_context(standalone=True, require=330)
        ctx.release()
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
