"""P0.4 — Determinism tests for physics.flock.

Split out of test_flock.py (file-size split) — core init/state tests and
P0.5 swarm-centre tests stay elsewhere; this file covers same-seed
bit-identical reproducibility across all force modes.
"""

import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock


def test_flock_rng_initialised(default_config):
    """PhysicsFlock has a self.rng attribute initialised from config.seed."""
    cfg = default_config
    cfg.seed = 42
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "rng"), "flock.rng must exist"
    assert isinstance(flock.rng, np.random.Generator), (
        "flock.rng must be np.random.Generator"
    )


def test_same_seed_bit_identical():
    """Two engines with same seed produce bit-identical positions after 100 steps.

    P0.4 requirement: same seed → bit-identical after 100 steps per mode.
    Tests projection mode (deterministic, no zero-speed reseed).
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 30
    cfg.mode = "projection"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(
        e1.flock.positions, e2.flock.positions,
        err_msg="Same seed must produce bit-identical positions after 100 steps"
    )
    np.testing.assert_array_equal(
        e1.flock.velocities, e2.flock.velocities,
        err_msg="Same seed must produce bit-identical velocities after 100 steps"
    )


def test_same_seed_bit_identical_spatial():
    """Spatial mode also produces bit-identical results with same seed."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 77
    cfg.num_boids = 30
    cfg.mode = "spatial"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)


def test_same_seed_bit_identical_spatial_with_noise():
    """Spatial mode with noise_scale > 0 is deterministic (I1.5 regression).

    This guards against noise_force being called without the seeded rng —
    a missing rng argument causes np.random.default_rng() which produces
    different values on every call.  Fixed by passing rng to noise_force.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 99
    cfg.num_boids = 40
    cfg.mode = "spatial"
    cfg.noise_scale = 1.5

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(50):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(
        e1.flock.positions, e2.flock.positions,
        err_msg="spatial mode with noise_scale > 0 must be deterministic (I1.5)"
    )
    np.testing.assert_array_equal(
        e1.flock.velocities, e2.flock.velocities,
        err_msg="spatial mode with noise_scale > 0 must be deterministic (I1.5)"
    )


def test_same_seed_bit_identical_field():
    """Field mode also produces bit-identical results with same seed."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 123
    cfg.num_boids = 30
    cfg.mode = "field"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)


def test_same_seed_bit_identical_vicsek():
    """Vicsek mode produces bit-identical results with same seed (I1.5)."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 50
    cfg.mode = "vicsek"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)
    np.testing.assert_array_equal(e1.flock.velocities, e2.flock.velocities)


def test_same_seed_bit_identical_influencer():
    """Influencer mode produces bit-identical results with same seed (I1.5)."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 77
    cfg.num_boids = 30
    cfg.mode = "influencer"

    e1 = SimulationEngine(cfg)
    e2 = SimulationEngine(cfg)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)
    np.testing.assert_array_equal(e1.flock.velocities, e2.flock.velocities)


def test_all_modes_deterministic():
    """Parametric: every force mode produces bit-identical results with same seed."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    for mode in ["projection", "spatial", "field", "vicsek", "influencer"]:
        cfg = SimConfig()
        cfg.seed = 123
        cfg.num_boids = 20
        cfg.mode = mode
        if mode == "spatial":
            cfg.noise_scale = 1.0  # exercise the noise RNG path (I1.5 regression guard)

        e1 = SimulationEngine(cfg)
        e2 = SimulationEngine(cfg)

        for _ in range(50):
            e1.step(1.0 / 60.0)
            e2.step(1.0 / 60.0)

        np.testing.assert_array_equal(
            e1.flock.positions, e2.flock.positions,
            err_msg=f"{mode}: same seed must produce bit-identical positions"
        )


@pytest.mark.parametrize("mode", ["projection", "spatial", "field", "vicsek", "influencer"])
def test_different_seeds_diverge(mode):
    """Different seeds produce different positions after 100 steps.

    Parametrized across all 5 force modes. Verifies that seed-based
    RNG pipeline works for every mode — different seed → different
    trajectory, which is the complement of the bit-identical test.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg1 = SimConfig()
    cfg1.seed = 42
    cfg1.num_boids = 30
    cfg1.mode = mode

    cfg2 = SimConfig()
    cfg2.seed = 99
    cfg2.num_boids = 30
    cfg2.mode = mode

    e1 = SimulationEngine(cfg1)
    e2 = SimulationEngine(cfg2)

    for _ in range(100):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    assert not np.array_equal(e1.flock.positions, e2.flock.positions), (
        f"{mode}: different seeds must produce different positions"
    )
