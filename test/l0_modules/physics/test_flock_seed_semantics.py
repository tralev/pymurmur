"""Unit tests for physics.flock — D6 seed=0 semantics, boundary radius factor.

Split out of test_flock.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock
from test.helpers import _step_flock  # noqa: E402 — shared test helper

# ── D6 + D20: Seed=0 spawning determinism (cross-cutting) ─────


def test_same_seed_zero_with_spawning_deterministic():
    """D6+D20: Two engines with seed=0 + identical spawns → bit-identical.

    Before D6: seed=0 was conflated with None, causing non-deterministic
    spawning across runs. After D6+D20: seed=0 is a valid deterministic
    seed, and spawn_at uses the cube-velocity law deterministically.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg1 = SimConfig()
    cfg1.seed = 0
    cfg1.num_boids = 10
    cfg1.mode = "spatial"

    cfg2 = SimConfig()
    cfg2.seed = 0
    cfg2.num_boids = 10
    cfg2.mode = "spatial"

    e1 = SimulationEngine(cfg1)
    e2 = SimulationEngine(cfg2)

    # Run a few steps then spawn birds (D20 cube-law velocity)
    for _ in range(5):
        e1.step(1.0 / 60.0)
        e2.step(1.0 / 60.0)

    # Identical spawn operations on both engines
    e1.enqueue_spawn((100, 200, 300))
    e2.enqueue_spawn((100, 200, 300))
    e1.enqueue_spawn((400, 500, 600))
    e2.enqueue_spawn((400, 500, 600))
    e1.drain_commands()
    e2.drain_commands()

    # Both engines must be bit-identical after spawning
    np.testing.assert_array_equal(
        e1.flock.positions, e2.flock.positions,
        err_msg="D6+D20: seed=0 with spawning must be deterministic"
    )
    np.testing.assert_array_equal(
        e1.flock.velocities, e2.flock.velocities,
        err_msg="D6+D20: seed=0 spawn velocities must be bit-identical"
    )


# ── D6: Seed semantics (0 ≠ None) ────────────────────────────────


class TestD6SeedSemantics:
    """D6: seed=0 determinism is distinct from seed=None (fresh entropy).

    The bug (now fixed) was:
        default_rng(config.seed if config.seed else 0)
    which conflated seed=0 with seed=None because 0 is falsy.
    The fix is:
        default_rng(config.seed)
    numpy interprets None correctly as "fresh entropy" and 0 as
    "deterministic seed 0".
    """

    def test_seed_zero_is_deterministic(self, default_config):
        """D6: Two flocks with seed=0 produce identical state."""
        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 30
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = 0
        cfg2.num_boids = 30
        flock2 = PhysicsFlock(cfg2)

        # Seed 0 should be honoured — both flocks must be bit-identical.
        np.testing.assert_array_equal(flock1.positions, flock2.positions)
        np.testing.assert_array_equal(flock1.velocities, flock2.velocities)
        np.testing.assert_array_equal(flock1.seeds, flock2.seeds)

    def test_seed_zero_diverges_from_seed_none(self, default_config):
        """D6: seed=0 produces different output than seed=None.

        If seed=None were being replaced with seed=0 (the original bug),
        both flocks would be identical.  They must differ."""
        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 30
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = None
        cfg2.num_boids = 30
        flock2 = PhysicsFlock(cfg2)

        # seed=None must produce fresh entropy, not seed=0.
        # The probability of two 90-dimensional random draws colliding
        # is astronomically small, so a single comparison suffices.
        assert not np.array_equal(flock1.positions, flock2.positions), (
            "seed=0 and seed=None must produce different positions"
        )
        assert not np.array_equal(flock1.velocities, flock2.velocities), (
            "seed=0 and seed=None must produce different velocities"
        )

    def test_seed_none_is_nondeterministic(self, default_config):
        """D6: Two flocks with seed=None produce different state."""
        cfg1 = default_config
        cfg1.seed = None
        cfg1.num_boids = 30
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = None
        cfg2.num_boids = 30
        flock2 = PhysicsFlock(cfg2)

        # seed=None should give fresh entropy each time.
        assert not np.array_equal(flock1.positions, flock2.positions), (
            "seed=None must produce fresh entropy each call"
        )

    def test_seed_zero_determinism_persists_after_steps(
        self, default_config,
    ):
        """D6: seed=0 remains deterministic after multiple integration steps."""
        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 20
        flock1 = PhysicsFlock(cfg1)

        cfg2 = default_config
        cfg2.seed = 0
        cfg2.num_boids = 20
        flock2 = PhysicsFlock(cfg2)

        # Step both flocks the same number of times
        for _ in range(5):
            _step_flock(flock1, cfg1, 1.0 / 60.0)
            _step_flock(flock2, cfg2, 1.0 / 60.0)

        np.testing.assert_array_equal(flock1.positions, flock2.positions)

    def test_seed_via_engine_zero_vs_none_diverge(
        self, default_config,
    ):
        """D6: At engine level, seed=0 and seed=None diverge after stepping."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg1 = default_config
        cfg1.seed = 0
        cfg1.num_boids = 20
        eng1 = SimulationEngine(cfg1)

        cfg2 = default_config
        cfg2.seed = None
        cfg2.num_boids = 20
        eng2 = SimulationEngine(cfg2)

        # Step both 10 times in headless mode
        eng1.step(1.0 / 60.0)
        eng2.step(1.0 / 60.0)

        # seed=0 and seed=None must produce different trajectories
        assert not np.array_equal(
            eng1.flock.positions, eng2.flock.positions,
        ), "seed=0 and seed=None must diverge at engine level"


def test_boundary_radius_factor_scales_sphere_clamp(default_config):
    """C3: boundary_radius_factor scales the effective sphere boundary."""
    cfg = default_config
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 100.0
    cfg.num_boids = 1
    cfg.boundary_radius_factor = 2.0

    flock = PhysicsFlock(cfg)
    center = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    # Place the bird far outside even the scaled radius, at rest.
    flock.positions[0] = center + np.array([500.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[0] = 0.0
    flock.accelerations[0] = 0.0

    flock.integrate(cfg, dt=1.0 / 60.0)

    dist = float(np.linalg.norm(flock.positions[0] - center))
    assert dist == pytest.approx(200.0, abs=1e-3), (
        f"Expected hard clamp at radius*factor=200, got {dist:.3f}"
    )


def test_boundary_radius_factor_default_is_noop(default_config):
    """C3: boundary_radius_factor=1.0 (default) matches unscaled behaviour."""
    cfg = default_config
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 100.0
    cfg.num_boids = 1
    assert cfg.boundary_radius_factor == 1.0

    flock = PhysicsFlock(cfg)
    center = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    flock.positions[0] = center + np.array([500.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[0] = 0.0
    flock.accelerations[0] = 0.0

    flock.integrate(cfg, dt=1.0 / 60.0)

    dist = float(np.linalg.norm(flock.positions[0] - center))
    assert dist == pytest.approx(100.0, abs=1e-3)
