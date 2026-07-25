"""Unit tests for physics.flock — PhysicsFlock init/state, determinism,
smoothed swarm centre.

Remaining core after the file-size split (spatial index tests moved to
test_spatial_index.py; species/stash/max-speed to test_flock_state.py;
spawn_at to test_flock_spawn_at.py/test_flock_spawn_at_engine.py; D6
seed semantics to test_flock_seed_semantics.py).
"""

import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock
from test.helpers import _step_flock  # noqa: E402 — shared test helper


def test_flock_init_creates_birds(default_config):
    """PhysicsFlock(config) has N_active == config.num_boids."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    assert flock.N_active == 50


def test_flock_init_positions_in_domain(default_config):
    """All positions within domain bounds."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    pos = flock.positions[flock.active]
    assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= cfg.width).all()


def test_flock_init_velocities_nonzero(default_config):
    """All velocities have non-zero norm."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
    assert (speeds > 0).all()


def test_flock_init_accelerations_zero(default_config):
    """All accelerations are zero after init."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    assert (flock.accelerations == 0.0).all()


def test_flock_add_boids(default_config):
    """add_boids(5) increases N_active by 5."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    initial = flock.N_active
    flock.add_boids(5, cfg)
    assert flock.N_active == initial + 5


def test_flock_remove_boids(default_config):
    """remove_boids(5) decreases N_active by 5."""
    cfg = default_config
    cfg.num_boids = 50
    flock = PhysicsFlock(cfg)
    initial = flock.N_active
    removed = flock.remove_boids(5)
    assert removed == 5
    assert flock.N_active == initial - 5


def test_flock_remove_boids_deactivates(default_config):
    """Removed birds have active[i] = False."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    # Find an active bird
    active_idx = np.where(flock.active)[0]
    target = active_idx[-1]
    flock.remove_boids(1)
    assert not flock.active[target]


def test_flock_step_runs(default_config):
    """flock.step(config, dt) completes without error."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    _step_flock(flock, cfg, 1.0 / 60.0)
    assert flock.N_active == 10


def test_flock_step_positions_change(default_config):
    """Positions change after step() with non-zero forces."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    pos_before = flock.positions[flock.active].copy()
    _step_flock(flock, cfg, 1.0 / 60.0)
    pos_after = flock.positions[flock.active]
    assert not np.allclose(pos_before, pos_after)


def test_flock_add_boids_initializes(default_config):
    """Added birds have non-zero positions and velocities."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    added = flock.add_boids(5, cfg)
    assert added == 5
    # New birds are at the end of the active array
    active_idx = np.where(flock.active)[0]
    new_birds = active_idx[-5:]
    for i in new_birds:
        assert not np.allclose(flock.positions[i], 0.0)
        assert np.linalg.norm(flock.velocities[i]) > 0


def test_flock_add_beyond_capacity(default_config):
    """add_boids() extends arrays when all slots filled."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    cap_before = flock.N_capacity
    # Try to add more than capacity
    added = flock.add_boids(cap_before + 100, cfg)
    assert added > 0
    assert flock.N_capacity > cap_before


def test_flock_remove_all(default_config):
    """Removing all birds leaves N_active = 0."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    flock.remove_boids(20)
    assert flock.N_active == 0


def test_flock_seeded_reproducible(default_config):
    """Same seed + same config → identical flock state."""
    from copy import copy

    cfg1 = copy(default_config)
    cfg1.seed = 42
    cfg1.num_boids = 30
    flock1 = PhysicsFlock(cfg1)

    cfg2 = copy(default_config)
    cfg2.seed = 42
    cfg2.num_boids = 30
    flock2 = PhysicsFlock(cfg2)

    assert np.allclose(flock1.positions, flock2.positions)
    assert np.allclose(flock1.velocities, flock2.velocities)



# ── P0.4 Determinism Tests ─────────────────────────────────────


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


# ── P0.5 Smoothed Swarm Centre Tests ───────────────────────────


def test_center_initialised_none(default_config):
    """D1: flock.center is initialised to domain centre (not None).

    Before D1: center was None before any step (snap-to-centroid on frame 0).
    After D1:  center starts at (W/2, H/2, D/2) so sphere boundary is
               always centred on domain centre from frame 0.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    assert flock.center is not None, "D1: center initialised to domain centre"
    assert flock.center.shape == (3,), "center must be (3,) float32"
    assert flock.center.dtype == np.float32
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    np.testing.assert_array_equal(flock.center, C)


def test_center_set_after_first_step(default_config):
    """D1: flock.center is initialised to domain centre before first step,
    and EMA-drifts toward the centroid after step().

    Before D1: center snapped to centroid on frame 0 (None → centroid).
    After D1:  center starts at domain centre, then EMA blends toward
               centroid: center ← center + 0.5·(centroid − center).
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    C_initial = flock.center.copy()
    _step_flock(flock, cfg, 1.0 / 60.0)
    assert flock.center is not None, "center must be set after first step"
    assert flock.center.shape == (3,), "center must be (3,) float32"
    assert flock.center.dtype == np.float32
    # D1: After EMA, center moves toward centroid (not identity snap)
    # It should differ from initial domain centre after the first step
    assert not np.allclose(flock.center, C_initial), (
        "center should EMA-drift from domain centre toward centroid"
    )


def test_center_close_to_centroid(default_config):
    """D1: After first step, centre is halfway between domain centre and centroid.

    Before D1: center snapped to centroid (EMA init snap, exact match).
    After D1:  center starts at domain centre, EMA: center += 0.5·(centroid − center).
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    C_initial = flock.center.copy()
    _step_flock(flock, cfg, 1.0 / 60.0)

    centroid = flock.positions[flock.active].mean(axis=0)
    # D1: After EMA, center = (C_initial + centroid) / 2
    expected = (C_initial + centroid) / 2.0
    np.testing.assert_allclose(
        flock.center, expected, rtol=0.05, atol=5.0,
        err_msg="center should be halfway between domain centre and centroid"
    )


def test_center_ema_smoothing(default_config):
    """Teleport flock — centre moves exactly 50% toward centroid (EMA α=0.5).

    Uses update_center() directly to isolate EMA behaviour from physics.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    # Initialise centre
    flock.update_center()
    old_center = flock.center.copy()

    # Teleport all birds far away (no step, no physics)
    flock.positions += np.array([500.0, 0.0, 0.0], dtype=np.float32)

    # Update centre directly — pure EMA, no force/integrate
    flock.update_center()

    new_centroid = flock.positions[flock.active].mean(axis=0)
    distance_center_moved = np.linalg.norm(flock.center - old_center)
    distance_to_centroid = np.linalg.norm(new_centroid - old_center)

    # Centre should have moved toward the centroid
    assert distance_center_moved > 0, "center should move after teleport"

    # Centre should NOT have reached the centroid in one step (EMA lag)
    assert distance_center_moved < distance_to_centroid, (
        f"EMA lag: center moved {distance_center_moved:.1f}, "
        f"but centroid moved {distance_to_centroid:.1f}"
    )

    # EMA α=0.5 → centre moves exactly 50% of the way
    expected_move = 0.5 * distance_to_centroid
    assert np.isclose(distance_center_moved, expected_move, atol=1e-4), (
        f"EMA α=0.5: expected move ≈ {expected_move:.1f}, "
        f"got {distance_center_moved:.1f}"
    )


def test_center_converges(default_config):
    """Pure EMA: after teleport, centre converges to <1% of centroid in ~7 frames.

    Uses update_center() directly (no physics) so convergence follows
    error = D · (0.5)^n exactly.  With D=500, error < 5.0 after 7 frames.
    """
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    # Initialise centre
    flock.update_center()

    # Teleport — zero velocities so no physics drift
    flock.positions += np.array([500.0, 0.0, 0.0], dtype=np.float32)
    flock.velocities[:] = 0.0
    flock.accelerations[:] = 0.0

    centroid = flock.positions[flock.active].mean(axis=0)
    for i in range(20):  # noqa: B007
        flock.update_center()
        error = np.linalg.norm(flock.center - centroid)
        if error < 0.01 * np.linalg.norm(centroid):
            break

    assert i < 10, (
        f"Pure EMA should converge within 10 frames, took {i + 1}"
    )


def test_center_no_active_birds(default_config):
    """update_center() is a no-op when all birds are inactive."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    _step_flock(flock, cfg, 1.0 / 60.0)

    # Deactivate all birds
    flock.active[:] = False
    center_before = flock.center.copy()

    flock.update_center()

    np.testing.assert_array_equal(
        flock.center, center_before,
        err_msg="center must not change when no active birds"
    )


