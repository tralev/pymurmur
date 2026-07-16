"""P2.9 — Holey-mask contract tests.

Verifies that every registered force mode handles partially-active
(holey) flocks correctly:

1. No exceptions when computing forces on a holey flock
2. Inactive birds' positions, velocities, and accelerations are
   unchanged after force computation
3. Active birds still receive forces (the flock isn't dead)
4. Determinism holds — same seed + holey mask → identical result
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces import MODE_REGISTRY, compute_all_forces


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    """Minimal SimConfig — 30 birds, seed 42."""
    c = SimConfig()
    c.num_boids = 30
    c.seed = 42
    return c


@pytest.fixture
def holey_flock(cfg):
    """PhysicsFlock with ~33% of birds deactivated (10 of 30)."""
    flock = PhysicsFlock(cfg)
    # Deactivate birds 10–19 (holes in the middle — not contiguous)
    flock.active[5:10] = False
    flock.active[15:20] = False
    # Rebuild index so inactive birds are excluded
    idx = flock.get_index()
    if idx is not None:
        idx.rebuild(flock.positions, flock.active)
    return flock


# ── Helpers ────────────────────────────────────────────────────────

def _call_mode(mode_name: str, flock: PhysicsFlock, config: SimConfig) -> None:
    """Route forces through MODE_REGISTRY for the given mode."""
    config.mode = mode_name
    compute_all_forces(flock, config)


# ── Tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode_name", sorted(MODE_REGISTRY.keys()))
def test_holey_mask_no_exception(mode_name, holey_flock, cfg):
    """Every registered force mode completes without exception on a holey flock."""
    _call_mode(mode_name, holey_flock, cfg)


@pytest.mark.parametrize("mode_name", sorted(MODE_REGISTRY.keys()))
def test_holey_mask_inactive_positions_unchanged(mode_name, holey_flock, cfg):
    """Inactive bird positions are never modified by force computation."""
    pos_before = holey_flock.positions[~holey_flock.active].copy()
    _call_mode(mode_name, holey_flock, cfg)
    pos_after = holey_flock.positions[~holey_flock.active]
    np.testing.assert_array_equal(
        pos_before, pos_after,
        err_msg=f"{mode_name}: inactive positions changed",
    )


@pytest.mark.parametrize("mode_name", sorted(MODE_REGISTRY.keys()))
def test_holey_mask_inactive_velocities_unchanged(mode_name, holey_flock, cfg):
    """Inactive bird velocities are never modified by force computation."""
    vel_before = holey_flock.velocities[~holey_flock.active].copy()
    _call_mode(mode_name, holey_flock, cfg)
    vel_after = holey_flock.velocities[~holey_flock.active]
    np.testing.assert_array_equal(
        vel_before, vel_after,
        err_msg=f"{mode_name}: inactive velocities changed",
    )


@pytest.mark.parametrize("mode_name", sorted(MODE_REGISTRY.keys()))
def test_holey_mask_active_forces_applied(mode_name, holey_flock, cfg):
    """Active birds receive force contributions — the flock isn't dead."""
    holey_flock.accelerations[:] = 0.0
    _call_mode(mode_name, holey_flock, cfg)

    active_acc = holey_flock.accelerations[holey_flock.active]
    inactive_acc = holey_flock.accelerations[~holey_flock.active]

    # Inactive birds must have zero acceleration
    assert np.allclose(
        inactive_acc, 0.0,
    ), f"{mode_name}: inactive birds received acceleration"

    # Vicsek sets velocity directly — accelerations stay zero.
    # All other modes produce non-zero acceleration on active birds
    # when neighbours are present.
    if mode_name != "vicsek":
        acc_mags = np.linalg.norm(active_acc, axis=1)
        n_with_force = int((acc_mags > 1e-6).sum())
        n_active = holey_flock.active.sum()
        assert n_with_force > 0, (
            f"{mode_name}: 0/{n_active} active birds felt force"
        )


@pytest.mark.parametrize("mode_name", sorted(MODE_REGISTRY.keys()))
def test_holey_mask_deterministic(mode_name, cfg):
    """Same holey mask + same seed → identical forces (P0.4)."""
    cfg1 = SimConfig()
    cfg1.num_boids = 20
    cfg1.seed = 99
    cfg1.mode = mode_name
    flock1 = PhysicsFlock(cfg1)
    flock1.active[5:10] = False  # holey mask

    cfg2 = SimConfig()
    cfg2.num_boids = 20
    cfg2.seed = 99
    cfg2.mode = mode_name
    flock2 = PhysicsFlock(cfg2)
    flock2.active[5:10] = False

    # Rebuild indices
    for f in (flock1, flock2):
        idx = f.get_index()
        if idx is not None:
            idx.rebuild(f.positions, f.active)

    compute_all_forces(flock1, cfg1)
    compute_all_forces(flock2, cfg2)

    np.testing.assert_array_equal(
        flock1.accelerations, flock2.accelerations,
        err_msg=f"{mode_name}: holey-mask determinism broken",
    )


@pytest.mark.parametrize("mode_name", sorted(MODE_REGISTRY.keys()))
def test_holey_mask_20_steps_no_exception(mode_name, holey_flock, cfg):
    """20 consecutive force+integrate steps on a holey flock — no crash,
    and inactive birds stay frozen."""
    from pymurmur.simulation.engine import SimulationEngine
    cfg.mode = mode_name
    engine = SimulationEngine(cfg)
    # Apply the same holey mask
    engine.flock.active[:] = holey_flock.active

    pos_before = engine.flock.positions[~engine.flock.active].copy()
    vel_before = engine.flock.velocities[~engine.flock.active].copy()

    for _ in range(20):
        engine.step(1.0 / 60.0)

    assert engine.frame == 20

    # P2.9 contract: inactive birds must be bit-identical after integration
    np.testing.assert_array_equal(
        engine.flock.positions[~engine.flock.active], pos_before,
        err_msg=f"{mode_name}: inactive positions changed after 20 steps",
    )
    np.testing.assert_array_equal(
        engine.flock.velocities[~engine.flock.active], vel_before,
        err_msg=f"{mode_name}: inactive velocities changed after 20 steps",
    )
