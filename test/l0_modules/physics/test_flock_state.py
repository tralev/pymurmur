"""Unit tests for physics.flock — species column, prev-position/acceleration stash, per-bird max speed, D6 seed semantics, boundary radius factor.

Split out of test_flock.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock
from test.helpers import _step_flock  # noqa: E402 — shared test helper

# ── P0.6 Species Column Tests ───────────────────────────────────


def test_is_predator_all_false_initially(default_config):
    """All birds start as prey (is_predator all False)."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "is_predator"), "flock.is_predator must exist"
    assert flock.is_predator.dtype == bool
    assert not flock.is_predator.any(), "all birds must be prey initially"
    assert len(flock.is_predator) == flock.N_capacity


def test_add_boids_predator_flag(default_config):
    """add_boids(is_predator=True) marks new birds as predators."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    added = flock.add_boids(5, cfg, is_predator=True)
    assert added == 5
    # New birds are at the end of active
    active_idx = np.where(flock.active)[0]
    new_birds = active_idx[-5:]
    assert flock.is_predator[new_birds].all(), (
        "new birds must be predators"
    )
    # Original birds are still prey
    original = active_idx[:10]
    assert not flock.is_predator[original].any(), (
        "original birds must remain prey"
    )


def test_add_boids_prey_default(default_config):
    """add_boids() without is_predator defaults to prey (False)."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    added = flock.add_boids(5, cfg)  # no is_predator arg
    assert added == 5
    active_idx = np.where(flock.active)[0]
    new_birds = active_idx[-5:]
    assert not flock.is_predator[new_birds].any(), (
        "default add_boids must produce prey"
    )


def test_species_survives_add_remove(default_config):
    """is_predator flag persists after remove_boids (flags on inactive birds survive).

    Per P0.6 spec: add 5 predators → 5 total, remove 3 active from end →
    flags on those inactive indices persist. is_predator.sum() stays at 5.
    """
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Add predators
    flock.add_boids(5, cfg, is_predator=True)
    assert flock.is_predator.sum() == 5

    # Remove 3 birds (last active ones — predators, since added at end)
    removed = flock.remove_boids(3)
    assert removed == 3
    # Flags persist on all 5 predator slots (3 now inactive, 2 still active)
    assert flock.is_predator.sum() == 5, (
        f"is_predator flags must persist on inactive birds, got {flock.is_predator.sum()}"
    )
    # Only 2 predators remain active
    assert flock.is_predator[flock.active].sum() == 2


def test_species_carried_through_extend(default_config):
    """is_predator preserved when arrays grow via add_boids beyond capacity."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Mark first 3 birds as predators
    active_idx = np.where(flock.active)[0]
    flock.is_predator[active_idx[:3]] = True
    cap_before = flock.N_capacity

    # Add more birds to force extend
    added = flock.add_boids(cap_before + 50, cfg)
    assert added > 0
    assert flock.N_capacity > cap_before

    # First 3 birds should still be predators after extend
    assert flock.is_predator[active_idx[:3]].all(), (
        "predator flags lost after _extend()"
    )


def test_is_predator_inactive_preserved(default_config):
    """Inactive birds' is_predator flags are preserved."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Make the last 2 active birds predators (they'll be removed first)
    active_idx = np.where(flock.active)[0]
    flock.is_predator[active_idx[-2:]] = True
    assert flock.is_predator.sum() == 2

    # Remove 2 (deactivates last 2 active, which are the predators)
    flock.remove_boids(2)

    # The deactivated birds should still have is_predator=True
    # (they're inactive but their flags persist)
    inactive = np.where(~flock.active)[0]
    assert len(inactive) == 2
    assert flock.is_predator[inactive].all(), (
        "inactive predators should retain their flags"
    )
    # And is_predator.sum() still counts them
    assert flock.is_predator.sum() == 2


# ── P0.7 Prev Positions + Acceleration Stash Tests ─────────────


def test_prev_positions_initialised(default_config):
    """prev_positions is (N, 3) float32, initially all zeros."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "prev_positions"), "prev_positions must exist"
    assert flock.prev_positions.shape == (flock.N_capacity, 3)
    assert flock.prev_positions.dtype == np.float32
    assert (flock.prev_positions == 0.0).all()


def test_last_accelerations_initialised(default_config):
    """last_accelerations is (N, 3) float32, initially all zeros."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "last_accelerations"), "last_accelerations must exist"
    assert flock.last_accelerations.shape == (flock.N_capacity, 3)
    assert flock.last_accelerations.dtype == np.float32
    assert (flock.last_accelerations == 0.0).all()


def test_prev_positions_stashed_before_integrate(default_config):
    """After step(), prev_positions holds the pre-integration positions."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)

    pos_before_step = flock.positions.copy()
    _step_flock(flock, cfg, 1.0 / 60.0)

    # prev_positions should equal positions from before the step
    np.testing.assert_array_equal(
        flock.prev_positions, pos_before_step,
        err_msg="prev_positions must capture pre-integration positions"
    )
    # positions should have changed (integration moved birds)
    assert not np.array_equal(flock.positions, pos_before_step), (
        "positions should change after step"
    )


def test_last_accelerations_stashed_before_reset(default_config):
    """After step(), last_accelerations holds the force-computed accelerations."""
    cfg = default_config
    cfg.num_boids = 20
    flock = PhysicsFlock(cfg)
    _step_flock(flock, cfg, 1.0 / 60.0)

    # After step(), accelerations are reset to zero (integrate does this)
    assert (flock.accelerations[flock.active] == 0.0).all(), (
        "accelerations should be zero after integrate"
    )
    # But last_accelerations should hold the pre-reset values
    # At minimum, it should be non-zero for at least some birds (forces exist)
    assert not (flock.last_accelerations[flock.active] == 0.0).all(), (
        "last_accelerations must capture non-zero force accelerations"
    )


def test_stash_arrays_survive_extend(default_config):
    """prev_positions and last_accelerations preserved after _extend()."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Set known values
    flock.prev_positions[:] = np.arange(30, dtype=np.float32).reshape(10, 3)
    flock.last_accelerations[:] = np.arange(30, 60, dtype=np.float32).reshape(10, 3)
    cap_before = flock.N_capacity

    # Force extend
    flock.add_boids(cap_before + 50, cfg)

    # First 10 rows should be preserved
    expected_prev = np.arange(30, dtype=np.float32).reshape(10, 3)
    expected_acc = np.arange(30, 60, dtype=np.float32).reshape(10, 3)
    np.testing.assert_array_equal(flock.prev_positions[:10], expected_prev)
    np.testing.assert_array_equal(flock.last_accelerations[:10], expected_acc)


# ── P0.8 Per-Bird Max Speed Tests ──────────────────────────────


def test_max_speed_default_none(default_config):
    """max_speed is None by default (scalar v0 fallback)."""
    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    assert hasattr(flock, "max_speed"), "max_speed must exist"
    assert flock.max_speed is None, "max_speed must be None by default"


def test_last_accelerations_nonzero_after_forces(default_config):
    """P0.7: last_accelerations captures actual force data, not just zeros."""
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "spatial"
    flock = PhysicsFlock(cfg)
    _step_flock(flock, cfg, 1.0 / 60.0)
    # After a step with spatial forces, accelerations should have been
    # non-zero before being reset (captured in last_accelerations)
    acc_mags = np.linalg.norm(flock.last_accelerations[flock.active], axis=1)
    assert acc_mags.max() > 0, (
        f"last_accelerations should capture non-zero forces, got max={acc_mags.max():.6f}"
    )


def test_max_speed_per_bird_lowers_cap(default_config):
    """Setting per-bird max_speed lowers the speed cap for those birds."""
    cfg = default_config
    cfg.num_boids = 5
    cfg.mode = "projection"
    flock = PhysicsFlock(cfg)

    # Give every bird a tight speed cap
    flock.max_speed = np.full(flock.N_capacity, 1.0, dtype=np.float32)
    # Set velocities above the cap
    flock.velocities[:] = np.array([[3.0, 0.0, 0.0]] * 5, dtype=np.float32)
    flock.accelerations[:] = 0.0

    _step_flock(flock, cfg, 1.0 / 60.0)

    speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
    # All speeds should be clamped to max_speed=1.0, not v0=4.0
    assert (speeds <= 1.01).all(), f"speeds={speeds} should ≤ 1.0"
    assert (speeds >= 0.29).all(), f"speeds={speeds} should ≥ 0.3"


def test_max_speed_different_per_bird(default_config):
    """Each bird can have a different max_speed cap."""
    cfg = default_config
    cfg.num_boids = 3
    cfg.mode = "projection"
    flock = PhysicsFlock(cfg)

    # Bird 0: cap=2.0, Bird 1: cap=1.0, Bird 2: cap=3.0
    flock.max_speed = np.array([2.0, 1.0, 3.0], dtype=np.float32)
    # All start at high speed
    flock.velocities[:] = np.array([[5.0, 0.0, 0.0]] * 3, dtype=np.float32)
    flock.accelerations[:] = 0.0

    _step_flock(flock, cfg, 1.0 / 60.0)

    speeds = np.linalg.norm(flock.velocities, axis=1)
    assert speeds[0] <= 2.01, f"bird 0 speed={speeds[0]}"
    assert speeds[1] <= 1.01, f"bird 1 speed={speeds[1]}"
    assert speeds[2] <= 3.01, f"bird 2 speed={speeds[2]}"
    # Each should be at their cap (since starting speed 5 > all caps)
    assert np.isclose(speeds[0], 2.0, atol=0.05)
    assert np.isclose(speeds[1], 1.0, atol=0.05)
    assert np.isclose(speeds[2], 3.0, atol=0.05)


def test_add_boids_uses_flock_rng_deterministically(default_config):
    """P0.4: add_boids uses flock.rng — same RNG state → same positions."""
    cfg = default_config
    flock = PhysicsFlock(cfg)
    # Re-initialise with known seed
    flock.rng = np.random.default_rng(42)

    # Test 1: add_boids uses flock.rng for positions
    flock.rng = np.random.default_rng(42)
    flock.add_boids(5, cfg)
    pos1 = flock.positions[-5:].copy()

    flock.rng = np.random.default_rng(42)
    flock.add_boids(5, cfg)
    pos2 = flock.positions[-5:].copy()
    # After adding 5 more, positions should match (same RNG state before add)
    assert np.array_equal(pos1, pos2), "add_boids not using flock.rng deterministically"


def test_max_speed_none_uses_scalar_v0(default_config):
    """When max_speed is None, the scalar v0 from config is used."""
    cfg = default_config
    cfg.num_boids = 5
    cfg.mode = "projection"
    cfg.v0 = 3.0  # non-default v0
    flock = PhysicsFlock(cfg)

    # max_speed is None → should use config.v0 = 3.0
    assert flock.max_speed is None
    flock.velocities[:] = np.array([[8.0, 0.0, 0.0]] * 5, dtype=np.float32)
    flock.accelerations[:] = 0.0

    _step_flock(flock, cfg, 1.0 / 60.0)

    speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
    assert (speeds <= 3.01).all(), f"speeds={speeds} should ≤ 3.0 (cfg.v0)"


def test_max_speed_with_ceiling_mode(default_config):
    """P0.8+P0.9: per-bird max_speed works with ceiling speed mode."""
    cfg = default_config
    cfg.num_boids = 4
    flock = PhysicsFlock(cfg)
    # Different caps per bird — bird 2 (cap=5) stays, bird 0 (cap=2) clamped
    flock.max_speed = np.array([2.0, 3.0, 5.0, 4.0], dtype=np.float32)
    flock.velocities = np.array([[8.0, 0, 0]] * 4, dtype=np.float32)
    flock.accelerations[:] = 0.0

    from pymurmur.physics.boid import integrate
    integrate(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, cfg.width, cfg.height, cfg.depth,
        4.0, "toroidal", 1.0 / 60.0,
        max_speed=flock.max_speed, speed_mode="ceiling",
    )
    speeds = np.linalg.norm(flock.velocities, axis=1)
    # Ceiling: speeds > cap are clamped down to cap
    s0, s1, s2, s3 = float(speeds[0]), float(speeds[1]), float(speeds[2]), float(speeds[3])
    assert s0 == pytest.approx(2.0, abs=0.05), f"bird 0 cap=2: speed={s0:.4f}"
    assert s1 == pytest.approx(3.0, abs=0.05), f"bird 1 cap=3: speed={s1:.4f}"
    assert s2 == pytest.approx(5.0, abs=0.05), f"bird 2 cap=5: speed={s2:.4f}"
    assert s3 == pytest.approx(4.0, abs=0.05), f"bird 3 cap=4: speed={s3:.4f}"


# ── P0.4 Determinism — AST scan for module-level np.random ───────


def test_no_module_level_np_random():
    """P0.4: No module-level np.random.* calls remain in pymurmur/.

    Scans every .py file under pymurmur/ for bare `np.random.` calls
    (i.e. calls on the module-level RNG, not on a local Generator instance).
    P0.4 requires all stochastic sites to use flock.rng or a local Generator.
    """
    import ast
    from pathlib import Path

    violations = []
    pymurmur_root = Path("pymurmur")

    for py_file in sorted(pymurmur_root.rglob("*.py")):
        if py_file.name == "__init__.py" and py_file.stat().st_size < 10:
            continue
        tree = ast.parse(py_file.read_text())

        for node in ast.walk(tree):
            # Match: np.random.<method>(...) — bare np.random call
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == "np"
                        and node.func.value.attr == "random"):
                    # Exclude: np.random.default_rng(...) — that's allowed (creates Generator)
                    if node.func.attr != "default_rng":
                        violations.append(
                            f"{py_file}:{node.lineno}: np.random.{node.func.attr}(...)"
                        )

    assert not violations, (
        f"P0.4 violation: {len(violations)} module-level np.random.* call(s) found:\n"
        + "\n".join(violations)
        + "\n\nAll stochastic sites must use flock.rng (or a local Generator)."
    )


