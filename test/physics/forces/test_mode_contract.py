"""P2.2 — ForceMode ABC + MODE_REGISTRY contract tests.

Verifies the ForceMode protocol, @register decorator, and registry
are correct independent entities.
"""

import numpy as np
import pytest

from pymurmur.physics.forces._mode import ForceMode, MODE_REGISTRY, register
from pymurmur.physics.forces import compute_all_forces
from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock


# ── P2.2: ForceMode ABC ───────────────────────────────────────────

def test_force_mode_abc_cannot_instantiate():
    """ForceMode is abstract — cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ForceMode()  # abstract because compute() is @abstractmethod


def test_force_mode_has_needs_index():
    """ForceMode ABC defines needs_index class attribute."""
    assert hasattr(ForceMode, 'needs_index')
    assert ForceMode.needs_index is False


def test_force_mode_has_compute():
    """ForceMode ABC defines compute() abstract staticmethod."""
    assert hasattr(ForceMode, 'compute')
    assert callable(ForceMode.compute)


# ── P2.2: @register decorator ─────────────────────────────────────

def test_register_decorator_adds_to_registry():
    """@register(name) adds the class to MODE_REGISTRY."""
    old = dict(MODE_REGISTRY)  # snapshot

    @register("__test_mode")
    class _TestMode(ForceMode):
        needs_index = False

        @staticmethod
        def compute(positions, velocities, accelerations, active,
                    index, rng, last_theta, config):
            pass

    assert "__test_mode" in MODE_REGISTRY
    assert MODE_REGISTRY["__test_mode"] is _TestMode

    # Cleanup: remove test entry
    MODE_REGISTRY.pop("__test_mode", None)


def test_register_decorator_returns_class():
    """@register returns the class unchanged (for stacking)."""
    @register("__test_mode2")
    class _TestMode(ForceMode):
        needs_index = False

        @staticmethod
        def compute(positions, velocities, accelerations, active,
                    index, rng, last_theta, config):
            pass

    # Decorator should return the class itself
    assert isinstance(_TestMode, type)
    assert issubclass(_TestMode, ForceMode)

    MODE_REGISTRY.pop("__test_mode2", None)


# ── P2.2: MODE_REGISTRY completeness ──────────────────────────────

def test_mode_registry_has_all_five_modes():
    """MODE_REGISTRY contains exactly the 5 registered modes."""
    expected = {'projection', 'spatial', 'field', 'vicsek', 'influencer'}
    assert set(MODE_REGISTRY.keys()) == expected


def test_mode_registry_needs_index_correct():
    """MODE_REGISTRY needs_index flags match mode requirements."""
    needs = {
        'projection': True,
        'spatial': True,
        'field': False,
        'vicsek': True,
        'influencer': False,
    }
    for name, cls in MODE_REGISTRY.items():
        assert cls.needs_index == needs[name], (
            f"{name}: expected needs_index={needs[name]}, got {cls.needs_index}"
        )


def test_every_registered_mode_is_force_mode_subclass():
    """Every entry in MODE_REGISTRY is a ForceMode subclass."""
    for name, cls in MODE_REGISTRY.items():
        assert issubclass(cls, ForceMode), (
            f"{name}: {cls.__name__} is not a ForceMode subclass"
        )


# ── P2.2: compute_all_forces dispatch ─────────────────────────────

def test_compute_all_forces_unknown_mode_raises():
    """compute_all_forces raises ValueError for unknown mode."""
    cfg = SimConfig()
    cfg.num_boids = 5
    cfg.mode = "nonexistent"
    flock = PhysicsFlock(cfg)
    with pytest.raises(ValueError, match="Unknown force mode"):
        compute_all_forces(flock, cfg)


def test_compute_all_forces_all_modes_no_crash():
    """compute_all_forces doesn't crash for any registered mode."""
    for name in sorted(MODE_REGISTRY.keys()):
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.mode = name
        flock = PhysicsFlock(cfg)
        idx = flock.get_index()
        if idx is not None:
            idx.rebuild(flock.positions, flock.active)
        # Should not raise
        compute_all_forces(flock, cfg)


# ═══════════════════════════════════════════════════════════════════
# Per-mode independent tests — each ForceMode subclass tested directly
# (not via MODE_REGISTRY dispatch). Verifies class-level attributes,
# direct compute() calls, zero-active handling, and backward compat aliases.
# ═══════════════════════════════════════════════════════════════════

# ── Shared helpers ────────────────────────────────────────────────

def _make_test_flock(n=20, seed=42):
    """Create a small PhysicsFlock with spatial index rebuilt for modes
    that need it. Does not set mode (caller controls engine dispatch)."""
    cfg = SimConfig()
    cfg.num_boids = n
    # P2.1: seed is in flock sub-config
    cfg.flock.seed = seed
    flock = PhysicsFlock(cfg)
    idx = flock.get_index()
    if idx is not None:
        idx.rebuild(flock.positions, flock.active)
    return flock, cfg


def _assert_finite(flock, mode_name):
    """All active accelerations must be finite after compute."""
    active = flock.active
    assert np.isfinite(flock.accelerations[active]).all(), (
        f"{mode_name}: NaN/inf in accelerations after compute"
    )


# ── ProjectionMode ─────────────────────────────────────────────────

def test_projection_mode_direct_instantiation():
    """ProjectionMode class is importable and has correct attributes."""
    from pymurmur.physics.forces.projection import ProjectionMode
    assert issubclass(ProjectionMode, ForceMode)
    assert ProjectionMode.needs_index is True
    assert MODE_REGISTRY["projection"] is ProjectionMode


def test_projection_mode_compute_direct():
    """Call ProjectionMode.compute() directly — not via registry dispatch."""
    from pymurmur.physics.forces.projection import ProjectionMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "projection"

    # Before
    acc_before = flock.accelerations.copy()

    ProjectionMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    _assert_finite(flock, "ProjectionMode")
    # At least some birds should have felt forces
    assert not np.allclose(flock.accelerations[flock.active], acc_before[flock.active], atol=1e-8), (
        "ProjectionMode.compute produced no acceleration change"
    )


def test_projection_mode_zero_active():
    """ProjectionMode.compute handles zero active birds (early return)."""
    from pymurmur.physics.forces.projection import ProjectionMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "projection"
    flock.active[:] = False

    # Must not crash
    ProjectionMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    # Accelerations must be unchanged (no birds to compute for)
    assert (flock.accelerations == 0.0).all()


def test_projection_mode_backward_compat_alias():
    """projection_forces alias is ProjectionMode.compute with same needs_index."""
    from pymurmur.physics.forces.projection import projection_forces, ProjectionMode
    assert projection_forces is ProjectionMode.compute
    assert projection_forces.needs_index is True


# ── SpatialMode ────────────────────────────────────────────────────

def test_spatial_mode_direct_instantiation():
    """SpatialMode class is importable and has correct attributes."""
    from pymurmur.physics.forces.spatial import SpatialMode
    assert issubclass(SpatialMode, ForceMode)
    assert SpatialMode.needs_index is True
    assert MODE_REGISTRY["spatial"] is SpatialMode


def test_spatial_mode_compute_direct():
    """Call SpatialMode.compute() directly — not via registry dispatch."""
    from pymurmur.physics.forces.spatial import SpatialMode
    # Use n=50 so SpatialHashGrid has meaningful neighbour density
    flock, cfg = _make_test_flock(50)
    cfg.mode = "spatial"

    acc_before = flock.accelerations.copy()

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    _assert_finite(flock, "SpatialMode")
    assert not np.allclose(flock.accelerations[flock.active], acc_before[flock.active], atol=1e-8), (
        "SpatialMode.compute produced no acceleration change"
    )


def test_spatial_mode_zero_active():
    """SpatialMode.compute handles zero active birds (early return)."""
    from pymurmur.physics.forces.spatial import SpatialMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "spatial"
    flock.active[:] = False

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    assert (flock.accelerations == 0.0).all()


def test_spatial_mode_backward_compat_alias():
    """spatial_forces alias is SpatialMode.compute with same needs_index."""
    from pymurmur.physics.forces.spatial import spatial_forces, SpatialMode
    assert spatial_forces is SpatialMode.compute
    assert spatial_forces.needs_index is True


# ── FieldMode ──────────────────────────────────────────────────────

def test_field_mode_direct_instantiation():
    """FieldMode class is importable and has correct attributes."""
    from pymurmur.physics.forces.field import FieldMode
    assert issubclass(FieldMode, ForceMode)
    assert FieldMode.needs_index is False
    assert MODE_REGISTRY["field"] is FieldMode


def test_field_mode_compute_direct():
    """Call FieldMode.compute() directly — not via registry dispatch."""
    from pymurmur.physics.forces.field import FieldMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "field"
    # Field mode doesn't need index — pass None

    acc_before = flock.accelerations.copy()

    FieldMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, None, flock.rng,
        flock.last_theta, cfg,
    )

    _assert_finite(flock, "FieldMode")
    assert not np.allclose(flock.accelerations[flock.active], acc_before[flock.active], atol=1e-8), (
        "FieldMode.compute produced no acceleration change"
    )


def test_field_mode_zero_active():
    """FieldMode.compute handles zero active birds (early return)."""
    from pymurmur.physics.forces.field import FieldMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "field"
    flock.active[:] = False

    FieldMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, None, flock.rng,
        flock.last_theta, cfg,
    )
    assert (flock.accelerations == 0.0).all()


def test_field_mode_backward_compat_alias():
    """field_forces alias is FieldMode.compute with same needs_index."""
    from pymurmur.physics.forces.field import field_forces, FieldMode
    assert field_forces is FieldMode.compute
    assert field_forces.needs_index is False


# ── VicsekMode ─────────────────────────────────────────────────────

def test_vicsek_mode_direct_instantiation():
    """VicsekMode class is importable and has correct attributes."""
    from pymurmur.physics.forces.vicsek import VicsekMode
    assert issubclass(VicsekMode, ForceMode)
    assert VicsekMode.needs_index is True
    assert MODE_REGISTRY["vicsek"] is VicsekMode


def test_vicsek_mode_compute_direct():
    """Call VicsekMode.compute() directly — not via registry dispatch.
    Vicsek mode sets velocities directly rather than accumulating
    accelerations."""
    from pymurmur.physics.forces.vicsek import VicsekMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "vicsek"

    vel_before = flock.velocities.copy()

    VicsekMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    # Vicsek sets velocities to v0=1.0 (constant speed)
    active_vel = flock.velocities[flock.active]
    speeds = np.linalg.norm(active_vel, axis=1)
    assert np.allclose(speeds, cfg.vicsek_velocity, atol=0.05), (
        f"Vicsek: speeds must be ~v0=1.0, got {speeds[:5]}"
    )
    # Velocities should have changed (memory+noise modifies direction)
    assert not np.allclose(flock.velocities[flock.active], vel_before[flock.active], atol=1e-6), (
        "VicsekMode.compute produced no velocity change"
    )


def test_vicsek_mode_zero_active():
    """VicsekMode.compute handles zero active birds (early return)."""
    from pymurmur.physics.forces.vicsek import VicsekMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "vicsek"
    vel_before = flock.velocities.copy()
    flock.active[:] = False

    VicsekMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    # Velocities unchanged (no active birds)
    assert np.array_equal(flock.velocities, vel_before)


def test_vicsek_mode_backward_compat_alias():
    """vicsek_forces alias is VicsekMode.compute with same needs_index."""
    from pymurmur.physics.forces.vicsek import vicsek_forces, VicsekMode
    assert vicsek_forces is VicsekMode.compute
    assert vicsek_forces.needs_index is True


# ── InfluencerMode ─────────────────────────────────────────────────

def test_influencer_mode_direct_instantiation():
    """InfluencerMode class is importable and has correct attributes."""
    from pymurmur.physics.forces.influencer import InfluencerMode
    assert issubclass(InfluencerMode, ForceMode)
    assert InfluencerMode.needs_index is False
    assert MODE_REGISTRY["influencer"] is InfluencerMode


def test_influencer_mode_compute_direct():
    """Call InfluencerMode.compute() directly — not via registry dispatch."""
    from pymurmur.physics.forces.influencer import InfluencerMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "influencer"

    acc_before = flock.accelerations.copy()

    InfluencerMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, None, flock.rng,
        flock.last_theta, cfg,
    )

    _assert_finite(flock, "InfluencerMode")
    assert not np.allclose(flock.accelerations[flock.active], acc_before[flock.active], atol=1e-8), (
        "InfluencerMode.compute produced no acceleration change"
    )


def test_influencer_mode_zero_active():
    """InfluencerMode.compute handles zero active birds (early return)."""
    from pymurmur.physics.forces.influencer import InfluencerMode
    flock, cfg = _make_test_flock(20)
    cfg.mode = "influencer"
    flock.active[:] = False

    InfluencerMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, None, flock.rng,
        flock.last_theta, cfg,
    )
    assert (flock.accelerations == 0.0).all()


def test_influencer_mode_backward_compat_alias():
    """influencer_forces alias is InfluencerMode.compute with same needs_index."""
    from pymurmur.physics.forces.influencer import influencer_forces, InfluencerMode
    assert influencer_forces is InfluencerMode.compute
    assert influencer_forces.needs_index is False
