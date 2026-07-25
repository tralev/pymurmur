"""S2.B3 — minimum-image (toroidal) predator escape + numba/numpy equivalence; P4.2/P4.3/P4.4 full-pipeline integration tests.

Split out of test_forces_hybrid.py (file-size split).
"""


import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock  # noqa: E402

# ── S2.B3: minimum-image (toroidal) predator escape ─────────────────

def test_predator_escape_min_image_across_wrap_boundary():
    """S2.B3: a predator just across a toroidal wrap boundary must be
    seen as adjacent (short escape distance), not as ~domain-width away."""
    from pymurmur.physics.forces._kernels import _numpy_predator_escape

    box_size = 1000.0
    box = np.array([box_size, box_size, box_size], dtype=np.float32)
    N, k = 2, 4
    positions = np.zeros((N, 3), dtype=np.float32)
    # Prey near the low edge, predator near the high edge -- raw distance
    # ~box_size-10, min-image distance ~10.
    positions[0] = [5.0, 500.0, 500.0]
    positions[1] = [box_size - 5.0, 500.0, 500.0]
    active = np.ones(N, dtype=bool)
    is_predator = np.array([False, True])
    threatened = np.array([True, False])
    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[0, 0] = 1  # prey 0 sees predator 1

    escape_factor, accel_boost = 1e6, 1.4

    esc_wrapped = np.zeros((N, 3), dtype=np.float32)
    _numpy_predator_escape(esc_wrapped, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost, box)

    esc_unwrapped = np.zeros((N, 3), dtype=np.float32)
    _numpy_predator_escape(esc_unwrapped, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost)

    mag_wrapped = np.linalg.norm(esc_wrapped[0])
    mag_unwrapped = np.linalg.norm(esc_unwrapped[0])
    # Min-image distance (~10) gives a much stronger 1/d^2 force than the
    # raw cross-domain distance (~990).
    assert mag_wrapped > mag_unwrapped * 100, (
        f"wrapped={mag_wrapped:.4f} unwrapped={mag_unwrapped:.4f}"
    )
    # Direction should point from the predator toward the near edge (+x),
    # i.e. away across the wrap seam, not across the whole domain (-x).
    assert esc_wrapped[0][0] > 0, f"expected +x escape, got {esc_wrapped[0]}"
    assert esc_unwrapped[0][0] < 0, f"expected -x escape without wrap, got {esc_unwrapped[0]}"


def test_predator_escape_numba_numpy_min_image_equivalence():
    """S2.B3: numba and numpy min-image escape agree."""
    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_predator_escape,
        _numpy_predator_escape,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    box = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
    N, k = 2, 4
    positions = np.zeros((N, 3), dtype=np.float32)
    positions[0] = [5.0, 500.0, 500.0]
    positions[1] = [995.0, 500.0, 500.0]
    active = np.ones(N, dtype=bool)
    is_predator = np.array([False, True])
    threatened = np.array([True, False])
    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[0, 0] = 1

    escape_factor, accel_boost = 1e6, 1.4
    esc_numba = np.zeros((N, 3), dtype=np.float32)
    esc_numpy = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(esc_numba, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost, box)
    _numpy_predator_escape(esc_numpy, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost, box)
    assert np.allclose(esc_numba, esc_numpy, atol=1e-3), (
        f"numba/numpy min-image escape mismatch: {np.abs(esc_numba - esc_numpy).max():.6f}"
    )


def test_numba_predator_detect_excludes_predators():
    """P4.10: Predators are never marked as threatened."""
    from pymurmur.physics.forces._kernels import _HAS_NUMBA, _numba_predator_detect
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 20, 8
    np.random.default_rng(42).uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[0] = True
    is_predator[1] = True  # two predators, each sees the other

    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[0, 0] = 1  # predator 0 sees predator 1
    n_idx[1, 0] = 0  # predator 1 sees predator 0

    threatened = np.zeros(N, dtype=bool)
    _numba_predator_detect(threatened, n_idx, is_predator, active)

    assert not threatened[0], "Predator 0 should not be threatened"
    assert not threatened[1], "Predator 1 should not be threatened"


def test_numba_predator_escape_direction():
    """P4.10: Escape force points away from nearest predator.

    Uses bird index 2 as predator (not 0) to avoid ambiguity with
    the zero-padding sentinel in neighbour arrays."""
    from pymurmur.physics.forces._kernels import _HAS_NUMBA, _numba_predator_escape
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 10, 5
    positions = np.random.default_rng(42).uniform(0, 100, (N, 3)).astype(np.float32)
    # Predator at (50, 50, 50), prey at (60, 50, 50) — 10 units +x
    positions[2] = [50.0, 50.0, 50.0]
    positions[3] = [60.0, 50.0, 50.0]
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[2] = True  # bird 2 is predator (non-zero index)
    threatened = np.zeros(N, dtype=bool)
    threatened[3] = True   # bird 3 is threatened prey

    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[3, 0] = 2  # bird 3 sees predator 2

    escape = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(escape, positions, n_idx, is_predator,
                            threatened, active, escape_factor=1e6, accel_boost=1.0)

    # Escape should push prey (+x) away from predator at x=50
    assert escape[3, 0] > 0, f"Escape should push +x, got {escape[3]}"
    assert abs(escape[3, 1]) < 1e-6 and abs(escape[3, 2]) < 1e-6, \
        "Escape should have no y/z component for collinear predator-prey"


def test_numba_predator_escape_scattered_zeros():
    """P4.10: Escape handles scattered zeros in neighbour array.

    Regression test for the break→continue bug — scattered zeros
    shouldn't prevent finding a predator further in the list."""
    from pymurmur.physics.forces._kernels import _HAS_NUMBA, _numba_predator_escape
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 10, 8
    positions = np.random.default_rng(42).uniform(0, 100, (N, 3)).astype(np.float32)
    positions[5] = [30.0, 40.0, 50.0]  # predator
    positions[7] = [38.0, 40.0, 50.0]  # prey, 8 units +x from predator
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[5] = True
    threatened = np.zeros(N, dtype=bool)
    threatened[7] = True

    # Scattered zeros: [0, 5, 0, 0, ...] — zero at position 0, predator at position 1
    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[7, 0] = 0   # zero sentinel (should be skipped)
    n_idx[7, 1] = 5   # predator (should be found)
    n_idx[7, 2] = 0   # zero sentinel
    n_idx[7, 3] = 1   # regular bird (not predator, should be skipped)

    escape = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(escape, positions, n_idx, is_predator,
                            threatened, active, escape_factor=1e6, accel_boost=1.0)

    # Must find predator 5 through the scattered zeros
    assert escape[7, 0] > 0, (
        f"Escape should push +x (predator at x=30, prey at x=38), got {escape[7]}"
    )


# ═══════════════════════════════════════════════════════════════════
# P4.1 + P4.10 Integration tests — full pipeline as a whole
# ═══════════════════════════════════════════════════════════════════

def test_p42_accel_scale_zero_produces_zero_forces(default_config):
    """P4.2: accel_scale=0 multiplies all accumulated forces to zero.

    With non-zero weights and neighbours, forces accumulate normally.
    But acceleration_scale=0 should zero everything (before noise).
    Setting noise_scale=0 too ensures final forces are exactly zero."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 10.0
    cfg.acceleration_scale = 0.0  # ← zeroes all accumulated forces

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    # All forces should be zero (accumulated forces × 0 + noise=0)
    assert np.allclose(flock.accelerations[flock.active], 0.0), (
        f"accel_scale=0 should zero all forces, got max={np.abs(flock.accelerations).max():.6f}"
    )


def test_full_pipeline_predator_with_hybrid_filter(default_config):
    """P4.1+P4.10: Predator escape + hybrid filter work together end-to-end.

    Sets up a predator, places prey nearby, runs SpatialMode.compute
    (which exercises: _query_neighbors → hybrid filter → force primitives
    → predator detect → predator escape → accumulate → clamp → noise).
    Verifies the prey gets escape force and alignment/cohesion are zeroed."""
    from pymurmur.physics.forces.spatial import SpatialMode, _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.visual_range = 500.0
    cfg.influence_count = 10
    cfg.noise_scale = 0.0
    cfg.max_force = 10.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.is_predator[0] = True
    # Place prey bird close to predator
    flock.positions[1] = flock.positions[0] + np.array([15.0, 0, 0], dtype=np.float32)
    flock.get_index().rebuild(flock.positions, flock.active)
    object.__setattr__(cfg, '_is_predator', flock.is_predator)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    # Prey (bird 1) should have escape force pointing +x
    assert flock.accelerations[1, 0] > 0, (
        f"Prey should get escape force +x, got {flock.accelerations[1]}"
    )
    # Predator should NOT get escape force
    assert not np.allclose(flock.accelerations[0], 0.0), \
        "Predator should still get flocking forces"

    # Verify hybrid filter capped neighbors
    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)
    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        assert len(valid) <= cfg.influence_count, \
            f"Bird {i}: {len(valid)} > {cfg.influence_count}"


def test_full_pipeline_determinism(default_config):
    """P4.1+P4.10: Same seed + same config → bit-identical forces.

    The numba kernels (with seeded RNG) must produce deterministic
    output. Runs two identical flocks through SpatialMode.compute."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.noise_scale = 0.1  # non-zero noise exercises RNG determinism

    flock1 = PhysicsFlock(cfg)
    flock1.accelerations[:] = 0.0
    flock1.get_index().rebuild(flock1.positions, flock1.active)

    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)

    SpatialMode.compute(
        flock1.positions, flock1.velocities, flock1.accelerations,
        flock1.active, flock1.get_index(), flock1.rng,
        flock1.last_theta, cfg,
    )
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng,
        flock2.last_theta, cfg,
    )

    assert np.array_equal(flock1.accelerations, flock2.accelerations), \
        "Identical seed+config must produce bit-identical forces"


def test_full_pipeline_influence_count_affects_forces(default_config):
    """P4.1+P4.10: Different influence_count → different force distribution.

    A tight cap (3) vs loose cap (20) should produce measurably
    different mean force magnitudes because more/fewer neighbours
    contribute to cohesion and alignment."""
    from pymurmur.physics.forces.spatial import SpatialMode

    forces = {}
    for ic in [3, 20]:
        cfg = default_config
        cfg.seed = 42  # D6: default seed is None — pin for determinism
        cfg.mode = "spatial"
        cfg.num_boids = 100
        cfg.influence_count = ic
        cfg.visual_range = 500.0
        cfg.noise_scale = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        SpatialMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        forces[ic] = float(mags.mean())

    # Different caps should produce meaningfully different force distributions.
    # Fewer neighbours → less averaging → more variable forces.
    # More neighbours → more averaging → smoother, more clustered forces.
    # The mean magnitudes can go either way, but the distributions must differ.
    ratio = max(forces[3], forces[20]) / min(forces[3], forces[20])
    assert abs(ratio - 1.0) > 0.001, (
        f"ic=3 mean={forces[3]:.4f}, ic=20 mean={forces[20]:.4f} — "
        f"different caps must produce measurably different forces (ratio={ratio:.4f})"
    )


def test_full_pipeline_clamp_then_noise(default_config):
    """P4.1+P4.2+P4.10: Post-clamp noise addition works through full pipeline.

    Noise is added AFTER the max_force clamp (P4.2). Verifies that
    with noise_scale > 0, some forces exceed max_force (noise is
    unclamped), but forces without noise stay ≤ max_force."""
    from pymurmur.physics.forces.spatial import SpatialMode

    # Without noise: all forces ≤ max_force
    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.noise_scale = 0.0
    cfg.max_force = 0.5

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    mags_no_noise = np.linalg.norm(flock.accelerations[flock.active], axis=1)
    assert mags_no_noise.max() <= cfg.max_force + 1e-4, \
        f"Without noise: max={mags_no_noise.max():.4f} > {cfg.max_force}"

    # With noise: some forces may exceed max_force
    cfg.noise_scale = 0.3
    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)

    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng,
        flock2.last_theta, cfg,
    )
    mags_noise = np.linalg.norm(flock2.accelerations[flock2.active], axis=1)
    # Noise increases mean force (added post-clamp)
    assert mags_noise.mean() > mags_no_noise.mean(), \
        f"Noise should increase mean force: {mags_no_noise.mean():.4f} → {mags_noise.mean():.4f}"


def test_full_pipeline_all_phase4_features(default_config):
    """P4.1+P4.2+P4.3+P4.5+P4.8+P4.10: All spatial features work together.

    Exercises the complete spatial pipeline with:
    - P4.1: hybrid filter (influence_count=7)
    - P4.2: accumulate → accel_scale → clamp → noise
    - P4.3: predator escape + zeroed align/coh
    - P4.5: jitter on sep/coh/align
    - P4.8: coherence gate (via _coherence_factor)
    - P4.10: numba-accelerated kernels"""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 60
    cfg.influence_count = 7
    cfg.visual_range = 300.0
    cfg.separation_weight = 4.5
    cfg.alignment_weight = 0.65
    cfg.cohesion_weight = 0.75
    cfg.noise_scale = 0.1
    cfg.max_force = 5.0
    cfg.acceleration_scale = 0.3
    cfg.jitter_separation = 0.1
    cfg.jitter_cohesion = 0.1
    cfg.jitter_alignment = 0.1

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.is_predator[0] = True
    # Place prey near predator
    flock.positions[1] = flock.positions[0] + np.array([12.0, 0, 0], dtype=np.float32)
    flock.get_index().rebuild(flock.positions, flock.active)
    object.__setattr__(cfg, '_is_predator', flock.is_predator)
    object.__setattr__(cfg, '_coherence_factor', 0.8)  # P4.8 gate

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)

    # Basic sanity: forces finite, non-zero, reasonable range
    assert np.isfinite(flock.accelerations).all(), "NaN/Inf in forces"
    assert mags.mean() > 0, "Forces should be non-zero"
    assert mags.max() < 20.0, f"Forces unexpectedly large: max={mags.max():.1f}"

    # Prey should have escape component (P4.3)
    assert flock.accelerations[1, 0] > 0, \
        f"Prey escape +x expected, got {flock.accelerations[1]}"

    # Predator should have flocking forces (not escape)
    assert not np.allclose(flock.accelerations[0], 0.0), \
        "Predator should get flocking forces"

    # Jitter should increase force variance vs no-jitter run (P4.5)
    cfg_no_jit = default_config
    for attr in ('mode', 'num_boids', 'influence_count', 'visual_range',
                 'separation_weight', 'alignment_weight', 'cohesion_weight',
                 'noise_scale', 'max_force', 'acceleration_scale'):
        setattr(cfg_no_jit, attr, getattr(cfg, attr))
    cfg_no_jit.jitter_separation = 0.0
    cfg_no_jit.jitter_cohesion = 0.0
    cfg_no_jit.jitter_alignment = 0.0

    flock_nj = PhysicsFlock(cfg_no_jit)
    flock_nj.accelerations[:] = 0.0
    flock_nj.get_index().rebuild(flock_nj.positions, flock_nj.active)

    SpatialMode.compute(
        flock_nj.positions, flock_nj.velocities, flock_nj.accelerations,
        flock_nj.active, flock_nj.get_index(), flock_nj.rng,
        flock_nj.last_theta, cfg_no_jit,
    )
    mags_nj = np.linalg.norm(flock_nj.accelerations[flock_nj.active], axis=1)

    # Jitter should increase force spread (different weights per frame)
    assert mags.std() > mags_nj.std() * 1.05, (
        f"Jitter should increase force variance: "
        f"jitter std={mags.std():.4f}, no-jitter std={mags_nj.std():.4f}"
    )


