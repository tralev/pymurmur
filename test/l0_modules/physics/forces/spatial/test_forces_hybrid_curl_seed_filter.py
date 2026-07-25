"""S2.B11 — shared curl-flow in SpatialMode, seed_sinusoidal noise mode; P4.1 hybrid filter unit tests; P4.10 numba kernel unit tests.

Split out of test_forces_hybrid.py (file-size split).
"""


import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock  # noqa: E402

# ── S2.B11: shared curl-flow in SpatialMode ─────────────────────────

def test_spatial_flow_weight_zero_is_baseline(default_config):
    """S2.B11: flow_weight=0 (default) → bit-identical to no-flow baseline."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.noise_scale = 0.0
    assert cfg.flow_weight == 0.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    acc_flow_zero = flock.accelerations.copy()

    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)
    object.__setattr__(cfg, 'flow_weight', 0.0)  # explicit, same as default
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng,
        flock2.last_theta, cfg,
    )
    np.testing.assert_array_equal(acc_flow_zero, flock2.accelerations)


def test_spatial_flow_weight_matches_shared_curl_flow_primitive(default_config):
    """S2.B11: SpatialMode's flow contribution equals curl_flow(...) * flow_weight * 0.22,
    the same L0 primitive FieldMode uses for its own curl-flow term."""
    from pymurmur.physics.forces._base import curl_flow
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42
    cfg.mode = "spatial"
    cfg.num_boids = 20
    cfg.noise_scale = 0.0
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.5

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    object.__setattr__(cfg, '_field_time', 1.25)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    active_idx = np.where(flock.active)[0]
    C = np.mean(flock.positions[active_idx], axis=0)
    U = 0.4 * min(cfg.width, cfg.height, cfg.depth)
    seeds = np.arange(len(active_idx), dtype=np.float32)
    expected_flow = curl_flow(
        flock.positions[active_idx], C, seeds, 1.25, U,
    ) * (cfg.flow_weight * 0.22) * cfg.acceleration_scale

    # With separation/alignment/cohesion all zeroed and no clamp triggered
    # (small magnitude), the total acceleration should equal the flow
    # contribution scaled by acceleration_scale (P4.2 pipeline step 2).
    np.testing.assert_allclose(
        flock.accelerations[active_idx], expected_flow, atol=1e-5,
    )


# ── S2.B11: seed_sinusoidal noise mode ──────────────────────────────

def test_spatial_seed_sinusoidal_deterministic_same_seeds_and_t(default_config):
    """S2.B11: seed_sinusoidal noise depends only on (seeds, t), not the
    rng stream — two identical runs at the same _field_time produce
    identical noise contributions even with different rng states."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 15
    cfg.noise_mode = "seed_sinusoidal"
    cfg.noise_scale = 0.18
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.0
    object.__setattr__(cfg, '_field_time', 2.5)

    flock1 = PhysicsFlock(cfg)
    flock1.rng = np.random.default_rng(1)  # different rng stream...
    flock1.accelerations[:] = 0.0
    flock1.get_index().rebuild(flock1.positions, flock1.active)

    flock2 = PhysicsFlock(cfg)
    flock2.positions[:] = flock1.positions
    flock2.velocities[:] = flock1.velocities
    flock2.rng = np.random.default_rng(999)  # ...shouldn't matter
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)

    SpatialMode.compute(
        flock1.positions, flock1.velocities, flock1.accelerations,
        flock1.active, flock1.get_index(), flock1.rng, flock1.last_theta, cfg,
    )
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng, flock2.last_theta, cfg,
    )
    np.testing.assert_array_equal(flock1.accelerations, flock2.accelerations)


def test_spatial_seed_sinusoidal_matches_seed_noise3_scaled(default_config):
    """S2.B11: seed_sinusoidal output equals seed_noise3(seeds, t) * (noise_scale/0.18)."""
    from pymurmur.core.types import seed_noise3
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 10
    cfg.noise_mode = "seed_sinusoidal"
    cfg.noise_scale = 0.36  # 2x the ±0.18 base range
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.0
    object.__setattr__(cfg, '_field_time', 4.0)

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    active_idx = np.where(flock.active)[0]
    seeds = np.arange(len(active_idx), dtype=np.float32)
    # Noise is added after the acceleration_scale step (same as "additive"
    # mode) — no extra scaling here.
    expected = seed_noise3(seeds, 4.0) * (cfg.noise_scale / 0.18)
    np.testing.assert_allclose(flock.accelerations[active_idx], expected, atol=1e-5)


def test_spatial_seed_sinusoidal_bounded_at_default_noise_scale(default_config):
    """S2.B11: at noise_scale=0.18 (the atom's native range), each axis of
    the per-bird noise contribution stays within ±0.18 (pre acceleration_scale)."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 40
    cfg.noise_mode = "seed_sinusoidal"
    cfg.noise_scale = 0.18
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.0
    cfg.acceleration_scale = 1.0
    object.__setattr__(cfg, '_field_time', 0.7)

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
    assert np.all(np.abs(flock.accelerations[flock.active]) <= 0.18 + 1e-5)


# ── P4.1: Hybrid filter unit tests ─────────────────────────

def test_hybrid_filter_caps_at_influence_count(default_config):
    """P4.1: After _query_neighbors, no bird has > influence_count neighbours."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 200
    cfg.influence_count = 5
    cfg.visual_range = 300.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        assert len(valid) <= cfg.influence_count, (
            f"Bird {i} has {len(valid)} neighbours, cap is {cfg.influence_count}"
        )


def test_hybrid_filter_visual_range_enforced(default_config):
    """P4.1: All accepted neighbours are within visual_range."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 100
    cfg.influence_count = 10
    cfg.visual_range = 80.0  # tight range

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        if len(valid) == 0:
            continue
        diffs = flock.positions[valid] - flock.positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        assert (dists <= cfg.visual_range + 1.0).all(), (
            f"Bird {i}: neighbour at dist {dists.max():.1f} > {cfg.visual_range}"
        )


def test_hybrid_filter_with_scattered_zeros(default_config):
    """P4.1+P4.10: Scattered zeros in neighbour array don't break filtering.

    This specifically tests the bug fixed in P4.10 where numba's `break`
    on encountering zero indices skipped valid neighbours."""
    from pymurmur.physics.forces._kernels import (
        _numba_hybrid_filter,
        _numpy_hybrid_filter,
    )

    N = 20
    k = 10
    positions = np.random.default_rng(42).uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)

    # Build neighbour arrays with scattered zeros: bird 0 has neighbours
    # [1, 0, 2, 0, 3, 4, 0, 5, 6, 0] — zeros interspersed with valid indices
    n_idx_numba = np.zeros((N, k), dtype=np.int32)
    n_idx_numpy = np.zeros((N, k), dtype=np.int32)
    scattered = [1, 0, 2, 0, 3, 4, 0, 5, 6, 0]
    n_idx_numba[0, :len(scattered)] = scattered
    n_idx_numpy[0, :len(scattered)] = scattered

    _numba_hybrid_filter(n_idx_numba, positions, active, visual_range=5000.0, influence_count=4)
    _numpy_hybrid_filter(n_idx_numpy, positions, active, visual_range=5000.0, influence_count=4)

    # Both must match and cap at 4
    assert np.array_equal(n_idx_numba, n_idx_numpy), \
        "numba and numpy hybrid filter must match with scattered zeros"
    valid = n_idx_numba[0][n_idx_numba[0] > 0]
    assert len(valid) <= 4, f"Scattered-zeros bird should have ≤4, got {len(valid)}"
    assert len(valid) > 0, "Should have at least some neighbours"


def test_hybrid_filter_empty_or_single(default_config):
    """P4.1: Zero or one active bird → no crash, empty neighbour array."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 1

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)
    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)
    assert n_idx.shape[1] == 0, f"Single bird should have 0-width neighbour array, got shape {n_idx.shape}"

    # Zero active
    cfg.num_boids = 10
    flock2 = PhysicsFlock(cfg)
    flock2.active[:] = False
    flock2.get_index().rebuild(flock2.positions, flock2.active)
    n_idx2 = _query_neighbors(flock2.positions, flock2.active, flock2.get_index(), cfg)
    assert n_idx2.shape[1] == 0, "Zero active should have 0-width neighbour array"


def test_influence_count_config_wired_to_filter(default_config):
    """P4.1: influence_count config field reaches the hybrid filter."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    for ic in [3, 7, 12]:
        cfg = default_config
        cfg.seed = 42  # D6: default seed is None — pin for determinism
        cfg.mode = "spatial"
        cfg.num_boids = 100
        cfg.influence_count = ic
        cfg.visual_range = 500.0

        flock = PhysicsFlock(cfg)
        flock.get_index().rebuild(flock.positions, flock.active)
        n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

        for i in range(cfg.num_boids):
            valid = n_idx[i][n_idx[i] > 0]
            assert len(valid) <= ic, (
                f"influence_count={ic}: bird {i} has {len(valid)} > {ic}"
            )


# ── P4.10: Numba kernel unit tests ─────────────────────────

def test_numba_numpy_hybrid_filter_equivalence():
    """P4.10: numba and numpy hybrid filter produce identical output."""
    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_hybrid_filter,
        _numpy_hybrid_filter,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    rng = np.random.default_rng(42)
    N, k = 50, 20
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)

    n_idx_numba = np.zeros((N, k), dtype=np.int32)
    n_idx_numpy = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        cands = np.array([j for j in range(N) if j != i], dtype=np.int32)
        chosen = rng.choice(cands, min(15, len(cands)), replace=False)
        n_idx_numba[i, :len(chosen)] = chosen
        n_idx_numpy[i, :len(chosen)] = chosen

    _numba_hybrid_filter(n_idx_numba, positions, active, visual_range=400.0, influence_count=6)
    _numpy_hybrid_filter(n_idx_numpy, positions, active, visual_range=400.0, influence_count=6)

    assert np.array_equal(n_idx_numba, n_idx_numpy), \
        "numba and numpy hybrid filter must be identical"


def test_numba_numpy_predator_detect_equivalence():
    """P4.10: numba and numpy predator detection produce identical output."""
    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_predator_detect,
        _numpy_predator_detect,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    rng = np.random.default_rng(42)
    N, k = 30, 10
    rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[0] = is_predator[3] = True  # two predators

    n_idx = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        if i in (0, 3):
            continue
        n_idx[i, 0] = 0  # bird 0 is a predator
        if i % 2 == 0:
            n_idx[i, 1] = 3  # bird 3 also predator for even-index birds

    threat_numba = np.zeros(N, dtype=bool)
    threat_numpy = np.zeros(N, dtype=bool)
    _numba_predator_detect(threat_numba, n_idx, is_predator, active)
    _numpy_predator_detect(threat_numpy, n_idx, is_predator, active)
    assert np.array_equal(threat_numba, threat_numpy)


def test_numba_numpy_predator_escape_equivalence():
    """P4.10: numba and numpy predator escape produce identical output."""
    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_predator_escape,
        _numpy_predator_escape,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    rng = np.random.default_rng(42)
    N, k = 20, 8
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    # Place prey right next to predator
    positions[1] = positions[0] + np.array([8.0, 0, 0], dtype=np.float32)
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[0] = True
    threatened = np.zeros(N, dtype=bool)
    threatened[1] = True

    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[1, 0] = 0  # bird 1 sees predator 0

    escape_factor, accel_boost = 1e6, 1.4
    esc_numba = np.zeros((N, 3), dtype=np.float32)
    esc_numpy = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(esc_numba, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost)
    _numpy_predator_escape(esc_numpy, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost)
    assert np.allclose(esc_numba, esc_numpy), \
        f"numba and numpy escape must match: {np.abs(esc_numba - esc_numpy).max():.6f}"


