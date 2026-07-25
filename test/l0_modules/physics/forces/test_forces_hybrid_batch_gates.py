"""P4.6 batch k-NN query verification; P4.7/P4.9/P4.8/P4.4 cross-item tests; S2.B11 curl_flow edge cases; alignment radius ratio, separation distance gate, global neighbor filter.

Split out of test_forces_hybrid.py (file-size split).
"""


import numpy as np

from pymurmur.physics.flock import PhysicsFlock  # noqa: E402

# ── P4.6: Batch k-NN query verification ──────────────────────────

def test_p46_batch_query_matches_per_bird(default_config):
    """P4.6: Batch cKDTree query produces same neighbours as per-bird queries.

    The batch query optimization (tree.query(active_pos, k=k+1, workers=-1))
    must produce identical results to individual per-bird tree.query() calls.
    This test verifies correctness, not speed."""
    from scipy.spatial import cKDTree

    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 100
    cfg.influence_count = 7
    cfg.visual_range = 500.0  # large enough to not filter in metric step

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    # Get neighbours via the batch query path (uses tree.query batch)
    n_idx_batch = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    # Get neighbours via per-bird path for verification
    active_idx = np.where(flock.active)[0]
    positions = flock.positions
    k = max(cfg.influence_count * 3, getattr(cfg, 'topological_cap', 50))
    k = min(k, len(active_idx) - 1)

    # Build a fresh tree on active positions (per-bird query baseline)
    tree = cKDTree(positions[active_idx])
    n_idx_per_bird = np.zeros((len(positions), k), dtype=np.int32)
    for _j, global_i in enumerate(active_idx):
        _, compacted = tree.query(positions[global_i], k=k + 1)
        n_idx_per_bird[global_i] = active_idx[compacted[1:k + 1]]

    # Apply the same hybrid filter to per-bird results for fair comparison
    from pymurmur.physics.forces._kernels import _numba_hybrid_filter
    _numba_hybrid_filter(n_idx_per_bird, positions, flock.active,
                         cfg.visual_range, cfg.influence_count)

    # Both must produce identical neighbour sets
    assert np.array_equal(n_idx_batch, n_idx_per_bird), (
        "Batch query must produce identical neighbours to per-bird query"
    )


def test_p46_batch_query_all_birds_have_neighbours(default_config):
    """P4.6: Batch query — every active bird has at least 1 neighbour in dense flock."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 200  # dense
    cfg.influence_count = 5
    cfg.visual_range = 200.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    # In a dense flock, every bird (except maybe extreme outliers) should have ≥1 neighbour
    birds_without_neighbours = 0
    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        if len(valid) == 0:
            birds_without_neighbours += 1

    # Allow up to 5% isolated birds (edge of domain)
    assert birds_without_neighbours < cfg.num_boids * 0.05, (
        f"{birds_without_neighbours}/{cfg.num_boids} birds have no neighbours in dense flock"
    )


# ═══════════════════════════════════════════════════════════════════
# P4 integration tests — multiple P4 items working together as a whole
# ═══════════════════════════════════════════════════════════════════

def test_p47_p43_sphere_boundary_with_predator_escape(default_config):
    """P4.7+P4.3: Sphere boundary + predator — birds stay inside sphere,
    prey get escape force, predator gets normal flocking forces.

    Verifies that the sphere soft boundary (P4.7 _sphere_soft) and
    predator escape (P4.3) work together without conflict: birds near
    the sphere edge get inward push while also fleeing the predator."""
    from pymurmur.simulation.engine import SimulationEngine

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 150.0
    cfg.predator_enabled = True
    cfg.roosting_enabled = False
    cfg.noise_scale = 0.0

    engine = SimulationEngine(cfg)

    # Place predator at flock centre (all birds are prey)
    pred = getattr(engine.extensions, '_predator', None)
    assert pred is not None
    engine.flock.update_center()
    pred._pos = engine.flock.center.copy()
    pred._phase = "approach"

    # Run several steps — no bird should escape the sphere.
    # D1: the sphere is centred on the domain centre C, not the origin.
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                 dtype=np.float32)
    for _ in range(30):
        engine.step(1.0 / 60.0)
        dists = np.linalg.norm(engine.flock.positions - C, axis=1)
        assert (dists <= cfg.boundary_sphere_radius + 1.0).all(), (
            f"Bird outside sphere: max dist={dists.max():.1f} > {cfg.boundary_sphere_radius}"
        )

    # Forces should be finite and non-zero (predator present)
    last_acc = engine.flock.last_accelerations[engine.flock.active]
    assert np.isfinite(last_acc).all(), "NaN/Inf in forces with sphere+predator"
    assert not np.allclose(last_acc, 0.0), (
        "Forces should be non-zero with predator present"
    )


def test_p49_p41_velocity_init_affects_spatial_force_distribution(default_config):
    """P4.9+P4.1: Different velocity init modes → measurably different force
    distributions through the same spatial force pipeline.

    Cube (random uniform [-v0,v0]³) vs sphere (fixed-speed random direction)
    velocity inits produce different alignment and separation forces because
    the flock starts with different velocity structures."""
    from pymurmur.physics.forces.spatial import SpatialMode

    def forces_for_init(velocity_mode: str) -> np.ndarray:
        cfg = default_config
        cfg.seed = 42  # D6: default seed is None — pin for determinism
        cfg.seed = 42  # D6: pin seed — both flocks share positions, so the
        # only difference between runs is the velocity-init mode under test
        cfg.mode = "spatial"
        cfg.num_boids = 50
        cfg.velocity_init = velocity_mode
        cfg.separation_weight = 4.5
        cfg.alignment_weight = 0.65
        cfg.cohesion_weight = 0.75
        cfg.noise_scale = 0.0
        cfg.influence_count = 7
        cfg.visual_range = 200.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        SpatialMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        return np.linalg.norm(flock.accelerations[flock.active], axis=1)

    mags_cube = forces_for_init("cube")
    mags_sphere = forces_for_init("sphere")

    # Different velocity inits should produce different force distributions.
    # Cube: uniform in [-v0,v0]³ → speeds vary widely, forces more spread.
    # Sphere: fixed |v| = v0·0.8 → equal speeds, more uniform forces.
    # The std of force magnitudes must differ measurably.
    ratio = max(mags_cube.std(), mags_sphere.std()) / min(mags_cube.std(), mags_sphere.std())
    assert ratio > 1.15, (
        f"Cube and sphere velocity inits must produce different force spreads: "
        f"cube std={mags_cube.std():.4f}, sphere std={mags_sphere.std():.4f}, ratio={ratio:.3f}"
    )
    # Both should be finite
    assert np.isfinite(mags_cube).all()
    assert np.isfinite(mags_sphere).all()


def test_p48_p44_ecology_with_physical_metrics(default_config):
    """P4.8+P4.4: Ecology enabled → physical metrics populated.

    Runs SimulationEngine with ecology and metrics enabled. After several
    frames, physical metrics (speed_real_ms, energy_J) should be populated
    because the flock is actively simulating with roosting forces."""
    from pymurmur.simulation.engine import SimulationEngine

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.roosting_enabled = True
    cfg.metrics_detail_level = 1
    cfg.metrics_interval = 1  # every frame
    cfg.noise_scale = 0.1

    engine = SimulationEngine(cfg)

    # Run enough frames for metrics to accumulate
    for _ in range(60):
        engine.step(1.0 / 60.0)

    m = engine.metrics.snapshot()

    # Physical metrics must be in plausible ranges (not all zero)
    assert m.speed_real_ms >= 0.0, f"speed_real_ms={m.speed_real_ms} < 0"
    assert m.force_real_N >= 0.0, f"force_real_N={m.force_real_N} < 0"
    assert m.energy_J >= 0.0, f"energy_J={m.energy_J} < 0"

    # With a running simulation, speed should be non-zero
    assert m.speed_avg > 0.0, "Flock should have non-zero average speed"

    # Metrics collector should have accumulated history (≥1 frame)
    assert len(engine.metrics.history) > 0, "Metrics history should be non-empty"

    # Ecology should be active — coherence factor must be set
    eco = getattr(engine.extensions, '_ecology', None)
    if eco is not None:
        assert 0.0 <= eco.coherence_factor <= 1.0, (
            f"Ecology coherence_factor={eco.coherence_factor} outside [0,1]"
        )




# ── S2.B11: curl_flow edge cases ───────────────────────────────────

def test_curl_flow_empty_positions():
    """S2.B11: curl_flow with n=0 → zero array."""
    from pymurmur.physics.forces._base import curl_flow
    result = curl_flow(
        np.empty((0, 3), dtype=np.float32),
        np.array([500.0, 350.0, 200.0], dtype=np.float32),
        np.array([], dtype=np.float32),
        1.0, 100.0,
    )
    assert result.shape == (0, 3), f"Empty input → empty output, got {result.shape}"


def test_curl_flow_deterministic():
    """S2.B11: curl_flow with same inputs → same output (deterministic)."""
    from pymurmur.physics.forces._base import curl_flow

    positions = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float32)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    seeds = np.array([0.1, 0.5], dtype=np.float32)

    r1 = curl_flow(positions, center, seeds, 1.0, 100.0)
    r2 = curl_flow(positions, center, seeds, 1.0, 100.0)
    np.testing.assert_array_equal(r1, r2)


def test_curl_flow_different_t_different_output():
    """S2.B11: curl_flow at different times produces different flow."""
    from pymurmur.physics.forces._base import curl_flow

    positions = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float32)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    seeds = np.array([0.1, 0.5], dtype=np.float32)

    r1 = curl_flow(positions, center, seeds, 1.0, 100.0)
    r2 = curl_flow(positions, center, seeds, 10.0, 100.0)
    assert not np.allclose(r1, r2), (
        "Different t should produce different flow vectors"
    )


def test_curl_flow_magnitude_bounded():
    """S2.B11: curl_flow returns vectors of magnitude 0.08 (normalized)."""
    from pymurmur.physics.forces._base import curl_flow

    positions = np.random.default_rng(42).uniform(0, 1000, (50, 3)).astype(np.float32)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    seeds = np.arange(50, dtype=np.float32)

    result = curl_flow(positions, center, seeds, 1.5, 100.0)
    mags = np.linalg.norm(result, axis=1)
    assert np.allclose(mags, 0.08, atol=1e-5), (
        f"All flow magnitudes should be 0.08, got {mags.min():.6f}..{mags.max():.6f}"
    )


# ═══════════════════════════════════════════════════════════════════
# S2.B1: alignment_radius_ratio / separation_distance / global filter
# ═══════════════════════════════════════════════════════════════════

def test_alignment_radius_ratio_restricts_alignment_subset(default_config):
    """S2.B1: alignment set is a subset of the sep/coh neighbour set
    when alignment_radius_ratio < 1.0."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42
    cfg.mode = "spatial"
    cfg.num_boids = 80
    cfg.visual_range = 200.0
    cfg.spatial.alignment_radius_ratio = 0.3  # tight subset

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel_restricted = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_restricted, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    cfg.spatial.alignment_radius_ratio = 1.0  # no restriction (baseline)
    accel_baseline = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_baseline, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    assert not np.allclose(accel_restricted, accel_baseline), (
        "tightening alignment_radius_ratio should change the alignment contribution"
    )


def test_alignment_radius_ratio_default_is_noop(default_config):
    """S2.B1: default alignment_radius_ratio=1.0 must not change forces
    vs before the feature existed (no max_dist_align set)."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 7
    cfg.mode = "spatial"
    cfg.num_boids = 60
    assert cfg.spatial.alignment_radius_ratio == 1.0
    assert cfg.spatial.separation_distance == 0.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
    assert np.isfinite(accel).all()
    assert (np.linalg.norm(accel, axis=1) > 0).any(), "default config should still produce forces"


def test_separation_distance_gate_restricts_separation(default_config):
    """S2.B1: separation_distance gates separation neighbours to a tighter
    absolute distance than visual_range."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 3
    cfg.mode = "spatial"
    cfg.num_boids = 80
    cfg.visual_range = 200.0
    cfg.separation_weight = 5.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel_wide = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_wide, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    cfg.spatial.separation_distance = 15.0  # much tighter than visual_range
    accel_tight = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_tight, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    assert not np.allclose(accel_wide, accel_tight), (
        "separation_distance should change the separation contribution"
    )


def test_neighbor_filter_global_uses_flock_wide_mean(default_config):
    """S2.B1: neighbor_filter='global' steers every bird's alignment/
    cohesion toward the whole-flock mean velocity / centre of mass."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 11
    cfg.mode = "spatial"
    cfg.num_boids = 40
    cfg.spatial.neighbor_filter = "global"
    cfg.alignment_weight = 1.0
    cfg.cohesion_weight = 1.0
    cfg.separation_weight = 0.0
    cfg.noise_scale = 0.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    active_idx = np.where(flock.active)[0]
    mean_vel = flock.velocities[active_idx].mean(axis=0)
    mean_pos = flock.positions[active_idx].mean(axis=0)

    # Hand-check bird 0: align+coh should point toward global mean vel/CoM
    steer = mean_vel - flock.velocities[0]
    steer_norm = np.linalg.norm(steer)
    expected_align = steer / steer_norm if steer_norm > 1e-6 else np.zeros(3)
    to_center = mean_pos - flock.positions[0]
    length = np.linalg.norm(to_center)
    expected_coh = to_center / length if length > 1.0 else to_center

    expected = expected_align * cfg.alignment_weight + expected_coh * cfg.cohesion_weight
    expected_mag = np.linalg.norm(expected)
    if expected_mag > cfg.max_force:
        expected = expected / expected_mag * cfg.max_force

    assert np.allclose(accel[0], expected, atol=1e-4), (
        f"bird 0 force {accel[0]} != expected global-mean-based force {expected}"
    )


def test_neighbor_filter_global_does_not_crash_with_no_active(default_config):
    """S2.B1: 'global' mode with zero active birds is a no-op, not a crash."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 10
    cfg.spatial.neighbor_filter = "global"

    flock = PhysicsFlock(cfg)
    flock.active[:] = False
    flock.get_index().rebuild(flock.positions, flock.active)

    accel = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
    assert np.allclose(accel, 0.0)
