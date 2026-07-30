"""Unit tests for physics.forces._base — force primitives."""

import numpy as np

from pymurmur.physics.forces._base import (
    alignment_force,
    cohesion_force,
    curl_flow,
    noise_force,
    separation_force,
)


def test_separation_force_no_neighbors(known_positions, known_velocities, neighbor_idx):
    """Birds with no neighbours get zero separation force."""
    N = len(known_positions)
    active = np.ones(N, dtype=bool)
    empty_idx = np.zeros((N, 0), dtype=np.int32)
    force = separation_force(known_positions, known_velocities, empty_idx, active)
    assert np.allclose(force, 0.0)


def test_separation_force_direction(known_positions, known_velocities):
    """Force points away from neighbour."""
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = separation_force(known_positions, known_velocities, idx, active)
    # Bird 0 pushed away from bird 1 at (10, 0, 0) → negative x
    assert force[0, 0] < 0


def test_alignment_force_no_neighbors(known_positions, known_velocities, neighbor_idx):
    """Birds with no neighbours get zero alignment force."""
    N = len(known_positions)
    active = np.ones(N, dtype=bool)
    empty_idx = np.zeros((N, 0), dtype=np.int32)
    force = alignment_force(known_positions, known_velocities, empty_idx, active)
    assert np.allclose(force, 0.0)


def test_cohesion_force_toward_center(known_positions, known_velocities):
    """Force points toward the centre of mass of neighbours."""
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = cohesion_force(known_positions, known_velocities, idx, active)
    # Bird 0's lone neighbour at (10, 0, 0) → cohesion toward +x
    assert force[0, 0] > 0


def test_noise_force_shape():
    """noise_force(N, s) returns (N, 3) float32."""
    f = noise_force(50, 0.5)
    assert f.shape == (50, 3)
    assert f.dtype == np.float32


def test_noise_force_zero_scale():
    """scale=0 produces all-zero array."""
    f = noise_force(20, 0.0)
    assert np.allclose(f, 0.0)


def test_force_primitives_inactive_rows(known_positions, known_velocities, neighbor_idx):
    """All primitives return zero force for inactive birds."""
    len(known_positions)
    active = np.array([True, True, False, False])
    sep = separation_force(known_positions, known_velocities, neighbor_idx, active)
    assert np.allclose(sep[~active], 0.0)
    align = alignment_force(known_positions, known_velocities, neighbor_idx, active)
    assert np.allclose(align[~active], 0.0)
    coh = cohesion_force(known_positions, known_velocities, neighbor_idx, active)
    assert np.allclose(coh[~active], 0.0)


def test_separation_force_zero_distance():
    """Two birds at identical positions — handled gracefully (no div-by-zero crash)."""
    pos = np.array([[100, 100, 100], [100, 100, 100]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    active = np.ones(2, dtype=bool)
    force = separation_force(pos, vel, idx, active)
    # Zero-distance neighbours are skipped (1e-6 guard), so force is zero
    assert np.all(np.isfinite(force))
    assert not np.any(np.isnan(force))


def test_separation_force_falls_with_distance(known_positions, known_velocities):
    """Force magnitude decreases as neighbour distance increases."""
    # Bird 0 sees bird 1 at (10, 0, 0) — far
    idx_far = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force_far = separation_force(known_positions, known_velocities, idx_far, active)

    # Move bird 1 closer to bird 0
    pos_near = known_positions.copy()
    pos_near[1] = np.array([1, 0, 0], dtype=np.float32)
    idx_near = np.array([[1], [0], [3], [2]], dtype=np.int32)
    force_near = separation_force(pos_near, known_velocities, idx_near, active)

    # Force should be stronger at close range
    assert np.linalg.norm(force_near[0]) > np.linalg.norm(force_far[0])


def test_separation_force_inactive_ignored():
    """Inactive birds get zero separation force computed for them."""
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    vel = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    # Bird 1 is inactive, bird 0 is active
    active = np.array([True, False])
    force = separation_force(pos, vel, idx, active)
    # Bird 0 is active and sees bird 1 — gets a force (neighbour filtering
    # by active mask is the caller's responsibility).
    assert np.isfinite(force[0]).all()
    # Bird 1 is inactive → force slot stays at initialised zero
    assert np.allclose(force[1], 0.0)


def test_alignment_force_parallel():
    """Two birds with identical velocities → alignment force is zero."""
    pos = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    active = np.ones(2, dtype=bool)
    force = alignment_force(pos, vel, idx, active)
    # Identical velocities → avg/norm == vi/norm → force ≈ 0
    assert np.allclose(force[0], 0.0, atol=1e-6)


def test_alignment_force_opposite():
    """Two birds with opposite velocities → alignment force is nonzero."""
    pos = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    idx = np.array([[1], [0]], dtype=np.int32)
    active = np.ones(2, dtype=bool)
    force = alignment_force(pos, vel, idx, active)
    assert not np.allclose(force[0], 0.0)
    assert np.isfinite(force).all()


def test_cohesion_force_single_neighbor(known_positions, known_velocities):
    """With one neighbour, cohesion force points directly toward it."""
    # Bird 0 has only bird 1 at (10, 0, 0) → cohesion should point toward +x
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = cohesion_force(known_positions, known_velocities, idx, active)
    # Force toward neighbour should have positive x component
    assert force[0, 0] > 0
    # Force should be along x axis only (neighbour is on x axis)
    assert abs(force[0, 1]) < 1e-6
    assert abs(force[0, 2]) < 1e-6


def test_cohesion_force_no_neighbors(known_positions, known_velocities):
    """Birds with no neighbours get zero cohesion force."""
    N = len(known_positions)
    active = np.ones(N, dtype=bool)
    empty_idx = np.zeros((N, 0), dtype=np.int32)
    force = cohesion_force(known_positions, known_velocities, empty_idx, active)
    assert np.allclose(force, 0.0)


def test_noise_force_unit_scale():
    """noise_force(N, 1.0) produces vectors with unit norm."""
    N = 500
    f = noise_force(N, 1.0)
    assert f.shape == (N, 3)
    norms = np.linalg.norm(f, axis=1)
    # All norms should be ~1.0 (noise is normalized after gaussian sampling)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ── S2.B11: shared curl_flow primitive ──────────────────────────────

def _curl_flow_inputs():
    rng = np.random.default_rng(0)
    n = 12
    positions = rng.uniform(-50, 50, (n, 3)).astype(np.float32)
    center = np.zeros(3, dtype=np.float32)
    seeds = np.arange(n, dtype=np.float32)
    t = 3.7
    U = 100.0
    return positions, center, seeds, t, U


def test_curl_flow_empty_returns_empty():
    empty = np.zeros((0, 3), dtype=np.float32)
    out = curl_flow(empty, np.zeros(3, dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0, 100.0)
    assert out.shape == (0, 3)


def test_curl_flow_deterministic_same_seeds_and_t():
    """S2.B11: identical (positions, seeds, t) → identical output."""
    positions, center, seeds, t, U = _curl_flow_inputs()
    out1 = curl_flow(positions, center, seeds, t, U)
    out2 = curl_flow(positions, center, seeds, t, U)
    np.testing.assert_array_equal(out1, out2)


def test_curl_flow_differs_with_different_t():
    """Different time parameter → different flow field (it's time-varying)."""
    positions, center, seeds, t, U = _curl_flow_inputs()
    out1 = curl_flow(positions, center, seeds, t, U)
    out2 = curl_flow(positions, center, seeds, t + 5.0, U)
    assert not np.allclose(out1, out2)


def test_curl_flow_is_normalized_before_base_scale():
    """Output magnitude is exactly the 0.08 base scale (unit direction × 0.08)."""
    positions, center, seeds, t, U = _curl_flow_inputs()
    out = curl_flow(positions, center, seeds, t, U)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 0.08, atol=1e-5)


# ── New kernel dispatch: separation "linear"/"nearest_only"/"bell_zone",
#    cohesion "bell_zone", alignment "fov_weighted"/"spherical_mean" ──

def test_separation_force_linear_kernel_dispatch(known_positions, known_velocities):
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = separation_force(known_positions, known_velocities, idx, active, kernel="linear")
    assert force[0, 0] < 0  # still pushes away from neighbour at +x
    assert np.isfinite(force).all()


def test_separation_force_nearest_only_kernel_dispatch():
    pos = np.array([[0, 0, 0], [10, 0, 0], [0, 8, 0], [0, 0, -3]], dtype=np.float32)
    vel = np.zeros((4, 3), dtype=np.float32)
    idx = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    active = np.array([True, True, True, True])
    force = separation_force(pos, vel, idx, active, kernel="nearest_only")
    # Bird 0's nearest neighbour is bird 3 at (0,0,-3) -> push toward +z
    assert force[0, 2] > 0
    assert abs(force[0, 0]) < 1e-5
    assert abs(force[0, 1]) < 1e-5


def test_separation_force_bell_zone_kernel_dispatch(known_positions, known_velocities):
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = separation_force(
        known_positions, known_velocities, idx, active,
        kernel="bell_zone", kernel_radius=10.0, kernel_zone_width=5.0,
    )
    assert np.isfinite(force).all()
    assert np.linalg.norm(force[0]) > 0


def test_cohesion_force_bell_zone_kernel_dispatch(known_positions, known_velocities):
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    force = cohesion_force(
        known_positions, known_velocities, idx, active,
        kernel="bell_zone", kernel_radius=10.0, kernel_zone_width=5.0,
    )
    assert np.isfinite(force).all()


def test_alignment_force_fov_weighted_kernel_dispatch():
    pos = np.array([[0, 0, 0], [10, 0, 0], [-10, 0, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
    idx = np.array([[1, 2], [0, 0], [0, 0]], dtype=np.int32)
    active = np.ones(3, dtype=bool)
    force = alignment_force(pos, vel, idx, active, kernel="fov_weighted", fov_min=0.0)
    assert np.isfinite(force).all()
    # Dead-ahead neighbour (bird 1) dominates -> steering has positive y pull
    assert force[0, 1] > 0


def test_alignment_force_spherical_mean_kernel_dispatch():
    pos = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([[1, 2], [0, 0], [0, 0]], dtype=np.int32)
    active = np.ones(3, dtype=bool)
    force = alignment_force(pos, vel, idx, active, kernel="spherical_mean")
    assert np.isfinite(force).all()


def test_alignment_force_bell_zone_kernel_dispatch():
    pos = np.array([[0, 0, 0], [10, 0, 0], [-10, 0, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
    idx = np.array([[1, 2], [0, 0], [0, 0]], dtype=np.int32)
    active = np.ones(3, dtype=bool)
    force = alignment_force(
        pos, vel, idx, active,
        kernel="bell_zone", kernel_radius=10.0, kernel_zone_width=5.0,
    )
    assert np.isfinite(force).all()


def test_alignment_force_unweighted_default_unchanged(known_positions, known_velocities):
    """Default kernel="unweighted" must reproduce the pre-existing plain-mean
    behavior byte-for-byte — new kernel param is opt-in."""
    idx = np.array([[1], [0], [3], [2]], dtype=np.int32)
    active = np.ones(4, dtype=bool)
    explicit = alignment_force(known_positions, known_velocities, idx, active, kernel="unweighted")
    implicit = alignment_force(known_positions, known_velocities, idx, active)
    np.testing.assert_array_equal(explicit, implicit)


def test_curl_flow_field_and_spatial_agree_up_to_gain():
    """S2.B11: FieldMode._compute_curl_flow and SpatialMode's flow_contrib
    both build on curl_flow() — for the same inputs, FieldMode's output
    should equal curl_flow(...) * flow * flow_pull exactly (its own gain
    formula), proving the two modes share one primitive rather than two
    independently-drifting implementations."""
    from pymurmur.physics.forces.field import _compute_curl_flow

    positions, center, seeds, t, U = _curl_flow_inputs()
    flow, flow_pull = 1.3, 0.7

    field_out = _compute_curl_flow(positions, center, seeds, t, U, flow, flow_pull)
    shared_out = curl_flow(positions, center, seeds, t, U) * flow * flow_pull
    np.testing.assert_allclose(field_out, shared_out, atol=1e-6)


# ── Modularity pass 10/11: stash_target_speed ────────────────────

class TestStashTargetSpeed:
    def test_scatters_active_speeds_and_defaults_inactive_to_v0(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.forces._base import stash_target_speed

        cfg = SimConfig()
        cfg.v0 = 4.0
        active_idx = np.array([1, 3])
        speeds = np.array([10.0, 20.0], dtype=np.float32)

        stash_target_speed(cfg, n_total=5, active_idx=active_idx, speeds_active=speeds)

        stashed = cfg._mode_target_speed
        assert stashed.shape == (5,)
        assert stashed[1] == 10.0
        assert stashed[3] == 20.0
        # Inactive rows (0, 2, 4) default to v0 -- masked out by `active`
        # downstream regardless, but must not be NaN/garbage.
        assert stashed[0] == 4.0
        assert stashed[2] == 4.0
        assert stashed[4] == 4.0

    def test_empty_active_idx_produces_all_v0(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.forces._base import stash_target_speed

        cfg = SimConfig()
        cfg.v0 = 4.0
        stash_target_speed(
            cfg, n_total=3, active_idx=np.array([], dtype=np.int64),
            speeds_active=np.array([], dtype=np.float32),
        )
        np.testing.assert_array_equal(cfg._mode_target_speed, [4.0, 4.0, 4.0])


class TestModeTargetSpeedIntegration:
    """flock.integrate() consumes config._mode_target_speed as the
    max_speed base, and clears it after reading (one-shot)."""

    def test_integrate_uses_mode_target_speed_as_base(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = "spatial"  # mode choice irrelevant here — testing integrate() directly
        cfg.num_boids = 3
        cfg.v0 = 4.0
        cfg.spatial.predator_speed_boost = 1.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:3] = np.array([[10, 0, 0], [10, 0, 0], [10, 0, 0]], dtype=np.float32)
        cfg._mode_target_speed = np.array([1.0, 2.0, 3.0] + [4.0] * (len(flock.positions) - 3), dtype=np.float32)

        flock.integrate(cfg, dt=1.0 / 60.0, speed_mode="fixed")

        speeds = np.linalg.norm(flock.velocities[:3], axis=1)
        np.testing.assert_allclose(speeds, [1.0, 2.0, 3.0], atol=1e-4)

    def test_integrate_clears_mode_target_speed_after_use(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 3
        flock = PhysicsFlock(cfg)
        cfg._mode_target_speed = np.full(len(flock.positions), 2.0, dtype=np.float32)

        flock.integrate(cfg, dt=1.0 / 60.0, speed_mode="fixed")

        assert cfg._mode_target_speed is None

    def test_integrate_without_mode_target_speed_falls_back_to_v0(self):
        """No stash present (e.g. spatial/projection/field modes, which
        never call stash_target_speed) -> unaffected, same as before
        this change."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 3
        cfg.v0 = 4.0
        cfg.spatial.predator_speed_boost = 1.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:3] = np.array([[10, 0, 0]] * 3, dtype=np.float32)
        assert getattr(cfg, '_mode_target_speed', None) is None

        flock.integrate(cfg, dt=1.0 / 60.0, speed_mode="fixed")

        speeds = np.linalg.norm(flock.velocities[:3], axis=1)
        np.testing.assert_allclose(speeds, [4.0, 4.0, 4.0], atol=1e-4)
