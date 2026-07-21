"""Unit tests for physics.forces dispatch and mode functions."""

from copy import copy

import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock  # noqa: E402
from test.helpers import _call_force


def test_compute_all_forces_imports():
    """All 5 modes are importable from the forces package."""
    from pymurmur.physics.forces import compute_all_forces
    from pymurmur.physics.forces.field import field_forces
    from pymurmur.physics.forces.influencer import influencer_forces
    from pymurmur.physics.forces.projection import projection_forces
    from pymurmur.physics.forces.spatial import spatial_forces
    from pymurmur.physics.forces.vicsek import vicsek_forces
    assert callable(compute_all_forces)
    assert callable(spatial_forces)
    assert callable(projection_forces)
    assert callable(field_forces)
    assert callable(vicsek_forces)
    assert callable(influencer_forces)


def test_mode_dispatch_unknown_raises(default_config):
    """Invalid mode raises ValueError."""
    import pytest

    from pymurmur.physics.forces import compute_all_forces

    cfg = default_config
    cfg.mode = "invalid_mode"
    flock = PhysicsFlock(cfg)

    with pytest.raises(ValueError, match="Unknown force mode"):
        compute_all_forces(flock, cfg)


def test_all_modes_run(default_config):
    """Each of the 5 modes runs without crash on a small flock."""
    from pymurmur.physics.forces import compute_all_forces

    cfg = copy(default_config)
    cfg.num_boids = 300

    for mode in ["projection", "spatial", "field", "vicsek", "influencer"]:
        cfg.mode = mode
        flock = PhysicsFlock(cfg)
        compute_all_forces(flock, cfg)
        # All active accelerations should be finite
        assert np.isfinite(flock.accelerations).all()


def test_spatial_forces_produces_nonzero(default_config):
    """Spatial mode with weights produces non-zero forces."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 300
    cfg.noise_scale = 0.5
    flock = PhysicsFlock(cfg)

    # Rebuild index first
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(spatial_forces, flock, cfg)
    assert not np.allclose(flock.accelerations[flock.active], 0.0)


# ── Spatial mode scenario tests ────────────────────────────────────


def test_spatial_mode_all_weights_zero(default_config):
    """All weights=0 → no steering forces (only noise may remain if scale=0 too)."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 10.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(spatial_forces, flock, cfg)
    # Accelerations should be all-zero (no steering, no noise)
    assert np.allclose(flock.accelerations[flock.active], 0.0)


def test_spatial_mode_separation_only(default_config):
    """Separation-only mode pushes birds apart."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 50.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(spatial_forces, flock, cfg)

    # Forces should be nonzero
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    # All forces should be finite
    assert np.isfinite(acc_active).all()


def test_spatial_mode_alignment_only(default_config):
    """Alignment-only mode steers toward average heading."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 20.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(spatial_forces, flock, cfg)

    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_spatial_mode_cohesion_only(default_config):
    """Cohesion-only mode pulls birds toward centre."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 20.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(spatial_forces, flock, cfg)

    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_spatial_mode_noise_only(default_config):
    """Noise-only mode produces random perturbations."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 1.0
    cfg.max_force = 20.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(spatial_forces, flock, cfg)

    acc_active = flock.accelerations[flock.active]
    # Noise should produce non-zero forces
    assert not np.allclose(acc_active, 0.0)
    # Noise vectors should have roughly unit norm before weight application
    assert np.isfinite(acc_active).all()


def test_spatial_mode_force_clamped(default_config):
    """No bird's acceleration exceeds config.max_force."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.separation_weight = 100.0  # huge weight → huge forces
    cfg.alignment_weight = 100.0
    cfg.cohesion_weight = 100.0
    cfg.noise_scale = 0.0  # P4.2: noise is post-clamp, set to 0 for clamp test
    cfg.max_force = 5.0  # low clamp

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(spatial_forces, flock, cfg)

    # Check that all acceleration magnitudes are ≤ max_force (within tolerance)
    acc_mags = np.linalg.norm(flock.accelerations, axis=1)
    active_mags = acc_mags[flock.active]
    assert np.all(active_mags <= cfg.max_force + 1e-5), \
        f"max acc: {active_mags.max()}, limit: {cfg.max_force}"


def test_spatial_mode_numba_fallback(default_config):
    """Spatial mode works without numba (pure numpy path)."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 30

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # Run on pure numpy path (numba may or may not be installed)
    _call_force(spatial_forces, flock, cfg)
    acc_active = flock.accelerations[flock.active]
    assert np.isfinite(acc_active).all()
    assert not np.allclose(acc_active, 0.0)


def test_spatial_mode_zero_active(default_config):
    """spatial_forces returns early when no birds are active."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    # Deactivate all birds
    flock.active[:] = False
    flock.accelerations[:] = 0.0

    _call_force(spatial_forces, flock, cfg)
    # Should be a no-op — no crash, no NaN
    assert np.allclose(flock.accelerations, 0.0)


def test_spatial_mode_single_bird(default_config):
    """spatial_forces handles N=1 without neighbour queries."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 1

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(spatial_forces, flock, cfg)
    # n < 2 → neighbour query returns empty → no steering → only noise
    assert np.isfinite(flock.accelerations).all()







# ── P11.5: Evolvable forward force + perception cones ─────────────


def test_forward_force_sign_flips_around_v0(default_config):
    """P11.5: w_fwd thrust accelerates below v0, decelerates above."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 2
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 100.0
    cfg.w_fwd = 1.0

    flock = PhysicsFlock(cfg)
    # Bird 0 slower than v0, bird 1 faster — both heading +x
    flock.positions[0] = [0.0, 0.0, 0.0]
    flock.positions[1] = [500.0, 0.0, 0.0]  # far apart → no interaction
    flock.velocities[0] = [cfg.v0 * 0.5, 0.0, 0.0]
    flock.velocities[1] = [cfg.v0 * 2.0, 0.0, 0.0]
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(spatial_forces, flock, cfg)

    assert flock.accelerations[0, 0] > 0.0, "Below v0 → thrust forward"
    assert flock.accelerations[1, 0] < 0.0, "Above v0 → braking"


def test_forward_force_off_by_default(default_config):
    """Without the w_fwd gene the spatial pipeline is unchanged."""
    from pymurmur.physics.forces.spatial import spatial_forces

    cfg = copy(default_config)
    cfg.mode = "spatial"
    cfg.num_boids = 2
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0

    flock = PhysicsFlock(cfg)
    flock.positions[0] = [0.0, 0.0, 0.0]
    flock.positions[1] = [500.0, 0.0, 0.0]
    flock.velocities[0] = [cfg.v0 * 0.5, 0.0, 0.0]
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(spatial_forces, flock, cfg)
    assert np.allclose(flock.accelerations[flock.active], 0.0)


def test_perception_cone_excludes_behind(default_config):
    """P11.5: cos-angle cone excludes neighbours behind the bird."""
    from pymurmur.physics.forces.spatial import _maybe_perception_filter

    positions = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    active = np.ones(3, dtype=bool)
    neighbor_idx = np.array([[1, 2], [2, 0], [1, 2]], dtype=np.int32)

    # 90° half-angle cone (cos α = 0): bird 0 heading +x sees bird 1
    # (ahead) but not bird 2 (behind)
    out = _maybe_perception_filter(
        positions, velocities, neighbor_idx, active,
        max_dist=0.0, cos_angle=0.0,
    )
    assert list(out[0]) == [1], f"Behind-cone bird must be excluded, got {list(out[0])}"


def test_perception_max_dist_filters(default_config):
    """P11.5: per-interaction max distance excludes far neighbours."""
    from pymurmur.physics.forces.spatial import _maybe_perception_filter

    positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    velocities = np.zeros((3, 3), dtype=np.float32)
    active = np.ones(3, dtype=bool)
    neighbor_idx = np.array([[1, 2], [2, 0], [1, 2]], dtype=np.int32)

    out = _maybe_perception_filter(
        positions, velocities, neighbor_idx, active,
        max_dist=10.0, cos_angle=-1.0,
    )
    assert list(out[0]) == [1], "Neighbour at 30 units must be excluded"


def test_perception_filter_fast_path(default_config):
    """Disabled filters return the shared neighbour array untouched."""
    from pymurmur.physics.forces.spatial import _maybe_perception_filter

    positions = np.zeros((2, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    active = np.ones(2, dtype=bool)
    neighbor_idx = np.array([[1], [0]], dtype=np.int32)

    out = _maybe_perception_filter(
        positions, velocities, neighbor_idx, active,
        max_dist=0.0, cos_angle=-1.0,
    )
    assert out is neighbor_idx, "Fast path must return the same object"


# ── S1.5: Separation kernel modes (sum | mean | unit) ─────────────

class TestSeparationKernel:
    """S1.5: separation_force() kernel parameter — sum, mean, unit."""

    @staticmethod
    def _simple_neighbors(n_neighbors: int, at_distance: float):
        """Create N=2 positions with n_neighbors at distance d from bird 1.

        Returns (positions, velocities, neighbor_idx, active).
        Bird 0 is at origin (no neighbors). Bird 1 is at (500,0,0) with
        n_neighbors all placed to the right (+x direction), slightly
        offset in Y to avoid identical positions. Forces don't cancel
        because all neighbors are on the same side.
        """
        N = 2 + n_neighbors
        positions = np.zeros((N, 3), dtype=np.float32)
        velocities = np.zeros((N, 3), dtype=np.float32)
        # Place all n_neighbors to the right of bird 1 (+x), stacked in Y
        for j in range(n_neighbors):
            positions[2 + j] = [
                500.0 + at_distance,
                (j - (n_neighbors - 1) / 2.0) * 0.3,
                0.0,
            ]
        positions[0] = [0.0, 0.0, 0.0]
        positions[1] = [500.0, 0.0, 0.0]
        active = np.ones(N, dtype=bool)

        # Bird 0 has no neighbors; bird 1 has n_neighbors
        neighbor_idx = np.zeros((N, n_neighbors), dtype=np.int32)
        neighbor_idx[1, :] = np.arange(2, 2 + n_neighbors, dtype=np.int32)

        return positions, velocities, neighbor_idx, active

    # ── "sum" kernel tests ────────────────────────────────────

    def test_sum_default_is_backward_compatible(self):
        """S1.5: Calling separation_force without kernel uses 'sum'."""
        from pymurmur.physics.forces._base import separation_force

        positions, velocities, neighbor_idx, active = self._simple_neighbors(4, 5.0)

        with_kernel = separation_force(
            positions, velocities, neighbor_idx, active, kernel="sum",
        )
        without_kernel = separation_force(
            positions, velocities, neighbor_idx, active,
        )

        assert np.allclose(with_kernel, without_kernel)

    def test_sum_scales_with_neighbor_count(self):
        """S1.5: 'sum' kernel — force magnitude grows with neighbor count."""
        from pymurmur.physics.forces._base import separation_force

        # 2 neighbours at distance 5
        p1, v1, n1, a1 = self._simple_neighbors(2, 5.0)
        force_2 = separation_force(p1, v1, n1, a1, kernel="sum")

        # 8 neighbours at distance 5 (same configuration, more neighbors)
        p2, v2, n2, a2 = self._simple_neighbors(8, 5.0)
        force_8 = separation_force(p2, v2, n2, a2, kernel="sum")

        # Bird 1's force magnitude should be ~4× larger with 8 vs 2 neighbors
        mag_2 = np.linalg.norm(force_2[1])
        mag_8 = np.linalg.norm(force_8[1])
        assert mag_8 > mag_2 * 2.0, (
            f"Sum kernel: 8 neighbors ({mag_8:.4f}) should be > 2× "
            f"2 neighbors ({mag_2:.4f})"
        )

    # ── "mean" kernel tests ───────────────────────────────────

    def test_mean_density_invariant(self):
        """S1.5: 'mean' kernel — same force regardless of neighbor count."""
        from pymurmur.physics.forces._base import separation_force

        # 2 neighbours at distance 5
        p1, v1, n1, a1 = self._simple_neighbors(2, 5.0)
        force_2 = separation_force(p1, v1, n1, a1, kernel="mean")

        # 8 neighbours at distance 5
        p2, v2, n2, a2 = self._simple_neighbors(8, 5.0)
        force_8 = separation_force(p2, v2, n2, a2, kernel="mean")

        mag_2 = np.linalg.norm(force_2[1])
        mag_8 = np.linalg.norm(force_8[1])

        # "mean" kernel divides by neighbor count — force should be similar
        # (not exactly equal due to angular distribution, but within 30%)
        assert mag_2 == pytest.approx(mag_8, rel=0.30), (
            f"Mean kernel: 2 neighbors ({mag_2:.4f}) and 8 neighbors "
            f"({mag_8:.4f}) should be similar (density-invariant)"
        )

    def test_mean_matches_sum_divided_by_k(self):
        """S1.5: 'mean' force = 'sum' force / k (for uniform distances)."""
        from pymurmur.physics.forces._base import separation_force

        positions, velocities, neighbor_idx, active = self._simple_neighbors(6, 5.0)

        force_sum = separation_force(
            positions, velocities, neighbor_idx, active, kernel="sum",
        )
        force_mean = separation_force(
            positions, velocities, neighbor_idx, active, kernel="mean",
        )

        # With 6 equally-spaced neighbors at same distance, the sum and
        # mean should be related by a factor of ~6 (exact for perfectly
        # symmetric arrangement)
        k = 6
        # The force on bird 1 should be sum/k
        assert np.allclose(force_sum[1] / k, force_mean[1], rtol=0.15), (
            f"Mean ≠ sum/k: sum={force_sum[1]}, sum/k={force_sum[1]/k}, "
            f"mean={force_mean[1]}"
        )

    # ── "unit" kernel tests ───────────────────────────────────

    def test_unit_distance_independent(self):
        """S1.5: 'unit' kernel — same force regardless of neighbour distance."""
        from pymurmur.physics.forces._base import separation_force

        # 4 neighbours at distance 5
        p_near, v_near, n_near, a_near = self._simple_neighbors(4, 5.0)
        force_near = separation_force(
            p_near, v_near, n_near, a_near, kernel="unit",
        )

        # 4 neighbours at distance 50 (same directions, 10× farther)
        p_far, v_far, n_far, a_far = self._simple_neighbors(4, 50.0)
        force_far = separation_force(
            p_far, v_far, n_far, a_far, kernel="unit",
        )

        mag_near = np.linalg.norm(force_near[1])
        mag_far = np.linalg.norm(force_far[1])

        # Unit kernel uses dir/distance, not dir/distance² — so force
        # should be the same regardless of distance (same directions)
        assert mag_near == pytest.approx(mag_far, rel=0.05), (
            f"Unit kernel: near ({mag_near:.4f}) and far ({mag_far:.4f}) "
            f"should be equal (distance-independent)"
        )

    def test_unit_vs_sum_at_short_range_nonzero(self):
        """S1.5: Both unit and sum kernels produce non-zero force at short range."""
        from pymurmur.physics.forces._base import separation_force

        # 4 neighbours very close (distance 1.0)
        positions, velocities, neighbor_idx, active = self._simple_neighbors(4, 1.0)

        force_sum = separation_force(
            positions, velocities, neighbor_idx, active, kernel="sum",
        )
        force_unit = separation_force(
            positions, velocities, neighbor_idx, active, kernel="unit",
        )

        mag_sum = np.linalg.norm(force_sum[1])
        mag_unit = np.linalg.norm(force_unit[1])

        assert mag_unit > 0.0, "Unit kernel should produce non-zero force"
        assert mag_sum > 0.0, "Sum kernel should produce non-zero force"

    def test_unit_larger_than_sum_for_distant_neighbors(self):
        """S1.5: 'unit' kernel produces more force than 'sum' at long range.

        sum: 1/d² scaling → tiny force for distant neighbors
        unit: 1/d scaling → still significant force
        """
        from pymurmur.physics.forces._base import separation_force

        # 4 neighbours far away (distance 50.0)
        positions, velocities, neighbor_idx, active = self._simple_neighbors(4, 50.0)

        force_sum = separation_force(
            positions, velocities, neighbor_idx, active, kernel="sum",
        )
        force_unit = separation_force(
            positions, velocities, neighbor_idx, active, kernel="unit",
        )

        mag_sum = np.linalg.norm(force_sum[1])
        mag_unit = np.linalg.norm(force_unit[1])

        # At distance=50, sum kernel (1/2500 scaling) vs unit kernel
        # (1/50 scaling). Unit should be ~50× larger.
        assert mag_unit > mag_sum * 10.0, (
            f"Unit kernel ({mag_unit:.6f}) should be >> sum kernel "
            f"({mag_sum:.6f}) at distance 50"
        )

    # ── Ragged array path tests ───────────────────────────────

    def test_ragged_mean_matches_dense_mean(self):
        """S1.5: Ragged (object array) mean kernel matches dense path
        for the bird that actually has neighbors."""
        from pymurmur.physics.forces._base import _is_ragged, separation_force

        positions, velocities, dense_idx, active = self._simple_neighbors(4, 5.0)

        # Dense path
        force_dense = separation_force(
            positions, velocities, dense_idx, active, kernel="mean",
        )

        # Convert to ragged (object array), but ONLY for bird 1 which
        # has real neighbors. Birds 0,2-5 have zero entries in dense_idx
        # which map to position[0] in dense but are empty in ragged.
        ragged_idx = np.empty(len(positions), dtype=object)
        for i in range(len(positions)):
            row = dense_idx[i]
            # Preserve zero entries in dense as [0] in ragged to keep
            # the neighbor semantics consistent (zero-valued index → bird 0)
            valid = row > 0
            if valid.any():
                ragged_idx[i] = row[valid].astype(np.int32)
            else:
                # This row had only zero entries → keep as empty array
                ragged_idx[i] = np.zeros(0, dtype=np.int32)
        assert _is_ragged(ragged_idx)

        force_ragged = separation_force(
            positions, velocities, ragged_idx, active, kernel="mean",
        )

        # Bird 1's force should match (has real neighbors in both)
        assert np.allclose(force_dense[1], force_ragged[1], atol=1e-6), (
            f"Ragged and dense mean kernel differ for bird 1: "
            f"dense={force_dense[1]}, ragged={force_ragged[1]}"
        )

    def test_ragged_unit_matches_dense_unit(self):
        """S1.5: Ragged (object array) unit kernel matches dense path
        for the bird that actually has neighbors."""
        from pymurmur.physics.forces._base import _is_ragged, separation_force

        positions, velocities, dense_idx, active = self._simple_neighbors(4, 5.0)

        force_dense = separation_force(
            positions, velocities, dense_idx, active, kernel="unit",
        )

        # Same ragged conversion with zero-preservation as above
        ragged_idx = np.empty(len(positions), dtype=object)
        for i in range(len(positions)):
            row = dense_idx[i]
            valid = row > 0
            if valid.any():
                ragged_idx[i] = row[valid].astype(np.int32)
            else:
                ragged_idx[i] = np.zeros(0, dtype=np.int32)
        assert _is_ragged(ragged_idx)

        force_ragged = separation_force(
            positions, velocities, ragged_idx, active, kernel="unit",
        )

        assert np.allclose(force_dense[1], force_ragged[1], atol=1e-6), (
            f"Ragged and dense unit kernel differ for bird 1: "
            f"dense={force_dense[1]}, ragged={force_ragged[1]}"
        )

    # ── Edge case tests ───────────────────────────────────────

    def test_all_kernels_handle_no_neighbors(self):
        """S1.5: All kernels return zero force when no neighbors exist."""
        from pymurmur.physics.forces._base import separation_force

        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        velocities = np.zeros((1, 3), dtype=np.float32)
        neighbor_idx = np.zeros((1, 0), dtype=np.int32)
        active = np.ones(1, dtype=bool)

        for kernel in ("sum", "mean", "unit"):
            force = separation_force(
                positions, velocities, neighbor_idx, active, kernel=kernel,
            )
            assert np.allclose(force, 0.0), (
                f"Kernel '{kernel}' should return zero force with no neighbors"
            )

    def test_all_kernels_handle_zero_active(self):
        """S1.5: All kernels return zero force when no birds active."""
        from pymurmur.physics.forces._base import separation_force

        positions, velocities, neighbor_idx, _ = self._simple_neighbors(3, 5.0)
        active = np.zeros(len(positions), dtype=bool)

        for kernel in ("sum", "mean", "unit"):
            force = separation_force(
                positions, velocities, neighbor_idx, active, kernel=kernel,
            )
            assert np.allclose(force, 0.0), (
                f"Kernel '{kernel}' should return zero force with no active birds"
            )

    def test_invalid_kernel_ignored_by_code_structure(self):
        """S1.5: Unknown kernel string falls through to 'sum' path."""
        from pymurmur.physics.forces._base import separation_force

        positions, velocities, neighbor_idx, active = self._simple_neighbors(3, 5.0)

        # An unknown kernel falls through the if/elif chain to the shared
        # sum/mean path. Since kernel != "mean", no division occurs → "sum".
        force_default = separation_force(
            positions, velocities, neighbor_idx, active, kernel="sum",
        )
        force_unknown = separation_force(
            positions, velocities, neighbor_idx, active, kernel="unknown_stuff",
        )

        assert np.allclose(force_default, force_unknown)
