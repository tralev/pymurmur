"""Unit tests for physics.forces — P11.5 evolvable forward force +
perception cones.

Split out of test_forces_forward_perception_kernel.py (file-size
split) — S1.5 separation/cohesion kernel-mode tests stay in the
original.
"""

from copy import copy

import numpy as np

from pymurmur.physics.flock import PhysicsFlock  # noqa: E402
from test.helpers import _call_force

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
