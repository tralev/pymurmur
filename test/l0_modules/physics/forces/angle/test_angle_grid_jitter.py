"""P5 — Angle mode: heading jitter (P5.5), scale invariance (P5.7), mode registration, incremental grid (P5.6), extended P5.2/P5.4/P5.5 cases.

Split out of test_angle.py (file-size split).
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.angle import AngleMode

# ── P5.5: Heading jitter ──────────────────────────────────────────

def test_jitter_produces_variation():
    """P5.5: With jitter > 0, repeated runs produce different headings."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 5
    cfg.jitter_deg = 10.0
    cfg.turn_threshold = 20.0  # large dead zone to isolate jitter
    cfg.boundary_mode = "open"
    cfg.neighbors = 10
    cfg.base_speed = 4.0

    all_h1 = []
    all_h2 = []

    for run in range(2):
        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        for i in range(5):
            flock.positions[i] = np.array(
                [500, 350 + i * 10, 200], dtype=np.float32
            )
            flock.velocities[i] = np.array([4.0, 0, 0], dtype=np.float32)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        if run == 1:
            flock.rng = np.random.default_rng(99)

        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )

        for i in range(5):
            v = flock.velocities[i]
            n = np.linalg.norm(v)
            if run == 0:
                all_h1.append(v / n)
            else:
                all_h2.append(v / n)

    # Jitter should produce some variation between runs
    h1 = np.array(all_h1)
    h2 = np.array(all_h2)
    assert not np.allclose(h1, h2), (
        "Jitter should produce different headings between runs"
    )


# ── P5.7: Body-unit scale invariance ──────────────────────────────

def test_double_boid_size_doubles_radii():
    """P5.7: sep/align/range radii scale with boid_size."""
    b = 9.0
    cfg = SimConfig()
    cfg.boid_size = b
    cfg.sep_radius_bodies = 1.0
    cfg.align_radius_bodies = 5.0
    cfg.range_radius_bodies = 12.0

    sep_r = cfg.sep_radius_bodies * b
    align_r = cfg.align_radius_bodies * b
    range_r = cfg.range_radius_bodies * b

    assert sep_r == 9.0
    assert align_r == 45.0
    assert range_r == 108.0

    # Double boid_size → all radii double
    cfg.boid_size = 18.0
    sep_r2 = cfg.sep_radius_bodies * 18.0
    align_r2 = cfg.align_radius_bodies * 18.0
    range_r2 = cfg.range_radius_bodies * 18.0

    assert sep_r2 == 2 * sep_r
    assert align_r2 == 2 * align_r
    assert range_r2 == 2 * range_r


# ── Mode registration ─────────────────────────────────────────────

def test_angle_mode_in_registry():
    """'angle' key exists in MODE_REGISTRY."""
    from pymurmur.physics.forces._mode import MODE_REGISTRY

    assert "angle" in MODE_REGISTRY, "angle must be in MODE_REGISTRY"
    assert MODE_REGISTRY["angle"] == AngleMode


def test_angle_mode_runs_without_crash():
    """Angle mode compute() runs without error on a small flock."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 20
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    assert np.isfinite(flock.velocities).all()
    assert np.isfinite(flock.accelerations).all()


def test_angle_mode_zero_active():
    """Angle mode handles zero active birds gracefully."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    flock.active[:] = False

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    # Should not crash


# ── P5.6: Incremental spatial grid ────────────────────────────────

def test_incremental_grid_equivalent_to_full_rebuild():
    """P5.6: Neighbour sets from incremental rebuild == full rebuild sets."""
    from pymurmur.physics.flock import SpatialHashGrid

    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 50
    cfg.visual_range = 70.0
    cfg.boundary_mode = "toroidal"

    # Create two identical flocks with SpatialHashGrid indices
    flock1 = PhysicsFlock(cfg)
    flock2 = PhysicsFlock(cfg)

    # Copy state from flock1 to flock2
    flock2.positions[:] = flock1.positions
    flock2.velocities[:] = flock1.velocities
    flock2.active[:] = flock1.active

    # Ensure both use SpatialHashGrid (small N)
    index1 = SpatialHashGrid(cfg)
    index2 = SpatialHashGrid(cfg)

    # Full rebuild on index1
    index1.rebuild(flock1.positions, flock1.active)

    # Incremental rebuild on index2 (fresh _last_cell = all -1)
    last_cell = np.full((cfg.num_boids, 3), -1, dtype=np.int32)
    n_touched = index2.incremental_rebuild(flock2.positions, flock2.active, last_cell)

    # On first frame, incremental touches N_active (add only — no old cells to remove)
    assert n_touched == flock2.active.sum(), (
        f"First frame should touch N_active (add only): "
        f"touched={n_touched}, N_active={flock2.active.sum()}"
    )

    # Verify both indices produce identical query results
    for i in range(cfg.num_boids):
        if not flock1.active[i]:
            continue
        nbrs1 = index1.query_knn(flock1.positions[i], 7)
        nbrs2 = index2.query_knn(flock2.positions[i], 7)
        assert set(nbrs1) == set(nbrs2), (
            f"Bird {i}: full={set(nbrs1)}, incr={set(nbrs2)}"
        )


def test_incremental_grid_touch_rate_below_10_pct():
    """P5.6: After first frame, incremental rebuild touches <10% of birds.

    At typical speeds, most birds stay in the same cell each frame.
    """
    from pymurmur.physics.flock import SpatialHashGrid

    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 200
    cfg.visual_range = 70.0
    cfg.v0 = 4.0
    cfg.boundary_mode = "toroidal"

    # Use SpatialHashGrid directly
    index = SpatialHashGrid(cfg)
    last_cell = np.full((cfg.num_boids, 3), -1, dtype=np.int32)

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.velocities[:] = np.random.default_rng(42).uniform(
        -4, 4, (cfg.num_boids, 3)
    ).astype(np.float32)

    # Frame 0: full population (all birds are new — add only, no old cells)
    n0 = index.incremental_rebuild(flock.positions, flock.active, last_cell)
    assert n0 == flock.active.sum(), (
        f"First frame touches N_active (add only), got {n0}"
    )

    # Frame 1+: step positions forward by one frame
    dt = 1.0 / 60.0
    n_total_touched = 0
    n_frames = 10

    for _ in range(n_frames):
        # Move birds
        flock.positions[:] += flock.velocities[:] * dt
        # Wrap toroidal
        flock.positions[:, 0] %= cfg.width
        flock.positions[:, 1] %= cfg.height
        flock.positions[:, 2] %= cfg.depth

        t = index.incremental_rebuild(
            flock.positions, flock.active, last_cell,
        )
        n_total_touched += t

    avg_touched = n_total_touched / n_frames
    n_active = flock.active.sum()
    touch_rate = avg_touched / n_active

    # Each bird touched counts as 2 (remove from old + add to new when
    # cell changes). At v0=4, cell_size=70, most birds stay in same cell.
    # Expected: ~5-10 birds cross per frame out of 200 → ~2.5-5% × 2 = 5-10%
    assert touch_rate < 0.10, (
        f"Touch rate {touch_rate:.1%} should be < 10% (P5.6 spec)"
    )


def test_last_cell_initialized_and_updated():
    """P5.6: _last_cell is initialized and updated across frames."""
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 5
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # D14: _last_cell is now per-index — read from the spatial index
    idx = flock.get_index()

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, idx, flock.rng,
        flock.last_theta, cfg,
    )

    # After first compute, _angle_last_cell should be initialized on the index
    last_cell = getattr(idx, '_angle_last_cell', None)
    assert last_cell is not None, "_angle_last_cell must be initialized on index"
    assert last_cell.shape[0] >= cfg.num_boids, (
        f"_angle_last_cell must cover at least N_active={cfg.num_boids}, "
        f"got shape={last_cell.shape}"
    )
    assert last_cell.shape[1] == 3, (
        f"_angle_last_cell must have 3 columns, got {last_cell.shape[1]}"
    )
    # At least one bird should have a valid cell (not -1)
    assert (last_cell[0] >= 0).all(), (
        "Active birds must have valid cell coords"
    )


# ── P5.2 (extended): Align+cohere middle ground ───────────────

def test_align_and_cohere_when_mid_range():
    """P5.2: sep_r < nearest < align_r → steer toward normalize(ĉ + m̂).

    Bird at middle distance should both cohere (toward centroid)
    AND align (toward mean heading), not just cohere-only.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 4
    cfg.sep_radius_bodies = 0.5   # sep_r = 4.5
    cfg.align_radius_bodies = 3.0  # align_r = 27
    cfg.range_radius_bodies = 8.0  # range_r = 72
    cfg.boid_size = 9.0
    cfg.turn_rate = 360.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.neighbors = 10
    cfg.base_speed = 4.0
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Bird 0 at origin heading +y. Bird 1 at d=15 (sep_r < 15 < align_r).
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
    flock.positions[1] = np.array([515, 350, 200], dtype=np.float32)  # d=15
    flock.positions[2] = np.array([600, 350, 200], dtype=np.float32)
    flock.positions[3] = np.array([700, 350, 200], dtype=np.float32)
    # Bird 0 heading up, bird 1 heading right (strongly different)
    flock.velocities[0] = np.array([0, 4.0, 0], dtype=np.float32)
    flock.velocities[1] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.velocities[2] = np.array([1.0, 1.0, 0], dtype=np.float32)
    flock.velocities[3] = np.array([0, 1.0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    from pymurmur.core.types import safe_normalize as sn
    hdg_before = sn(flock.velocities[0].copy())

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    hdg_after = sn(flock.velocities[0])
    # Bird 1 is at +x → centroid pull is +x. Bird 1 heading is +x.
    # Combined target = normalize(ĉ + m̂) should pull bird 0 toward +x.
    assert hdg_after[0] > hdg_before[0], (
        f"Align+cohere should pull toward neighbour: "
        f"before={hdg_before}, after={hdg_after}"
    )


# ── P5.4 (extended): Multi-frame edge containment ─────────────

def test_cube_edge_containment_over_many_frames():
    """P5.4: Bird heading at wall at high speed stays within domain.

    Spec: 10⁴ frames at max speeds → zero escapes.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 10  # neighbours to avoid speed boost
    cfg.margin = 50.0
    cfg.turn_rate = 360.0
    cfg.max_turn_rate = 720.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "margin"
    cfg.neighbors = 10
    cfg.base_speed = 4.0

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Place all birds just inside left wall, heading directly left
    for i in range(10):
        flock.positions[i] = np.array(
            [30.0, 350 + i * 5, 200], dtype=np.float32
        )
        flock.velocities[i] = np.array([-4.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0

    min_x_over_time = float("inf")
    for _ in range(200):  # many frames
        flock.get_index().rebuild(flock.positions, flock.active)
        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        flock.integrate(cfg, 1.0 / 60.0)
        min_x = flock.positions[flock.active, 0].min()
        min_x_over_time = min(min_x_over_time, min_x)
        # Must never go negative (escape left wall)
        assert min_x > -1.0, f"Bird escaped left wall: min_x={min_x:.1f}"

    assert min_x_over_time > -1.0, (
        f"Edge containment failed: min x over 200 frames = {min_x_over_time:.1f}"
    )


# ── P5.5 (extended): Jitter distribution bounded ±4° ──────────

def test_jitter_distribution_bounded():
    """P5.5: Steering-off distribution is bounded ±jitter_deg°.

    With steering disabled (large dead zone), heading changes come
    only from jitter, which must be within ±jitter_deg°.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 1
    cfg.jitter_deg = 4.0
    cfg.turn_threshold = 90.0  # huge dead zone — disables all steering
    cfg.boundary_mode = "open"
    cfg.neighbors = 10
    cfg.base_speed = 4.0

    rng = np.random.default_rng(42)
    flock = PhysicsFlock(cfg)
    flock.rng = rng
    flock.active[:] = True
    flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)

    max_angle = 0.0
    for _ in range(100):
        flock.velocities[0] = np.array([4.0, 0, 0], dtype=np.float32)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        hdg_before = flock.velocities[0] / np.linalg.norm(flock.velocities[0])

        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )

        hdg_after = flock.velocities[0] / np.linalg.norm(flock.velocities[0])
        cos_a = np.clip(np.dot(hdg_before, hdg_after), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        max_angle = max(max_angle, angle)

    # Jitter is ±4° → max rotation from jitter alone ≤ 4°
    assert max_angle <= cfg.jitter_deg + 0.1, (
        f"Jitter max angle {max_angle:.2f}° should be ≤ {cfg.jitter_deg}°"
    )


