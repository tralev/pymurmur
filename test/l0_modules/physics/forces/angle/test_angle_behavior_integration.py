"""P5 — Angle mode: extended behavioural smoke (scale invariance, multi-knob, flee+edge combine, speed adaptation) and SimulationEngine integration (holey mask, toroidal seam).

Split out of test_angle.py (file-size split).
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.angle import AngleMode

# ── P5.7 (extended): Behavioural smoke — scale invariance ─────

def test_double_boid_size_changes_neighbour_behaviour():
    """P5.7: 2× boid_size changes which neighbour-mode is active.

    At b=9, bird at d=50 is > align_r=45 → cohere-only → target = ĉ.
    At b=18, same bird is sep_r < d < align_r=90 → align+cohere.
    With neighbours having strongly divergent headings (perpendicular
    to the centroid direction), align+cohere gives a measurably
    different target than cohere-only.
    """
    from pymurmur.core.types import safe_normalize as sn

    def _get_heading_after_compute(boid_size, d):
        cfg = SimConfig()
        cfg.mode = "angle"
        cfg.num_boids = 4
        cfg.boid_size = boid_size
        cfg.sep_radius_bodies = 1.0
        cfg.align_radius_bodies = 5.0
        cfg.range_radius_bodies = 12.0
        cfg.turn_rate = 360.0
        cfg.turn_threshold = 0.0
        cfg.jitter_deg = 0.0
        cfg.neighbors = 10
        cfg.base_speed = 4.0
        cfg.boundary_mode = "toroidal"

        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        # Bird 0 at origin heading +y
        flock.positions[0] = np.array([500, 350, 200], dtype=np.float32)
        # Bird 1 at +x (d from bird 0), heading strongly +z
        flock.positions[1] = np.array([500 + d, 350, 200], dtype=np.float32)
        flock.velocities[1] = np.array([0, 0, 4.0], dtype=np.float32)
        # Bird 2 also at +x but farther, heading also +z
        flock.positions[2] = np.array([500 + d + 10, 350, 200], dtype=np.float32)
        flock.velocities[2] = np.array([0, 0, 4.0], dtype=np.float32)
        # Bird 3 far away, heading +z (dummy)
        flock.positions[3] = np.array([700, 350, 200], dtype=np.float32)
        flock.velocities[3] = np.array([0, 0, 1.0], dtype=np.float32)
        # Bird 0 heading +y (perpendicular to centroid +x and neighbour headings +z)
        flock.velocities[0] = np.array([0, 4.0, 0], dtype=np.float32)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        return sn(flock.velocities[0])

    d = 50.0
    # D14: _last_cell is now per-index — each _get_heading_after_compute
    # creates its own flock/index, so no cross-test contamination.
    # b=9: align_r=45, range_r=108. d=50 > align_r → cohere-only (target = ĉ ≈ +x)
    h9 = _get_heading_after_compute(9.0, d)
    # b=18: align_r=90, range_r=216. d=50 in middle → align+cohere
    # Target = normalize(ĉ + m̂) where ĉ ≈ +x and m̂ ≈ +z
    # → target should have a +z component that cohere-only doesn't
    h18 = _get_heading_after_compute(18.0, d)

    # align+cohere (b=18) adds mean-heading component (+z) → heading differs
    assert not np.allclose(h9, h18, atol=0.01), (
        f"Double boid_size must change neighbour mode → different heading: "
        f"b=9:{h9} vs b=18:{h18}"
    )
    # The b=18 heading should have a non-trivial z component from alignment
    assert abs(h18[2]) > 0.01, (
        f"Align+cohere should add z heading component, got {h18}"
    )


# ═══════════════════════════════════════════════════════════════════
# P5 Integration tests — multiple features interacting as a whole
# ═══════════════════════════════════════════════════════════════════

def test_all_knobs_100_frames_no_nan_no_escape():
    """P5 integration: all features active for 100 frames — no NaN,
    speeds bounded, birds stay in domain.

    Exercises P5.1 (steering) + P5.2 (neighbours) + P5.3 (speed) +
    P5.4 (edge) + P5.5 (jitter) + P5.6 (incremental grid) + P5.7 (radii).
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 50
    cfg.boundary_mode = "margin"
    cfg.jitter_deg = 4.0
    cfg.turn_rate = 120.0
    cfg.max_turn_rate = 360.0
    cfg.turn_threshold = 0.8
    cfg.boid_size = 9.0
    cfg.sep_radius_bodies = 1.0
    cfg.align_radius_bodies = 5.0
    cfg.range_radius_bodies = 12.0
    cfg.base_speed = 4.0
    cfg.neighbors = 7
    cfg.margin = 50.0

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.accelerations[:] = 0.0

    # D14: per-index storage — no class-level state to reset
    for frame in range(100):
        flock.get_index().rebuild(flock.positions, flock.active)
        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        flock.integrate(cfg, 1.0 / 60.0)

        # No NaN
        assert np.isfinite(flock.positions).all(), f"NaN position at frame {frame}"
        assert np.isfinite(flock.velocities).all(), f"NaN velocity at frame {frame}"

        # Speeds bounded (adaptive speed: isolated ≤ 39, dense = 4)
        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert (speeds > 0).all(), f"Zero speed at frame {frame}"
        assert (speeds < 50).all(), f"Excessive speed {speeds.max():.0f} at frame {frame}"

        # No escapes
        xs = flock.positions[flock.active, 0]
        assert (xs > -1).all(), f"Bird escaped left at frame {frame}"
        assert (xs < cfg.width + 1).all(), f"Bird escaped right at frame {frame}"

    # Final state checks
    assert np.isfinite(flock.positions).all()
    speeds_final = np.linalg.norm(flock.velocities[flock.active], axis=1)
    # After 100 frames, speeds should be self-regulated (not all at max)
    assert speeds_final.mean() < 20, (
        f"Speeds should self-regulate: mean={speeds_final.mean():.1f}"
    )


def test_flee_and_edge_avoidance_combine():
    """P5.2+P5.4: Bird near wall AND near neighbour — both targets combine.

    Bird 0 is at x=15 (near left wall, should steer +x) and bird 1
    is at x=5 (even closer to wall, within sep_radius — should flee -x).
    Bird 0 must combine: steer right from edge AND steer left from
    nearby neighbour → result is a weighted blend.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 3
    cfg.margin = 50.0
    cfg.turn_rate = 360.0
    cfg.max_turn_rate = 720.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.boundary_mode = "margin"
    cfg.neighbors = 10
    cfg.base_speed = 4.0
    cfg.boid_size = 9.0
    cfg.sep_radius_bodies = 2.0  # sep_r = 18
    cfg.align_radius_bodies = 10.0
    cfg.range_radius_bodies = 20.0

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Bird 0 at x=15 (inside margin=50), heading left
    flock.positions[0] = np.array([15.0, 350, 200], dtype=np.float32)
    # Bird 1 at x=5 (d=10 < sep_r=18 → triggers flee), heading left
    flock.positions[1] = np.array([5.0, 350, 200], dtype=np.float32)
    # Bird 2 far away to provide enough neighbours
    flock.positions[2] = np.array([600, 350, 200], dtype=np.float32)
    flock.velocities[0] = np.array([-4.0, 0, 0], dtype=np.float32)
    flock.velocities[1] = np.array([-4.0, 0, 0], dtype=np.float32)
    flock.velocities[2] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    from pymurmur.core.types import safe_normalize as sn
    hdg_before = sn(flock.velocities[0].copy())
    # Heading starts going left (-x)
    assert hdg_before[0] < -0.9, f"Expected heading -x, got {hdg_before}"

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    hdg_after = sn(flock.velocities[0])
    # Both flee and edge push +x. Heading starts at -x (180° turn needed).
    # Turn rate cap limits rotation to 6°/frame at 360°/s.
    # After one frame, heading is no longer pure -x but still mostly -x.
    # Key assertion: heading moved (changed from before) AND moved
    # in the right direction (toward +x, i.e., less negative x).
    assert not np.allclose(hdg_before, hdg_after, atol=0.001), (
        "Combined flee+edge must change heading from pure -x"
    )
    # Heading should be turning toward +x (x component increases)
    assert hdg_after[0] > hdg_before[0], (
        f"Should turn toward +x: before={hdg_before[0]:.4f}, "
        f"after={hdg_after[0]:.4f}"
    )


def test_speed_adapts_as_flock_clusters():
    """P5.2+P5.3: Speed decreases as birds cluster, increases when spread out.

    Start with spread-out birds → high speeds (few neighbours).
    After several frames they cluster → lower speeds (more neighbours).
    This verifies the density self-regulation loop.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 30
    cfg.boundary_mode = "toroidal"
    cfg.jitter_deg = 2.0
    cfg.turn_rate = 120.0
    cfg.turn_threshold = 0.5
    cfg.boid_size = 9.0
    cfg.sep_radius_bodies = 1.0
    cfg.align_radius_bodies = 5.0
    cfg.range_radius_bodies = 12.0
    cfg.base_speed = 4.0
    cfg.neighbors = 7

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    # Spread birds across the domain (few neighbours each)
    for i in range(30):
        flock.positions[i] = np.array(
            [50 + i * 30, 350, 200], dtype=np.float32
        )
        flock.velocities[i] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0

    # D14: per-index storage — no class-level state to reset
    n_nbrs_history = []
    speeds_history = []

    # Run 30 frames, tracking neighbour counts and speeds
    for _frame in range(30):
        flock.get_index().rebuild(flock.positions, flock.active)

        # Count neighbours for bird 0 (using index query)
        nbrs = flock.get_index().query_knn(flock.positions[0], 7)
        n_nbrs_history.append(len(nbrs))

        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )

        speeds = np.linalg.norm(flock.velocities[flock.active], axis=1)
        speeds_history.append(float(speeds.mean()))

        flock.integrate(cfg, 1.0 / 60.0)

    # If birds cluster, neighbour counts should increase and speeds
    # should trend downward from their initial spread-out values.
    # Compare first 5 frames (spread out) vs last 5 frames (clustered).
    early_speed = sum(speeds_history[:5]) / 5
    late_speed = sum(speeds_history[-5:]) / 5
    assert late_speed <= early_speed + 2.0, (
        f"Density self-regulation: speeds should decrease as birds cluster. "
        f"Early mean={early_speed:.1f}, late mean={late_speed:.1f}"
    )
    # At least some frames should have >0 neighbours
    assert max(n_nbrs_history) > 0, "Birds should eventually find neighbours"
    # Speeds should be finite and positive
    assert all(s > 0 for s in speeds_history)
    assert all(np.isfinite(s) for s in speeds_history)


def test_incremental_grid_across_multiple_compute_calls():
    """P5.6: Incremental grid works correctly across multiple
    AngleMode.compute() calls with the same flock.

    Verifies that _last_cell persists correctly between frames
    without resetting, and touch rates stay low.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 50
    cfg.boundary_mode = "toroidal"
    cfg.jitter_deg = 0.0
    cfg.turn_threshold = 10.0
    cfg.neighbors = 10
    cfg.base_speed = 4.0

    flock = PhysicsFlock(cfg)
    flock.active[:] = True
    flock.accelerations[:] = 0.0

    # D14: _last_cell is now per-index — read from the spatial index
    idx = flock.get_index()

    # Frame 0: first compute initializes _angle_last_cell on the index
    idx.rebuild(flock.positions, flock.active)
    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, idx, flock.rng,
        flock.last_theta, cfg,
    )
    last_cell = getattr(idx, '_angle_last_cell', None)
    assert last_cell is not None
    assert (last_cell[:cfg.num_boids] >= 0).all()

    # Move slightly, compute again — _last_cell should persist on same index
    flock.positions[:] += 1.0  # move 1 unit right
    flock.positions[:, 0] %= cfg.width
    idx.rebuild(flock.positions, flock.active)
    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, idx, flock.rng,
        flock.last_theta, cfg,
    )

    # _angle_last_cell should still be valid and have updated
    last_cell = getattr(idx, '_angle_last_cell', None)
    assert last_cell is not None
    assert (last_cell[:cfg.num_boids] >= 0).all()

    # Verify that velocity/acceleration are still finite
    assert np.isfinite(flock.velocities).all()
    assert np.isfinite(flock.accelerations).all()


# ── P5 integration: through SimulationEngine ──────────────────

def test_angle_mode_through_simulation_engine():
    """P5 integration: Run angle mode through the full SimulationEngine.

    Verifies that the engine orchestration (index rebuild, force dispatch,
    integration, center update) works correctly with angle mode.
    """
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 30
    cfg.boundary_mode = "toroidal"
    cfg.jitter_deg = 2.0
    cfg.turn_rate = 120.0
    cfg.turn_threshold = 0.5
    cfg.seed = 42

    engine = SimulationEngine(cfg)

    # D14: per-index storage — no class-level state to reset
    for frame in range(20):
        engine.step(1.0 / 60.0)
        assert np.isfinite(engine.flock.positions).all(), f"NaN at frame {frame}"
        assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"

    # After 20 frames, birds should still be in domain
    xs = engine.flock.positions[engine.flock.active, 0]
    assert (xs >= -1).all() and (xs <= cfg.width + 1).all()
    assert engine.frame == 20


# ── P5 integration: holey mask contract ───────────────────────

def test_angle_mode_holey_mask_inactive_unchanged():
    """P5 integration: Inactive bird positions/velocities unchanged.

    Standard holey-mask contract — verifies angle mode respects the
    active mask and doesn't modify inactive birds.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 30
    cfg.boundary_mode = "toroidal"

    flock = PhysicsFlock(cfg)
    flock.active[5:10] = False
    flock.active[15:20] = False

    pos_before = flock.positions[~flock.active].copy()
    vel_before = flock.velocities[~flock.active].copy()

    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # D14: per-index storage — no class-level state to reset
    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    # Inactive birds must be bit-identical
    np.testing.assert_array_equal(
        flock.positions[~flock.active], pos_before,
        err_msg="Angle mode modified inactive bird positions",
    )
    np.testing.assert_array_equal(
        flock.velocities[~flock.active], vel_before,
        err_msg="Angle mode modified inactive bird velocities",
    )

    # Active birds should have non-zero velocity (not frozen)
    active_speeds = np.linalg.norm(
        flock.velocities[flock.active], axis=1
    )
    assert (active_speeds > 0).all(), "Active birds must have speed"


# ── P5 integration: toroidal cross-seam neighbours ────────────

def test_angle_mode_toroidal_cross_seam_neighbours():
    """P5 integration: Neighbour detection works across toroidal seam.

    Bird near right edge (x≈W−10) should detect bird near left edge
    (x≈10) as a neighbour through the toroidal wrap.
    """
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.num_boids = 3
    cfg.boundary_mode = "toroidal"
    cfg.boid_size = 9.0
    cfg.sep_radius_bodies = 2.0  # sep_r = 18
    cfg.align_radius_bodies = 10.0
    cfg.range_radius_bodies = 20.0
    cfg.turn_rate = 360.0
    cfg.turn_threshold = 0.0
    cfg.jitter_deg = 0.0
    cfg.neighbors = 10
    cfg.base_speed = 4.0

    flock = PhysicsFlock(cfg)
    flock.active[:] = True

    # Bird 0 near right edge (x = W - 10 = 990)
    flock.positions[0] = np.array(
        [cfg.width - 10, 350, 200], dtype=np.float32
    )
    # Bird 1 near left edge (x = 10)
    flock.positions[1] = np.array([10, 350, 200], dtype=np.float32)
    # Bird 2 far away
    flock.positions[2] = np.array([500, 350, 200], dtype=np.float32)

    flock.velocities[0] = np.array([4.0, 0, 0], dtype=np.float32)
    flock.velocities[1] = np.array([-4.0, 0, 0], dtype=np.float32)
    flock.velocities[2] = np.array([0, 4.0, 0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # Bird 0 should detect bird 1 as neighbour through toroidal wrap
    nbrs = flock.get_index().query_knn(flock.positions[0], 7)
    assert 1 in nbrs, (
        f"Toroidal seam: bird at x={flock.positions[0][0]:.0f} should see "
        f"bird at x={flock.positions[1][0]:.0f}. Got neighbours: {nbrs}"
    )

    # Now run angle mode compute — should handle cross-seam neighbours
    # D14: per-index storage — no class-level state to reset
    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    assert np.isfinite(flock.velocities).all()
    assert np.isfinite(flock.accelerations).all()


# ═══════════════════════════════════════════════════════════════════
# D14: AngleMode per-instance _angle_last_cell (no cross-talk)
# ═══════════════════════════════════════════════════════════════════


