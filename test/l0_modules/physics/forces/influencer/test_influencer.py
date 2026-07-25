"""Phase 7 — Influencer core: persistent tick + Lissajous target (P7.1), move-then-steer (P7.2), rank-by-target-distance influence (P7.3), density-scaled init (P7.4), distance diagnostics (P7.5).

Split out of test_influencer.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.influencer import (
    _lissajous_target,
    influencer_density_init,
    influencer_forces,
)
from test.helpers import _call_force


class TestInfluencerModeCore:
    """P7.1-P7.5: tick/Lissajous, move-then-steer, rank influence, density init, distance diagnostics."""

    # ── P7.1: Persistent tick + Lissajous target ─────────────────

    def test_tick_persists_across_calls(self):
        """P7.1: Tick counter persists across multiple compute() calls."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 10
        cfg.influencer_substeps = 1

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0])

        cfg._influencer_tick = 0.0
        _call_force(influencer_forces, flock, cfg)
        tick1 = cfg._influencer_tick

        _call_force(influencer_forces, flock, cfg)
        tick2 = cfg._influencer_tick

        assert tick1 > 0
        assert tick2 > tick1
        assert tick2 - tick1 == pytest.approx(
            cfg.influencer_tick_rate * cfg.influencer_substeps
        )

    def test_lissajous_deterministic(self):
        """P7.1: Same tick → same position; different tick → different position."""
        C = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        s_val = 1.0 * min(1000.0 / 460.0, 700.0 / 460.0, 400.0 / 254.0)

        t0 = _lissajous_target(0.0, C, s_val)
        assert np.isfinite(t0).all()

        t0b = _lissajous_target(0.0, C, s_val)
        np.testing.assert_array_equal(t0, t0b)

        t100 = _lissajous_target(100.0, C, s_val)
        assert not np.allclose(t0, t100)

    def test_lissajous_exact_values(self):
        """P7.1: T(t) at t∈{0, 970, 2170} equals hand-computed values."""
        C = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        s_val = 1.0 * min(1000.0 / 460.0, 700.0 / 460.0, 400.0 / 254.0)

        # Hand-computed using the spec formula
        # t=0: x=sin(0)*200+cos(0)*30=30, y=cos(53/29)*200+sin(47/13)*30≈..., z=cos(61/41)*100+sin(13/7)*27+40
        # With C=(500,350,200) and s≈1.521739:
        # T(0) = (545.652, 251.873, 312.898)
        expected = {
            0.0: np.array([545.652174, 251.873216, 312.898192], dtype=np.float32),
            970.0: np.array([323.472853, 77.439608, 446.284422], dtype=np.float32),
            2170.0: np.array([348.809836, 446.473766, 149.039338], dtype=np.float32),
        }

        for t, expected_val in expected.items():
            result = _lissajous_target(t, C, s_val)
            np.testing.assert_allclose(
                result, expected_val, rtol=1e-5, atol=1e-4,
                err_msg=f"t={t}: Lissajous mismatch"
            )
        """P7.1: Target stays inside domain for scale=0.5 (conservative containment)."""
        W, H, D = 460.0, 460.0, 254.0
        C = np.array([W / 2.0, H / 2.0, D / 2.0], dtype=np.float32)
        s_val = 0.5  # scale=0.5 ensures compact containment

        for t in np.linspace(0, 10000, 200):
            target = _lissajous_target(float(t), C, s_val)
            assert 0 <= target[0] <= W, (
                f"t={t:.1f}: x={target[0]:.1f} out of [0,{W}]"
            )
            assert 0 <= target[1] <= H, (
                f"t={t:.1f}: y={target[1]:.1f} out of [0,{H}]"
            )
            assert 0 <= target[2] <= D, (
                f"t={t:.1f}: z={target[2]:.1f} out of [0,{D}]"
            )

    # ── P7.2: Move-then-steer ────────────────────────────────────

    def test_produces_velocity_changes(self):
        """P7.2: Influencer steers velocities (not acceleration-based)."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 3

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.random.randn(*flock.velocities.shape).astype(np.float32)
        old_vels = flock.velocities.copy()

        _call_force(influencer_forces, flock, cfg)

        vel_diffs = np.linalg.norm(
            flock.velocities[flock.active] - old_vels[flock.active], axis=1
        )
        assert np.all(vel_diffs > 1e-6), (
            f"Not all birds steered: {np.sum(vel_diffs > 1e-6)}/{len(vel_diffs)}"
        )

    def test_velocity_clamped_to_v0(self):
        """P7.2: Speed strictly fixed to v0 (speed_mode='fixed')."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 10
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = (
            np.random.randn(*flock.velocities.shape).astype(np.float32) * 100.0
        )
        _call_force(influencer_forces, flock, cfg)

        v_mags = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(v_mags, cfg.v0), (
            f"Speed not fixed to v0 ({cfg.v0}): mean={v_mags.mean():.3f}"
        )

    def test_lissajous_steering_updates_each_frame(self):
        """P7.2: Velocity changes each frame as Lissajous target drifts."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 10
        cfg.influencer_substeps = 3
        cfg.influencer_scale = 0.5

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        cfg._influencer_tick = 0.0
        _call_force(influencer_forces, flock, cfg)
        vel1 = flock.velocities.copy()

        _call_force(influencer_forces, flock, cfg)
        vel2 = flock.velocities.copy()

        # Velocities change because Lissajous target moves between steps
        assert not np.allclose(vel1, vel2), (
            "Velocity should change as target moves between steps"
        )
        for step_vel in [vel1, vel2]:
            v_mags = np.linalg.norm(step_vel[flock.active], axis=1)
            assert np.allclose(v_mags, cfg.v0, atol=1e-4)

    def test_move_then_steer_lag(self):
        """P7.2: Velocity changes each call (target moves, steering updates)."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 10
        cfg.influencer_substeps = 1

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = 0.0

        cfg._influencer_tick = 0.0
        _call_force(influencer_forces, flock, cfg)
        vel1 = flock.velocities.copy()

        _call_force(influencer_forces, flock, cfg)
        vel2 = flock.velocities.copy()

        assert not np.allclose(vel1, vel2), "Velocity should change as target moves"

    # ── P7.3: Rank-by-target-distance influence ─────────────────

    def test_rank_influence_monotone(self):
        """P7.3: Influence monotone non-increasing in target distance."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 1.8

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_dirs = (
            flock.velocities[flock.active]
            / np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True)
        )

        _call_force(influencer_forces, flock, cfg)

        new_dirs = (
            flock.velocities[flock.active]
            / (np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True) + 1e-10)
        )
        angles = np.arccos(np.clip(np.sum(old_dirs * new_dirs, axis=1), -1.0, 1.0))

        t_last = cfg._influencer_tick - cfg.influencer_tick_rate
        C = np.array(
            [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0], dtype=np.float32
        )
        s_val = cfg.influencer_scale * min(
            cfg.width / 460.0, cfg.height / 460.0, cfg.depth / 254.0
        )
        target = _lissajous_target(float(t_last), C, s_val)
        dists = np.linalg.norm(flock.positions[flock.active] - target, axis=1)

        sort_idx = np.argsort(dists)
        sorted_angles = angles[sort_idx]
        diffs = np.diff(sorted_angles)
        assert np.mean(diffs) <= 0.05, "Average turn should decrease with distance"

    def test_closer_birds_turn_more(self):
        """P7.3: Birds closer to target turn more toward it than farther birds."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 100
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 2.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_dirs = (
            flock.velocities[flock.active]
            / np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True)
        )

        np.random.seed(42)
        _call_force(influencer_forces, flock, cfg)

        new_dirs = (
            flock.velocities[flock.active]
            / (np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True) + 1e-10)
        )
        angles = np.arccos(np.clip(np.sum(old_dirs * new_dirs, axis=1), -1.0, 1.0))

        t_last = cfg._influencer_tick - cfg.influencer_tick_rate
        C = np.array(
            [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0], dtype=np.float32
        )
        s_val = cfg.influencer_scale * min(
            cfg.width / 460.0, cfg.height / 460.0, cfg.depth / 254.0
        )
        target = _lissajous_target(float(t_last), C, s_val)
        dists = np.linalg.norm(flock.positions[flock.active] - target, axis=1)

        n = len(dists)
        close_idx = np.argsort(dists)[: max(n // 3, 1)]
        far_idx = np.argsort(dists)[-max(n // 3, 1):]

        close_avg = np.mean(angles[close_idx])
        far_avg = np.mean(angles[far_idx])

        assert close_avg > far_avg, (
            f"Closer birds turn more: close={close_avg:.4f} rad, far={far_avg:.4f} rad"
        )

    def test_rank_exponent_zero_equal_influence(self):
        """P7.3: rank_exp=0 → all birds get similar influence (within tolerance)."""
        cfg = SimConfig()
        cfg.seed = 42  # D6: default seed is None — pin so geometry is deterministic
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 0.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_dirs = flock.velocities[flock.active].copy()

        _call_force(influencer_forces, flock, cfg)

        new_dirs = (
            flock.velocities[flock.active]
            / (np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True) + 1e-10)
        )
        angles = np.arccos(np.clip(np.sum(old_dirs * new_dirs, axis=1), -1.0, 1.0))

        std_dev = np.std(angles)
        avg = np.mean(angles)
        # With equal influence, variation comes only from different target directions
        assert std_dev < max(avg * 0.6, 1e-6), (
            f"Turn magnitudes vary too much: std={std_dev:.4f}, avg={avg:.4f}"
        )

    def test_exactly_one_bird_at_max_influence(self):
        """P7.3: Closest bird turns most, with strictly larger angle than second-closest.

        Uses controlled geometry: all birds on one ray from the frame-0
        target, so the direction-to-target is identical for every bird and
        turn magnitude is strictly monotone in the influence weight. (With
        arbitrary positions the property doesn't hold — a low-influence bird
        whose target direction opposes its velocity can out-turn the closest.)
        """
        cfg = SimConfig()
        cfg.seed = 42  # D6: default seed is None — pin so geometry is deterministic
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 1.8

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Controlled geometry: bird i at target + (50 + 10·i)·ŷ, so ranks
        # are exactly the bird order and t̂ = −ŷ for all birds.
        C = np.array(
            [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0], dtype=np.float32
        )
        s_val = cfg.influencer_scale * min(
            cfg.width / 460.0, cfg.height / 460.0, cfg.depth / 254.0
        )
        target0 = _lissajous_target(0.0, C, s_val)
        for i in range(cfg.num_boids):
            flock.positions[i] = target0 + np.array(
                [0.0, 50.0 + 10.0 * i, 0.0], dtype=np.float32
            )
        old_dirs = (
            flock.velocities[flock.active]
            / np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True)
        )

        _call_force(influencer_forces, flock, cfg)

        new_dirs = (
            flock.velocities[flock.active]
            / (np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True) + 1e-10)
        )
        angles = np.arccos(np.clip(np.sum(old_dirs * new_dirs, axis=1), -1.0, 1.0))

        closest_idx = 0  # by construction
        max_angle_idx = np.argmax(angles)
        assert closest_idx == max_angle_idx, (
            f"Closest bird idx={closest_idx} should turn most, but idx={max_angle_idx} does"
        )
        # Closest bird (influence=1.0) must have strictly larger turn than second-closest
        assert angles[closest_idx] > angles[np.argsort(angles)[-2]], (
            "Closest bird should have strictly larger turn than second-closest (influence=1.0 vs <1.0)"
        )

    def test_min_influence_floor(self):
        """P7.3: Farthest bird influence ≈ 0.055 (rank_exp=1.8, 20 birds).

        Controlled geometry (birds on one ray from the frame-0 target) so
        turn magnitude is strictly monotone in the influence weight — see
        test_exactly_one_bird_at_max_influence.
        """
        cfg = SimConfig()
        cfg.seed = 42  # D6: default seed is None — pin so geometry is deterministic
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 1.8

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        C = np.array(
            [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0], dtype=np.float32
        )
        s_val = cfg.influencer_scale * min(
            cfg.width / 460.0, cfg.height / 460.0, cfg.depth / 254.0
        )
        target0 = _lissajous_target(0.0, C, s_val)
        for i in range(cfg.num_boids):
            flock.positions[i] = target0 + np.array(
                [0.0, 50.0 + 10.0 * i, 0.0], dtype=np.float32
            )
        old_dirs = (
            flock.velocities[flock.active]
            / np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True)
        )

        _call_force(influencer_forces, flock, cfg)

        new_dirs = (
            flock.velocities[flock.active]
            / (np.linalg.norm(flock.velocities[flock.active], axis=1, keepdims=True) + 1e-10)
        )
        angles = np.arccos(np.clip(np.sum(old_dirs * new_dirs, axis=1), -1.0, 1.0))

        # The farthest bird (rank = (N-1)) gets influence = (1 - 0.8)^1.8 = 0.2^1.8 ≈ 0.055
        # Its turn should be the smallest
        farthest_idx = cfg.num_boids - 1  # by construction
        min_angle_idx = np.argmin(angles)
        assert farthest_idx == min_angle_idx, (
            f"Farthest bird idx={farthest_idx} should turn least, but idx={min_angle_idx} does"
        )

        # The floor influence for 20 birds with rank_exp=1.8
        # farthest rank = (N-1)/(N-1) = 1.0
        # inf = (1 - 1.0 * 0.8)^1.8 = 0.2^1.8 ≈ 0.055
        expected_floor = 0.2 ** 1.8
        assert 0.03 < expected_floor < 0.07, (
            f"Floor influence should be ~0.055, got {expected_floor:.4f}"
        )

    # ── P7.4: Density-scaled init ───────────────────────────────

    def test_density_scaled_init_shape(self):
        """P7.4: Density init produces correct shape."""
        rng = np.random.default_rng(42)
        positions = influencer_density_init(
            n=100, width=1000.0, height=700.0, depth=400.0,
            scale=1.0, separation=0.5, rng=rng,
        )
        assert positions.shape == (100, 3)
        assert positions.dtype == np.float32

    def test_density_init_centred(self):
        """P7.4: Positions cluster around domain centre."""
        rng = np.random.default_rng(42)
        C = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        positions = influencer_density_init(
            n=100, width=1000.0, height=700.0, depth=400.0,
            scale=1.0, separation=0.5, rng=rng,
        )
        com = positions.mean(axis=0)
        assert np.linalg.norm(com - C) < 100.0, f"CoM {com} far from centre {C}"

    def test_density_init_scale_matters(self):
        """P7.4: Density similar across N∈{100, 1000}."""
        rng = np.random.default_rng(42)

        pos100 = influencer_density_init(
            n=100, width=1000.0, height=700.0, depth=400.0,
            scale=1.0, separation=0.5, rng=rng,
        )
        pos1000 = influencer_density_init(
            n=1000, width=1000.0, height=700.0, depth=400.0,
            scale=1.0, separation=0.5, rng=rng,
        )

        std100 = np.std(pos100, axis=0).mean()
        std1000 = np.std(pos1000, axis=0).mean()
        ratio = std1000 / max(std100, 1e-10)
        # sigma ~ N^(1/3) => ratio ~ 10^(1/3) ~ 2.154; the S2.E4 shared
        # offset shifts the mean, not the spread, so std is untouched by it.
        assert 1.5 < ratio < 2.6, f"Density ratio out of expected N^(1/3) band: {ratio:.2f}"

    def test_density_init_large_n(self):
        """P7.4: Init works correctly at N=8000 (large-scale density)."""
        rng = np.random.default_rng(42)
        positions = influencer_density_init(
            n=8000, width=1000.0, height=700.0, depth=400.0,
            scale=1.0, separation=0.5, rng=rng,
        )
        assert positions.shape == (8000, 3)
        assert positions.dtype == np.float32
        assert np.isfinite(positions).all()
        # All positions should be within reasonable bounds
        assert positions[:, 0].min() >= -500 and positions[:, 0].max() <= 1500

    def test_init_density_consistent(self):
        """P7.4: σ scales as N^(1/3) so spread ratio matches expectation.

        S2.E4: U(0,10s)³ is now a single SHARED offset per call (not
        per-bird jitter), so it displaces each call's cloud centre by an
        unpredictable amount relative to the fixed domain centre C —
        measuring spread from C would make this test flaky. Measure from
        each cloud's own centroid instead, which isolates sigma's N^(1/3)
        scaling from the shared-offset displacement.
        """
        rng = np.random.default_rng(42)
        sep = 0.5
        s = 1.0

        spreads = []
        for n in [100, 1000]:
            positions = influencer_density_init(
                n=n, width=1000.0, height=700.0, depth=400.0,
                scale=s, separation=sep, rng=rng,
            )
            centroid = positions.mean(axis=0)
            dists = np.linalg.norm(positions - centroid, axis=1)
            spreads.append(np.mean(dists))

        # sigma ~ N^(1/3) ~ 2.154 for a 10x increase in N.
        actual_ratio = spreads[1] / spreads[0]
        assert 1.5 < actual_ratio < 2.6, (
            f"Spread ratio {actual_ratio:.2f} out of expected N^(1/3) band"
        )

    def test_frame_0_headings_proportional(self):
        """P7.4: Frame-0 headings point toward target, proportional to influence."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 1
        cfg.influencer_scale = 0.5
        cfg.influencer_rank_exponent = 2.0

        flock = PhysicsFlock(cfg)
        # Start all with zero velocity so computed direction is pure target pull
        flock.velocities[:] = 0.0

        cfg._influencer_tick = 0.0
        _call_force(influencer_forces, flock, cfg)

        # All birds should have nonzero velocity
        v_mags = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.all(v_mags > 0), "Zero-velocity birds should get steered"

        # Velocities should generally point toward the target
        C = np.array(
            [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0], dtype=np.float32
        )
        s_val = cfg.influencer_scale * min(
            cfg.width / 460.0, cfg.height / 460.0, cfg.depth / 254.0
        )
        target = _lissajous_target(0.0, C, s_val)

        dirs = flock.velocities[flock.active]
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10)
        to_target = target - flock.positions[flock.active]
        to_target = to_target / (np.linalg.norm(to_target, axis=1, keepdims=True) + 1e-10)

        # At least 50% of birds should have positive dot product with target direction
        dots = np.sum(dirs * to_target, axis=1)
        assert np.mean(dots > 0.0) > 0.5, (
            f"Only {np.mean(dots>0.0)*100:.0f}% of birds point toward target"
        )

    # ── P7.5: Distance diagnostics ──────────────────────────────

    def test_distance_diagnostics_populated(self):
        """P7.5: config._target_dist_min/max populated after compute()."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 2

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        _call_force(influencer_forces, flock, cfg)

        assert hasattr(cfg, '_target_dist_min'), "target_dist_min not set"
        assert hasattr(cfg, '_target_dist_max'), "target_dist_max not set"
        assert cfg._target_dist_min > 0
        assert cfg._target_dist_max >= cfg._target_dist_min
        assert cfg._target_dist_max < (cfg.width + cfg.height + cfg.depth)

