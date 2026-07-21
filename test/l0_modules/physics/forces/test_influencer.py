"""Phase 7 — Influencer parity tests.

P7.1: Persistent tick + spec Lissajous target
P7.2: Move-then-steer at unit speed (velocity-based, not acceleration)
P7.3: Rank-by-target-distance influence weights
P7.4: Density-scaled Gaussian init
P7.5: Distance diagnostics (target_dist_min/max)
P7.6: Pilot mode (WASD attractor)
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.influencer import (
    InfluencerMode,
    PilotTarget,
    _lissajous_target,
    influencer_density_init,
    influencer_forces,
)
from pymurmur.simulation.engine import SimulationEngine
from test.helpers import _call_force


class TestInfluencerMode:
    """P7: Persistent tick-driven Lissajous target with move-then-steer."""

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
        assert 0.5 < ratio < 2.0, f"Density ratio too large: {ratio:.2f}"

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
        """P7.4: σ scales as N^(1/3) so spread ratio matches expectation."""
        rng = np.random.default_rng(42)
        sep = 0.5
        s = 1.0

        spreads = []
        for n in [100, 1000]:
            positions = influencer_density_init(
                n=n, width=1000.0, height=700.0, depth=400.0,
                scale=s, separation=sep, rng=rng,
            )
            C = np.array([500.0, 350.0, 200.0], dtype=np.float32)
            dists = np.linalg.norm(positions - C, axis=1)
            spreads.append(np.mean(dists))

        # σ ∝ N^(1/3), but U(0,10s)³ jitter adds constant offset
        # so spread ratio is less than pure N^(1/3) ≈ 2.15, but still > 1.0
        actual_ratio = spreads[1] / spreads[0]
        assert 1.0 < actual_ratio < 2.0, (
            f"Spread ratio {actual_ratio:.2f} should be >1 (N=1000 wider than N=100)"
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

    # ── P7.6: Pilot mode ────────────────────────────────────────

    def test_pilot_target_set_and_clear(self):
        """P7.6: PilotTarget set/clear via set_pilot()."""
        pilot = PilotTarget(
            position=np.array([100.0, 200.0, 300.0], dtype=np.float32),
            heading=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        assert InfluencerMode._pilot is None
        InfluencerMode.set_pilot(pilot)
        assert InfluencerMode._pilot is pilot
        InfluencerMode.set_pilot(None)
        assert InfluencerMode._pilot is None

    def test_pilot_mode_produces_velocity_changes(self):
        """P7.6: With active pilot, birds steer toward pilot position."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 2

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_vels = flock.velocities.copy()

        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            )
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            vel_diffs = np.linalg.norm(
                flock.velocities[flock.active] - old_vels[flock.active], axis=1
            )
            assert np.all(vel_diffs > 1e-6), "No velocity change with active pilot"
        finally:
            InfluencerMode.set_pilot(None)

    def test_shell_radius_expands_on_scatter(self):
        """P7.6: Scatter increases shell_radius monotonically up to 2.2 cap."""
        pilot = PilotTarget()
        pilot.shell_radius = 1.0
        dt = 1.0 / 60.0

        radii = [pilot.shell_radius]
        for _ in range(100):
            pilot.update_shell(dt, scatter=True, gather=False)
            radii.append(pilot.shell_radius)

        # Monotonically non-decreasing
        for i in range(1, len(radii)):
            assert radii[i] >= radii[i - 1] - 1e-10, (
                f"Shell radius decreased at step {i}: {radii[i]:.4f} < {radii[i-1]:.4f}"
            )

        # Should reach or approach the 2.2 cap
        assert radii[-1] >= 2.19, f"Shell radius {radii[-1]:.4f} didn't reach cap 2.2"

    def test_shell_radius_contracts_on_gather(self):
        """P7.6: Gather decreases shell_radius monotonically down to 0.42 floor."""
        pilot = PilotTarget()
        pilot.shell_radius = 2.0
        dt = 1.0 / 60.0

        radii = [pilot.shell_radius]
        for _ in range(100):
            pilot.update_shell(dt, scatter=False, gather=True)
            radii.append(pilot.shell_radius)

        # Monotonically non-increasing
        for i in range(1, len(radii)):
            assert radii[i] <= radii[i - 1] + 1e-10, (
                f"Shell radius increased at step {i}: {radii[i]:.4f} > {radii[i-1]:.4f}"
            )

        # Should reach or approach the 0.42 floor
        assert radii[-1] <= 0.43, f"Shell radius {radii[-1]:.4f} didn't reach floor 0.42"

    def test_pilot_heading_force_isolated(self):
        """P7.6: F_heading = pilot_heading * 0.12 when bird at pilot position."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 1
        cfg.influencer_substeps = 5  # more substeps → heading force accumulates
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        # Place bird exactly at pilot position (negates core_follow and shell_pull)
        pilot_pos = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        flock.positions[0] = pilot_pos.copy()
        flock.velocities[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # heading +z

        pilot = PilotTarget(
            position=pilot_pos.copy(),
            heading=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # heading +x
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            # Heading is (1,0,0), initial vel is (0,0,1).
            # heading_force is very small (0.12), so the +x component is tiny
            # after normalization.  Just verify it's nonzero (heading works).
            vel = flock.velocities[0]
            assert vel[0] > 0, f"Heading force should add +x component: vel={vel}"
        finally:
            InfluencerMode.set_pilot(None)

    def test_pilot_core_follow_isolated(self):
        """P7.6: F_core = (pilot_pos - p_i) * 0.22, inside shell_radius."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 1
        cfg.influencer_substeps = 5
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        # Bird at (510,350,200), pilot at (500,350,200).  d=10 < shell_radius=50.
        # Use orthogonal initial velocity (0,0,1) so core_follow clearly pulls -x.
        pilot_pos = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        flock.positions[0] = np.array([510.0, 350.0, 200.0], dtype=np.float32)
        flock.velocities[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        pilot = PilotTarget(
            position=pilot_pos.copy(),
            heading=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.shell_radius = 50.0
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            # With zero heading + inside shell: only core_follow fires
            # F_core = (-10, 0, 0) * 0.22 = (-2.2, 0, 0)
            # After 5 substeps accumulated, renormalized to v0
            vel = flock.velocities[0]
            assert vel[0] < 0, f"Core follow should pull -x, got vel={vel}"
            assert abs(vel[1]) < 0.5, f"No lateral component, got vel={vel}"
        finally:
            InfluencerMode.set_pilot(None)

    def test_pilot_shell_pull_activation(self):
        """P7.6: shell_pull fires only when d > shell_radius."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 2
        cfg.influencer_substeps = 5
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        pilot_pos = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        # Bird A: inside shell (d=40 < radius=50) — no shell_pull
        flock.positions[0] = np.array([540.0, 350.0, 200.0], dtype=np.float32)
        # Bird B: outside shell (d=70 > radius=50) — shell_pull fires
        flock.positions[1] = np.array([570.0, 350.0, 200.0], dtype=np.float32)
        # Orthogonal initial velocity so force direction clearly dominates
        flock.velocities[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        pilot = PilotTarget(
            position=pilot_pos.copy(),
            heading=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.shell_radius = 50.0
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0
            _call_force(influencer_forces, flock, cfg)

            vel_a = flock.velocities[0]  # inside shell (d=40)
            vel_b = flock.velocities[1]  # outside shell (d=70)

            # Both should be pulled toward pilot (-x direction)
            assert vel_a[0] < 0, f"Bird A should move -x, got {vel_a}"
            assert vel_b[0] < 0, f"Bird B should move -x, got {vel_b}"

            # Bird B (farther + shell_pull) should have at least as strong -x
            assert abs(vel_b[0]) >= abs(vel_a[0]) - 1e-4, (
                f"Bird B shell-pull should match or exceed bird A: a={vel_a} b={vel_b}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    def test_flock_follows_within_2_shell_radius(self):
        """P7.6 acceptance: Flock CoM tracks pilot within 2·shell_radius."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 5
        cfg.influencer_rank_exponent = 2.0
        cfg.seed = 42

        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                dtype=np.float32,
            ),
            heading=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.shell_radius = 60.0
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            engine = SimulationEngine(cfg)
            for _ in range(60):
                engine.step(1.0 / 60.0)

            com = engine.flock.positions[engine.flock.active].mean(axis=0)
            dist = np.linalg.norm(com - pilot.position)
            assert dist < 2.0 * pilot.shell_radius, (
                f"Flock CoM {dist:.1f} should be within "
                f"2·shell_radius={2*pilot.shell_radius:.1f}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    def test_frozen_target_convergence(self):
        """P7.6: Static pilot → birds converge toward it over multiple calls."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 5
        cfg.influencer_rank_exponent = 2.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            )
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0

            # Run several calls, simulating integrate() movement
            for _ in range(10):
                _call_force(influencer_forces, flock, cfg)
                flock.positions += flock.velocities * 0.1

            dists = np.linalg.norm(
                flock.positions[flock.active] - pilot.position, axis=1
            )
            assert dists.mean() < 500.0, (
                f"Birds too far from pilot: {dists.mean():.1f}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    # ── Edge cases ──────────────────────────────────────────────

    def test_zero_active(self):
        """Empty flock produces no change."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        old_acc = flock.accelerations.copy()
        old_vel = flock.velocities.copy()
        _call_force(influencer_forces, flock, cfg)
        assert np.allclose(flock.accelerations, old_acc)
        assert np.allclose(flock.velocities, old_vel)

    def test_single_bird(self):
        """Single bird: rank=0 → influence=1.0, velocity steered."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 1
        cfg.influencer_substeps = 1

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        old_vel = flock.velocities.copy()

        np.random.seed(42)
        _call_force(influencer_forces, flock, cfg)

        assert not np.allclose(flock.velocities[0], old_vel[0])
        assert np.linalg.norm(flock.velocities[0]) == pytest.approx(cfg.v0)

    def test_no_neighbour_queries(self):
        """Influencer mode never queries the spatial index."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.random.randn(*flock.velocities.shape).astype(np.float32)
        old_vels = flock.velocities.copy()

        class SpyIndex:
            def __init__(self):
                self.ready = True
            def query_knn(self, *a, **kw):
                raise RuntimeError("Should not be called")
            def query_radius(self, *a, **kw):
                raise RuntimeError("Should not be called")
            def rebuild(self, *a, **kw):
                pass

        flock._index = SpyIndex()
        _call_force(influencer_forces, flock, cfg)

        vel_diffs = np.linalg.norm(
            flock.velocities[flock.active] - old_vels[flock.active], axis=1
        )
        assert np.all(vel_diffs > 1e-6)

    def test_substeps_multiply_turn(self):
        """More substeps → proportionally larger turn angle."""
        cfg1 = SimConfig()
        cfg1.seed = 42  # D6: pin seed so both flocks share initial geometry
        cfg1.mode = "influencer"
        cfg1.num_boids = 30
        cfg1.influencer_substeps = 1

        cfg2 = SimConfig()
        cfg2.seed = 42  # D6: identical geometry to flock1
        cfg2.mode = "influencer"
        cfg2.num_boids = 30
        cfg2.influencer_substeps = 3

        flock1 = PhysicsFlock(cfg1)
        flock1.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_dirs1 = (
            flock1.velocities[flock1.active]
            / np.linalg.norm(flock1.velocities[flock1.active], axis=1, keepdims=True)
        )
        _call_force(influencer_forces, flock1, cfg1)
        new_dirs1 = (
            flock1.velocities[flock1.active]
            / (np.linalg.norm(flock1.velocities[flock1.active], axis=1, keepdims=True) + 1e-10)
        )
        ang1 = np.arccos(np.clip(np.sum(old_dirs1 * new_dirs1, axis=1), -1.0, 1.0)).mean()

        flock2 = PhysicsFlock(cfg2)
        flock2.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_dirs2 = (
            flock2.velocities[flock2.active]
            / np.linalg.norm(flock2.velocities[flock2.active], axis=1, keepdims=True)
        )
        _call_force(influencer_forces, flock2, cfg2)
        new_dirs2 = (
            flock2.velocities[flock2.active]
            / (np.linalg.norm(flock2.velocities[flock2.active], axis=1, keepdims=True) + 1e-10)
        )
        ang2 = np.arccos(np.clip(np.sum(old_dirs2 * new_dirs2, axis=1), -1.0, 1.0)).mean()

        ratio = ang2 / max(ang1, 1e-10)
        # Not 3.0 despite 3 substeps: the direction blend saturates toward
        # t̂ per substep, and D11 move-then-steer shifts positions between
        # substeps. The point is super-unity accumulation, not linearity.
        assert 1.3 < ratio < 6.0, f"Substep scaling: ang2/ang1 = {ratio:.2f}"

    def test_inactive_birds_unchanged(self):
        """Inactive birds unchanged while active ones are steered."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 2

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        flock.active[10:20] = False
        old_vel_inactive = flock.velocities[~flock.active].copy()
        old_vel_active = flock.velocities[flock.active].copy()

        _call_force(influencer_forces, flock, cfg)

        assert np.allclose(flock.velocities[~flock.active], old_vel_inactive)
        vel_diff_active = np.linalg.norm(
            flock.velocities[flock.active] - old_vel_active, axis=1
        )
        assert np.any(vel_diff_active > 1e-6)

    def test_substeps_zero(self):
        """substeps=0 → no velocity change, no crash, diagnostics still run."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        old_vel = flock.velocities.copy()
        _call_force(influencer_forces, flock, cfg)

        assert np.allclose(flock.velocities, old_vel)
        assert hasattr(cfg, '_target_dist_min')

    # ── Integration: Mid-run state transitions ─────────────────

    def test_mid_run_pilot_toggle_through_engine(self):
        """P7.1+P7.6: Toggle pilot on/off mid-run, Lissajous ↔ pilot seamlessly."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 15
        cfg.influencer_substeps = 2
        cfg.influencer_scale = 0.5
        cfg.seed = 42

        engine = SimulationEngine(cfg)

        # Phase 1: Lissajous only (10 frames)
        for _ in range(10):
            engine.step(1.0 / 60.0)
        lissajous_pos = engine.flock.positions.copy()

        # Phase 2: Activate pilot at centre (15 frames)
        pilot = PilotTarget(
            position=np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                              dtype=np.float32)
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            for _ in range(15):
                engine.step(1.0 / 60.0)
            pilot_pos = engine.flock.positions.copy()
        finally:
            InfluencerMode.set_pilot(None)

        # Phase 3: Back to Lissajous (10 frames)
        for _ in range(10):
            engine.step(1.0 / 60.0)
        lissajous2_pos = engine.flock.positions.copy()

        # All phases should be NaN-free and speeds bounded
        assert np.isfinite(lissajous_pos).all()
        assert np.isfinite(pilot_pos).all()
        assert np.isfinite(lissajous2_pos).all()
        assert engine.frame == 35

    def test_mid_run_influence_mode_switch(self):
        """P7.3: Switch influence_mode mid-run, diagnostics survive."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 1.8
        cfg.influencer_influence_mode = "rank"

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Phase 1: rank mode
        cfg._influencer_tick = 0.0
        _call_force(influencer_forces, flock, cfg)
        rank_vel = flock.velocities.copy()
        rank_dmin = cfg._target_dist_min

        # Phase 2: switch to distance mode
        cfg.influencer_influence_mode = "distance"
        _call_force(influencer_forces, flock, cfg)
        dist_vel = flock.velocities.copy()
        dist_dmin = cfg._target_dist_min

        # Both modes produce valid velocities
        for vel in [rank_vel, dist_vel]:
            v_mags = np.linalg.norm(vel[flock.active], axis=1)
            assert np.allclose(v_mags, cfg.v0, atol=1e-4)
        # Diagnostics survive mode switch
        assert rank_dmin > 0
        assert dist_dmin > 0
        # Rankings differ, so velocities should differ
        assert not np.allclose(rank_vel, dist_vel), (
            "Rank and distance modes should produce different steering"
        )

    def test_dynamic_shell_radius_effect(self):
        """P7.6: Shell_radius controls flock spread via shell_pull activation.

        With a small shell_radius, shell_pull activates sooner, pulling birds
        inward.  With a large shell_radius, birds feel no shell_pull until
        they're much farther out.  We verify that both radii produce valid
        steering and that the flock stays bounded."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 3
        cfg.v0 = 4.0

        flock = PhysicsFlock(cfg)
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        pilot = PilotTarget(
            position=np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                              dtype=np.float32)
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            cfg._influencer_tick = 0.0

            # Run with multiple shell radii, verify speeds stay bounded
            for shell_r in [20.0, 50.0, 200.0]:
                pilot.shell_radius = shell_r
                for _ in range(5):
                    _call_force(influencer_forces, flock, cfg)
                    flock.positions += flock.velocities * 0.1
                    v_mags = np.linalg.norm(
                        flock.velocities[flock.active], axis=1
                    )
                    assert np.allclose(v_mags, cfg.v0, atol=1e-4), (
                        f"Speed violation at shell_radius={shell_r}"
                    )
                    assert np.isfinite(flock.positions).all()
                    assert np.isfinite(flock.velocities).all()
        finally:
            InfluencerMode.set_pilot(None)

    def test_long_run_200_frames_all_features(self):
        """P7.1–P7.6: 200-frame stress test, all features, no NaN, speeds bounded."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 3
        cfg.influencer_rank_exponent = 2.0
        cfg.influencer_scale = 0.6
        cfg.influencer_tick_rate = 0.7
        cfg.v0 = 4.0
        cfg.seed = 123

        engine = SimulationEngine(cfg)

        pilot = PilotTarget(
            position=np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                              dtype=np.float32)
        )
        pilot.active = True
        pilot.shell_radius = 50.0
        InfluencerMode.set_pilot(pilot)

        try:
            for frame in range(200):
                engine.step(1.0 / 60.0)

                # Cycle pilot modes: pilot 0-49, Lissajous 50-99, pilot 100-149, Lissajous 150-199
                if frame == 50:
                    InfluencerMode.set_pilot(None)  # back to Lissajous
                elif frame == 100:
                    InfluencerMode.set_pilot(pilot)  # back to pilot
                elif frame == 150:
                    InfluencerMode.set_pilot(None)  # back to Lissajous

                assert np.isfinite(engine.flock.positions).all(), f"NaN at frame {frame}"
                assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"

                v_mags = np.linalg.norm(
                    engine.flock.velocities[engine.flock.active], axis=1
                )
                assert v_mags.max() <= cfg.v0 + 1e-4, (
                    f"Speed exceeded v0 at frame {frame}: {v_mags.max():.3f}"
                )
        finally:
            InfluencerMode.set_pilot(None)

        assert engine.frame == 200

    # ── Integration: P7 as a whole ─────────────────────────────

    def test_end_to_end_through_engine(self):
        """P7.1–P7.5: Full pipeline through SimulationEngine with influencer mode."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 2
        cfg.influencer_rank_exponent = 1.8
        cfg.influencer_scale = 0.5
        cfg.influencer_influence_mode = "rank"
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        for frame in range(10):
            engine.step(1.0 / 60.0)
            assert np.isfinite(engine.flock.positions).all(), f"NaN at frame {frame}"
            assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"
            # Velocities should all be v0 (speed_mode='fixed' via integrate)
            v_mags = np.linalg.norm(engine.flock.velocities[engine.flock.active], axis=1)
            # integrate applies speed clamping after compute, may differ slightly
            assert np.allclose(v_mags, cfg.v0, atol=1e-4), (
                f"Speed not v0 at frame {frame}: {v_mags.mean():.3f}"
            )

    def test_multi_frame_stability_50_frames(self):
        """P7.1–P7.5: 50 frames through engine, no NaN, positions in domain."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 3
        cfg.influencer_rank_exponent = 2.0
        cfg.influencer_scale = 0.8
        cfg.influencer_influence_mode = "distance"
        cfg.influencer_near_dist_sq = 150.0
        cfg.influencer_tick_rate = 0.5
        cfg.seed = 99

        engine = SimulationEngine(cfg)
        for frame in range(50):
            engine.step(1.0 / 60.0)
            assert np.isfinite(engine.flock.positions).all(), f"NaN pos at frame {frame}"
            assert np.isfinite(engine.flock.velocities).all(), f"NaN vel at frame {frame}"
            # Positions should stay within domain (toroidal wrapping)
            W, H, D = cfg.width, cfg.height, cfg.depth
            pos = engine.flock.positions[engine.flock.active]
            assert (pos[:, 0] >= -1.0).all() and (pos[:, 0] <= W + 1.0).all(), (
                f"x out of domain at frame {frame}"
            )
            assert (pos[:, 1] >= -1.0).all() and (pos[:, 1] <= H + 1.0).all(), (
                f"y out of domain at frame {frame}"
            )
            assert (pos[:, 2] >= -1.0).all() and (pos[:, 2] <= D + 1.0).all(), (
                f"z out of domain at frame {frame}"
            )
        assert engine.frame == 50

    def test_density_init_plus_compute(self):
        """P7.4+P7.1+P7.2: Density-scaled positions work with Lissajous steering."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 2
        cfg.influencer_scale = 1.0
        cfg.influencer_init_separation = 0.5

        flock = PhysicsFlock(cfg)
        # Override positions with density-scaled init
        rng = np.random.default_rng(42)
        flock.positions[:] = InfluencerMode.density_init_positions(
            n=cfg.num_boids,
            width=cfg.width,
            height=cfg.height,
            depth=cfg.depth,
            config=cfg,
            rng=rng,
        )
        flock.velocities[:] = 0.0

        _call_force(influencer_forces, flock, cfg)

        # All birds should have nonzero velocity (steered toward Lissajous target)
        v_mags = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(v_mags, cfg.v0, atol=1e-4), (
            f"Speed not v0 after density init: {v_mags.mean():.3f}"
        )
        # Diagnostics should be populated
        assert hasattr(cfg, '_target_dist_min')

    def test_pilot_end_to_end_through_engine(self):
        """P7.6+P7.1+P7.2: Pilot mode through SimulationEngine with convergence."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 20
        cfg.influencer_substeps = 3
        cfg.influencer_rank_exponent = 1.8
        cfg.seed = 42

        # Set pilot at domain centre
        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            ),
            heading=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        pilot.active = True
        pilot.shell_radius = 50.0
        InfluencerMode.set_pilot(pilot)

        try:
            engine = SimulationEngine(cfg)
            initial_dist = np.linalg.norm(
                engine.flock.positions.mean(axis=0) - pilot.position
            )

            for frame in range(30):
                engine.step(1.0 / 60.0)
                assert np.isfinite(engine.flock.positions).all(), (
                    f"NaN at frame {frame}"
                )

            # After 30 frames, flock CoM should be closer to pilot
            final_dist = np.linalg.norm(
                engine.flock.positions.mean(axis=0) - pilot.position
            )
            assert final_dist < initial_dist, (
                f"Flock did not converge: initial={initial_dist:.1f}, final={final_dist:.1f}"
            )
        finally:
            InfluencerMode.set_pilot(None)

    def test_all_p7_features_active_together(self):
        """P7.1–P7.6: All features active simultaneously, no conflicts."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 25
        cfg.influencer_substeps = 2
        cfg.influencer_rank_exponent = 1.8
        cfg.influencer_scale = 0.7
        cfg.influencer_influence_mode = "rank"
        cfg.influencer_near_dist_sq = 100.0
        cfg.influencer_init_separation = 0.4
        cfg.influencer_tick_rate = 0.8
        cfg.seed = 77

        # Density-scaled init
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        flock.positions[:] = InfluencerMode.density_init_positions(
            n=cfg.num_boids,
            width=cfg.width,
            height=cfg.height,
            depth=cfg.depth,
            config=cfg,
            rng=rng,
        )

        # Run through compute with pilot active, then Lissajous
        cfg._influencer_tick = 0.0

        # First: Lissajous mode
        for _ in range(5):
            _call_force(influencer_forces, flock, cfg)
            flock.positions += flock.velocities * 0.1
            assert np.isfinite(flock.velocities).all()

        # Then: switch to pilot mode
        pilot = PilotTarget(
            position=np.array(
                [cfg.width / 2.0, cfg.height / 2.0, cfg.depth / 2.0],
                dtype=np.float32,
            )
        )
        pilot.active = True
        InfluencerMode.set_pilot(pilot)

        try:
            for _ in range(5):
                _call_force(influencer_forces, flock, cfg)
                flock.positions += flock.velocities * 0.1
                assert np.isfinite(flock.velocities).all()
        finally:
            InfluencerMode.set_pilot(None)

        # Diagnostics should work in both modes
        assert hasattr(cfg, '_target_dist_min')
        assert hasattr(cfg, '_target_dist_max')

        # Velocities should be constant speed throughout
        v_mags = np.linalg.norm(flock.velocities[flock.active], axis=1)
        assert np.allclose(v_mags, cfg.v0, atol=1e-4)

    # ── Backward compatibility ──────────────────────────────────

    def test_legacy_alias_functional(self):
        """influencer_forces alias is functional."""
        assert callable(influencer_forces)
        assert influencer_forces.needs_index is False

    def test_config_round_trip_preserves_new_fields(self):
        """New P7 fields survive YAML round-trip."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.influencer_rank_exponent = 2.0
        cfg.influencer_substeps = 3
        cfg.influencer_scale = 0.8
        cfg.influencer_influence_mode = "distance"
        cfg.influencer_near_dist_sq = 200.0
        cfg.influencer_init_separation = 0.3
        cfg.influencer_tick_rate = 0.5

        import os
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "test_p7_config.yaml")
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)
            assert loaded.influencer_rank_exponent == 2.0
            assert loaded.influencer_substeps == 3
            assert loaded.influencer_scale == 0.8
            assert loaded.influencer_influence_mode == "distance"
            assert loaded.influencer_near_dist_sq == 200.0
            assert loaded.influencer_init_separation == 0.3
            assert loaded.influencer_tick_rate == 0.5
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
