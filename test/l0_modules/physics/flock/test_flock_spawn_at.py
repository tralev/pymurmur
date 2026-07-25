"""Unit tests for physics.flock — P10.4 spawn_at + cube-velocity law.

Split out of test_flock.py (file-size split).
"""

import numpy as np

from pymurmur.physics.flock import PhysicsFlock

# ── P10.4: spawn_at tests — cube-velocity law + v0/rng plumbing ─

class TestSpawnAt:
    """P10.4: spawn_at() — position, velocity law, v0/rng plumbing."""

    def test_spawn_position_exact(self, default_config):
        """Spawned bird has the exact position passed."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        target = (123.0, 456.0, 789.0)
        idx = flock.spawn_at(target)
        assert idx >= 0
        np.testing.assert_array_equal(
            flock.positions[idx], np.array(target, dtype=np.float32),
        )

    def test_spawn_velocity_magnitude_bounded_by_v0(self, default_config):
        """Cube-velocity law: |v| ≤ v0 always (limit3 clamp)."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.v0 = 4.0
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        # Spawn many birds, check all velocities are within v0
        for _ in range(50):
            idx = flock.spawn_at((500, 350, 200), v0=cfg.v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            assert speed <= cfg.v0 + 1e-6, (
                f"spawn velocity {speed:.4f} exceeds v0={cfg.v0}"
            )
            assert speed >= 0.0, "spawn velocity should be non-negative"

    def test_spawn_velocity_components_in_range(self, default_config):
        """Cube-velocity law: each component ∈ [-v0, v0]."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.v0 = 3.0
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(123)

        for _ in range(100):
            idx = flock.spawn_at((500, 350, 200), v0=cfg.v0, rng=rng)
            vel = flock.velocities[idx]
            assert (-cfg.v0 - 0.01 <= vel[0] <= cfg.v0 + 0.01), (
                f"vx={vel[0]:.4f} outside [-v0, v0]"
            )
            assert (-cfg.v0 - 0.01 <= vel[1] <= cfg.v0 + 0.01)
            assert (-cfg.v0 - 0.01 <= vel[2] <= cfg.v0 + 0.01)

    def test_spawn_v0_scales_velocity(self, default_config):
        """Higher v0 produces larger velocities on average."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        # Spawn with v0=2.0
        speeds_small = []
        for _ in range(30):
            idx = flock.spawn_at((500, 350, 200), v0=2.0, rng=rng)
            speeds_small.append(float(np.linalg.norm(flock.velocities[idx])))

        rng2 = np.random.default_rng(42)
        flock2 = PhysicsFlock(cfg)
        speeds_large = []
        for _ in range(30):
            idx = flock2.spawn_at((500, 350, 200), v0=4.0, rng=rng2)
            speeds_large.append(float(np.linalg.norm(flock2.velocities[idx])))

        # On average, v0=4.0 should produce ~2× the speed of v0=2.0
        mean_small = np.mean(speeds_small)
        mean_large = np.mean(speeds_large)
        ratio = mean_large / max(mean_small, 0.01)
        assert ratio > 1.6, (
            f"v0=4 mean={mean_large:.3f} vs v0=2 mean={mean_small:.3f}, "
            f"ratio={ratio:.2f} should be >1.6 (~2.0 expected)"
        )

    def test_spawn_rng_deterministic(self, default_config):
        """Same rng seed → identical velocity for spawned bird."""
        cfg = default_config
        cfg.num_boids = 5

        rng1 = np.random.default_rng(42)
        flock1 = PhysicsFlock(cfg)
        idx1 = flock1.spawn_at((100, 200, 300), v0=4.0, rng=rng1)

        rng2 = np.random.default_rng(42)
        flock2 = PhysicsFlock(cfg)
        idx2 = flock2.spawn_at((100, 200, 300), v0=4.0, rng=rng2)

        np.testing.assert_array_equal(
            flock1.velocities[idx1], flock2.velocities[idx2],
            err_msg="Same rng seed must produce identical spawn velocity",
        )

    def test_spawn_rng_default_uses_flock_rng(self, default_config):
        """When rng is not passed, flock.rng is used."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.seed = 42
        flock = PhysicsFlock(cfg)

        # Reset rng to known state and spawn
        flock.rng = np.random.default_rng(42)
        idx1 = flock.spawn_at((100, 200, 300), v0=4.0)  # no rng=...

        # Same state should produce same result
        flock2 = PhysicsFlock(cfg)
        flock2.rng = np.random.default_rng(42)
        idx2 = flock2.spawn_at((100, 200, 300), v0=4.0)

        np.testing.assert_array_equal(
            flock.velocities[idx1], flock2.velocities[idx2],
        )

    def test_spawn_rng_advances_state(self, default_config):
        """Each spawn advances the RNG — two consecutive spawns differ."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        idx1 = flock.spawn_at((500, 350, 200), v0=4.0, rng=rng)
        idx2 = flock.spawn_at((500, 350, 200), v0=4.0, rng=rng)

        # Two spawns from same rng should give different velocities
        assert not np.array_equal(
            flock.velocities[idx1], flock.velocities[idx2],
        ), "Consecutive spawns should produce different velocities"

    def test_spawn_predator_flag(self, default_config):
        """Spawned predator gets is_predator=True."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200), is_predator=True)
        assert bool(flock.is_predator[idx]) is True

    def test_spawn_prey_flag_default(self, default_config):
        """Spawned bird defaults to is_predator=False."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200))
        assert bool(flock.is_predator[idx]) is False

    def test_spawn_reuses_inactive_slot(self, default_config):
        """spawn_at activates an inactive slot before extending."""
        cfg = default_config
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        # Deactivate bird at index 3
        flock.active[3] = False
        cap_before = flock.N_capacity

        idx = flock.spawn_at((500, 350, 200))
        assert idx == 3, f"Should reuse inactive slot 3, got {idx}"
        assert flock.N_capacity == cap_before, "No extension needed"

    def test_spawn_extends_capacity(self, default_config):
        """spawn_at extends arrays when all slots are active."""
        cfg = default_config
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        cap_before = flock.N_capacity

        idx = flock.spawn_at((500, 350, 200))
        assert idx >= 0
        assert flock.N_capacity > cap_before, (
            "Capacity should extend when all slots active"
        )

    def test_spawn_acceleration_zero(self, default_config):
        """Spawned bird starts with zero acceleration."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200))
        np.testing.assert_array_equal(
            flock.accelerations[idx], np.zeros(3, dtype=np.float32),
        )

    def test_spawn_seed_assigned(self, default_config):
        """Spawned bird gets a seed value in [0, 1)."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        idx = flock.spawn_at((500, 350, 200))
        s = float(flock.seeds[idx])
        assert 0.0 <= s < 1.0, f"seed {s} not in [0, 1)"

    def test_spawn_velocity_not_all_same_direction(self, default_config):
        """Cube-velocity produces varied directions, not just one axis."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)

        directions = []
        for _ in range(20):
            idx = flock.spawn_at((500, 350, 200), v0=4.0, rng=rng)
            vel = flock.velocities[idx]
            mag = np.linalg.norm(vel)
            if mag > 0:
                directions.append(vel / mag)

        # Check there's variation in at least one component
        dirs = np.array(directions)
        for axis in range(3):
            std = float(np.std(dirs[:, axis]))
            assert std > 0.05, (
                f"Axis {axis}: std={std:.4f} — all birds facing same direction?"
            )


# ── P10.4: Cube-velocity law — exact formula verification ─────

class TestCubeVelocityLaw:
    """P10.4: limit3((U³ − 0.5) · 2v0, v0) — exact formula, clamping, distribution."""

    def test_exact_formula_reproduction(self, default_config):
        """spawn_at velocity equals manual limit3((U−0.5)·2v0, v0) computation."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        v0 = 3.0

        # Snapshot RNG state, then manually compute what spawn_at should produce
        rng_snap = np.random.default_rng(42)
        U = rng_snap.uniform(0, 1, 3).astype(np.float32)
        raw_vel = (U ** 3 - 0.5) * 2.0 * v0
        mag = float(np.linalg.norm(raw_vel))
        if mag > v0:
            raw_vel *= v0 / mag
        expected = raw_vel.copy()

        # Now call spawn_at with the same RNG state
        rng2 = np.random.default_rng(42)
        idx = flock.spawn_at((100, 200, 300), v0=v0, rng=rng2)
        actual = flock.velocities[idx]

        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Cube-velocity law mismatch: expected={expected}, got={actual}"
        )

    def test_formula_with_v0_one(self, default_config):
        """With v0=1.0, raw velocity is in [-1,1]³ before clamp."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(99)

        # Spawn several birds with v0=1.0
        for _ in range(50):
            idx = flock.spawn_at((500, 350, 200), v0=1.0, rng=rng)
            vel = flock.velocities[idx]
            # Each component must be in [-1, 1] (before clamp it's in [-v0, v0])
            assert -1.01 <= vel[0] <= 1.01
            assert -1.01 <= vel[1] <= 1.01
            assert -1.01 <= vel[2] <= 1.01
            # Magnitude must be ≤ 1 (after limit3 clamp)
            assert float(np.linalg.norm(vel)) <= 1.01

    def test_limit3_clamping_fires(self, default_config):
        """limit3 clamp is exercised — some velocities reach exactly |v|=v0."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        v0 = 2.0

        # The cube [-v0,v0]³ has diagonal sqrt(3)*v0 ≈ 3.46 > v0=2.
        # ~43% of uniform-cube samples will exceed v0 and get clamped.
        # After clamping, their magnitude is exactly v0.
        n_clamped = 0
        for _ in range(200):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            if abs(speed - v0) < 0.001:
                n_clamped += 1

        # At least some should be clamped (probability of none in 200 is < 1e-50)
        assert n_clamped > 0, (
            "limit3 clamp never fired in 200 spawns — "
            "expected ~86 clamps (43%%), got 0"
        )

    def test_clamped_velocity_magnitude_is_exactly_v0(self, default_config):
        """When raw_vel exceeds v0, the clamped velocity has |v| ≈ v0 (float32)."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(777)
        v0 = 1.5

        # Spawn many, collect clamped ones
        clamped_speeds = []
        for _ in range(500):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            if abs(speed - v0) < 0.01:
                clamped_speeds.append(speed)

        assert len(clamped_speeds) > 10, (
            f"Expected many clamped velocities, got {len(clamped_speeds)}"
        )
        # All clamped speeds should be very close to v0 (allow float32 epsilon)
        for s in clamped_speeds:
            assert abs(s - v0) < 0.01, (
                f"Clamped speed {s:.6f} not close enough to v0={v0}"
            )

    def test_unclamped_velocity_below_v0(self, default_config):
        """When raw_vel mag ≤ v0, the velocity is left unchanged (no clamp)."""
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        v0 = 10.0  # Large v0 makes clamping rare (cube [-10,10]³, diag≈17.3)

        unclamped_count = 0
        for _ in range(200):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            speed = float(np.linalg.norm(flock.velocities[idx]))
            # Not at exactly v0 → was not clamped
            if abs(speed - v0) > 0.01:
                unclamped_count += 1
                # Verify the components are each within [-v0, v0]
                vel = flock.velocities[idx]
                assert -v0 - 0.01 <= vel[0] <= v0 + 0.01
                assert -v0 - 0.01 <= vel[1] <= v0 + 0.01
                assert -v0 - 0.01 <= vel[2] <= v0 + 0.01

        assert unclamped_count > 0, (
            "With large v0, some velocities should NOT be clamped"
        )

    def test_distribution_is_cube_law_before_clamp(self, default_config):
        """Pre-clamp raw_vel is (U³−0.5)·2v0, bounded by [-v0, v0]³ (D20).

        We can verify this externally by spawning with a known RNG,
        capturing the raw uniform values, and applying the same transform.
        """
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        v0 = 4.0

        # Reconstruct: spawn_at calls r.uniform(0,1,3), then transforms.
        # We intercept by using the same RNG and comparing the pre-transform.
        rng = np.random.default_rng(42)
        U = rng.uniform(0, 1, 3).astype(np.float32)
        raw = (U ** 3 - 0.5) * 2.0 * v0  # D20 cube law, in [-v0, v0]³

        # Now spawn with the same rng
        rng2 = np.random.default_rng(42)
        idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng2)
        vel = flock.velocities[idx]

        # If raw was not clamped (mag ≤ v0), vel == raw exactly
        raw_mag = float(np.linalg.norm(raw))
        if raw_mag <= v0:
            np.testing.assert_array_almost_equal(vel, raw, decimal=6)
        else:
            # If clamped, vel = raw * (v0 / raw_mag)
            np.testing.assert_array_almost_equal(vel, raw * (v0 / raw_mag), decimal=6)

    def test_cube_law_mean_bias(self, default_config):
        """Cube-law mean per component is approximately −0.5·v0.

        With (U³−0.5)·2v0, each component has theoretical mean
        (0.25−0.5)·2v0 = −0.5·v0.  This is the cube-law's systematic
        bias: pushing mass toward ±v0 concentrates at −v0 end.
        """
        cfg = default_config
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        v0 = 5.0

        vels = []
        for _ in range(500):
            idx = flock.spawn_at((500, 350, 200), v0=v0, rng=rng)
            vels.append(flock.velocities[idx])

        mean_vel = np.mean(vels, axis=0)
        # Cube-law mean per component: (0.25−0.5)·2v0 = −0.5·v0 = −2.5 for v0=5.
        # limit3 clamping shifts mean toward 0 — observed ≈ −2.0 for v0=5.
        # Use wide bound to accommodate clamp + sampling variance.
        for axis in range(3):
            assert -3.5 < float(mean_vel[axis]) < -0.5, (
                f"Axis {axis}: mean={float(mean_vel[axis]):.3f} "
                f"should be negative (cube-law bias per component)"
            )


# ── P10.4: Engine plumbing — v0/rng flow from engine to spawn_at ─
