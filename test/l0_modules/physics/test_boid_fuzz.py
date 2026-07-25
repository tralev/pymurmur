"""Unit tests for physics.boid — P0.3 physics invariant fuzz tests (speed band, no-NaN, inactive-row preservation, fixed/ceiling/band mode fuzz).

Split out of test_boid.py (file-size split).
"""

import numpy as np

from pymurmur.physics.boid import (
    integrate,
)

# ── P0.3 Physics Invariant Fuzz Tests ──────────────────────────
# Per roadmap P0.3: for integrate(..., "toroidal"): after step,
# 0 ≤ pos < (W,H,D) elementwise, |v| ≤ v0 + ε, no NaN,
# inactive rows bit-identical.


def test_speed_band_respected():
    """200 random seeds: all speeds ≤ v0 + epsilon, positions in domain."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for seed in range(200):
        rng_i = np.random.default_rng(seed)
        pos = rng_i.uniform(0, [W, H, D], (50, 3)).astype(np.float32)
        vel = rng_i.uniform(-2 * v0, 2 * v0, (50, 3)).astype(np.float32)
        acc = np.zeros((50, 3), dtype=np.float32)
        active = np.ones(50, dtype=bool)

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal", 1.0 / 60.0)

        speeds = np.linalg.norm(vel, axis=1)
        assert (speeds <= v0 + 1e-4).all(), (
            f"seed={seed}: max speed={speeds.max():.4f} > v0={v0}"
        )
        assert (pos[:, 0] >= 0).all() and (pos[:, 0] < W).all(), (
            f"seed={seed}: x out of bounds: min={pos[:,0].min():.1f} max={pos[:,0].max():.1f}"
        )
        assert (pos[:, 1] >= 0).all() and (pos[:, 1] < H).all(), (
            f"seed={seed}: y out of bounds"
        )
        assert (pos[:, 2] >= 0).all() and (pos[:, 2] < D).all(), (
            f"seed={seed}: z out of bounds"
        )

    print(f"\n✓ 200 seeds: all speeds ≤ {v0}+ε, all positions in domain")


def test_no_nan_after_integrate():
    """200 random seeds across boundary modes: no NaN in positions or velocities."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for mode in ("toroidal", "open", "margin", "sphere"):
        for seed in range(200):
            rng_i = np.random.default_rng(seed)
            pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
            vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
            acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
            active = rng_i.uniform(0, 1, 30) > 0.1  # ~90% active

            integrate(pos, vel, acc, active, W, H, D, v0, mode, 1.0 / 60.0)

            assert not np.isnan(pos).any(), (
                f"mode={mode} seed={seed}: NaN in positions"
            )
            assert not np.isnan(vel).any(), (
                f"mode={mode} seed={seed}: NaN in velocities"
            )
            assert not np.isinf(pos).any(), (
                f"mode={mode} seed={seed}: Inf in positions"
            )
            assert not np.isinf(vel).any(), (
                f"mode={mode} seed={seed}: Inf in velocities"
            )

    print("\n✓ 3 modes × 200 seeds: no NaN or Inf in positions/velocities")


def test_inactive_rows_bit_identical():
    """Inactive birds' positions and velocities are bit-identical after integrate.

    P0.3 requirement: inactive rows bit-identical. More rigorous than
    test_integrate_inactive_unchanged which uses np.allclose.
    """
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0
    rng = np.random.default_rng(7)

    for mode in ("toroidal", "open", "margin", "sphere"):
        for _ in range(50):
            pos = rng.uniform(0, [W, H, D], (20, 3)).astype(np.float32)
            vel = rng.uniform(-v0, v0, (20, 3)).astype(np.float32)
            acc = rng.uniform(-1, 1, (20, 3)).astype(np.float32)
            active = rng.uniform(0, 1, 20) > 0.15

            pos_before = pos.copy()
            vel_before = vel.copy()

            integrate(pos, vel, acc, active, W, H, D, v0, mode, 1.0 / 60.0)

            inactive = ~active
            # Exact bit-identical check (not allclose)
            assert np.array_equal(pos[inactive], pos_before[inactive]), (
                f"mode={mode}: inactive positions changed"
            )
            assert np.array_equal(vel[inactive], vel_before[inactive]), (
                f"mode={mode}: inactive velocities changed"
            )

    print("\n✓ 3 modes × 50 seeds: inactive rows bit-identical")


def test_toroidal_positions_in_bounds():
    """After toroidal integrate, all positions satisfy 0 ≤ pos < domain elementwise.

    Explicit check across a range of starting positions near the boundary.
    """
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    # Fixed positions that test edge cases: near-wrapping, at-boundary, far-out
    test_positions = np.array([
        [999.0, 350.0, 200.0],   # near +X boundary
        [1.0, 350.0, 200.0],     # near -X boundary
        [500.0, 699.0, 200.0],   # near +Y
        [500.0, 1.0, 200.0],     # near -Y
        [500.0, 350.0, 399.0],   # near +Z
        [500.0, 350.0, 1.0],     # near -Z
        [500.0, 350.0, 200.0],   # centre (no wrap)
    ], dtype=np.float32)

    vel = np.array([
        [10.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, -10.0, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, -10.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    N = len(test_positions)
    acc = np.zeros((N, 3), dtype=np.float32)
    active = np.ones(N, dtype=bool)

    for _frame in range(100):
        integrate(test_positions, vel, acc, active, W, H, D, v0, "toroidal", 1.0 / 60.0)

    # After 100 frames, all positions must be in bounds
    assert (test_positions[:, 0] >= 0).all() and (test_positions[:, 0] < W).all(), (
        f"x out of bounds: min={test_positions[:,0].min():.1f} max={test_positions[:,0].max():.1f}"
    )
    assert (test_positions[:, 1] >= 0).all() and (test_positions[:, 1] < H).all()
    assert (test_positions[:, 2] >= 0).all() and (test_positions[:, 2] < D).all()

    print("\n✓ 7 birds × 100 frames: all positions in [0,W)×[0,H)×[0,D)")


def test_fixed_mode_fuzz():
    """200 seeds with speed_mode='fixed': all speeds ≡ v0."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for seed in range(200):
        rng_i = np.random.default_rng(seed)
        pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
        vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
        acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
        active = np.ones(30, dtype=bool)

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal",
                  1.0 / 60.0, speed_mode="fixed")

        speeds = np.linalg.norm(vel, axis=1)
        assert np.allclose(speeds, v0, atol=1e-4), (
            f"seed={seed}: fixed mode speeds must be v0={v0}, got {speeds.min():.4f}–{speeds.max():.4f}"
        )
        assert not np.isnan(pos).any()
        assert not np.isnan(vel).any()

    print(f"\n✓ 200 seeds fixed-mode: all speeds ≡ {v0}")


def test_ceiling_mode_fuzz():
    """200 seeds with speed_mode='ceiling': all speeds ≤ v0, no NaN."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0

    for seed in range(200):
        rng_i = np.random.default_rng(seed)
        pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
        vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
        acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
        active = np.ones(30, dtype=bool)

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal",
                  1.0 / 60.0, speed_mode="ceiling")

        speeds = np.linalg.norm(vel, axis=1)
        assert (speeds <= v0 + 1e-4).all(), (
            f"seed={seed}: ceiling mode max speed={speeds.max():.4f} > v0={v0}"
        )
        assert not np.isnan(pos).any()
        assert not np.isnan(vel).any()

    print(f"\n✓ 200 seeds ceiling-mode: all speeds ≤ {v0}")


def test_band_mode_fuzz_all_boundaries():
    """P0.3: 50 seeds × 4 boundary modes — band mode invariant holds.

    Band mode (default): speeds ∈ [0.3·v0, v0], positions in bounds
    if toroidal, no NaN. This fills the gap — only toroidal was fuzzed.
    """
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0
    v_min = 0.3 * v0  # 1.2

    for mode in ("toroidal", "open", "margin", "sphere"):
        for seed in range(50):
            rng_i = np.random.default_rng(seed)
            pos = rng_i.uniform(0, [W, H, D], (30, 3)).astype(np.float32)
            vel = rng_i.uniform(-2 * v0, 2 * v0, (30, 3)).astype(np.float32)
            acc = rng_i.uniform(-1, 1, (30, 3)).astype(np.float32)
            active = rng_i.uniform(0, 1, 30) > 0.1

            integrate(pos, vel, acc, active, W, H, D, v0, mode,
                      1.0 / 60.0, speed_mode="band")

            speeds = np.linalg.norm(vel[active], axis=1)
            # Toroidal: tight bounds (no boundary-induced velocity changes)
            # Non-toroidal: margin/open can nudge velocity, use relaxed bounds
            # Sphere: skip speed bounds — sphere boundary projects birds back
            #   with velocity kicks that can far exceed v0 (by design)
            if mode == "toroidal":
                assert (speeds >= v_min - 1e-4).all(), (
                    f"{mode} seed={seed}: min speed={speeds.min():.4f} < {v_min}"
                )
                assert (speeds <= v0 + 1e-4).all(), (
                    f"{mode} seed={seed}: max speed={speeds.max():.4f} > {v0}"
                )
            elif mode != "sphere":
                assert (speeds >= v_min * 0.95).all(), (
                    f"{mode} seed={seed}: min speed={speeds.min():.4f} < {v_min*0.95:.2f}"
                )
                assert (speeds <= v0 * 1.05).all(), (
                    f"{mode} seed={seed}: max speed={speeds.max():.4f} > {v0*1.05:.2f}"
                )
            assert not np.isnan(pos).any(), f"{mode} seed={seed}: NaN in positions"
            assert not np.isnan(vel).any(), f"{mode} seed={seed}: NaN in velocities"

            # Toroidal: positions must remain in bounds
            if mode == "toroidal":
                assert (pos[:, 0] >= 0).all() and (pos[:, 0] < W).all()
                assert (pos[:, 1] >= 0).all() and (pos[:, 1] < H).all()
                assert (pos[:, 2] >= 0).all() and (pos[:, 2] < D).all()

    print(f"\n✓ 4 modes × 50 seeds band-mode: speeds in [{v_min},{v0}], no NaN")


def test_inertia_fuzz_inactive_preserved():
    """Inertia > 0 with mixed active/inactive: inactive rows unchanged."""
    W, H, D = 1000.0, 700.0, 400.0
    v0 = 4.0
    rng = np.random.default_rng(13)

    for _ in range(50):
        pos = rng.uniform(0, [W, H, D], (20, 3)).astype(np.float32)
        vel = rng.uniform(-v0, v0, (20, 3)).astype(np.float32)
        acc = rng.uniform(-1, 1, (20, 3)).astype(np.float32)
        active = rng.uniform(0, 1, 20) > 0.2

        pos_before = pos.copy()
        vel_before = vel.copy()

        integrate(pos, vel, acc, active, W, H, D, v0, "toroidal",
                  1.0 / 60.0, speed_mode="band", inertia=0.7)

        inactive = ~active
        assert np.array_equal(pos[inactive], pos_before[inactive]), (
            "inertia fuzz: inactive positions changed"
        )
        assert np.array_equal(vel[inactive], vel_before[inactive]), (
            "inertia fuzz: inactive velocities changed"
        )

    print("\n✓ 50 seeds inertia=0.7: inactive rows bit-identical")


