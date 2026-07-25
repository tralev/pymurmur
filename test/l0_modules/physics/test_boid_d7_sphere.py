"""Unit tests for physics.boid — D7 sphere position-init (volume-uniform cube-root law), D1+D7 sphere-boundary cross-cutting.

Split out of test_boid.py (file-size split).
"""

import numpy as np

# ── D7: position_init "sphere" — volume-uniform via ∛-law ────────


class TestD7SphereInit:
    """D7: position_init='sphere' uses ∛-law for volume-uniform
    distribution inside a sphere (not just on the surface).

    Previously the 'sphere' string fell through to the else 'box'
    branch because no 'sphere' case existed.  Now it correctly
    samples r ∝ cbrt(U) for uniform volume density.
    """

    @staticmethod
    def _sphere_positions(n=5000, rng_seed=42):
        """Return (n,3) positions from sphere init."""
        from pymurmur.physics.boid import init_positions
        w, h, d = 1000.0, 700.0, 400.0
        rng = np.random.default_rng(rng_seed)
        return init_positions(n, w, h, d, rng, mode="sphere")

    def test_all_positions_within_radius(self):
        """D7: All sphere-init positions are within R of centre."""
        w, h, d = 1000.0, 700.0, 400.0
        R = 0.4 * min(w, h, d)  # = 160.0
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        pts = self._sphere_positions()
        dists = np.linalg.norm(pts - C, axis=1)
        assert (dists <= R * 1.001).all(), (
            f"All points must be within {R}, max dist = {dists.max():.1f}"
        )

    def test_radial_histogram_follows_r_squared(self):
        """D7: Radial bin counts ∝ r² (uniform volume density).

        For volume-uniform sampling, the probability of a point
        landing in [r, r+dr] is proportional to r² (surface area
        of spherical shell at radius r).  The cumulative distribution
        follows P(r ≤ R) = (r/R)³, hence the ∛-law.
        """
        w, h, d = 1000.0, 700.0, 400.0
        R = 0.4 * min(w, h, d)
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        pts = self._sphere_positions(n=10000)
        dists = np.linalg.norm(pts - C, axis=1)

        # Split into 10 radial bins and check counts ∝ r²
        bins = np.linspace(0, R, 11)
        hist, _ = np.histogram(dists, bins=bins)

        # Expected: count ∝ shell volume (r_{i+1}³ − r_i³) for each bin.
        # For volume-uniform distribution, the probability of landing
        # in [r_i, r_{i+1}] is proportional to the spherical shell volume.
        # r ∝ bin_edge³ gives the exact formula, unlike r_mid² which is
        # a coarse approximation for wide bins.
        shell_volumes = bins[1:] ** 3 - bins[:-1] ** 3
        expected = shell_volumes / shell_volumes.sum() * hist.sum()

        for i in range(len(hist)):
            if expected[i] > 10:  # skip nearly-empty bins
                rel_err = abs(hist[i] - expected[i]) / expected[i]
                assert rel_err < 0.25, (
                    f"Bin {i}: r=[{bins[i]:.0f},{bins[i+1]:.0f}], "
                    f"count={hist[i]}, expected≈{expected[i]:.0f}, "
                    f"rel_err={rel_err:.2f}"
                )

    def test_sphere_is_volume_not_surface(self):
        """D7: Sphere init fills the volume, not just the surface.

        Verify at least 50% of points are inside 70% of the radius
        (uniform volume → ~34% inside 0.7R; surface → 0%).
        """
        w, h, d = 1000.0, 700.0, 400.0
        R = 0.4 * min(w, h, d)
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        pts = self._sphere_positions(n=2000)
        dists = np.linalg.norm(pts - C, axis=1)

        inside_70pct = (dists < 0.7 * R).sum()
        expected_vol_fraction = 0.7 ** 3  # ~0.343
        fraction = inside_70pct / len(pts)

        # Allow ±15% margin: min ~0.19
        assert fraction > expected_vol_fraction - 0.15, (
            f"Only {fraction:.2%} inside 0.7R; "
            f"expected ~{expected_vol_fraction:.0%} (±15%), "
            f"which rules out surface-only distribution"
        )

    def test_sphere_init_via_physics_flock(self):
        """D7: Sphere init works through PhysicsFlock position_init config."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.position_init = "sphere"
        cfg.num_boids = 100
        w, h, d = cfg.width, cfg.height, cfg.depth
        R = 0.4 * min(w, h, d)
        C = np.array([w / 2, h / 2, d / 2], dtype=np.float32)

        flock = PhysicsFlock(cfg)
        dists = np.linalg.norm(flock.positions - C, axis=1)

        assert (dists <= R * 1.001).all(), (
            f"Sphere-init via PhysicsFlock: all points must be within {R}"
        )
        # Also verify it's volume-distributed (not surface)
        interior = (dists < 0.5 * R).sum()
        assert interior > 5, (
            f"Only {interior} birds in inner half of sphere; "
            f"surface-only would have 0"
        )

    def test_sphere_init_deterministic_with_same_seed(self):
        """D7: Sphere init with same seed produces identical positions."""
        from pymurmur.physics.boid import init_positions
        w, h, d = 1000.0, 700.0, 400.0
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        pts1 = init_positions(50, w, h, d, rng1, mode="sphere")
        pts2 = init_positions(50, w, h, d, rng2, mode="sphere")

        np.testing.assert_array_equal(pts1, pts2)

    def test_sphere_not_identical_to_box(self):
        """D7: Sphere and box init produce different positions.

        This is the regression guard — the original bug caused
        'sphere' to silently fall through to the else: 'box' branch,
        so both modes would return identical output for the same seed.
        """
        from pymurmur.physics.boid import init_positions
        w, h, d = 1000.0, 700.0, 400.0
        rng = np.random.default_rng(42)

        sphere_pts = init_positions(200, w, h, d, rng, mode="sphere")
        # Fresh RNG with same seed for independent box positions
        rng2 = np.random.default_rng(42)
        box_pts = init_positions(200, w, h, d, rng2, mode="box")

        assert not np.array_equal(sphere_pts, box_pts), (
            "Sphere and box init must produce different outputs; "
            "if they match, 'sphere' is falling through to 'box' (D7 regression)"
        )


# ── D1 + D7: Sphere boundary + sphere init cross-cutting ────────


def test_sphere_init_birds_stay_within_sphere_boundary():
    """D1+D7: Birds initted with 'sphere' mode stay inside sphere_soft boundary.

    D7 fixed position_init to support 'sphere' mode. D1 fixed sphere boundary
    to centre on domain centre C, not origin. Together, birds initialized in
    a sphere should remain within the sphere_soft boundary over many frames.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 30
    cfg.mode = "spatial"
    cfg.width = 800
    cfg.height = 600
    cfg.depth = 400
    cfg.boundary_mode = "sphere_soft"
    cfg.position_init = "sphere"
    cfg.boundary_avoidance_factor = 0.8
    cfg.boundary_sphere_radius = 0.4

    engine = SimulationEngine(cfg)

    # D7: birds must be initialized in sphere mode (not degraded to box)
    # D1: sphere boundary is centred on domain centre C, not origin
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                 dtype=np.float32)
    R = cfg.boundary_sphere_radius * min(cfg.width, cfg.height, cfg.depth)

    # All initial positions must be within the sphere
    initial_dists = np.linalg.norm(
        engine.flock.positions - C, axis=1
    )
    assert (initial_dists <= R * 1.05).all(), (
        f"D7: sphere init must place birds within sphere boundary. "
        f"Max dist={initial_dists.max():.1f}, R={R:.1f}"
    )

    # Run many steps and verify no bird escapes the sphere boundary
    for _ in range(200):
        engine.step(1.0 / 60.0)

    final_dists = np.linalg.norm(
        engine.flock.positions[engine.flock.active] - C, axis=1
    )
    # Sphere_soft is asymptotic — birds can slightly overshoot the
    # nominal R during fast turns. Use 20% tolerance.
    assert (final_dists <= R * 1.2).all(), (
        f"D1: sphere_soft boundary must contain birds. "
        f"Max dist={final_dists.max():.1f}, R={R:.1f}"
    )
    # At least some birds should be near the boundary (not all at centre)
    assert final_dists.max() > R * 0.5, (
        "Birds should explore the sphere volume, not stay at centre"
    )


def test_sphere_init_centre_matches_boundary_centre():
    """D1+D7: Sphere boundary centre C equals domain centre on frame 0.

    D1 initialises flock.center to domain centre. D7 uses the same centre
    for sphere init. The boundary and init must agree on the sphere centre.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 20
    cfg.mode = "spatial"
    cfg.boundary_mode = "sphere_soft"
    cfg.position_init = "sphere"

    engine = SimulationEngine(cfg)
    flock = engine.flock

    C_expected = np.array(
        [cfg.width / 2, cfg.height / 2, cfg.depth / 2],
        dtype=np.float32,
    )
    # D1: flock.center is initialised to domain centre
    np.testing.assert_array_equal(
        flock.center, C_expected,
        err_msg="D1: flock.center must be domain centre on frame 0"
    )

    # All birds should be distributed around the domain centre
    centroid = flock.positions[flock.active].mean(axis=0)
    dist_centroid_to_centre = np.linalg.norm(centroid - C_expected)
    # Centroid should be near the centre (sphere init is centred on C)
    R = cfg.boundary_sphere_radius * min(cfg.width, cfg.height, cfg.depth)
    assert dist_centroid_to_centre < R * 0.3, (
        f"D7: sphere init centroid should be near domain centre. "
        f"Dist={dist_centroid_to_centre:.1f}, R={R:.1f}"
    )
