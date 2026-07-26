"""Array init helpers — position/velocity generation strategies.

Level 0 — depends only on numpy. Split out of boid.py (file-size split,
pure extraction, no behavior change) — boid.py keeps the integration
kernel; this module holds the position/velocity init strategies that
integrate() itself never calls.
"""

from __future__ import annotations

import numpy as np

# ── Array helpers ─────────────────────────────────────────────────

def random_positions(
    n: int,
    width: float,
    height: float,
    depth: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """N random points uniformly distributed in the 3D domain volume.

    Returns (N, 3) float32.
    """
    rng = rng or np.random.default_rng()
    return rng.uniform(
        low=(0.0, 0.0, 0.0),
        high=(width, height, depth),
        size=(n, 3),
    ).astype(np.float32)


def random_unit_sphere(
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """N random directions uniformly on the 3D unit sphere.

    Uses Marsaglia's method — rejection-free, uniform.
    Returns (N, 3) float32.
    """
    rng = rng or np.random.default_rng()
    # Generate random points in [-1, 1]³ and reject those outside sphere
    pts = rng.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def init_positions(
    n: int,
    width: float,
    height: float,
    depth: float,
    rng: np.random.Generator | None = None,
    mode: str = "box",
    separation: float = 9.0,
) -> np.ndarray:
    """Generate initial flock positions using one of 5 strategies.

    Args:
        n: number of birds
        width, height, depth: domain bounds
        rng: seeded random generator
        mode: "box" | "sphere_shell" | "gaussian" | "grid" | "blob"
        separation: bird body size for overlap prevention

    Returns:
        (n, 3) float32 position array

    For blob velocities, use init_velocities_blob() — P3.10 drift-biased tangentials.
    """
    rng = rng or np.random.default_rng()
    C = np.array([width / 2, height / 2, depth / 2], dtype=np.float32)

    if mode == "sphere_shell":
        R = 0.4 * min(width, height, depth)
        dirs = random_unit_sphere(n, rng)
        return (C + dirs * R).astype(np.float32)

    elif mode == "sphere":
        # D7: Volume-uniform positions inside a sphere — ∛-law.
        # Radial distribution P(r) ∝ r² for uniform volume density.
        # cbrt of uniform [0,1] gives the correct radial CDF.
        R = 0.4 * min(width, height, depth)
        r = rng.uniform(0, 1, (n, 1)).astype(np.float32)
        r = np.cbrt(r) * R  # ∛-law: uniform in volume
        dirs = random_unit_sphere(n, rng)
        return (C + dirs * r).astype(np.float32)

    elif mode == "gaussian":
        sigma = n ** (1.0 / 3.0) * separation
        pts = rng.normal(0.0, sigma, (n, 3)).astype(np.float32)
        return C + pts

    elif mode == "grid":
        pts_per_axis = int(np.ceil(n ** (1.0 / 3.0)))
        xs = np.linspace(0, width * 0.9, pts_per_axis, dtype=np.float32)
        ys = np.linspace(0, height * 0.9, pts_per_axis, dtype=np.float32)
        zs = np.linspace(0, depth * 0.9, pts_per_axis, dtype=np.float32)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        # Trim to exact n (grid always has >= n points)
        return grid_pts[:n].astype(np.float32)

    elif mode == "blob":
        # 5 centres roughly evenly distributed in domain
        centres = C + np.array([
            [-0.48,  0.18,  0.12],
            [ 0.36, -0.20, -0.28],
            [ 0.12,  0.34,  0.42],
            [-0.16, -0.30,  0.34],
            [ 0.48,  0.16,  0.18],
        ], dtype=np.float32) * min(width, height, depth) * 0.4
        radii = rng.uniform(
            0.0, 1.0, (n, 1)
        ).astype(np.float32)  # ∛-uniform via cbrt later
        radii = np.cbrt(radii) * (0.22 + rng.uniform(0.0, 1.0, (n, 1)).astype(np.float32) * 0.28)
        radii *= min(width, height, depth) * 0.4
        dirs = random_unit_sphere(n, rng)
        centre_idx = rng.integers(0, len(centres), n)
        centre_pos = centres[centre_idx]
        jitter = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32) * 0.045 * min(width, height, depth)
        return (centre_pos + dirs * radii + jitter).astype(np.float32)

    else:  # "box" (legacy)
        return random_positions(n, width, height, depth, rng)


# ── P3.10: Blob velocity init (drift-biased tangential) ────────────

def init_velocities_blob(
    n: int,
    v0: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """P3.10: Drift-biased tangential velocities for blob mode.

    v = ((0.34 ± 0.08), ±0.16, (0.08 ± 0.08)) · v0 · 0.5 + jitter(0.05·v0)

    Creates a gentle forward drift (positive x) with modest y/z spread.
    S2.A9: an independent per-axis jitter term (magnitude 0.05·v0) is
    added on top of the drift vector so per-bird velocities aren't
    perfectly correlated across axes at the v0·0.5 scale.

    Returns (n, 3) float32 velocity array.
    """
    rng = rng or np.random.default_rng()
    v = np.empty((n, 3), dtype=np.float32)
    # x: 0.34 ± 0.08 → uniform(0.26, 0.42)
    v[:, 0] = 0.34 + rng.uniform(-0.08, 0.08, n).astype(np.float32)
    # y: ±0.16 → uniform(-0.16, 0.16)
    v[:, 1] = rng.uniform(-0.16, 0.16, n).astype(np.float32)
    # z: 0.08 ± 0.08 → uniform(0.0, 0.16)
    v[:, 2] = 0.08 + rng.uniform(-0.08, 0.08, n).astype(np.float32)
    v = v * v0 * 0.5
    # S2.A9: jitter(0.05·v0) — independent per-axis noise
    jitter = rng.uniform(-0.05, 0.05, (n, 3)).astype(np.float32) * v0
    return (v + jitter).astype(np.float32)


# ── P4.9: Velocity-init variants ─────────────────────────

def init_velocities_cube(
    n: int,
    v0: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """P4.9: Velocities uniformly distributed in a cube [−v0, v0]³.

    Produces a wider speed distribution than sphere sampling —
    birds near cube corners have |v| ≈ v0√3 ≈ 1.73·v0.
    Mean speed ≈ 0.96·v0 (expected value of ‖U(−1,1)³‖).

    S2.B9: verified against the roadmap's own bug list — this mode is
    NOT flagged as divergent (only speed_uniform's lower bound and
    tangential's constant speed are), despite the roadmap's separate
    *math* line writing the law as `(U³−0.5)·2v0`. Left as-is.

    Returns (n, 3) float32 velocity array.
    """
    rng = rng or np.random.default_rng()
    return rng.uniform(-v0, v0, (n, 3)).astype(np.float32)


def init_velocities_speed_uniform(
    n: int,
    v0: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """S2.B9: Uniform speed in [min(1, 0.3·v0), v0] with random sphere directions.

    Unlike fixed-speed sphere, this produces a flat speed histogram,
    giving natural variation in individual bird speeds from the start.
    The lower bound keeps a floor speed at low v0 (matches
    speed_min_factor's 0.3 band-clamp default) rather than allowing
    near-zero initial speeds.

    Returns (n, 3) float32 velocity array.
    """
    rng = rng or np.random.default_rng()
    dirs = random_unit_sphere(n, rng)
    lo = min(1.0, 0.3 * v0)
    speeds = rng.uniform(lo, v0, (n, 1)).astype(np.float32)
    return (dirs * speeds).astype(np.float32)


def init_velocities_tangential(
    n: int,
    v0: float,
    rng: np.random.Generator | None = None,
    center: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """S2.B9: Tangential velocities — perpendicular to radial from centre.

    Each bird orbits the flock centre rather than moving radially.
    Uses Gram–Schmidt to find a random perpendicular direction from
    the radial vector, then scales by a random speed in U(1, v0) (was
    a constant v0 — the spec wants per-bird speed variation here too).

    Args:
        n: number of birds
        v0: cruise speed
        rng: seeded generator
        center: (3,) float32 — orbit centre (defaults to origin)
        positions: (n, 3) float32 — bird positions for radial computation

    Returns (n, 3) float32 velocity array.
    """
    rng = rng or np.random.default_rng()
    if center is None:
        center = np.zeros(3, dtype=np.float32)
    speeds = rng.uniform(min(1.0, v0), max(1.0, v0), n).astype(np.float32)
    if positions is None:
        # No positions → fall back to random_unit_sphere (reasonable default)
        return random_unit_sphere(n, rng) * speeds[:, np.newaxis]

    v = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        radial = positions[i] - center
        r_norm = np.linalg.norm(radial)
        if r_norm < 1e-6:
            # At centre → use random direction
            v[i] = random_unit_sphere(1, rng).ravel() * speeds[i]
            continue
        radial /= r_norm
        # Pick a random vector and Gram–Schmidt out the radial component
        rand_vec = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        tangent = rand_vec - radial * np.dot(radial, rand_vec)
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-6:
            tangent = np.cross(radial, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                tangent = np.cross(radial, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                t_norm = np.linalg.norm(tangent)
        v[i] = (tangent / t_norm) * speeds[i]
    return v


def init_velocities_fixed(
    n: int,
    v0: float,
    direction: tuple[float, float, float] = (0.6, 0.0, 0.4),
) -> np.ndarray:
    """P4.9: All birds get the same fixed velocity direction at v0.

    Deterministic — no RNG needed. Useful for testing and for
    configurations where uniform initial heading is desired.

    Args:
        n: number of birds
        v0: cruise speed
        direction: (dx, dy, dz) — unit direction (normalised internally)

    Returns (n, 3) float32 velocity array.
    """
    d = np.array(direction, dtype=np.float32)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-6:
        d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        d = d / d_norm
    return np.tile(d * v0, (n, 1)).astype(np.float32)


# ── Unified velocity-init dispatch ─────────────────────────

def init_velocities(
    n: int,
    v0: float,
    rng: np.random.Generator | None = None,
    mode: str = "sphere",
    center: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """P4.9: Generate initial velocities using one of 6 strategies.

    Args:
        n: number of birds
        v0: cruise speed
        rng: seeded random generator
        mode: "sphere" | "blob" | "drift" | "cube" | "speed_uniform" |
              "tangential" | "fixed" ("drift" is a C3 alias for "blob" —
              the drift-biased tangential velocity init doubles as both)
        center: (3,) orbit centre for tangential mode
        positions: (n, 3) for tangential mode radial computation

    Returns:
        (n, 3) float32 velocity array
    """
    rng = rng or np.random.default_rng()

    if mode in ("blob", "drift"):  # C3: "drift" aliases "blob"
        return init_velocities_blob(n, v0, rng)
    elif mode == "cube":
        return init_velocities_cube(n, v0, rng)
    elif mode == "speed_uniform":
        return init_velocities_speed_uniform(n, v0, rng)
    elif mode == "tangential":
        return init_velocities_tangential(n, v0, rng, center, positions)
    elif mode == "fixed":
        return init_velocities_fixed(n, v0)
    else:  # "sphere" (default, backward-compatible)
        return random_unit_sphere(n, rng) * v0 * 0.8
