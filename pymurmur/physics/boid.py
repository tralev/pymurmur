"""Single-bird integration kernel and array helpers.

Level 0 — depends only on core.types and core.config.
NEVER imports physics.flock or physics.forces.

The entire flock is integrated in one vectorised call — no per-bird
Python loops.  Boundary modes operate on flat arrays via boolean masks.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Vec3


# ── Integration kernel ────────────────────────────────────────────

def integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    active: np.ndarray,
    width: float,
    height: float,
    depth: float,
    v0: float,
    boundary_mode: str,
    dt: float,
    sphere_radius: float = 300.0,
    avoidance_factor: float = 0.05,
    rng: np.random.Generator | None = None,
    max_speed: np.ndarray | None = None,
    speed_mode: str = "band",
    inertia: float = 0.0,
    move: bool = True,
    speed_min_factor: float = 0.3,
    center: np.ndarray | None = None,
) -> None:
    """Vectorised Euler integration over the entire flock.

    Operates on flat arrays — no Python per-bird loop. All parameters
    are passed explicitly to avoid a SimConfig import at the hot-path level.

    speed_mode: "band" (clamp [min, cap]), "fixed" (exact renormalisation),
                "ceiling" (≤ cap only), "none" (no clamp).
    inertia: 0.0–1.0 lerp between raw and clamped velocity.
    move: if False, skip position update (caller owns positions).
    """
    # 0. Safety rails: dt clamp (P0.10)
    dt = float(np.clip(dt, 0.0, 0.05))

    # 1. Apply accumulated forces (only active birds)
    velocities[active] += accelerations[active]

    # 2. Build per-bird caps
    N = len(velocities)
    if max_speed is not None:
        caps = max_speed.astype(np.float32)
    else:
        caps = np.full(N, v0, dtype=np.float32)
    min_speed = caps * speed_min_factor

    # 3. Speed clamp — save raw velocity for inertia
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    raw_vel = velocities.copy() if inertia > 0 else None

    if speed_mode == "band":
        too_fast = (speeds.ravel() > caps).ravel() & active
        too_slow = (speeds.ravel() < min_speed).ravel() & active
        if too_fast.any():
            velocities[too_fast] = (
                velocities[too_fast] / speeds[too_fast]
            ) * caps[too_fast, np.newaxis]
        if too_slow.any():
            velocities[too_slow] = (
                velocities[too_slow] / (speeds[too_slow] + 1e-10)
            ) * min_speed[too_slow, np.newaxis]

    elif speed_mode == "fixed":
        # Exact renormalisation to cap, 0-safe: zero-velocity
        # birds get deterministic direction (1, 0, 0) to avoid NaN.
        safe_speeds = speeds + 1e-10
        dirs = velocities / safe_speeds
        zero_mask = (speeds.ravel() < 1e-6) & active
        if zero_mask.any():
            dirs[zero_mask.ravel(), 0] = 1.0
            dirs[zero_mask.ravel(), 1] = 0.0
            dirs[zero_mask.ravel(), 2] = 0.0
        velocities[active] = dirs[active] * caps[active, np.newaxis]

    elif speed_mode == "ceiling":
        too_fast = (speeds.ravel() > caps).ravel() & active
        if too_fast.any():
            velocities[too_fast] = (
                velocities[too_fast] / speeds[too_fast]
            ) * caps[too_fast, np.newaxis]
        # No lower bound — slow speeds left as-is

    elif speed_mode == "none":
        pass  # no speed clamp

    # 4. Inertia: lerp between raw and clamped velocity
    if inertia > 0 and raw_vel is not None:
        velocities[active] = (
            velocities[active] * (1.0 - inertia)
            + raw_vel[active] * inertia
        )

    # 5. Zero-speed deterministic fallback — (minSpeed, 0, 0) for all modes
    #    Prevents NaN in normalise() and keeps replay bit-identical.
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    zero_speed = (speeds.ravel() < 1e-6) & active
    if zero_speed.any():
        velocities[zero_speed, 0] = min_speed[zero_speed]
        velocities[zero_speed, 1] = 0.0
        velocities[zero_speed, 2] = 0.0

    # 6. Move forward
    if move:
        positions[active] += velocities[active] * dt

    # 7. Boundary enforcement
    _apply_boundary(positions, velocities, active,
                    width, height, depth, boundary_mode,
                    sphere_radius, avoidance_factor)

    # 8. Reset accelerations for next frame
    accelerations[active] = np.float32(0.0)

    # 9. NaN guard: reset any non-finite positions to centre (P0.10)
    if center is not None:
        bad = (~np.isfinite(positions)).any(axis=1) & active
        if bad.any():
            positions[bad] = center.astype(np.float32)
            velocities[bad] = 0.0


def _apply_boundary(
    positions: np.ndarray,
    velocities: np.ndarray,
    active: np.ndarray,
    width: float,
    height: float,
    depth: float,
    mode: str,
    sphere_radius: float,
    avoidance_factor: float,
) -> None:
    """Enforce boundary conditions on active birds."""
    if mode == "toroidal":
        mask = active
        positions[mask, 0] %= width
        positions[mask, 1] %= height
        positions[mask, 2] %= depth

    elif mode == "open":
        pass  # birds may leave freely

    elif mode == "margin":
        _margin_push(positions, velocities, active, width, height, depth,
                     avoidance_factor)

    elif mode == "sphere":
        _sphere_soft(positions, velocities, active, sphere_radius,
                     avoidance_factor)


def _margin_push(
    positions: np.ndarray,
    velocities: np.ndarray,
    active: np.ndarray,
    width: float,
    height: float,
    depth: float,
    factor: float,
    margin: float = 50.0,
) -> None:
    """Nudge velocity away from domain walls when within margin."""
    for axis, size in enumerate([width, height, depth]):
        v = velocities[:, axis]
        p = positions[:, axis]

        lo = (p < margin) & active
        hi = (p > size - margin) & active

        v[lo] += factor * (margin - p[lo]) / margin
        v[hi] -= factor * (p[hi] - (size - margin)) / margin

        p[lo] = np.maximum(p[lo], 0.0)
        p[hi] = np.minimum(p[hi], size)


def _sphere_soft(
    positions: np.ndarray,
    velocities: np.ndarray,
    active: np.ndarray,
    radius: float,
    factor: float,
) -> None:
    """Asymptotic push toward sphere interior. 1/(R - r) soft boundary."""
    dists = np.linalg.norm(positions, axis=1)
    outside = (dists > radius) & active

    if not outside.any():
        return

    radial = positions[outside] / dists[outside, np.newaxis]
    positions[outside] = radial * radius
    velocities[outside] -= radial * factor * (dists[outside, np.newaxis] - radius)


# ── BoidView — lightweight single-bird access ─────────────────────

class BoidView:
    """Read-only view into one bird's state within flat SoA arrays.

    Used by occlusion and force functions that need to address
    individual birds. Extremely lightweight — __slots__ only.
    """

    __slots__ = ("idx", "_positions", "_velocities", "_thetas")

    def __init__(
        self,
        idx: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        last_theta: np.ndarray | None = None,
    ) -> None:
        self.idx = idx
        self._positions = positions
        self._velocities = velocities
        self._thetas = last_theta

    @property
    def pos(self) -> Vec3:
        """Position as (3,) float32 view."""
        return self._positions[self.idx]

    @property
    def vel(self) -> Vec3:
        """Velocity as (3,) float32 view."""
        return self._velocities[self.idx]

    @property
    def theta(self) -> float:
        """Cached internal opacity from last projection step."""
        if self._thetas is None:
            return 0.0
        return float(self._thetas[self.idx])


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

    v = ((0.34 ± 0.08), ±0.16, (0.08 ± 0.08)) · v0 · 0.5

    Creates a gentle forward drift (positive x) with modest y/z spread.

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
    return (v * v0 * 0.5).astype(np.float32)
