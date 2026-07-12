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
) -> None:
    """Vectorised Euler integration over the entire flock.

    Operates on flat arrays — no Python per-bird loop. All parameters
    are passed explicitly to avoid a SimConfig import at the hot-path level.
    """
    # 1. Apply accumulated forces (only active birds)
    velocities[active] += accelerations[active]

    # 2. Speed clamp: [0.3 * v0, v0] — vectorised
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    too_fast = (speeds > v0).ravel()
    too_slow = (speeds < v0 * 0.3).ravel()

    if too_fast.any():
        velocities[too_fast] = (velocities[too_fast]
                                / speeds[too_fast]) * v0
    if too_slow.any():
        velocities[too_slow] = (velocities[too_slow]
                                / speeds[too_slow]) * v0 * 0.3

    # 2b. Zero-speed re-seed: random unit sphere for speeds < 1e-6
    zero_speed = (speeds.ravel() < 1e-6) & active
    if zero_speed.any():
        nz = zero_speed.sum()
        velocities[zero_speed] = random_unit_sphere(nz) * v0 * 0.5

    # 3. Move forward
    positions[active] += velocities[active] * dt

    # 4. Boundary enforcement
    _apply_boundary(positions, velocities, active,
                    width, height, depth, boundary_mode,
                    sphere_radius, avoidance_factor)

    # 5. Reset accelerations for next frame
    accelerations[active] = np.float32(0.0)


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
        positions[:, 0] %= width
        positions[:, 1] %= height
        positions[:, 2] %= depth

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
