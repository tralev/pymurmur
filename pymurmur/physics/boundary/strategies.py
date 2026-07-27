"""The 5 registered boundary strategies.

Pure extraction from boid.py's former _apply_boundary/_margin_push/
_sphere_soft/_sphere_soft_asymptotic — no behavior change, verified
byte-identical (see test_boundary_registry.py).

Level 0 — pure numpy, zero pymurmur imports.
"""

from __future__ import annotations

import numpy as np

from ._mode import BoundaryMode, register


@register("toroidal")
class ToroidalBoundary(BoundaryMode):
    """Wrap positions modulo the domain bounds on all 3 axes."""

    @staticmethod
    def apply(
        positions: np.ndarray,
        velocities: np.ndarray,
        active: np.ndarray,
        width: float,
        height: float,
        depth: float,
        sphere_radius: float,
        avoidance_factor: float,
        center: np.ndarray | None = None,
    ) -> None:
        mask = active
        positions[mask, 0] %= width
        positions[mask, 1] %= height
        positions[mask, 2] %= depth


@register("open")
class OpenBoundary(BoundaryMode):
    """No boundary — birds may leave freely."""

    @staticmethod
    def apply(
        positions: np.ndarray,
        velocities: np.ndarray,
        active: np.ndarray,
        width: float,
        height: float,
        depth: float,
        sphere_radius: float,
        avoidance_factor: float,
        center: np.ndarray | None = None,
    ) -> None:
        pass  # birds may leave freely


@register("margin")
class MarginBoundary(BoundaryMode):
    """Nudge velocity away from domain walls when within margin."""

    @staticmethod
    def apply(
        positions: np.ndarray,
        velocities: np.ndarray,
        active: np.ndarray,
        width: float,
        height: float,
        depth: float,
        sphere_radius: float,
        avoidance_factor: float,
        center: np.ndarray | None = None,
        margin: float = 50.0,
    ) -> None:
        for axis, size in enumerate([width, height, depth]):
            v = velocities[:, axis]
            p = positions[:, axis]

            lo = (p < margin) & active
            hi = (p > size - margin) & active

            v[lo] += avoidance_factor * (margin - p[lo]) / margin
            v[hi] -= avoidance_factor * (p[hi] - (size - margin)) / margin

            p[lo] = np.maximum(p[lo], 0.0)
            p[hi] = np.minimum(p[hi], size)


@register("sphere")
class SphereBoundary(BoundaryMode):
    """Hard sphere boundary at radius from centre C.

    Birds outside radius are projected back onto the sphere surface
    and given an inward velocity correction proportional to overshoot.

    Uses ‖p−C‖ (not ‖p‖) — the sphere is centred on the domain centre.
    """

    @staticmethod
    def apply(
        positions: np.ndarray,
        velocities: np.ndarray,
        active: np.ndarray,
        width: float,
        height: float,
        depth: float,
        sphere_radius: float,
        avoidance_factor: float,
        center: np.ndarray | None = None,
    ) -> None:
        if center is None:
            center = np.zeros(3, dtype=np.float32)

        offsets = positions - center
        dists = np.linalg.norm(offsets, axis=1)
        outside = (dists > sphere_radius) & active

        if not outside.any():
            return

        radial = offsets[outside] / dists[outside, np.newaxis]
        positions[outside] = center + radial * sphere_radius
        velocities[outside] -= radial * avoidance_factor * (dists[outside, np.newaxis] - sphere_radius)


@register("sphere_soft")
class SphereSoftBoundary(BoundaryMode):
    """Asymptotic soft sphere boundary — never hard-projects positions.

    Birds near the boundary get a gentle inward velocity push:
        Δv = −factor · r̂ / max(R−r, 0.05·R)
    No position clamping — birds can briefly overshoot and are pushed back
    smoothly. Uses ‖p−C‖ (sphere centred on domain centre).
    """

    @staticmethod
    def apply(
        positions: np.ndarray,
        velocities: np.ndarray,
        active: np.ndarray,
        width: float,
        height: float,
        depth: float,
        sphere_radius: float,
        avoidance_factor: float,
        center: np.ndarray | None = None,
    ) -> None:
        if center is None:
            center = np.zeros(3, dtype=np.float32)

        offsets = positions - center
        dists = np.linalg.norm(offsets, axis=1)

        # Apply to birds near or outside the boundary
        near = (dists > sphere_radius * 0.9) & active
        if not near.any():
            return

        # Soft margin: 10% of radius for asymptotic kick
        gap = sphere_radius - dists[near]
        # Clamp gap to avoid divide-by-zero; max push when r → R
        safe_gap = np.maximum(gap, 0.05 * sphere_radius)

        radial = offsets[near] / dists[near, np.newaxis]
        # Push grows as 1/gap — stronger near the boundary
        push_strength = avoidance_factor * sphere_radius / safe_gap
        velocities[near] -= radial * push_strength[:, np.newaxis]
