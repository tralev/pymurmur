"""Single-bird integration kernel.

Level 0 — depends only on core.types and core.config.
NEVER imports physics.flock or physics.forces.

The entire flock is integrated in one vectorised call — no per-bird
Python loops.  Boundary modes operate on flat arrays via boolean masks.

Array init helpers (random_positions, init_positions, init_velocities,
etc.) live in boid_init.py (file-size split) and are re-exported below
so `from pymurmur.physics.boid import init_positions` etc. keep working.
"""

from __future__ import annotations

import numpy as np

from .boid_init import (  # noqa: F401 — re-exported for back-compat
    init_positions,
    init_velocities,
    init_velocities_blob,
    init_velocities_cube,
    init_velocities_fixed,
    init_velocities_speed_uniform,
    init_velocities_tangential,
    random_positions,
    random_unit_sphere,
)

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
    velocity_noise: np.ndarray | None = None,
    damping: float = 0.0,
) -> None:
    """Vectorised Euler integration over the entire flock.

    Operates on flat arrays — no Python per-bird loop. All parameters
    are passed explicitly to avoid a SimConfig import at the hot-path level.

    speed_mode: "band"/"clamp" (clamp [min, cap]), "fixed" (exact
                renormalisation), "ceiling" (≤ cap only), "none" (no clamp).
                "clamp" is the SpatialConfig.speed_mode default vocabulary
                and aliases "band" (D11 — an unrecognised value would
                silently disable speed enforcement).
    inertia: 0.0–1.0 lerp between raw and clamped velocity.
    move: if False, skip position update (caller owns positions).
    velocity_noise: S2.B2 — (N, 3) additive velocity-domain noise applied
                    right after v+=a and before the speed clamp (matches
                    the spec pipeline order: accumulate -> accel_scale ->
                    clamp(force) -> v+=a -> velocity noise -> ceiling
                    limit -> move). None = no-op (default, back-compat).
    damping: general velocity damping/friction, v *= (1 - damping*dt),
             applied AFTER the speed-mode clamp/inertia (step 4b) so it
             has a uniform effect regardless of speed_mode — applying it
             earlier would be immediately undone by "band"'s floor or
             "fixed"'s exact renormalisation. 0.0 = no-op (default).
    """
    # 0. Safety rails: dt clamp (P0.10)
    dt = float(np.clip(dt, 0.0, 0.05))

    # 0a. D1: Default center to domain centre when not provided.
    #     Ensures sphere/sphere_soft boundary is always centred on C,
    #     never origin, for ALL callers (not just PhysicsFlock.integrate).
    if center is None:
        center = np.array([width / 2, height / 2, depth / 2], dtype=np.float32)

    # 1. Apply accumulated forces (only active birds)
    velocities[active] += accelerations[active]

    # 1a. S2.B2: velocity-domain noise, applied after v+=a and before
    # the speed clamp (so it is not itself clamped by max_force).
    if velocity_noise is not None:
        velocities[active] += velocity_noise[active]

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

    if speed_mode in ("band", "clamp"):
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

    # 4b. General velocity damping/friction — applied after the speed-mode
    # clamp/inertia so it's visible regardless of speed_mode (see docstring).
    if damping > 0.0:
        velocities[active] *= (1.0 - damping * dt)

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
                    sphere_radius, avoidance_factor,
                    center=center)

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
    center: np.ndarray | None = None,
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
                     avoidance_factor, center=center)

    elif mode == "sphere_soft":
        _sphere_soft_asymptotic(positions, velocities, active, sphere_radius,
                                avoidance_factor, center=center)


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
    center: np.ndarray | None = None,
) -> None:
    """Hard sphere boundary at radius from centre C.

    Birds outside radius are projected back onto the sphere surface
    and given an inward velocity correction proportional to overshoot.

    Uses ‖p−C‖ (not ‖p‖) — the sphere is centred on the domain centre.
    """
    if center is None:
        center = np.zeros(3, dtype=np.float32)

    offsets = positions - center
    dists = np.linalg.norm(offsets, axis=1)
    outside = (dists > radius) & active

    if not outside.any():
        return

    radial = offsets[outside] / dists[outside, np.newaxis]
    positions[outside] = center + radial * radius
    velocities[outside] -= radial * factor * (dists[outside, np.newaxis] - radius)


def _sphere_soft_asymptotic(
    positions: np.ndarray,
    velocities: np.ndarray,
    active: np.ndarray,
    radius: float,
    factor: float,
    center: np.ndarray | None = None,
) -> None:
    """Asymptotic soft sphere boundary — never hard-projects positions.

    Birds near the boundary get a gentle inward velocity push:
        Δv = −factor · r̂ / max(R−r, 0.05·R)
    No position clamping — birds can briefly overshoot and are pushed back
    smoothly. Uses ‖p−C‖ (sphere centred on domain centre).
    """
    if center is None:
        center = np.zeros(3, dtype=np.float32)

    offsets = positions - center
    dists = np.linalg.norm(offsets, axis=1)

    # Apply to birds near or outside the boundary
    near = (dists > radius * 0.9) & active
    if not near.any():
        return

    # Soft margin: 10% of radius for asymptotic kick
    gap = radius - dists[near]
    # Clamp gap to avoid divide-by-zero; max push when r → R
    safe_gap = np.maximum(gap, 0.05 * radius)

    radial = offsets[near] / dists[near, np.newaxis]
    # Push grows as 1/gap — stronger near the boundary
    push_strength = factor * radius / safe_gap
    velocities[near] -= radial * push_strength[:, np.newaxis]

