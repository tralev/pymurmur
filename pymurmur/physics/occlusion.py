"""3D spherical-cap occlusion — Pearce et al. 2014.

Level 0 — pure numpy. No project imports beyond core.types.
Extended to 3D from the original 2D model.

Computes δ̂ (boundary-length-weighted projection direction),
visible neighbors (closest-first, occluded birds excluded),
and internal opacity Θ ∈ [0,1].
"""

from __future__ import annotations

import numpy as np



def spherical_cap_occlusion(
    observer_pos: np.ndarray,
    observer_vel: np.ndarray,
    neighbour_positions: np.ndarray,
    neighbour_velocities: np.ndarray,
    boid_size: float = 9.0,
    blind_cos: float | None = None,
    anisotropy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute δ̂, visible neighbors, and internal opacity Θ for one observer.

    Args:
        observer_pos: (3,) float32 — observer position.
        observer_vel: (3,) float32 — observer velocity (defines forward).
        neighbour_positions: (M, 3) float32 — neighbour positions.
        neighbour_velocities: (M, 3) float32 — neighbour velocities.
        boid_size: body radius for cap size calculation.
        blind_cos: cos(half blind angle), neighbours behind this excluded.
        anisotropy: body axis ratio a/b (1.0 = isotropic).

    Returns:
        delta: (3,) float32 — boundary-length-weighted mean direction.
               |δ̂| ∈ [0,1]; ≈1 at edge, →0 when fully surrounded.
        visible_idx: (K,) int32 — indices of visible neighbours (closest-first).
        theta: float — internal opacity Θ ∈ [0,1].
    """
    M = len(neighbour_positions)
    if M == 0:
        return np.zeros(3, dtype=np.float32), np.array([], dtype=np.int32), 0.0

    obs_forward = observer_vel / (np.linalg.norm(observer_vel) + 1e-10)

    # 1. Compute distances and sort closest-first
    diffs = neighbour_positions - observer_pos
    dists = np.linalg.norm(diffs, axis=1)
    order = np.argsort(dists)

    # 2. Process neighbours in closest-first order
    visible: list[int] = []
    theta = 0.0
    delta = np.zeros(3, dtype=np.float32)

    for j in order:
        d = dists[j]
        if d < 1e-6:
            continue  # skip self

        direction = diffs[j] / d

        # Blind angle check
        if blind_cos is not None:
            cos_angle = np.dot(direction, -obs_forward)
            if cos_angle >= blind_cos:
                continue  # behind observer

        # Effective radius (anisotropic body)
        b_eff = boid_size
        if anisotropy != 1.0:
            neighbour_forward = neighbour_velocities[j]
            n_norm = np.linalg.norm(neighbour_forward)
            if n_norm > 1e-6:
                neighbour_forward /= n_norm
                cos_psi = abs(np.dot(direction, neighbour_forward))
                sin_psi = np.sqrt(1.0 - cos_psi * cos_psi)
                b_eff = np.sqrt(
                    (boid_size * sin_psi) ** 2 +
                    (boid_size / anisotropy * cos_psi) ** 2
                )

        # Angular radius of the cap
        cap_radius = b_eff / (d + 1e-10)
        if cap_radius > 1.0:
            cap_radius = 1.0  # too close — covers whole view

        visible.append(j)
        theta += cap_radius

    visible_arr: np.ndarray = np.array([], dtype=np.int32)
    if visible:
        # δ̂ = boundary-length-weighted mean of visible directions
        visible_arr = np.array(visible, dtype=np.int32)
        visible_dirs = diffs[visible_arr] / (dists[visible_arr, np.newaxis] + 1e-10)
        cap_radii = np.minimum(boid_size / (dists[visible_arr] + 1e-10), 1.0)
        delta = np.sum(visible_dirs * cap_radii[:, np.newaxis], axis=0)
        delta_mag = np.linalg.norm(delta)
        if delta_mag > 1.0:
            delta /= delta_mag

    return delta.astype(np.float32), visible_arr.astype(np.int32), min(theta, 1.0)


def spherical_cap_occlusion_soa(
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
    nbr_positions: np.ndarray,
    nbr_velocities: np.ndarray,
    boid_size: float = 9.0,
    blind_cos: float | None = None,
    anisotropy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Adapter: wraps occlusion for flattened SoA arrays.

    Creates temporary views for σ neighbours only (6-7, not thousands).
    Bit-identical to spherical_cap_occlusion for identical inputs.
    """
    M = len(nbr_positions)
    if M == 0:
        return np.zeros(3, dtype=np.float32), np.array([], dtype=np.int32), 0.0

    # Sort by distance closest-first
    diffs = nbr_positions - obs_pos
    dists = np.linalg.norm(diffs, axis=1)
    order = np.argsort(dists)

    sorted_pos = nbr_positions[order]
    sorted_vel = nbr_velocities[order]

    return spherical_cap_occlusion(
        obs_pos, obs_vel, sorted_pos, sorted_vel,
        boid_size, blind_cos, anisotropy,
    )
