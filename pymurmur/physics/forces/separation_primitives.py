"""Separation-specific force machinery — kernel dispatch and
separation_force.

Level 0 — pure numpy functions operating on flat arrays.

Separation supports the most kernel variants of the three primitives
(11 vs. alignment's 4 and cohesion's 3 — see kernel_registry.py), so
its dispatch machinery is the most complex and warrants its own file.

Split out of _base.py (file-size split) — alignment_force, cohesion_force,
curl_flow, noise_force, and the ForceTerm/composeForces composition
framework stay in the original, which imports separation_force and
_is_ragged back from here (_is_ragged moved here too, rather than
staying in _base.py, to avoid a circular import — alignment_force and
cohesion_force also use it).
"""

from __future__ import annotations

import numpy as np

from ..plugins.kernel_registry import SEPARATION_KERNEL_REGISTRY, KernelInfo


def _is_ragged(neighbor_idx: np.ndarray) -> bool:
    """True if neighbor_idx is an object array (ragged), not dense 2D."""
    return neighbor_idx.dtype == np.dtype('object')


def _dispatch_separation_kernel(
    kernel: str,
    diffs: np.ndarray,
    dists: np.ndarray,
    close: np.ndarray,
    radius: float | None = None,
    zone_width: float | None = None,
    closing_speed: np.ndarray | None = None,
    heading: np.ndarray | None = None,
) -> np.ndarray:
    """Kernel-name -> kernels.py function dispatch via SEPARATION_KERNEL_REGISTRY.
    Unrecognized kernel names silently fall back to "sum" (matches pre-refactor
    behavior, explicitly tested by test_invalid_kernel_ignored_by_code_structure)."""
    info: KernelInfo | None = SEPARATION_KERNEL_REGISTRY.get(kernel)
    if info is None:
        info = SEPARATION_KERNEL_REGISTRY["sum"]

    kwargs: dict[str, Any] = {}
    if info.needs_radius:
        kwargs["radius"] = radius
    if info.needs_zone_width:
        kwargs["zone_width"] = zone_width
    if info.needs_closing_speed:
        kwargs["closing_speed"] = closing_speed
    if info.needs_heading:
        kwargs["heading"] = heading

    return info.fn(diffs, dists, close, **kwargs)


def _closing_speed(diffs: np.ndarray, dists: np.ndarray, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
    """Rate of distance decrease between i and each neighbor j (positive
    = approaching). unit = diffs/dist (points i -> j); closing speed is
    the component of (v_i - v_j) along that direction."""
    dists_safe = np.where(dists > 1e-6, dists, 1.0)
    unit = diffs / dists_safe[..., np.newaxis]
    return np.sum(unit * (v_i - v_j), axis=-1)


def separation_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbor_idx: np.ndarray,
    active: np.ndarray,
    kernel: str = "sum",
    kernel_radius: float = 20.0,
    kernel_zone_width: float = 10.0,
) -> np.ndarray:
    """Separation: push away from nearby neighbours.

    S1.5/kernel registry (pymurmur.physics.forces.kernels) — how
    neighbour contributions are combined:

    kernel="sum"    → Σ r̂/d² (Reynolds default; magnitude falls as 1/d²)
    kernel="mean"   → (1/k) Σ r̂/d² (density-invariant)
    kernel="unit"   → Σ −û(d_ij) (unit direction, distance-independent)
    kernel="exp"          → exponential-decay weight on unit direction
    kernel="linear_ramp"  → linear ramp weight on unit direction
    kernel="asymptotic"   → r/d − 1 weight on unit direction
    kernel="velocity_weighted" → "sum" base scaled by closing speed
                                 (receding neighbours contribute nothing)
    kernel="cosine_zone" → "sum" base scaled by a continuous (1+cosθ)/2
                            weight on bearing vs. own heading (§08-style
                            continuous zone, vs. a hard FOV cone cutoff)
    kernel="linear"       → Σ r̂/d (plain 1/d falloff)
    kernel="nearest_only" → only the single closest neighbour contributes
    kernel="bell_zone"    → cosine-bell weight peaking at kernel_radius
                            (zone center), falling off symmetrically on
                            both sides over kernel_zone_width

    kernel_radius is consulted by exp/linear_ramp/asymptotic/bell_zone
    (as the zone center for bell_zone). kernel_zone_width is only
    consulted by bell_zone.
    Returns (N, 3) float32.
    """
    N = len(positions)
    force = np.zeros((N, 3), dtype=np.float32)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return force

    needs_closing_speed = kernel == "velocity_weighted"
    needs_heading = kernel == "cosine_zone"

    if _is_ragged(neighbor_idx):
        # Ragged object array — per-bird fallback
        for i in active_idx:
            nbrs = neighbor_idx[i]
            if len(nbrs) == 0:
                continue
            diffs = positions[nbrs] - positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            close = dists > 1e-6
            if not close.any():
                continue
            closing_speed = (
                _closing_speed(diffs, dists, velocities[i], velocities[nbrs])
                if needs_closing_speed else None
            )
            heading = velocities[i] if needs_heading else None
            force[i] = _dispatch_separation_kernel(
                kernel, diffs, dists, close,
                radius=kernel_radius, zone_width=kernel_zone_width,
                closing_speed=closing_speed, heading=heading,
            )
        return force

    # Dense 2D int array — vectorised gather+reduce
    k = neighbor_idx.shape[1] if neighbor_idx.ndim == 2 else 0
    if k == 0:
        return force

    nbr_idx = neighbor_idx[active_idx]           # (n_active, k)
    p_i = positions[active_idx]                  # (n_active, 3)
    p_j = positions[nbr_idx]                     # (n_active, k, 3)

    diffs = p_j - p_i[:, np.newaxis, :]           # (n_active, k, 3)
    dists = np.linalg.norm(diffs, axis=2)         # (n_active, k)
    close = dists > 1e-6

    closing_speed = None
    heading = None
    if needs_closing_speed:
        v_i = velocities[active_idx]              # (n_active, 3)
        v_j = velocities[nbr_idx]                 # (n_active, k, 3)
        closing_speed = _closing_speed(diffs, dists, v_i[:, np.newaxis, :], v_j)
    if needs_heading:
        heading = velocities[active_idx]          # (n_active, 3)

    contrib = _dispatch_separation_kernel(
        kernel, diffs, dists, close,
        radius=kernel_radius, zone_width=kernel_zone_width,
        closing_speed=closing_speed, heading=heading,
    )
    force[active_idx] = contrib.astype(np.float32)
    return force
