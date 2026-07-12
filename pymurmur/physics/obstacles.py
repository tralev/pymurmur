"""Signed distance function (SDF) primitives for obstacle scenes.

Level 0 — pure numpy, zero project imports. Five analytical SDF
primitives, CSG boolean operators, numerical gradient, collision
detection, and kinematic surface correction.

All SDF functions accept (N,3) position arrays and return (N,)
signed distances. Negative = inside the obstacle.

z-up convention: cylinder radial component uses dx²+dz².
"""

from __future__ import annotations

import numpy as np


# ── SDF primitives ────────────────────────────────────────────────

def sdf_sphere(
    p: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Signed distance to a sphere.

    Args:
        p: (N, 3) float32 positions
        center: (3,) sphere centre
        radius: sphere radius

    Returns:
        (N,) signed distances — negative inside, positive outside
    """
    return np.linalg.norm(p - center, axis=1) - radius


def sdf_box(
    p: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Signed distance to an axis-aligned box.

    Args:
        p: (N, 3) float32 positions
        center: (3,) box centre
        half_extents: (3,) half-widths [bx, by, bz]

    Returns:
        (N,) signed distances
    """
    q = np.abs(p - center) - half_extents
    # SDF = ‖max(q, 0)‖ + min(max(q), 0)
    #   where max(q) is along axis 1 and we take the max component
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)  # deepest penetration
    return outside + inside


def sdf_cylinder(
    p: np.ndarray,
    center: np.ndarray,
    radius: float,
    half_height: float,
) -> np.ndarray:
    """Signed distance to a cylinder aligned with the z-axis.

    Uses z-up convention: radial component = sqrt(dx² + dz²),
    vertical component = |dy - center_y|.

    Args:
        p: (N, 3) float32 positions
        center: (3,) cylinder centre
        radius: cylinder radius
        half_height: half the cylinder height

    Returns:
        (N,) signed distances
    """
    d = p - center
    dx = d[:, 0]
    dy = d[:, 1]
    dz = d[:, 2]

    # Radial distance in XZ plane
    r = np.sqrt(dx * dx + dz * dz)

    # Vertical distance along Y (height axis)
    y_dist = np.abs(dy) - half_height

    # Two components: radial (infinite cylinder) and vertical (caps)
    radial = r - radius
    height = y_dist

    # SDF = length(max(radial, height, 0)) + min(max(radial, height), 0)
    q_radial = np.maximum(radial, 0.0)
    q_height = np.maximum(height, 0.0)
    outside = np.sqrt(q_radial * q_radial + q_height * q_height)
    inside = np.minimum(np.maximum(radial, height), 0.0)
    return outside + inside


def sdf_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CSG union: min(a, b) — combine two SDFs.

    Args:
        a: (N,) SDF values from first shape
        b: (N,) SDF values from second shape

    Returns:
        (N,) combined SDF values
    """
    return np.minimum(a, b)


def sdf_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CSG subtraction: max(a, −b) — subtract b from a.

    Args:
        a: (N,) SDF values from base shape
        b: (N,) SDF values from shape to subtract

    Returns:
        (N,) combined SDF values
    """
    return np.maximum(a, -b)


# ── Utility helpers ───────────────────────────────────────────────

def sdf_gradient(
    sdf_fn,
    p: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Numerical gradient of an SDF via central finite differences.

    Args:
        sdf_fn: callable that takes (N, 3) positions and returns (N,) SDF values
        p: (N, 3) positions at which to compute gradient
        eps: step size for finite differences

    Returns:
        (N, 3) gradient vectors (unnormalized)
    """
    grad = np.zeros_like(p, dtype=np.float32)
    for axis in range(3):
        offset = np.zeros_like(p, dtype=np.float32)
        offset[:, axis] = eps
        fwd = sdf_fn(p + offset)
        bwd = sdf_fn(p - offset)
        grad[:, axis] = (fwd - bwd) / (2.0 * eps)
    return grad


def collision_detected(
    sdf_old: np.ndarray,
    sdf_new: np.ndarray,
) -> np.ndarray:
    """Detect collisions — sign of SDF changed from positive to negative.

    Args:
        sdf_old: (N,) SDF values at previous positions
        sdf_new: (N,) SDF values at current positions

    Returns:
        (N,) bool array — True where a collision occurred (entered obstacle)
    """
    return (np.sign(sdf_old) > 0) & (np.sign(sdf_new) < 0)


def kinematic_correction(
    p: np.ndarray,
    sdf_fn,
    eps: float = 1e-4,
) -> np.ndarray:
    """Push colliding positions to the obstacle surface.

    Uses a Newton-like step:
        p ← p − SDF(p) · ∇SDF / ‖∇SDF‖²

    Only corrects positions that are inside (SDF < 0).
    Falls back to a perturbed gradient when the gradient
    is near-zero (e.g. at sphere centre).

    Args:
        p: (N, 3) positions to correct
        sdf_fn: callable for the combined SDF scene
        eps: step size for gradient computation

    Returns:
        (N, 3) corrected positions
    """
    sdf_val = sdf_fn(p)
    inside = sdf_val < 0.0
    if not inside.any():
        return p.copy()

    grad = sdf_gradient(sdf_fn, p, eps)
    grad_norm_sq = np.sum(grad * grad, axis=1, keepdims=True)

    # Handle near-zero gradient: perturb position slightly so gradient
    # is well-defined (e.g. at sphere centre where gradient is zero)
    zero_grad = (grad_norm_sq.ravel() < 1e-20) & inside
    if zero_grad.any():
        p_perturbed = p.copy()
        p_perturbed[zero_grad] += eps * 2.0
        grad_perturbed = sdf_gradient(sdf_fn, p_perturbed, eps)
        grad[zero_grad] = grad_perturbed[zero_grad]
        grad_norm_sq = np.sum(grad * grad, axis=1, keepdims=True)

    # Only correct where inside AND gradient is non-degenerate
    valid = inside & (grad_norm_sq.ravel() > 1e-20)
    if not valid.any():
        return p.copy()

    result = p.copy().astype(np.float32)
    result[valid] -= (
        sdf_val[valid, np.newaxis]
        * grad[valid]
        / grad_norm_sq[valid]
    )
    return result
