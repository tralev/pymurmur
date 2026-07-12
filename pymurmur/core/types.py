"""Fundamental types shared by all subsystems.

Level 0 — no project imports beyond numpy. Every module in pymurmur
agrees on these data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

# ── The vector ────────────────────────────────────────────────────
Vec3 = np.ndarray  # shape (3,), dtype float32


# ── Force function protocol ───────────────────────────────────────
class ForceFunc(Protocol):
    """A force function receives a flock and a config, mutates
    flock.accelerations in-place. Returns nothing."""

    def __call__(self, flock: "PhysicsFlock", config: "SimConfig") -> None: ...


# ── numba JIT kernel signature ────────────────────────────────────
# Receives flat arrays + scalar params. No Python objects.
ForceKernel = Callable[
    [
        np.ndarray,  # positions      (N, 3) float32
        np.ndarray,  # velocities     (N, 3) float32
        np.ndarray,  # accelerations  (N, 3) float32
        np.ndarray,  # active          (N,)  bool
        np.ndarray,  # neighbor_idx   (N, k) int32
        float,       # separation_weight
        float,       # alignment_weight
        float,       # cohesion_weight
        float,       # noise_scale
        float,       # v0
        float,       # max_force
    ],
    None,
]


# ── Flock state container ─────────────────────────────────────────
@dataclass
class FlockArrays:
    """Structure-of-Arrays — flat numpy arrays, no per-bird objects.

    Memory budget at 300K: ~13.5 MB (vs 60 MB for per-bird objects).
    Enables vectorised numpy ops with zero Python per-bird loops.
    """

    positions: np.ndarray      # (N, 3) float32
    velocities: np.ndarray     # (N, 3) float32
    accelerations: np.ndarray  # (N, 3) float32
    seeds: np.ndarray          # (N,)  float32
    last_theta: np.ndarray     # (N,)  float32  — internal opacity per bird (projection mode)
    active: np.ndarray         # (N,)  bool     — True = alive, False = slot available for reuse

    @property
    def N_active(self) -> int:
        """Count of currently active birds."""
        return int(self.active.sum())

    @property
    def N_capacity(self) -> int:
        """Total allocated slots (active + inactive)."""
        return len(self.active)


# ── Math helpers (L0, numpy-only) ─────────────────────────────────

def safe_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalize vectors to unit length with zero-vector guard.

    Zero-magnitude vectors are returned as-is (zero) instead of NaN.

    Args:
        v: (3,) or (N,3) float array
        eps: threshold below which vector is treated as zero

    Returns:
        Unit vector(s) of same shape as input
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > eps else np.zeros(3, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    safe = np.where(norms < eps, 1.0, norms)
    return v / safe


def limit3(v: np.ndarray, max_mag: float) -> np.ndarray:
    """Clamp vector magnitudes to max_mag, preserving direction.

    Args:
        v: (3,) or (N,3) float array
        max_mag: maximum allowed magnitude per vector

    Returns:
        Clamped vector(s) of same shape as input
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v * (max_mag / norm) if norm > max_mag else v.copy()
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.where(norms > max_mag, max_mag / norms, 1.0)
    return v * scale


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation: a + t * (b - a).

    Args:
        a: start value (scalar or array)
        b: end value (scalar or array)
        t: interpolation factor, typically in [0, 1]

    Returns:
        Interpolated value
    """
    return a + t * (b - a)


def rotate_about(v: np.ndarray, k: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula: rotate v around axis k by angle.

    Args:
        v: (3,) or (N,3) vector(s) to rotate
        k: (3,) rotation axis (will be normalized)
        angle: rotation angle in radians

    Returns:
        Rotated vector(s) of same shape as v
    """
    k = safe_normalize(k)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    if v.ndim == 1:
        return v * cos_a + np.cross(k, v) * sin_a + k * np.dot(v, k) * (1 - cos_a)
    dot_vk = np.dot(v, k)  # (N,)
    return v * cos_a + np.cross(k, v) * sin_a + k * np.expand_dims(dot_vk, 1) * (1 - cos_a)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Smooth Hermite interpolation between 0 and 1.

    Returns 0 for x ≤ edge0, 1 for x ≥ edge1, and smooth
    cubic interpolation in between.

    Args:
        edge0: lower edge
        edge1: upper edge
        x: input value(s), scalar or array

    Returns:
        Clamped and smoothed value(s) in [0, 1]
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-30), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hash01(x: np.ndarray) -> np.ndarray:
    """Deterministic hash mapping floats to [0, 1).

    Uses fract(sin(x·12.9898)·43758.5453) — a common
    GLSL-style pseudo-random hash.

    Args:
        x: float or array of floats

    Returns:
        Hash value(s) in [0, 1)
    """
    hashed = np.sin(x * 12.9898) * 43758.5453
    return hashed - np.floor(hashed)


def min_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Per-axis minimum-image distance (toroidal wrapping).

    Maps each component to [-box/2, box/2] by adding/subtracting
    multiples of the box size.

    Args:
        delta: (N, 3) displacement vectors
        box: (3,) domain size [W, H, D]

    Returns:
        (N, 3) minimum-image displacement vectors
    """
    return delta - box * np.round(delta / box)


def min_image_distance(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image distance (norm after toroidal wrapping).

    Args:
        delta: (N, 3) displacement vectors
        box: (3,) domain size [W, H, D]

    Returns:
        (N,) minimum-image distances
    """
    return np.linalg.norm(min_image(delta, box), axis=1)


def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately evenly-distributed points on S².

    Uses the golden-angle spiral method.

    Args:
        n: number of points (0 or 1 are handled)

    Returns:
        (n, 3) float32 array of unit vectors
    """
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n == 1:
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    i = np.arange(n, dtype=np.float32)
    y = 1.0 - (i / (n - 1)) * 2.0
    radius = np.sqrt(1.0 - y * y)
    theta = i * np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.column_stack([x, y, z]).astype(np.float32)


def seed_noise3(seeds: np.ndarray, t: float) -> np.ndarray:
    """Deterministic sinusoidal noise per bird, bounded in [-0.18, 0.18]/axis.

    Each axis uses a different sinusoidal modulation of the per-bird
    seed value and time parameter.

    Args:
        seeds: (N,) float32 per-bird seed values
        t: time parameter

    Returns:
        (N, 3) float32 noise vectors with per-axis range [-0.18, 0.18]
    """
    seeds = np.asarray(seeds, dtype=np.float32)
    x = np.sin(seeds * 17.3 + t * 2.9 + 1.3) * 0.18
    y = np.sin(seeds * 23.7 + t * 3.4 + 2.7) * 0.18
    z = np.sin(seeds * 31.1 + t * 2.1 + 4.1) * 0.18
    return np.column_stack([x, y, z]).astype(np.float32)
