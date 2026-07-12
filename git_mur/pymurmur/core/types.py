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
