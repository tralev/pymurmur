"""Deterministic 3D value-noise field sampled at world positions.

Level 0 — no project imports beyond numpy. Distinct from
`seed_noise3` (types.py), which is a per-boid-seed sinusoid keyed on
(seed, t) and is NOT spatially coherent; this module produces a true
continuous noise field over 3D space, used by the speed_noise
extension to drive organic slow/fast zones.
"""

from __future__ import annotations

import numpy as np

# Squirrel3-style integer bit-mix hash (Eiserloh) — avoids the float32
# periodicity/banding artifacts a sin()*C float hash can show at scale.
_BIT_NOISE1 = np.uint32(0xB5297A4D)
_BIT_NOISE2 = np.uint32(0x68E31DA4)
_BIT_NOISE3 = np.uint32(0x1B56C4E9)
_PRIME_X = np.uint32(198491317)
_PRIME_Y = np.uint32(6542989)
_PRIME_Z = np.uint32(357239)


def _squirrel3(n: np.ndarray, seed: int) -> np.ndarray:
    """Vectorized integer bit-noise hash over a uint32 array."""
    n = n.astype(np.uint32) * _BIT_NOISE1
    n = n + np.uint32(seed & 0xFFFFFFFF)
    n = n ^ (n >> np.uint32(8))
    n = n + _BIT_NOISE2
    n = n ^ (n << np.uint32(8))
    n = n * _BIT_NOISE3
    n = n ^ (n >> np.uint32(8))
    return n


def _lattice_hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Hash integer lattice coordinates to a deterministic value in [0, 1)."""
    combined = (
        ix.astype(np.int64).astype(np.uint32) * _PRIME_X
        + iy.astype(np.int64).astype(np.uint32) * _PRIME_Y
        + iz.astype(np.int64).astype(np.uint32) * _PRIME_Z
    )
    hashed = _squirrel3(combined, seed)
    return hashed.astype(np.float64) / 4294967296.0


def value_noise3(positions: np.ndarray, frequency: float, seed: int) -> np.ndarray:
    """Vectorized 3D lattice value-noise, sampled at world positions.

    Pure function of (positions, frequency, seed) — no RNG draws, so
    output is bit-identical across independent calls with the same
    arguments, unlike drawing from a stateful np.random.Generator.

    Args:
        positions: (N, 3) array of world-space positions
        frequency: spatial frequency (cycles/unit) — higher values
            shrink the noise "zones"; lower values widen them
        seed: integer seed distinguishing independent noise fields

    Returns:
        (N,) float32 array of noise samples in [0, 1]
    """
    positions = np.asarray(positions, dtype=np.float64)
    p = positions * frequency
    p0 = np.floor(p)
    frac = p - p0
    ix0 = p0[:, 0].astype(np.int64)
    iy0 = p0[:, 1].astype(np.int64)
    iz0 = p0[:, 2].astype(np.int64)

    # Quintic fade (Perlin's improved fade, C2-continuous) per axis.
    fade = frac * frac * frac * (frac * (frac * 6.0 - 15.0) + 10.0)
    fx, fy, fz = fade[:, 0], fade[:, 1], fade[:, 2]

    def corner(dx: int, dy: int, dz: int) -> np.ndarray:
        return _lattice_hash01(ix0 + dx, iy0 + dy, iz0 + dz, seed)

    c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
    c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
    c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
    c011, c111 = corner(0, 1, 1), corner(1, 1, 1)

    x00 = c000 + fx * (c100 - c000)
    x10 = c010 + fx * (c110 - c010)
    x01 = c001 + fx * (c101 - c001)
    x11 = c011 + fx * (c111 - c011)

    y0 = x00 + fy * (x10 - x00)
    y1 = x01 + fy * (x11 - x01)

    result = y0 + fz * (y1 - y0)
    return result.astype(np.float32)
