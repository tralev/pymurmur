"""A15 (Young et al. 2013): 3D point-cloud generators for comparing
robustness across distributions of varying "orderedness" -- uniform
random, grid+Gaussian-noise, and Halton (quasi-random, low-
discrepancy) sequences, per Fig. S7 (/tmp/1302.3195v1.txt:493-498).

All three generators produce points within a rectangular-prism box
(span x span x span*thickness_scale on the z axis), matching the
paper's own "rectangular prism" setup and this codebase's established
thickness convention (compute_shape's sqrt(lambda3/lambda1)).
"""
from __future__ import annotations

import numpy as np


def generate_uniform_flock(
    N: int, thickness_scale: float, rng: np.random.Generator, span: float = 100.0,
) -> np.ndarray:
    """Uniform-random points in a rectangular prism -- the paper's
    baseline ("least ordered") distribution."""
    xy = rng.uniform(-span, span, (N, 2))
    z = rng.uniform(-span * thickness_scale, span * thickness_scale, (N, 1))
    return np.hstack([xy, z]).astype(np.float32)


def generate_grid_noise_flock(
    N: int, thickness_scale: float, rng: np.random.Generator,
    span: float = 100.0, noise_frac: float = 0.3,
) -> np.ndarray:
    """Points on a regular 3D grid perturbed by Gaussian noise -- an
    "ordered" distribution (Fig. S7's magenta curve).

    Grid dimensions are sized to roughly match the box's aspect ratio
    (1, 1, thickness_scale) with at least N cells, then N points are
    sampled without replacement and jittered by Gaussian noise scaled
    to a fraction of the grid spacing.
    """
    thickness_scale = max(thickness_scale, 1e-3)
    nx = max(1, int(np.ceil((N / thickness_scale) ** (1.0 / 3.0))))
    ny = nx
    nz = max(1, int(np.ceil(nx * thickness_scale)))
    while nx * ny * nz < N:
        nx += 1
        ny += 1

    xs = np.linspace(-span, span, nx)
    ys = np.linspace(-span, span, ny)
    zs = np.linspace(-span * thickness_scale, span * thickness_scale, nz)
    grid = np.array(np.meshgrid(xs, ys, zs)).reshape(3, -1).T

    if len(grid) > N:
        idx = rng.choice(len(grid), size=N, replace=False)
        points = grid[idx]
    else:
        points = grid

    spacing = (2.0 * span) / max(nx - 1, 1)
    noise = rng.normal(0.0, noise_frac * spacing, points.shape)
    return (points + noise).astype(np.float32)


def _van_der_corput(n: int, base: int, start_index: int = 1) -> np.ndarray:
    """Van der Corput low-discrepancy sequence in the given base,
    values in [0, 1)."""
    seq = np.empty(n)
    for i in range(n):
        idx = start_index + i
        f, r = 1.0, 0.0
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq[i] = r
    return seq


def generate_halton_flock(
    N: int, thickness_scale: float, span: float = 100.0, start_index: int = 1,
) -> np.ndarray:
    """3D Halton sequence (bases 2, 3, 5 for x/y/z) -- the paper's
    "more ordered" quasi-random distribution (Fig. S7's green curve).

    Deterministic given (N, thickness_scale, start_index) -- no rng
    needed, matching Halton sequences' own low-discrepancy, non-random
    construction. start_index lets callers draw different (still
    deterministic) samples without overlapping sequence prefixes.
    """
    x = (_van_der_corput(N, 2, start_index) - 0.5) * 2.0 * span
    y = (_van_der_corput(N, 3, start_index) - 0.5) * 2.0 * span
    z = (_van_der_corput(N, 5, start_index) - 0.5) * 2.0 * span * thickness_scale
    return np.column_stack([x, y, z]).astype(np.float32)
