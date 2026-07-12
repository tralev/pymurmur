"""JerBoon cosmic influencer mode — Lissajous target follow.

No neighbour queries. Each bird follows a single 3D Lissajous target
with rank-based influence gradient. Fully vectorised per substep.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig


def influencer_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Compute influencer forces — all birds follow a Lissajous target.

    The target moves along a 3D Lissajous curve. Each bird's attraction
    is weighted by its rank distance from the flock centre.
    """
    active = flock.active
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return

    substeps = config.influencer_substeps
    rank_exp = config.influencer_rank_exponent
    positions = flock.positions

    for _ in range(substeps):
        # Compute 3D Lissajous target position
        t = flock.rng.uniform(0, 2 * np.pi)
        target = np.array([
            200 * np.sin(t * 2.3),
            200 * np.sin(t * 3.7),
            40 * np.sin(t * 5.1),
        ], dtype=np.float32)

        # Rank birds by distance from CoM (lower rank = closer = more influence)
        com = np.mean(positions[active_idx], axis=0)
        dists = np.linalg.norm(positions[active_idx] - com, axis=1)
        ranks = np.argsort(dists).argsort() / n_active  # [0, 1] per bird

        # Influence gradient: closer birds follow target more tightly
        influence = (1.0 - ranks) ** rank_exp  # shape (n_active,)

        # Vectorised: all birds pulled toward target at once
        to_target = target - positions[active_idx]  # (n_active, 3)
        dists_to_target = np.linalg.norm(to_target, axis=1, keepdims=True) + 1e-10
        directions = to_target / dists_to_target

        flock.accelerations[active_idx] += (
            directions * influence[:, np.newaxis] * 0.1
        )

    # Clamp to max_force
    acc_mags = np.linalg.norm(flock.accelerations, axis=1)
    too_strong = (acc_mags > config.max_force) & active
    if too_strong.any():
        flock.accelerations[too_strong] = (
            flock.accelerations[too_strong] /
            acc_mags[too_strong, np.newaxis] * config.max_force
        )
