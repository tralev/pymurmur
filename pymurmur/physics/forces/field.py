"""crs48 field/blob mode — vectorised per-bird terms, O(N), no neighbour queries.

Shell force, slot repulsion, ripple envelopes, flow field, fold noise,
tangential orbital, buoyancy, viscous drag, drift alignment, target pull.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig


def field_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Compute field/blob anchor forces — O(N), fully vectorised."""
    active = flock.active
    n_active = active.sum()
    if n_active == 0:
        return

    positions = flock.positions
    velocities = flock.velocities
    acc = flock.accelerations

    # CoM (center of mass) and average velocity — compute once
    com = np.mean(positions[active], axis=0)
    avg_vel = np.mean(velocities[active], axis=0)

    # ── Shell force (vectorised): pull toward CoM ──
    to_com = com - positions[active]
    dists = np.linalg.norm(to_com, axis=1)
    mask = dists > 1e-6
    shell_dir = np.zeros_like(to_com)
    shell_dir[mask] = to_com[mask] / dists[mask, np.newaxis]
    acc[active] += shell_dir * config.field_cohesion * 0.01

    # ── Drift alignment (vectorised): steer toward CoM velocity ──
    acc[active] += (avg_vel - velocities[active]) * config.field_alignment * 0.01

    # ── Flow noise (vectorised): pseudo-random curl field ──
    p = positions[active]
    noise = np.column_stack([
        np.sin(p[:, 1] * 0.01 + p[:, 2] * 0.007),
        np.sin(p[:, 2] * 0.01 + p[:, 0] * 0.007),
        np.sin(p[:, 0] * 0.01 + p[:, 1] * 0.007),
    ]).astype(np.float32)
    acc[active] += noise * config.field_flow * 0.1

    # ── Slot repulsion (vectorised, O(N)): sparse offset-based ──
    # Uses fixed stride offsets (±1, ±7, ±31) instead of O(N²) all-pairs
    if config.field_separation > 0 and n_active > 1:
        active_idx = np.where(active)[0]
        offsets = np.array([1, 7, 31], dtype=np.int32)
        for offset in offsets:
            if offset >= n_active:
                continue
            # Forward offset
            src = active_idx[:-offset]
            dst = active_idx[offset:]
            diffs = positions[dst] - positions[src]
            inv_dist = 1.0 / (np.linalg.norm(diffs, axis=1) + 1e-6)
            force = diffs * (inv_dist ** 2)[:, np.newaxis] * (-config.field_separation * 0.01)
            acc[dst] += force
            acc[src] -= force  # Newton's third law

    # ── Clamp ──
    acc_mags = np.linalg.norm(acc, axis=1)
    too_strong = (acc_mags > config.max_force) & active
    if too_strong.any():
        acc[too_strong] = (
            acc[too_strong] / acc_mags[too_strong, np.newaxis] * config.max_force)
