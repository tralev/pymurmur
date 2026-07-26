"""NeighborAdaptiveSpeed extension — isolated boids fly faster.

Generalizes angle.py's deficit-based speed law (P5.3/S2.C3) across all
7 modes without touching angle.py's own tested code: queries
flock._index directly (mode-agnostic — every mode's PhysicsFlock has
one, per SpatialIndex Protocol's query_knn), counts how many of a
fixed-k query fall within the configured radius, and maps the neighbor
deficit to a per-boid speed-cap multiplier, published as
flock.neighbor_adaptive_speed_mult. PhysicsFlock.integrate() multiplies
this into max_speed the same way SpeedNoise does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..adaptive_speed import adaptive_speed_bonus
from ._base import Extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ._base import StepContext


class NeighborAdaptiveSpeed(Extension):
    """Noise-free, deterministic per-boid speed cap driven by local
    neighbor count: boids with fewer neighbors than the configured
    target speed up (matching angle.py's precedent); boids at or above
    the target get no bonus (multiplier 1.0).

    Same floor-and-ceiling-scale note as SpeedNoise applies here: under
    speed_mode="band"/"clamp" this multiplier scales both bounds of the
    clamp band, which is what lets isolated boids actually move faster
    on average, not just be allowed to.

    Known limitation: field/influencer/marl modes declare needs_index=
    False, so the engine never calls flock._index.rebuild() while one
    of them is active — the index's `ready` stays False forever, and
    this extension correctly no-ops (multiplier 1.0 for every boid)
    rather than querying a stale/empty index. spatial/projection/
    vicsek/angle all rebuild the index every tick and get real
    adaptive-speed behaviour.
    """

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        cfg = ctx.config.neighbor_adaptive_speed
        active = flock.active
        n = len(flock.positions)
        mult = np.ones(n, dtype=np.float32)

        index = flock._index
        if index is None or not getattr(index, "ready", False):
            flock.neighbor_adaptive_speed_mult = mult
            return

        active_idx = np.where(active)[0]
        n_nbrs = np.zeros(len(active_idx), dtype=np.float32)
        k = cfg.neighbor_adaptive_speed_target
        radius_sq = cfg.neighbor_adaptive_speed_radius ** 2
        for row, i in enumerate(active_idx):
            nbrs = index.query_knn(flock.positions[i], k)
            if len(nbrs) == 0:
                continue
            diffs = flock.positions[nbrs] - flock.positions[i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            n_nbrs[row] = np.count_nonzero(dists_sq <= radius_sq)

        deficit = k - n_nbrs
        deficit_cap = float(k * k)
        bonus = adaptive_speed_bonus(
            deficit, cfg.neighbor_adaptive_speed_mode, deficit_cap,
            cfg.neighbor_adaptive_speed_linear_scale,
        )
        mult[active_idx] = 1.0 + bonus / max(ctx.config.v0, 1e-6)
        flock.neighbor_adaptive_speed_mult = mult
