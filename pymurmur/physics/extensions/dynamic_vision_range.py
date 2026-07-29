"""DynamicVisionRange extension — flock-wide adaptive perception radius.

Scoped deliberately (see class docstring): a single flock-wide scalar
multiplier, not a genuinely per-boid one, and consumed only by
spatial_helpers.py::_query_neighbors (spatial + projection modes) —
angle.py/vicsek.py/field.py/marl.py/influencer.py have their own,
differently-shaped perception knobs and are unaffected. Making this
fully per-boid and universal across all 7 modes would mean editing
each mode's own neighbor-query call site individually.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._base import Extension
from .extension_registry import register_extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ._base import StepContext


@register_extension("dynamic_vision_range_enabled", None)
class DynamicVisionRange(Extension):
    """Nudges a flock-wide visual_range multiplier up when boids see too
    few neighbors on average, down when they see too many — a §11-style
    feedback loop (visionDistance += 1 / -= 1 per frame), simplified to
    one shared multiplier rather than a per-boid value.

    Stateful (unlike SpeedNoise): the multiplier persists and drifts by
    a small step each frame rather than being recomputed from scratch,
    matching the reference implementation's incremental-adjustment shape.
    """

    def __init__(self) -> None:
        self._mult: float = 1.0

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        cfg = ctx.config.dynamic_vision_range
        index = flock._index
        active_idx = np.where(flock.active)[0]

        if index is None or not getattr(index, "ready", False) or len(active_idx) == 0:
            ctx.config._dynamic_visual_range_mult = self._mult
            return

        radius = ctx.config.visual_range * self._mult
        radius_sq = radius * radius
        k = cfg.dynamic_vision_range_sample_k

        # Sample average neighbor count across active boids (reuses the
        # existing index — no new index construction).
        counts = np.zeros(len(active_idx), dtype=np.float32)
        for row, i in enumerate(active_idx):
            nbrs = index.query_knn(flock.positions[i], k)
            if len(nbrs) == 0:
                continue
            diffs = flock.positions[nbrs] - flock.positions[i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            counts[row] = np.count_nonzero(dists_sq <= radius_sq)

        avg_count = float(np.mean(counts))
        ideal = cfg.dynamic_vision_range_ideal_count
        step = cfg.dynamic_vision_range_step

        if avg_count < ideal:
            self._mult += step
        elif avg_count > ideal:
            self._mult -= step

        self._mult = float(np.clip(
            self._mult,
            cfg.dynamic_vision_range_min_mult,
            cfg.dynamic_vision_range_max_mult,
        ))
        ctx.config._dynamic_visual_range_mult = self._mult
