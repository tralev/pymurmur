"""SpeedNoise extension — organic slow/fast zones from a 3D noise field.

Samples pymurmur.core.noise.value_noise3 at each boid's position and
maps the result to a per-boid speed-cap multiplier, published as
flock.speed_noise_mult. PhysicsFlock.integrate() multiplies this into
the per-boid max_speed array consumed by boid.integrate(), so the
effect applies under every speed_mode ("band"/"clamp", "fixed",
"ceiling") and every flocking mode — no changes needed to boid.py or
any ForceMode.

Stateless: unlike Wander's internal clock accumulator, frame/dt come
directly from StepContext each call, so output only ever depends on
(flock.positions, ctx.frame, ctx.dt, ctx.config) — no ordering
requirement across calls for determinism to hold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.noise import value_noise3
from ._base import Extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ._base import StepContext


class SpeedNoise(Extension):
    """Noise-modulated per-boid speed cap.

    Note: under speed_mode="band"/"clamp", boid.integrate() derives
    min_speed = max_speed * speed_min_factor — so this multiplier
    scales BOTH the floor and the ceiling of the clamp band, not just
    the ceiling. That's intentional: it's what produces "slow zones"
    (both bounds shrink together) rather than only capping top speed.
    """

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        cfg = ctx.config.speed_noise
        t_offset = ctx.frame * ctx.dt * cfg.speed_noise_time_scale
        if t_offset != 0.0:
            sample_pos = flock.positions + np.array(
                [t_offset, 0.0, 0.0], dtype=np.float32,
            )
        else:
            sample_pos = flock.positions

        noise = value_noise3(
            sample_pos, cfg.speed_noise_frequency, ctx.config.seed or 0,
        )
        mult = (
            cfg.speed_noise_min_mult
            + (cfg.speed_noise_max_mult - cfg.speed_noise_min_mult)
            * noise ** cfg.speed_noise_power
        )
        flock.speed_noise_mult = mult.astype(np.float32)
