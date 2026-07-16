"""Ecology extension — day/night cycle and dusk roost pull.

O(1) per frame. Day-length model, temperature model, logistic dusk ramp.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import Extension

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ..flock import PhysicsFlock
    from ._base import StepContext


class Ecology(Extension):
    """Day/night cycle with dusk roosting behaviour."""

    def __init__(self, config: SimConfig) -> None:
        self._day: float = 172.0   # summer solstice
        self._day_dt: float = 1.0 / 600.0  # ~1 day per 10 seconds at 60fps
        self._roost_pos = np.array(config.ecology_roost, dtype=np.float32)
        self._critical_mass = config.ecology_critical_mass
        self.predator_active: bool = True  # updated in apply(); public for I5.4
        self._last_int_day: int = int(self._day)

    def day_length(self, day: float) -> float:
        """Hours of daylight for a given day of year."""
        return 12.0 + 4.5 * np.cos(2 * np.pi * (day - 172) / 365)

    def temperature(self, day: float) -> float:
        """Temperature in °C for a given day of year."""
        return 9.0 - 8.0 * np.cos(2 * np.pi * (day - 20) / 365)

    @staticmethod
    def predator_present(day: int) -> bool:
        """Deterministic per-day predator presence via Knuth multiplicative hash.

        Returns True roughly 30% of days, deterministically.
        """
        # Knuth multiplicative hash: h = (day * 2654435769) >> 32
        day_int = int(day) & 0xFFFFFFFF
        h = (day_int * 2654435769) & 0xFFFFFFFF
        # Use upper bits for uniformity, threshold at ~30%
        return (h >> 24) < 77  # 77/256 ≈ 30%

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Advance day and apply dusk roost pull."""
        self._day += self._day_dt

        # Check for predator presence on day boundary
        int_day = int(self._day)
        if int_day != self._last_int_day:
            self._last_int_day = int_day
            self.predator_active = self.predator_present(int_day)

        day_len = self.day_length(self._day)
        dusk = 12.0 + day_len / 2.0
        hour = (self._day % 1) * 24.0

        # Dusk pull within last hour before sunset
        if dusk - 1.0 < hour < dusk:
            ramp = (dusk - hour) / 1.0  # [0, 1]
            # Critical mass: dampen roost pull below ~500 birds
            n_active = flock.active.sum()
            # Smoothstep: 0 at N=0, 1 at N=500, clamped above 500
            t = min(n_active / self._critical_mass, 1.0)
            mass_factor = t * t * (3.0 - 2.0 * t)
            ramp *= mass_factor

            active = flock.active
            for i in np.where(active)[0]:
                to_roost = self._roost_pos - flock.positions[i]
                flock.accelerations[i] += to_roost * 0.01 * ramp
