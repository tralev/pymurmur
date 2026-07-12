"""Wander extension — bounded attractor motion for the flock centre.

O(1) per frame. Multi-frequency composite trig + radial pulse.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import Extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock


class Wander(Extension):
    """Bounded flock-centre wander using multi-frequency trig."""

    def __init__(self) -> None:
        self._t: float = 0.0

    def apply(self, flock: PhysicsFlock) -> None:
        """Move the wander attractor and apply gentle pull on all birds."""
        self._t += 0.001

        target = np.array([
            100 * np.sin(self._t * 1.3) * np.cos(self._t * 0.7),
            100 * np.sin(self._t * 1.7) * np.sin(self._t * 0.5),
            50 * np.sin(self._t * 2.1),
        ], dtype=np.float32)

        active = flock.active
        for i in np.where(active)[0]:
            to_target = target - flock.positions[i]
            dist = np.linalg.norm(to_target)
            if dist > 1e-6:
                flock.accelerations[i] += (
                    to_target / dist * min(dist * 0.0005, 0.1)
                )
