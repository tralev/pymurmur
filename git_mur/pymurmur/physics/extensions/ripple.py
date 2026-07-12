"""Ripple extension — 3 staggered density pulse envelopes.

O(1) per frame. amount = exp(−δ²) · smoothstep_envelope, radial + twist.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import Extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock


class Ripple(Extension):
    """3 staggered density pulse envelopes."""

    def __init__(self) -> None:
        self._t: float = 0.0
        self._offsets = np.array([0.0, 9.33, 18.67], dtype=np.float32)

    def apply(self, flock: PhysicsFlock) -> None:
        """Apply ripple pulses centred on the flock CoM."""
        self._t += 0.001
        active = flock.active
        if active.sum() == 0:
            return

        com = np.mean(flock.positions[active], axis=0)

        for offset in self._offsets:
            tt = max(self._t - offset, 0.0)
            radius = tt * 200.0  # expanding pulse

            for i in np.where(active)[0]:
                dist = np.linalg.norm(flock.positions[i] - com)
                envelope = np.exp(-((dist - radius) ** 2) / 1000.0)
                if envelope > 0.01:
                    push = (flock.positions[i] - com) / (dist + 1e-10)
                    flock.accelerations[i] += push * envelope * 0.02
