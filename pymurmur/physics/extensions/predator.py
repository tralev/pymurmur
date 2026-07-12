"""Predator extension — autonomous threat agent.

O(N) per frame. Approach/egress FSM, pass-through targeting with arc
offset, threat force on nearby boids.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import Extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock


class Predator(Extension):
    """Autonomous predator that approaches and passes through the flock."""

    def __init__(self) -> None:
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._phase: str = "approach"

    def apply(self, flock: PhysicsFlock) -> None:
        """Update predator state and apply threat forces."""
        active = flock.active
        if active.sum() == 0:
            return

        com = np.mean(flock.positions[active], axis=0)

        if self._phase == "approach":
            # Move toward flock centre
            to_com = com - self._pos
            dist = np.linalg.norm(to_com)
            if dist > 1e-6:
                self._vel = to_com / dist * 8.0
            if dist < 50.0:
                self._phase = "pass_through"
        else:
            # Pass through with offset
            self._phase = "approach"
            self._pos = com + flock.rng.normal(scale=200, size=3).astype(np.float32)

        self._pos += self._vel * 0.016

        # Threat force + panic boost on nearby birds
        panicked: list[int] = []
        for i in np.where(active)[0]:
            diff = flock.positions[i] - self._pos
            d = np.linalg.norm(diff)
            if d < 200.0 and d > 0:
                # Threat force: radial push away from predator
                flock.accelerations[i] += (
                    diff / d * 0.5 * (200.0 - d) / 200.0
                )
                # Track panicked birds (within half threat radius)
                if d < 100.0:
                    panicked.append(i)

        # Panic speed boost + blackening (cohesion pull) for panicked birds
        if panicked:
            com_panicked = np.mean(flock.positions[panicked], axis=0)
            for i in panicked:
                # Speed boost: increase velocity by 50%
                flock.velocities[i] *= 1.5
                # Blackening: cohesion toward panic group centre
                to_centre = com_panicked - flock.positions[i]
                dist_centre = np.linalg.norm(to_centre)
                if dist_centre > 1e-6:
                    flock.accelerations[i] += to_centre / dist_centre * 0.3
