"""Extension protocol — pluggable behavioural modules.

Each extension is a class implementing apply(flock, ctx). The
ExtensionManager assembles whichever are enabled in config.

Iteration 5: StepContext provides per-frame context (dt, config, rng,
frame, centre) so extensions can react to T/K toggles and use correct
integration timesteps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig


@dataclass
class StepContext:
    """Per-frame context passed to every extension.

    Provides dt, config, RNG, frame counter, and swarm centre so
    extensions don't need to hardcode timesteps or reach into engine
    internals.  threat_prox is an optional per-bird threat proximity
    array populated by the Predator extension (I5.4).
    """
    frame: int
    dt: float
    rng: "np.random.Generator"
    center: "np.ndarray | None"
    config: "SimConfig"
    threat_prox: "np.ndarray | None" = None


class Extension(ABC):
    """Base protocol for behavioural extensions."""

    @abstractmethod
    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Apply the extension's effect to the flock."""
        ...
