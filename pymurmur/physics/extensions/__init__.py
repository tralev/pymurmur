"""ExtensionManager — assembles enabled extensions and applies them in pre_step()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import Extension
from .predator import Predator
from .ecology import Ecology
from .wander import Wander
from .ripple import Ripple

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig


class ExtensionManager:
    """Manages pluggable behavioural extensions.

    Instantiate once; call pre_step() before each simulation step.
    Only enabled extensions (per config) are instantiated.
    Ecology runs first (advances day, sets predator presence), then
    Predator is conditionally applied based on predator_present(day).
    """

    def __init__(self, config: SimConfig) -> None:
        self._extensions: list[Extension] = []
        self._ecology: Ecology | None = None
        self._predator: Predator | None = None
        if config.predator_enabled:
            self._predator = Predator()
            self._extensions.append(self._predator)
        if config.roosting_enabled:
            self._ecology = Ecology()
            self._extensions.append(self._ecology)
        if config.wander_enabled:
            self._extensions.append(Wander())
        if config.ripple_enabled:
            self._extensions.append(Ripple())

    def pre_step(self, flock: PhysicsFlock) -> None:
        """Run all enabled extensions before force computation.

        Ecology runs first to advance the day and update predator_present.
        Predator only runs when ecology says predator is present (or when
        ecology is not enabled, in which case predator is always active).
        """
        eco = self._ecology
        pred = self._predator

        # Run ecology first (advances day, sets _predator_active flag)
        if eco is not None:
            eco.apply(flock)

        # Run predator only if present (or ecology not enabled → always present)
        if pred is not None:
            if eco is None or eco._predator_active:
                pred.apply(flock)

        # Run remaining extensions (wander, ripple)
        for ext in self._extensions:
            if ext is not eco and ext is not pred:
                ext.apply(flock)

    @property
    def count(self) -> int:
        return len(self._extensions)
